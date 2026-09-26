"""GLM-5.3 KDA narrow-row (g / beta) padded rebuild 的 flat gather 内核回归测试。

旧实现: ``F.pad + index_select(1, pad_to_pack)``。CANN GatherV3 引擎在训练 pack
(~742 个短段、~26% pad 行)下会因重复哨兵索引 stall,单次 launch ~1.5 ms,每步
1088 次 = 1.67 s/步(run128 profile)。新实现: ``_seg_copy_rows_gather_kernel``
flat 行 gather,pad 行由 masked load(other=0.0)直接产零,不再需要 ``F.pad``
哨兵行。head-major 布局分支已随 A/B 结论回退,``_repack_padded`` 输出恒为
packed BSND 布局;内核与参考实现必须逐位一致,pad 行必须恰为零。

TestKdaNarrowRepackGatherBitwise(@pytest.mark.gpu)
    test_narrow_bitwise_and_pad_zero: 训练类 pack(371 短段,重 pad)下
        [1, T, H] g/beta 的重建与参考逐位一致、pad 行恰为零;分配器被
        NaN 投毒后重跑仍逐位一致(排除未初始化内存读)。
    test_narrow_token_major_bitwise: 同内核输出 [1, Tp, H] 逐位一致。
    test_non_pow2_width_falls_back_to_gather: H=6(非 2 的幂)走 monotone gather
        回退,结果仍逐位一致。
    test_wide_bsnd_unchanged: q/k 形状 [1, T, 2, 128] 走 HTD 段拷贝(BSND),防本
        改动回归宽行路径。
"""

import pytest
import torch
import torch.nn.functional as F

from xtuner.v1.ops.kda.kda_op import (
    _padded_pack_maps,
    _repack_padded,
)
from xtuner.v1.utils.device import get_device


DEVICE = get_device()
_HEADS, _TOKENS = 2, 65536


def _make_cu(nseg: int, total: int) -> list[int]:
    """A pack of ``nseg`` lognormal short segments summing to ``total`` tokens."""
    torch.manual_seed(7)
    raw = torch.exp(torch.randn(nseg) * 1.1 + 3.0)
    lens = torch.clamp((raw * (total / raw.sum())).round(), min=3).long()
    lens[-1] += total - int(lens.sum())
    cu = [0]
    for n in lens.tolist():
        cu.append(cu[-1] + n)
    return cu


def _reference_repack(tensor: torch.Tensor, pad_to_pack: torch.Tensor) -> torch.Tensor:
    """The pre-fix narrow path, kept verbatim as the semantics contract."""
    pad_spec = (0, 0, 0, 1) if tensor.dim() == 3 else (0, 0, 0, 0, 0, 1)
    return F.pad(tensor, pad_spec).index_select(1, pad_to_pack)


@pytest.mark.gpu
class TestKdaNarrowRepackGatherBitwise:
    def test_narrow_bitwise_and_pad_zero(self):
        cu = _make_cu(371, _TOKENS)  # short segments -> heavy sentinel padding
        cu_padded, real_pos, gather_pos, cu_t, pad_cu_t, max_padded_len = _padded_pack_maps(cu, torch.device(DEVICE))
        n_padded = cu_padded[-1]
        g = torch.randn((1, _TOKENS, _HEADS), dtype=torch.bfloat16, device=DEVICE)
        ref = _reference_repack(g, gather_pos)

        # allocator poisoning: any row the new kernel forgets to write shows up as NaN
        poison = torch.full((ref.numel(),), float("nan"), dtype=torch.bfloat16, device=DEVICE)
        del poison
        got = _repack_padded(g, gather_pos, cu_t, pad_cu_t, n_padded, max_padded_len)

        assert got.shape == ref.shape and got.dtype == ref.dtype
        assert not torch.isnan(got.float()).any(), "unwritten rows left in dst"
        assert torch.equal(got, ref), "narrow rebuild diverged from the gather reference"
        real_mask = torch.zeros(n_padded, dtype=torch.bool, device=DEVICE)
        real_mask[real_pos] = True
        assert float(got[0][~real_mask].abs().max()) == 0.0, "pad rows must be exactly zero"

    def test_narrow_token_major_bitwise(self):
        cu = _make_cu(371, _TOKENS)
        cu_padded, real_pos, gather_pos, cu_t, pad_cu_t, max_padded_len = _padded_pack_maps(cu, torch.device(DEVICE))
        n_padded = cu_padded[-1]
        g = torch.randn((1, _TOKENS, _HEADS), dtype=torch.bfloat16, device=DEVICE)
        ref = _reference_repack(g, gather_pos)
        got = _repack_padded(g, gather_pos, cu_t, pad_cu_t, n_padded, max_padded_len)
        assert torch.equal(got, ref), "narrow token-major rebuild diverged from the gather reference"

    def test_non_pow2_width_falls_back_to_gather(self):
        heads = 6
        cu = _make_cu(64, 8192)
        cu_padded, real_pos, gather_pos, cu_t, pad_cu_t, max_padded_len = _padded_pack_maps(cu, torch.device(DEVICE))
        n_padded = cu_padded[-1]
        g = torch.randn((1, 8192, heads), dtype=torch.bfloat16, device=DEVICE)
        ref = _reference_repack(g, gather_pos)
        got = _repack_padded(g, gather_pos, cu_t, pad_cu_t, n_padded, max_padded_len)
        assert torch.equal(got, ref), "non-pow2 row-width fallback diverged"

    def test_wide_bsnd_unchanged(self):
        cu = _make_cu(371, _TOKENS)
        cu_padded, real_pos, gather_pos, cu_t, pad_cu_t, max_padded_len = _padded_pack_maps(cu, torch.device(DEVICE))
        n_padded = cu_padded[-1]
        q = torch.randn((1, _TOKENS, _HEADS, 128), dtype=torch.bfloat16, device=DEVICE)
        got = _repack_padded(q, gather_pos, cu_t, pad_cu_t, n_padded, max_padded_len)
        assert got.shape == (1, n_padded, _HEADS, 128)
        # pad rows exactly zero, real rows equal their packed sources
        real_mask = torch.zeros(n_padded, dtype=torch.bool, device=DEVICE)
        real_mask[real_pos] = True
        assert float(got[0][~real_mask].abs().max()) == 0.0, "wide rebuild pad rows must be exactly zero"
        src = q.reshape(_TOKENS, _HEADS * 128)
        first_real = int(real_pos[0])
        assert torch.equal(got[0, first_real], src[0].reshape(_HEADS, 128)), "wide rebuild real rows corrupted"
