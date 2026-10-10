"""GLM-5.3 mHC fused split-Sinkhorn 缓冲缓存（wave-3 host-bound 消灭）的契约测试。

``xtuner/v1/ops/mhc_fused.py`` 的前向/反传把每调用的四次 ``F.pad`` 列填充与两块
``torch.empty`` 中间量换成模块级缓存：``_padded_comb_buffer``（pad 列一次性预填 0.0 /
-inf，每调用只重写前 hc_mult 列）与 ``_comb_scratch_buffer``（每调用被 part1 核整体重写的
comb_tmp）。``comb_flat_padded`` 有意保持 fresh ``torch.empty``——返回的 ``comb`` 是它的
切片视图，缓存会被下一次调用改写、脏读 autograd 图仍持有的张量。缓存超过 64 项时下一次
miss 先整体清空再写入。

数值基线：缓存只是去掉分配/填充的 host 派发，核输入逐位不变（实现侧 kernel-input spy
已证），因此前向输出必须与「缓存前实现」的 F.pad 复刻逐位一致。反传的 hc_scale / hc_base
梯度是 bf16 atomic_add 累加：累加次序随物理地址布局抖动 ±1 ULP（冷/热分配不同、跨实现、
跨进程皆然，run112 基线同现象），因此不与 F.pad 参考做梯度对拍，重复调用按
rtol=0.02 / atol=0.05 锚定（训练侧由 ±0.1 锚点容差吸收）；mixes 梯度无 atomic_add，
同进程重复调用逐位一致。

TestSinkhornCachedBuffers
    test_forward_matches_fpad_reference_bitwise: 缓存路径前向 == 旧实现复刻（part1 ->
        F.pad(-inf) -> fresh flat -> part2），pre/post/comb 逐位一致。
    test_comb_isolation_across_cache_hit: 不同输入的第二次调用（缓存命中）不得改写第一次
        返回的 comb（comb_flat_padded 不缓存的守卫），且两次 comb 确实不同（守卫有效性）。
    test_deterministic_repeat_calls: 同输入两次调用 pre/post/comb 逐位一致（含缓存命中）。
    test_varying_batch_seq_sizes_and_row_pad_tail: bs=1024/768（各自独立缓存项）逐位对拍；
        bs=700 非 group_part2=32 整除：前向仍逐位对拍（核内按 batch_seq_size 掩码），反传
        row_pad 分支端到端跑通、梯度有限且非零。
    test_cache_eviction_on_miss: 预置 65 项的缓存字典在下一次 miss 时被整体清空并只写入
        新项；同一形状的第二次调用复用同一缓冲对象（缓存确实命中、不再增长）。
    test_backward_deterministic_and_finite: 同进程重复调用，mixes 梯度逐位一致；hc_scale /
        hc_base 梯度为 bf16 atomic_add，按 rtol=0.02 / atol=0.05 锚定；三路梯度全部有限
        且非零。
"""

import pytest
import torch
import triton

import xtuner.v1.ops.mhc_fused as mhc_fused
from xtuner.v1.ops.mhc_fused import hc_split_sinkhorn_fused
from xtuner.v1.utils.device import get_device


DEVICE = get_device()
_requires_accelerator = pytest.mark.skipif(DEVICE == "cpu", reason="requires an NPU/CUDA accelerator")

_HC = 4
_FEAT = (2 + _HC) * _HC
_ITERS = 20
_EPS = 1e-6
_BLOCK_ALIGN = 8
_GROUP_PART1 = 256
_GROUP_PART2 = 32


def _make_args(bs: int, seed: int, *, requires_grad: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """生产量级 sinkhorn 入参（mixes / hc_scale / hc_base，bf16）。

    Args:
        bs (int): batch_seq_size（``b * s``，batch=1）。
        seed (int): 全局 RNG seed。
        requires_grad (bool): 三个入参是否 ``requires_grad_``。

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ``(mixes, hc_scale, hc_base)``。
    """
    torch.manual_seed(seed)
    mixes = torch.randn(1, bs, _FEAT, device=DEVICE, dtype=torch.bfloat16, requires_grad=requires_grad)
    hc_scale = torch.randn(3, device=DEVICE, dtype=torch.bfloat16, requires_grad=requires_grad)
    hc_base = torch.randn(_FEAT, device=DEVICE, dtype=torch.bfloat16, requires_grad=requires_grad)
    return mixes, hc_scale, hc_base


def _fpad_reference_forward(
    mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, iters: int = _ITERS, eps: float = _EPS
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """缓存前实现的逐位参考：per-call ``F.pad`` + fresh 中间量，launch 参数同生产前向。

    Args:
        mixes (torch.Tensor): mHC mix logits ``[1, bs, (2+hc_mult)*hc_mult]``。
        hc_scale (torch.Tensor): Per-output learned scales ``[3]``。
        hc_base (torch.Tensor): Learned biases ``[(2+hc_mult)*hc_mult]``。
        iters (int): Sinkhorn-Knopp iteration count.
        eps (float): Stabilising epsilon.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ``pre`` / ``post`` ``[1, bs, hc]``
            与 ``comb`` ``[1, bs, hc, hc]``（``[1, bs, hc, 8]`` 底 buffer 的切片视图）。
    """
    b, s, _ = mixes.shape
    bs = b * s
    mixes_flat = mixes.reshape(-1, _FEAT).contiguous()
    pre_flat = torch.empty((bs, _HC), dtype=mixes.dtype, device=mixes.device)
    post_flat = torch.empty((bs, _HC), dtype=mixes.dtype, device=mixes.device)
    comb_tmp = torch.empty((bs, _HC, _HC), dtype=mixes.dtype, device=mixes.device)

    mhc_fused._hc_split_sinkhorn_kernel_part1[(triton.cdiv(bs, _GROUP_PART1),)](
        mixes_flat, hc_scale, hc_base, pre_flat, post_flat, comb_tmp, bs, eps, _FEAT, _HC, _GROUP_PART1
    )
    comb_pad = torch.nn.functional.pad(comb_tmp, (0, _BLOCK_ALIGN - _HC), mode="constant", value=float("-inf"))
    flat = torch.empty((bs, _HC * _BLOCK_ALIGN), dtype=mixes.dtype, device=mixes.device)
    mhc_fused._hc_split_sinkhorn_kernel_part2[(triton.cdiv(bs, _GROUP_PART2),)](
        comb_pad, flat, bs, _HC, iters, eps, _GROUP_PART2, BLOCK_ALIGN=_BLOCK_ALIGN
    )
    return pre_flat.view(b, s, _HC), post_flat.view(b, s, _HC), flat.view(b, s, _HC, _BLOCK_ALIGN)[:, :, :, :_HC]


@pytest.mark.gpu
class TestSinkhornCachedBuffers:
    """缓存缓冲的前向逐位守卫、comb 隔离、确定性、尺寸矩阵、缓存上限与反传确定性。"""

    @_requires_accelerator
    def test_forward_matches_fpad_reference_bitwise(self):
        # 缓存（cold）路径前向必须与旧实现（per-call F.pad + fresh empty）逐位一致。
        mixes, hc_scale, hc_base = _make_args(1024, 0)
        pre, post, comb = hc_split_sinkhorn_fused(mixes, hc_scale, hc_base, _HC, _ITERS, _EPS)
        pre_r, post_r, comb_r = _fpad_reference_forward(mixes, hc_scale, hc_base)
        assert comb.shape == (1, 1024, _HC, _HC), comb.shape
        assert pre.dtype == post.dtype == comb.dtype == torch.bfloat16
        assert torch.equal(pre, pre_r), "pre != F.pad 参考"
        assert torch.equal(post, post_r), "post != F.pad 参考"
        assert torch.equal(comb, comb_r), "comb != F.pad 参考"

    @_requires_accelerator
    def test_comb_isolation_across_cache_hit(self):
        # 第二次调用（缓存命中）不得改写第一次返回的 comb：comb_flat_padded 必须保持 fresh。
        mixes1, hc_scale1, hc_base1 = _make_args(1024, 0)
        _, _, comb1 = hc_split_sinkhorn_fused(mixes1, hc_scale1, hc_base1, _HC, _ITERS, _EPS)
        snapshot = comb1.clone()

        mixes2, hc_scale2, hc_base2 = _make_args(1024, 7)
        _, _, comb2 = hc_split_sinkhorn_fused(mixes2, hc_scale2, hc_base2, _HC, _ITERS, _EPS)
        assert torch.equal(comb1, snapshot), "第二次调用改写了第一次返回的 comb（缓存脏写）"
        assert not torch.equal(comb2, comb1), "两次输入不同，comb 必须不同（守卫有效性）"

    @_requires_accelerator
    def test_deterministic_repeat_calls(self):
        # 同输入两次调用（第二次全缓存命中）必须逐位一致，缓存不能引入非确定性。
        mixes, hc_scale, hc_base = _make_args(1024, 42)
        out1 = hc_split_sinkhorn_fused(mixes, hc_scale, hc_base, _HC, _ITERS, _EPS)
        out2 = hc_split_sinkhorn_fused(mixes, hc_scale, hc_base, _HC, _ITERS, _EPS)
        for a, b, name in zip(out1, out2, ("pre", "post", "comb")):
            assert torch.equal(a, b), f"{name} 重复调用不一致"

    @_requires_accelerator
    def test_varying_batch_seq_sizes_and_row_pad_tail(self):
        # 不同 bs 各自建立缓存项且逐位正确；非 32 整除的 bs=700 覆盖前向掩码与反传 row_pad。
        for bs, seed in ((1024, 3), (768, 5)):
            mixes, hc_scale, hc_base = _make_args(bs, seed)
            pre, post, comb = hc_split_sinkhorn_fused(mixes, hc_scale, hc_base, _HC, _ITERS, _EPS)
            pre_r, post_r, comb_r = _fpad_reference_forward(mixes, hc_scale, hc_base)
            assert torch.equal(pre, pre_r) and torch.equal(post, post_r) and torch.equal(comb, comb_r), bs

        # bs=700：group_part2=32 不整除 —— 前向核内按 batch_seq_size 掩码，仍逐位对拍。
        mixes, hc_scale, hc_base = _make_args(700, 11, requires_grad=True)
        pre, post, comb = hc_split_sinkhorn_fused(mixes, hc_scale, hc_base, _HC, _ITERS, _EPS)
        pre_r, post_r, comb_r = _fpad_reference_forward(mixes, hc_scale, hc_base)
        assert torch.equal(pre, pre_r) and torch.equal(post, post_r) and torch.equal(comb, comb_r)

        # 反传 row_pad 分支端到端：不崩溃、梯度有限且非零（反传核在 batch_seq_size 之外
        # 的 pad 行会溢出读，实现端用 F.pad 把三个入参补齐到 group_part2 的倍数）。
        grads = torch.autograd.grad(comb.sum(), [mixes, hc_scale, hc_base])
        expected_grad_shapes = {"mixes": (1, 700, _FEAT), "hc_scale": (3,), "hc_base": (_FEAT,)}
        for g, (name, shape) in zip(grads, expected_grad_shapes.items()):
            assert tuple(g.shape) == shape, (name, g.shape)
            assert torch.isfinite(g.float()).all(), f"{name} 梯度含非有限值"
            assert g.float().abs().sum().item() > 0.0, f"{name} 梯度全零"

    @_requires_accelerator
    def test_cache_eviction_on_miss(self, monkeypatch: pytest.MonkeyPatch):
        # 预置 65 项的缓存字典：下一次 miss 必须先整体清空再写入（上限 64 的淘汰契约），
        # 且同一形状的第二次调用复用同一缓冲对象（缓存确实命中、不再增长）。
        sentinel = torch.zeros(1, device=DEVICE, dtype=torch.bfloat16)
        padded_fake: dict[tuple, torch.Tensor] = {
            ("stale", i, str(DEVICE), torch.bfloat16, 0.0): sentinel for i in range(65)
        }
        scratch_fake: dict[tuple, torch.Tensor] = {((i,), str(DEVICE), torch.bfloat16): sentinel for i in range(65)}
        monkeypatch.setattr(mhc_fused, "_PADDED_COMB_CACHE", padded_fake)
        monkeypatch.setattr(mhc_fused, "_COMB_SCRATCH_CACHE", scratch_fake)

        mixes, hc_scale, hc_base = _make_args(256, 21)
        hc_split_sinkhorn_fused(mixes, hc_scale, hc_base, _HC, _ITERS, _EPS)
        # 前向恰好各写一项：comb_tmp scratch 与 fwd_comb_tmp padded（其余 65 项全被清掉）。
        assert len(scratch_fake) == 1, sorted(scratch_fake)
        assert len(padded_fake) == 1, sorted(padded_fake)
        assert all(key[0] != "stale" for key in padded_fake), "padded 缓存未被清空"

        padded_buf = next(iter(padded_fake.values()))
        scratch_buf = next(iter(scratch_fake.values()))
        hc_split_sinkhorn_fused(mixes, hc_scale, hc_base, _HC, _ITERS, _EPS)
        assert len(padded_fake) == 1 and len(scratch_fake) == 1, "缓存命中后不应新增条目"
        assert next(iter(padded_fake.values())) is padded_buf, "padded 缓冲未被复用（缓存未命中）"
        assert next(iter(scratch_fake.values())) is scratch_buf, "scratch 缓冲未被复用（缓存未命中）"

    @_requires_accelerator
    def test_backward_deterministic_and_finite(self):
        # 同进程重复调用（第二次全缓存命中）：mixes 梯度逐位一致；hc_scale / hc_base 梯度
        # 走 bf16 atomic_add，累加顺序随地址布局抖动（冷/热分配不同 → ±1 bf16 ULP，
        # run112 基线同现象，训练侧由 ±0.1 锚点容差吸收），按 rtol=0.02 / atol=0.05 锚定。
        mixes, hc_scale, hc_base = _make_args(512, 11, requires_grad=True)
        _, _, comb1 = hc_split_sinkhorn_fused(mixes, hc_scale, hc_base, _HC, _ITERS, _EPS)
        g_comb = torch.randn_like(comb1)
        g1 = torch.autograd.grad(comb1, [mixes, hc_scale, hc_base], grad_outputs=g_comb)
        _, _, comb2 = hc_split_sinkhorn_fused(mixes, hc_scale, hc_base, _HC, _ITERS, _EPS)
        g2 = torch.autograd.grad(comb2, [mixes, hc_scale, hc_base], grad_outputs=g_comb)

        assert torch.equal(g1[0], g2[0]), "mixes 梯度重复调用不一致"
        torch.testing.assert_close(g1[1].float(), g2[1].float(), rtol=0.02, atol=0.05)
        torch.testing.assert_close(g1[2].float(), g2[2].float(), rtol=0.02, atol=0.05)
        for g, name in zip(g1, ("mixes", "hc_scale", "hc_base")):
            assert torch.isfinite(g.float()).all(), f"{name} 梯度含非有限值"
            assert g.float().abs().sum().item() > 0.0, f"{name} 梯度全零"
