"""``act_fn_type_map_npu["silu"]`` → ``npu_silu`` 注册与对拍测试（ops/act_fn.py）。

``npu_silu`` 是 NPU map ``"silu"`` 条目的 ``npu_*`` 形态（与 npu_swiglu/npu_gelu 同构）：
NPU 张量走 ``torch_npu.npu_silu``（与 ``F.silu`` 在 NPU 上 Lower 到同一 aclnn Silu 核，
bf16/fp16/fp32 fwd+bwd 逐位一致），CPU 驻留张量或无 torch_npu 环境回退 ``native_silu``
（同一 eager 路径，逐位一致）。注册后 ``get_act_fn("silu")`` 与 ``get_gated_act_fn``
的 eager 回退闭包在 NPU 盒上解析到 ``npu_silu``——数值行为不变。

TestNpuSilu（CPU 盒可跑部分 + NPU 对拍）
    test_map_identity: ``act_fn_type_map_npu["silu"] is npu_silu``（cuda map 仍为 native）。
    test_cpu_fallback_bitwise: CPU 张量回退 native_silu，与 F.silu 逐位（bf16/fp32）。
    test_npu_bitwise（NPU）: bf16/fp16/fp32 的 fwd+bwd 与 F.silu 逐位一致。
    test_cpu_tensor_on_npu_box_fallback（NPU）: NPU 盒上 CPU 驻留张量走 native 回退（逐位）。
    test_get_act_fn_dispatch（NPU）: NPU 盒上 ``get_act_fn("silu")`` 即 ``npu_silu``，
        输出与 F.silu 逐位。
"""

import pytest
import torch
from torch.nn import functional as F

import xtuner.v1.ops.act_fn as af
from xtuner.v1.utils.device import get_device


DEVICE = get_device()
_requires_npu = pytest.mark.skipif(DEVICE != "npu", reason="requires an NPU (torch_npu.npu_silu)")

_SEED = 20261010
_DTYPES = [torch.bfloat16, torch.float16, torch.float32]
# 混合幅值（±3σ 覆盖 silu 两端渐近区）+ 精确 0/负 0 附近（silu 对称性敏感区）。
_SHAPE = (128, 512)


def _mixed(dtype: torch.dtype, seed: int, device: str = "cpu") -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    t = (torch.randn(_SHAPE, generator=g, dtype=torch.float32) * 3.0).to(dtype)
    t[0, :4] = torch.tensor([0.0, -0.0, 1e-3, -1e-3], dtype=dtype)
    return t.to(device)


class TestNpuSilu:
    """npu_silu 注册身份、回退矩阵与 NPU 对拍（fwd+bwd 逐位）。"""

    def test_map_identity(self) -> None:
        """NPU map 的 "silu" 即 npu_silu；cuda map 仍为 native_silu。"""
        assert af.act_fn_type_map_npu["silu"] is af.npu_silu
        assert af.act_fn_type_map_cuda["silu"] is af.native_silu

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_cpu_fallback_bitwise(self, dtype: torch.dtype) -> None:
        """CPU 张量回退 native_silu：与 F.silu 逐位一致（无 torch_npu 依赖）。"""
        x = _mixed(dtype, _SEED)
        assert torch.equal(af.npu_silu(x), F.silu(x))

    @_requires_npu
    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_npu_bitwise(self, dtype: torch.dtype) -> None:
        """NPU 张量：npu_silu 的 fwd+bwd 与 F.silu 逐位一致（bf16/fp16/fp32）。"""
        x = _mixed(dtype, _SEED + 1, device=DEVICE).requires_grad_(True)
        g = x.detach().clone().requires_grad_(True)
        out_f = af.npu_silu(x)
        out_e = F.silu(g)
        assert torch.equal(out_f, out_e)
        (out_f * torch.arange(out_f.numel(), device=DEVICE, dtype=dtype).reshape(_SHAPE)).sum().backward()
        (out_e * torch.arange(out_e.numel(), device=DEVICE, dtype=dtype).reshape(_SHAPE)).sum().backward()
        assert x.grad is not None and g.grad is not None
        assert torch.equal(x.grad, g.grad)

    @_requires_npu
    @pytest.mark.gpu
    def test_cpu_tensor_on_npu_box_fallback(self) -> None:
        """NPU 盒上 CPU 驻留张量走 native 回退：与 F.silu 逐位一致（防 host 指针进 NPU 后端）。"""
        x = _mixed(torch.bfloat16, _SEED + 2)
        assert torch.equal(af.npu_silu(x), F.silu(x))

    @_requires_npu
    @pytest.mark.gpu
    def test_get_act_fn_dispatch(self) -> None:
        """NPU 盒上 get_act_fn("silu") 解析到 npu_silu 且输出与 F.silu 逐位一致。"""
        fn = af.get_act_fn("silu")
        assert fn is af.npu_silu
        x = _mixed(torch.bfloat16, _SEED + 3, device=DEVICE)
        assert torch.equal(fn(x), F.silu(x))
