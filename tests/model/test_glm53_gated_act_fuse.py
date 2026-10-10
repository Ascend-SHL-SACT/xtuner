"""GLM-5.3-Flash 融合激活核（ops/act_fn.py clamped-SwiGLU 双布局 triton）契约与对拍测试。

``_clamped_silu_mul_fwd_kernel`` / ``_clamped_silu_mul_bwd_kernel`` 用每方向一个 triton 核替换
eager 4-op 链（aclnnClamp x2 + aclnnSilu + aclnnMul），核内复刻 eager 链的 bf16 物化点（fp32
计算、silu 结果 bf16 舍入、fp32 乘、bf16 存）并复刻 aten clamp backward 的边界语义（掩码测
pre-clamp 值且含边界：g0<=limit、-limit<=u0<=limit → gate=limit 精确边界 dg 泄漏、
limit+1bf16ulp 处 dg==0）。服务两种布局：独立 gate/up（GS=US=C、U_OFF=0，shared/dense MLP）
与 fused ``[..., 2C]``（GS=US=2C、U_OFF=C，routed experts gate_up）。

入口与门控：``npu_clamped_swiglu_bitexact``（``_fused_layout_ok``）→ ``_ClampedSwigluFusedFn``；
``get_gated_act_fn("silu", swiglu_limit=...)``（NPU）→ ``_fused_gated_clamped_silu`` →
``_GatedClampedSiluFn``（``_gated_fused_ok``）；任何门控不满足回退 eager 链（应与 eager 逐位一致）。

实测精度基线（本盒 910C，前序数值探针已完成）：fwd 对 eager 链逐位一致（torch.equal）；bwd 与
eager autograd 的不等元素比例 ~4e-8（≈5/104M，源于 aclnnSiluBackward 内部代数序），且每个不等
元素差 == 该量级 1 bf16 ulp。bwd 断言：逐元素不等比例 < 1e-6（小张量放宽到绝对个数）+ 每个
不等元素差 == 1 bf16 ulp（相邻 bf16 值的间距 == 两侧 binade 间距的较小者）。

TestGatedActFuseGating（CPU 盒全量可跑；NPU 盒上回退路径以 patch _NPU_AVAILABLE=False 仿真）
    test_gated_ok_condition_matrix: ``_gated_fused_ok`` 条件矩阵（patch _NPU_AVAILABLE 测纯条件
        分支）：bf16 同形状连续 C%512==0 基线 True；fp32/fp16、形状不等、非连续、dim==0（不得在
        shape[-1] 抛 IndexError）、C%512!=0、C==0、numel==0 → False。
    test_fused_layout_ok_condition_matrix: ``_fused_layout_ok`` 条件矩阵：基线/split_dim=dim-1
        True；末维奇数、半维%512!=0、split_dim=-2、末维 0、numel 0、fp32、非连续、dim==0 → False。
    test_get_gated_act_fn_dispatch_availability: patch _NPU_AVAILABLE 仿真 NPU 条件：NPU 条件
        返回融合闭包（qualname 判别）、非 NPU 返回 plain lambda。
    test_gated_ok_unpatched_availability / test_fused_layout_ok_unpatched_availability: 未 patch
        的真实可用性语义（门控无 device 检查：CPU 盒 False、NPU 盒 True）。
    test_get_gated_act_fn_non_npu_not_fused: patch _NPU_AVAILABLE=False（非 NPU 条件）下
        ``get_gated_act_fn("silu", swiglu_limit=...)`` 返回 plain lambda（qualname 判别），输出
        与 eager 链逐位，spy 断言未派发融合 Function。
    test_get_gated_act_fn_plain_no_limit: 无 limit 仍是 plain silu*up（逐位）。
    test_no_npu_clamped_swiglu_fallback_bitwise: patch _NPU_AVAILABLE=False（非 NPU 条件）下
        ``npu_clamped_swiglu_bitexact`` == ``native_clamped_swiglu`` 逐位（多 shape + 3D +
        split_dim=-2），spy 断言未派发融合 Function（回退即同代码路径）。
TestGatedActFuseForward（NPU）
    test_fused_layout_forward_bitwise: A1 fused 布局 [(33940,3072)、(128,3072)、(17,2048)、
        (2,64,3072)]：``npu_clamped_swiglu_bitexact`` vs ``native_clamped_swiglu`` 逐位，spy
        断言派发一次；输入含 ~10% 饱和（|x|>limit）+ 精确边界（±limit、±limit+1bf16ulp）。
    test_gated_layout_forward_bitwise: A2 独立布局 [(8192,1536)、(17,2048)、(2,64,1536)]：
        ``get_gated_act_fn`` 闭包 vs eager 链逐位，spy 断言派发一次。
TestGatedActFuseBackward（NPU）
    test_fused_layout_backward_1ulp: 同 A1 shapes：fwd 逐位 + bwd 不等比例 <1e-6 且不等元素差
        == 1 bf16 ulp。
    test_gated_layout_backward_1ulp: 同 A2 shapes：同上（gate/up 双梯度）。
    test_boundary_leak_matches_eager: 边界泄漏回归（fused/gated 两布局）：gate=limit 精确边界
        dg 非零且与 eager 一致（1-ulp 内）；gate=limit+1bf16ulp dg==0；up 双侧同理（±limit
        泄漏、±(limit+1ulp) 为 0）。
TestGatedActFuseAutograd（NPU）
    test_gated_fn_full_chain_grads: ``_GatedClampedSiluFn.apply`` torch.autograd.grad 全链 vs
        eager autograd（1-ulp）。
    test_fused_fn_full_chain_grads: ``_ClampedSwigluFusedFn.apply`` 同上。
    test_backward_non_contiguous_grad_out: 非连续 grad_out（backward 内 .contiguous() 分支）：
        直接调 backward + autograd.grad 全链（1-ulp）。
    test_backward_needs_input_grad_contract: needs_input_grad 矩阵：单侧 False → 对应梯度 None、
        第三参 None；fused fn [0]=False → (None, None, None)。
TestGatedActFuseProduction（NPU）
    test_moe_act_fn_config_build_partial: ``MoEActFnConfig(act_type="clamped_swiglu",
        clip_limit=10.0).build()`` 的 partial（func 即 npu_clamped_swiglu_bitexact）调
        ``(gate_up_out, split_dim=-1)`` 与直接调逐位一致。
    test_act_fn_type_map_npu_identity: ``act_fn_type_map_npu["clamped_swiglu"] is
        npu_clamped_swiglu_bitexact``。
    test_get_gated_act_fn_closure_npu: NPU 上 ``get_gated_act_fn("silu", swiglu_limit=10.0)``
        闭包 fwd 逐位 + bwd 1-ulp，spy 断言派发。
TestGatedActFusePersistentGrid（NPU）
    test_fwd_kernel_grid_48_vs_24 / test_bwd_kernel_grid_48_vs_24: 直接调核，grid (48,)（aiv
        物理块数 cap）vs (24,)（陈旧缓存 aicore clamp）逐位一致（R=4096 → 512 个 tile 远超
        cap、每程序跨步多 tile；R=39 覆盖尾行 mask），并对 eager 参照（fwd 逐位 / bwd 1-ulp）。
"""

from types import ModuleType

import pytest
import torch
from torch.nn import functional as F

import xtuner.v1.ops.act_fn as af
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEActFnConfig
from xtuner.v1.utils.device import get_device


DEVICE = get_device()
# NPU-only, not just "any accelerator": every decorated case exercises the fused triton
# NPU backend (or asserts NPU-map dispatch), which does not exist on a CUDA box -- there
# _NPU_AVAILABLE is False and the assertions below would fail spuriously.
_requires_accelerator = pytest.mark.skipif(DEVICE != "npu", reason="requires an NPU (fused triton NPU backend)")

LIMIT = 10.0
# 10.0 的上一 bf16 值（binade [8,16) 步距 2^-4=0.0625）——clamp 掩码外的第一个可表示值。
_LIMIT_NEXT = 10.0625
# 混合输入：~90% N(0,1) + ~10% 饱和（|x|>limit，跨 clamp 上/下界）+ 精确边界值
# （±limit、±limit+1bf16ulp、±(limit-1ulp)）；边界泄漏语义由 _boundary_rows 单独构造覆盖。
_SEED = 20260930
# 形状矩阵：A1 = routed experts fused gate_up（(33940,3072) 生产形状、(17,2048) 非整 tile、
# (2,64,3072) 3D）；A2 = shared/dense MLP 独立 gate/up（(8192,1536) 生产 shared-expert）。
_FUSED_SHAPES: list[tuple[tuple[int, ...], int]] = [
    ((33940,), 3072),
    ((128,), 3072),
    ((17,), 2048),
    ((2, 64), 3072),
]
_GATED_SHAPES: list[tuple[tuple[int, ...], int]] = [
    ((8192,), 1536),
    ((17,), 2048),
    ((2, 64), 1536),
]
# bwd 不等比例界（实测 ~4e-8 ≈ 5/104M）；小张量按绝对个数放宽（合法 1-ulp 噪声非回归对象）。
_ULP_RATIO_BOUND = 1e-6
_ULP_ABS_TOL = 4
# 边界泄漏回归的行取值：gate clamp 只有 max=limit（负侧无边界，-limit 仍泄漏）；up clamp 双侧。
_BOUNDARY_GATE_ROWS = [LIMIT, _LIMIT_NEXT, LIMIT - 0.0625, 0.0, -3.0, -LIMIT]
_BOUNDARY_UP_ROWS = [LIMIT, _LIMIT_NEXT, -LIMIT, -_LIMIT_NEXT, LIMIT - 0.0625, -(LIMIT - 0.0625)]


def _shape_id(lead: tuple[int, ...], c: int) -> str:
    return "x".join(str(d) for d in (*lead, c))


def _mixed_bf16(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    """构造混合输入：~90% N(0,1) + ~10% 饱和（|x|>limit）+ 精确边界值（CPU 生成、确定性）。

    Args:
        shape (tuple[int, ...]): 目标形状（最后一维为 2C 时即 fused 布局输入）。
        seed (int): CPU Generator 种子，保证跨运行/跨设备一致。

    Returns:
        torch.Tensor: bf16 张量（CPU），含饱和值与 ±limit / ±(limit±1bf16ulp) 边界值。
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = 1
    for d in shape:
        n *= d
    t = torch.randn(n, generator=g, dtype=torch.float32).to(torch.bfloat16)
    sat = torch.rand(n, generator=g) < 0.10
    sign = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0)
    mag = LIMIT + 20.0 * torch.rand(n, generator=g)
    t[sat] = (sign[sat] * mag[sat]).to(torch.bfloat16)
    edges = torch.tensor(
        [LIMIT, -LIMIT, _LIMIT_NEXT, -_LIMIT_NEXT, LIMIT - 0.0625, -(LIMIT - 0.0625)],
        dtype=torch.bfloat16,
    )
    n_edge = min(n, 64)
    idx = torch.randint(0, n, (n_edge,), generator=g)
    t[idx] = edges[torch.arange(n_edge) % edges.numel()]
    return t.reshape(shape)


def _boundary_rows(vals: list[float], c: int) -> torch.Tensor:
    """每行一个常数值的 (len(vals), C) 张量，用于 clamp 边界泄漏的行级断言。

    Args:
        vals (list[float]): 每行的 gate/up 取值（含精确边界与边界外 1 bf16 ulp）。
        c (int): 通道数（须 %512==0 以通过融合门控）。

    Returns:
        torch.Tensor: bf16 CPU 张量，形状 (len(vals), c)，第 i 行全为 vals[i]。
    """
    return torch.tensor(vals, dtype=torch.bfloat16).reshape(-1, 1).expand(len(vals), c).contiguous()


def _non_contig(t: torch.Tensor) -> torch.Tensor:
    """构造 shape 不变、非连续的视图（transpose(-1,-2) 连续化后转回）。"""
    return t.transpose(-1, -2).contiguous().transpose(-1, -2)


def _spy_apply(monkeypatch: pytest.MonkeyPatch, fn_name: str, mod: ModuleType = af) -> list[int]:
    """把 ``mod.<fn_name>.apply`` 包一层计数 spy（每次派发追加 1），返回计数字列表。"""
    calls: list[int] = []
    orig_apply = getattr(mod, fn_name).apply

    class _SpyFunction:
        @staticmethod
        def apply(*args: object, **kwargs: object) -> object:
            calls.append(1)
            return orig_apply(*args, **kwargs)

    monkeypatch.setattr(mod, fn_name, _SpyFunction)
    return calls


def _bf16_key(t: torch.Tensor) -> torch.Tensor:
    """bf16 位型 → 单调整数键（IEEE sign-magnitude 重映射为全序）：相邻 bf16 值键差恰为 1。

    Args:
        t (torch.Tensor): bf16 张量。

    Returns:
        torch.Tensor: int64 键，-0.0 与 +0.0 同键；随数值单调，键差 == bf16 ulp 距离。
    """
    bits = t.contiguous().cpu().view(torch.int16).int()
    return torch.where(bits < 0, -32768 - bits, bits).long()


def _assert_grad_1ulp(
    got: torch.Tensor,
    ref: torch.Tensor,
    name: str,
    ratio_bound: float = _ULP_RATIO_BOUND,
    abs_tol: int = 0,
) -> None:
    """bwd 对拍：不等元素比例 < ratio_bound（或个数 <= abs_tol）且每个不等元素差 == 1 bf16 ulp。

    「1 bf16 ulp」按位型判定：got/ref 为相邻 bf16 网格点（键差 == 1，含跨 2 的幂边界的
    半间距情形）。

    Args:
        got (torch.Tensor): 融合路径梯度（bf16）。
        ref (torch.Tensor): eager autograd 梯度（bf16）。
        name (str): 断言消息中的梯度名。
        ratio_bound (float): 允许的不等元素比例上界。
        abs_tol (int): 小张量下允许的不等元素绝对个数上界（比例界在 <1e4 元素时过严）。
    """
    assert got.shape == ref.shape and got.dtype == ref.dtype, name
    gf, rf = got.float(), ref.float()
    neq = gf != rf
    n = int(neq.sum().item())
    ratio = n / neq.numel()
    assert ratio < ratio_bound or n <= abs_tol, (name, "unequal ratio", ratio, n, neq.numel())
    if n:
        ulps = (_bf16_key(got) - _bf16_key(ref)).abs()[neq.cpu()]
        bad = int((ulps != 1).sum().item())
        assert bad == 0, (name, "diff != 1 bf16 ulp", bad, ulps[:8])


def _fused_pair_grads(
    fused_x: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """fused 布局 (融合, eager) 两路反传：loss = (out * w).sum()，w 为共享 grad_out 权重。

    Args:
        fused_x (torch.Tensor): ``[..., 2C]`` 输入（无需 requires_grad，内部各自克隆叶子）。
        w (torch.Tensor): 与输出同形的权重张量（两侧共享，保证 grad_out 逐位一致）。

    Returns:
        tuple[torch.Tensor, ...]: (out_fused, dg_fused, out_eager, dg_eager)。
    """
    xf = fused_x.detach().clone().requires_grad_(True)
    out_f = af.npu_clamped_swiglu_bitexact(xf, -1, LIMIT)
    (out_f * w).sum().backward()
    xe = fused_x.detach().clone().requires_grad_(True)
    out_e = af.native_clamped_swiglu(xe, -1, LIMIT)
    (out_e * w).sum().backward()
    assert xf.grad is not None and xe.grad is not None
    return out_f, xf.grad, out_e, xe.grad


def _gated_pair_grads(
    gate: torch.Tensor, up: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """独立布局 (融合, eager) 两路反传：loss = (out * w).sum()，w 为共享 grad_out 权重。

    Args:
        gate (torch.Tensor): gate 投影输出 ``[..., C]``。
        up (torch.Tensor): up 投影输出，同 gate 形状。
        w (torch.Tensor): 与输出同形的权重张量（两侧共享）。

    Returns:
        tuple[torch.Tensor, ...]: (out_fused, dg_fused, du_fused, out_eager, dg_eager, du_eager)。
    """
    fn = af.get_gated_act_fn("silu", swiglu_limit=LIMIT)
    gf = gate.detach().clone().requires_grad_(True)
    uf = up.detach().clone().requires_grad_(True)
    out_f = fn(gf, uf)
    (out_f * w).sum().backward()
    ge = gate.detach().clone().requires_grad_(True)
    ue = up.detach().clone().requires_grad_(True)
    out_e = F.silu(ge.clamp(max=LIMIT)) * ue.clamp(min=-LIMIT, max=LIMIT)
    (out_e * w).sum().backward()
    assert gf.grad is not None and uf.grad is not None and ge.grad is not None and ue.grad is not None
    return out_f, gf.grad, uf.grad, out_e, ge.grad, ue.grad


class _FakeCtx:
    """autograd ctx 替身：直接调 Function.forward/backward 验证返回契约（不经引擎）。"""

    def __init__(self, needs: tuple[bool, ...]) -> None:
        self.needs_input_grad = needs
        self.limit = LIMIT
        self._saved: tuple[torch.Tensor, ...] = ()

    def save_for_backward(self, *ts: torch.Tensor) -> None:
        self._saved = ts

    @property
    def saved_tensors(self) -> tuple[torch.Tensor, ...]:
        return self._saved


class TestGatedActFuseGating:
    """门控矩阵与 CPU 回退：servable 判定、非 NPU 下必走 eager 且逐位一致。"""

    def test_gated_ok_condition_matrix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``_gated_fused_ok`` 纯条件分支（patch _NPU_AVAILABLE/_fused_device_ok）：基线 True、各退化输入 False。"""
        monkeypatch.setattr(af, "_NPU_AVAILABLE", True)
        monkeypatch.setattr(af, "_fused_device_ok", lambda t: True)
        gate = torch.zeros(4, 512, dtype=torch.bfloat16)
        up = torch.zeros(4, 512, dtype=torch.bfloat16)
        assert af._gated_fused_ok(gate, up) is True
        # 非 bf16
        assert af._gated_fused_ok(gate.float(), up.float()) is False
        assert af._gated_fused_ok(gate.half(), up.half()) is False
        # 形状不等
        assert af._gated_fused_ok(gate, up[:, :256]) is False
        # 非连续
        assert af._gated_fused_ok(_non_contig(gate), up) is False
        # dim == 0：短路与链在 dim>=1 处截断，不得在 shape[-1] 抛 IndexError
        d0 = torch.zeros((), dtype=torch.bfloat16)
        assert af._gated_fused_ok(d0, d0.clone()) is False
        # C % 512 != 0 / C == 0 / numel == 0
        c500 = torch.zeros(4, 500, dtype=torch.bfloat16)
        c0 = torch.zeros(4, 0, dtype=torch.bfloat16)
        n0 = torch.zeros(0, 512, dtype=torch.bfloat16)
        assert af._gated_fused_ok(c500, c500.clone()) is False
        assert af._gated_fused_ok(c0, c0.clone()) is False
        assert af._gated_fused_ok(n0, n0.clone()) is False

    def test_fused_layout_ok_condition_matrix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``_fused_layout_ok`` 纯条件分支（patch _NPU_AVAILABLE/_fused_device_ok）：基线 True、各退化输入 False。"""
        monkeypatch.setattr(af, "_NPU_AVAILABLE", True)
        monkeypatch.setattr(af, "_fused_device_ok", lambda t: True)
        x = torch.zeros(4, 1024, dtype=torch.bfloat16)
        assert af._fused_layout_ok(x, -1) is True
        assert af._fused_layout_ok(x, x.dim() - 1) is True
        # split_dim 不是末维
        assert af._fused_layout_ok(x, -2) is False
        # 末维奇数 / 半维 % 512 != 0
        assert af._fused_layout_ok(torch.zeros(4, 1023, dtype=torch.bfloat16), -1) is False
        assert af._fused_layout_ok(torch.zeros(4, 1022, dtype=torch.bfloat16), -1) is False
        # 非 bf16 / 非连续 / dim == 0
        assert af._fused_layout_ok(x.float(), -1) is False
        assert af._fused_layout_ok(_non_contig(x), -1) is False
        assert af._fused_layout_ok(torch.zeros((), dtype=torch.bfloat16), -1) is False
        # 末维 0 / numel 0
        assert af._fused_layout_ok(torch.zeros(4, 0, dtype=torch.bfloat16), -1) is False
        assert af._fused_layout_ok(torch.zeros(0, 1024, dtype=torch.bfloat16), -1) is False

    def test_get_gated_act_fn_dispatch_availability(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """仿真 NPU 条件（patch _NPU_AVAILABLE）：NPU 条件返回融合闭包、非 NPU 返回 plain lambda。"""
        monkeypatch.setattr(af, "_NPU_AVAILABLE", True)
        fn_on = af.get_gated_act_fn("silu", swiglu_limit=LIMIT)
        assert fn_on.__qualname__ == "_fused_gated_clamped_silu.<locals>._act"
        monkeypatch.setattr(af, "_NPU_AVAILABLE", False)
        fn_off = af.get_gated_act_fn("silu", swiglu_limit=LIMIT)
        assert fn_off.__qualname__ == "get_gated_act_fn.<locals>.<lambda>"

    def test_gated_ok_unpatched_availability(self) -> None:
        """未 patch 的真实 _NPU_AVAILABLE 语义：NPU 盒上 NPU 张量 True、CPU 盒 False；
        CPU 驻留张量在 NPU 盒上也 False（门控含张量级 device 检查，防 host 指针进 NPU 后端）。"""
        gate = torch.zeros(4, 512, dtype=torch.bfloat16, device=DEVICE)
        assert af._gated_fused_ok(gate, gate.clone()) == (DEVICE == "npu")
        gate_cpu = torch.zeros(4, 512, dtype=torch.bfloat16)
        assert not af._gated_fused_ok(gate_cpu, gate_cpu.clone())
        x = torch.zeros(4, 1024, dtype=torch.bfloat16, device=DEVICE)
        assert af._fused_layout_ok(x, -1) == (DEVICE == "npu")

    def test_fused_layout_ok_unpatched_availability(self) -> None:
        """同上，fused 布局门控（与上一断言互为冗余守卫，防条件重排回归）。"""
        x = torch.zeros(4, 1024, dtype=torch.bfloat16, device=DEVICE)
        assert af._fused_layout_ok(x, -1) == (DEVICE == "npu")
        x_cpu = torch.zeros(4, 1024, dtype=torch.bfloat16)
        assert not af._fused_layout_ok(x_cpu, -1)

    def test_get_gated_act_fn_non_npu_not_fused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """非 NPU 条件（patch _NPU_AVAILABLE=False）下 get_gated_act_fn 返回 plain lambda：与 eager 链逐位、未派发。"""
        monkeypatch.setattr(af, "_NPU_AVAILABLE", False)
        calls = _spy_apply(monkeypatch, "_GatedClampedSiluFn")
        fn = af.get_gated_act_fn("silu", swiglu_limit=LIMIT)
        assert fn.__qualname__ == "get_gated_act_fn.<locals>.<lambda>"
        gate = _mixed_bf16((128, 1536), _SEED + 1)
        up = _mixed_bf16((128, 1536), _SEED + 2)
        out = fn(gate, up)
        ref = F.silu(gate.clamp(max=LIMIT)) * up.clamp(min=-LIMIT, max=LIMIT)
        assert calls == []
        assert out.shape == ref.shape and out.dtype == ref.dtype
        assert torch.equal(out, ref)

    def test_get_gated_act_fn_plain_no_limit(self) -> None:
        """无 limit 仍是 plain silu*up（逐位）。"""
        fn = af.get_gated_act_fn("silu")
        gate = _mixed_bf16((17, 512), _SEED + 3)
        up = _mixed_bf16((17, 512), _SEED + 4)
        assert torch.equal(fn(gate, up), F.silu(gate) * up)

    def test_no_npu_clamped_swiglu_fallback_bitwise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """非 NPU 条件（patch _NPU_AVAILABLE=False）下 npu_clamped_swiglu_bitexact == native 逐位（回退即同代码路径）。"""
        monkeypatch.setattr(af, "_NPU_AVAILABLE", False)
        calls = _spy_apply(monkeypatch, "_ClampedSwigluFusedFn")
        shapes: list[tuple[tuple[int, ...], int]] = [((4,), 512), ((17,), 2048), ((2, 5), 500)]
        for lead, c in shapes:
            fused_x = _mixed_bf16(lead + (2 * c,), _SEED + 5)
            out = af.npu_clamped_swiglu_bitexact(fused_x, -1, LIMIT)
            ref = af.native_clamped_swiglu(fused_x, -1, LIMIT)
            assert out.shape == ref.shape and out.dtype == ref.dtype
            assert torch.equal(out, ref), (lead, c)
        # 3D + 非末维 split_dim 亦走 eager 且逐位（中维取偶数，chunk 对半不报错）
        fused_3d = _mixed_bf16((2, 8, 1024), _SEED + 6)
        assert torch.equal(
            af.npu_clamped_swiglu_bitexact(fused_3d, -2, LIMIT),
            af.native_clamped_swiglu(fused_3d, -2, LIMIT),
        )
        assert calls == []


class TestGatedActFuseForward:
    """fwd 对拍：fused/独立两布局，融合核 vs eager 链逐位（含饱和与精确边界输入）。"""

    @_requires_accelerator
    @pytest.mark.gpu
    @pytest.mark.parametrize(("lead", "c"), _FUSED_SHAPES, ids=[_shape_id(*s) for s in _FUSED_SHAPES])
    def test_fused_layout_forward_bitwise(
        self, monkeypatch: pytest.MonkeyPatch, lead: tuple[int, ...], c: int
    ) -> None:
        """fused 布局：npu_clamped_swiglu_bitexact 派发一次且输出与 native_clamped_swiglu 逐位一致。"""
        fused_x = _mixed_bf16(lead + (2 * c,), _SEED).to(DEVICE)
        calls = _spy_apply(monkeypatch, "_ClampedSwigluFusedFn")
        out = af.npu_clamped_swiglu_bitexact(fused_x, -1, LIMIT)
        ref = af.native_clamped_swiglu(fused_x, -1, LIMIT)
        assert calls == [1]
        assert out.shape == ref.shape and out.dtype == ref.dtype
        assert torch.equal(out, ref), _shape_id(lead, c)

    @_requires_accelerator
    @pytest.mark.gpu
    @pytest.mark.parametrize(("lead", "c"), _GATED_SHAPES, ids=[_shape_id(*s) for s in _GATED_SHAPES])
    def test_gated_layout_forward_bitwise(
        self, monkeypatch: pytest.MonkeyPatch, lead: tuple[int, ...], c: int
    ) -> None:
        """独立布局：get_gated_act_fn 闭包派发 _GatedClampedSiluFn 一次且输出与 eager 链逐位一致。"""
        gate = _mixed_bf16(lead + (c,), _SEED + 51).to(DEVICE)
        up = _mixed_bf16(lead + (c,), _SEED + 52).to(DEVICE)
        calls = _spy_apply(monkeypatch, "_GatedClampedSiluFn")
        fn = af.get_gated_act_fn("silu", swiglu_limit=LIMIT)
        out = fn(gate, up)
        ref = F.silu(gate.clamp(max=LIMIT)) * up.clamp(min=-LIMIT, max=LIMIT)
        assert calls == [1]
        assert out.shape == ref.shape and out.dtype == ref.dtype
        assert torch.equal(out, ref), _shape_id(lead, c)


class TestGatedActFuseBackward:
    """bwd 对拍：fwd 逐位 + bwd 不等比例 <1e-6 且不等元素差 == 1 bf16 ulp + 边界泄漏回归。"""

    @_requires_accelerator
    @pytest.mark.gpu
    @pytest.mark.parametrize(("lead", "c"), _FUSED_SHAPES, ids=[_shape_id(*s) for s in _FUSED_SHAPES])
    def test_fused_layout_backward_1ulp(self, lead: tuple[int, ...], c: int) -> None:
        """fused 布局反传：fwd 逐位 + dg 不等比例 <1e-6 且不等元素差 == 1 bf16 ulp。"""
        fused_x = _mixed_bf16(lead + (2 * c,), _SEED + 61).to(DEVICE)
        w = _mixed_bf16(lead + (c,), _SEED + 62).to(DEVICE)
        out_f, dg_f, out_e, dg_e = _fused_pair_grads(fused_x, w)
        assert torch.equal(out_f, out_e), _shape_id(lead, c)
        _assert_grad_1ulp(dg_f, dg_e, f"dg@{_shape_id(lead, c)}")

    @_requires_accelerator
    @pytest.mark.gpu
    @pytest.mark.parametrize(("lead", "c"), _GATED_SHAPES, ids=[_shape_id(*s) for s in _GATED_SHAPES])
    def test_gated_layout_backward_1ulp(self, lead: tuple[int, ...], c: int) -> None:
        """独立布局反传：fwd 逐位 + dg/du 不等比例 <1e-6 且不等元素差 == 1 bf16 ulp。"""
        gate = _mixed_bf16(lead + (c,), _SEED + 63).to(DEVICE)
        up = _mixed_bf16(lead + (c,), _SEED + 64).to(DEVICE)
        w = _mixed_bf16(lead + (c,), _SEED + 65).to(DEVICE)
        out_f, dg_f, du_f, out_e, dg_e, du_e = _gated_pair_grads(gate, up, w)
        assert torch.equal(out_f, out_e), _shape_id(lead, c)
        _assert_grad_1ulp(dg_f, dg_e, f"dg@{_shape_id(lead, c)}")
        _assert_grad_1ulp(du_f, du_e, f"du@{_shape_id(lead, c)}")

    @_requires_accelerator
    @pytest.mark.gpu
    @pytest.mark.parametrize("fused", [True, False], ids=["fused", "gated"])
    def test_boundary_leak_matches_eager(self, fused: bool) -> None:
        """clamp 边界泄漏回归：gate=limit 精确边界 dg 非零（与 eager 一致）；limit+1bf16ulp dg==0；up 双侧同理。

        行矩阵（每行一个常数取值，C=512）：gate 侧行取 [limit, limit+1ulp, limit-1ulp, 0, -3,
        -limit]（gate clamp 只有 max，负侧无边界）；up 侧行取 [±limit, ±(limit+1ulp),
        ±(limit-1ulp)]（up clamp 双侧）。全零 dout 行级判定由 grad_out=ones 权重给出。
        """
        c = 512
        assert torch.tensor(_LIMIT_NEXT, dtype=torch.bfloat16).item() == _LIMIT_NEXT
        gate_b = _boundary_rows([LIMIT, _LIMIT_NEXT, LIMIT - 0.0625, 0.0, -3.0, -LIMIT], c).to(DEVICE)
        up_b = _boundary_rows(_BOUNDARY_UP_ROWS, c).to(DEVICE)
        w = torch.ones(len(_BOUNDARY_GATE_ROWS), c, dtype=torch.bfloat16, device=DEVICE)
        if fused:
            fused_x = torch.cat([gate_b, up_b], dim=-1)
            out_f, dg_f, out_e, dg_e = _fused_pair_grads(fused_x, w)
            du_f = dg_f[:, c:]
            du_e = dg_e[:, c:]
            dg_f, dg_e = dg_f[:, :c], dg_e[:, :c]
        else:
            out_f, dg_f, du_f, out_e, dg_e, du_e = _gated_pair_grads(gate_b, up_b, w)
        assert torch.equal(out_f, out_e), fused
        _assert_grad_1ulp(dg_f, dg_e, "dg@boundary", abs_tol=_ULP_ABS_TOL)
        _assert_grad_1ulp(du_f, du_e, "du@boundary", abs_tol=_ULP_ABS_TOL)
        for r, v in enumerate(_BOUNDARY_GATE_ROWS):
            leaked = dg_f[r].abs().sum().item() > 0
            assert leaked == (v <= LIMIT), ("gate row", r, v, leaked)
            assert (dg_e[r] != 0).any().item() == (v <= LIMIT), ("gate row eager", r, v)
        for r, v in enumerate(_BOUNDARY_UP_ROWS):
            leaked = du_f[r].abs().sum().item() > 0
            assert leaked == (abs(v) <= LIMIT), ("up row", r, v, leaked)
            assert (du_e[r] != 0).any().item() == (abs(v) <= LIMIT), ("up row eager", r, v)


class TestGatedActFuseAutograd:
    """autograd 契约：全链 grad、非连续 grad_out、needs_input_grad 矩阵。"""

    @_requires_accelerator
    @pytest.mark.gpu
    def test_gated_fn_full_chain_grads(self) -> None:
        """``_GatedClampedSiluFn.apply`` torch.autograd.grad 全链 vs eager autograd（1-ulp）。"""
        lead, c = (128,), 3072
        gate = _mixed_bf16(lead + (c,), _SEED + 71).to(DEVICE)
        up = _mixed_bf16(lead + (c,), _SEED + 72).to(DEVICE)
        go = _mixed_bf16(lead + (c,), _SEED + 73).to(DEVICE)
        gf = gate.clone().requires_grad_(True)
        uf = up.clone().requires_grad_(True)
        out_f = af._GatedClampedSiluFn.apply(gf, uf, LIMIT)
        grads_f = torch.autograd.grad(out_f, [gf, uf], grad_outputs=[go])
        ge = gate.clone().requires_grad_(True)
        ue = up.clone().requires_grad_(True)
        out_e = F.silu(ge.clamp(max=LIMIT)) * ue.clamp(min=-LIMIT, max=LIMIT)
        grads_e = torch.autograd.grad(out_e, [ge, ue], grad_outputs=[go])
        _assert_grad_1ulp(grads_f[0], grads_e[0], "dg", abs_tol=_ULP_ABS_TOL)
        _assert_grad_1ulp(grads_f[1], grads_e[1], "du", abs_tol=_ULP_ABS_TOL)

    @_requires_accelerator
    @pytest.mark.gpu
    def test_fused_fn_full_chain_grads(self) -> None:
        """``_ClampedSwigluFusedFn.apply`` torch.autograd.grad 全链 vs eager autograd（1-ulp）。"""
        lead, c = (128,), 3072
        fused_x = _mixed_bf16(lead + (2 * c,), _SEED + 74).to(DEVICE)
        go = _mixed_bf16(lead + (c,), _SEED + 75).to(DEVICE)
        xf = fused_x.clone().requires_grad_(True)
        out_f = af._ClampedSwigluFusedFn.apply(xf, -1, LIMIT)
        (dx_f,) = torch.autograd.grad(out_f, [xf], grad_outputs=[go])
        xe = fused_x.clone().requires_grad_(True)
        out_e = af.native_clamped_swiglu(xe, -1, LIMIT)
        (dx_e,) = torch.autograd.grad(out_e, [xe], grad_outputs=[go])
        _assert_grad_1ulp(dx_f, dx_e, "dx", abs_tol=_ULP_ABS_TOL)

    @_requires_accelerator
    @pytest.mark.gpu
    def test_backward_non_contiguous_grad_out(self) -> None:
        """非连续 grad_out：backward 内 .contiguous() 分支——直接调 backward 契约 + 全链对拍。"""
        rows, c = 128, 3072
        go_nc = _mixed_bf16((c, rows), _SEED + 76).to(DEVICE).t()
        assert not go_nc.is_contiguous()
        gate = _mixed_bf16((rows, c), _SEED + 77).to(DEVICE)
        up = _mixed_bf16((rows, c), _SEED + 78).to(DEVICE)
        ctx = _FakeCtx((True, True, True))
        af._GatedClampedSiluFn.forward(ctx, gate, up, LIMIT)
        dg_f, du_f, none = af._GatedClampedSiluFn.backward(ctx, go_nc)
        assert none is None and dg_f is not None and du_f is not None
        ge = gate.clone().requires_grad_(True)
        ue = up.clone().requires_grad_(True)
        out_e = F.silu(ge.clamp(max=LIMIT)) * ue.clamp(min=-LIMIT, max=LIMIT)
        grads_e = torch.autograd.grad(out_e, [ge, ue], grad_outputs=[go_nc])
        _assert_grad_1ulp(dg_f, grads_e[0], "dg_nc", abs_tol=_ULP_ABS_TOL)
        _assert_grad_1ulp(du_f, grads_e[1], "du_nc", abs_tol=_ULP_ABS_TOL)
        # fused 布局同一非连续 grad_out
        fused_x = _mixed_bf16((rows, 2 * c), _SEED + 79).to(DEVICE)
        ctx_f = _FakeCtx((True, True, True))
        af._ClampedSwigluFusedFn.forward(ctx_f, fused_x, -1, LIMIT)
        (dx_f, none1, none2) = af._ClampedSwigluFusedFn.backward(ctx_f, go_nc)
        assert none1 is None and none2 is None and dx_f is not None
        xe = fused_x.clone().requires_grad_(True)
        (dx_e,) = torch.autograd.grad(af.native_clamped_swiglu(xe, -1, LIMIT), [xe], grad_outputs=[go_nc])
        _assert_grad_1ulp(dx_f, dx_e, "dx_nc", abs_tol=_ULP_ABS_TOL)

    @_requires_accelerator
    @pytest.mark.gpu
    def test_backward_needs_input_grad_contract(self) -> None:
        """needs_input_grad 矩阵：单侧 False → 对应梯度 None、第三参 None；fused fn [0]=False → 全 None。"""
        rows, c = 128, 3072
        gate = _mixed_bf16((rows, c), _SEED + 81).to(DEVICE)
        up = _mixed_bf16((rows, c), _SEED + 82).to(DEVICE)
        go = _mixed_bf16((rows, c), _SEED + 83).to(DEVICE)
        for needs in [(True, True, True), (False, True, True), (True, False, True)]:
            ctx = _FakeCtx(needs)
            af._GatedClampedSiluFn.forward(ctx, gate, up, LIMIT)
            grads = af._GatedClampedSiluFn.backward(ctx, go)
            assert isinstance(grads, tuple) and len(grads) == 3
            assert grads[2] is None
            assert (grads[0] is None) == (not needs[0]), needs
            assert (grads[1] is None) == (not needs[1]), needs
            if needs[0]:
                assert grads[0].shape == gate.shape and grads[0].dtype == gate.dtype
            if needs[1]:
                assert grads[1].shape == up.shape and grads[1].dtype == up.dtype
        fused_x = _mixed_bf16((rows, 2 * c), _SEED + 84).to(DEVICE)
        for needs in [(True, True, True), (False, True, True)]:
            ctx = _FakeCtx(needs)
            af._ClampedSwigluFusedFn.forward(ctx, fused_x, -1, LIMIT)
            grads = af._ClampedSwigluFusedFn.backward(ctx, go)
            assert isinstance(grads, tuple) and len(grads) == 3
            if needs[0]:
                assert grads[0].shape == fused_x.shape and grads[0].dtype == fused_x.dtype
                assert grads[1] is None and grads[2] is None
            else:
                assert grads == (None, None, None)


class TestGatedActFuseProduction:
    """生产入口接线：MoEActFnConfig partial、act_fn_type_map_npu 身份、get_gated_act_fn 闭包。"""

    @_requires_accelerator
    @pytest.mark.gpu
    def test_moe_act_fn_config_build_partial(self) -> None:
        """``MoEActFnConfig(act_type="clamped_swiglu", clip_limit=10.0).build()`` 接线到融合入口且逐位一致。"""
        built = MoEActFnConfig(act_type="clamped_swiglu", clip_limit=LIMIT).build()
        assert built.keywords == {"limit": LIMIT}
        assert built.func is af.act_fn_type_map_npu["clamped_swiglu"]
        lead, c = (2048,), 3072
        gate_up = _mixed_bf16(lead + (2 * c,), _SEED + 91).to(DEVICE)
        out = built(gate_up, split_dim=-1)
        ref = af.npu_clamped_swiglu_bitexact(gate_up, -1, LIMIT)
        assert out.shape == ref.shape and out.dtype == ref.dtype
        assert torch.equal(out, ref)

    @_requires_accelerator
    @pytest.mark.gpu
    def test_act_fn_type_map_npu_identity(self) -> None:
        """``act_fn_type_map_npu["clamped_swiglu"] is npu_clamped_swiglu_bitexact``。"""
        assert af.act_fn_type_map_npu["clamped_swiglu"] is af.npu_clamped_swiglu_bitexact

    @_requires_accelerator
    @pytest.mark.gpu
    def test_get_gated_act_fn_closure_npu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """NPU 上 get_gated_act_fn("silu", swiglu_limit=10.0) 闭包：fwd 逐位 + bwd 1-ulp + 派发一次。"""
        calls = _spy_apply(monkeypatch, "_GatedClampedSiluFn")
        fn = af.get_gated_act_fn("silu", swiglu_limit=LIMIT)
        assert fn.__qualname__ == "_fused_gated_clamped_silu.<locals>._act"
        lead, c = (8192,), 1536
        gate = _mixed_bf16(lead + (c,), _SEED + 92).to(DEVICE)
        up = _mixed_bf16(lead + (c,), _SEED + 93).to(DEVICE)
        out = fn(gate, up)
        ref = F.silu(gate.clamp(max=LIMIT)) * up.clamp(min=-LIMIT, max=LIMIT)
        assert calls == [1]
        assert torch.equal(out, ref)
        # bwd：闭包链 vs eager 链（1-ulp）
        go = _mixed_bf16(lead + (c,), _SEED + 94).to(DEVICE)
        gf = gate.clone().requires_grad_(True)
        uf = up.clone().requires_grad_(True)
        grads_f = torch.autograd.grad(fn(gf, uf), [gf, uf], grad_outputs=[go])
        ge = gate.clone().requires_grad_(True)
        ue = up.clone().requires_grad_(True)
        eager_out = F.silu(ge.clamp(max=LIMIT)) * ue.clamp(min=-LIMIT, max=LIMIT)
        grads_e = torch.autograd.grad(eager_out, [ge, ue], grad_outputs=[go])
        _assert_grad_1ulp(grads_f[0], grads_e[0], "dg", abs_tol=_ULP_ABS_TOL)
        _assert_grad_1ulp(grads_f[1], grads_e[1], "du", abs_tol=_ULP_ABS_TOL)


class TestGatedActFusePersistentGrid:
    """持久化网格回归：直接调核，grid (48,) vs (24,) 逐位一致（含尾行 mask）。"""

    @_requires_accelerator
    @pytest.mark.gpu
    @pytest.mark.parametrize("rows", [4096, 39], ids=["r4096", "tail39"])
    def test_fwd_kernel_grid_48_vs_24(self, rows: int) -> None:
        """fwd 核直接调：grid (48,) vs (24,) 逐位一致（R=4096 每程序跨步多 tile；R=39 尾行 mask）。"""
        c = 512
        fused_x = _mixed_bf16((rows, 2 * c), _SEED + 101).to(DEVICE)
        ref = af.native_clamped_swiglu(fused_x, -1, LIMIT)
        outs: list[torch.Tensor] = []
        for progs in (af._GATED_ACT_MAX_PROGS, af._GATED_ACT_MAX_PROGS // 2):
            out = torch.empty(rows, c, dtype=torch.bfloat16, device=DEVICE)
            af._clamped_silu_mul_fwd_kernel[(progs,)](
                fused_x, fused_x, out, rows, c, 2 * c, 2 * c, c, LIMIT, BR=8, BC=512, num_warps=8
            )
            outs.append(out)
        assert torch.equal(outs[0], outs[1]), rows
        assert torch.equal(outs[0], ref), rows

    @_requires_accelerator
    @pytest.mark.gpu
    @pytest.mark.parametrize("rows", [4096, 39], ids=["r4096", "tail39"])
    def test_bwd_kernel_grid_48_vs_24(self, rows: int) -> None:
        """bwd 核直接调：grid (48,) vs (24,) 逐位一致，且对 eager autograd 参照 1-ulp。"""
        c = 512
        fused_x = _mixed_bf16((rows, 2 * c), _SEED + 102).to(DEVICE)
        dout = _mixed_bf16((rows, c), _SEED + 103).to(DEVICE)
        dxs: list[torch.Tensor] = []
        for progs in (af._GATED_ACT_MAX_PROGS, af._GATED_ACT_MAX_PROGS // 2):
            dx = torch.empty(rows, 2 * c, dtype=torch.bfloat16, device=DEVICE)
            af._clamped_silu_mul_bwd_kernel[(progs,)](
                fused_x,
                fused_x,
                dout,
                dx,
                dx,
                rows,
                c,
                2 * c,
                2 * c,
                c,
                2 * c,
                2 * c,
                c,
                LIMIT,
                BR=8,
                BC=512,
                num_warps=8,
            )
            dxs.append(dx)
        assert torch.equal(dxs[0], dxs[1]), rows
        xe = fused_x.detach().clone().requires_grad_(True)
        af.native_clamped_swiglu(xe, -1, LIMIT).backward(dout)
        assert xe.grad is not None
        _assert_grad_1ulp(dxs[0][:, :c], xe.grad[:, :c], "dg", abs_tol=_ULP_ABS_TOL)
        _assert_grad_1ulp(dxs[0][:, c:], xe.grad[:, c:], "du", abs_tol=_ULP_ABS_TOL)
