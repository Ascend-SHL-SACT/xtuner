import os
from functools import partial
from typing import Callable

import torch
import triton
import triton.language as tl
from torch.nn import functional as F


# Kill switch for the fused KDA sigmoid-mul gate kernels (fwd+bwd): ``0`` restores the
# eager sigmoid/mul chain -- a bit-exactness escape hatch, default on.
_GATED_ACT_FUSED = os.environ.get("XTUNER_GLM53_GATED_ACT_FUSED", "1") == "1"

# Persistent-grid cap, at the aiv-mode physical block count (48): binaries compiled
# WITH TRITON_ALL_BLOCKS_PARALLEL=1 replay correctly past the physical count (see
# kda_op._SEG_COPY_MAX_PROGS), while a stale cache compiled without the flag still
# silently drops blocks past the aicore-mode clamp of 24 -- the same guard every
# capped kernel here carries.
_GATED_ACT_MAX_PROGS = 48

try:
    _NPU_AVAILABLE = torch.npu.is_available()  # type: ignore[attr-defined]
except (ImportError, AttributeError):  # pragma: no cover - CPU-only environments
    _NPU_AVAILABLE = False


def native_swiglu(fused_x: torch.Tensor, split_dim=-1) -> torch.Tensor:
    x1, x2 = torch.chunk(fused_x, 2, dim=split_dim)
    return F.silu(x1) * x2


def npu_swiglu(fused_x: torch.Tensor, split_dim: int = -1) -> torch.Tensor:
    import torch_npu

    return torch_npu.npu_swiglu(fused_x, dim=split_dim)


def native_clipped_swiglu(fused_x: torch.Tensor, split_dim=-1, alpha=1.702, limit=7) -> torch.Tensor:
    gate, up = torch.chunk(fused_x, 2, dim=split_dim)
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    gated_output = (up + 1) * glu
    return gated_output


def native_clamped_swiglu(fused_x: torch.Tensor, split_dim=-1, limit=10.0) -> torch.Tensor:
    """GLM-5.3-Flash's clamped SwiGLU: ``silu(clamp_max(gate, limit)) *
    clamp(up, -limit, limit)``.

    Distinct from :func:`native_clipped_swiglu` (GPT-OSS's ``(up+1) * gate * sigmoid(alpha*gate)``)
    -- same activation family, different formula, so this is a separate function rather than a
    parameterization of the existing one.
    """
    gate, up = torch.chunk(fused_x, 2, dim=split_dim)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


def get_gated_act_fn(hidden_act: str, swiglu_limit: float | None = None) -> Callable[..., torch.Tensor]:
    """Build the gated activation for MLPs that keep gate/up in **separate**
    projections.

    The fused-``gate_up`` counterpart is :class:`~xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEActFnConfig`
    (routed experts); this is its unfused twin, used by ``DenseMLP`` / ``MoEMLP``. Resolving the
    variant once here keeps the formula -- and the branch -- out of every MLP's forward.

    Args:
        hidden_act (str): Elementwise activation applied to the gate, as named in
            ``transformers.activations.ACT2CLS``.
        swiglu_limit (float | None): ``None`` gives the plain ``act_fn(gate) * up`` every model
            uses. A float selects GLM-5.3-Flash's clamped SwiGLU,
            ``act_fn(clamp_max(gate, limit)) * clamp(up, -limit, limit)``.

    Returns:
        Callable[..., torch.Tensor]: ``(gate, up) -> activated``.
    """
    act_fn = get_act_fn(hidden_act)
    if swiglu_limit is None:
        return lambda gate, up: act_fn(gate) * up
    return lambda gate, up: act_fn(gate.clamp(max=swiglu_limit)) * up.clamp(min=-swiglu_limit, max=swiglu_limit)


def native_gelu(x: torch.Tensor, approximate: str | None = None) -> torch.Tensor:
    if approximate is not None:
        return F.gelu(x, approximate=approximate)
    return F.gelu(x)


def npu_gelu(x: torch.Tensor, approximate: str | None = None) -> torch.Tensor:
    import torch_npu

    if approximate is not None:
        return torch_npu.npu_gelu(x, approximate=approximate)
    return torch_npu.npu_gelu(x)


def npu_clipped_swiglu(fused_x: torch.Tensor, split_dim=-1, alpha=1.702, limit=7) -> torch.Tensor:
    raise NotImplementedError


def native_silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


@triton.jit
def _sigmoid_mul_fwd_kernel(
    y_ptr,
    g_ptr,
    o_ptr,
    R,
    C,
    YS,
    GS,
    OS,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    """out = y * sigmoid(g), one pass.

    Serves the GLM-5.3 KDA ``o_norm`` gate: the eager chain materializes a bf16
    sigmoid result between aclnnSigmoid and aclnnMul, so the kernel reproduces
    that rounding point (fp32 sigmoid from bf16 ``g``, bf16 round of the
    sigmoid result, fp32 multiply, bf16 store) to match bit for bit.
    """
    pid = tl.program_id(0)
    nprogs = tl.num_programs(0)
    c_tiles = C // BC
    r_off = tl.arange(0, BR)
    c_off = tl.arange(0, BC)
    for t in range(pid, tl.cdiv(R, BR) * c_tiles, nprogs):
        rt = t // c_tiles
        ct = t % c_tiles
        rows = rt * BR + r_off
        rmask = rows < R
        cols = ct * BC + c_off
        r64 = rows[:, None].to(tl.int64)
        y = tl.load(y_ptr + r64 * YS + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        g = tl.load(g_ptr + r64 * GS + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        s = tl.sigmoid(g)
        # mirror the eager chain's materialized bf16 sigmoid output before the mul
        s = s.to(tl.bfloat16).to(tl.float32)
        tl.store(o_ptr + r64 * OS + cols[None, :], (s * y).to(o_ptr.dtype.element_ty), mask=rmask[:, None])


@triton.jit
def _sigmoid_mul_bwd_kernel(
    y_ptr,
    g_ptr,
    dout_ptr,
    dy_ptr,
    dg_ptr,
    R,
    C,
    YS,
    GS,
    DYS,
    DGS,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    """Backward of the fused sigmoid gate ``y * sigmoid(g)``.

    Mirrors the eager autograd chain: aclnnMulBackward produces a bf16
    ``dout * y`` that feeds aclnnSigmoidBackward, which takes the bf16 forward
    sigmoid output ``s`` as its ``y`` input and computes ``ds * s * (1 - s)``
    in fp32 (bf16 store). ``s`` is recomputed from ``g`` with the same bf16
    round as the forward's materialized intermediate; ``dg`` and ``dy`` are
    stored bf16.
    """
    pid = tl.program_id(0)
    nprogs = tl.num_programs(0)
    c_tiles = C // BC
    r_off = tl.arange(0, BR)
    c_off = tl.arange(0, BC)
    for t in range(pid, tl.cdiv(R, BR) * c_tiles, nprogs):
        rt = t // c_tiles
        ct = t % c_tiles
        rows = rt * BR + r_off
        rmask = rows < R
        cols = ct * BC + c_off
        r64 = rows[:, None].to(tl.int64)
        y = tl.load(y_ptr + r64 * YS + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        g = tl.load(g_ptr + r64 * GS + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        dout = tl.load(dout_ptr + r64 * C + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        s = tl.sigmoid(g).to(tl.bfloat16).to(tl.float32)
        ds = (dout * y).to(tl.bfloat16).to(tl.float32)
        dg = ds * s * (1.0 - s)
        dy = dout * s
        tl.store(dy_ptr + r64 * DYS + cols[None, :], dy.to(dy_ptr.dtype.element_ty), mask=rmask[:, None])
        tl.store(dg_ptr + r64 * DGS + cols[None, :], dg.to(dg_ptr.dtype.element_ty), mask=rmask[:, None])


def _sigmoid_mul_bc(c: int) -> int | None:
    """Largest tile width in {512, 256, 128, 64} (all pow2) dividing ``c``.

    Args:
        c (int): Last-dim width of the sigmoid-gate operands.

    Returns:
        int | None: The tile width, or ``None`` when ``c`` is not divisible by
        any supported pow2 width (the caller falls back to eager).
    """
    for bc in (512, 256, 128, 64):
        if c % bc == 0:
            return bc
    return None


def _sigmoid_mul_ok(y: torch.Tensor, g: torch.Tensor) -> bool:
    """Gate for the fused sigmoid gate: NPU, bf16, contiguous, same shape.

    Args:
        y (torch.Tensor): Normed input ``[..., C]``.
        g (torch.Tensor): Gate, same shape as ``y``.

    Returns:
        bool: Whether :class:`_SigmoidMulFn` can serve the pair.
    """
    if not (
        _GATED_ACT_FUSED
        and _NPU_AVAILABLE
        and y.dtype == torch.bfloat16
        and g.dtype == torch.bfloat16
        and y.shape == g.shape
        and y.is_contiguous()
        and g.is_contiguous()
        and y.dim() >= 1
        and y.numel() > 0
    ):
        return False
    return _sigmoid_mul_bc(y.shape[-1]) is not None


class _SigmoidMulFn(torch.autograd.Function):
    """Fused ``y * sigmoid(g)`` for the KDA ``o_norm`` gate.

    One triton kernel per direction replaces the eager 3-op forward (Cast +
    Sigmoid + Mul) and 2-op backward (MulBackward + SigmoidBackward) -- the
    per-call cost of the GLM-5.3 attention output gate. Rounding points mirror
    the eager chain's materialized bf16 intermediates; the A/B harness in
    tests/model/test_glm53_sigmoid_mul_fuse.py asserts bitwise equality on NPU.
    """

    @staticmethod
    def forward(ctx, y: torch.Tensor, g: torch.Tensor) -> torch.Tensor:  # noqa: ANN001
        c = y.shape[-1]
        rows = y.numel() // c
        out = torch.empty_like(y)
        bc = _sigmoid_mul_bc(c)
        _sigmoid_mul_fwd_kernel[(_GATED_ACT_MAX_PROGS,)](y, g, out, rows, c, c, c, c, BR=8, BC=bc, num_warps=8)
        ctx.save_for_backward(y, g)
        ctx.bc = bc
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:  # noqa: ANN001
        y, g = ctx.saved_tensors
        if not (ctx.needs_input_grad[0] or ctx.needs_input_grad[1]):
            return None, None
        c = y.shape[-1]
        rows = y.numel() // c
        grad_y = torch.empty_like(y)
        grad_g = torch.empty_like(g)
        _sigmoid_mul_bwd_kernel[(_GATED_ACT_MAX_PROGS,)](
            y,
            g,
            grad_out.contiguous(),
            grad_y,
            grad_g,
            rows,
            c,
            c,
            c,
            c,
            c,
            BR=8,
            BC=ctx.bc,
            num_warps=8,
        )
        return (
            grad_y if ctx.needs_input_grad[0] else None,
            grad_g if ctx.needs_input_grad[1] else None,
        )


def fused_sigmoid_mul(y: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """NPU fused ``y * sigmoid(g)`` for the GLM-5.3 KDA ``o_norm`` gate.

    Bit-exact drop-in for ``y * torch.sigmoid(g)`` on NPU bf16: one triton
    kernel per direction instead of the aclnn Cast+Sigmoid+Mul forward chain
    (and MulBackward+SigmoidBackward backward chain). Anything failing the gate
    (:func:`_sigmoid_mul_ok`) falls back to the eager chain unchanged.

    Args:
        y (torch.Tensor): Normed input ``[..., C]`` (bf16).
        g (torch.Tensor): Gate, same shape as ``y`` (bf16).

    Returns:
        torch.Tensor: ``y * sigmoid(g)`` (bf16), same shape.
    """
    if _sigmoid_mul_ok(y, g):
        return _SigmoidMulFn.apply(y, g)
    return y * torch.sigmoid(g)


act_fn_type_map_cuda = {
    "swiglu": native_swiglu,
    "clipped_swiglu": native_clipped_swiglu,
    "clamped_swiglu": native_clamped_swiglu,
    "gelu": native_gelu,
    "gelu_pytorch_tanh": partial(native_gelu, approximate="tanh"),
    "silu": native_silu,
}
act_fn_type_map_npu = {
    "swiglu": npu_swiglu,
    "clipped_swiglu": npu_clipped_swiglu,
    "clamped_swiglu": native_clamped_swiglu,
    "gelu": npu_gelu,
    "gelu_pytorch_tanh": partial(npu_gelu, approximate="tanh"),
    "silu": native_silu,
}


def get_act_fn(act_type):
    from xtuner.v1.utils.device import get_device

    device = get_device()
    if device == "npu":
        return act_fn_type_map_npu[act_type]
    else:
        return act_fn_type_map_cuda[act_type]
