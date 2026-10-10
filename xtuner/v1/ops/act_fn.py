from functools import partial
from typing import Callable

import torch
import triton
import triton.language as tl
from torch.nn import functional as F


# Persistent-grid cap, at the aiv-mode physical block count (48): binaries compiled
# WITH TRITON_ALL_BLOCKS_PARALLEL=1 replay correctly past the physical count (see
# kda_op._SEG_COPY_MAX_PROGS; conv-bwd/mhc_fused have run
# 48, this kernel family probed NaN-free and bitwise-equal at 48 vs 24), while a
# stale cache compiled without the flag still silently drops blocks past the
# aicore-mode clamp of 24 -- the same guard every capped kernel here carries.
_GATED_ACT_MAX_PROGS = 48

try:
    _NPU_AVAILABLE = torch.npu.is_available()  # type: ignore[attr-defined]
except (ImportError, AttributeError):  # pragma: no cover - CPU-only environments
    _NPU_AVAILABLE = False


def _fused_device_ok(t: torch.Tensor) -> bool:
    """Per-tensor device gate for the fused kernels.

    ``_NPU_AVAILABLE`` alone is a machine-level property: a CPU-resident tensor on an NPU
    box would pass it and reach the triton NPU backend with a host pointer -- a hard launch
    failure instead of the eager fallback. Same convention as ``xtuner.v1.ops.mhc_fused`` (norm-linear and
    stream-combine). A separate helper (not an inline ``device.type`` check) so the
    CPU-box condition-matrix tests can patch it alongside ``_NPU_AVAILABLE``.

    Args:
        t (torch.Tensor): Candidate operand.

    Returns:
        bool: Whether ``t`` lives on an NPU device.
    """
    return t.device.type == "npu"


@triton.jit
def _clamped_silu_mul_fwd_kernel(
    g_ptr,
    u_ptr,
    o_ptr,
    R,
    C,
    GS,
    US,
    U_OFF,
    limit,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    """out = silu(gate.clamp(max=limit)) * up.clamp(±limit), one pass.

    Serves both layouts: separate projections (``g_ptr``/``u_ptr`` distinct,
    ``U_OFF=0``) and the fused ``[..., 2C]`` gate/up tensor (same base pointer,
    ``GS=US=2C``, ``U_OFF=C``). The eager chain materializes two bf16
    intermediates (clamp outputs, silu output) before the multiply; the kernel
    reproduces those rounding points (fp32 compute, bf16 round of the silu
    result, fp32 multiply, bf16 store) so the output matches the aclnn
    clamp+silu+mul chain bit for bit.
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
        base_g = r64 * GS + cols[None, :]
        base_u = r64 * US + U_OFF + cols[None, :]
        g = tl.load(g_ptr + base_g, mask=rmask[:, None], other=0.0).to(tl.float32)
        u = tl.load(u_ptr + base_u, mask=rmask[:, None], other=0.0).to(tl.float32)
        # PropagateNan.ALL: the default (NONE) would clamp a NaN input to a finite
        # bound and silently swallow it, while the eager clamp/silu chain propagates
        # NaN -- the fused path must keep the loud failure the zero-nan acceptance
        # invariant relies on. Finite inputs are unaffected bit for bit.
        g = tl.minimum(g, limit, propagate_nan=tl.PropagateNan.ALL)
        u = tl.minimum(
            tl.maximum(u, -limit, propagate_nan=tl.PropagateNan.ALL), limit, propagate_nan=tl.PropagateNan.ALL
        )
        s = g * tl.sigmoid(g)
        # mirror the eager chain's materialized bf16 silu output before the mul
        s = s.to(tl.bfloat16).to(tl.float32)
        tl.store(o_ptr + r64 * C + cols[None, :], (s * u).to(o_ptr.dtype.element_ty), mask=rmask[:, None])


@triton.jit
def _clamped_silu_mul_bwd_kernel(
    g_ptr,
    u_ptr,
    dout_ptr,
    dg_ptr,
    du_ptr,
    R,
    C,
    GS,
    US,
    U_OFF,
    DGS,
    DUS,
    DU_OFF,
    limit,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    """Backward of the fused clamped SwiGLU (both layouts).

    Mirrors the eager autograd chain's op order and rounding: ``d_silu_in =
    dout * up_c`` (bf16 materialize), silu-backward in fp32 (bf16 store), then
    the two clamp select masks. The masks test the ORIGINAL (pre-clamp) values
    and pass the boundary inclusive -- ``g <= limit`` / ``|u| <= limit`` --
    identical to aten clamp backward (and the retired glm53/act.py mirror's
    kernels under backup/dead_mirrors/).
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
        g0 = tl.load(g_ptr + r64 * GS + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        u0 = tl.load(u_ptr + r64 * US + U_OFF + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        dout = tl.load(dout_ptr + r64 * C + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        # PropagateNan.ALL mirrors the forward clamp and aten clamp semantics: the
        # select masks below still zero the NaN positions' own clamp gradient (NaN
        # fails both comparisons, as in eager), while the cross terms keep the NaN.
        g = tl.minimum(g0, limit, propagate_nan=tl.PropagateNan.ALL)
        u = tl.minimum(
            tl.maximum(u0, -limit, propagate_nan=tl.PropagateNan.ALL), limit, propagate_nan=tl.PropagateNan.ALL
        )
        sig = tl.sigmoid(g)
        s = (g * sig).to(tl.bfloat16).to(tl.float32)
        d_silu_in = (dout * u).to(tl.bfloat16).to(tl.float32)
        dg = d_silu_in * sig * (1.0 + g * (1.0 - sig))
        du = dout * s
        dg = tl.where(g0 <= limit, dg, 0.0)
        du = tl.where((u0 >= -limit) & (u0 <= limit), du, 0.0)
        tl.store(dg_ptr + r64 * DGS + cols[None, :], dg.to(dg_ptr.dtype.element_ty), mask=rmask[:, None])
        tl.store(du_ptr + r64 * DUS + DU_OFF + cols[None, :], du.to(du_ptr.dtype.element_ty), mask=rmask[:, None])


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


def _gated_fused_ok(gate: torch.Tensor, up: torch.Tensor) -> bool:
    """Gate for the fused separate-gate/up path: NPU, bf16, contiguous, C % 512 == 0.

    Args:
        gate (torch.Tensor): Gate projection output ``[..., C]``.
        up (torch.Tensor): Up projection output, same shape as ``gate``.

    Returns:
        bool: Whether :class:`_GatedClampedSiluFn` can serve the pair.
    """
    return (
        _NPU_AVAILABLE
        and _fused_device_ok(gate)
        and _fused_device_ok(up)
        and gate.dtype == torch.bfloat16
        and up.dtype == torch.bfloat16
        and gate.shape == up.shape
        and gate.is_contiguous()
        and up.is_contiguous()
        and gate.dim() >= 1
        and gate.shape[-1] % 512 == 0
        and gate.shape[-1] > 0
        and gate.numel() > 0
    )


class _GatedClampedSiluFn(torch.autograd.Function):
    """Fused ``silu(clamp_max(gate, L)) * clamp(up, ±L)`` for separate projections.

    One triton kernel per direction replaces the eager 4-op forward chain
    (aclnnClamp x2, aclnnSilu, aclnnMul) -- the per-call cost of the shared
    expert / dense MLP activations. Rounding points mirror the eager chain's
    materialized bf16 intermediates; the A/B harness in
    tests/model/test_glm53_gated_act_fuse.py asserts bitwise equality on NPU.

    Backward serves the pure-bf16 production graph: ``grad_out`` is consumed as
    given and ``dg``/``du`` are stored bf16 (``empty_like`` of the inputs); an
    fp32 ``grad_out`` from a mixed-precision graph would be silently narrowed.
    """

    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:  # noqa: ANN001
        c = gate.shape[-1]
        rows = gate.numel() // c
        out = torch.empty_like(gate)
        _clamped_silu_mul_fwd_kernel[(_GATED_ACT_MAX_PROGS,)](
            gate, up, out, rows, c, c, c, 0, limit, BR=8, BC=512, num_warps=8
        )
        ctx.save_for_backward(gate, up)
        ctx.limit = limit
        return out

    @staticmethod
    def backward(  # noqa: ANN001
        ctx, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None]:
        gate, up = ctx.saved_tensors
        c = gate.shape[-1]
        rows = gate.numel() // c
        # Both buffers are allocated whenever either half is needed: the kernel
        # stores both halves unconditionally, so neither lane can alias an input.
        grad_gate = torch.empty_like(gate)
        grad_up = torch.empty_like(up)
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            _clamped_silu_mul_bwd_kernel[(_GATED_ACT_MAX_PROGS,)](
                gate,
                up,
                grad_out.contiguous(),
                grad_gate,
                grad_up,
                rows,
                c,
                c,
                c,
                0,
                c,
                c,
                0,
                ctx.limit,
                BR=8,
                BC=512,
                num_warps=8,
            )
        return (grad_gate if ctx.needs_input_grad[0] else None, grad_up if ctx.needs_input_grad[1] else None, None)


def _fused_gated_clamped_silu(
    limit: float, act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build the NPU fused callable for ``(gate, up) -> silu(gate.clamp(max=L)) * up.clamp(±L)``.

    Args:
        limit (float): ``swiglu_limit`` clamp bound.
        act_fn (Callable[[torch.Tensor], torch.Tensor]): The gate activation resolved by
            ``get_gated_act_fn``; used by the eager fallback so both paths stay in lockstep
            if the ``"silu"`` map entry ever changes.

    Returns:
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]: The activation;
        falls back to the eager chain whenever the fused gate rejects the pair.
    """

    def _act(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        if _gated_fused_ok(gate, up):
            return _GatedClampedSiluFn.apply(gate, up, limit)
        return act_fn(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)

    return _act


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
        _NPU_AVAILABLE
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


def _fused_layout_ok(fused_x: torch.Tensor, split_dim: int) -> bool:
    """Gate for the fused ``[..., 2C]`` clamped-SwiGLU kernel path.

    Args:
        fused_x (torch.Tensor): Concatenated ``[..., gate; up]`` tensor, the
            routed-expert gate_up projection output.
        split_dim (int): The dimension the gate/up halves live along.

    Returns:
        bool: Whether :class:`_ClampedSwigluFusedFn` can serve this tensor.
    """
    return (
        _NPU_AVAILABLE
        # Per-tensor device check, same rationale as _gated_fused_ok.
        and _fused_device_ok(fused_x)
        and split_dim in (-1, fused_x.dim() - 1)
        and fused_x.dtype == torch.bfloat16
        and fused_x.is_contiguous()
        and fused_x.dim() >= 1
        and fused_x.shape[-1] % 2 == 0
        and (fused_x.shape[-1] // 2) % 512 == 0
        and fused_x.shape[-1] > 0
        and fused_x.numel() > 0
    )


class _ClampedSwigluFusedFn(torch.autograd.Function):
    """Fused ``silu(clamp_max(gate, L)) * clamp(up, ±L)`` for ``[..., 2C]`` layout.

    Bit-exact replacement of the eager ``native_clamped_swiglu`` chain: the two
    chunk views are read in-place via strides (no unpack copy), and backward
    writes both halves of ``dx`` in a single kernel. Rounding points mirror the
    eager chain's materialized bf16 intermediates, and the clamp masks test the
    pre-clamp values with the boundary included (aten clamp-backward semantics).

    Backward serves the pure-bf16 production graph: ``grad_out`` is consumed as
    given and ``dx`` is stored bf16; an fp32 ``grad_out`` from a mixed-precision
    graph would be silently narrowed.
    """

    @staticmethod
    def forward(ctx, fused_x: torch.Tensor, split_dim: int, limit: float) -> torch.Tensor:  # noqa: ANN001
        two_c = fused_x.shape[-1]
        c = two_c // 2
        rows = fused_x.numel() // two_c
        out = torch.empty(fused_x.shape[:-1] + (c,), dtype=fused_x.dtype, device=fused_x.device)
        _clamped_silu_mul_fwd_kernel[(_GATED_ACT_MAX_PROGS,)](
            fused_x, fused_x, out, rows, c, two_c, two_c, c, limit, BR=8, BC=512, num_warps=8
        )
        ctx.save_for_backward(fused_x)
        ctx.limit = limit
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple[torch.Tensor | None, None, None]:  # noqa: ANN001
        (fused_x,) = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None, None, None
        two_c = fused_x.shape[-1]
        c = two_c // 2
        rows = fused_x.numel() // two_c
        dx = torch.empty_like(fused_x)
        _clamped_silu_mul_bwd_kernel[(_GATED_ACT_MAX_PROGS,)](
            fused_x,
            fused_x,
            grad_out.contiguous(),
            dx,
            dx,
            rows,
            c,
            two_c,
            two_c,
            c,
            two_c,
            two_c,
            c,
            ctx.limit,
            BR=8,
            BC=512,
            num_warps=8,
        )
        return dx, None, None


def npu_clamped_swiglu_bitexact(fused_x: torch.Tensor, split_dim: int = -1, limit: float = 10.0) -> torch.Tensor:
    """Fused (one kernel per direction) clamped SwiGLU over ``[..., 2C]`` input.

    Bit-exact drop-in for :func:`native_clamped_swiglu` on NPU -- the routed
    experts' gate_up activation. Replaces the eager chunk + clamp + silu + mul
    chain (Clamp_Slice / ClipByValueV2 / Silu / Mul launches per call) with one
    triton kernel per direction; anything failing the fused gate
    (:func:`_fused_layout_ok`) falls back to the eager chain unchanged.

    Args:
        fused_x (torch.Tensor): Concatenated ``[..., gate; up]`` tensor.
        split_dim (int): Dimension holding the gate/up halves; only the last
            dimension is fused, others take the eager path.
        limit (float): ``clip_limit`` clamp bound.

    Returns:
        torch.Tensor: ``silu(clamp_max(gate, limit)) * clamp(up, -limit, limit)``
        of shape ``fused_x.shape[:-1] + (fused_x.shape[-1] // 2,)``.
    """
    if _fused_layout_ok(fused_x, split_dim):
        return _ClampedSwigluFusedFn.apply(fused_x, split_dim, limit)
    return native_clamped_swiglu(fused_x, split_dim=split_dim, limit=limit)


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
    if hidden_act == "silu" and _NPU_AVAILABLE:
        # One fused triton kernel instead of the 4-op aclnn chain (Clamp x2 +
        # Silu + Mul); falls back to the eager chain whenever the pair fails
        # the fused gate (non-bf16, non-contiguous, C % 512 != 0).
        return _fused_gated_clamped_silu(swiglu_limit, act_fn)
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


def npu_silu(x: torch.Tensor) -> torch.Tensor:
    """Silu via the NPU op, falling back to :func:`native_silu` off-device.

    The ``act_fn_type_map_npu["silu"]`` entry, shaped like its ``npu_*`` siblings (swiglu,
    gelu). ``torch_npu.npu_silu`` lowers to the same aclnn Silu kernel ``F.silu`` dispatches
    to on NPU tensors -- forward and backward are bitwise-equal on bf16/fp16/fp32 (asserted
    in tests/model/test_glm53_npu_silu.py) -- so the entry is behaviour-preserving and only
    gives the map a uniform ``npu_*`` surface with a single place for a future
    specialization.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: ``silu(x)``.
    """
    try:
        import torch_npu  # noqa: F401
    except ImportError:  # pragma: no cover - CPU-only environments
        return native_silu(x)
    if x.device.type != "npu":
        return native_silu(x)
    return torch_npu.npu_silu(x)


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
    "clamped_swiglu": npu_clamped_swiglu_bitexact,
    "gelu": npu_gelu,
    "gelu_pytorch_tanh": partial(npu_gelu, approximate="tanh"),
    "silu": npu_silu,
}


def get_act_fn(act_type):
    from xtuner.v1.utils.device import get_device

    device = get_device()
    if device == "npu":
        return act_fn_type_map_npu[act_type]
    else:
        return act_fn_type_map_cuda[act_type]
