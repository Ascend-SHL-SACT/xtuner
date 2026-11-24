"""NPU grouped GEMM for ``GroupedLinear`` with an overflow-safe backward.

Forward is byte-identical to the former mindspeed path (the same CANN
``npu_gmm`` op on the same inputs). The backward re-issues the same CANN
``npu_gmm_backward`` op on a power-of-two-scaled operand (the smaller of the
upstream gradient / saved activation, so the transient copy fits) when the
operand magnitudes would overflow the op's fp32 accumulator: saturated-gate
packs drive finite-but-huge (~1e24-1e26) expert gradients whose elementwise
|grad_out| * |activation| products pass the fp32 cap (2^128) inside the dW
GEMM, turning per-expert partials into Inf and then NaN -- the FFN/expert
backward fp32 overflow that remains after the KDA band-16 fix.

Power-of-two scaling is an exact representation change (the mantissa is
untouched, only the exponent shifts): every scaled product and partial sum
stays inside the fp32 range and the unscaled result equals the true gradient
up to normal fp rounding. Healthy steps compute scale = 1.0 and take the
zero-copy branch -- the op receives the original tensors, bit-identical to
the former path (verified bitwise-equal on healthy training steps).
Genuinely out-of-range true values still propagate as Inf after the unscale
(no detection, retry, or masking: this is a representation fix, not a
fallback).

The exponent shift is computed device-side (two amax reductions + scalar
kernels) and pulled to the host as an int (one small D2H sync) to branch to
the zero-copy healthy path: s == 0 whenever the product-exponent budget holds
or either operand is already non-finite (a non-finite operand keeps the
original unscaled call semantics, so its honest Inf/NaN propagation is
unchanged), and the guards outside the healthy grad-norm band only (see the
arming flag in ops/moe/npu/fused_a2a_gmm.py) skip even that sync.
"""

from __future__ import annotations

import math

import torch

from mindspeed.ops.gmm import GMMFunction

# Keep every dW elementwise product <= 2^96. A per-expert partial sum then
# stays below 2^96 * max_tokens_per_expert (<= 2^17 rows) = 2^113, safely under
# the fp32 accumulator cap 2^128 with headroom for the dX GEMM's weight
# exponent (~2^4).
_PRODUCT_EXPONENT_BUDGET = 96


def _bwd_power2_scale(grad_out: torch.Tensor, x: torch.Tensor) -> int:
    """Host-side exponent shift for the backward rescale of a (grad, activation) pair.

    Returns ``s = max(0, floor(log2|grad_out|.amax) + floor(log2|x|.amax) -
    _PRODUCT_EXPONENT_BUDGET)`` as a host int (one .item() sync). 0 when the
    budget already holds, when either tensor is all-zero, or when either amax
    is non-finite (keep honest Inf/NaN propagation). The device-side floor of
    an amax may tip +/-1 at integer boundaries -- harmless, the 32-bit budget
    headroom dwarfs it. Callers apply the factor as an exact host-side
    ``math.ldexp(1.0, -+s)`` float: the NPU fp32 exp2 scalar kernel is not
    correctly rounded (observed ~1e-7 relative error), so the factor must not
    be materialized with it when exactness is claimed.
    """
    tiny = torch.finfo(grad_out.dtype).tiny
    go_max = grad_out.abs().amax()
    x_max = x.abs().amax()
    e = (
        torch.log2(go_max.clamp_min(tiny)).floor()
        + torch.log2(x_max.clamp_min(tiny)).floor()
    )
    finite = torch.isfinite(go_max) & torch.isfinite(x_max)
    e = torch.where(finite, e, torch.zeros_like(e))
    s = (e - _PRODUCT_EXPONENT_BUDGET).clamp_min(0)
    return int(s.item())


def _rescale(t: torch.Tensor, factor: float, inplace: bool = False) -> torch.Tensor:
    """Multiply ``t`` by the exact power-of-two host float ``factor``.

    No dtype-widening copy: a python-float power-of-two factor converts to the
    tensor's dtype exactly (it only shifts the exponent; the operand's
    mantissa is untouched), so the product is exact whenever the factor
    (2^±s, |s| <= 126) and the result are representable in the dtype.
    ``inplace=True`` is for op-fresh outputs only (the dW/dX grads this
    module just produced and still solely owns): the multi-GB dW cannot
    tolerate even one same-size duplicate at training-time memory pressure,
    so the unscale mutates the op's output buffer in place (x * 1.0 == x, so
    the healthy path is bit-identical). |s| > 126 cannot arise from
    bf16-representable operands without the upstream already being Inf; such
    values keep their honest propagation (the factor itself leaves the
    storage dtype's range there).
    """
    return t.mul_(factor) if inplace else t * factor


class _OverflowSafeGroupedGemm(torch.autograd.Function):
    """Grouped GEMM (``out = x @ weight`` per expert group) whose backward
    guards the CANN op's fp32 accumulator with an exact power-of-two rescale.

    Mirrors mindspeed ``GMMFunction`` (group_list_type=0, no bias, no
    gemm_fusion) for the layouts ``GroupedLinear`` feeds it: ``weight_t`` is
    the transposed view ``[E, in, out]`` and ``group_list_cum`` the cumulative
    int64 token counts on device.
    """

    @staticmethod
    def forward(ctx, x, weight_t, group_list_cum):
        op = GMMFunction.builder.load()
        outputs = op.npu_gmm([x], [weight_t], [], group_list_cum, 0, 0)
        ctx.save_for_backward(x, weight_t, group_list_cum)
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_out):
        from xtuner.v1.ops.moe.npu.fused_a2a_gmm import overflow_guard_deep_armed

        x, weight_t, group_list_cum = ctx.saved_tensors
        op = GMMFunction.builder.load()
        # Disarmed, armed-shallow, and armed-healthy (s == 0) all feed the
        # ORIGINAL tensors to the op -- zero copies, bit-identical to the
        # former mindspeed backward. Only past the deep-arm band does a call
        # pay the amax + sync measurement. Without the s == 0 branch even a
        # healthy armed step would duplicate a multi-GB operand and OOM (the
        # dY1 grad of the first MoE layer alone is ~9 GiB).
        s = _bwd_power2_scale(grad_out, x) if overflow_guard_deep_armed() else 0
        if s == 0:
            dx, dw, _dbias = op.npu_gmm_backward([grad_out], [x], [weight_t], group_list_cum, 0)
            return dx[0], dw[0], None
        scale = math.ldexp(1.0, -s)
        unscale = math.ldexp(1.0, s)
        if x.numel() <= grad_out.numel():
            # Scale the smaller operand (the saved activation): the dX leg
            # (grad_out @ weight) does not read x, so dX stays exact unscaled
            # and only dW carries the rescale -- unscaled in place on the
            # op-fresh output (any same-size duplicate OOMs).
            x_s = _rescale(x, scale)
            dx, dw, _dbias = op.npu_gmm_backward([grad_out], [x_s], [weight_t], group_list_cum, 0)
            return dx[0], _rescale(dw[0], unscale, inplace=True), None
        grad_out_s = _rescale(grad_out, scale)
        dx, dw, _dbias = op.npu_gmm_backward([grad_out_s], [x], [weight_t], group_list_cum, 0)
        return _rescale(dx[0], unscale, inplace=True), _rescale(dw[0], unscale, inplace=True), None


def npu_group_gemm(x: torch.Tensor, weights: torch.Tensor, split_sizes: torch.Tensor) -> torch.Tensor:
    weights = weights.transpose(1, 2)

    group_list = torch.cumsum(split_sizes, dim=0).to('npu')
    return _OverflowSafeGroupedGemm.apply(x, weights, group_list)
