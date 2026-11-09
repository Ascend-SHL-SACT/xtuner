"""Int8 W8A8 quantization kernels for linear layers.

The weight grid is a self-defined contract: the per-output-channel fp32 scale
(``amax(|w|) / 127``) and the int8 levels both ship in the checkpoint, and the deployment
engine multiplies them back at serve time, so the math here only has to match itself
across training (fake-quant STE), export (checkpoint int8 + scale) and the
``npu_quant_gmm`` real-op arm. The activation grid is the deployed ``W8A8_DYNAMIC``
scheme's per-token dynamic quantization.

On NPU both grids run through the deployment kernel itself --
``torch.ops.npu.npu_dynamic_quant``, the exact op vllm-ascend's ``W8A8_DYNAMIC`` path
calls at serve time -- replacing the ~8-op eager chain with one fused quant kernel plus
the dequant (4 kernels per call site in situ: quant, the int8->fp32 promote cast, the
fp32 multiply, the cast back; the micro-bench's 2.3-3.6x was the whole-chain figure).
Its grid differs from the CPU replica chain by two probed, bounded deltas: the fp32
scale is ``amax * (1 / 127)`` (reciprocal multiply, at most 1 ulp from IEEE division on
~1.5% of rows) and the rounding flips ~0.03% of knife-edge levels by exactly one. An
all-zero row gets scale 0 instead of the CPU chain's ``1e-12`` floor; its levels are 0
either way, so the dequantized value stays 0. Per-device self-consistency is what the
contract needs: the export quantizes each weight on the training device when one is
available (the same kernel), so NPU training exports the kernel scale it trained
against; a CPU-only export falls back to this eager chain and carries the deltas above.

The dequant multiply has two selectable grids (``XTUNER_QUANT_LINEAR_DEQUANT``): the
default ``"fp32"`` promotes the int8 levels to fp32, multiplies against the kernel's
fp32 scale and casts back -- bitwise the deployment reconstruction -- while ``"native"``
multiplies in the tensor's own dtype, re-rounding the fp32 scale once (at most 2^-8
relative; the int8 levels are exact in bf16) to roughly halve the dequant's memory
traffic on the bandwidth-bound shapes. The option affects this NPU fused path only; the
eager replica below keeps the fp32 grid (it mirrors the deployment reconstruction off
NPU). ``npu_anti_quant`` is not usable for either grid -- it accepts per-tensor /
per-K-channel scales, not per-row.

Off NPU (or on fp32 input, which the kernel does not accept) the eager torch chain
applies the original grid: IEEE ``amax / 127`` with the ``1e-12`` floor, ``torch.round``
(half-to-even); 910C-safe by construction (no fp8/int4 operators, all quantization
arithmetic in fp32).
"""

import torch

from ..kv.kernels import int8_fake_quant_dynamic, int8_fake_quant_static
from .config import _dequant_grid_from_env


# ``npu_dynamic_quant`` accepts half/bf16 input only; fp32 (or CPU) tensors keep the eager
# chain. Reaching the NPU branch implies torch_npu is loaded in the process, so the
# ``torch.ops.npu`` namespace is registered by then.
_NPU_FUSED_DTYPES = (torch.bfloat16, torch.float16)


def per_channel_weight_quant(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``w`` to int8 levels with a per-output-channel fp32 scale.

    On NPU this returns the deployment kernel's own levels and scale; the scale is its
    ``amax * (1 / 127)`` fp32 value and an all-zero channel gets scale 0 (levels 0, so
    the dequantized value is 0 either way -- the eager chain's zero-division guard is
    unnecessary here).

    Args:
        w (torch.Tensor): Weight with the channel axis second-to-last, e.g. ``[out, in]``
            for a linear layer or ``[E, out, in]`` for a grouped one.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(q, scale)`` where ``q`` is the int8 level
        tensor with ``w``'s shape and ``scale`` is the fp32 scale with ``w``'s shape minus
        the last axis plus a trailing 1 (e.g. ``[out, 1]`` / ``[E, out, 1]``).
    """
    if _use_npu_fused(w):
        q, scale = _npu_dynamic_quant_rows(w)
        return q.reshape(w.shape), scale.reshape(w.shape[:-1]).unsqueeze(-1)
    scale = _per_channel_weight_scale(w)
    q = torch.clamp(torch.round(w.float() / _safe_scale(scale)), -127.0, 127.0)
    return q.to(torch.int8), scale


def per_channel_weight_fake_quant(w: torch.Tensor) -> torch.Tensor:
    """Fake-quantize ``w`` on the per-output-channel int8 grid (straight-through).

    Grid-identical to :func:`per_channel_weight_quant` (the same kernel on NPU, same
    scale, same rounding, same clamp), dequantized back to ``w``'s dtype; the backward
    passes gradients through unchanged.

    Args:
        w (torch.Tensor): Weight with the channel axis second-to-last, e.g. ``[out, in]``
            for a linear layer or ``[E, out, in]`` for a grouped one.

    Returns:
        torch.Tensor: Dequantized tensor with the same shape and dtype as ``w``.
    """
    if _use_npu_fused(w):
        return _NpuFusedRowwiseFakeQuant.apply(w)
    return int8_fake_quant_static(w, _safe_scale(_per_channel_weight_scale(w)), clip=False)


def per_token_act_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations to int8 levels with a per-token fp32 scale.

    On NPU this is the deployment kernel itself (``npu_dynamic_quant``), so the
    ``npu_quant_gmm`` real-op arm consumes exactly the levels and scales serve time
    computes. Off NPU it replicates the eager chain's scale math (``amax / 127`` per
    token with the ``1e-12`` floor). All-zero rows get scale 0 on NPU and ``1e-12`` off
    it; their levels are 0 either way, so the dequantized value stays 0.

    Args:
        x (torch.Tensor): Activations with the token axis first and the channel axis
            last, e.g. ``[M, C]``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(q, scale)`` where ``q`` is the int8 level
        tensor with ``x``'s shape and ``scale`` is the fp32 per-token scale ``[M]``.
    """
    if _use_npu_fused(x):
        q, scale = _npu_dynamic_quant_rows(x)
        return q.reshape(x.shape), scale.reshape(x.shape[:-1])
    x32 = x.float()
    scale = x32.abs().amax(dim=-1, keepdim=True).div_(127.0).clamp_min_(1e-12)
    q = torch.clamp(torch.round(x32 / scale), -127.0, 127.0)
    return q.to(torch.int8), scale.squeeze(-1)


def per_token_act_fake_quant(x: torch.Tensor) -> torch.Tensor:
    """Fake-quantize activations per token on the deployed dynamic grid (straight-through).

    The linear layers' forward entry: on NPU the fused deployment kernel plus the
    dequant (4 kernels in situ vs the eager chain's ~8; dequant grid selectable via
    ``XTUNER_QUANT_LINEAR_DEQUANT``); off NPU the eager replica chain
    (:func:`~xtuner.v1.quantization.kv.kernels.int8_fake_quant_dynamic`). The
    fused a2a+gmm path consumes this through the modules' ``fused_act_fake_quant`` hook,
    so the separated and fused paths share the same grid.

    Args:
        x (torch.Tensor): Activations with the channel axis last, e.g. ``[M, C]``.

    Returns:
        torch.Tensor: Dequantized tensor with the same shape and dtype as ``x``.
    """
    if _use_npu_fused(x):
        return _NpuFusedRowwiseFakeQuant.apply(x)
    return int8_fake_quant_dynamic(x)


def _use_npu_fused(x: torch.Tensor) -> bool:
    return x.device.type == "npu" and x.dtype in _NPU_FUSED_DTYPES


def _npu_dynamic_quant_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Row-wise int8 dynamic quant through the deployment kernel (the exact op
    # vllm-ascend's W8A8_DYNAMIC path calls): one fused amax+scale+round+clamp kernel. A
    # row is the last axis; leading axes flatten into rows, and per-row scales are
    # row-local, so a ``[E, out, in]`` grouped weight quantizes bitwise-identically to
    # quantizing each expert's ``[out, in]`` slice alone.
    rows = x.reshape(-1, x.shape[-1])
    return torch.ops.npu.npu_dynamic_quant(rows, dst_type=torch.int8)


class _NpuFusedRowwiseFakeQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        # Two selectable dequant grids, parsed per call (XTUNER_QUANT_LINEAR_DEQUANT).
        # "fp32" (the default) promotes the int8 levels to fp32, multiplies against the
        # kernel's fp32 scale and casts back -- bitwise the deployment reconstruction.
        # "native" casts to x's dtype and multiplies there: roughly half the dequant
        # memory traffic on the bandwidth-bound shapes, at the cost of re-rounding the
        # fp32 scale once (at most 2^-8 relative; int8 levels are exact in bf16).
        q, scale = _npu_dynamic_quant_rows(x)
        if _dequant_grid_from_env() == "native":
            return (q.to(x.dtype) * scale.to(x.dtype).unsqueeze(-1)).reshape(x.shape)
        return (q * scale.unsqueeze(-1)).to(x.dtype).reshape(x.shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


def _per_channel_weight_scale(w: torch.Tensor) -> torch.Tensor:
    # fp32 per-output-channel scale along the last (input) axis. IEEE division (not the
    # reciprocal multiply the deployment activation kernel uses): the scale ships in the
    # checkpoint, so the only requirement is self-consistency across training / export /
    # the real-op arm (the #150 calibrator precedent). ``aminmax`` + ``max(mx, -mn)`` reads
    # ``w`` once without the ``abs()`` copy; selection over the raw dtype is exact, so the
    # value is bit-identical to ``amax(w.float().abs())``.
    mn, mx = torch.aminmax(w.detach(), dim=-1, keepdim=True)
    return torch.maximum(mx, mn.neg()).float().div_(127.0)


def _safe_scale(scale: torch.Tensor) -> torch.Tensor:
    # An all-zero output channel has scale 0 and would turn the quantization division
    # into 0/0 = NaN; quantize such channels against 1.0 (their levels are exactly 0, so
    # the dequantized value stays 0 either way).
    return torch.where(scale > 0, scale, torch.ones_like(scale))


__all__ = [
    "int8_fake_quant_dynamic",
    "per_channel_weight_fake_quant",
    "per_channel_weight_quant",
    "per_token_act_fake_quant",
    "per_token_act_quant",
]
