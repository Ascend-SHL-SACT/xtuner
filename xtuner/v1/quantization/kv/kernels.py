"""Pure-torch int8 fake-quantization kernels shared by QAT training, export and deployment.

910C-safe by construction: no fp8/int4 operators, all arithmetic in fp32 with the original
dtype in/out. The backward is a straight-through estimator: gradients bypass the rounding
entirely (``clip=True`` additionally zeroes gradients of values outside the int8 range).

Rounding uses ``torch.round`` (half-to-even); the deployment kernel's rounding mode must be
aligned with this before claiming bit-exact training/inference parity.
"""

import torch


class _Int8FakeQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor | None, clip: bool) -> torch.Tensor:
        x32 = x.float()
        if scale is None:
            # Per-token dynamic: the scale is derived inside the Function so no gradient
            # flows through the amax path.
            scale = x32.abs().amax(dim=-1, keepdim=True).div_(127.0).clamp_min_(1e-12)
        ctx.clip = clip
        if clip:
            ctx.save_for_backward(x32.abs().le(127.0 * scale))
        q = torch.clamp(torch.round(x32 / scale), -127.0, 127.0)
        return q.mul(scale).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.clip:
            (mask,) = ctx.saved_tensors
            # torch.where avoids materializing a mask-sized float copy on every backward
            # pass (the N3 review item) and never turns a masked-out grad into NaN.
            zero = torch.zeros((), dtype=grad_output.dtype, device=grad_output.device)
            grad_output = torch.where(mask, grad_output, zero)
        return grad_output, None, None


def int8_fake_quant_dynamic(x: torch.Tensor, clip: bool = False) -> torch.Tensor:
    """Fake-quantize ``x`` on the symmetric int8 grid with a per-token dynamic scale.

    Args:
        x (torch.Tensor): Input with the quantization axis last, e.g. ``[B, S, C]``.
        clip (bool): When ``True``, zero gradients of values outside the int8 range instead of
            passing them through.

    Returns:
        torch.Tensor: Dequantized tensor with the same shape and dtype as ``x``.
    """
    return _Int8FakeQuant.apply(x, None, clip)


def int8_fake_quant_static(x: torch.Tensor, scale: torch.Tensor, clip: bool = False) -> torch.Tensor:
    """Fake-quantize ``x`` on the symmetric int8 grid with a frozen ``scale``.

    Args:
        x (torch.Tensor): Input with the quantization axis last, e.g. ``[B, S, C]``.
        scale (torch.Tensor): Frozen scale broadcast against the last axis of ``x`` (e.g.
            per-channel ``[C]`` or per-tensor scalar). Must be float32 on the same device as
            ``x``: fp32 keeps the quantization math aligned with the deployment kernel.
        clip (bool): When ``True``, zero gradients of values outside the int8 range instead of
            passing them through.

    Returns:
        torch.Tensor: Dequantized tensor with the same shape and dtype as ``x``.
    """
    if scale.dtype != torch.float32:
        raise ValueError(f"scale must be float32 (deployment-aligned), got {scale.dtype}.")
    if scale.device != x.device:
        raise ValueError(f"scale must live on x's device ({x.device}), got {scale.device}.")
    try:
        torch.broadcast_shapes(x.shape, scale.shape)
    except RuntimeError as e:
        raise ValueError(f"scale shape {tuple(scale.shape)} cannot broadcast against x shape {tuple(x.shape)}.") from e
    return _Int8FakeQuant.apply(x, scale, clip)
