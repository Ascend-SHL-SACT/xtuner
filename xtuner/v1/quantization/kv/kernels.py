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
        if not clip:
            # No clip mask to save, so run the fused chain: one fp32 materialization
            # (``x / s``) plus in-place rounding ops instead of the generic path's six
            # full-size fp32 temporaries. Bit-exact against the generic path below:
            # ``max(mx, -mn)`` selected over the raw dtype equals ``amax(x.float().abs())``
            # (sign flip and the low-to-high dtype upcast are exact), and a low-precision
            # operand divided by an fp32 scale promotes to the same fp32 dividend. The
            # clamp stays: it is provably dead for a dynamic scale (|x| <= amax = 127 * s)
            # but live for a caller-supplied static scale (values can exceed the range).
            s = scale
            if s is None:
                mn, mx = torch.aminmax(x, dim=-1, keepdim=True)
                s = torch.maximum(mx, mn.neg()).float().div_(127.0).clamp_min_(1e-12)
            q = torch.div(x, s).round_().clamp_(-127.0, 127.0).mul_(s)
            ctx.clip = False
            return q.to(x.dtype)
        x32 = x.float()
        if scale is None:
            # Per-token dynamic: the scale is derived inside the Function so no gradient
            # flows through the amax path.
            scale = x32.abs().amax(dim=-1, keepdim=True).div_(127.0).clamp_min_(1e-12)
        ctx.clip = clip
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
            passing them through. Meaningful only for a caller-supplied static scale: a dynamic
            scale derives from the amax itself, so the clip boundary ``127 * scale`` equals the
            amax and clipping is a semantic no-op (fp32 rounding can even spuriously zero the
            amax element's gradient).

    Returns:
        torch.Tensor: Dequantized tensor with the same shape and dtype as ``x``.
    """
    return _Int8FakeQuant.apply(x, None, clip)


# The deployment kernel's only supported block shape (torch_npu.npu_dynamic_block_quant,
# row_block_size=1 / col_block_size=128): one symmetric-int8 scale per token per 128-channel
# tile of the ckv latent. Hardcoded to the consumer contract; not a free parameter.
_DYNAMIC_BLOCK_TILE = 128


class _Int8FakeQuantDynamicBlock(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, clip: bool) -> torch.Tensor:
        if x.shape[-1] % _DYNAMIC_BLOCK_TILE != 0:
            raise ValueError(
                f"Last dimension {x.shape[-1]} must be a multiple of the dynamic-block tile {_DYNAMIC_BLOCK_TILE}."
            )
        x32 = x.float()
        tiles = x32.reshape(*x32.shape[:-1], x32.shape[-1] // _DYNAMIC_BLOCK_TILE, _DYNAMIC_BLOCK_TILE)
        # On-device probe of the deployment op (claude_added/probe_npu_dynamic_block_20261106.py):
        # scale = amax * (1/127) in fp32 -- multiplying by the reciprocal constant, not dividing;
        # an all-zero tile yields scale == 0 with y == 0 (no NaN, no minimum-scale clamp).
        scale = tiles.abs().amax(dim=-1, keepdim=True).mul_(1.0 / 127.0)
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        ctx.clip = clip
        if clip:
            ctx.save_for_backward(tiles.abs().le(127.0 * scale).reshape(x32.shape))
        q = torch.clamp(torch.round(tiles / safe), -127.0, 127.0)
        return q.mul(scale).reshape(x32.shape).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.clip:
            (mask,) = ctx.saved_tensors
            zero = torch.zeros((), dtype=grad_output.dtype, device=grad_output.device)
            grad_output = torch.where(mask, grad_output, zero)
        return grad_output, None


def int8_fake_quant_dynamic_block(x: torch.Tensor, clip: bool = False) -> torch.Tensor:
    """Fake-quantize ``x`` on the symmetric int8 grid with one dynamic scale per (1, 128) tile.

    Grid-aligned with the deployment kernel vllm-ascend SFA C8 uses for this latent
    (``torch_npu.npu_dynamic_block_quant(row_block_size=1, col_block_size=128)``): per-token,
    per-128-channel-tile fp32 scales, int8 levels clamped to [-127, 127], straight-through
    backward. The last dimension must be a multiple of 128 (the ckv latent is 512).

    Args:
        x (torch.Tensor): Input with the quantization axis last, e.g. ``[B, S, C]`` where
            ``C`` is a multiple of 128.
        clip (bool): When ``True``, zero gradients of values outside the int8 range instead of
            passing them through. Meaningful only for a caller-supplied static scale: a dynamic
            tile scale derives from the tile amax itself, so the clip boundary ``127 * scale``
            equals the amax and clipping is a semantic no-op (fp32 rounding can even spuriously
            zero the amax element's gradient).

    Returns:
        torch.Tensor: Dequantized tensor with the same shape and dtype as ``x``.
    """
    return _Int8FakeQuantDynamicBlock.apply(x, clip)


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
