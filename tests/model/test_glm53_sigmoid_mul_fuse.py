"""GLM-5.3 KDA ``o_norm`` sigmoid-gate fused kernel (``y * sigmoid(g)``) regression tests.

The fused triton kernel (:func:`fused_sigmoid_mul`) replaces the eager
``y * torch.sigmoid(g)`` chain (aclnn Cast+Sigmoid+Mul forward and
MulBackward+SigmoidBackward backward) for NPU bf16 inputs whose last dim is a
supported pow2 width. The kernel mirrors the eager chain's bf16 rounding points
(materialized bf16 sigmoid intermediate in forward; bf16 ``dout*y`` in backward)
so the output and both gradients must match the eager autograd chain bit for bit.

Run with ``XTUNER_DETERMINISTIC=false``: the project's deterministic patch
breaks the triton wheel's autotune, which would otherwise make the fused kernel
unreachable or non-deterministic under the default setting.

TestSigmoidMulFuseBitwise(@pytest.mark.gpu)
    test_o_norm_shape_bitwise: production o_norm shape [1, 131072, 8, 128] --
        forward and both backward grads (dy, dg) bitwise equal to eager autograd.
    test_small_shape_bitwise_and_backward: [1, 8192, 8, 128] -- same forward +
        backward bitwise checks, plus allocator poisoning (NaN-fill then del) to
        prove no unwritten dst rows.
    test_various_pow2_widths: last-dim in {64, 128, 256, 512} with [1, 2048, C] --
        forward bitwise equal.
    test_non_contiguous_falls_back_to_eager: non-contiguous bf16 g -- gate rejects,
        eager fallback bitwise equal to ``y * torch.sigmoid(g)``.
    test_fp16_falls_back_to_eager: fp16 inputs -- gate rejects, eager fallback
        bitwise equal to ``y * torch.sigmoid(g)``.
"""

import pytest
import torch

from xtuner.v1.ops.act_fn import _sigmoid_mul_ok, fused_sigmoid_mul
from xtuner.v1.utils.device import get_device


DEVICE = get_device()


def _eager_vs_fused_fwd_bwd(
    y: torch.Tensor,
    g: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run eager and fused forward+backward with a ones seed grad, return both.

    Compares ``y * torch.sigmoid(g)`` (eager autograd) against
    :func:`fused_sigmoid_mul` (fused triton) for the forward output and both
    backward gradients. Both lanes use ``torch.ones_like(out)`` as the seed
    gradient so the backward comparison is apples-to-apples.

    Args:
        y (torch.Tensor): Normed input (bf16, contiguous).
        g (torch.Tensor): Gate, same shape as ``y`` (bf16, contiguous).

    Returns:
        tuple: ``(out_eager, out_fused, dy_eager, dy_fused, dg_eager, dg_fused)``.
    """
    y_e = y.detach().clone().requires_grad_(True)
    g_e = g.detach().clone().requires_grad_(True)
    out_e = y_e * torch.sigmoid(g_e)
    out_e.backward(torch.ones_like(out_e))

    y_f = y.detach().clone().requires_grad_(True)
    g_f = g.detach().clone().requires_grad_(True)
    out_f = fused_sigmoid_mul(y_f, g_f)
    out_f.backward(torch.ones_like(out_f))

    assert y_e.grad is not None and g_e.grad is not None
    assert y_f.grad is not None and g_f.grad is not None
    return out_e, out_f, y_e.grad, y_f.grad, g_e.grad, g_f.grad


@pytest.mark.gpu
class TestSigmoidMulFuseBitwise:
    def test_o_norm_shape_bitwise(self) -> None:
        shape = [1, 131072, 8, 128]
        y = torch.randn(shape, dtype=torch.bfloat16, device=DEVICE)
        g = torch.randn(shape, dtype=torch.bfloat16, device=DEVICE)
        out_e, out_f, dy_e, dy_f, dg_e, dg_f = _eager_vs_fused_fwd_bwd(y, g)
        assert torch.equal(out_f, out_e), "o_norm forward diverged from eager"
        assert torch.equal(dy_f, dy_e), "o_norm dy diverged from eager"
        assert torch.equal(dg_f, dg_e), "o_norm dg diverged from eager"

    def test_small_shape_bitwise_and_backward(self) -> None:
        shape = [1, 8192, 8, 128]
        y = torch.randn(shape, dtype=torch.bfloat16, device=DEVICE)
        g = torch.randn(shape, dtype=torch.bfloat16, device=DEVICE)
        out_e, out_f, dy_e, dy_f, dg_e, dg_f = _eager_vs_fused_fwd_bwd(y, g)
        assert torch.equal(out_f, out_e), "small forward diverged from eager"
        assert torch.equal(dy_f, dy_e), "small dy diverged from eager"
        assert torch.equal(dg_f, dg_e), "small dg diverged from eager"

        # allocator poisoning: any row the kernel forgets to write stays NaN
        poison = torch.full((y.numel(),), float("nan"), dtype=torch.bfloat16, device=DEVICE)
        del poison
        got = fused_sigmoid_mul(y, g)
        assert not torch.isnan(got.float()).any(), "unwritten rows left in dst"
        assert torch.equal(got, y * torch.sigmoid(g)), "poisoned forward diverged from eager"

    @pytest.mark.parametrize("c", [64, 128, 256, 512])
    def test_various_pow2_widths(self, c: int) -> None:
        shape = [1, 2048, c]
        y = torch.randn(shape, dtype=torch.bfloat16, device=DEVICE)
        g = torch.randn(shape, dtype=torch.bfloat16, device=DEVICE)
        ref = y * torch.sigmoid(g)
        got = fused_sigmoid_mul(y, g)
        assert torch.equal(got, ref), f"width {c} forward diverged from eager"

    def test_non_contiguous_falls_back_to_eager(self) -> None:
        base = torch.randn([128, 256], dtype=torch.bfloat16, device=DEVICE)
        g = base.t().contiguous().t()  # shape [128, 256], non-contiguous
        y = torch.randn([128, 256], dtype=torch.bfloat16, device=DEVICE)
        assert not _sigmoid_mul_ok(y, g), "gate should reject non-contiguous g"
        ref = y * torch.sigmoid(g)
        got = fused_sigmoid_mul(y, g)
        assert torch.equal(got, ref), "non-contiguous eager fallback diverged"

    def test_fp16_falls_back_to_eager(self) -> None:
        y = torch.randn([1, 2048, 128], dtype=torch.float16, device=DEVICE)
        g = torch.randn([1, 2048, 128], dtype=torch.float16, device=DEVICE)
        assert not _sigmoid_mul_ok(y, g), "gate should reject fp16"
        ref = y * torch.sigmoid(g)
        got = fused_sigmoid_mul(y, g)
        assert torch.equal(got, ref), "fp16 eager fallback diverged"
