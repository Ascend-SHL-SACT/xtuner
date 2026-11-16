# Copyright (c) OpenMMLab. All rights reserved.
"""Regression tests for the glm53 causal-conv1d triton backward.

The production backward launches a persistent-grid triton kernel whose row
length ``T`` varies per step. Two hazards are pinned here:

- ``T`` must stay out of the specialization key so only the startup-probed
  binary set exists (a cache-reused extra variant faulted with an MTE DDR
  out-of-bounds on masked-off load lanes);
- every load row index is clamped into ``[0, T)`` so no address can leave the
  tensor even if a masked MTE issue is lowered unmasked.

Both properties must keep the triton output numerically equivalent to an
fp32 eager reference across the T values and segment patterns production
hits. The reference is evaluated in fp32 because the triton kernel is
strictly more accurate than the bf16 eager column sums (its approved
precision exception): a bf16 reference accumulates its own product rounding
over T and false-fails the dw comparison at large T.
"""

import pytest
import torch

from xtuner.v1.ops.kda import causal_conv1d_ascendc as conv_mod
from xtuner.v1.ops.kda.causal_conv1d_ascendc import (
    _conv_bwd_eager,
    _conv_bwd_triton,
    _conv_bwd_triton_available,
)


pytestmark = pytest.mark.gpu


def _cu_from_lens(lens: list[int]) -> list[int]:
    cu = [0]
    for length in lens:
        cu.append(cu[-1] + length)
    return cu


class TestConvBwdTriton:
    @pytest.mark.parametrize(
        "total,lens",
        [
            (64, [64]),  # single segment: HAS_SEG=False binary
            (70, [31, 39]),  # startup-probe shape, odd T
            (1023, [1, 2, 1020]),  # odd T, unit-length leading segments
            (1024, [1024]),  # div-16 T, single segment
            (1024, [512, 512]),  # div-16 T, multi segment
            (4096, [300] * 13 + [196]),  # production-like segment count
            (8192, [8192]),  # SP16-era packed row length, spans many tiles
        ],
    )
    def test_matches_eager_across_shapes_and_segments(self, total: int, lens: list[int]):
        torch.manual_seed(0)
        device = torch.device("cuda")
        channels = 256
        x = torch.randn(total, channels, dtype=torch.bfloat16, device=device)
        w = torch.randn(4, channels, dtype=torch.bfloat16, device=device)
        y = torch.randn(1, total, channels, dtype=torch.bfloat16, device=device)
        dy = torch.randn(1, total, channels, dtype=torch.bfloat16, device=device)
        cu = _cu_from_lens(lens)

        out = _conv_bwd_triton(x, y, w, dy, cu, activation=1)
        ref = _conv_bwd_eager(x.float(), y.float(), w.float(), dy.float(), cu, activation=1)

        assert out is not None and ref is not None
        dx, dw = out
        assert dx.shape == x.shape and dw.shape == w.shape
        assert torch.isfinite(dx.float()).all()
        assert torch.isfinite(dw.float()).all()
        assert torch.allclose(dx.float(), ref[0], atol=5e-2, rtol=5e-2)
        assert torch.allclose(dw.float(), ref[1], atol=5e-2, rtol=5e-2)

    def test_matches_eager_without_activation(self):
        torch.manual_seed(1)
        device = torch.device("cuda")
        total, channels = 70, 256
        x = torch.randn(total, channels, dtype=torch.bfloat16, device=device)
        w = torch.randn(4, channels, dtype=torch.bfloat16, device=device)
        dy = torch.randn(1, total, channels, dtype=torch.bfloat16, device=device)
        cu = [0, 31, 31, total]

        out = _conv_bwd_triton(x, None, w, dy, cu, activation=0)
        ref = _conv_bwd_eager(x.float(), None, w.float(), dy.float(), cu, activation=0)

        assert out is not None and ref is not None
        assert torch.allclose(out[0].float(), ref[0], atol=5e-2, rtol=5e-2)
        assert torch.allclose(out[1].float(), ref[1], atol=5e-2, rtol=5e-2)

    def test_t_is_not_specialized(self):
        marked = getattr(conv_mod._conv_bwd_kernel, "do_not_specialize", [])
        marked = [marked] if isinstance(marked, str) else list(marked)
        assert "T" in marked

    def test_abp_constexpr_in_signature(self):
        assert "ABP" in conv_mod._conv_bwd_kernel.arg_names

    def test_startup_probe_available(self):
        assert _conv_bwd_triton_available(torch.device("cuda"))
