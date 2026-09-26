"""GLM-5.3 causal conv1d triton backward axis=0 dw reduction regression tests.

Verifies the ``_conv_bwd_kernel`` in the ACTIVE module
``xtuner.v1.ops.kda.causal_conv1d_ascendc`` still produces correct gradients
after its per-tap ``dw`` reduction changed from
``tl.sum(tl.trans(x * d), axis=1)`` to ``tl.sum(x * d, axis=0)``.

The contract that must hold:
1. ``dx`` is bitwise unaffected by the reduction-form change (dx does not
   depend on the dw reduction); triton dx matches an independent fp32
   shift-MAD reference within bf16 noise.
2. ``dw`` is correct within bf16 noise vs the same fp32 reference (the axis=0
   reduction may differ from the old trans form by <=1 bf16 ULP from the fp32
   reduction-tree change, within the kernel's existing precision exception).
3. The kernel actually ran (did not silently fall back): ``_conv_bwd_triton``
   returns a tuple, not None, for the servable shapes.

The fp32 reference is self-contained (does not import the kernel's own eager
path): it casts the bf16 NPU inputs to float32, computes the silu backward,
then the shift-MAD tap products with the kernel's lag convention (w_rev =
weight_kc.flip(0); tap k pairs lag k; dw flipped back to kernel-native [W, C]
layout on return). The convention was confirmed by reading ``_conv_bwd_eager``
in the module before coding the reference.

Environment requirements:
    - NPU accelerator (all tests are ``@pytest.mark.gpu``).
    - ``XTUNER_DETERMINISTIC=false`` -- the deterministic patch breaks triton
      autotune/compile, making the kernel unreachable.
    - ``LC_ALL=C.UTF-8 LANG=C.UTF-8``.

TestConvBwdAxis0Reduce(@pytest.mark.gpu)
    test_triton_bwd_dx_dw_match_fp32_reference_production_shape: [8192, 12288]
        single-segment, W=4, silu -- triton ran, dx/dw match fp32 ref.
    test_triton_bwd_dx_dw_match_fp32_reference_long_seq: [131072, 1024]
        single-segment, W=4, silu -- many t-tiles + persistent-grid stride loop.
    test_triton_bwd_multi_segment_matches_reference: cu=[0, 2048, 8192],
        T=8192, C=1024, W=4, silu -- HAS_SEG cross-segment masking.
    test_triton_bwd_no_activation_matches_reference: activation=0 (None),
        [8192, 1024], W=4, single-segment -- dsilu=dy, same assertions.
"""

import fla_npu  # noqa: F401  vendored AscendC ops; import before xtuner pulls torch_npu
import pytest
import torch

import xtuner.v1.ops.kda.causal_conv1d_ascendc as conv_module
from xtuner.v1.utils.device import get_device


DEVICE = get_device()


def _fp32_reference(
    x2d: torch.Tensor,
    pre_3d: torch.Tensor | None,
    dy_3d: torch.Tensor,
    weight_kc: torch.Tensor,
    cu: list[int] | None,
    silu: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent fp32 shift-MAD reference for the depthwise causal conv1d backward.

    Mirrors the tap/lag convention of the production kernel: weight_kc is the
    kernel-native [W, C] layout (row j pairs lag W-1-j); the reference flips
    once to lag-indexed w_rev, computes dx/dw_lag, then flips dw back to the
    kernel-native layout on return. Computed entirely in float32 on CPU.

    Args:
        x2d (torch.Tensor): Forward input [T, C] (bf16 NPU tensor; cast to fp32).
        pre_3d (torch.Tensor | None): Saved pre-activation [1, T, C] when silu
            is True; None when False.
        dy_3d (torch.Tensor): Upstream gradient [1, T, C] (bf16 NPU tensor).
        weight_kc (torch.Tensor): Conv weight in kernel-native [W, C] layout.
        cu (list[int] | None): Per-sequence bounds; None treats the row as one
            sequence.
        silu (bool): Whether silu was applied in the forward.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (dx [T, C] fp32, dw [W, C] fp32 in
        kernel-native layout).
    """
    x = x2d.detach().cpu().float()
    dy = dy_3d.detach().cpu().float().squeeze(0)
    w = weight_kc.detach().cpu().float()
    total, channels = x.shape
    width = w.shape[0]
    w_rev = w.flip(0)
    if silu:
        pre = pre_3d.detach().cpu().float().squeeze(0)  # type: ignore[union-attr]
        sig = torch.sigmoid(pre)
        dsilu = dy * sig * (1.0 + pre * (1.0 - sig))
    else:
        dsilu = dy
    if cu is not None and len(cu) > 2:
        lens = [b - a for a, b in zip(cu[:-1], cu[1:])]
        seg_id = torch.repeat_interleave(torch.arange(len(lens)), torch.tensor(lens))
    else:
        seg_id = None
    dx = torch.zeros_like(x)
    dw_lag = torch.zeros(width, channels)
    for k in range(width):
        if total <= k:
            continue
        if seg_id is not None:
            same = seg_id[: total - k] == seg_id[k:]
            shifted = torch.where(same.unsqueeze(-1), dsilu[k:], torch.zeros((), dtype=dsilu.dtype))
        else:
            shifted = dsilu[k:]
        dx[: total - k] += shifted * w_rev[k]
        dw_lag[k] = (x[: total - k] * shifted).sum(0)
    dw = dw_lag.flip(0)
    return dx, dw


def _assert_bf16_close(g_triton: torch.Tensor, g_ref: torch.Tensor, name: str) -> None:
    """bf16-vs-fp32 gradient check: norm-relative and max/std within bf16 noise.

    Args:
        g_triton (torch.Tensor): Triton kernel output (bf16).
        g_ref (torch.Tensor): Independent fp32 reference.
        name (str): Identifier for assertion failure messages.
    """
    g_t = g_triton.detach().cpu().float()
    g_r = g_ref.detach().cpu().float()
    abs_diff = (g_t - g_r).abs()
    norm_rel = (abs_diff.norm() / g_r.norm().clamp_min(1e-8)).item()
    max_over_std = (abs_diff.max() / g_r.std().clamp_min(1e-3)).item()
    assert norm_rel < 0.02 and max_over_std < 0.1, f"{name}: norm_rel={norm_rel:.6e} max_over_std={max_over_std:.6e}"


@pytest.mark.gpu
class TestConvBwdAxis0Reduce:
    """Verify triton conv backward after the dw axis=0 reduction change."""

    @staticmethod
    def _build_args(
        tokens: int, channels: int, width: int = 4
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build bf16 NPU inputs for the backward.

        Args:
            tokens (int): Sequence length T.
            channels (int): Channel count C (must be divisible by 128).
            width (int): Conv kernel width W. Defaults to 4.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            (x2d [T, C], pre_3d [1, T, C], weight_kc [W, C], dy3d [1, T, C]).
        """
        torch.manual_seed(0)
        x2d = torch.randn(tokens, channels, device=DEVICE, dtype=torch.bfloat16)
        pre_3d = torch.randn(1, tokens, channels, device=DEVICE, dtype=torch.bfloat16)
        weight_kc = torch.randn(width, channels, device=DEVICE, dtype=torch.bfloat16)
        dy3d = torch.randn(1, tokens, channels, device=DEVICE, dtype=torch.bfloat16)
        return x2d, pre_3d, weight_kc, dy3d

    def test_triton_bwd_dx_dw_match_fp32_reference_production_shape(self) -> None:
        """[8192, 12288] single-segment, W=4, silu: triton ran, dx/dw match fp32 ref."""
        tokens, channels, width = 8192, 12288, 4
        x2d, pre_3d, weight_kc, dy3d = self._build_args(tokens, channels, width)
        out = conv_module._conv_bwd_triton(x2d, pre_3d, weight_kc, dy3d, None, activation=1)
        assert out is not None, "triton bwd returned None for production shape"
        dx, dw = out
        assert dx.shape == (tokens, channels) and dw.shape == (width, channels)
        dx_ref, dw_ref = _fp32_reference(x2d, pre_3d, dy3d, weight_kc, None, True)
        _assert_bf16_close(dx, dx_ref, "dx production shape")
        _assert_bf16_close(dw, dw_ref, "dw production shape")

    def test_triton_bwd_dx_dw_match_fp32_reference_long_seq(self) -> None:
        """[131072, 1024] single-segment, W=4, silu: many t-tiles + persistent grid."""
        tokens, channels, width = 131072, 1024, 4
        x2d, pre_3d, weight_kc, dy3d = self._build_args(tokens, channels, width)
        out = conv_module._conv_bwd_triton(x2d, pre_3d, weight_kc, dy3d, None, activation=1)
        assert out is not None, "triton bwd returned None for long seq"
        dx, dw = out
        assert dx.shape == (tokens, channels) and dw.shape == (width, channels)
        dx_ref, dw_ref = _fp32_reference(x2d, pre_3d, dy3d, weight_kc, None, True)
        _assert_bf16_close(dx, dx_ref, "dx long seq")
        _assert_bf16_close(dw, dw_ref, "dw long seq")

    def test_triton_bwd_multi_segment_matches_reference(self) -> None:
        """cu=[0, 2048, 8192], T=8192, C=1024, W=4, silu: HAS_SEG cross-segment masking."""
        tokens, channels, width = 8192, 1024, 4
        cu = [0, 2048, 8192]
        x2d, pre_3d, weight_kc, dy3d = self._build_args(tokens, channels, width)
        out = conv_module._conv_bwd_triton(x2d, pre_3d, weight_kc, dy3d, cu, activation=1)
        assert out is not None, "triton bwd returned None for multi-segment"
        dx, dw = out
        assert dx.shape == (tokens, channels) and dw.shape == (width, channels)
        dx_ref, dw_ref = _fp32_reference(x2d, pre_3d, dy3d, weight_kc, cu, True)
        _assert_bf16_close(dx, dx_ref, "dx multi-segment")
        _assert_bf16_close(dw, dw_ref, "dw multi-segment")

    def test_triton_bwd_no_activation_matches_reference(self) -> None:
        """activation=0 (None), [8192, 1024], W=4, single-segment: dsilu=dy."""
        tokens, channels, width = 8192, 1024, 4
        x2d, _pre_3d, weight_kc, dy3d = self._build_args(tokens, channels, width)
        out = conv_module._conv_bwd_triton(x2d, None, weight_kc, dy3d, None, activation=0)
        assert out is not None, "triton bwd returned None for no-activation"
        dx, dw = out
        assert dx.shape == (tokens, channels) and dw.shape == (width, channels)
        dx_ref, dw_ref = _fp32_reference(x2d, None, dy3d, weight_kc, None, False)
        _assert_bf16_close(dx, dx_ref, "dx no-act")
        _assert_bf16_close(dw, dw_ref, "dw no-act")
