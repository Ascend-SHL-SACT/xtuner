# Copyright © 2026 Huawei Technologies Co., Ltd.
"""NPU (fla_npu) backend re-exposing the fla KDA surface used by GLM-5.3-Flash.

`xtuner.v1.module.attention.kda` and `xtuner.v1.ops.kda` speak the fla 0.4.2 surface:
``fused_kda_gate`` / ``chunk_kda`` / ``fused_recurrent_kda`` / ``causal_conv1d`` /
``rms_norm_gated`` plus the ``FusedRMSNormGated`` / ``ShortConvolution`` base modules. On 910C
the ``fla`` wheel is not installed -- only ``fla_npu`` (Ascend C kernels), whose validated
pipeline lives in ``xtuner.v1.ops.kda.kda_op`` (ported from MindSpeed-MM's
``fsdp/models/glm5_next/modeling_glm5_next.py``). This module re-exposes that pipeline under the
fla names, so the calling code stays byte-identical and the backend choice is made in exactly
two places: the getters in ``xtuner/v1/ops/kda/__init__.py`` and the import block of
``xtuner/v1/module/attention/kda.py``.

Gate contract (the one deliberate deviation from fla): fla precomputes the forget gate with
``fused_kda_gate`` and passes the fp32 log-space gate into ``chunk_kda``. The fla_npu chunk
kernel computes the gate *inside* the kernel from raw inputs ``(g_raw, A_log, dt_bias,
lower_bound)`` -- the exact configuration every bitwise loss anchor in this repo was captured
with. To serve both call shapes, ``fused_kda_gate`` records the four inputs and returns ``g_raw``
unchanged, and ``chunk_kda`` consumes the record (validated by tensor identity) and lets the
kernel gate internally. A ``chunk_kda`` call without a matching preceding ``fused_kda_gate``
call is a programming error and raises. Autograd is unaffected: gradients for ``g_raw``,
``A_log`` and ``dt_bias`` are produced by the kernel backward itself.

All heavy imports are lazy, so importing this module is cheap and CPU-only environments only pay
for what they call.
"""

from __future__ import annotations

import math
import os
from typing import Any, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "FusedRMSNormGated",
    "ShortConvolution",
    "causal_conv1d",
    "chunk_kda",
    "fused_kda_gate",
    "fused_recurrent_kda",
    "npu_impl_selected",
    "rms_norm_gated",
]

_BACKEND_ENV = "XTUNER_KDA_BACKEND"


class _GateBundle(TypedDict):
    g_raw: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor | None
    lower_bound: float | None


# One-slot handoff from `fused_kda_gate` to the following `chunk_kda` / `fused_recurrent_kda`
# call (see module docstring). Emptied on consumption and validated by tensor identity, so a
# stale or missing record fails loudly instead of silently mis-gating.
_GATE_BUNDLE: _GateBundle | None = None


def npu_impl_selected() -> bool:
    """Whether the KDA operator surface should resolve to the NPU (fla_npu) backend.

    Set ``XTUNER_KDA_BACKEND`` to force a backend: ``npu`` selects this module, ``fla`` selects
    the original fla imports. Unset, the decision falls to the current accelerator (NPU -> this
    backend, anything else -> fla).

    Returns:
        bool: Whether the NPU backend is selected.
    """
    backend = os.getenv(_BACKEND_ENV, "").strip().lower()
    if backend:
        return backend == "npu"
    from xtuner.v1.utils import get_device

    return get_device() == "npu"


def fused_kda_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Record the gate parameters and pass ``g`` through unchanged.

    The fla_npu chunk kernel computes the gate from these parameters internally (see module
    docstring). This is a no-op for autograd: ``g`` is returned as-is, and the kernel backward
    produces the gradients for ``g_raw``, ``A_log`` and ``dt_bias`` directly.

    Args:
        g (torch.Tensor): Raw gate projection of shape ``[B, T, H, K]``.
        A_log (torch.Tensor): Per-head fp32 decay parameter of shape ``[H]``.
        dt_bias (torch.Tensor | None): Per-head fp32 bias of shape ``[H * K]``.
        lower_bound (float | None): Safe-gate lower bound, or ``None`` to disable it.
        **kwargs (Any): Unsupported fla keywords; ``output_dtype`` must stay at the fp32 default.

    Returns:
        torch.Tensor: ``g``, unchanged.
    """
    if kwargs.get("output_dtype") not in (None, torch.float32):
        raise NotImplementedError(
            "the npu chunk kernel computes the gate in its internal dtype; pass no output_dtype."
        )
    global _GATE_BUNDLE
    _GATE_BUNDLE = {"g_raw": g, "A_log": A_log, "dt_bias": dt_bias, "lower_bound": lower_bound}
    return g


def _consume_gate_bundle(g: torch.Tensor) -> _GateBundle:
    global _GATE_BUNDLE
    bundle = _GATE_BUNDLE
    _GATE_BUNDLE = None
    if bundle is None or bundle["g_raw"] is not g:
        raise RuntimeError(
            "the npu chunk kernel computes the gate internally, so every chunk_kda/"
            "fused_recurrent_kda call must be preceded by fused_kda_gate on the same g tensor."
        )
    return bundle


def _chunk_kda_generic_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float | None,
    use_qk_l2norm: bool,
    safe_gate: bool,
    lower_bound: float | None,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    # Fallback for shapes/dtypes the fla_npu Ascend C kernels reject (fp32 activations, head
    # dim != 128: small parity-test geometries). Same math as `kda_op.chunk_kda_eager` minus
    # its 30B-geometry validation -- the chunk scan helper is imported, not duplicated, so this
    # path shares the bitwise-validated reference math.
    from .kda_op import CHUNK_SIZE, _as_cu_list, _chunk_kda_core, _kda_gate, _l2norm_eager

    if scale is None:
        scale = q.shape[-1] ** -0.5
    if use_qk_l2norm:
        q = _l2norm_eager(q)
        k = _l2norm_eager(k)
    g = _kda_gate(g_raw, A_log, dt_bias, lower_bound if safe_gate else None)
    q_f, k_f, v_f, g_f, b_f = q.float(), k.float(), v.float(), g, beta.float()
    cu = _as_cu_list(cu_seqlens)
    if cu is None:
        return _chunk_kda_core(q_f, k_f, v_f, g_f, b_f, scale, CHUNK_SIZE).to(q.dtype)
    outs = [
        _chunk_kda_core(q_f[:, s:e], k_f[:, s:e], v_f[:, s:e], g_f[:, s:e], b_f[:, s:e], scale, CHUNK_SIZE)
        for s, e in zip(cu[:-1], cu[1:])
    ]
    return torch.cat(outs, dim=1).to(q.dtype)


def _chunk_kda_core(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None,
    cu_seqlens: torch.Tensor | None,
    safe_gate: bool,
) -> tuple[torch.Tensor, None]:
    from .kda_op import chunk_kda_fla_npu

    bundle = _consume_gate_bundle(g)
    dt_bias = bundle["dt_bias"]
    if dt_bias is None:
        # The kernel computes the gate from raw inputs, so dt_bias is not optional here.
        raise RuntimeError("the npu gate-in-kernel path requires dt_bias; fused_kda_gate got none.")
    # The Ascend C kernels only accept fp16/bf16 activations (fp32 A_log/dt_bias) and a fixed
    # K=V=128 head dim -- the published GLM-5.3-Flash geometry. Anything else (fp32 tests,
    # eager parity checks) runs the eager torch path in place.
    use_ascendc = (
        q.device.type == "npu"
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.shape[-1] == 128
        and v.shape[-1] == 128
    )
    if use_ascendc:
        o = chunk_kda_fla_npu(
            q=q,
            k=k,
            v=v,
            g_raw=g,
            beta=beta,
            A_log=bundle["A_log"],
            dt_bias=dt_bias,
            scale=scale,
            use_qk_l2norm=True,
            safe_gate=safe_gate,
            lower_bound=bundle["lower_bound"],
            cu_seqlens=cu_seqlens,
        )
    else:
        o = _chunk_kda_generic_eager(
            q,
            k,
            v,
            g,
            beta,
            bundle["A_log"],
            dt_bias,
            scale,
            use_qk_l2norm=True,
            safe_gate=safe_gate,
            lower_bound=bundle["lower_bound"],
            cu_seqlens=cu_seqlens,
        )
    return o, None


def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    transpose_state_layout: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Chunked Kimi Delta Attention on the fla_npu Ascend C kernels (or eager on CPU).

    Args:
        q (torch.Tensor): Queries of shape ``[B, T, H, K]``.
        k (torch.Tensor): Keys of shape ``[B, T, H, K]``.
        v (torch.Tensor): Values of shape ``[B, T, H, V]``.
        g (torch.Tensor): Raw (pre-gate) projection ``[B, T, H, K]`` as returned by the
            preceding ``fused_kda_gate`` call; the kernel computes the gate internally.
        beta (torch.Tensor): Betas of shape ``[B, T, H]``.
        scale (float | None): Attention scale; defaults to ``K ** -0.5``.
        cu_seqlens (torch.Tensor | None): Packed-sequence offsets, as in the varlen attention
            API.
        safe_gate (bool): Clamp the gate against the recorded lower bound inside the kernel.
        lower_bound (float | None): Ignored; the bound recorded by ``fused_kda_gate`` is
            authoritative, which is what the validated anchors were captured with.
        transpose_state_layout (bool): Accepted for call-shape compatibility; no final state is
            returned, so no state layout exists to transpose.
        use_qk_l2norm_in_kernel (bool): Must stay ``True``; GLM-5.3-Flash always L2-norms q/k,
            and the backward depends on the ``rstd`` that path produces.
        **kwargs (Any): Unsupported fla keywords.

    Returns:
        tuple[torch.Tensor, None]: Outputs ``[B, T, H, V]``, and ``None`` for the final state,
        which this entry point does not produce.
    """
    if not use_qk_l2norm_in_kernel:
        raise NotImplementedError(
            "the npu chunk kernel always L2-norms q/k; the backward depends on the rstd it keeps."
        )
    for unsupported in ("initial_state", "output_final_state", "cp_context", "A_log", "dt_bias"):
        if kwargs.get(unsupported):
            raise NotImplementedError(f"the npu chunk_kda does not support {unsupported!r}.")
    return _chunk_kda_core(q, k, v, g, beta, scale, cu_seqlens, safe_gate)


def fused_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    transpose_state_layout: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """KDA entry for short (unpacked) sequences, mirroring fla's recurrent kernel.

    The fla_npu Ascend C chunk kernel pads ``T`` to its chunk width and handles short sequences
    natively, so this routes through the same validated path as ``chunk_kda`` -- the recurrent
    dispatch exists only to keep the fla call surface. Call-shape contract matches
    ``chunk_kda``; see there for the argument descriptions.

    Returns:
        tuple[torch.Tensor, None]: Outputs ``[B, T, H, V]``, and ``None`` for the final state,
        which this entry point does not produce.
    """
    if not use_qk_l2norm_in_kernel:
        raise NotImplementedError("the npu kernel always L2-norms q/k.")
    for unsupported in ("initial_state", "output_final_state", "cp_context", "A_log", "dt_bias"):
        if kwargs.get(unsupported):
            raise NotImplementedError(f"the npu fused_recurrent_kda does not support {unsupported!r}.")
    return _chunk_kda_core(q, k, v, g, beta, scale, cu_seqlens, safe_gate)


def _causal_conv1d_dense(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
) -> torch.Tensor:
    # x: [B, T, D], weight: [D, W]; depthwise conv with left padding, trimmed causal.
    y = F.conv1d(
        x.transpose(1, 2),
        weight.unsqueeze(1),
        bias=bias,
        stride=1,
        padding=weight.shape[-1] - 1,
        groups=x.shape[-1],
    )[:, :, : x.shape[1]].transpose(1, 2)
    if activation in ("silu", "swish"):
        y = F.silu(y)
    return y


def _causal_conv1d_varlen_dense(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    cu: list[int],
) -> torch.Tensor:
    outs = []
    for start, end in zip(cu[:-1], cu[1:]):
        if end > start:
            outs.append(_causal_conv1d_dense(x[:, start:end], weight, bias, activation))
    return torch.cat(outs, dim=1)


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    backend: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Causal depthwise short convolution: fla_npu Ascend C op on NPU, dense eager elsewhere.

    Args:
        x (torch.Tensor): Input of shape ``[B, T, D]``.
        weight (torch.Tensor): Depthwise kernel of shape ``[D, W]`` (or ``[D, 1, W]``).
        bias (torch.Tensor | None): Optional per-channel bias ``[D]``.
        activation (str | None): ``"silu"``/``"swish"`` or ``None``.
        backend (str | None): fla dispatcher knob; the NPU entry has a single fused path, so it
            is accepted but ignored (the original ``KDAShortConvolution`` passes its base's
            ``self.backend`` through).
        cu_seqlens (torch.Tensor | None): Packed-sequence offsets, so the convolution never
            reads across a document boundary.
        **kwargs (Any): Unsupported fla keywords.

    Returns:
        tuple[torch.Tensor, None]: The convolved tensor, and ``None`` for the final state, which
        this entry point does not produce.
    """
    for unsupported in ("residual", "initial_state", "output_final_state", "cp_context"):
        if kwargs.get(unsupported):
            raise NotImplementedError(f"the npu causal_conv1d does not support {unsupported!r}.")
    if weight.dim() == 3:
        weight = weight.squeeze(1)
    cu = [int(v) for v in cu_seqlens.tolist()] if cu_seqlens is not None else None
    y: torch.Tensor | None = None
    # The Ascend C op takes fp16/bf16 activations with a channel count divisible by 16, and
    # raises (rather than returning None) on other dtypes/channels, so those route straight to
    # the dense path.
    if (
        x.device.type == "npu"
        and x.shape[0] == 1
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.shape[-1] % 16 == 0
    ):
        from .causal_conv1d_ascendc import causal_conv1d_ascendc

        # The Ascend C op rejects (returns None) unsupported shapes: bias, kernel width > 4,
        # batch > 1, missing backend probe. Fall back to the dense path then.
        y = causal_conv1d_ascendc(x, weight, bias, activation, cu)
    if y is None:
        if cu is not None:
            y = _causal_conv1d_varlen_dense(x, weight, bias, activation, cu)
        else:
            y = _causal_conv1d_dense(x, weight, bias, activation)
    return y, None


def rms_norm_gated(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = "sigmoid",
    residual: torch.Tensor | None = None,
    eps: float = 1e-5,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
) -> torch.Tensor:
    """RMSNorm followed by a sigmoid gate (fla ``rms_norm_gated`` surface).

    Port of the q35-validated ``Glm53RMSNormGated``: on NPU the norm runs through
    ``torch_npu.npu_rms_norm`` in two dtype regimes selected by ``XTUNER_NPU_FUSED_GATED_NORM``
    (default on) -- the bf16 regime keeps the activation in bf16 in and out of the kernel with
    fp32 gamma/rstd inside, and the fp32 regime (env ``0``) matches the MindSpeed reference
    bit-for-bit. Elsewhere a strict fp32 eager norm runs.

    Args:
        x (torch.Tensor): Input of shape ``[..., D]``.
        g (torch.Tensor): Gate, broadcastable against ``x``.
        weight (torch.Tensor): Norm scale ``[D]`` (already unsharded by the caller).
        bias (torch.Tensor | None): Must be ``None``; the checkpoint has no gated-norm bias and
            the validated pipeline never adds one.
        activation (str | None): Must be ``"sigmoid"``.
        residual (torch.Tensor | None): Must be ``None`` (no residual path is validated).
        eps (float): Variance epsilon.
        prenorm (bool): Must be ``False``.
        residual_in_fp32 (bool): Must be ``False``.

    Returns:
        torch.Tensor: Gated normed output, same dtype as ``x``.
    """
    if bias is not None:
        raise NotImplementedError("the npu rms_norm_gated does not support a bias.")
    if activation != "sigmoid":
        raise NotImplementedError(f"the npu rms_norm_gated only supports 'sigmoid', got {activation!r}.")
    if residual is not None or prenorm or residual_in_fp32:
        raise NotImplementedError("the npu rms_norm_gated does not support residual/prenorm paths.")

    try:
        import torch_npu  # noqa: F401

        npu_module_loaded = True
    except ImportError:  # pragma: no cover - CPU-only environments
        npu_module_loaded = False
    # Key the fused regimes on the *input* device, not host availability: the module import
    # succeeds on a mixed box while CPU tensors would fail the dispatch into npu_rms_norm.
    npu_input = npu_module_loaded and x.device.type == "npu"
    fused_gated_norm = os.environ.get("XTUNER_NPU_FUSED_GATED_NORM", "1") == "1"
    input_dtype = x.dtype

    if npu_input and input_dtype in (torch.bfloat16, torch.float16) and fused_gated_norm:
        from torch_npu import npu_rms_norm

        y = npu_rms_norm(x, weight.to(torch.float32), eps)[0]
        return y * torch.sigmoid(g.to(input_dtype))

    x = x.to(torch.float32)
    if npu_input:
        from torch_npu import npu_rms_norm

        x = npu_rms_norm(x, weight.to(torch.float32), eps)[0]
    else:
        # Strict FP32 norm (do not downcast on the weights).
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        x = weight.to(torch.float32) * x
    return (x * torch.sigmoid(g.to(torch.float32))).to(input_dtype)


class FusedRMSNormGated(nn.Module):
    """fla-surface RMSNorm + sigmoid gate module (NPU base).

    ``xtuner.v1.module.attention.kda`` subclasses this and overrides ``forward`` to unshard
    ``weight`` first under EP, so only the ``__init__`` state (the attributes its forward reads)
    matters here.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5, activation: str | None = "sigmoid") -> None:
        super().__init__()
        if activation != "sigmoid":
            raise NotImplementedError(f"FusedRMSNormGated only supports 'sigmoid', got {activation!r}.")
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias: nn.Parameter | None = None
        self.eps = eps
        self.activation = activation

    def forward(  # type: ignore[empty-body]
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        residual: torch.Tensor | None = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ) -> torch.Tensor:
        """Overridden by the EP-unsharding subclass in ``xtuner.v1.module.attention.kda``."""
        raise NotImplementedError("overridden by xtuner.v1.module.attention.kda.FusedRMSNormGated")


class ShortConvolution(nn.Module):
    """fla-surface causal depthwise short convolution module (NPU base).

    ``xtuner.v1.module.attention.kda`` subclasses this and overrides ``forward`` to unshard the
    weight under EP/SP, so only the ``__init__`` state matters here. Weight layout matches fla's
    ``[hidden, 1, kernel_size]``; no bias, matching the published checkpoint.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 4,
        activation: str | None = "silu",
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.conv_dim = hidden_size
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(hidden_size, 1, kernel_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if bias else None
        self.activation = activation
        # The caller passes this through to `causal_conv1d`, where it is accepted but ignored.
        self.backend = "ascendc"
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # nn.Conv1d's default init (kaiming uniform with a=sqrt(5)), matching the MindSpeed
        # tree the validated NPU runs used.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.shape[0]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(  # type: ignore[empty-body]
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Overridden by the unsharding subclass in ``xtuner.v1.module.attention.kda``."""
        raise NotImplementedError("overridden by xtuner.v1.module.attention.kda.KDAShortConvolution")
