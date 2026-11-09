"""W8A8 linear quantization configuration."""

import os
from typing import Literal

from pydantic import BaseModel, ConfigDict


_LINEAR_QUANT_ENV = "XTUNER_QUANT_LINEAR"

# Optional routed-experts GEMM engine override: "fake" (default) runs the fake-quant
# grouped GEMM with autograd; "npu_quant_gmm" additionally runs the real aclnn op
# (MindSpeed ``npu_quant_gmm``) in a dual-path forward.
_LINEAR_QUANT_GMM_ENV = "XTUNER_QUANT_LINEAR_GMM"

# Optional dequant-grid override for the NPU fused fake-quant path: "fp32" (default)
# dequantizes with the fp32 promote-mul, bitwise the deployment reconstruction; "native"
# multiplies in the tensor's own dtype instead, re-rounding the kernel's fp32 scale once
# (at most 2^-8 relative) to cut the dequant's memory traffic.
_LINEAR_QUANT_DEQUANT_ENV = "XTUNER_QUANT_LINEAR_DEQUANT"


class LinearQuantConfig(BaseModel):
    """Configuration for W8A8 linear quantization-aware training.

    The ``w8a8_dynamic`` mode fake-quantizes the routed dense/expert linear weights on
    the per-output-channel int8 grid and the activations on the per-token dynamic int8
    grid -- the grids the deployed engine's ``W8A8_DYNAMIC`` scheme (vllm-ascend
    ``AscendW8A8DynamicLinearMethod`` / ``AscendW8A8DynamicFusedMoEMethod``) applies at
    serve time -- so the master weights adapt to the quantized values. With
    ``linear_quant="off"`` (the default) every factory returns the original module
    class and the forward stays bit-identical to the unquantized path.

    Args:
        linear_quant (Literal["off", "w8a8_dynamic"]): Linear quantization mode.
            ``"w8a8_dynamic"`` fake-quantizes weight (per output channel) and activation
            (per token) on the symmetric int8 grid of the deployed ``W8A8_DYNAMIC``
            scheme.
        gmm_engine (Literal["fake", "npu_quant_gmm"]): Routed-experts grouped-GEMM
            engine. ``"fake"`` computes the quantized grouped GEMM with the native
            ``group_gemm`` (autograd flows through the STE graph). ``"npu_quant_gmm"``
            runs the real aclnn op (MindSpeed ``npu_quant_gmm``) as a dual-path forward:
            the value comes from the real op, the backward graph from the fake path, so
            gradients stay bit-identical to the ``"fake"`` engine.
    """

    model_config = ConfigDict(extra="forbid")

    linear_quant: Literal["off", "w8a8_dynamic"] = "off"
    gmm_engine: Literal["fake", "npu_quant_gmm"] = "fake"


def _gmm_engine_from_env() -> Literal["fake", "npu_quant_gmm"]:
    raw = os.environ.get(_LINEAR_QUANT_GMM_ENV, "fake").strip().lower()
    if raw in ("", "fake"):
        return "fake"
    if raw == "npu_quant_gmm":
        return "npu_quant_gmm"
    raise ValueError(f"{_LINEAR_QUANT_GMM_ENV}={raw!r} is invalid; expected 'fake' or 'npu_quant_gmm'.")


def _dequant_grid_from_env() -> Literal["fp32", "native"]:
    raw = os.environ.get(_LINEAR_QUANT_DEQUANT_ENV, "fp32").strip().lower()
    if raw in ("", "fp32"):
        return "fp32"
    if raw == "native":
        return "native"
    raise ValueError(f"{_LINEAR_QUANT_DEQUANT_ENV}={raw!r} is invalid; expected 'fp32' or 'native'.")


def linear_quant_config_from_env() -> LinearQuantConfig | None:
    """Build a :class:`LinearQuantConfig` from ``XTUNER_QUANT_LINEAR``.

    The ``w8a8_dynamic`` mode also reads the optional ``XTUNER_QUANT_LINEAR_GMM``
    routed-experts engine override (default ``"fake"``). Setting the engine variable to
    ``"npu_quant_gmm"`` while ``XTUNER_QUANT_LINEAR`` is off is an error (mirroring the
    KV aux-env rejection convention); leaving it unset or ``"fake"`` is allowed so
    launch scripts exporting both variables unconditionally keep working.

    The optional ``XTUNER_QUANT_LINEAR_DEQUANT`` selects the NPU fused path's dequant
    grid (default ``"fp32"``). Invalid values of either aux variable raise regardless of
    the mode (a typo should never pass silently); ``"npu_quant_gmm"`` / ``"native"`` with
    the mode off are semantic errors, while unset / default values with the mode off are
    allowed so unconditional launch exports keep working.

    Returns:
        LinearQuantConfig | None: Parsed config, or ``None`` when the variable is unset
        or selects "off".
    """
    # Both aux variables are parsed before the mode branch so an invalid value raises
    # regardless of the main switch (a typo must never pass silently); the parsed values
    # then drive the semantic cross-checks below.
    gmm = _gmm_engine_from_env()
    dequant = _dequant_grid_from_env()
    raw = os.environ.get(_LINEAR_QUANT_ENV, "off").strip().lower()
    if raw in ("", "off", "0", "false"):
        if gmm == "npu_quant_gmm":
            raise ValueError(
                f"{_LINEAR_QUANT_GMM_ENV}='npu_quant_gmm' requires {_LINEAR_QUANT_ENV}='w8a8_dynamic'; "
                f"unset it or switch {_LINEAR_QUANT_ENV} on."
            )
        if dequant == "native":
            raise ValueError(
                f"{_LINEAR_QUANT_DEQUANT_ENV}='native' requires {_LINEAR_QUANT_ENV}='w8a8_dynamic'; "
                f"unset it or switch {_LINEAR_QUANT_ENV} on."
            )
        return None
    if raw == "w8a8_dynamic":
        return LinearQuantConfig(linear_quant="w8a8_dynamic", gmm_engine=gmm)
    raise ValueError(f"{_LINEAR_QUANT_ENV}={raw!r} is invalid; expected 'off' or 'w8a8_dynamic'.")
