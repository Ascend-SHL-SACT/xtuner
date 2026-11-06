"""KV-cache quantization configuration."""

import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


_KV_QUANT_ENV = "XTUNER_QUANT_KV"

# Optional per-run override of KVQuantConfig.kv_calib_num_batches for the static modes:
# calibration rides the first training steps, so keep the threshold below the planned step
# count -- a run that ends earlier never freezes and writes no sidecar.
_KV_QUANT_CALIB_BATCHES_ENV = "XTUNER_QUANT_KV_CALIB_BATCHES"

_DEFAULT_KV_QUANT_SCOPE: list[Literal["nope_dsa_ckv"]] = ["nope_dsa_ckv"]

# The static modes freeze their scale once (lazy in-training calibration or checkpoint
# sidecar) instead of deriving it per forward call.
STATIC_KV_QUANT_MODES: tuple[Literal["per_tensor_static"], ...] = ("per_tensor_static",)


class KVQuantConfig(BaseModel):
    """Configuration for KV-cache quantization-aware training.

    The `kv_quant` entry fake-quantizes the DSA compressed-KV latent -- the tensor a
    deployed engine stores as its int8 KV cache -- during training, so the master weights
    adapt to the quantized values. With `kv_quant="off"` (the default) the forward stays
    bit-identical to the unquantized path.

    Args:
        kv_quant (Literal["off", "per_token_dynamic", "per_tensor_static"]):
            Quantization mode for the compressed-KV latent. ``"per_token_dynamic"`` scales
            each token by its own ``amax / 127``. ``"per_tensor_static"`` freezes one scale
            per layer (scalar); the frozen scale comes from the first ``kv_calib_num_batches``
            training steps or from a checkpoint sidecar.
        kv_quant_scope (list[Literal["nope_dsa_ckv"]]): Closed set of face names the KV
            fake-quant applies to. ``"nope_dsa_ckv"`` maps to the post-``kv_a_layernorm``
            injection point in NoPE-DSA attention. Extend the Literal when a new face lands.
        kv_calib_num_batches (int): Static modes only: number of training steps the lazy
            calibration collects before freezing the scales. No dataloader is touched.
    """

    model_config = ConfigDict(extra="forbid")

    kv_quant: Literal["off", "per_token_dynamic", "per_tensor_static"] = "off"
    # Closed Literal instead of free strings: a typo'd face name must fail at config time
    # rather than silently gating the fake-quant off.
    kv_quant_scope: list[Literal["nope_dsa_ckv"]] = Field(default_factory=lambda: list(_DEFAULT_KV_QUANT_SCOPE))
    kv_calib_num_batches: int = Field(default=64, gt=0)


def kv_fakequant_enabled(kv_quant_cfg: KVQuantConfig | None, face: str) -> bool:
    """Whether the KV fake-quant applies to ``face`` under ``kv_quant_cfg``.

    Args:
        kv_quant_cfg (KVQuantConfig | None): Active KV quantization config, or ``None`` when the feature is off.
        face (str): Quantization face name, e.g. ``"nope_dsa_ckv"``.

    Returns:
        bool: ``True`` when the face is in scope and ``kv_quant`` is not ``"off"``.
    """
    if kv_quant_cfg is None or kv_quant_cfg.kv_quant == "off":
        return False
    return face in kv_quant_cfg.kv_quant_scope


def _kv_calib_batches_from_env() -> int:
    raw = os.environ.get(_KV_QUANT_CALIB_BATCHES_ENV, "").strip()
    if not raw:
        return KVQuantConfig.model_fields["kv_calib_num_batches"].default
    try:
        value = int(raw)
    except ValueError as e:
        raise ValueError(f"{_KV_QUANT_CALIB_BATCHES_ENV}={raw!r} is not a valid integer.") from e
    if value <= 0:
        raise ValueError(f"{_KV_QUANT_CALIB_BATCHES_ENV}={raw!r} must be a positive integer.")
    return value


def kv_quant_config_from_env() -> KVQuantConfig | None:
    """Build a :class:`KVQuantConfig` from the ``XTUNER_QUANT_KV`` environment variable.

    The static modes also read the optional ``XTUNER_QUANT_KV_CALIB_BATCHES`` override of
    ``kv_calib_num_batches``; setting that variable with ``per_token_dynamic`` is an error.

    Returns:
        KVQuantConfig | None: Parsed config, or ``None`` when the variable is unset or selects
        "off".
    """
    raw = os.environ.get(_KV_QUANT_ENV, "off").strip().lower()
    if raw in ("", "off", "0", "false"):
        return None
    if raw == "per_token_dynamic":
        if os.environ.get(_KV_QUANT_CALIB_BATCHES_ENV, "").strip():
            raise ValueError(
                f"{_KV_QUANT_CALIB_BATCHES_ENV} only applies to the static KV quant modes; unset it or switch "
                "to 'per_tensor_static'."
            )
        return KVQuantConfig(kv_quant="per_token_dynamic")
    if raw == "per_tensor_static":
        return KVQuantConfig(kv_quant="per_tensor_static", kv_calib_num_batches=_kv_calib_batches_from_env())
    raise ValueError(
        f"{_KV_QUANT_ENV}={raw!r} is invalid; expected 'off', 'per_token_dynamic' or 'per_tensor_static'."
    )
