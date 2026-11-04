"""KV-cache quantization configuration."""

import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


_KV_QUANT_ENV = "XTUNER_QUANT_KV"

_DEFAULT_KV_QUANT_SCOPE: list[Literal["nope_dsa_ckv"]] = ["nope_dsa_ckv"]


class KVQuantConfig(BaseModel):
    """Configuration for KV-cache quantization-aware training.

    The `kv_quant` entry fake-quantizes the DSA compressed-KV latent -- the tensor a
    deployed engine stores as its int8 KV cache -- during training, so the master weights
    adapt to the quantized values. With `kv_quant="off"` (the default) the forward stays
    bit-identical to the unquantized path.

    Args:
        kv_quant (Literal["off", "per_token_dynamic"]): Quantization mode for the compressed-KV
            latent. ``"per_token_dynamic"`` scales each token by its own ``amax / 127``; the
            per-channel static variant (calibrated, frozen scale) arrives with the calibration
            change.
        kv_quant_scope (list[Literal["nope_dsa_ckv"]]): Closed set of face names the KV
            fake-quant applies to. ``"nope_dsa_ckv"`` maps to the post-``kv_a_layernorm``
            injection point in NoPE-DSA attention. Extend the Literal when a new face lands.
    """

    model_config = ConfigDict(extra="forbid")

    kv_quant: Literal["off", "per_token_dynamic"] = "off"
    # Closed Literal instead of free strings: a typo'd face name must fail at config time
    # rather than silently gating the fake-quant off.
    kv_quant_scope: list[Literal["nope_dsa_ckv"]] = Field(default_factory=lambda: list(_DEFAULT_KV_QUANT_SCOPE))


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


def kv_quant_config_from_env() -> KVQuantConfig | None:
    """Build a :class:`KVQuantConfig` from the ``XTUNER_QUANT_KV`` environment variable.

    Returns:
        KVQuantConfig | None: Parsed config, or ``None`` when the variable is unset or selects
        "off".
    """
    raw = os.environ.get(_KV_QUANT_ENV, "off").strip().lower()
    if raw in ("", "off", "0", "false"):
        return None
    if raw == "per_token_dynamic":
        return KVQuantConfig(kv_quant="per_token_dynamic")
    raise ValueError(f"{_KV_QUANT_ENV}={raw!r} is invalid; expected 'off' or 'per_token_dynamic'.")
