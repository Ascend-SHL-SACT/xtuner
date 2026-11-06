"""KV-cache quantization-aware training (QAT).

Design constraints (Ascend 910C/910_93, no fp8): master weights always train in their
original precision; int8 exists only as a fake-quant view computed in the forward pass with
a straight-through-estimator backward. The same quantization math is shared by training,
export and the deployment kernel so the three stay bit-aligned.
"""

from .calibrate import (
    KVScaleLazyCalibrator,
    maybe_load_kv_scale_sidecar,
    maybe_save_kv_scale_sidecar,
    needs_kv_calibration,
)
from .config import STATIC_KV_QUANT_MODES, KVQuantConfig, kv_fakequant_enabled, kv_quant_config_from_env
from .export import maybe_write_hf_kv_scales
from .kernels import int8_fake_quant_dynamic, int8_fake_quant_static


__all__ = [
    "STATIC_KV_QUANT_MODES",
    "KVQuantConfig",
    "KVScaleLazyCalibrator",
    "int8_fake_quant_dynamic",
    "int8_fake_quant_static",
    "kv_fakequant_enabled",
    "kv_quant_config_from_env",
    "maybe_load_kv_scale_sidecar",
    "maybe_save_kv_scale_sidecar",
    "maybe_write_hf_kv_scales",
    "needs_kv_calibration",
]
