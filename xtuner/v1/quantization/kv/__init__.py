"""KV-cache quantization-aware training (QAT).

Design constraints (Ascend 910C/910_93, no fp8): master weights always train in their
original precision; int8 exists only as a fake-quant view computed in the forward pass with
a straight-through-estimator backward. The same quantization math is shared by training,
export and the deployment kernel so the three stay bit-aligned.
"""

from .config import KVQuantConfig, kv_fakequant_enabled, kv_quant_config_from_env
from .kernels import int8_fake_quant_dynamic, int8_fake_quant_static


__all__ = [
    "KVQuantConfig",
    "int8_fake_quant_dynamic",
    "int8_fake_quant_static",
    "kv_fakequant_enabled",
    "kv_quant_config_from_env",
]
