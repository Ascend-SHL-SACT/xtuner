"""Linear-layer quantization-aware training (W8A8 dynamic).

Design constraints (same spine as the KV QAT package): master weights always train in
their original precision; int8 exists only as a fake-quant view computed in the forward
pass with a straight-through-estimator backward, on the grids the deployed
``W8A8_DYNAMIC`` scheme (vllm-ascend) applies at serve time. The same quantization math is
shared by training, export and the optional ``npu_quant_gmm`` real-op arm so the three stay
grid-aligned.
"""

from typing import TYPE_CHECKING, Any

from .config import LinearQuantConfig, linear_quant_config_from_env
from .kernels import (
    int8_fake_quant_dynamic,
    per_channel_weight_fake_quant,
    per_channel_weight_quant,
    per_token_act_fake_quant,
    per_token_act_quant,
)


if TYPE_CHECKING:
    from .export import maybe_write_hf_w8a8_linear
    from .modules import Int8W8A8DynamicGroupedLinear, Int8W8A8DynamicLinear


__all__ = [
    "Int8W8A8DynamicGroupedLinear",
    "Int8W8A8DynamicLinear",
    "LinearQuantConfig",
    "int8_fake_quant_dynamic",
    "linear_quant_config_from_env",
    "maybe_write_hf_w8a8_linear",
    "per_channel_weight_fake_quant",
    "per_channel_weight_quant",
    "per_token_act_fake_quant",
    "per_token_act_quant",
]


# ``.modules`` and ``.export`` load lazily (PEP 562): both subclass/import from
# ``xtuner.v1.module.linear.linear`` and ``...moe_group_linear``, which import this
# package's ``.config`` at module level. Importing them eagerly here would close an import
# cycle (linear.py -> this package -> .modules -> linear.py, half-initialized) the moment
# any ``xtuner.v1.module`` module loads first. The factories already import them lazily from
# their submodules; this hook keeps ``from xtuner.v1.quantization.linear import ...`` working
# as a flat API. By the time any of these names is requested, the package init (config,
# kernels) has completed, so the deferred submodules can never re-enter a half-run init.
_LAZY_EXPORTS: dict[str, str] = {
    "Int8W8A8DynamicGroupedLinear": ".modules",
    "Int8W8A8DynamicLinear": ".modules",
    "maybe_write_hf_w8a8_linear": ".export",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        import importlib

        value = getattr(importlib.import_module(_LAZY_EXPORTS[name], __name__), name)
        globals()[name] = value  # cache as a real attribute; the hook never fires again
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
