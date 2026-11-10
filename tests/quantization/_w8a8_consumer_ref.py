"""Verbatim excerpts of the vllm / vllm-ascend consumer code for W8A8_DYNAMIC linear weights.

These excerpts exist so ``TestW8A8ConsumerContract`` can exercise the real consumer logic
(the ModelSlim description lookup, the ``W8A8_DYNAMIC`` weight-shape contract and the
routed-experts scale-load dispatch) without importing the vllm/vllm-ascend packages. Every
function body is a byte-identical copy of the upstream source, cited per block. Deviations
are limited to the import/plumbing layer and are documented here:

- ``AscendW8A8DynamicLinearMethod.apply`` / ``prepare_input_for_overlap`` and
  ``AscendW8A8DynamicFusedMoEMethod.apply`` / ``__init__`` are not copied: they wrap the
  torch_npu runtime ops (``npu_dynamic_quant`` / ``npu_quant_matmul`` / the fused MoE
  kernel) and read the live serving-engine config; neither participates in the on-disk
  weight contract under test. Tests construct schemes directly.
- The ``@register_scheme`` decorators are registry plumbing, not contract logic.
- ``WeightSwitchGatherSpec`` class attributes, ``supports_weight_switch`` and the MoE
  ``quant_type: QuantType`` attribute are dropped (they import vllm types); the
  weight-switch / quant-type metadata is not part of the checkpoint contract.
- The scheme base classes become ``_AscendLinearSchemeHarness`` / ``_AscendMoESchemeHarness``
  stand-ins: upstream they only carry the registry plumbing plus abstract ``apply`` hooks.
  ``validate_weight_shapes`` (a shape cross-check against the live MoE config) becomes a
  no-op stub so the ``create_weights`` call site stays verbatim; the MoE harness
  deliberately has no ``group_size`` attribute because W8A8_DYNAMIC has none — the
  ``hasattr(self.quant_method, "group_size")`` branch in ``create_weights`` must stay on
  the "no group_size" leg, which is what keeps the scales on the CHANNEL loader.
- ``maybe_trans_nz`` becomes the identity: the ND->NZ fractal re-layout is serving-time
  device-memory plumbing with no on-disk meaning; the ``transpose(0, 1)`` before it IS the
  contract and stays verbatim.
- ``AscendModelSlimLookupConfig`` mirrors the description-lookup half of upstream
  ``AscendModelSlimConfig``: its ``__init__`` takes the description dict directly, and its
  ``get_quant_type`` mirrors the lookup core of upstream ``get_quant_method`` minus the
  layer-type dispatch (returns the quant-type string instead of a QuantizeMethodBase) and
  minus the kimi_k3 mixed-KDA branch (never applies to glm5_next). The older
  KV-flavored excerpt of the same upstream class is pinned separately in
  ``_modelslim_consumer_ref.py`` (@9300a2f9); the two excerpts share no code.
- ``AscendMoEMethodHarness.__init__`` mirrors upstream minus the ``super().__init__``
  (``FusedMoEMethodBase`` only stores the config; the harness stores ``moe_config`` itself).
- ``RoutedExpertsScaleLoaderHarness.load_scale`` copies the ``weight_loader`` lines that
  decide the scale-loading contract (``base_quant_method``, ``is_transposed``, the
  ``shard_id`` / ``shard_dim`` / ``full_load`` / ``expert_data`` head and the
  quant_method-keyed dispatch). The gpt-oss, online-MoE re-quantization, expert-id mapping
  and shared-expert-online-loader branches ahead of them only reshape which expert/rank a
  tensor belongs to and are dropped; the terminal loaders become recording stubs because
  their real bodies (``_load_w13`` / ``_load_w2`` / ``copy_weight_`` narrowing) depend on
  the live MoE config. The w13 half-mapping is asserted structurally in the tests instead.
- ``set_weight_attrs`` drops the TPU-only ``current_platform`` weight-loader sync tail
  (imports vllm at call time; TPU-only, unreachable on NPU hosts).
- ``modelslim_config`` upstream does ``import regex as re``; the single auto-discovery
  pattern copied here is stdlib-``re``-compatible, so the stdlib module is used.
- ``logger`` is a stdlib logging shim (only ``.error`` is used by the copied code).

Sources (pinned):
- vllm-ascend @ 28e9e97ba254b076a88275e8a275556cb9a11664 (2026-11-05):
  quantization/configs/modelslim_config.py (:169-304 mappings, :307-439 helpers,
  :662-784 lookup methods, :812-826 lookup core),
  quantization/methods/w8a8/w8a8_dynamic.py (:61-143 linear scheme, :146-201 MoE scheme),
  quantization/method_adapters.py (:277-279 adapter init, :290-323 create_weights).
- vllm @ dfebb42d096d5c6a04aa5a1722195e37cb8e0b2d (2026-11-05):
  model_executor/layers/fused_moe/routed_experts.py (:65-69 enum, :704-712 signature,
  :729, :787-823 head, :945-999 scale dispatch),
  model_executor/utils.py (:39-71 set_weight_attrs).
"""

import logging
import re
from copy import deepcopy
from enum import Enum
from typing import Any, Callable

import torch

from ._modelslim_consumer_ref import WeightsMapper


logger = logging.Logger("w8a8_consumer_ref")


# --- vllm-ascend modelslim_config.py:165-304 (verbatim) ---------------------------------------


# key: model_type
# value: dict of fused module name -> list of original module names
# Note: Currently, only models that do not have the `packed_modules_mapping` attribute
# in the vLLM upstream need to be added here.
UPDATED_PACKED_MODULES_MAPPING: dict[str, dict[str, list[str]]] = {
    # GLM-5.3-Flash (glm5_next): KDA layers ship a fused q/k/v/b/f/a/g_a
    # projection; sparse-MLA layers keep the DeepSeek-style q_a/kv_a pair.
    # Native HF FP8 checkpoints leave the KDA projections in bf16 via
    # modules_to_not_convert; this mapping is for ModelSlim w8a8 later.
    "glm5_next": {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "experts": ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
        "fused_qkvbfg_a_proj": ["q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj"],
    },
    "glm5_next_mtp": {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "experts": ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
        "fused_qkvbfg_a_proj": ["q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj"],
    },
    "deepseek_mtp": {
        "gate_up_proj": ["gate_proj", "up_proj"],
    },
    "pangu_ultra_moe_mtp": {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    },
    "qwen3_vl_moe": {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    },
    "longcat_flash": {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    },
    "bailing_hybrid": {
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
        "o_proj": ["dense"],
    },
    # The step3.5 MTP draft (speculative.py sets model_type="step3p5_mtp")
    # reuses the same fused module layout as the verifier.
    "step3p5_mtp": {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
        "experts": ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"],
    },
}


QUANT_MODEL_PREFIX_MAPPINGS = {
    "deepseek_v4": {
        # Multimodal checkpoints keep the text-backbone quantization keys in
        # the original causal-LM namespace (``model.*`` / ``lm_head.*``),
        # while the vLLM wrapper nests the module under ``language_model``.
        "language_model.model.": "model.",
        "language_model.lm_head.": "lm_head.",
        # ``WeightsMapper`` treats the trailing dot literally, while the
        # quantization lookup also queries the module prefix itself.
        "language_model.lm_head": "lm_head",
        "layers.": "model.layers.",
        "embed.": "model.embed_tokens.",
        "head.": "lm_head.",
    },
    "deepseek_v41": {
        # V4.1 ModelSlim descriptions keep the original checkpoint names,
        # while the runtime uses Ascend module names. Map runtime prefixes
        # back to the checkpoint namespace for quant-scheme lookup.
        "language_model.model.layers.": "layers.",
        "language_model.model.embed_tokens.": "embed.",
        "language_model.model.embed_tokens": "embed",
        "language_model.lm_head.": "head.",
        "language_model.lm_head": "head",
        "model.layers.": "layers.",
        "model.embed_tokens.": "embed.",
        "model.embed_tokens": "embed",
        "lm_head.": "head.",
        "lm_head": "head",
    },
}


QUANT_MODEL_SUBSTR_MAPPINGS = {
    "deepseek_v4": {
        ".attn.": ".self_attn.",
        ".w1.": ".gate_proj.",
        ".w2.": ".down_proj.",
        ".w3.": ".up_proj.",
        ".ffn.": ".mlp.",
        ".ffn_norm.": ".post_attention_layernorm.",
        ".attn_norm.": ".input_layernorm.",
    },
    "deepseek_v41": {
        ".self_attn.": ".attn.",
        ".gate_proj.": ".w1.",
        ".gate_proj": ".w1",
        ".down_proj.": ".w2.",
        ".down_proj": ".w2",
        ".up_proj.": ".w3.",
        ".up_proj": ".w3",
        ".mlp.": ".ffn.",
        ".post_attention_layernorm.": ".ffn_norm.",
        ".post_attention_layernorm": ".ffn_norm",
        ".input_layernorm.": ".attn_norm.",
        ".input_layernorm": ".attn_norm",
    },
    # The step3.5 MTP draft nests its decoder block under ".mtp_block.", but the
    # checkpoint's quant_model_description.json keys it without that infix
    # (e.g. "model.layers.45.self_attn.q_proj.weight"). Strip it so the quant
    # lookup matches the on-disk naming.
    "step3p5_mtp": {
        ".mtp_block.": ".",
    },
    # GLM-5 MTP nests the decoder under ``mtp_block`` while the ModelSlim
    # description retains checkpoint-style names.
    "glm5_next_mtp": {
        ".mtp_block.": ".",
    },
    # Gemma4 MoE renames ".experts." to ".moe.experts." in the vLLM module tree
    # (gemma4.py weight mapper), but the ModelSlim quant description keeps the
    # original ".experts." naming. Strip the ".moe" infix so the quant lookup
    # matches the on-disk keys.
    "gemma4": {
        ".moe.experts": ".experts",
    },
    "gemma4_text": {
        ".moe.experts": ".experts",
    },
}


# --- vllm-ascend modelslim_config.py:307-439 (verbatim) ---------------------------------------


def _is_missing_v_shard(shard_key: str, quant_description: dict[str, Any]) -> bool:
    """Return whether the missing shard is Gemma4's replicated v_proj.

    k_eq_v layers do not have a dedicated v_proj entry because v_proj is
    replicated from k_proj at load time. Keep q_proj/k_proj mandatory so other
    malformed packed quant descriptions still fail as before.
    """
    if not shard_key.endswith(".v_proj.weight"):
        return False
    shard_prefix = shard_key[: -len("v_proj.weight")]
    return f"{shard_prefix}q_proj.weight" in quant_description and f"{shard_prefix}k_proj.weight" in quant_description


def _find_quant_description_key(quant_description: dict[str, Any], prefix: str) -> str | None:
    """Return the weight or bare ModelSlim entry for ``prefix``."""
    weight_key = prefix + ".weight"
    if weight_key in quant_description:
        return weight_key
    if prefix in quant_description:
        return prefix
    return None


def _resolve_fused_packed_shard_keys(
    quant_description: dict[str, Any],
    prefix: str,
    shard_prefixes: list[str],
) -> list[str] | None:
    """Resolve descriptions that keep a runtime packed module fused.

    ModelSlim may describe a packed runtime module directly. Routed experts
    may also keep gate/up fused while vLLM exposes split logical shards. Only
    use these entries when none of the normal expanded shard entries exists;
    callers reject partially mixed descriptions.
    """
    direct_key = _find_quant_description_key(quant_description, prefix)
    if direct_key is not None:
        return [direct_key]

    if prefix.rpartition(".")[-1] != "experts":
        return None

    expected_suffixes = {
        shard_prefix.removeprefix(prefix + ".").partition(".")[-1]
        for shard_prefix in shard_prefixes
        if shard_prefix.startswith(prefix + ".")
    }
    if not {"gate_proj", "up_proj", "down_proj"}.issubset(expected_suffixes):
        return None

    gate_up_key = _find_quant_description_key(quant_description, f"{prefix}.gate_up_proj")
    down_key = _find_quant_description_key(quant_description, f"{prefix}.down_proj")
    if gate_up_key is None or down_key is None:
        return None
    return [gate_up_key, down_key]


def get_quant_type_for_layer(
    quant_description: dict[str, Any],
    prefix: str,
    packed_modules_mapping: dict[str, Any] | None = None,
    prefix_mapper: Callable[[str], str] | None = None,
) -> str | None:
    """Determine the quantization type for a layer.

    Args:
        quant_description: The quantization description dictionary.
        prefix: The layer prefix.
        packed_modules_mapping: Mapping for packed/fused modules.
        prefix_mapper: Map expanded module names to quantization description keys.

    Returns:
        The quantization type string (e.g., "W8A8_DYNAMIC").
        Note: If the quantization type is "FLOAT", it will return None to indicate no quantization.
    """
    if packed_modules_mapping is None:
        packed_modules_mapping = dict()

    proj_name = prefix.split(".")[-1]
    if proj_name in packed_modules_mapping:
        quant_type = None
        # Replace only the trailing component; str.replace would also rewrite
        # any earlier occurrence of proj_name inside the prefix.
        shard_prefixes = [
            prefix.removesuffix(proj_name) + shard_proj_name for shard_proj_name in packed_modules_mapping[proj_name]
        ]
        if prefix_mapper is not None:
            prefix = prefix_mapper(prefix)
            shard_prefixes = [prefix_mapper(shard_prefix) for shard_prefix in shard_prefixes]
        shard_keys = [_find_quant_description_key(quant_description, shard_prefix) for shard_prefix in shard_prefixes]
        missing_v_shards = [
            key is None and _is_missing_v_shard(shard_prefix + ".weight", quant_description)
            for key, shard_prefix in zip(shard_keys, shard_prefixes)
        ]
        normal_keys = [key for key in shard_keys if key is not None]
        missing_normal_shards = [
            shard_prefix
            for key, shard_prefix, missing_v in zip(shard_keys, shard_prefixes, missing_v_shards)
            if key is None and not missing_v
        ]

        if missing_normal_shards:
            # Never combine a partial normal description with fused aliases;
            # that would hide malformed or mixed quantization metadata.
            if normal_keys:
                raise KeyError(missing_normal_shards[0] + ".weight")
            fused_keys = _resolve_fused_packed_shard_keys(quant_description, prefix, shard_prefixes)
            if fused_keys is None:
                raise KeyError(missing_normal_shards[0] + ".weight")
            shard_keys_to_check = fused_keys
        else:
            # Only Gemma4 k_eq_v is allowed to omit v_proj.
            shard_keys_to_check = normal_keys

        for shard_key in shard_keys_to_check:
            shard_quant_type = quant_description[shard_key]

            if quant_type is None:
                quant_type = shard_quant_type
            elif shard_quant_type != quant_type:
                err_msg = (
                    f"Not all shards of {prefix} are quantized with same quant type. "
                    f"Shard {proj_name} uses {shard_quant_type}, but another shard "
                    f"uses {quant_type}. Please check quantization config."
                )
                logger.error(err_msg)
                raise ValueError(err_msg)
    else:
        if prefix_mapper is not None:
            prefix = prefix_mapper(prefix)
        quant_key = _find_quant_description_key(quant_description, prefix)
        quant_type = quant_description.get(quant_key) if quant_key is not None else None
    return quant_type if quant_type != "FLOAT" else None


class AscendModelSlimLookupConfig:
    """Description-lookup harness for vllm-ascend ``AscendModelSlimConfig`` @28e9e97b.

    The upstream class is the ModelSlim quant-config entry point; only its description
    bookkeeping and the W8A8-relevant lookup methods are mirrored here (see the module
    docstring for what ``get_quant_type`` omits relative to upstream ``get_quant_method``).

    Args:
        quant_description (dict[str, Any]): The parsed ``quant_model_description.json``.
        packed_modules_mapping (dict[str, Any] | None): vLLM-side packed-module mapping
            seeds; the glm5_next deltas from ``UPDATED_PACKED_MODULES_MAPPING`` are merged
            in by ``_update_packed_modules_mapping``.
    """

    def __init__(
        self, quant_description: dict[str, Any], packed_modules_mapping: dict[str, Any] | None = None
    ) -> None:
        self.quant_description = quant_description
        self.packed_modules_mapping: dict[str, Any] = dict(packed_modules_mapping or {})
        self.model_type: str | None = None

    # vllm-ascend modelslim_config.py:662-675 (verbatim)
    def _has_quant_weight(self, prefix: str) -> bool:
        if _find_quant_description_key(self.quant_description, prefix) is not None:
            return True
        proj_name = prefix.split(".")[-1]
        if proj_name in self.packed_modules_mapping:
            return all(
                _find_quant_description_key(
                    self.quant_description,
                    prefix.removesuffix(proj_name) + shard_proj_name,
                )
                is not None
                for shard_proj_name in self.packed_modules_mapping[proj_name]
            )
        return False

    # vllm-ascend modelslim_config.py:677-738 (verbatim)
    def quant_prefix_mapper(self, model_type: str, prefix: str) -> str:
        self.model_type = model_type
        # Some model paths, e.g. qwen3-vl and qwen3_5_moe MTP drafter,
        # initialize lm_head with prefix="lm_head", while the quant description
        # key is mapped to "language_model.lm_head.weight".
        if (
            prefix == "lm_head"
            and "lm_head.weight" not in self.quant_description
            and "language_model.lm_head.weight" in self.quant_description
        ):
            prefix = "language_model.lm_head"
        prefix_mapping = QUANT_MODEL_PREFIX_MAPPINGS.get(model_type)
        substr_mapping = QUANT_MODEL_SUBSTR_MAPPINGS.get(model_type)
        if prefix_mapping or substr_mapping:
            hf_to_vllm_mapper = WeightsMapper(
                orig_to_new_prefix=prefix_mapping or {},
                orig_to_new_substr=substr_mapping or {},
            )
            prefix = hf_to_vllm_mapper._map_name(prefix)

        if model_type in ("glm5_next", "glm5_next_text"):
            candidate = None
            if prefix.startswith("language_model.model."):
                candidate = prefix.replace(
                    "language_model.model.",
                    "model.language_model.",
                    1,
                )
            elif prefix.startswith("language_model.lm_head"):
                candidate = prefix.replace("language_model.lm_head", "lm_head", 1)
            elif prefix.startswith("model.layers."):
                candidate = prefix.replace(
                    "model.layers.",
                    "language_model.model.layers.",
                    1,
                )
            if candidate and not self._has_quant_weight(prefix):
                if self._has_quant_weight(candidate):
                    return candidate

        if model_type == "glm5_next_mtp" and prefix.startswith("model.layers."):
            candidate = prefix.replace(
                "model.layers.",
                "model.language_model.layers.",
                1,
            )
            if not self._has_quant_weight(prefix) and self._has_quant_weight(candidate):
                return candidate

        if model_type == "step3p5_mtp" and prefix.startswith("model.layers."):
            # Step3P5 MTP and newly generated Step3P7 W8A8 MTP checkpoints use
            # ``model.layers.*``.  The Step3P7 vLLM wrapper mapper rewrites
            # current ``model.layers.*`` quant descriptions to
            # ``language_model.model.layers.*``.  The MTP draft module itself
            # is still Step3P5-shaped and queries ``model.layers.*``, so try
            # the Step3P7 wrapper alias only when the direct Step3P5/new-key
            # lookup misses.
            if not self._has_quant_weight(prefix):
                for candidate in (prefix.replace("model.layers.", "language_model.model.layers.", 1),):
                    if self._has_quant_weight(candidate):
                        return candidate
        return prefix

    # vllm-ascend modelslim_config.py:740-784 (verbatim)
    def _update_packed_modules_mapping(self, model_type: str) -> None:
        """Update packed_modules_mapping for the given model type.

        Upstream vLLM seeds packed_modules_mapping from the model's own
        definition (passed by reference), so only the deltas are applied here:
        - extra fused modules listed in UPDATED_PACKED_MODULES_MAPPING,
        - the split KDA projections of Kimi K3,
        - expert shard names auto-discovered from quant_description, so that
          the consistency verification in get_quant_type_for_layer applies to
          all expert shards (gate/up/down or w1/w2/w3).
        """
        # Mutating the mapping we were handed by reference would leak these
        # deltas into the model class's own mapping, which vLLM's LoRA manager
        # reads to pick its packing branch. Work on a private copy instead.
        self.packed_modules_mapping = deepcopy(self.packed_modules_mapping)
        # Only update packed_modules_mapping if the upstream model definition not satisfies our scenario.
        mapping_model_type = "glm5_next" if model_type == "glm5_next_text" else model_type
        if mapping_model_type in UPDATED_PACKED_MODULES_MAPPING:
            self.packed_modules_mapping.update(UPDATED_PACKED_MODULES_MAPPING[mapping_model_type])
        if model_type in ("kimi_k3", "kimi_linear"):
            from vllm.models.kimi_k3.nvidia.model import KimiLinearModel

            mapping = {
                packed_name: list(shard_names)
                for packed_name, shard_names in KimiLinearModel.packed_modules_mapping.items()
            }
            qkv_shards = [name for name in mapping["in_proj_qkvgfab"] if name in ("q_proj", "k_proj", "v_proj")]
            gate_shards = ["g_proj", "f_a_proj", "b_proj"]
            mapping.update(
                {
                    "experts": ["experts.0.w1", "experts.0.w2", "experts.0.w3"],
                    "in_proj_qkvgfab": qkv_shards + gate_shards,
                    "in_proj_qkv": qkv_shards,
                    "in_proj_gfab": gate_shards,
                }
            )
            self.packed_modules_mapping.update(mapping)
        if "experts" not in self.packed_modules_mapping:
            shard_names: set[str] = set()
            for key in self.quant_description:
                m = re.search(r"\.experts\.\d+\.(\w+)\.weight$", key)
                if m:
                    shard_names.add(m.group(1))
            if shard_names:
                self.packed_modules_mapping["experts"] = [f"experts.0.{name}" for name in sorted(shard_names)]

    # vllm-ascend modelslim_config.py:812-826 lookup core (adapted; see class docstring)
    def get_quant_type(self, model_type: str, prefix: str) -> str | None:
        """Return the ModelSlim quant type for a runtime module prefix.

        Args:
            model_type (str): The serving model type (e.g. "glm5_next").
            prefix (str): The runtime module prefix (e.g.
                "model.language_model.layers.0.mlp.experts").

        Returns:
            str | None: The quant type string, or ``None`` for unquantized / "FLOAT".
        """
        self._update_packed_modules_mapping(model_type)
        runtime_prefix = prefix
        prefix = self.quant_prefix_mapper(model_type, prefix)
        quant_type = get_quant_type_for_layer(
            self.quant_description,
            runtime_prefix,
            self.packed_modules_mapping,
            prefix_mapper=lambda name: self.quant_prefix_mapper(model_type, name),
        )
        return quant_type


# --- vllm-ascend methods/w8a8/w8a8_dynamic.py:61-143 (adapted; see module docstring) -----------


def maybe_trans_nz(tensor: torch.Tensor) -> torch.Tensor:
    """Identity stand-in for vllm-ascend ``maybe_trans_nz`` (see module docstring)."""
    return tensor


class _AscendLinearSchemeHarness:
    """Stand-in for vllm-ascend ``AscendLinearScheme`` (see module docstring)."""


class AscendW8A8DynamicLinearMethod(_AscendLinearSchemeHarness):
    """Linear method for Ascend W8A8_DYNAMIC.

    This scheme uses dynamic per-token quantization for activations
    and per-channel quantization for weights.
    """

    act_quant_type: torch.dtype = torch.int8

    def __init__(self):
        pass

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        params_dict = {"weight": torch.empty(output_size, input_size, dtype=torch.int8)}
        return params_dict

    def get_perchannel_param(
        self,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(output_size, 1, dtype=params_dtype)
        params_dict["weight_offset"] = torch.empty(output_size, 1, dtype=params_dtype)
        return params_dict

    def process_weights_after_loading(self, layer):
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        # cast quantized weight tensors in NZ format for higher inference speed
        if self.act_quant_type == torch.int8:
            layer.weight.data = maybe_trans_nz(layer.weight.data)
        layer.weight_scale.data = layer.weight_scale.data.flatten()
        layer.weight_scale_fp32 = layer.weight_scale.data.to(torch.float32)
        layer.weight_offset.data = layer.weight_offset.data.flatten()


# --- vllm-ascend methods/w8a8/w8a8_dynamic.py:146-201 (adapted; see module docstring) ----------


class _AscendMoESchemeHarness:
    """Stand-in for vllm-ascend ``AscendMoEScheme`` (see module docstring)."""

    def validate_weight_shapes(self, layer: torch.nn.Module, intermediate_size_per_partition: int) -> None:
        pass


class AscendW8A8DynamicFusedMoEMethod(_AscendMoESchemeHarness):
    """FusedMoE method for Ascend W8A8_DYNAMIC."""

    supports_eplb = True
    act_quant_type: torch.dtype = torch.int8
    fused_activations = frozenset({"silu", "swigluoai_uninterleave"})

    def get_weight(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, hidden_sizes, dtype=torch.int8
        )
        param_dict["w2_weight"] = torch.empty(
            num_experts, hidden_sizes, intermediate_size_per_partition, dtype=torch.int8
        )
        return param_dict

    def get_dynamic_quant_param(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight_scale"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, 1, dtype=params_dtype
        )
        param_dict["w13_weight_offset"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, 1, dtype=params_dtype
        )
        param_dict["w2_weight_scale"] = torch.empty(num_experts, hidden_sizes, 1, dtype=params_dtype)
        param_dict["w2_weight_offset"] = torch.empty(num_experts, hidden_sizes, 1, dtype=params_dtype)
        return param_dict


# --- vllm routed_experts.py:65-69 (verbatim) + vllm utils.py:39-71 (adapted) -------------------


class FusedMoeWeightScaleSupported(Enum):
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"


def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: dict[str, Any] | None,
):
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.

    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(weight, key), f"Overwriting existing tensor attribute: {key}"

        # NOTE(woosuk): During weight loading, we often do something like:
        # narrowed_tensor = param.data.narrow(0, offset, len)
        # narrowed_tensor.copy_(real_weight)
        # expecting narrowed_tensor and param.data to share the same storage.
        # However, on TPUs, narrowed_tensor will lazily propagate to the base
        # tensor, which is param.data, leading to the redundant memory usage.
        # This sometimes causes OOM errors during model loading. To avoid this,
        # we sync the param tensor after its weight loader is called.
        # TODO(woosuk): Remove this hack once we have a better solution.
        # (Adaptation: the TPU-only current_platform weight-loader sync that follows in
        # upstream is dropped; it imports vllm at call time.)
        setattr(weight, key, value)


# --- vllm-ascend method_adapters.py:277-279,290-323 (adapted; see module docstring) ------------


class AscendMoEMethodHarness:
    """Harness mirroring vllm-ascend ``AscendFusedMoEMethod``'s weight creation.

    Args:
        scheme (AscendW8A8DynamicFusedMoEMethod): The MoE quantization scheme.
        moe_config (Any): Stand-in for ``FusedMoEConfig`` (unused by ``create_weights``).
    """

    def __init__(self, scheme: AscendW8A8DynamicFusedMoEMethod, moe_config: Any) -> None:
        # (Adaptation: upstream calls super().__init__(moe_config), which only stores it.)
        self.moe_config = moe_config
        self.quant_method = scheme

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        self.quant_method.validate_weight_shapes(layer, intermediate_size_per_partition)

        weight_param = self.quant_method.get_weight(
            num_experts, intermediate_size_per_partition, hidden_size, params_dtype
        )
        for param_key, param_value in weight_param.items():
            param = torch.nn.Parameter(param_value, requires_grad=False)
            layer.register_parameter(param_key, param)
            set_weight_attrs(param, extra_weight_attrs)

        extra_weight_attrs.update({"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value})
        per_group_param = ["weight_scale_second", "weight_offset_second", "scale_bias"] + (
            ["weight_scale", "weight_offset"]
            if hasattr(self.quant_method, "group_size") and self.quant_method.group_size > 0
            else []
        )
        dynamic_quant_param = self.quant_method.get_dynamic_quant_param(
            num_experts, intermediate_size_per_partition, hidden_size, params_dtype
        )
        for param_key, param_value in dynamic_quant_param.items():
            param = torch.nn.Parameter(param_value, requires_grad=False)
            layer.register_parameter(param_key, param)
            set_weight_attrs(param, extra_weight_attrs)
            if any(fields in param_key for fields in per_group_param):
                param.quant_method = FusedMoeWeightScaleSupported.GROUP.value


# --- vllm routed_experts.py:704-712,729,787-823,945-999 (adapted; see module docstring) --------


class RoutedExpertsScaleLoaderHarness:
    """Recording harness for vllm ``RoutedExperts.weight_loader``'s scale path.

    ``load_scale`` reproduces verbatim every line that decides the scale-loading contract
    (the ``quant_method`` attribute dispatch and the ``shard_id`` / ``shard_dim`` /
    ``expert_data`` slicing); the terminal loaders record their call instead of copying
    data (see the module docstring).

    Args:
        moe_config (Any): Stand-in for ``FusedMoEConfig``; only ``tp_rank`` is read.
        quant_method (Any | None): Stand-in for the MoE quant method whose class
            attributes (e.g. ``weight_scale_refine``) the dispatch consults.
    """

    def __init__(self, moe_config: Any, quant_method: Any = None) -> None:
        self.moe_config = moe_config
        self.quant_method = quant_method
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _load_per_channel_weight_scale(self, **kwargs: Any) -> None:
        self.calls.append(("per_channel", kwargs))

    def _load_model_weight_or_group_weight_scale(self, **kwargs: Any) -> None:
        self.calls.append(("group", kwargs))

    def _load_per_tensor_weight_scale(self, **kwargs: Any) -> None:
        self.calls.append(("per_tensor", kwargs))

    def load_scale(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        base_quant_method = self.quant_method

        # is_transposed: if the dim to shard the weight
        # should be flipped. Required by GPTQ/AWQ (K-first format).
        # should be whatever dimension intermediate_size_per_partition is
        is_transposed = getattr(param, "is_transposed", False)

        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"shard_id must be ['w1','w2','w3'] but got {shard_id}.")

        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size_per_partition is used.
        SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}
        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
        if is_transposed:
            shard_dim = int(not shard_dim)

        full_load = len(loaded_weight.shape) == 3
        if full_load:
            shard_dim += 1

        expert_data = param.data if full_load else param.data[expert_id]

        # Case weight scales, zero_points and offset, weight/input global scales
        if "scale" in weight_name or "zero" in weight_name or "offset" in weight_name:
            # load the weight scales and zp based on the quantization scheme
            # supported weight scales/zp can be found in
            # FusedMoeWeightScaleSupported
            # TODO @dsikka: once hardened, refactor to use vLLM Parameters
            # specific to each case
            quant_method = getattr(param, "quant_method", None)
            if quant_method == FusedMoeWeightScaleSupported.CHANNEL.value:
                self._load_per_channel_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.moe_config.tp_rank,
                )
            elif quant_method in [
                FusedMoeWeightScaleSupported.GROUP.value,
                FusedMoeWeightScaleSupported.BLOCK.value,
            ]:
                scale_refine = getattr(base_quant_method, "weight_scale_refine", None)
                if quant_method == FusedMoeWeightScaleSupported.BLOCK.value and scale_refine is not None:
                    # FP8 block scales are stored per (block_n, block_k) tile
                    # of the unsharded weight, while the TP-sharded parameters
                    # use a refined block grid (see Fp8MoEMethod). Upsample the
                    # scales to the refined grid (lossless: the refined block
                    # divides the checkpoint block) so per-rank slicing stays
                    # exact. Dim -2 is the weight's N dim, dim -1 is K.
                    loaded_weight = loaded_weight.repeat_interleave(scale_refine[0], dim=-2).repeat_interleave(
                        scale_refine[1], dim=-1
                    )
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.moe_config.tp_rank,
                    is_scale=True,
                )
            elif quant_method == FusedMoeWeightScaleSupported.TENSOR.value:
                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
            else:
                WEIGHT_SCALE_SUPPORTED = [e.value for e in FusedMoeWeightScaleSupported]
                raise ValueError(f"quant method must be one of {WEIGHT_SCALE_SUPPORTED}")
            return True if return_success else None

        raise AssertionError(f"load_scale harness reached an unhandled weight_name: {weight_name}")
