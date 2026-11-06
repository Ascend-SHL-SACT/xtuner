"""Verbatim excerpts of the vllm-ascend ModelSlim/FAKQuant consumer code.

These excerpts exist so ``TestPerTensorModelSlimConsumer`` can exercise the real
consumer logic without importing the vllm/vllm-ascend packages (whose import graph
pulls ~20 third-party wheels unavailable on offline hosts). Every function body is a
byte-identical copy of the upstream source, cited per block. Deviations are limited to
the import/plumbing layer and are documented inline:

- ``logger`` is a stdlib logging shim providing vllm's ``warning_once``/``info_once``
  (identical behavior minus the once-dedup).
- ``get_tensor_model_parallel_rank``/``get_tensor_model_parallel_world_size`` are
  single-rank stand-ins for ``vllm.distributed`` (rank 0 / size 1); the scalar-fill
  branch under test never reaches TP sharding.
- The hardware-profile wrapper pins the A3 capability set (this host is the 910C
  family); the ``supports()`` semantics and the capability sets themselves are
  verbatim. Detection (SoC probing + ``AscendDeviceType``) is not copied.
- ``AscendFAQuantAttentionMethod.__init__`` reads the serving engine's vllm_config and
  is not copied; the tests bypass construction via ``__new__``. The ``@register_scheme``
  decorator is registry plumbing, not contract logic.
- ``get_model_file``'s remote-repo branch references vllm's ``envs`` (unbound here);
  it is unreachable for local model dirs and its except-all degrades to ``None``.

Sources (pinned):
- vllm-ascend @ 9300a2f9bfee2fec19c0f7fa2634490d77cde9de (2026-11-04):
  quantization/utils.py, quantization/configs/modelslim_config.py,
  quantization/methods/kv_cache/kv_c8.py, device/hardware_profile.py, utils.py.
- vllm @ 790be385c8a69f45ec9a7a85ef00836d1684060f (2026-11-04):
  model_executor/models/utils.py (WeightsMapper),
  model_executor/layers/quantization/base_config.py (get_cache_scale_mapper),
  config/vllm.py (get_current_vllm_config_or_none).
"""

import glob
import json
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, TypeAlias

import torch


logger = logging.Logger("modelslim_consumer_ref")
logger.warning_once = logger.warning  # type: ignore[method-assign]
logger.info_once = logger.info  # type: ignore[method-assign]

# vllm/config/vllm.py:4102 (verbatim; the engine assembles/sets the config in serving).
_current_vllm_config: Any = None


def get_current_vllm_config_or_none() -> Any:
    return _current_vllm_config


# vllm-ascend utils.py:56 / vllm-ascend utils.py:1404 / modelslim_config.py:163,66-68,72-77 (verbatim).
ASCEND_QUANTIZATION_METHOD = "ascend"
C8_MXFP_KV_CACHE_DTYPE = "mxfp8"
MODELSLIM_CONFIG_FILENAME = "quant_model_description.json"
_MXFP_C8_KV_CACHE_TYPE = "K_DYNAMIC_V_STATIC_MXFP8_PER_CHANNEL"
_MXFP_C8_LAYER_QUANT_TYPE_SUFFIX = ".self_attn.quant_type"
_MXFP_C8_LAYER_QUANT_TYPES = ("QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL",)
_MXFP_C8_UNUSED_FA_SUFFIXES = (
    ".fa_q.scale",
    ".fa_k.scale",
    ".fa_q.offset",
    ".fa_k.offset",
)


# Single-rank stand-ins for vllm.distributed (see module docstring).
def get_tensor_model_parallel_rank() -> int:
    return 0


def get_tensor_model_parallel_world_size() -> int:
    return 1


# --- vllm-ascend device/hardware_profile.py:37-171,255-287 (verbatim) -------------------------


class HardwareCapability(Enum):
    """Binary contracts consumed by shared business logic.

    ``profile.supports(X)`` answers only whether the exact contract documented
    for ``X`` is available. Capabilities are intentionally narrow so callers
    do not infer unrelated behavior from a device generation. New default or
    implementation-selection fields should normally be modeled as a policy,
    mode, or family rather than added here.

    A few historical members still describe a default or implementation
    selection. Their comments call this out explicitly; do not interpret them
    as broad claims about the underlying silicon.
    """

    # Legacy default: replace vLLM's custom-op list with ``["all"]`` during
    # platform configuration. Runtime extension loading is a separate contract.
    AUTO_ENABLE_CUSTOM_OPS = auto()
    # Register ATB extension entry points eagerly during worker construction.
    ATB_EXTENSIONS = auto()
    # Run the ATB matmul warmup that avoids first-request cache-write latency.
    ATB_WARMUP = auto()
    # Register fake/meta implementations for the custom BGMV and SGMV LoRA ops.
    BGMV_SGMV_META_REGISTRATION = auto()
    # SK scatter kernel for paged cache writes (A2/A3/A5 ABI).
    SCATTER_ND_CACHE_STORE = auto()
    # Extend SK cache writes to non-overlapping row-strided destinations.
    # A5 is excluded until arch35 handles their storage bounds correctly.
    SCATTER_ND_STRIDED_CACHE_STORE = auto()
    # Extend SK cache writes to FP8 E4M3FN/E5M2 (A5 ABI).
    SCATTER_ND_FP8_CACHE_STORE = auto()
    # Allow the CANN MegaMoe fused-MC2 path when its model, EP, and config checks pass.
    CANN_MEGAMOE = auto()
    # Allow A5 MegaMoe's MXFP-only path and its A5-specific calling conventions.
    CANN_MEGAMOE_MXFP = auto()
    # Split a mixed chunked-prefill batch into separate prefill and decode FIA calls.
    CHUNKED_PREFILL_PHASE_SPLIT = auto()
    # CANN device-side FlashAttn metadata with strided paged GQA K/V.
    DEVICE_LENGTH_FLASH_ATTN = auto()
    # Build CPU-affinity pools from the cluster-aware CPU topology.
    CLUSTER_CPU_TOPOLOGY = auto()
    # Legacy implementation selector: install the 310P-specific operator overrides.
    COMPATIBILITY_OP_IMPLEMENTATIONS = auto()
    # Install distributed-communication adaptations required by the compatibility path.
    DISTRIBUTED_COMMUNICATION_ADAPTATION = auto()
    # Use DSA C128-state kernel block sizes ``[4, 8, 16]`` instead of ``[8, 16, 32]``.
    DSA_C128_STATE_SMALL_BLOCK_SIZES = auto()
    # Use the DeepSeek-V4/DSA compressed-KV-cache layout and compressor/indexer flow.
    DSV4_COMPRESSED_CACHE = auto()
    # Use the DeepSeek-V4.1 mixed-quant cache and packaged QLI/QSMLA ABI.
    DSV41_PACKED_CACHE = auto()
    # Triton RMS/dot/gate/residual fusion for unrotated Engram inputs.
    ENGRAM_UNROTATED_GATE = auto()
    # Decode native FP8 codes and E8M0 group scales in Engram table lookups.
    ENGRAM_MXFP8 = auto()
    # Enable dynamic-MX norm fusion and the associated ``wo_a`` weight-layout contract.
    DYNAMIC_MX_QUANT_FUSION = auto()
    # Select DynamicMxQuantV3 ``scale_alg=1`` for model paths that require it.
    DYNAMIC_MX_QUANT_SCALE_ALG_ONE = auto()
    # Allow GDN prefill to use the external FLA NPU fused operator.
    FLA_GDN_PREFILL = auto()
    # Enable the FP8/C8 attention KV-cache ABI and matching attention preprocess paths.
    # This is not a general statement that every FP8 operation is supported.
    FP8_ATTENTION = auto()
    # CANN PoolKeyIndexer quant_mode=0 with E4M3 query/key and FP32 descales.
    POOL_KEY_INDEXER_FP8 = auto()
    # Select the compatibility grouped-top-k router used by the fused-MoE path.
    FUSED_MOE_COMPATIBILITY = auto()
    # Use npu_rotary_mul with adjacent pairs (rotary_mode="interleave").
    FUSED_ROTARY_MUL_INTERLEAVE = auto()
    # Pass ``glu_alpha`` and ``glu_bias`` to the fused dequant-SwiGLU-quant operator.
    FUSED_SWIGLU_TUNING_ARGS = auto()
    # Select the compatibility GatedDeltaNet core and state-dtype implementation.
    GDN_COMPATIBILITY = auto()
    # Register the FX graph rewrite that fuses a supported matmul/reduce-scatter pattern.
    GRAPH_MM_REDUCE_SCATTER_FUSION = auto()
    # Register the FX graph rewrite that fuses the supported muls-plus-add pattern.
    GRAPH_MULS_ADD_FUSION = auto()
    # Register the FX graph rewrites for supported RMSNorm-plus-quant patterns.
    GRAPH_NORM_QUANT_FUSION = auto()
    # Register the FX graph rewrite that fuses q/k RMSNorm, RoPE and the weight-less
    # v RMSNorm into the SIMD ``split_qkv_rmsnorm_rope_vnorm`` Triton kernel.
    GRAPH_QKV_NORM_ROPE_FUSION = auto()
    # Let inplace_partial_rotary_mul negate sine internally; profiles without
    # this contract negate the sine input explicitly and pass negate_sin=False.
    INPLACE_PARTIAL_ROTARY_MUL_NEGATE_SIN = auto()
    # Reserve host CPUs for interrupt handling when constructing worker CPU pools.
    IRQ_CPU_RESERVATION = auto()
    # Create the local communication resource used by supported KV-transfer deployments.
    LOCAL_KV_COMM_RESOURCE = auto()
    # Use vLLM-Ascend's custom BGMV/SGMV LoRA kernels when rank constraints also pass.
    LORA_CUSTOM_OPS = auto()
    # Use the token-fused MLA cache protocol for eligible FlashMLA models.
    MLA_FLASH = auto()
    # Allow MLAPO with native floating-point projection weights, not only quantized weights.
    MLAPO_NATIVE_WEIGHTS = auto()
    # Accept ``fullmesh_v2`` as the MC2 communication algorithm.
    MC2_FULLMESH_V2_COMM = auto()
    # Accept hierarchical MC2 communication, subject to its expert-count constraints.
    MC2_HIERARCHY_COMM = auto()
    # Pass TP group/rank metadata required by the newer MoE dispatch/combine ABI.
    MOE_DISPATCH_EXTRA_ARGS = auto()
    # Pass shared-expert, expert-scale, quant-mode, and output-dtype metadata to MoE dispatch.
    MOE_DISPATCH_SHARED_EXPERT_ARGS = auto()
    # Route DeepSeek-V4 vision and hash rows through the fused
    # ``moe_gating_top_k_hash`` ABI with ``bias_vl`` and image sentinels.
    MOE_GATING_TOP_K_HASH_VISION = auto()
    # Allow the extended NPU graph backend; static-kernel mode depends on this contract.
    MM_REDUCE_SCATTER_AI_CPU_INFERENCE = auto()
    NPUGRAPH_EX = auto()
    # Use ``torch_npu.npu_top_k_top_p`` for sampling instead of pytorch fallback.
    NPU_TOP_K_TOP_P = auto()
    # Allow shared paged-attention decode when graph mode, shape, and model checks pass.
    PAGED_ATTENTION = auto()
    # Inspect PCIe topology to distinguish 310P Root-Complex and endpoint deployments.
    RC_DEVICE_DISCOVERY = auto()
    # Fused RMSNorm+cast is available for the A2/A3 (CANN 9.1) path.
    RMS_NORM_CAST = auto()
    # Import and register the compiled vLLM-Ascend custom-op library at runtime.
    # This is independent of whether custom ops are enabled by default.
    RUNTIME_CUSTOM_OPS = auto()
    # Legacy implementation selector: install the standard worker patch bundle.
    STANDARD_WORKER_PATCHES = auto()
    # Legacy implementation selector: install the standard Mamba platform patch.
    STANDARD_MAMBA_PATCH = auto()
    # Reserved contract for the A5 SwiGLU OAI MX-quant path; it currently has no consumer.
    SWIGLU_OAI_MX_QUANT = auto()
    # Use the Triton batch-memcpy kernel for Mamba state copies.
    TRITON_BATCH_MEMCPY = auto()
    # Native TurboQuant 4-bit non-causal MLA cache and SFA kernels.
    TURBOQUANT_4BIT_NC_CACHE = auto()
    # Honor MLAPO enablement on any pipeline role; other profiles limit it to decode consumers.
    UNRESTRICTED_MLAPO = auto()


_STANDARD_CAPABILITIES = frozenset(
    {
        HardwareCapability.AUTO_ENABLE_CUSTOM_OPS,
        HardwareCapability.ATB_EXTENSIONS,
        HardwareCapability.ATB_WARMUP,
        HardwareCapability.BGMV_SGMV_META_REGISTRATION,
        HardwareCapability.FLA_GDN_PREFILL,
        HardwareCapability.FUSED_ROTARY_MUL_INTERLEAVE,
        HardwareCapability.FUSED_SWIGLU_TUNING_ARGS,
        HardwareCapability.GRAPH_MM_REDUCE_SCATTER_FUSION,
        HardwareCapability.GRAPH_MULS_ADD_FUSION,
        HardwareCapability.GRAPH_NORM_QUANT_FUSION,
        HardwareCapability.GRAPH_QKV_NORM_ROPE_FUSION,
        HardwareCapability.INPLACE_PARTIAL_ROTARY_MUL_NEGATE_SIN,
        HardwareCapability.IRQ_CPU_RESERVATION,
        HardwareCapability.LORA_CUSTOM_OPS,
        HardwareCapability.MC2_HIERARCHY_COMM,
        HardwareCapability.MOE_GATING_TOP_K_HASH_VISION,
        HardwareCapability.NPUGRAPH_EX,
        HardwareCapability.PAGED_ATTENTION,
        HardwareCapability.RUNTIME_CUSTOM_OPS,
        HardwareCapability.SCATTER_ND_CACHE_STORE,
        HardwareCapability.SCATTER_ND_STRIDED_CACHE_STORE,
        HardwareCapability.STANDARD_MAMBA_PATCH,
        HardwareCapability.STANDARD_WORKER_PATCHES,
        HardwareCapability.TRITON_BATCH_MEMCPY,
        HardwareCapability.TURBOQUANT_4BIT_NC_CACHE,
    }
)
_A3_CAPABILITIES = _STANDARD_CAPABILITIES | {
    HardwareCapability.MC2_FULLMESH_V2_COMM,
    HardwareCapability.RMS_NORM_CAST,
}


@dataclass(frozen=True, slots=True)
class _ConsumerHardwareProfile:
    """Trimmed HardwareProfile carrying only the ``supports`` contract surface.

    The full dataclass (hardware_profile.py:223-253) adds family/policy fields that
    need the family enums; they are not queried on this path. ``supports`` below is
    verbatim. This host (Atlas 910C) pins the A3 capability set, which does NOT
    contain FP8_ATTENTION — the same branch production takes on 910C.
    """

    capabilities: frozenset[HardwareCapability]

    def supports(self, capability: HardwareCapability) -> bool:
        """Return whether the exact documented ``capability`` contract is available."""

        return capability in self.capabilities


_CURRENT_HARDWARE_PROFILE = _ConsumerHardwareProfile(capabilities=_A3_CAPABILITIES)


def get_current_hardware_profile() -> _ConsumerHardwareProfile:
    """Return the profile selected by the current device configuration."""

    return _CURRENT_HARDWARE_PROFILE


# --- vllm model_executor/models/utils.py:30-49,57-175 (verbatim; apply() not copied) ----------

ShardId: TypeAlias = str | int | tuple[int, ...]
"""One shard of a stacked parameter. A tuple is a single contiguous span."""
ShardIds: TypeAlias = ShardId | list[ShardId]
"""One shard, or a list of shards the same weight is loaded into in turn."""

WeightsTransformFn: TypeAlias = Callable[[Iterable[tuple[str, torch.Tensor]]], Iterable[tuple[str, torch.Tensor]]]
"""A lazy transform over a raw checkpoint ``(name, tensor)`` stream, passing
unrecognized tensors through unchanged."""


@dataclass
class WeightsMapper:
    """Maps the name of each weight if they match the following patterns.

    If a key maps to a value of `None`, the corresponding weight is ignored.

    ``weights_transforms`` rewrite the raw weight stream in list order before
    any name mapping."""

    orig_to_new_renaming: list[Any] = field(default_factory=list)
    orig_to_new_regex: Mapping[re.Pattern, str | None] = field(default_factory=dict)
    orig_to_new_substr: Mapping[str, str | None] = field(default_factory=dict)
    orig_to_new_stacked: Mapping[str, tuple[str, ShardIds]] = field(default_factory=dict)
    orig_to_new_prefix: Mapping[str, str | None] = field(default_factory=dict)
    orig_to_new_suffix: Mapping[str, str | None] = field(default_factory=dict)
    weights_transforms: list[WeightsTransformFn] = field(default_factory=list)

    def __or__(self, other: "WeightsMapper") -> "WeightsMapper":
        """Combine two `WeightsMapper`s by merging their mappings."""
        return WeightsMapper(
            orig_to_new_renaming=[
                *self.orig_to_new_renaming,
                *other.orig_to_new_renaming,
            ],
            orig_to_new_regex={**self.orig_to_new_regex, **other.orig_to_new_regex},
            orig_to_new_substr={**self.orig_to_new_substr, **other.orig_to_new_substr},
            orig_to_new_stacked={
                **self.orig_to_new_stacked,
                **other.orig_to_new_stacked,
            },
            orig_to_new_prefix={**self.orig_to_new_prefix, **other.orig_to_new_prefix},
            orig_to_new_suffix={**self.orig_to_new_suffix, **other.orig_to_new_suffix},
            weights_transforms=[*self.weights_transforms, *other.weights_transforms],
        )

    def map_name(self, key: str) -> str | None:
        """Map a weight name; returns ``None`` if the weight should be ignored."""
        result = self._map_name_with_shard(key)
        return result[0] if result is not None else None

    def _map_name(self, key: str) -> str | None:
        """Backward-compatible alias for :meth:`map_name`."""
        return self.map_name(key)

    def _map_name_with_shard(self, key: str) -> tuple[str, ShardIds | None] | None:
        """Map a weight name and extract any shard_id metadata.

        Returns:
            (mapped_name, shard_id) if the name should be kept. A list of shard
            ids means the weight is loaded into each of those shards in turn.
            None if the name should be dropped.

        """
        # Deprecation warnings
        if key.endswith(".kv_scale"):
            logger.warning_once(
                "DEPRECATED. Found kv_scale in the checkpoint. "
                "This format is deprecated in favor of separate k_scale and "
                "v_scale tensors and will be removed in a future release. "
                "Functionally, we will remap kv_scale to k_scale and duplicate "
                "k_scale to v_scale"
            )

        for renaming in self.orig_to_new_renaming:
            key, _ = renaming.rename_source_key(key)

        for pattern, new_key in self.orig_to_new_regex.items():
            if pattern.search(key):
                if new_key is None:
                    return None

                key = pattern.sub(new_key, key)

        for substr, new_key in self.orig_to_new_substr.items():
            if substr in key:
                if new_key is None:
                    return None

                key = key.replace(substr, new_key, 1)

        shard_id: ShardIds | None = None
        for substr, (new_key, new_shard_id) in self.orig_to_new_stacked.items():
            if substr in key:
                key = key.replace(substr, new_key, 1)
                shard_id = new_shard_id

        for prefix, new_key in self.orig_to_new_prefix.items():
            if key.startswith(prefix):
                if new_key is None:
                    return None

                key = key.replace(prefix, new_key, 1)

        for suffix, new_key in self.orig_to_new_suffix.items():
            if key.endswith(suffix):
                if new_key is None:
                    return None

                key = new_key.join(key.rsplit(suffix, 1))

        return key, shard_id


# --- vllm model_executor/layers/quantization/base_config.py:232-260 (verbatim) ----------------
# The inner "from vllm.model_executor.models.utils import WeightsMapper" is dropped:
# WeightsMapper is module-level here (verbatim copy above).


class QuantizationConfig:
    @staticmethod
    def get_cache_scale_mapper() -> "WeightsMapper":
        """Mapping from checkpoint KV-cache scale names to vLLM scale names.

        Returning a mapper here causes `AutoWeightsLoader` to apply it to the
        weight stream automatically; individual model `load_weights` methods
        do not need to know about KV-cache scales.
        """
        orig_to_new_regex = {
            # Deprecated fused kv_scale -> attn.k_scale
            re.compile(r"\.kv_scale$"): r".attn.k_scale",
            # ModelOpt: .self_attn.{k,v}_proj.{k,v}_scale -> .self_attn.attn.*
            re.compile(r"\.self_attn\.[kv]_proj\.([kv])_scale$"): (r".self_attn.attn.\1_scale"),
            # Fused QKV / qkqkv proj: .self_attn.qk(qk)v_proj.{k,v}_scale -> attn
            re.compile(r"\.self_attn\.qk(?:qk)?v_proj\.([kv])_scale$"): (r".self_attn.attn.\1_scale"),
            # NemotronH: .mixer.{k,v}_proj.{k,v}_scale -> .mixer.attn.*
            re.compile(r"\.mixer\.[kv]_proj\.([kv])_scale$"): r".mixer.attn.\1_scale",
            # HYV3: .self_attn.q.scale -> .self_attn.attn.q_scale
            re.compile(r"\.self_attn\.q\.scale$"): r".self_attn.attn.q_scale",
            # HYV3: .self_attn.{k,v}_cache.scale -> .self_attn.attn.{k,v}_scale
            re.compile(r"\.self_attn\.([kv])_cache\.scale$"): (r".self_attn.attn.\1_scale"),
            # Default: .{q,k,v}_scale -> .attn.{q,k,v}_scale (unless already .attn)
            re.compile(r"(?<!\.attn)\.([qkv])_scale$"): r".attn.\1_scale",
            re.compile(r"(?<!\.attn)\.([qkv])_zero_point$"): r".attn.\1_zero_point",
        }
        return WeightsMapper(orig_to_new_regex=orig_to_new_regex)


# --- vllm-ascend quantization/utils.py:87-134 (verbatim) --------------------------------------


def get_model_file(
    model: str | Path,
    filename: str,
    revision: str | None = None,
) -> Path | None:
    """Get a file from local model directory or download from remote repo.

    This function handles both local paths and remote repository IDs,
    automatically downloading files from HuggingFace Hub or ModelScope
    if they are not already cached.

    Args:
        model: Local directory path or HuggingFace/ModelScope repo id.
        filename: Name of the file to retrieve (e.g., "config.json").
        revision: Optional revision (branch, tag, or commit hash) for remote repos.

    Returns:
        Path to the file if found, None otherwise.
    """
    # Check if it's a local path
    model_path = Path(model) if isinstance(model, str) else model
    if model_path.exists():
        file_path = model_path / filename
        return file_path if file_path.exists() else None

    # Remote repo: try to download from HF Hub or ModelScope
    try:
        if envs.VLLM_USE_MODELSCOPE:  # type: ignore[name-defined] # noqa: F821
            from modelscope.hub.file_download import model_file_download  # type: ignore[import-untyped]

            downloaded_path = model_file_download(
                model_id=str(model),
                file_path=filename,
                revision=revision,
            )
            return Path(downloaded_path)
        else:
            from huggingface_hub import hf_hub_download

            downloaded_path = hf_hub_download(
                repo_id=str(model),
                filename=filename,
                revision=revision,
            )
            return Path(downloaded_path)
    except Exception as e:
        logger.warning("Could not download %s from %s: %s", filename, model, e)
        return None


# --- vllm-ascend quantization/utils.py:137-167 (verbatim; inner import adapted) ---------------


def detect_quantization_method(model: str, revision: str | None = None) -> str | None:
    """Auto-detect the ModelSlim (Ascend) quantization method from model files.

    Only ModelSlim needs file-based auto-detection here: ModelSlim-quantized
    checkpoints carry their quantization description in a dedicated
    ``quant_model_description.json`` file which upstream vLLM does not know
    about.  All other quantization formats (compressed-tensors, fp8, gptq, ...)
    declare ``quantization_config.quant_method`` in ``config.json`` and are
    already auto-detected by upstream vLLM's
    ``ModelConfig._verify_quantization``, so re-detecting them here would be
    redundant.

    Args:
        model: Local directory path **or** HuggingFace / ModelScope repo id.
        revision: Optional model revision (branch, tag, or commit id).

    Returns:
        ``"ascend"`` for ModelSlim models, or ``None`` if no ModelSlim
        quantization signature is found.
    """
    modelslim_path = get_model_file(model, MODELSLIM_CONFIG_FILENAME, revision=revision)
    if modelslim_path is not None:
        return ASCEND_QUANTIZATION_METHOD
    return None


# --- vllm-ascend quantization/methods/kv_cache/kv_c8.py (verbatim) ----------------------------


def _fa_quant_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor):
    """Weight loader for MLA-based C8 (FAKQuant) models."""
    if param.numel() == 1 and loaded_weight.numel() == 1:
        param.data.fill_(loaded_weight.item())
    else:
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        shard_size = loaded_weight.shape[0] // tp_size
        loaded_weight = loaded_weight.narrow(0, shard_size * tp_rank, shard_size)
        assert param.size() == loaded_weight.size(), (
            "[vllm-ascend/FAKQuant] Attempted to load weight "
            f"({loaded_weight.size()}) into parameter ({param.size()}) "
            f"when TP size is {tp_size} and TP rank is {tp_rank}."
        )

        param.data.copy_(loaded_weight)


class AscendFAQuantAttentionMethod:
    """FAKQuant attention scheme (kv_c8.py:31-72).

    ``__init__`` (reads the serving engine's vllm_config for ``kv_lora_rank`` /
    ``qk_rope_head_dim``) and the ``@register_scheme("FAKQuant", "attention")``
    decorator are not copied; the tests bypass construction via ``__new__`` and set
    the two ranks explicitly. Everything below is verbatim.
    """

    def create_weights(self, layer: torch.nn.Module) -> None:
        extra_module_names = ["fa_q", "fa_k", "fa_v"]
        for name in extra_module_names:
            setattr(layer, name, torch.nn.Module())
        params_dict = {}
        dtype = torch.get_default_dtype()
        params_dict["fa_q.scale"] = torch.empty((layer.num_heads, 1), dtype=dtype)
        params_dict["fa_k.scale"] = torch.empty((layer.num_kv_heads, 1), dtype=dtype)
        params_dict["fa_v.scale"] = torch.empty((layer.num_kv_heads, 1), dtype=dtype)
        params_dict["fa_q.offset"] = torch.empty((layer.num_heads, 1), dtype=torch.int8)
        params_dict["fa_k.offset"] = torch.empty((layer.num_kv_heads, 1), dtype=torch.int8)
        params_dict["fa_v.offset"] = torch.empty((layer.num_kv_heads, 1), dtype=torch.int8)

        for name, weight in params_dict.items():
            module_name, weight_name = name.rsplit(".", 1)
            module = getattr(layer, module_name)
            weight_param = torch.nn.Parameter(weight, requires_grad=False)
            module.register_parameter(weight_name, weight_param)
            # When loading weights, segment them according to TP
            weight_param.weight_loader = _fa_quant_weight_loader

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        fa_k_scale = torch.squeeze(layer.fa_k.scale).unsqueeze(0)
        layer.fak_descale_float = torch.nn.Parameter(fa_k_scale.to(torch.float), requires_grad=False)
        layer.fak_descale = torch.nn.Parameter(fa_k_scale, requires_grad=False)
        if get_current_hardware_profile().supports(HardwareCapability.FP8_ATTENTION):
            layer.fak_descale_reciprocal = 1.0 / torch.nn.Parameter(fa_k_scale.to(torch.float), requires_grad=False)
        else:
            layer.fak_descale_reciprocal = 1.0 / torch.nn.Parameter(fa_k_scale, requires_grad=False)
        fa_k_offset = torch.squeeze(layer.fa_k.offset).unsqueeze(0)
        layer.fak_offset = torch.nn.Parameter(fa_k_offset.to(layer.fak_descale.dtype), requires_grad=False)

        repeated_quant_kscale = fa_k_scale.repeat(self.kv_lora_rank)
        layer.quant_kscale = repeated_quant_kscale.view(1, self.kv_lora_rank)
        layer.quant_kscale = 1.0 / torch.nn.Parameter(layer.quant_kscale.to(torch.float), requires_grad=False)


# --- vllm-ascend quantization/configs/modelslim_config.py (verbatim; see per-method notes) ----


class AscendModelSlimConfig(QuantizationConfig):
    """Config class for Ascend ModelSlim quantization (modelslim_config.py:478+).

    Only the methods the consumer contract tests exercise are copied; unused
    methods (get_quant_method, apply_vllm_mapper, prefix mappers, ...) stay in the
    upstream file. Bodies are verbatim; the single adaptation is
    ``maybe_update_config``'s inner ``from vllm_ascend.quantization.utils import
    get_model_file`` which resolves to the module-level verbatim copy.
    """

    # Assigned in _add_kvcache_quant_metadata; declared here so the types
    # resolve regardless of method order (get_cache_scale_mapper reads and
    # conditionally rewrites enable_fa_quant before the assignment site).
    enable_fa_quant: bool
    kvcache_quant_layers: list[int]
    enable_indexer_quant: bool
    indexer_quant_layers: list[int]
    enable_c8_quant: bool
    c8_quant_layers: list[int]

    def __init__(self, quant_config: dict[str, Any] | None = None):
        super().__init__()
        # This will be updated by upstream vLLM with model-specific mappings.
        self.packed_modules_mapping: dict[str, list[str]] = {}
        self.quant_description = quant_config if quant_config is not None else {}
        self._format_metadata: dict[str, Any] = {}
        self._apply_extra_quant_adaptations()
        self.model_type: str | None = None
        self.hf_to_vllm_mapper: WeightsMapper | None = None
        self._mapper_applied = False
        self._add_kvcache_quant_metadata()

    @classmethod
    def get_name(cls) -> str:
        return ASCEND_QUANTIZATION_METHOD

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.int8, torch.float16, torch.bfloat16]

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        # Return empty list so that vllm's get_quant_config() skips the
        # file-based lookup (which raises an unfriendly "Cannot find the
        # config file for ascend" error when the model is not quantized).
        # Instead, the config file is loaded in maybe_update_config(),
        # which can provide a user-friendly error message.
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AscendModelSlimConfig":
        # Format-only HF metadata is not a per-parameter quantization description.
        metadata_keys = {"quant_method", "model_quant_type"}
        if config.get("quant_method") == ASCEND_QUANTIZATION_METHOD and set(config) <= metadata_keys:
            result = cls()
            result._format_metadata = dict(config)
            return result
        return cls(config)

    def get_cache_scale_mapper(self) -> "WeightsMapper":
        """Upstream use staticmethod, but we need to use instance attribute"""
        suffix_map: dict[str, str | None] = {}
        regex_map = {}
        if self._has_mxfp_c8_recipe and not self.enable_mxfp_c8_quant:
            logger.warning_once(
                "[quantization] The checkpoint carries an MXFP8 KV recipe, but "
                "--kv-cache-dtype is not 'mxfp8'; the recipe is not enabled. "
                "Pass --kv-cache-dtype mxfp8 to serve with C8-MXFP."
            )
        if self.enable_mxfp_c8_quant and self.enable_fa_quant:
            # A checkpoint can carry fa_quant_type alongside an MXFP8 KV
            # recipe; FAKQuant would win every shared dispatch and rename
            # fa_v.scale into an MLA submodule this dense model does not
            # have, silently dropping the scale. Stand FAKQuant down.
            logger.warning_once(
                "[quantization] fa_quant_type=%s is declared alongside an MXFP8 KV cache; ignoring FAKQuant.",
                self.quant_description.get("fa_quant_type", ""),
            )
            self.enable_fa_quant = False
        if self.enable_c8_quant:
            suffix_map.update(
                {
                    ".k_proj.kv_cache_scale": ".attn.k_cache_scale",
                    ".k_proj.kv_cache_offset": ".attn.k_cache_offset",
                    ".v_proj.kv_cache_scale": ".attn.v_cache_scale",
                    ".v_proj.kv_cache_offset": ".attn.v_cache_offset",
                }
            )
        if self.enable_mxfp_c8_quant:
            # MXFP C8 quantizes K dynamically, but V uses the static E8M0
            # per-channel scale stored in the ModelSlim checkpoint
            # (fa_v.scale, reusing FAKQuant's key for a different recipe).
            suffix_map[".fa_v.scale"] = ".attn.v_cache_scale"
            # The remaining FAKQuant legacy tensors have no MXFP8 consumer.
            suffix_map[".fa_v.offset"] = None
            suffix_map.update(dict.fromkeys(_MXFP_C8_UNUSED_FA_SUFFIXES, None))
        if self.enable_fa_quant:
            # Some models (e.g., Kimi-K2.6) have a nested module and call AutoWeightsLoader twice, to avoid double
            # mapping, we use regex mapping.
            regex_map.update(
                {
                    re.compile(r"(?<!\.mla_attn\.mla_attn)\.fa_q\.scale$"): ".mla_attn.mla_attn.fa_q.scale",
                    re.compile(r"(?<!\.mla_attn\.mla_attn)\.fa_k\.scale$"): ".mla_attn.mla_attn.fa_k.scale",
                    re.compile(r"(?<!\.mla_attn\.mla_attn)\.fa_v\.scale$"): ".mla_attn.mla_attn.fa_v.scale",
                    re.compile(r"(?<!\.mla_attn\.mla_attn)\.fa_q\.offset$"): ".mla_attn.mla_attn.fa_q.offset",
                    re.compile(r"(?<!\.mla_attn\.mla_attn)\.fa_k\.offset$"): ".mla_attn.mla_attn.fa_k.offset",
                    re.compile(r"(?<!\.mla_attn\.mla_attn)\.fa_v\.offset$"): ".mla_attn.mla_attn.fa_v.offset",
                }
            )
        if self.enable_indexer_quant:
            regex_map.update(
                {
                    re.compile(r"(?<!\.mla_attn\.mla_attn)\.indexer\.q_rot$"): ".mla_attn.mla_attn.indexer.q_rot",
                    re.compile(r"(?<!\.mla_attn\.mla_attn)\.indexer\.k_rot$"): ".mla_attn.mla_attn.indexer.k_rot",
                }
            )
        if not suffix_map and not regex_map:
            return QuantizationConfig.get_cache_scale_mapper()
        cache_scale_mapper = WeightsMapper(orig_to_new_suffix=suffix_map, orig_to_new_regex=regex_map)
        return cache_scale_mapper | QuantizationConfig.get_cache_scale_mapper()

    @property
    def enable_mxfp_c8_quant(self) -> bool:
        """C8-MXFP is enabled with ``--kv-cache-dtype mxfp8``.

        A Model-config-time query: returns False while the engine config is
        still being assembled or absent (e.g. unit tests without a
        ``set_current_vllm_config`` context), so no quant-config constructor
        or mapper may take a C8 decision through this switch.
        """
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is None:
            return False
        return vllm_config.cache_config.cache_dtype == C8_MXFP_KV_CACHE_DTYPE

    @property
    def _has_mxfp_c8_recipe(self) -> bool:
        """Whether the checkpoint carries an MXFP8 KV recipe (model-wide
        ``kv_cache_type`` or a per-layer ``quant_type``). This is NOT the
        switch -- ``--kv-cache-dtype mxfp8`` is; the recipe only decides
        whether the fa_v.scale weights are expected."""
        kv_quant_type = self.quant_description.get("kv_cache_type", "")
        return kv_quant_type == _MXFP_C8_KV_CACHE_TYPE or any(
            key.endswith(_MXFP_C8_LAYER_QUANT_TYPE_SUFFIX) and value in _MXFP_C8_LAYER_QUANT_TYPES
            for key, value in self.quant_description.items()
        )

    def maybe_update_config(
        self,
        model_name: str,
        hf_config: Any = None,
        revision: str | None = None,
    ) -> None:
        """Load the ModelSlim quantization config from model directory.

        This method is called by vllm after get_quant_config() returns
        successfully. Since we return an empty list from get_config_filenames()
        to bypass vllm's built-in file lookup, we do the actual config loading
        here and provide user-friendly error messages when the config is missing.

        Args:
            model_name: Path to the model directory or HuggingFace /
                ModelScope repo id.
            hf_config: The Hugging Face config of the model
            revision: Optional revision (branch, tag, or commit hash) for
                remote repos.
        """
        # If quant_description is already populated (e.g. from from_config()),
        # there is nothing to do.
        if self.quant_description:
            return

        # Try to get the config file (local or remote)
        config_path = get_model_file(model_name, MODELSLIM_CONFIG_FILENAME, revision=revision)

        if config_path is not None:
            with open(config_path) as f:
                self.quant_description = {**self._format_metadata, **json.load(f)}
            self._apply_extra_quant_adaptations()
            self._add_kvcache_quant_metadata()
            return

        # Collect diagnostic info for the error message
        json_names: list[str] = []
        if os.path.isdir(model_name):
            json_files = glob.glob(os.path.join(model_name, "*.json"))
            json_names = [os.path.basename(f) for f in json_files]

        # Config file not found - raise a friendly error message
        logger.error(
            "ModelSlim quantization config not found for model '%s'. Searched path: %s. Found JSON files: %s.",
            model_name,
            model_name,
            json_names if json_names else "N/A",
        )
        raise ValueError(
            "\n"
            + "=" * 80
            + "\n"
            + "ERROR: ModelSlim Quantization Config Not Found\n"
            + "=" * 80
            + "\n"
            + "\n"
            + f"You have enabled '--quantization {ASCEND_QUANTIZATION_METHOD}' "
            + "(ModelSlim quantization),\n"
            + f"but the model '{model_name}' does not contain the required\n"
            + f"quantization config file ('{MODELSLIM_CONFIG_FILENAME}').\n"
            + "\n"
            + "This usually means the model weights are NOT quantized by "
            + "ModelSlim.\n"
            + "\n"
            + "Please choose one of the following solutions:\n"
            + "\n"
            + "  Solution 1: Remove the quantization option "
            + "(for float/unquantized models)\n"
            + "  "
            + "-" * 58
            + "\n"
            + f"    Remove '--quantization {ASCEND_QUANTIZATION_METHOD}' from "
            + "your command if you want to\n"
            + "    run the model with the original (float) weights.\n"
            + "\n"
            + "    Example:\n"
            + f"      vllm serve {model_name}\n"
            + "\n"
            + "  Solution 2: Quantize your model weights with ModelSlim first\n"
            + "  "
            + "-" * 58
            + "\n"
            + "    Quantize your model weights with ModelSlim\n"
            + "    before deployment.\n"
            + "    After quantization, the model directory should contain "
            + f"'{MODELSLIM_CONFIG_FILENAME}'.\n"
            + "    For more information, please refer to:\n"
            + "    https://gitee.com/ascend/msit/tree/master/msmodelslim\n"
            + "\n"
            + (f"  (Found JSON files in model directory: {json_names})\n" if json_names else "")
            + "=" * 80
        )

    def _apply_extra_quant_adaptations(self) -> None:
        """Apply extra adaptations to the quant_description dict.

        This handles known key transformations such as shared_head and
        weight_packed mappings.
        """
        if "hc_head_fn" in self.quant_description:
            extra_quant_dict = {}
            for name in self.quant_description:
                new_name = name
                if not name.startswith("model"):
                    new_name = f"model.{name}"
                extra_quant_dict[new_name] = self.quant_description[name]
            self.quant_description.update(extra_quant_dict)

            extra_quant_dict = {}
            for name in self.quant_description:
                new_name = name
                if "attn" in name and "self_attn" not in name:
                    new_name = name.replace(".attn.", ".self_attn.")
                extra_quant_dict[new_name] = self.quant_description[name]
            self.quant_description.update(extra_quant_dict)

            extra_quant_dict = {}
            for name in self.quant_description:
                new_name = name
                if "ffn" in name:
                    new_name = name.replace("ffn", "mlp")
                extra_quant_dict[new_name] = self.quant_description[name]
            self.quant_description.update(extra_quant_dict)

            extra_quant_dict = {}
            for name in self.quant_description:
                new_name = name
                if "w1" in name:
                    new_name = name.replace(".w1.", ".gate_proj.")
                if "w2" in name:
                    new_name = name.replace(".w2.", ".down_proj.")
                if "w3" in name:
                    new_name = name.replace(".w3.", ".up_proj.")

                if "head" in name and "lm_head" not in name:
                    new_name = name.replace("head", "lm_head")
                if "embed" in name and "embed_tokens" not in name:
                    new_name = name.replace("embed", "embed_tokens")
                extra_quant_dict[new_name] = self.quant_description[name]
            self.quant_description.update(extra_quant_dict)

        extra_quant_dict = {}
        for k in self.quant_description:
            if "shared_head" in k:
                new_k = k.replace(".shared_head.", ".")
                extra_quant_dict[new_k] = self.quant_description[k]
            if "transformer.shared_head.output." in k:
                # Step3.5 MTP checkpoints describe per-layer draft logits heads
                # as ``transformer.shared_head.output``. The vLLM model module
                # exposes the same parameter as ``shared_head.head``.
                new_k = k.replace(
                    "transformer.shared_head.output.",
                    "shared_head.head.",
                )
                extra_quant_dict[new_k] = self.quant_description[k]
            if "transformer.shared_head.norm." in k:
                new_k = k.replace(
                    "transformer.shared_head.norm.",
                    "shared_head.norm.",
                )
                extra_quant_dict[new_k] = self.quant_description[k]
            if "weight_packed" in k:
                new_k = k.replace("weight_packed", "weight")
                extra_quant_dict[new_k] = self.quant_description[k]
        self.quant_description.update(extra_quant_dict)

    def _add_kvcache_quant_metadata(self):
        fa_quant_type = self.quant_description.get("fa_quant_type", "")
        self.enable_fa_quant = fa_quant_type != ""
        self.kvcache_quant_layers = []
        indexer_quant_type = self.quant_description.get("indexer_quant_type", "")
        self.enable_indexer_quant = indexer_quant_type != ""
        self.indexer_quant_layers = []
        kv_quant_type = self.quant_description.get("kv_cache_type", "")
        self.enable_c8_quant = kv_quant_type == "C8"
        self.c8_quant_layers = []
        if self.enable_fa_quant or self.enable_indexer_quant or self.enable_c8_quant:
            for key in self.quant_description:
                _id = "".join(re.findall(r"\.(\d+)\.", key))
                if "fa_k.scale" in key:
                    self.kvcache_quant_layers.append(int(_id))
                if "indexer.quant_type" in key:
                    self.indexer_quant_layers.append(int(_id))
                if "k_proj.kv_cache_scale" in key:
                    self.c8_quant_layers.append(int(_id))
