"""Carry W8A8 dynamic linear quantization into HuggingFace-format export directories.

The training side fake-quantizes on the deployed ``W8A8_DYNAMIC`` grids but keeps training
in bf16; the deployment side needs int8 weights plus the ModelSlim artifacts it actually
consumes. The export reads the freshly written bf16 weights back from the shards (the
async saver may already have advanced the in-memory parameters), quantizes each weight on
the same per-output-channel grid the forward fake path used -- on an NPU box the
deployment kernel itself (bitwise the training grid), on a CPU-only box the eager chain
(which differs from the kernel grid by the bounded deltas documented in ``kernels.py``)
-- and rewrites the shard in place. Fused grouped weights map to their per-expert HF keys through the model's
``to_hf_key_list``; each expert slice is quantized independently on exactly the per-row
grid the fused training path used (``amax`` over the same rows), so the checkpoint grid is
the grid the master weights were trained against.

The consumer contract (vllm-ascend ``AscendW8A8DynamicLinearMethod`` /
``AscendW8A8DynamicFusedMoEMethod``, pinned in ``tests/quantization/_w8a8_consumer_ref.py``):
``<module>.weight`` int8 ``[out, in]``, ``<module>.weight_scale`` fp32 ``[out, 1]``, and
``<module>.weight_offset`` fp32 ``[out, 1]`` — the engine loads the offset but never
applies it in the matmul, so it exports as zeros. Detection keys on
``quant_model_description.json``; every quantized module is marked ``W8A8_DYNAMIC`` by its
bare prefix, with routed experts marked through their ``experts.0.*`` template keys (the
consumer's packed-module mapping expands an experts prefix into exactly those). The
description is read-merge-written so entries written moments earlier by the KV FAKQuant
export survive; a conflicting value for the same key fails fast. ``config.json`` is
deliberately left untouched: the ModelSlim path activates on the description file when
``config.json`` declares no ``quantization_config``, while an XTuner-specific
``quant_method`` there would make upstream vLLM reject the whole checkpoint.
"""

import json
from pathlib import Path
from typing import Any, Callable, cast

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from torch import nn

from .kernels import _NPU_FUSED_DTYPES, per_channel_weight_quant
from .modules import Int8W8A8DynamicGroupedLinear, Int8W8A8DynamicLinear


# Export artifact names (shared with the KV export; the description file is the ModelSlim
# recipe the deployment engine detects).
_HF_INDEX_FILENAME = "model.safetensors.index.json"
_MODELSLIM_DESCRIPTION_FILENAME = "quant_model_description.json"
_MODELSLIM_DESCRIPTION_VERSION = "1.0.0"
_MODELSLIM_W8A8_DYNAMIC = "W8A8_DYNAMIC"
_W8A8_SCALE_SUFFIX = "weight_scale"
_W8A8_OFFSET_SUFFIX = "weight_offset"


def maybe_write_hf_w8a8_linear(model: nn.Module, hf_dir: str | Path) -> None:
    """Rewrite the HF export's W8A8 linear weights as int8 with ModelSlim scale artifacts.

    Rank 0 only; a no-op when the model has no W8A8 quantized linear module (feature off).
    Every exported weight must already be listed in ``model.safetensors.index.json``
    (checked upfront); the affected shards are then rewritten one at a time. Routed-expert
    keys exist for all experts because the HF save gathers the full model on rank 0. The
    ``quant_model_description.json`` recipe is merged into, never overwriting existing
    entries (see the module docstring).

    Args:
        model (nn.Module): The training model the W8A8 modules live in.
        hf_dir (str | Path): HuggingFace-format export directory written by the engine.
    """
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    plan = _w8a8_export_plan(model)
    if not plan:
        return
    hf_dir = Path(hf_dir)
    index_path = hf_dir / _HF_INDEX_FILENAME
    if not index_path.exists():
        raise FileNotFoundError(f"W8A8 linear export requires a sharded HF checkpoint index: {index_path} not found.")
    index = cast(dict[str, Any], json.loads(index_path.read_text()))
    weight_map: dict[str, str] = index["weight_map"]
    missing = [key for _, keys in plan for key in keys if key not in weight_map]
    if missing:
        raise KeyError(
            f"W8A8 linear export found HF weights missing from the index: {missing[:4]} (and possibly more)."
        )

    description: dict[str, str] = {}
    shard_keys: dict[str, list[str]] = {}
    for _, keys in plan:
        for key in keys:
            shard_keys.setdefault(weight_map[key], []).append(key)
        for desc_key in _description_keys(keys):
            description[desc_key] = _MODELSLIM_W8A8_DYNAMIC

    for shard, keys in shard_keys.items():
        shard_path = hf_dir / shard
        tensors = load_file(shard_path)
        edits: dict[str, torch.Tensor] = {}
        for hf_key in keys:
            if hf_key not in tensors:
                raise KeyError(f"W8A8 linear export found {hf_key!r} in the index but not in shard {shard!r}.")
            w = tensors[hf_key]
            if w.ndim != 2:
                raise ValueError(f"Expected a 2D linear weight for {hf_key!r}, got shape {tuple(w.shape)}.")
            q, scale = _export_quantize_weight(w)
            module_key = hf_key.removesuffix(".weight")
            edits[hf_key] = q
            edits[f"{module_key}.{_W8A8_SCALE_SUFFIX}"] = scale
            edits[f"{module_key}.{_W8A8_OFFSET_SUFFIX}"] = torch.zeros_like(scale)
        collisions = [key for key in edits if not key.endswith(".weight") and key in tensors]
        if collisions:
            raise ValueError(
                f"W8A8 artifact {collisions[0]!r} already exists in shard {shard!r}; refusing to overwrite."
            )
        tensors.update(edits)
        save_file(tensors, shard_path, metadata={"format": "pt"})
        for key in edits:
            weight_map[key] = shard
    index_path.write_text(json.dumps(index, indent=4) + "\n")

    _merge_description(hf_dir, description)


def _hf_weight_keys(module_name: str, to_hf_key_list: Callable[[str], list[str]]) -> list[str]:
    # Training-side module path -> exported HF weight names: drop the text-model root, drop
    # the activation-checkpointing wrapper segments, and let the model apply its own torch
    # -> HF mapping (``model.language_model`` prefix, MTP rewrite, fused-experts expansion).
    key = module_name.replace("._checkpoint_wrapped_module.", ".")
    if key.startswith("language_model."):
        key = key[len("language_model.") :]
    return to_hf_key_list(f"{key}.weight")


def _to_hf_key_list_fn(model: nn.Module) -> Callable[[str], list[str]]:
    # W8A8 linear quant lives on the text model's MLP layers; the compose model holds the
    # text model under ``language_model`` (plain models own ``to_hf_key_list`` directly).
    text_model = getattr(model, "language_model", model)
    to_hf_key_list = getattr(text_model, "to_hf_key_list", None)
    if to_hf_key_list is None:
        raise AttributeError(
            "The model owning W8A8 linear quant modules must implement ``to_hf_key_list`` so the exported weights "
            "get the same HF names as the rest of the checkpoint."
        )
    return cast("Callable[[str], list[str]]", to_hf_key_list)


def _w8a8_export_plan(model: nn.Module) -> list[tuple[str, list[str]]]:
    # One entry per quantized module: its training-side name and the HF weight names its
    # exported weight maps to (one for dense modules, all experts' slices for grouped ones).
    # Empty when the feature is off — the export is then a complete no-op, and
    # ``to_hf_key_list`` is only demanded of the model once a quant module actually exists
    # (mirroring the KV export, so feature-off models without the mapping stay untouched).
    quant_modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, (Int8W8A8DynamicLinear, Int8W8A8DynamicGroupedLinear))
    ]
    if not quant_modules:
        return []
    to_hf_key_list = _to_hf_key_list_fn(model)
    plan: list[tuple[str, list[str]]] = []
    for name, module in quant_modules:
        keys = _hf_weight_keys(name, to_hf_key_list)
        if isinstance(module, Int8W8A8DynamicLinear) and len(keys) != 1:
            raise ValueError(f"Module {name!r} mapped to {keys} HF keys; expected exactly one.")
        plan.append((name, keys))
    return plan


def _description_keys(hf_keys: list[str]) -> list[str]:
    # The consumer's packed-module mapping expands a routed-experts prefix into its
    # ``experts.0.*`` shard prefixes, so routed experts are marked through those template
    # keys only; every other module is marked by its own bare prefix. Shared experts never
    # match ``.experts.0.`` (their keys carry ``shared_experts.`` without a numeric
    # segment — the shared-expert MLP is a single fused module, not per-expert children).
    bare = [key.removesuffix(".weight") for key in hf_keys]
    expert0 = [key for key in bare if ".experts.0." in key]
    return expert0 if expert0 else bare


def _merge_description(hf_dir: Path, description: dict[str, str]) -> None:
    # Read-merge-write: the KV FAKQuant export writes the same file moments earlier (blind
    # overwrite), so its ``version``/``fa_quant_type``/FA entries must survive. A key
    # mapped to a different value means two features claim one layer — fail fast rather
    # than silently re-quantize.
    desc_path = hf_dir / _MODELSLIM_DESCRIPTION_FILENAME
    if desc_path.exists():
        merged = cast(dict[str, Any], json.loads(desc_path.read_text()))
    else:
        merged = {"version": _MODELSLIM_DESCRIPTION_VERSION}
    for key, value in description.items():
        if key in merged and merged[key] != value:
            raise ValueError(
                f"{_MODELSLIM_DESCRIPTION_FILENAME} already maps {key!r} to {merged[key]!r}; "
                f"refusing to overwrite with {value!r}."
            )
        merged[key] = value
    desc_path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")


def _npu_export_available() -> bool:
    # torch_npu registers the ``npu`` namespace on ``torch``; without the plugin the
    # attribute does not exist at all, and a CPU-only box must keep the eager chain.
    npu = getattr(torch, "npu", None)
    return npu is not None and bool(npu.is_available())


def _export_quantize_weight(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Export grid = training grid: with torch_npu loaded, the shard weight is quantized by
    # the deployment kernel itself -- bitwise the grid the NPU training forward
    # fake-quantized against (kernels.py dispatches on device + dtype, so a bf16/fp16
    # shard takes the fused path). safetensors requires CPU tensors, so the artifacts are
    # moved back. A CPU-only export keeps the eager chain, which differs from the kernel
    # grid by the bounded deltas documented in ``kernels.py`` (scale off by at most 1 ulp
    # fp32, level flips on ~0.05-0.09% of elements).
    if w.dtype in _NPU_FUSED_DTYPES and _npu_export_available():
        q, scale = per_channel_weight_quant(w.to("npu"))
        return q.cpu(), scale.cpu()
    return per_channel_weight_quant(w)


__all__ = [
    "maybe_write_hf_w8a8_linear",
]
