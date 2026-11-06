"""Carry frozen static-mode KV scales into HuggingFace-format export directories.

The training side persists the scales to ``kv_scale_sidecar.pt`` next to the DCP weights so a
resume reproduces the exact quantization grid; the deployment side needs the same scales inside
the exported HF checkpoint in a format a deployment engine actually consumes.

The export format targets the strictest verified consumer: ``per_tensor_static`` ships
vllm-ascend's ModelSlim/FAKQuant artifact set. That path does not read
``config.json`` at all: its detection keys on the presence of ``quant_model_description.json``
in the model directory (``detect_quantization_method``/``maybe_update_config``), and the
ascend quantization method activates precisely when ``config.json`` declares *no*
``quantization_config`` (``override_quantization_method``). Writing one there would instead
make upstream vLLM reject the whole checkpoint at config time (``ValueError`` on unknown
``quant_method``). So the export ships the ModelSlim artifact set: a
``quant_model_description.json`` recipe declaring ``fa_quant_type: FAKQuant`` plus one
``.fa_*.{scale,offset}`` weight entry per quantized attention layer, following the shapes the
consumer's ``AscendFAQuantAttentionMethod.create_weights`` registers (``[num_kv_heads, 1]``;
MLA latent caches have exactly one KV head) and the symmetric-int8 convention of the training
kernel (``scale = amax / 127``, no zero point, so all offsets are zero; the Q/V entries the
scheme registers but the MLA runtime never reads are exported at their identity values so the
loader initializes them instead of leaving ``torch.empty`` garbage).
"""

import json
from pathlib import Path
from typing import Any, Callable, cast

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch import nn

from .calibrate import StaticKVQuantModule, _named_static_kv_modules


# Export artifact names.
_HF_KV_SCALES_FILENAME = "kv_scale.safetensors"
_HF_INDEX_FILENAME = "model.safetensors.index.json"
_KV_SCALE_META_FILENAME = "kv_scale_meta.json"
_KV_QUANT_PRODUCER = "xtuner-kv-qat"
_EXPORT_FORMAT_VERSION = 1

# ModelSlim (vllm-ascend) artifacts for per-tensor static mode. The recipe filename is the
# consumer's MODELSLIM_CONFIG_FILENAME detection key; ``fa_quant_type: FAKQuant`` selects the
# registered ``@register_scheme("FAKQuant", "attention")`` scheme; the per-layer entry values
# are opaque to the consumer (only the key names drive layer membership), but the canonical
# ModelSlim fixtures mark the FA entries with the scheme name.
_MODELSLIM_DESCRIPTION_FILENAME = "quant_model_description.json"
_MODELSLIM_DESCRIPTION_VERSION = "1.0.0"
_MODELSLIM_FA_QUANT_TYPE = "FAKQuant"
_MODELSLIM_FA_ENTRY = "FAQuant"
_MODELSLIM_FA_SCALE_SUFFIX = "fa_k.scale"
_MODELSLIM_FA_TENSOR_SUFFIXES = (
    "fa_q.scale",
    "fa_k.scale",
    "fa_v.scale",
    "fa_q.offset",
    "fa_k.offset",
    "fa_v.offset",
)
# MLA latent caches hold a single KV head (the compressed ckv latent is shared across heads), so
# the consumer's ``fa_*.scale/offset`` parameters collapse to one scalar per tensor.
_MODELSLIM_FA_TENSOR_SHAPE = (1, 1)
_MODELSLIM_FA_OFFSET_DTYPE = torch.int8


def _hf_scale_key(
    module_name: str,
    to_hf_key_list: Callable[[str], list[str]],
    suffix: str,
) -> str:
    # Training-side module path -> exported HF weight name: drop the text-model root, drop the
    # activation-checkpointing wrapper segments, append the scale tensor name, and let the model
    # apply its own torch->HF mapping (``model.language_model`` prefix, MTP rewrite, ...).
    key = module_name.replace("._checkpoint_wrapped_module.", ".")
    if key.startswith("language_model."):
        key = key[len("language_model.") :]
    keys = to_hf_key_list(f"{key}.{suffix}")
    if len(keys) != 1:
        raise ValueError(f"Module {module_name!r} mapped to {keys} HF keys; expected exactly one.")
    return keys[0]


def _to_hf_key_list_fn(model: nn.Module) -> Callable[[str], list[str]]:
    # Static-mode KV quant lives on the text model's attention layers; the compose model holds
    # the text model under ``language_model`` (plain models own ``to_hf_key_list`` directly).
    text_model = getattr(model, "language_model", model)
    to_hf_key_list = getattr(text_model, "to_hf_key_list", None)
    if to_hf_key_list is None:
        raise AttributeError(
            "The model owning static-mode KV quant modules must implement ``to_hf_key_list`` so the exported "
            "scales get the same HF weight names as the rest of the checkpoint."
        )
    return cast("Callable[[str], list[str]]", to_hf_key_list)


def _modelslim_per_tensor_export(
    modules: list[tuple[str, StaticKVQuantModule]],
    to_hf_key_list: Callable[[str], list[str]],
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    # Build the ModelSlim FAKQuant artifact set for per-tensor scales: one ``[1, 1]`` fp32
    # ``.fa_k.scale`` weight entry per quantized layer plus the five companion tensors the
    # consumer's scheme registers (symmetric int8 has no zero point, so offsets export as zero;
    # the Q/V scales the MLA runtime never reads export at their identity 1.0), and the recipe
    # JSON describing the same layers.
    entries: dict[str, torch.Tensor] = {}
    description: dict[str, str] = {
        "version": _MODELSLIM_DESCRIPTION_VERSION,
        "fa_quant_type": _MODELSLIM_FA_QUANT_TYPE,
    }
    for name, module in modules:
        scale = module._kv_scale
        assert scale is not None  # implied by _kv_scale_ready
        layer_path = _hf_scale_key(name, to_hf_key_list, _MODELSLIM_FA_SCALE_SUFFIX).removesuffix(
            f".{_MODELSLIM_FA_SCALE_SUFFIX}"
        )
        collision = [s for s in _MODELSLIM_FA_TENSOR_SUFFIXES if f"{layer_path}.{s}" in entries]
        if collision:
            raise ValueError(f"HF scale key {collision[0]!r} collides between {name!r} and an earlier module.")
        entries[f"{layer_path}.{_MODELSLIM_FA_SCALE_SUFFIX}"] = (
            scale.detach().to(device="cpu", dtype=torch.float32).reshape(_MODELSLIM_FA_TENSOR_SHAPE).clone()
        )
        for suffix in _MODELSLIM_FA_TENSOR_SUFFIXES:
            if suffix == _MODELSLIM_FA_SCALE_SUFFIX:
                continue
            if suffix.endswith("offset"):
                tensor = torch.zeros(_MODELSLIM_FA_TENSOR_SHAPE, dtype=_MODELSLIM_FA_OFFSET_DTYPE)
            elif suffix == "fa_q.scale" or suffix == "fa_v.scale":
                tensor = torch.ones(_MODELSLIM_FA_TENSOR_SHAPE, dtype=torch.float32)
            else:  # pragma: no cover - _MODELSLIM_FA_TENSOR_SUFFIXES only holds the cases above
                raise ValueError(f"Unexpected ModelSlim FA tensor suffix {suffix!r}.")
            entries[f"{layer_path}.{suffix}"] = tensor
        for fa_suffix in _MODELSLIM_FA_TENSOR_SUFFIXES:
            description[f"{layer_path}.{fa_suffix}"] = _MODELSLIM_FA_ENTRY
    return entries, description


def _kv_scale_export(model: nn.Module) -> tuple[dict[str, torch.Tensor], str, dict[str, str]]:
    # Collect the frozen scales into the ModelSlim FAKQuant artifact set, or an empty dict when
    # the export must be a no-op: no static-mode module (feature off), or calibration still
    # collecting (a partial export would describe a quantization grid the checkpoint does not
    # fully carry).
    modules = _named_static_kv_modules(model)
    if not modules or any(not module._kv_scale_ready for _, module in modules):
        return {}, "", {}
    mode = modules[0][1]._kv_quant_mode
    entries, description = _modelslim_per_tensor_export(modules, _to_hf_key_list_fn(model))
    return entries, mode, description


def maybe_write_hf_kv_scales(model: nn.Module, hf_dir: str | Path) -> None:
    """Write frozen static-mode KV scales into a HuggingFace-format export directory.

    Rank 0 only; a no-op when the model has no static-mode KV quant module or calibration has
    not frozen the scales yet (all-or-nothing, mirroring :func:`maybe_save_kv_scale_sidecar`).
    Writes the ModelSlim-consumable artifact set (see the module docstring):
    ``quant_model_description.json`` plus ``kv_scale.safetensors`` entries named
    ``<HF layer path>.fa_{q,k,v}.{scale,offset}`` appended to ``model.safetensors.index.json``
    when present. ``config.json`` is deliberately left untouched: the ModelSlim path activates
    on the description-file presence only when ``config.json`` carries no
    ``quantization_config``, while any XTuner-specific ``quant_method`` there would make
    upstream vLLM reject the whole checkpoint.

    Args:
        model (nn.Module): The training model the static-mode modules live in.
        hf_dir (str | Path): HuggingFace-format export directory written by the engine.
    """
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    entries, mode, description = _kv_scale_export(model)
    if not entries:
        return
    hf_dir = Path(hf_dir)
    save_file(entries, hf_dir / _HF_KV_SCALES_FILENAME, metadata={"format": "pt"})

    index_path = hf_dir / _HF_INDEX_FILENAME
    if index_path.exists():
        index = cast(dict[str, Any], json.loads(index_path.read_text()))
        weight_map: dict[str, str] = index.setdefault("weight_map", {})
        for key in entries:
            weight_map[key] = _HF_KV_SCALES_FILENAME
        index_path.write_text(json.dumps(index, indent=4) + "\n")

    (hf_dir / _MODELSLIM_DESCRIPTION_FILENAME).write_text(json.dumps(description, indent=2, sort_keys=True) + "\n")

    meta = {
        "format_version": _EXPORT_FORMAT_VERSION,
        "kv_quant_mode": mode,
        "kv_scales_file": _HF_KV_SCALES_FILENAME,
        "producer": _KV_QUANT_PRODUCER,
        "consumer_recipe": _MODELSLIM_DESCRIPTION_FILENAME,
    }
    (hf_dir / _KV_SCALE_META_FILENAME).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
