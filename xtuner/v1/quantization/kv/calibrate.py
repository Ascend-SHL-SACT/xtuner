"""In-training calibration and scale persistence for the static KV quantization modes.

Per-token dynamic quantization derives its scale from the data itself on every forward, but the
static mode (``per_tensor_static``) needs one frozen scale per module:
the activation scale cannot be re-derived from the weights, so it is calibrated over the first
``kv_calib_num_batches`` training steps (amax over ``kv_a_layernorm`` outputs, MAX-reduced across
all ranks so DP shards and SP sequence shards agree on the same grid), frozen, and persisted to a
sidecar file next to the DCP weights -- re-calibrating after a resume would silently change the
frozen grid. The calibration rides the training loop's own forwards and owns no dataloader: every
extra ``iter()`` spawns a fresh dataloader worker generation that re-tokenizes the same huge
samples, and stacked generations exhausted host memory (run38/run40).
"""

from pathlib import Path
from typing import Callable, Protocol, cast

import torch
import torch.distributed as dist
from torch import nn

from .config import STATIC_KV_QUANT_MODES


_KV_SCALE_SIDECAR_FILENAME = "kv_scale_sidecar.pt"


class StaticKVQuantModule(Protocol):
    """Contract a static-mode KV quant injection point must satisfy.

    See ``NoPEDSAMultiLatentAttention`` for the reference implementation: the scale lives in a
    plain attribute (not a registered buffer) so meta-device construction and DCP state dicts
    never touch it, and the lazy calibrator / checkpoint sidecar fills it before the first
    quantized forward.
    """

    _kv_quant_mode: str
    _kv_scale: torch.Tensor | None
    _kv_scale_ready: bool
    _kv_calibrating: bool
    kv_calib_num_batches: int
    kv_a_layernorm: nn.Module


class KVScaleLazyCalibrator:
    """Freeze static-mode KV scales from the training loop's own forwards, no dataloader of its own.

    On construction it collects every static-mode module that still lacks a frozen scale, flips
    them into passthrough (``_kv_calibrating``), and installs a running-max hook on each
    ``kv_a_layernorm`` output. The trainer reports every completed step via ``on_train_step_end``;
    once ``kv_calib_num_batches`` steps are collected, each record is MAX-reduced across all ranks
    (covering both DP data shards and SP sequence shards, since the fake-quant applies before the
    SP gather), ``scale = amax / 127`` is frozen, the hooks are removed, and the modules switch to
    their quantized forward. Collection steps therefore stay numerically identical to a
    non-quantized run, and no extra forward is ever issued. Upfront dataloader passes were removed
    on purpose: every extra ``iter()`` spawns a fresh dataloader worker generation that
    re-tokenizes the same huge samples, and stacked generations exhausted host memory
    (run38/run40). A run shorter than ``kv_calib_num_batches`` steps never freezes (and writes no
    sidecar); keep the threshold below the planned step count.

    Args:
        model (nn.Module): The training model, already on its target device.
    """

    def __init__(self, model: nn.Module) -> None:
        self._todo: list[tuple[str, StaticKVQuantModule]] = [
            (name, module) for name, module in _named_static_kv_modules(model) if not module._kv_scale_ready
        ]
        num_batches_set = {module.kv_calib_num_batches for _, module in self._todo}
        if len(num_batches_set) > 1:
            raise ValueError(
                f"Inconsistent kv_calib_num_batches across static KV quant modules: {sorted(num_batches_set)}."
            )
        self._num_batches = num_batches_set.pop() if num_batches_set else 0
        self._modules_by_name = dict(self._todo)
        self._records: dict[str, torch.Tensor | None] = {name: None for name, _ in self._todo}
        self._consumed = 0
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        for name, module in self._todo:
            module._kv_calibrating = True
            self._handles.append(module.kv_a_layernorm.register_forward_hook(self._record_hook(name)))

    def _record_hook(self, name: str) -> Callable[[nn.Module, tuple, torch.Tensor], None]:
        records = self._records

        def hook(_module: nn.Module, _args: tuple, output: torch.Tensor) -> None:
            amax = _kv_amax(output)
            prev = records[name]
            records[name] = amax if prev is None else torch.maximum(prev, amax)

        return hook

    @property
    def active(self) -> bool:
        """Whether calibration is still collecting training steps.

        Returns:
            bool: ``True`` while unready static modules remain and the step threshold has not
            been reached; ``False`` once frozen (or when there was nothing to calibrate), where
            every further call is a no-op.
        """
        return bool(self._todo)

    def on_train_step_end(self) -> None:
        """Count one completed training step and freeze the scales once the threshold is reached.

        At the threshold each record is MAX-reduced across all ranks and frozen as
        ``scale = amax / 127`` (clamped away from zero); afterwards the hooks are removed, the
        passthrough flag is cleared, and further calls are no-ops.
        """
        if not self._todo or self._consumed >= self._num_batches:
            return
        self._consumed += 1
        if self._consumed < self._num_batches:
            return
        for name, module in self._todo:
            record = self._records[name]
            assert record is not None, f"calibration hook for {name!r} never fired"
            if dist.is_initialized():
                # One world-wide grid: every rank quantizes its own pre-gather shard with the same scale.
                dist.all_reduce(record, op=dist.ReduceOp.MAX)
            module._kv_scale = record.div(127.0).clamp_min(1e-12)
            module._kv_scale_ready = True
        for handle in self._handles:
            handle.remove()
        for _, module in self._todo:
            module._kv_calibrating = False
        self._todo = []


def _named_static_kv_modules(model: nn.Module) -> list[tuple[str, StaticKVQuantModule]]:
    return [
        (name, cast(StaticKVQuantModule, module))
        for name, module in model.named_modules()
        if getattr(module, "_kv_quant_mode", "") in STATIC_KV_QUANT_MODES
    ]


def _kv_amax(x: torch.Tensor) -> torch.Tensor:
    # abs and amax are exact in the input dtype (amax returns one of the elements), so widening
    # only the reduced result is bitwise identical to float()-first while skipping a full fp32
    # copy of ``x`` on every observed forward during collection.
    return x.detach().abs().amax().float()


def needs_kv_calibration(model: nn.Module) -> bool:
    """Whether any static-mode KV quant module still lacks its frozen scale.

    Args:
        model (nn.Module): The training model.

    Returns:
        bool: ``True`` when at least one static-mode module has no frozen scale yet.
    """
    return any(not module._kv_scale_ready for _, module in _named_static_kv_modules(model))


def maybe_load_kv_scale_sidecar(model: nn.Module, checkpoint_path: str | Path | None) -> None:
    """Load frozen KV scales from a checkpoint sidecar into the static-mode modules.

    A no-op when there is no checkpoint path, no sidecar file, or no static-mode module (the
    feature is off). A sidecar that does not match the current model tree fails fast: re-using
    scales from a different architecture or ``kv_quant`` mode, or re-calibrating only a subset,
    would silently corrupt the frozen grid.

    Args:
        model (nn.Module): The training model, already on its target device.
        checkpoint_path (str | Path | None): Resume checkpoint root (the directory holding the
            ``weights`` dir), or ``None`` when starting fresh.
    """
    if checkpoint_path is None:
        return
    modules = _named_static_kv_modules(model)
    if not modules:
        return
    sidecar = Path(checkpoint_path) / _KV_SCALE_SIDECAR_FILENAME
    if not sidecar.exists():
        return
    scales: dict[str, torch.Tensor] = torch.load(sidecar, map_location="cpu", weights_only=True)
    unknown = set(scales) - {name for name, _ in modules}
    if unknown:
        raise ValueError(
            f"{sidecar} contains scales for modules absent from this model ({sorted(unknown)}); the checkpoint "
            "was written for a different architecture."
        )
    for name, module in modules:
        if name not in scales:
            raise ValueError(f"{sidecar} is missing a scale for {name!r}.")
        scale = scales[name]
        layernorm_weight = cast(torch.Tensor, module.kv_a_layernorm.weight)
        # Shape must match the frozen grid: a [C] vector would silently broadcast where a
        # scalar is required, so a cross-mode sidecar would quantize on the wrong grid.
        shape_ok = scale.ndim == 0
        if not shape_ok:
            raise ValueError(
                f"{sidecar} scale for {name!r} has shape {tuple(scale.shape)} but mode "
                f"{module._kv_quant_mode!r} expects a 0-dim scalar: the sidecar was written under a different "
                "kv_quant mode or architecture."
            )
        module._kv_scale = scale.to(layernorm_weight.device)
        module._kv_scale_ready = True


def maybe_save_kv_scale_sidecar(model: nn.Module, checkpoint_path: str | Path) -> None:
    """Persist the frozen static-mode KV scales into the checkpoint directory.

    Writes ``kv_scale_sidecar.pt`` (module name -> float32 scale) on rank 0 only; a no-op when no
    static-mode module has a frozen scale, so plain runs never grow a stray file. The write is
    all-or-nothing: while lazy calibration is still collecting (some module not yet ready) the
    sidecar is skipped, because a partial sidecar would fail the resume load with "missing a
    scale".

    Args:
        model (nn.Module): The training model.
        checkpoint_path (str | Path): Checkpoint directory the sidecar file is written into.
    """
    modules = _named_static_kv_modules(model)
    if not modules or any(not module._kv_scale_ready for _, module in modules):
        return
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    scales = {
        name: module._kv_scale.detach().cpu().clone() for name, module in modules if module._kv_scale is not None
    }
    torch.save(scales, Path(checkpoint_path) / _KV_SCALE_SIDECAR_FILENAME)
