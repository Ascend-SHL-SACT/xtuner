# Copyright (c) 2024-2026, XTuner contributors. All rights reserved.
#
# Ported from MindSpeed-MM (mindspeed_mm/tools/profiler.py and
# mindspeed_mm/fsdp/tools/profiler.py, Apache-2.0). This module is the
# MindSpeed-ported NPU profiling layer for xtuner: a fully-capable
# ``torch_npu.profiler`` collection path that exercises every
# ``_ExperimentalConfig`` field plus ``with_modules``/``with_flops``,
# ``export_memory_timeline``, ``add_metadata`` and the async trace handler,
# and produces the Ascend ``*_ascend_pt/`` CSV/DB artifacts consumable by the
# ``profiling-analysis`` skill and ``msprof-analyze``.
#
# Wiring: ``xtuner.v1.profiler.__init__`` routes the existing
# ``profiling_time(profile_dir)`` symbol to this module only on NPU when
# ``XTUNER_PROFILE_ENABLE=1``; on NPU with env=0 the original ``npu_profile``
# (HEAD) is used; GPU always uses ``cuda_profile``. No trainer / worker /
# config edits, no new env vars, no new files.
"""MindSpeed-ported Ascend NPU profiling layer (functionally complete).

This module is wired into the existing framework on NPU through the
``XTUNER_PROFILE_ENABLE`` environment variable (see
``xtuner/v1/profiler/__init__.py``); GPU always uses ``cuda_profile``.

Every ``ascend_pytorch_profiler`` capability is exposed (see the user guide at
``pytorch/docs/zh/ascend_pytorch_profiler/ascend_pytorch_profiler_user_guide.md``):

- All 15 ``_ExperimentalConfig`` fields: ``profiler_level``, ``aic_metrics``,
  ``l2_cache``, ``msprof_tx``, ``mstx``, ``data_simplification``,
  ``record_op_args``, ``op_attr``, ``gc_detect_threshold``, ``export_type``,
  ``host_sys``, ``mstx_domain_include``, ``mstx_domain_exclude``, ``sys_io``,
  ``sys_interconnection``.
- ``profile()`` extras: ``with_modules``, ``with_flops``.
- ``tensorboard_trace_handler`` knobs: ``analyse_flag``, ``async_mode``.
- ``prof.export_memory_timeline`` (auto-satisfies its prerequisites).
- ``prof.add_metadata`` / ``add_metadata_json``.
- ``dynamic_profile`` (dp.init / dp.step) loop-scoped mode.

The capability switches beyond the env vars live in a ``profiler_config.json``
written by the launch script and located via the *existing* env var
``XTUNER_PROFILE_DYNAMIC_CONFIG_PATH`` (no new env vars). Fields present in the
JSON override the env-var defaults; absent fields fall back to env defaults.
The JSON follows the guide's ``profiler_config.json`` schema so the same file
also drives ``dynamic_profile`` natively.

Public surface:

- :class:`StaticParam` / :class:`DynamicParam` / :class:`ExperimentalConfigJson`
  / :class:`FullProfileConfig` / :class:`ProfilingConfig`: pydantic config.
- :func:`profiling_config_from_env`: build a :class:`ProfilingConfig` from
  ``XTUNER_PROFILE_*`` env vars + optional ``profiler_config.json``.
- :func:`Profiler`: MindSpeed ``Profiler`` port (start/step/stop lifecycle with
  a contiguous ``[start_step, end_step)`` schedule window; honours
  ``XTUNER_PROFILE_TYPE=dynamic``).
- :func:`profiling_time`: drop-in replacement of the legacy
  ``profiling_time(profile_dir)`` context manager used by
  ``trainer._maybe_profiling`` / ``worker._maybe_profiling`` -- one
  self-contained per-step trace, fully configurable, producing the full
  CSV/DB set online.
- :func:`analyse` + ``__main__``: offline ``torch_npu`` analyse CLI
  (daemon-safe fallback).
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from xtuner.v1.utils import get_logger


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence


logger = get_logger()


# ---------------------------------------------------------------------------
# Config (mirrors MindSpeed-MM fsdp/params/tools_args.py + the guide's
# profiler_config.json schema)
# ---------------------------------------------------------------------------


class StaticParam(BaseModel):
    """Static-mode profiler parameters (env-driven, backward compatible).

    Each field mirrors the MindSpeed-MM ``StaticParam`` dataclass and is
    configurable via ``XTUNER_PROFILE_*`` env through
    :func:`profiling_config_from_env`. Coarse-grained switches that also appear
    in ``profiler_config.json`` are overridden by the JSON when present.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    level: str = Field(default="level1", description="profiling level: level0|level1|level2|level_none.")
    with_stack: bool = Field(default=False, description="collect operator call stacks.")
    with_memory: bool = Field(default=False, description="collect per-operator memory usage (profile_memory).")
    record_shapes: bool = Field(default=False, description="collect operator input shapes/types.")
    with_cpu: bool = Field(default=True, description="collect CPU events alongside NPU.")
    save_path: str | None = Field(default=None, description="output directory; None lets the caller (trainer) choose.")
    start_step: int = Field(
        default=10, description="first step to record (skip_first in schedule terms; Profiler class only)."
    )
    end_step: int = Field(default=11, description="step at which recording stops, exclusive (Profiler class only).")
    data_simplification: bool = Field(
        default=False, description="enable torch_npu data-simplification mode (False keeps full data)."
    )
    aic_metrics_type: str = Field(
        default="PipeUtilization",
        description="AI Core metric: one of the 9 documented AiCMetrics names.",
    )
    analyse_flag: bool = Field(default=True, description="run online analyse on trace finalization -> CSV/DB.")
    config_path: str | None = Field(
        default=None,
        description="path to profiler_config.json (from XTUNER_PROFILE_DYNAMIC_CONFIG_PATH).",
    )


class DynamicParam(BaseModel):
    """Dynamic-mode parameters (runtime-switchable profiling)."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    config_path: str | None = Field(default=None, description="profiler_config.json for torch_npu dynamic_profile.")


class ExperimentalConfigJson(BaseModel):
    """``experimental_config`` subset of ``profiler_config.json``.

    Mirrors all 15 fields of ``torch_npu.profiler._ExperimentalConfig``. Every
    field is optional (``None`` = not overridden); absent fields fall back to
    the env default or the ``torch_npu`` built-in default. ``extra="forbid"``
    follows the xtuner.v1 pydantic policy; an unknown field rejects the whole
    JSON (caught in :func:`_load_full_config`, which warns and falls back to
    env defaults), so every capability field is modeled explicitly.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    profiler_level: str | None = Field(default=None, description="level0|level1|level2|level_none.")
    aic_metrics: str | None = Field(default=None, description="one of the 9 AiCMetrics names.")
    l2_cache: bool | None = Field(default=None, description="L2 cache data.")
    msprof_tx: bool | None = Field(default=None, description="legacy msproftx marking data.")
    mstx: bool | None = Field(default=None, description="mstx (MUSA Tools Extension) marking data.")
    data_simplification: bool | None = Field(
        default=None, description="enable data-simplification (False keeps full data)."
    )
    record_op_args: bool | None = Field(
        default=None, description="record operator args (guide: do not co-enable with other data)."
    )
    op_attr: bool | None = Field(default=None, description="operator attribute data.")
    gc_detect_threshold: float | None = Field(
        default=None, description="GC detect threshold (seconds); None = disabled."
    )
    export_type: list[str] | None = Field(
        default=None, description="subset of [text, db]; None defaults to [text, db]."
    )
    host_sys: list[str] | None = Field(
        default=None,
        description="subset of [CPU, MEM, DISK, NETWORK, OSRT]; DISK needs iotop, OSRT needs perf+ltrace.",
    )
    mstx_domain_include: list[str] | None = Field(default=None, description="mstx domains to include.")
    mstx_domain_exclude: list[str] | None = Field(default=None, description="mstx domains to exclude.")
    sys_io: bool | None = Field(default=None, description="system I/O data.")
    sys_interconnection: bool | None = Field(default=None, description="system interconnection data.")


class FullProfileConfig(BaseModel):
    """Full-capability ``profiler_config.json`` (guide schema + xtuner extras).

    Drives every ``ascend_pytorch_profiler`` capability. In static per-step
    mode, fields present here override the env-var defaults; absent fields fall
    back to env defaults. In dynamic mode ``torch_npu`` reads this file
    natively (so the same file serves both modes).
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    activities: list[str] | None = Field(default=None, description="CPU and/or NPU (dynamic mode).")
    prof_dir: str | None = Field(default=None, description="output dir (dynamic mode).")
    analyse: bool | None = Field(default=None, description="analyse flag (dynamic mode).")
    async_mode: bool | None = Field(
        default=None, description="async trace handler; overrides tensorboard_trace_handler default."
    )
    record_shapes: bool | None = Field(default=None, description="collect operator input shapes.")
    profile_memory: bool | None = Field(default=None, description="collect per-operator memory.")
    with_stack: bool | None = Field(default=None, description="collect operator call stacks.")
    with_flops: bool | None = Field(default=None, description="collect FLOPs (pythonic flops).")
    with_modules: bool | None = Field(default=None, description="annotate operators with owning module.")
    active: int | None = Field(default=None, description="active steps (dynamic mode).")
    warmup: int | None = Field(default=None, description="warmup steps (dynamic mode).")
    start_step: int | None = Field(default=None, description="start step (dynamic mode).")
    is_rank: bool | None = Field(default=None, description="rank gating (dynamic mode).")
    rank_list: list[int] | None = Field(default=None, description="ranks to profile (dynamic mode).")
    metadata: dict[str, Any] | None = Field(
        default=None, description="metadata dict applied via prof.add_metadata_json."
    )
    experimental_config: ExperimentalConfigJson | None = Field(
        default=None, description="all 15 _ExperimentalConfig fields."
    )
    # xtuner extension: the guide calls export_memory_timeline as an API; here it
    # is auto-invoked from the custom on_trace_ready handler.
    export_memory_timeline: bool | None = Field(
        default=None, description="emit memory_timeline.html (auto-satisfies prerequisites)."
    )
    memory_timeline_device: str | None = Field(
        default=None, description='device for export_memory_timeline, e.g. "npu:0".'
    )


class ProfilingConfig(BaseModel):
    """Top-level profiling config.

    Args:
        enable (bool): master switch; when False the Profiler / profiling_time
            is inert.
        profile_type (str): ``static`` (default) or ``dynamic``.
        ranks (list[int]): ranks to profile; ``[-1]`` means all ranks.
        static_param (StaticParam): static-mode parameters.
        dynamic_param (DynamicParam): dynamic-mode parameters.
        full_config (FullProfileConfig | None): loaded from
            ``profiler_config.json``; ``None`` when no JSON is configured.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, protected_namespaces=())

    enable: bool = Field(default=False, description="enable profiling.")
    profile_type: str = Field(default="static", description="static|dynamic.")
    ranks: list[int] = Field(default_factory=lambda: [0], description="ranks to profile; [-1] = all.")
    static_param: StaticParam = Field(default_factory=StaticParam)
    dynamic_param: DynamicParam = Field(default_factory=DynamicParam)
    full_config: FullProfileConfig | None = Field(
        default=None, description="loaded from profiler_config.json (None if absent)."
    )


# ---------------------------------------------------------------------------
# Env -> Config
# ---------------------------------------------------------------------------


def _get_bool_env(name: str, default: bool) -> bool:
    """Read a boolean env var (1/true/yes/on -> True)."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int_env(name: str, default: int) -> int:
    """Read an int env var with fallback."""
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return int(value.strip())


def _get_ranks_env(name: str, default: list[int]) -> list[int]:
    """Read a ranks env var: ``-1`` -> ``[-1]`` (all); ``0,1`` -> ``[0,1]``."""
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return [int(p) for p in parts]


def _load_full_config(path: str | None) -> FullProfileConfig | None:
    """Load ``profiler_config.json`` from ``path`` into a
    :class:`FullProfileConfig`.

    Returns ``None`` (with a rank-0 warning) when the path is empty, the file is
    missing, or validation fails, so a malformed config never breaks training.
    """
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        logger.warning("[profiler_v2] profiler_config.json not found at %s; ignoring.", path)
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("[profiler_v2] failed to read %s: %s; ignoring.", path, e)
        return None
    try:
        return FullProfileConfig.model_validate(data)
    except ValidationError as e:
        logger.warning("[profiler_v2] invalid profiler_config.json at %s: %s; ignoring.", path, e)
        return None


def profiling_config_from_env() -> ProfilingConfig | None:
    """Build a :class:`ProfilingConfig` from ``XTUNER_PROFILE_*`` env vars.

    Also loads ``profiler_config.json`` from ``XTUNER_PROFILE_DYNAMIC_CONFIG_PATH``
    (existing env var; no new env introduced) into ``full_config``. Returns
    ``None`` when ``XTUNER_PROFILE_ENABLE`` is unset / not truthy, so callers
    can treat a ``None`` result as "profiling disabled".
    """
    if not _get_bool_env("XTUNER_PROFILE_ENABLE", False):
        return None
    config_path = os.environ.get("XTUNER_PROFILE_DYNAMIC_CONFIG_PATH") or None
    sp = StaticParam(
        level=os.environ.get("XTUNER_PROFILE_LEVEL", "level1"),
        with_stack=_get_bool_env("XTUNER_PROFILE_WITH_STACK", False),
        with_memory=_get_bool_env("XTUNER_PROFILE_WITH_MEMORY", False),
        record_shapes=_get_bool_env("XTUNER_PROFILE_RECORD_SHAPES", False),
        with_cpu=_get_bool_env("XTUNER_PROFILE_WITH_CPU", True),
        save_path=os.environ.get("XTUNER_PROFILE_SAVE_PATH") or None,
        start_step=_get_int_env("XTUNER_PROFILE_START_STEP", 10),
        end_step=_get_int_env("XTUNER_PROFILE_END_STEP", 11),
        data_simplification=_get_bool_env("XTUNER_PROFILE_DATA_SIMPLIFICATION", False),
        aic_metrics_type=os.environ.get("XTUNER_PROFILE_AIC_METRICS", "PipeUtilization"),
        analyse_flag=_get_bool_env("XTUNER_PROFILE_ANALYSE_FLAG", True),
        config_path=config_path,
    )
    full = _load_full_config(config_path)
    cfg = ProfilingConfig(
        enable=True,
        profile_type=os.environ.get("XTUNER_PROFILE_TYPE", "static"),
        ranks=_get_ranks_env("XTUNER_PROFILE_RANKS", [0]),
        static_param=sp,
        dynamic_param=DynamicParam(config_path=config_path),
        full_config=full,
    )
    log_rank0_info(cfg)
    return cfg


def log_rank0_info(cfg: ProfilingConfig) -> None:
    """Log the resolved profiling config (all capabilities) once on rank 0."""
    if _rank() != 0:
        return
    sp = cfg.static_param
    full = cfg.full_config
    ec = full.experimental_config if full and full.experimental_config else ExperimentalConfigJson()
    info = {
        "enable": cfg.enable,
        "type": cfg.profile_type,
        "ranks": cfg.ranks,
        "level": sp.level,
        "aic": sp.aic_metrics_type,
        "with_cpu": sp.with_cpu,
        "with_stack": sp.with_stack,
        "with_memory": sp.with_memory,
        "record_shapes": sp.record_shapes,
        "data_simpl": sp.data_simplification,
        "analyse": sp.analyse_flag,
        "full_config_loaded": full is not None,
        "ec.profiler_level": ec.profiler_level,
        "ec.aic_metrics": ec.aic_metrics,
        "ec.l2_cache": ec.l2_cache,
        "ec.msprof_tx": ec.msprof_tx,
        "ec.mstx": ec.mstx,
        "ec.record_op_args": ec.record_op_args,
        "ec.op_attr": ec.op_attr,
        "ec.gc_detect_threshold": ec.gc_detect_threshold,
        "ec.export_type": ec.export_type,
        "ec.host_sys": ec.host_sys,
        "ec.sys_io": ec.sys_io,
        "ec.sys_interconnection": ec.sys_interconnection,
        "full.with_modules": full.with_modules if full else None,
        "full.with_flops": full.with_flops if full else None,
        "full.async_mode": full.async_mode if full else None,
        "full.export_memory_timeline": full.export_memory_timeline if full else None,
        "full.metadata": bool(full.metadata) if full else False,
    }
    logger.info("[profiler_v2] resolved config: %s", json.dumps(info, ensure_ascii=False, default=str))


# ---------------------------------------------------------------------------
# torch_npu enum resolution (lazy import; version-tolerant)
# ---------------------------------------------------------------------------


def _resolve_profiler_level(level: str) -> object:
    """Map ``level0/1/2/_none`` to ``torch_npu.profiler.ProfilerLevel``."""
    import torch_npu  # noqa: PLC0415  lazy: keep torch_npu out of non-NPU import graph

    mapping = {
        "level0": torch_npu.profiler.ProfilerLevel.Level0,
        "level1": torch_npu.profiler.ProfilerLevel.Level1,
        "level2": torch_npu.profiler.ProfilerLevel.Level2,
        "level_none": torch_npu.profiler.ProfilerLevel.Level_none,
        "levelnone": torch_npu.profiler.ProfilerLevel.Level_none,
    }
    key = level.strip().lower()
    if key not in mapping:
        raise ValueError(f"profiler_level only supports level0|level1|level2|level_none, but gets {level}")
    return mapping[key]


def _resolve_aic_metrics(aic_metrics_type: str) -> object:
    """Map a metric name to ``torch_npu.profiler.AiCMetrics`` (all 9 names)."""
    import torch_npu  # noqa: PLC0415  lazy

    mapping = {
        "aicorenone": torch_npu.profiler.AiCMetrics.AiCoreNone,
        "pipeutilization": torch_npu.profiler.AiCMetrics.PipeUtilization,
        "arithmeticutilization": torch_npu.profiler.AiCMetrics.ArithmeticUtilization,
        "l2cache": torch_npu.profiler.AiCMetrics.L2Cache,
        "memory": torch_npu.profiler.AiCMetrics.Memory,
        "memoryaccess": torch_npu.profiler.AiCMetrics.MemoryAccess,
        "memoryl0": torch_npu.profiler.AiCMetrics.MemoryL0,
        "memoryub": torch_npu.profiler.AiCMetrics.MemoryUB,
        "resourceconflictratio": torch_npu.profiler.AiCMetrics.ResourceConflictRatio,
    }
    key = aic_metrics_type.strip().lower()
    if key not in mapping:
        raise ValueError(f"aic_metrics supports the 9 documented AiCMetrics names, but gets {aic_metrics_type}")
    return mapping[key]


def _default_export_type() -> list[object] | None:
    """Return ``[ExportType.Text, ExportType.Db]`` if available, else ``None``.

    ``export_type`` is what makes ``analyse_flag=True`` online also emit
    ``analysis.db`` (without it only text CSVs are produced). Older
    ``torch_npu`` may lack ``ExportType``; fall back to ``None`` then and rely
    on the offline :func:`analyse` CLI.
    """
    try:
        import torch_npu  # noqa: PLC0415  lazy

        return [torch_npu.profiler.ExportType.Text, torch_npu.profiler.ExportType.Db]
    except AttributeError:
        logger.warning(
            "[profiler_v2] torch_npu.profiler.ExportType unavailable; analysis.db will require offline analyse()"
        )
        return None
    except ImportError:
        return None


def _resolve_export_type(values: list[str] | None) -> list[object] | None:
    """Resolve a list of ``text``/``db`` strings to ``ExportType`` enums.

    ``None`` (not configured) yields the full ``[Text, Db]`` default so the
    online analyse emits both CSVs and ``analysis.db``.
    """
    if values is None:
        return _default_export_type()
    import torch_npu  # noqa: PLC0415  lazy

    mapping = {
        "text": torch_npu.profiler.ExportType.Text,
        "db": torch_npu.profiler.ExportType.Db,
    }
    resolved = [mapping[v.strip().lower()] for v in values if v.strip()]
    return resolved or _default_export_type()


def _resolve_host_sys(values: list[str] | None) -> list[object] | None:
    """Resolve ``CPU/MEM/DISK/NETWORK/OSRT`` strings to ``HostSystem`` enums.

    ``None``/empty -> ``None`` (no host-side collection). Note ``DISK`` requires
    ``iotop`` and ``OSRT`` requires ``perf`` + ``ltrace``; the launch script
    only enables the subset whose backing tools are installed.
    """
    if not values:
        return None
    import torch_npu  # noqa: PLC0415  lazy

    mapping = {
        "cpu": torch_npu.profiler.HostSystem.CPU,
        "mem": torch_npu.profiler.HostSystem.MEM,
        "disk": torch_npu.profiler.HostSystem.DISK,
        "network": torch_npu.profiler.HostSystem.NETWORK,
        "osrt": torch_npu.profiler.HostSystem.OSRT,
    }
    resolved = [mapping[v.strip().lower()] for v in values if v.strip()]
    return resolved or None


# ---------------------------------------------------------------------------
# Rank gating
# ---------------------------------------------------------------------------


def _rank() -> int:
    """Return the current distributed rank (0 if dist is not initialized)."""
    import torch.distributed as dist  # noqa: PLC0415  lazy

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def _should_profile_rank(ranks: list[int]) -> bool:
    """Return True when the current rank should collect a trace.

    Mirrors MindSpeed ``_enable_profile``: ``[-1]`` means all ranks; otherwise
    the rank must be in the list.
    """
    if ranks == [-1]:
        return True
    return _rank() in ranks


def _default_npu_device() -> str:
    """Return the current NPU device id as ``"npu:{id}"`` (best-effort).

    ``torch.npu`` is injected at runtime by ``import torch_npu``, so it is not
    visible to static analysis; it is read via ``getattr`` (which yields
    ``Any``) to stay mypy-strict-clean.
    """
    try:
        import torch  # noqa: PLC0415  lazy
        import torch_npu  # noqa: PLC0415, F401  side-effect: injects the torch.npu namespace
    except ImportError:
        return f"npu:{_rank() % 8}"
    npu = getattr(torch, "npu", None)
    try:
        if npu is not None and hasattr(npu, "current_device"):
            return f"npu:{npu.current_device()}"
    except (RuntimeError, AttributeError):
        pass
    return f"npu:{_rank() % 8}"


# ---------------------------------------------------------------------------
# Shared builder: merge env defaults + JSON overrides -> torch_npu objects
# ---------------------------------------------------------------------------


@dataclass
class _ProfileBuild:
    """Resolved torch_npu profiler inputs produced by
    :func:`_build_profile_objects`."""

    experimental_config: Any
    activities: list[Any]
    schedule: Any
    trace_handler: Callable[[Any], None]
    with_stack: bool
    record_shapes: bool
    profile_memory: bool
    with_modules: bool
    with_flops: bool
    metadata: dict[str, Any] | None


def _or_else(value: Any, default: Any) -> Any:
    """Return ``value`` when not ``None`` else ``default``."""
    return value if value is not None else default


def _make_trace_handler(
    save_path: Path,
    *,
    analyse_flag: bool,
    async_mode: bool,
    export_memory_timeline: bool,
    memory_timeline_device: str,
) -> Callable[[Any], None]:
    """Build an ``on_trace_ready`` handler.

    Emits ``memory_timeline.html`` first (if enabled) then runs the standard
    ``tensorboard_trace_handler`` (online analyse + persist). Ordering matters:
    ``export_memory_timeline`` reads the in-memory trace, so it must run before
    the handler persists / analyses it.
    """
    import torch_npu  # noqa: PLC0415  lazy

    inner = torch_npu.profiler.tensorboard_trace_handler(
        str(save_path), analyse_flag=analyse_flag, async_mode=async_mode
    )
    mem_tl_path = str(Path(str(save_path)) / "memory_timeline.html")

    def _handler(prof: Any) -> None:
        if export_memory_timeline:
            export_fn = getattr(prof, "export_memory_timeline", None)
            if export_fn is not None:
                try:
                    export_fn(output_path=mem_tl_path, device=memory_timeline_device)
                except (RuntimeError, ValueError, AttributeError) as e:
                    logger.warning("[profiler_v2] export_memory_timeline failed: %s", e)
            else:
                logger.warning("[profiler_v2] export_memory_timeline unavailable on this torch_npu; skipping.")
        inner(prof)

    return _handler


def _build_profile_objects(
    sp: StaticParam,
    full: FullProfileConfig | None,
    save_path: Path,
    *,
    active: int,
    skip_first: int,
) -> _ProfileBuild:
    """Single source of truth: merge env defaults + JSON overrides -> torch_npu.

    Constructs the ``_ExperimentalConfig`` (all 15 fields), the activity list,
    the schedule, the custom ``on_trace_ready`` handler, the ``profile()``
    kwargs (``with_stack``/``record_shapes``/``profile_memory``/
    ``with_modules``/``with_flops``) and the metadata dict. Used by both the
    per-step :func:`profiling_time` path and the :class:`Profiler` loop path,
    eliminating the prior duplication.
    """
    import torch_npu  # noqa: PLC0415  lazy

    ec = full.experimental_config if full and full.experimental_config else ExperimentalConfigJson()

    # Effective values: JSON override (if set) else env default.
    level_str = _or_else(ec.profiler_level, sp.level)
    aic_str = _or_else(ec.aic_metrics, sp.aic_metrics_type)
    profiler_level = _resolve_profiler_level(level_str)
    aic_metrics = _resolve_aic_metrics(aic_str)

    ec_kwargs: dict[str, Any] = {
        "profiler_level": profiler_level,
        "aic_metrics": aic_metrics,
        "data_simplification": _or_else(ec.data_simplification, sp.data_simplification),
        "l2_cache": _or_else(ec.l2_cache, False),
        "msprof_tx": _or_else(ec.msprof_tx, False),
        "mstx": _or_else(ec.mstx, False),
        "record_op_args": _or_else(ec.record_op_args, False),
        "op_attr": _or_else(ec.op_attr, False),
        "sys_io": _or_else(ec.sys_io, False),
        "sys_interconnection": _or_else(ec.sys_interconnection, False),
    }
    if ec.gc_detect_threshold is not None:
        ec_kwargs["gc_detect_threshold"] = ec.gc_detect_threshold
    export_type = _resolve_export_type(ec.export_type)
    if export_type is not None:
        ec_kwargs["export_type"] = export_type
    host_sys = _resolve_host_sys(ec.host_sys)
    if host_sys is not None:
        ec_kwargs["host_sys"] = host_sys
    if ec.mstx_domain_include:
        ec_kwargs["mstx_domain_include"] = ec.mstx_domain_include
    if ec.mstx_domain_exclude:
        ec_kwargs["mstx_domain_exclude"] = ec.mstx_domain_exclude

    experimental_config = torch_npu.profiler._ExperimentalConfig(**ec_kwargs)

    # profile() kwargs (JSON override else env default).
    with_stack = _or_else(full.with_stack if full else None, sp.with_stack)
    record_shapes = _or_else(full.record_shapes if full else None, sp.record_shapes)
    profile_memory = _or_else(full.profile_memory if full else None, sp.with_memory)
    with_modules = _or_else(full.with_modules if full else None, False)
    with_flops = _or_else(full.with_flops if full else None, False)

    # export_memory_timeline: auto-satisfy its documented prerequisites
    # (record_shapes + profile_memory + (with_stack or with_modules)).
    export_mem_tl = bool(full.export_memory_timeline) if full else False
    if export_mem_tl:
        if not record_shapes:
            logger.info("[profiler_v2] export_memory_timeline=True -> forcing record_shapes=True.")
            record_shapes = True
        if not profile_memory:
            logger.info("[profiler_v2] export_memory_timeline=True -> forcing profile_memory=True.")
            profile_memory = True
        if not with_stack and not with_modules:
            logger.info("[profiler_v2] export_memory_timeline=True -> forcing with_modules=True.")
            with_modules = True

    activities: list[Any] = [torch_npu.profiler.ProfilerActivity.NPU]
    if sp.with_cpu:
        activities.append(torch_npu.profiler.ProfilerActivity.CPU)

    async_mode = bool(_or_else(full.async_mode if full else None, False))
    mem_device = _or_else(full.memory_timeline_device if full else None, _default_npu_device())
    trace_handler = _make_trace_handler(
        save_path,
        analyse_flag=sp.analyse_flag,
        async_mode=async_mode,
        export_memory_timeline=export_mem_tl,
        memory_timeline_device=mem_device,
    )

    schedule = torch_npu.profiler.schedule(wait=0, warmup=0, active=active, repeat=1, skip_first=skip_first)
    metadata = full.metadata if full and full.metadata else None

    return _ProfileBuild(
        experimental_config=experimental_config,
        activities=activities,
        schedule=schedule,
        trace_handler=trace_handler,
        with_stack=with_stack,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_modules=with_modules,
        with_flops=with_flops,
        metadata=metadata,
    )


def _apply_metadata(prof: Any, metadata: dict[str, Any]) -> None:
    """Attach ``metadata`` to ``prof`` via ``add_metadata`` /
    ``add_metadata_json``.

    Best-effort: missing methods or per-key failures are warned and skipped so a
    metadata error never aborts the trace. Uses ``getattr`` to stay
    ``torch_npu``-version-agnostic and mypy-safe.
    """
    add_json = getattr(prof, "add_metadata_json", None)
    add = getattr(prof, "add_metadata", None)
    for key, value in metadata.items():
        try:
            if isinstance(value, str) and add is not None:
                add(key, value)
            elif add_json is not None:
                add_json(key, json.dumps(value, ensure_ascii=False))
            else:
                logger.warning("[profiler_v2] add_metadata unavailable; skipping key=%s", key)
        except (RuntimeError, ValueError, AttributeError) as e:
            logger.warning("[profiler_v2] add_metadata(%s) failed: %s", key, e)


# ---------------------------------------------------------------------------
# Profiler class (MindSpeed port: start/step/stop, contiguous window)
# ---------------------------------------------------------------------------


class Profiler:
    """MindSpeed-ported profiler with a contiguous ``[start, end)`` window.

    Reserved public API: the live trainer / worker path uses
    :func:`profiling_time` (per-step drop-in), so this class has no call site in
    the current codebase. It is retained as the complete MindSpeed port for
    future loop-scoped collection and as the only path that honours
    ``XTUNER_PROFILE_TYPE=dynamic`` (see :class:`DynamicParam`).

    Use this for loop-scoped collection::

        prof = Profiler(config, save_path)
        prof.start()
        for step in range(total_steps):
            train_one_step()
            prof.step()
        prof.stop()

    For the existing per-step ``_maybe_profiling`` integration (discrete
    steps), prefer :func:`profiling_time` -- it builds a self-contained
    one-step profile per call and is the drop-in the trainer already calls.

    Args:
        config (ProfilingConfig): resolved profiling config.
        save_path (Path): output directory for the trace + CSV/DB.
    """

    def __init__(self, config: ProfilingConfig, save_path: Path) -> None:
        self.enable = config.enable
        self.profile_type = config.profile_type
        self.ranks = config.ranks
        self._sp = config.static_param
        self._full = config.full_config
        self._save_path = Path(save_path)
        self._dp_config_path = config.dynamic_param.config_path
        self.prof, self._metadata = self._build_profile()

    def _build_profile(self) -> tuple[Any, dict[str, Any] | None]:
        """Construct the torch_npu profiler object (NPU only)."""
        if self.profile_type == "static":
            return self._build_static_profile()
        if self.profile_type == "dynamic":
            from torch_npu.profiler import dynamic_profile as dp  # noqa: PLC0415  lazy

            return dp, None
        raise ValueError(f"profile_type only supports static and dynamic, but gets {self.profile_type}")

    def _build_static_profile(self) -> tuple[Any, dict[str, Any] | None]:
        """Build the static NPU profile object via the shared builder."""
        import torch_npu  # noqa: PLC0415  lazy

        build = _build_profile_objects(
            self._sp,
            self._full,
            self._save_path,
            active=self._sp.end_step - self._sp.start_step,
            skip_first=self._sp.start_step,
        )
        prof = torch_npu.profiler.profile(
            activities=build.activities,
            schedule=build.schedule,
            on_trace_ready=build.trace_handler,
            with_stack=build.with_stack,
            record_shapes=build.record_shapes,
            profile_memory=build.profile_memory,
            with_modules=build.with_modules,
            with_flops=build.with_flops,
            experimental_config=build.experimental_config,
        )
        return prof, build.metadata

    def _enable_profile(self) -> bool:
        """Return True when the current rank should profile."""
        if not self.enable:
            return False
        return _should_profile_rank(self.ranks)

    def start(self) -> None:
        """Start profiling (or init dynamic mode) and attach metadata."""
        if not self._enable_profile():
            return
        if self.profile_type == "static":
            self.prof.start()
            if self._metadata:
                _apply_metadata(self.prof, self._metadata)
        else:
            self.prof.init(self._dp_config_path)

    def step(self) -> None:
        """Advance the schedule by one step."""
        if not self._enable_profile():
            return
        self.prof.step()

    def stop(self) -> None:
        """Stop profiling (no-op for dynamic mode; dp finalizes via atexit)."""
        if not self._enable_profile():
            return
        if self.profile_type == "static":
            self.prof.stop()


# ---------------------------------------------------------------------------
# profiling_time -- drop-in for the legacy per-step context manager
# ---------------------------------------------------------------------------


@contextmanager
def profiling_time(profile_dir: Path) -> Iterator[None]:
    """Collect one self-contained torch_npu trace into ``profile_dir``.

    Drop-in replacement of the legacy ``npu_profile.profiling_time`` (NPU): same
    signature so ``trainer.py`` / ``worker.py`` need no change. GPU always uses
    ``cuda_profile.profiling_time`` directly (this module is NPU-only). Reads
    config from :func:`profiling_config_from_env` (``XTUNER_PROFILE_*`` env +
    optional ``profiler_config.json``). When profiling is disabled or the
    current rank is not selected, this is a zero-overhead passthrough.

    Each call builds a one-step ``torch_npu.profiler.profile`` with
    ``schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0)``. With
    ``analyse_flag=True`` + ``export_type=[Text, Db]`` it finalizes two output
    trees under ``profile_dir``: ``ASCEND_PROFILER_OUTPUT/`` (the Python-style
    unsuffixed set: ``step_trace_time.csv``, ``op_statistic.csv``,
    ``kernel_details.csv``, ``operator_details.csv``, ``analysis.db``,
    ``ascend_pytorch_profiler_{Rank}.db``) and
    ``PROF_*/mindstudio_profiler_output/`` (the CANN-native suffixed set:
    ``op_statistic_<ts>.csv``, ``op_summary_<ts>.csv``,
    ``api_statistic_<ts>.csv``, ``task_time_<ts>.csv``, ``msprof_<ts>.db``).
    When ``export_memory_timeline=True`` it also emits
    ``memory_timeline.html``. ``msprof-analyze cluster -d <profile_dir>`` then
    consumes the PROF_ tree.

    Args:
        profile_dir (Path): output directory for this step's trace + CSV/DB.
    """
    cfg = profiling_config_from_env()
    if cfg is None or not cfg.enable:
        yield
        return
    sp = cfg.static_param
    if not _should_profile_rank(cfg.ranks):
        yield
        return
    if cfg.profile_type == "dynamic" and _rank() == 0:
        logger.warning(
            "[profiler_v2] XTUNER_PROFILE_TYPE=dynamic is ignored by the per-step "
            "profiling_time path: dynamic_profile is a long-lived singleton, "
            "incompatible with a per-step context manager. Falling back to a "
            "one-step static trace. Use the Profiler class for loop-scoped "
            "dynamic collection."
        )
    import torch_npu  # noqa: PLC0415  lazy

    save_path = Path(profile_dir)
    build = _build_profile_objects(sp, cfg.full_config, save_path, active=1, skip_first=0)
    save_path.mkdir(parents=True, exist_ok=True)
    with torch_npu.profiler.profile(
        activities=build.activities,
        schedule=build.schedule,
        on_trace_ready=build.trace_handler,
        with_stack=build.with_stack,
        record_shapes=build.record_shapes,
        profile_memory=build.profile_memory,
        with_modules=build.with_modules,
        with_flops=build.with_flops,
        experimental_config=build.experimental_config,
    ) as prof:
        if build.metadata:
            _apply_metadata(prof, build.metadata)
        yield
        prof.step()


# ---------------------------------------------------------------------------
# Offline analyse CLI (produces the CSV/DB set; daemon-safe fallback)
# ---------------------------------------------------------------------------


def analyse(
    profiler_path: str | Path,
    max_process_number: int | None = None,
    export_type: Sequence[str] = ("text", "db"),
) -> None:
    """Run the torch_npu offline analyse to materialize CSV/DB artifacts.

    Required when online ``analyse_flag=True`` could not run (daemon processes
    such as Ray workers silently no-op) or when ``analysis.db`` was not emitted.
    Also re-runnable on an existing trace dir.

    Args:
        profiler_path (str | Path): the profiling data directory.
        max_process_number (int | None): parallelism; None lets torch_npu pick.
        export_type (Sequence[str]): ``("text", "db")`` to emit both CSVs and
            ``analysis.db``.
    """
    from torch_npu.profiler.profiler import analyse as _analyse  # noqa: PLC0415  lazy

    _analyse(
        profiler_path=str(profiler_path),
        max_process_number=max_process_number,
        export_type=list(export_type),
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="xtuner profile offline analysing tool")
    parser.add_argument("--profiler-path", required=True, help="Path to the profiler data directory.")
    parser.add_argument(
        "--max-process-number", type=int, default=None, help="Maximum process number (default: CPU cores / 2)."
    )
    parser.add_argument(
        "--export-type",
        action="append",
        choices=["text", "db"],
        default=None,
        help="Export type(s): text, db. Repeatable. Default: text db.",
    )
    args = parser.parse_args()
    export_type = tuple(args.export_type) if args.export_type else ("text", "db")
    analyse(profiler_path=args.profiler_path, max_process_number=args.max_process_number, export_type=export_type)


if __name__ == "__main__":
    _main()
