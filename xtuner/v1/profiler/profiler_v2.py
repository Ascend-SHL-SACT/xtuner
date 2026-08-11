# Copyright (c) 2024-2026, XTuner contributors. All rights reserved.
#
# Ported from MindSpeed-MM (mindspeed_mm/tools/profiler.py and
# mindspeed_mm/fsdp/tools/profiler.py, Apache-2.0). This module is the
# MindSpeed-ported NPU profiling layer for xtuner: a configurable
# ``torch_npu.profiler`` collection path that also produces the Ascend
# ``*_ascend_pt/`` CSV/DB artifacts consumable by the ``profiling-analysis``
# skill.
#
# Wiring: ``xtuner.v1.profiler.__init__`` routes the existing
# ``profiling_time(profile_dir)`` symbol to this module only on NPU when
# ``XTUNER_PROFILE_ENABLE=1``; on NPU with env=0 the original ``npu_profile``
# (HEAD) is used; GPU always uses ``cuda_profile``. No trainer / worker /
# config edits.
"""MindSpeed-ported Ascend NPU profiling layer.

This module is wired into the existing framework on NPU through the
``XTUNER_PROFILE_ENABLE`` environment variable (see
``xtuner/v1/profiler/__init__.py``); GPU always uses ``cuda_profile``.
It exposes:

- :class:`ProfilingConfig` / :class:`StaticParam` / :class:`DynamicParam`:
  Pydantic config mirroring MindSpeed-MM ``fsdp/params/tools_args.py``.
- :func:`profiling_config_from_env`: build a :class:`ProfilingConfig` from
  ``XTUNER_PROFILE_*`` env vars (returns ``None`` when disabled).
- :class:`Profiler`: MindSpeed ``Profiler`` port (start/step/stop lifecycle
  with a contiguous ``[start_step, end_step)`` schedule window; NPU only;
  dynamic mode).
- :func:`profiling_time`: drop-in replacement of the legacy
  ``profiling_time(profile_dir)`` context manager used by
  ``trainer._maybe_profiling`` / ``worker._maybe_profiling`` -- one
  self-contained per-step trace, fully configurable, producing the full
  CSV/DB set online.
- :func:`analyse` + ``__main__``: offline ``torch_npu`` analyse CLI.

The critical addition over the legacy path is the explicit
``export_type=[Text, Db]`` in ``_ExperimentalConfig``: ``analyse_flag=True``
online otherwise yields only the text CSVs and omits ``analysis.db``, and
is a silent no-op inside daemon processes (e.g. Ray workers). The offline
:func:`analyse` CLI is the robust fallback.
"""

from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from xtuner.v1.utils import get_logger


if TYPE_CHECKING:
    from collections.abc import Sequence


logger = get_logger()


# ---------------------------------------------------------------------------
# Config (mirrors MindSpeed-MM fsdp/params/tools_args.py)
# ---------------------------------------------------------------------------


class StaticParam(BaseModel):
    """Static-mode profiler parameters.

    Each field mirrors the MindSpeed-MM ``StaticParam`` dataclass and is
    configurable via ``XTUNER_PROFILE_*`` env through
    :func:`profiling_config_from_env`.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    level: str = Field(default="level1", description="profiling level: level0|level1|level2.")
    with_stack: bool = Field(default=False, description="collect operator call stacks.")
    with_memory: bool = Field(default=False, description="collect per-operator memory usage (profile_memory).")
    record_shapes: bool = Field(default=False, description="collect operator input shapes/types.")
    with_cpu: bool = Field(default=True, description="collect CPU events alongside NPU.")
    save_path: str | None = Field(default=None, description="output directory; None lets the caller (trainer) choose.")
    start_step: int = Field(default=10, description="first step to record (skip_first in schedule terms).")
    end_step: int = Field(default=11, description="step at which recording stops (exclusive).")
    data_simplification: bool = Field(default=False, description="enable torch_npu data-simplification mode.")
    aic_metrics_type: str = Field(
        default="PipeUtilization", description="AI Core metric: PipeUtilization|ArithmeticUtilization."
    )

    analyse_flag: bool = Field(default=True, description="run online analyse on trace finalization -> CSV/DB.")


class DynamicParam(BaseModel):
    """Dynamic-mode parameters (runtime-switchable profiling)."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    config_path: str | None = Field(
        default=None, description="folder or profiler_config.json for torch_npu dynamic_profile."
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
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, protected_namespaces=())

    enable: bool = Field(default=False, description="enable profiling.")
    profile_type: str = Field(default="static", description="static|dynamic.")
    ranks: list[int] = Field(default_factory=lambda: [0], description="ranks to profile; [-1] = all.")
    static_param: StaticParam = Field(default_factory=StaticParam)
    dynamic_param: DynamicParam = Field(default_factory=DynamicParam)


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


def profiling_config_from_env() -> ProfilingConfig | None:
    """Build a :class:`ProfilingConfig` from ``XTUNER_PROFILE_*`` env vars.

    Returns ``None`` when ``XTUNER_PROFILE_ENABLE`` is unset / not truthy, so
    callers can treat a ``None`` result as "profiling disabled".
    """
    if not _get_bool_env("XTUNER_PROFILE_ENABLE", False):
        return None
    cfg = ProfilingConfig(
        enable=True,
        profile_type=os.environ.get("XTUNER_PROFILE_TYPE", "static"),
        ranks=_get_ranks_env("XTUNER_PROFILE_RANKS", [0]),
        static_param=StaticParam(
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
        ),
        dynamic_param=DynamicParam(
            config_path=os.environ.get("XTUNER_PROFILE_DYNAMIC_CONFIG_PATH") or None,
        ),
    )
    log_rank0_info(cfg)
    return cfg


def log_rank0_info(cfg: ProfilingConfig) -> None:
    """Log the resolved profiling config once on rank 0."""
    if _rank() != 0:
        return
    sp = cfg.static_param
    logger.info(
        "[profiler_v2] "
        f"enable={cfg.enable} type={cfg.profile_type} ranks={cfg.ranks} "
        f"level={sp.level} aic={sp.aic_metrics_type} "
        f"with_cpu={sp.with_cpu} with_stack={sp.with_stack} "
        f"with_memory={sp.with_memory} record_shapes={sp.record_shapes} "
        f"data_simpl={sp.data_simplification} "
        f"start={sp.start_step} end={sp.end_step} analyse={sp.analyse_flag}"
    )


# ---------------------------------------------------------------------------
# torch_npu enum resolution (lazy import; version-tolerant)
# ---------------------------------------------------------------------------


def _resolve_profiler_level(level: str) -> "object":
    """Map ``level0/1/2`` to ``torch_npu.profiler.ProfilerLevel``."""
    import torch_npu  # noqa: PLC0415  lazy: keep torch_npu out of non-NPU import graph

    mapping = {
        "level0": torch_npu.profiler.ProfilerLevel.Level0,
        "level1": torch_npu.profiler.ProfilerLevel.Level1,
        "level2": torch_npu.profiler.ProfilerLevel.Level2,
    }
    if level not in mapping:
        raise ValueError(f"profiler_level only supports level0, level1, level2, but gets {level}")
    return mapping[level]


def _resolve_aic_metrics(aic_metrics_type: str) -> "object":
    """Map ``PipeUtilization``/``ArithmeticUtilization`` to ``torch_npu`` enum."""
    import torch_npu  # noqa: PLC0415  lazy

    if aic_metrics_type == "PipeUtilization":
        return torch_npu.profiler.AiCMetrics.PipeUtilization
    if aic_metrics_type == "ArithmeticUtilization":
        return torch_npu.profiler.AiCMetrics.ArithmeticUtilization
    raise ValueError("aic_metrics_type only supports PipeUtilization and ArithmeticUtilization")


def _resolve_export_type() -> "list[object] | None":
    """Return ``[ExportType.Text, ExportType.Db]`` if available, else ``None``.

    ``export_type`` is what makes ``analyse_flag=True`` online also emit
    ``analysis.db`` (without it only text CSVs are produced). Older
    ``torch_npu`` may lack ``ExportType``; in that case fall back to ``None``
    and rely on the offline :func:`analyse` CLI.
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


# ---------------------------------------------------------------------------
# Profiler class (MindSpeed port: start/step/stop, contiguous window)
# ---------------------------------------------------------------------------


class Profiler:
    """MindSpeed-ported profiler with a contiguous ``[start, end)`` window.

    Reserved public API: the live trainer / worker path uses
    :func:`profiling_time` (per-step drop-in), so this class has no call site
    in the current codebase. It is retained as the complete MindSpeed port for
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
        sp = config.static_param
        self.sp_level = sp.level
        self.sp_with_stack = sp.with_stack
        self.sp_with_memory = sp.with_memory
        self.sp_record_shapes = sp.record_shapes
        self.sp_with_cpu = sp.with_cpu
        self.sp_save_path = str(save_path)
        self.sp_start_step = sp.start_step
        self.sp_end_step = sp.end_step
        self.sp_data_simplification = sp.data_simplification
        self.sp_analyse_flag = sp.analyse_flag
        self.aic_metrics_type = sp.aic_metrics_type
        self.dp_config_path = config.dynamic_param.config_path
        self.prof = self._build_profile()

    def _build_profile(self) -> Any:
        """Construct the torch_npu profiler object (NPU only)."""
        return self._build_npu_profile()

    def _build_npu_profile(self) -> Any:
        """Build the NPU (torch_npu.profiler) profile object."""
        import torch_npu  # noqa: PLC0415  lazy

        if self.profile_type == "static":
            profiler_level = _resolve_profiler_level(self.sp_level)
            aic_metrics_type = _resolve_aic_metrics(self.aic_metrics_type)
            export_type = _resolve_export_type()
            kwargs: dict[str, object] = {
                "aic_metrics": aic_metrics_type,
                "profiler_level": profiler_level,
                "data_simplification": self.sp_data_simplification,
            }
            if export_type is not None:
                kwargs["export_type"] = export_type
            experimental_config = torch_npu.profiler._ExperimentalConfig(**kwargs)
            skip_first = self.sp_start_step
            active = self.sp_end_step - self.sp_start_step
            activities = [torch_npu.profiler.ProfilerActivity.NPU]
            if self.sp_with_cpu:
                activities.append(torch_npu.profiler.ProfilerActivity.CPU)
            return torch_npu.profiler.profile(
                with_stack=self.sp_with_stack,
                record_shapes=self.sp_record_shapes,
                profile_memory=self.sp_with_memory,
                activities=activities,
                schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=active, repeat=1, skip_first=skip_first),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    self.sp_save_path, analyse_flag=self.sp_analyse_flag
                ),
                experimental_config=experimental_config,
            )
        if self.profile_type == "dynamic":
            from torch_npu.profiler import dynamic_profile as dp  # noqa: PLC0415  lazy

            return dp
        raise ValueError(f"profile_type only supports static and dynamic, but gets {self.profile_type}")

    def _enable_profile(self) -> bool:
        """Return True when the current rank should profile."""
        if not self.enable:
            return False
        return _should_profile_rank(self.ranks)

    def start(self) -> None:
        """Start profiling (or init dynamic mode)."""
        if not self._enable_profile():
            return
        if self.profile_type == "static":
            self.prof.start()
        else:
            self.prof.init(self.dp_config_path)

    def step(self) -> None:
        """Advance the schedule by one step."""
        if not self._enable_profile():
            return
        self.prof.step()

    def stop(self) -> None:
        """Stop profiling (no-op for dynamic mode)."""
        if not self._enable_profile():
            return
        if self.profile_type == "static":
            self.prof.stop()


# ---------------------------------------------------------------------------
# profiling_time -- drop-in for the legacy per-step context manager
# ---------------------------------------------------------------------------


@contextmanager
def profiling_time(profile_dir: Path):
    """Collect one self-contained torch_npu trace into ``profile_dir``.

    Drop-in replacement of the legacy ``npu_profile.profiling_time`` (NPU):
    same signature so ``trainer.py`` / ``worker.py`` need no change. GPU
    always uses ``cuda_profile.profiling_time`` directly (this module is
    NPU-only). Reads config from
    :func:`profiling_config_from_env` (``XTUNER_PROFILE_*``). When profiling
    is disabled or the current rank is not selected, this is a zero-overhead
    passthrough.

    Each call builds a one-step ``torch_npu.profiler.profile`` with
    ``schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0)`` and, with
    ``analyse_flag=True`` + ``export_type=[Text, Db]``, finalizes two output
    trees under ``profile_dir``: ``ASCEND_PROFILER_OUTPUT/`` (the
    Python-style unsuffixed set: ``step_trace_time.csv``,
    ``op_statistic.csv``, ``kernel_details.csv``, ``operator_details.csv``,
    ``analysis.db``) and ``PROF_*/mindstudio_profiler_output/`` (the
    CANN-native suffixed set: ``op_statistic_<ts>.csv``,
    ``op_summary_<ts>.csv``, ``api_statistic_<ts>.csv``,
    ``task_time_<ts>.csv``). Note ``op_summary_*.csv`` lives ONLY in
    ``mindstudio_profiler_output/`` (not ``ASCEND_PROFILER_OUTPUT/``); the
    ``profiling-analysis`` skill's recursive ``**/op_summary_*.csv`` glob
    finds it there.

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
            "[profiler_v2] XTUNER_PROFILE_TYPE=dynamic is ignored by the "
            "per-step profiling_time path: dynamic_profile is a long-lived "
            "singleton, incompatible with a per-step context manager. "
            "Falling back to a one-step static trace. Use the Profiler class "
            "for loop-scoped dynamic collection."
        )
    import torch_npu  # noqa: PLC0415  lazy

    profiler_level = _resolve_profiler_level(sp.level)
    aic_metrics_type = _resolve_aic_metrics(sp.aic_metrics_type)
    export_type = _resolve_export_type()
    kwargs: dict[str, object] = {
        "aic_metrics": aic_metrics_type,
        "profiler_level": profiler_level,
        "data_simplification": sp.data_simplification,
    }
    if export_type is not None:
        kwargs["export_type"] = export_type
    experimental_config = torch_npu.profiler._ExperimentalConfig(**kwargs)
    activities = [torch_npu.profiler.ProfilerActivity.NPU]
    if sp.with_cpu:
        activities.append(torch_npu.profiler.ProfilerActivity.CPU)
    profile_dir.mkdir(parents=True, exist_ok=True)
    with torch_npu.profiler.profile(
        with_stack=sp.with_stack,
        record_shapes=sp.record_shapes,
        profile_memory=sp.with_memory,
        activities=activities,
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            str(profile_dir), analyse_flag=sp.analyse_flag
        ),
        experimental_config=experimental_config,
    ) as prof:
        yield
        prof.step()


# ---------------------------------------------------------------------------
# Offline analyse CLI (produces the CSV/DB set; daemon-safe fallback)
# ---------------------------------------------------------------------------


def analyse(
    profiler_path: str | Path,
    max_process_number: int | None = None,
    export_type: "Sequence[str]" = ("text", "db"),
) -> None:
    """Run the torch_npu offline analyse to materialize CSV/DB artifacts.

    Required when online ``analyse_flag=True`` could not run (daemon
    processes such as Ray workers silently no-op) or when ``analysis.db`` was
    not emitted. Also re-runnable on an existing trace dir.

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
