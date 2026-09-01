"""Regression tests for profiler_v2 env-var defaults and routing-switch name.

Locks two alignments with the launch scripts (``shell/xtuner_512/*.sh``):

* ``XTUNER_PROFILE_ANALYSE_FLAG`` defaults to **off**. Online ``analyse()`` runs
  in-process and synchronously (~500s per profiled step, inflating the next
  step's TGS), so it must be opt-in; the sh scripts already export ``...:=0``.
  Before this change the code default was ``True``, so forgetting the env var
  silently paid the cost.
* The v2-routing switch is ``XTUNER_NPU_PROFILE_V2_ENABLE`` (renamed from
  ``XTUNER_PROFILE_ENABLE``): ``=1`` routes ``profiling_time`` to the NPU
  ``profiler_v2`` (Ascend csv/db), ``=0``/unset keeps the legacy
  ``npu_profile`` (chrome trace). The legacy name no longer routes.

CPU-only: ``profiling_config_from_env`` is pure env->pydantic parsing
(``torch_npu`` is lazy-imported), so no NPU / process group is needed.
"""

from __future__ import annotations

import pytest

from xtuner.v1.profiler.profiler_v2 import profiling_config_from_env


# Every XTUNER_PROFILE_* env var profiling_config_from_env reads, so each test
# starts from a clean slate and owns only the switch under test.
_PROFILE_ENVS: tuple[str, ...] = (
    "XTUNER_NPU_PROFILE_V2_ENABLE",
    "XTUNER_PROFILE_ENABLE",
    "XTUNER_PROFILE_ANALYSE_FLAG",
    "XTUNER_PROFILE_LEVEL",
    "XTUNER_PROFILE_TYPE",
    "XTUNER_PROFILE_RANKS",
    "XTUNER_PROFILE_WITH_CPU",
    "XTUNER_PROFILE_WITH_MEMORY",
    "XTUNER_PROFILE_WITH_STACK",
    "XTUNER_PROFILE_RECORD_SHAPES",
    "XTUNER_PROFILE_DATA_SIMPLIFICATION",
    "XTUNER_PROFILE_AIC_METRICS",
    "XTUNER_PROFILE_START_STEP",
    "XTUNER_PROFILE_END_STEP",
    "XTUNER_PROFILE_SAVE_PATH",
    "XTUNER_PROFILE_DYNAMIC_CONFIG_PATH",
)


@pytest.fixture(autouse=True)
def _clean_profile_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every XTUNER_PROFILE_* env var so each test owns its switch state."""
    for key in _PROFILE_ENVS:
        monkeypatch.delenv(key, raising=False)


class TestProfilerV2Switches:
    """Defaults and v2-routing switch name for profiling_config_from_env."""

    def test_default_off_returns_none(self) -> None:
        assert profiling_config_from_env() is None

    def test_legacy_enable_name_does_not_route(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XTUNER_PROFILE_ENABLE", "1")
        assert profiling_config_from_env() is None

    def test_new_enable_name_routes_to_v2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XTUNER_NPU_PROFILE_V2_ENABLE", "1")
        cfg = profiling_config_from_env()
        assert cfg is not None
        assert cfg.enable is True

    def test_analyse_flag_defaults_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XTUNER_NPU_PROFILE_V2_ENABLE", "1")
        cfg = profiling_config_from_env()
        assert cfg is not None
        assert cfg.static_param.analyse_flag is False

    def test_analyse_flag_opt_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XTUNER_NPU_PROFILE_V2_ENABLE", "1")
        monkeypatch.setenv("XTUNER_PROFILE_ANALYSE_FLAG", "1")
        cfg = profiling_config_from_env()
        assert cfg is not None
        assert cfg.static_param.analyse_flag is True
