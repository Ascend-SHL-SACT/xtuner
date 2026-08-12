"""Regression test for the A2A_CACHE_SPLITS recompute-on gate.

``_a2a_split_cache`` is a LIFO stack: the forward push (in ``_dispatch``) is
popped by the backward recompute. If activation-checkpoint recompute is OFF, the
backward never re-enters ``_dispatch`` -> no pop -> the stack grows across
steps (unbounded leak). The fix gates ``use_cache`` on ``RECOMPUTE_RATIO > 0``
-- the same env the launch script exports and the example config feeds into
``fsdp_config.recompute_ratio`` (the value ``moe.py``'s ``_should_recompute``
reads to decide whether to wrap layers with ``checkpoint_wrapper``). With
recompute off (ratio 0, or env unset -> default 0) the cache never pushes, so
there is nothing to leak.

CPU-only: drives the gate directly via env vars (no process group, no NPU, no
``_dispatch`` call).
"""
import os


def _set(a2a, ratio):
    """Set the A2A + RECOMPUTE_RATIO envs, return the module.

    Pass None for either to pop it (test the unset-default path).
    """
    if a2a is None:
        os.environ.pop("XTUNER_MOE_A2A_CACHE_SPLITS", None)
    else:
        os.environ["XTUNER_MOE_A2A_CACHE_SPLITS"] = str(a2a)
    if ratio is None:
        os.environ.pop("RECOMPUTE_RATIO", None)
    else:
        os.environ["RECOMPUTE_RATIO"] = str(ratio)
    from xtuner.v1.module.dispatcher import torch_all2all as t

    return t


class TestA2aCacheRecomputeGate:
    def test_off_when_recompute_off_env_on(self):
        # The leak case the gate prevents: A2A on but recompute off (ratio 0) ->
        # no push, so the stack never accumulates (no leak).
        t = _set("1", "0")
        assert t._a2a_cache_enabled() is False

    def test_on_only_when_both_env_and_recompute_on(self):
        t = _set("1", "1.0")
        assert t._a2a_cache_enabled() is True

    def test_off_when_a2a_off_even_if_recompute_on(self):
        t = _set("0", "1.0")
        assert t._a2a_cache_enabled() is False

    def test_off_when_a2a_unset_even_if_recompute_on(self):
        t = _set(None, "1.0")
        assert t._a2a_cache_enabled() is False

    def test_off_when_recompute_ratio_unset(self):
        # Unset RECOMPUTE_RATIO defaults to 0 (safe: cache off, no leak).
        t = _set("1", None)
        assert t._a2a_cache_enabled() is False

    def test_off_when_recompute_ratio_garbage(self):
        # Non-numeric ratio -> treat as off (safe).
        t = _set("1", "auto")
        assert t._a2a_cache_enabled() is False
