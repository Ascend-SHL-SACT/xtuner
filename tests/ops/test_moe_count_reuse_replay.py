# Copyright (c) OpenMMLab. All rights reserved.
"""``XTUNER_MOE_COUNT_REUSE_REPLAY``: gate semantics and replay-reuse wiring.

Pure CPU. ``_compute_counts`` itself needs a device and a process group, so the wiring is
checked against a fake count function while the checkpoint frame is driven through the real
``ContextVar`` that ``reuse_during_recompute`` reads.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast

import pytest
import torch
import torch.distributed as dist

from xtuner.v1.model.utils import checkpointing
from xtuner.v1.ops.moe.npu import fused_a2a_gmm


FAKE_SEND = [1, 2, 0]
FAKE_RECV = [0, 1, 2]
FAKE_GROUP = [3]
NO_TOKENS = cast(torch.Tensor, None)
NO_GROUP = cast(dist.ProcessGroup, None)


class TestCountReuseGate:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            (None, False),
            ("", False),
            ("0", False),
            ("1", True),
            ("2", False),
            ("true", False),
            (" 1 ", False),
        ],
    )
    def test_env_semantics(self, monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: bool) -> None:
        if raw is None:
            monkeypatch.delenv("XTUNER_MOE_COUNT_REUSE_REPLAY", raising=False)
        else:
            monkeypatch.setenv("XTUNER_MOE_COUNT_REUSE_REPLAY", raw)
        assert fused_a2a_gmm.count_reuse_on_replay() is expected

    def test_router_debug_disables_reuse(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XTUNER_MOE_COUNT_REUSE_REPLAY", "1")
        monkeypatch.setenv("XTUNER_ROUTER_DEBUG", "true")
        assert fused_a2a_gmm.count_reuse_on_replay() is False


@contextmanager
def _frame(frame: checkpointing._ActivationCheckpointFrame, is_replay: bool) -> Iterator[None]:
    token = checkpointing._CURRENT_ACTIVATION_CHECKPOINT.set((frame, is_replay))
    try:
        yield
    finally:
        checkpointing._CURRENT_ACTIVATION_CHECKPOINT.reset(token)


class TestCountReuseOnReplay:
    @staticmethod
    def _patch_counts(monkeypatch: pytest.MonkeyPatch, calls: list[int], sends: list[list[int]] | None = None) -> None:
        payloads = sends or [FAKE_SEND]

        def fake_counts(topk_ids: object, n_routed_experts: int, ep_group: object) -> tuple[list[int], ...]:
            calls.append(1)
            return payloads[(len(calls) - 1) % len(payloads)], FAKE_RECV, FAKE_GROUP

        monkeypatch.setattr(fused_a2a_gmm, "_compute_counts", fake_counts)

    def test_gate_off_recomputes_on_every_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("XTUNER_MOE_COUNT_REUSE_REPLAY", raising=False)
        calls: list[int] = []
        self._patch_counts(monkeypatch, calls)

        frame = checkpointing._ActivationCheckpointFrame()
        with _frame(frame, False):
            fused_a2a_gmm._compute_counts_for_call(NO_TOKENS, 4, NO_GROUP)
        with _frame(checkpointing._ActivationCheckpointFrame(), True):
            fused_a2a_gmm._compute_counts_for_call(NO_TOKENS, 4, NO_GROUP)

        assert len(calls) == 2
        assert not frame.outputs

    def test_gate_on_reuses_forward_counts_during_replay(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XTUNER_MOE_COUNT_REUSE_REPLAY", "1")
        calls: list[int] = []
        self._patch_counts(monkeypatch, calls)

        frame = checkpointing._ActivationCheckpointFrame()
        with _frame(frame, False):
            forward = fused_a2a_gmm._compute_counts_for_call(NO_TOKENS, 4, NO_GROUP)
        with _frame(frame, True):
            replayed = fused_a2a_gmm._compute_counts_for_call(NO_TOKENS, 4, NO_GROUP)

        assert len(calls) == 1
        assert replayed == forward
        assert replayed[0] is FAKE_SEND
        frame.assert_consumed()

    def test_gate_on_replays_multi_save_frame_in_fifo_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XTUNER_MOE_COUNT_REUSE_REPLAY", "1")
        calls: list[int] = []
        sends = [FAKE_SEND, [4, 5, 6]]
        self._patch_counts(monkeypatch, calls, sends)

        frame = checkpointing._ActivationCheckpointFrame()
        with _frame(frame, False):
            forward = [fused_a2a_gmm._compute_counts_for_call(NO_TOKENS, 4, NO_GROUP) for _ in range(2)]
        with _frame(frame, True):
            replayed = [fused_a2a_gmm._compute_counts_for_call(NO_TOKENS, 4, NO_GROUP) for _ in range(2)]

        assert len(calls) == 2
        assert [item[0] for item in replayed] == [item[0] for item in forward] == sends
        frame.assert_consumed()

    def test_gate_on_without_checkpoint_calls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XTUNER_MOE_COUNT_REUSE_REPLAY", "1")
        calls: list[int] = []
        self._patch_counts(monkeypatch, calls)

        assert fused_a2a_gmm._compute_counts_for_call(NO_TOKENS, 4, NO_GROUP) == (FAKE_SEND, FAKE_RECV, FAKE_GROUP)
        assert len(calls) == 1

    def test_gate_on_replay_without_forward_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XTUNER_MOE_COUNT_REUSE_REPLAY", "1")
        calls: list[int] = []
        self._patch_counts(monkeypatch, calls)

        with _frame(checkpointing._ActivationCheckpointFrame(), True):
            with pytest.raises(RuntimeError, match="no matching reusable output"):
                fused_a2a_gmm._compute_counts_for_call(NO_TOKENS, 4, NO_GROUP)
        assert not calls

    def test_gate_on_extra_replay_pops_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XTUNER_MOE_COUNT_REUSE_REPLAY", "1")
        calls: list[int] = []
        self._patch_counts(monkeypatch, calls)

        frame = checkpointing._ActivationCheckpointFrame()
        with _frame(frame, False):
            fused_a2a_gmm._compute_counts_for_call(NO_TOKENS, 4, NO_GROUP)
        with _frame(frame, True):
            fused_a2a_gmm._compute_counts_for_call(NO_TOKENS, 4, NO_GROUP)
            with pytest.raises(RuntimeError, match="no matching reusable output"):
                fused_a2a_gmm._compute_counts_for_call(NO_TOKENS, 4, NO_GROUP)
        assert len(calls) == 1
