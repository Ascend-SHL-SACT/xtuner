"""Completed microbatches detach scalar logs; offload cleanup runs at the next step's start."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from xtuner.v1.engine import train_engine
from xtuner.v1.model.base import ModelOutputs


class _Outputs(ModelOutputs):
    mtp_loss: torch.Tensor


@pytest.mark.parametrize("fail_second", [False, True])
def test_microbatch_loss_detach_preserves_accumulation(monkeypatch, fail_second):
    monkeypatch.setattr(train_engine, "DEVICE", "cpu")
    monkeypatch.setattr(train_engine, "ProberList", Mock())
    weight = torch.tensor(2.0, requires_grad=True)
    runtime_cache = []
    outputs = []
    clears = []

    def clear():
        # glm53f releases offload state at the START of a train step (#97/#110), so the
        # recorded gradient belongs to the previous step and the recorded cache length is
        # the previous step's leftover runtime entries.
        clears.append((weight.grad.item() if weight.grad is not None else None, len(runtime_cache)))
        runtime_cache.clear()

    monkeypatch.setattr(train_engine, "OffloadManager", lambda: SimpleNamespace(clear_step=clear))

    class Model:
        def pre_micro_batch_forward(self, batches):
            outputs.clear()
            return {}

        def __call__(self, **kwargs):
            assert len(runtime_cache) == len(outputs)
            if outputs:
                assert outputs[-1].loss.grad_fn is None
                assert outputs[-1].mtp_loss.grad_fn is None
            runtime_cache.append(object())
            if outputs and fail_second:
                raise RuntimeError("second forward failed")
            output = _Outputs(loss=weight.square(), mtp_loss=weight * 3)
            outputs.append(output)
            return output

        def post_micro_batch_forward(self, results):
            assert results == outputs
            assert len(runtime_cache) == len(outputs)
            return {}

    engine = object.__new__(train_engine.TrainEngine)
    engine.model = Model()
    engine.intra_layer_micro_batch = 1
    engine._count = 1
    engine._maybe_precompute_float8_dynamic_scale_for_fsdp = lambda: None
    batches = [{"seq_ctx": None, "loss_ctx": None}] * 2
    if fail_second:
        # The failed step leaves its leftovers behind; the next step's start-of-step clear
        # releases them while preserving the first microbatch's accumulated gradient —
        # the same cleanup coverage main gets from a try/finally at step exit.
        with pytest.raises(RuntimeError, match="second forward failed"):
            engine.train_step(batches)
        with pytest.raises(RuntimeError, match="second forward failed"):
            engine.train_step(batches)
        assert clears == [(None, 0), (7.0, 2)]
    else:
        result = engine.train_step(batches)
        assert result["total_loss"] == 20.0
        result = engine.train_step(batches)
        assert result["total_loss"] == 20.0
        assert clears == [(None, 0), (14.0, 2)]
    assert all(out.loss.item() == 4 and out.mtp_loss.item() == 6 for out in outputs)
    assert weight.grad.item() == (14.0 if fail_second else 28.0)
