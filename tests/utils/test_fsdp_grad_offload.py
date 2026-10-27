"""Tests for the FSDP fp32 gradient CPU offload manager (host-side accumulation)."""

import threading
from typing import Any

import pytest
import torch
from torch.distributed._composable_state import _insert_module_state
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup

from xtuner.v1.utils.fsdp_grad_offload import (
    FSDPGradOffloadManager,
    fsdp_grad_offload_enabled,
    install_fsdp_grad_offload,
)


class _FakeDTensor:
    """Duck-typed DTensor: the manager only touches ``_local_tensor``."""

    def __init__(self, local_tensor: torch.Tensor) -> None:
        self._local_tensor = local_tensor


class _FakeEvent:
    def synchronize(self) -> None:
        return None

    def query(self) -> bool:
        return True


class _BrokenEvent:
    def synchronize(self) -> None:
        raise ValueError("event broken")


class _GatedQueryEvent:
    """Event whose ``query()`` reports fired only after the test sets the gate."""

    def __init__(self, fired: threading.Event) -> None:
        self._fired = fired

    def query(self) -> bool:
        return self._fired.is_set()

    def synchronize(self) -> None:
        return None


class _FakeStreamEvent:
    """Stream whose per-copy event blocks synchronize() until the test sets the gate."""

    def __init__(self, gate: threading.Event) -> None:
        self._gate = gate

    def record_event(self) -> "_FakeStreamEvent":
        return self

    def synchronize(self) -> None:
        self._gate.wait()

    def query(self) -> bool:
        # The producer's inflight event must never block these tests.
        return True


class _FakeStream:
    def __init__(self, event: _FakeEvent | _BrokenEvent | None = None) -> None:
        self._event = event if event is not None else _FakeEvent()
        self.recorded = 0

    def record_event(self) -> _FakeEvent | _BrokenEvent:
        self.recorded += 1
        return self._event


class _FakeStreamCtx:
    def __enter__(self) -> _FakeStream | None:
        return None

    def __exit__(self, *args: object) -> bool:
        return False


class _RecordingCopyStream:
    """Duck-typed dedicated copy stream: records gate waits and per-chunk events."""

    def __init__(self, chunk_event: _FakeEvent | _BrokenEvent | None = None) -> None:
        self.waited: list[object] = []
        self.recorded = 0
        self._chunk_event = chunk_event if chunk_event is not None else _FakeEvent()

    def wait_event(self, event: object) -> None:
        self.waited.append(event)

    def record_event(self) -> _FakeEvent | _BrokenEvent:
        self.recorded += 1
        return self._chunk_event


class _FakeDeviceHandle:
    def __init__(self, chunk_event: _FakeEvent | _BrokenEvent | None = None) -> None:
        self.copy_stream = _RecordingCopyStream(chunk_event)
        self.entered: list[object] = []

    def Stream(self, device: object) -> _RecordingCopyStream:
        return self.copy_stream

    def stream(self, stream: object) -> _FakeStreamCtx:
        self.entered.append(stream)
        return _FakeStreamCtx()

    def current_stream(self) -> _FakeStream:
        return _FakeStream()


class _FakeModuleState:
    def __init__(self, param_group: FSDPParamGroup) -> None:
        self._fsdp_param_group = param_group


def _fake_param(params: list[object]) -> FSDPParamGroup:
    param_group = object.__new__(FSDPParamGroup)
    # Plain objects: a real Parameter would reject the duck-typed grad on assignment.
    fsdp_params: list[Any] = [type("FakeFSDPParam", (), {"sharded_param": param})() for param in params]
    param_group.fsdp_params = fsdp_params  # type: ignore[attr-defined]
    comm_ctx: Any = type("FakeCommCtx", (), {"reduce_scatter_stream": _FakeStream()})()
    param_group.comm_ctx = comm_ctx  # type: ignore[attr-defined]
    param_group.device_handle = _FakeDeviceHandle()  # type: ignore[attr-defined]
    return param_group


def _attach_param_group(module: torch.nn.Module, param_group: FSDPParamGroup) -> None:
    state = _FakeModuleState(param_group)
    # `_insert_module_state` stores a weak reference; the module must keep the state alive.
    module._fsdp_grad_offload_test_state = state  # type: ignore[attr-defined]
    _insert_module_state(module, state)


class TestFsdpGradOffloadGate:
    def test_default_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("XTUNER_FSDP_GRAD_OFFLOAD", raising=False)
        assert fsdp_grad_offload_enabled() is False

    def test_disabled_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for value in ("0", "false", "off", ""):
            monkeypatch.setenv("XTUNER_FSDP_GRAD_OFFLOAD", value)
            assert fsdp_grad_offload_enabled() is False, value

    def test_enabled_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for value in ("1", "true", "on", "TRUE"):
            monkeypatch.setenv("XTUNER_FSDP_GRAD_OFFLOAD", value)
            assert fsdp_grad_offload_enabled() is True, value

    def test_unknown_value_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XTUNER_FSDP_GRAD_OFFLOAD", "reduce")
        with pytest.raises(ValueError, match="XTUNER_FSDP_GRAD_OFFLOAD"):
            fsdp_grad_offload_enabled()


class TestFSDPGradOffloadManager:
    def test_offload_accumulates_across_microbatches(self) -> None:
        mgr = FSDPGradOffloadManager(enabled=True)
        p0 = type("FakeParam", (), {})()
        p1 = type("FakeParam", (), {})()
        pg = _fake_param([p0, p1])
        grad0 = _FakeDTensor(torch.arange(8, dtype=torch.float32))
        grad1 = _FakeDTensor(torch.ones(4, dtype=torch.float32))
        p0.grad = grad0  # type: ignore[attr-defined]
        p1.grad = grad1  # type: ignore[attr-defined]

        mgr.note_backward()
        mgr.maybe_offload_group(pg)

        assert p0.grad is None and p1.grad is None  # type: ignore[attr-defined]
        assert mgr._stash[p0] is grad0 and mgr._stash[p1] is grad1

        # second microbatch: fresh device grads are accumulated into the host buffers
        p0.grad = _FakeDTensor(torch.arange(8, dtype=torch.float32) * 2)  # type: ignore[attr-defined]
        p1.grad = _FakeDTensor(torch.ones(4, dtype=torch.float32) * 3)  # type: ignore[attr-defined]
        mgr.maybe_offload_group(pg)
        mgr.restore()

        assert p0.grad is grad0 and p1.grad is grad1  # type: ignore[attr-defined]
        assert torch.equal(grad0._local_tensor, torch.arange(8, dtype=torch.float32) * 3)
        assert torch.equal(grad1._local_tensor, torch.ones(4, dtype=torch.float32) * 4)
        assert not mgr._stash

    def test_restore_reattaches_original_shard_shape(self) -> None:
        # Regression (run76): restore re-attached the flat pinned accumulation buffer
        # directly, so the DTensor local came back 1-D where FSDP's shard was shaped.
        # clip_grad only norms each grad independently and passed, but the optimizer's
        # zeros_like(param)-shaped AdamW states then met a flat grad in the first
        # foreach lerp — aclnnInplaceLerps error 161002 on all 16 ranks. The
        # re-attached local must carry the original shard shape, values bit-identical.
        mgr = FSDPGradOffloadManager(enabled=True)
        param = type("FakeParam", (), {})()
        pg = _fake_param([param])
        shaped = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        grad = _FakeDTensor(shaped)
        param.grad = grad  # type: ignore[attr-defined]

        mgr.note_backward()
        mgr.maybe_offload_group(pg)
        assert param.grad is None  # type: ignore[attr-defined]
        mgr.restore()

        assert param.grad is grad  # type: ignore[attr-defined]
        local = grad._local_tensor  # type: ignore[attr-defined]
        assert local.shape == (3, 4)
        assert local.is_contiguous()
        assert torch.equal(local, shaped)

    def test_unarmed_group_is_noop(self) -> None:
        mgr = FSDPGradOffloadManager(enabled=True)
        param = type("FakeParam", (), {})()
        pg = _fake_param([param])
        grad = _FakeDTensor(torch.arange(8, dtype=torch.float32))
        param.grad = grad  # type: ignore[attr-defined]

        mgr.note_backward()
        mgr._armed = False
        mgr.maybe_offload_group(pg)

        assert param.grad is grad  # type: ignore[attr-defined]
        assert not mgr._stash

    def test_disabled_manager_is_noop(self) -> None:
        mgr = FSDPGradOffloadManager(enabled=False)
        param = type("FakeParam", (), {})()
        pg = _fake_param([param])
        grad = _FakeDTensor(torch.arange(8, dtype=torch.float32))
        param.grad = grad  # type: ignore[attr-defined]

        mgr.note_backward()
        mgr.maybe_offload_group(pg)

        assert param.grad is grad  # type: ignore[attr-defined]
        assert not mgr._stash

    def test_plain_tensor_grad_is_skipped(self) -> None:
        mgr = FSDPGradOffloadManager(enabled=True)
        param = type("FakeParam", (), {})()
        pg = _fake_param([param])
        param.grad = torch.arange(8, dtype=torch.float32)  # type: ignore[attr-defined]

        mgr.note_backward()
        mgr.maybe_offload_group(pg)

        assert param.grad is not None  # type: ignore[attr-defined]
        assert not mgr._stash

    def test_restore_is_noop_when_nothing_offloaded(self) -> None:
        mgr = FSDPGradOffloadManager(enabled=True)
        mgr.restore()
        assert not mgr._stash

    def test_begin_step_resets_accumulation(self) -> None:
        mgr = FSDPGradOffloadManager(enabled=True)
        param = type("FakeParam", (), {})()
        pg = _fake_param([param])
        param.grad = _FakeDTensor(torch.full((4,), 10.0))  # type: ignore[attr-defined]
        mgr.note_backward()
        mgr.maybe_offload_group(pg)
        mgr.restore()
        assert torch.equal(param.grad._local_tensor, torch.full((4,), 10.0))  # type: ignore[attr-defined]

        # An aborted step: the next step's first contribution must overwrite, not add.
        mgr.begin_step()
        param.grad = _FakeDTensor(torch.full((4,), 3.0))  # type: ignore[attr-defined]
        mgr.note_backward()
        mgr.maybe_offload_group(pg)
        mgr.restore()

        assert torch.equal(param.grad._local_tensor, torch.full((4,), 3.0))  # type: ignore[attr-defined]

    def test_restore_restarts_accumulation_for_next_step(self) -> None:
        # Consecutive steps call restore() but not begin_step(); the second step's first
        # contribution must overwrite the accumulation buffer, not add to it.
        mgr = FSDPGradOffloadManager(enabled=True)
        param = type("FakeParam", (), {})()
        pg = _fake_param([param])
        param.grad = _FakeDTensor(torch.full((4,), 10.0))  # type: ignore[attr-defined]
        mgr.note_backward()
        mgr.maybe_offload_group(pg)
        mgr.restore()
        assert torch.equal(param.grad._local_tensor, torch.full((4,), 10.0))  # type: ignore[attr-defined]

        param.grad = _FakeDTensor(torch.full((4,), 3.0))  # type: ignore[attr-defined]
        mgr.note_backward()
        mgr.maybe_offload_group(pg)
        mgr.restore()

        assert torch.equal(param.grad._local_tensor, torch.full((4,), 3.0))  # type: ignore[attr-defined]

    def test_begin_step_recovers_from_aborted_step(self) -> None:
        # An aborted step never reaches restore(); begin_step() must drop its stash so the
        # next step's first contribution overwrites the partial accumulation.
        mgr = FSDPGradOffloadManager(enabled=True)
        param = type("FakeParam", (), {})()
        pg = _fake_param([param])
        param.grad = _FakeDTensor(torch.full((4,), 10.0))  # type: ignore[attr-defined]
        mgr.note_backward()
        mgr.maybe_offload_group(pg)
        assert param.grad is None  # type: ignore[attr-defined]

        mgr.begin_step()
        param.grad = _FakeDTensor(torch.full((4,), 3.0))  # type: ignore[attr-defined]
        mgr.note_backward()
        mgr.maybe_offload_group(pg)
        mgr.restore()

        assert torch.equal(param.grad._local_tensor, torch.full((4,), 3.0))  # type: ignore[attr-defined]

    def test_worker_error_surfaces_at_restore(self) -> None:
        mgr = FSDPGradOffloadManager(enabled=True)
        param = type("FakeParam", (), {})()
        broken_pg = object.__new__(FSDPParamGroup)
        broken_pg.fsdp_params = [type("FakeFSDPParam", (), {"sharded_param": param})()]  # type: ignore[attr-defined]
        comm_ctx: Any = type("FakeCommCtx", (), {"reduce_scatter_stream": _FakeStream(_BrokenEvent())})()
        broken_pg.comm_ctx = comm_ctx  # type: ignore[attr-defined]
        # The per-chunk completion events come from the dedicated copy stream now, so the
        # breakage is injected there (the rs gate event is only waited on, never synced).
        broken_pg.device_handle = _FakeDeviceHandle(chunk_event=_BrokenEvent())  # type: ignore[attr-defined]
        param.grad = _FakeDTensor(torch.arange(8, dtype=torch.float32))  # type: ignore[attr-defined]

        mgr.note_backward()
        mgr.maybe_offload_group(broken_pg)

        with pytest.raises(RuntimeError, match="worker failed"):
            mgr.restore()

    def test_producer_never_touches_d2h_path_while_worker_wedged(self) -> None:
        # Regression (run69): producer-side waits on the D2H path (ring slots, copy
        # events) desynchronize this rank's backward progress from the other ranks',
        # and the misplaced collective rendezvous deadlocks HCCL across ranks. The
        # whole D2H path must live on the worker thread: with the worker wedged on a
        # copy event, the producer must still queue every task and return.
        mgr = FSDPGradOffloadManager(enabled=True)
        params = [type("FakeParam", (), {})() for _ in range(4)]
        pg = _fake_param(params)
        for index, param in enumerate(params):
            param.grad = _FakeDTensor(torch.full((8,), float(index)))  # type: ignore[attr-defined]
        gate = threading.Event()
        pg.comm_ctx = type(  # type: ignore[attr-defined]
            "FakeCommCtx", (), {"reduce_scatter_stream": _FakeStreamEvent(gate)}
        )()

        acquire_callers: list[str] = []
        orig_acquire = mgr._acquire_slot

        def _spy() -> int:
            acquire_callers.append(threading.current_thread().name)
            return orig_acquire()

        mgr._acquire_slot = _spy  # type: ignore[method-assign]
        mgr.note_backward()
        mgr.maybe_offload_group(pg)

        assert len(mgr._stash) == 4
        for param in params:
            assert param.grad is None  # type: ignore[attr-defined]

        gate.set()
        mgr.restore()
        for index, param in enumerate(params):
            assert torch.equal(param.grad._local_tensor, torch.full((8,), float(index)))  # type: ignore[attr-defined]
        assert acquire_callers and set(acquire_callers) == {"fsdp-grad-offload"}

    def test_d2h_runs_on_dedicated_copy_stream_gated_by_rs_event(self) -> None:
        # Regression (run73): the D2H used to be enqueued on the reduce-scatter stream
        # itself, so every chunk's completion event waited behind the whole collective
        # queue — effective bandwidth collapsed to ~16 MB/s, the producer throttle
        # lockstepped the backward ~15x, and the stretched backward OOMed the allocator.
        # Copies must instead run on the manager's dedicated copy stream, gated once per
        # contribution by the producing group's reduce-scatter event, with the gradient
        # storage released via record_stream on that copy stream.
        mgr = FSDPGradOffloadManager(enabled=True)
        param = type("FakeParam", (), {})()
        pg = _fake_param([param])
        grad = _FakeDTensor(torch.arange(8, dtype=torch.float32))
        source = grad._local_tensor
        param.grad = grad  # type: ignore[attr-defined]

        releases: list[torch.Tensor] = []
        orig_release = mgr._release_device_source

        def _spy(local: torch.Tensor) -> None:
            releases.append(local)
            orig_release(local)

        mgr._release_device_source = _spy  # type: ignore[method-assign]
        mgr.note_backward()
        mgr.maybe_offload_group(pg)
        gate = mgr._outstanding[0][0]
        mgr.restore()

        handle = pg.device_handle  # type: ignore[attr-defined]
        rs_stream = pg.comm_ctx.reduce_scatter_stream  # type: ignore[attr-defined]
        # One gate event per contribution on the reduce-scatter stream: the old code also
        # recorded a per-chunk completion event there, one per 32 MiB chunk.
        assert rs_stream.recorded == 1
        assert mgr._copy_stream is handle.copy_stream
        # The dedicated copy stream is the only stream the worker ever enters; the old
        # code entered the reduce-scatter stream instead.
        assert handle.entered and set(handle.entered) == {handle.copy_stream}
        # The gate is waited exactly once per contribution (not per chunk), and the
        # per-chunk completion events come from the copy stream.
        assert handle.copy_stream.waited == [gate]
        assert handle.copy_stream.recorded == 1
        # The device source went through the release path instead of staying
        # task-referenced until its copy finished executing.
        assert len(releases) == 1 and releases[0] is source


class TestInstallFsdpGradOffload:
    def test_wraps_post_backward_once(self) -> None:
        module = torch.nn.Linear(2, 2)
        mgr = FSDPGradOffloadManager(enabled=True)
        pg = _fake_param([type("FakeParam", (), {})()])
        _attach_param_group(module, pg)

        assert install_fsdp_grad_offload(module, manager=mgr) == 1
        assert pg.__dict__.get("post_backward") is not None
        assert install_fsdp_grad_offload(module, manager=mgr) == 0

    def test_disabled_install_is_noop(self) -> None:
        module = torch.nn.Linear(2, 2)
        mgr = FSDPGradOffloadManager(enabled=False)
        pg = _fake_param([type("FakeParam", (), {})()])
        _attach_param_group(module, pg)

        assert install_fsdp_grad_offload(module, manager=mgr) == 0
        assert pg.__dict__.get("post_backward") is None

    def test_wrapped_post_backward_offloads_while_armed(self) -> None:
        module = torch.nn.Linear(2, 2)
        mgr = FSDPGradOffloadManager(enabled=True)
        param = type("FakeParam", (), {})()
        pg = _fake_param([param])
        calls: list[int] = []
        # Replace the group's post_backward with a stub: the wrapper must still
        # call it and then feed the manager (the real post_backward needs a fully
        # initialized FSDPParamGroup that a fake cannot provide).
        pg.post_backward = lambda *args, **kwargs: calls.append(1)  # type: ignore[method-assign]
        _attach_param_group(module, pg)

        assert install_fsdp_grad_offload(module, manager=mgr) == 1

        mgr.note_backward()
        grad = _FakeDTensor(torch.arange(8, dtype=torch.float32))
        param.grad = grad  # type: ignore[attr-defined]
        pg.post_backward()

        assert calls == [1]
        assert param.grad is None  # type: ignore[attr-defined]
        assert mgr._stash[param] is grad

        mgr.restore()
        assert param.grad is grad  # type: ignore[attr-defined]


class TestFsdpGradOffloadBudget:
    """The host budget selects a subset; the rest keeps device accumulation."""

    def _mgr(self, budget_bytes: int) -> FSDPGradOffloadManager:
        # budget_gib is a float in GiB, so tests express byte budgets as fractions of 2**30.
        return FSDPGradOffloadManager(enabled=True, budget_gib=budget_bytes / 2**30)

    def test_budget_keeps_oversized_parameter_on_device(self) -> None:
        mgr = self._mgr(32)  # exactly one 8-float fp32 shard
        p_big = type("FakeParam", (), {})()
        p_small = type("FakeParam", (), {})()
        pg = _fake_param([p_big, p_small])
        big_grad = _FakeDTensor(torch.arange(8, dtype=torch.float32))
        small_grad = _FakeDTensor(torch.arange(4, dtype=torch.float32))
        p_big.grad = big_grad  # type: ignore[attr-defined]
        p_small.grad = small_grad  # type: ignore[attr-defined]

        mgr.note_backward()
        mgr.maybe_offload_group(pg)

        assert p_big.grad is None  # type: ignore[attr-defined]
        assert mgr._stash[p_big] is big_grad
        # The 4-float shard does not fit in the remaining budget: it stays on device.
        assert p_small.grad is small_grad  # type: ignore[attr-defined]
        assert p_small not in mgr._stash
        assert p_small in mgr._skipped_params
        assert mgr._offload_bytes == 32
        assert mgr.budget_bytes == 32

    def test_skipped_parameter_keeps_accumulating_on_device(self) -> None:
        mgr = self._mgr(0)  # unlimited budget for the first shard is not what is tested here
        mgr._budget_gib = 32 / 2**30
        p_big = type("FakeParam", (), {})()
        p_small = type("FakeParam", (), {})()
        pg = _fake_param([p_big, p_small])
        p_big.grad = _FakeDTensor(torch.arange(8, dtype=torch.float32))  # type: ignore[attr-defined]
        p_small.grad = _FakeDTensor(torch.arange(4, dtype=torch.float32))  # type: ignore[attr-defined]

        mgr.note_backward()
        mgr.maybe_offload_group(pg)

        second_small = _FakeDTensor(torch.full((4,), 5.0))
        p_big.grad = _FakeDTensor(torch.arange(8, dtype=torch.float32) * 2)  # type: ignore[attr-defined]
        p_small.grad = second_small  # type: ignore[attr-defined]
        mgr.maybe_offload_group(pg)

        # The skipped parameter is never taken over: its newest device gradient is intact.
        assert p_small.grad is second_small  # type: ignore[attr-defined]
        assert p_small not in mgr._stash

    def test_chunked_ring_reproduces_exact_accumulation(self) -> None:
        mgr = FSDPGradOffloadManager(enabled=True)
        mgr._ring_chunk_bytes = 16  # 4 fp32 elements per ring slot -> 4 chunks for 16 elements
        param = type("FakeParam", (), {})()
        pg = _fake_param([param])
        first = torch.arange(16, dtype=torch.float32)
        param.grad = _FakeDTensor(first)  # type: ignore[attr-defined]

        mgr.note_backward()
        mgr.maybe_offload_group(pg)
        for scale in (2.0, 3.0):
            param.grad = _FakeDTensor(torch.arange(16, dtype=torch.float32) * scale)  # type: ignore[attr-defined]
            mgr.maybe_offload_group(pg)
        mgr.restore()

        # 1 + 2 + 3 = 6 per element, exactly what device-side accumulation would produce.
        expected = torch.arange(16, dtype=torch.float32) * 6
        assert torch.equal(param.grad._local_tensor, expected)  # type: ignore[attr-defined]
        assert len(mgr._ring) == 8
        assert all(release.is_set() for release in mgr._ring_release)

    def test_ring_slots_are_released_after_each_contribution(self) -> None:
        mgr = FSDPGradOffloadManager(enabled=True)
        mgr._ring_chunk_bytes = 32  # 8 fp32 elements per slot
        params = [type("FakeParam", (), {})() for _ in range(12)]
        pg = _fake_param(params)
        for index, param in enumerate(params):
            param.grad = _FakeDTensor(torch.full((8,), float(index)))  # type: ignore[attr-defined]

        mgr.note_backward()
        mgr.maybe_offload_group(pg)  # 12 tasks > 8 slots: reuse must not deadlock
        mgr.restore()

        for index, param in enumerate(params):
            assert torch.equal(param.grad._local_tensor, torch.full((8,), float(index)))  # type: ignore[attr-defined]

    def test_ring_slots_sized_by_chunk_bytes_not_first_param(self) -> None:
        # Regression (run71-75): the ring was one-shot sized from the FIRST selected
        # parameter, so a tiny first param (norm weight, 16 B) shrank every slot to the
        # 4 KiB floor — a later 605 MiB parameter then copied in ~155k chunks paying the
        # ~200 us event sync per chunk (~30 s per parameter), stretching the backward
        # until the queued tasks' reduce-output storages OOMed the allocator. Slots must
        # always be _RING_CHUNK_BYTES, whatever the first selected parameter's size.
        mgr = FSDPGradOffloadManager(enabled=True)
        p_tiny = type("FakeParam", (), {})()
        p_big = type("FakeParam", (), {})()
        pg = _fake_param([p_tiny, p_big])
        p_tiny.grad = _FakeDTensor(torch.arange(4, dtype=torch.float32))  # type: ignore[attr-defined]
        p_big.grad = _FakeDTensor(torch.arange(1024 * 1024, dtype=torch.float32))  # type: ignore[attr-defined]

        mgr.note_backward()
        mgr.maybe_offload_group(pg)
        mgr.restore()

        assert mgr._ring_slot_bytes == mgr._ring_chunk_bytes
        expected = torch.arange(1024 * 1024, dtype=torch.float32)
        assert torch.equal(p_big.grad._local_tensor, expected)  # type: ignore[attr-defined]

    def test_single_param_with_more_chunks_than_ring_slots(self) -> None:
        # Regression (run65-68): one task per contribution deadlocked when a parameter
        # needed more chunks than the ring has slots — the producer blocked in
        # _acquire_slot on a slot only the worker can release, while the worker had no
        # task to consume because the queue was only filled after the enqueue loop.
        # GLM-5.3-Flash hits this on step-1 backward: its first sharded parameter alone
        # spans >8 x 32 MiB chunks.
        mgr = FSDPGradOffloadManager(enabled=True)
        mgr._ring_chunk_bytes = 16  # 4 fp32 elements per slot -> 16 chunks for 64 elements
        param = type("FakeParam", (), {})()
        pg = _fake_param([param])
        param.grad = _FakeDTensor(torch.arange(64, dtype=torch.float32))  # type: ignore[attr-defined]

        mgr.note_backward()
        mgr.maybe_offload_group(pg)
        param.grad = _FakeDTensor(torch.arange(64, dtype=torch.float32) * 2)  # type: ignore[attr-defined]
        mgr.maybe_offload_group(pg)
        mgr.restore()

        expected = torch.arange(64, dtype=torch.float32) * 3
        assert torch.equal(param.grad._local_tensor, expected)  # type: ignore[attr-defined]
        assert all(release.is_set() for release in mgr._ring_release)

    def test_non_contiguous_shard_stays_on_device(self) -> None:
        mgr = FSDPGradOffloadManager(enabled=True)
        param = type("FakeParam", (), {})()
        pg = _fake_param([param])
        base = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        param.grad = _FakeDTensor(base.t())  # type: ignore[attr-defined]  # non-contiguous view

        mgr.note_backward()
        mgr.maybe_offload_group(pg)

        assert param.grad is not None  # type: ignore[attr-defined]
        assert param in mgr._skipped_params
        assert not mgr._stash

    def test_budget_env_unset_means_unlimited(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("XTUNER_FSDP_GRAD_OFFLOAD_BUDGET_GIB", raising=False)
        assert FSDPGradOffloadManager(enabled=True).budget_bytes == 0

    def test_budget_env_reads_gib(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XTUNER_FSDP_GRAD_OFFLOAD_BUDGET_GIB", "3")
        assert FSDPGradOffloadManager(enabled=True).budget_bytes == 3 * 2**30

    def test_budget_env_rejects_negative(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XTUNER_FSDP_GRAD_OFFLOAD_BUDGET_GIB", "-1")
        with pytest.raises(ValueError, match="XTUNER_FSDP_GRAD_OFFLOAD_BUDGET_GIB"):
            _ = FSDPGradOffloadManager(enabled=True).budget_bytes


class TestFSDPGradOffloadInflightThrottle:
    def test_group_event_recorded_and_cleared_by_restore(self) -> None:
        # One selected parameter pins its whole group's reduce-scatter output storage,
        # so the inflight accounting charges that buffer's storage size (per-param
        # padding included) — not a per-shard numel sum.
        mgr = FSDPGradOffloadManager(enabled=True)
        p0 = type("FakeParam", (), {})()
        p1 = type("FakeParam", (), {})()
        pg = _fake_param([p0, p1])
        p0.grad = _FakeDTensor(torch.arange(8, dtype=torch.float32))  # type: ignore[attr-defined]
        p1.grad = _FakeDTensor(torch.ones(4, dtype=torch.float32))  # type: ignore[attr-defined]

        mgr.note_backward()
        mgr.maybe_offload_group(pg)

        # In the fake the shards do not share a buffer, so the charge is the first
        # shard's own storage; in FSDP it is the whole group's padded reduce output.
        assert mgr._inflight_bytes == 8 * 4
        assert len(mgr._outstanding) == 1
        mgr.restore()
        assert not mgr._outstanding and mgr._inflight_bytes == 0

    def test_non_selected_group_is_not_accounted(self) -> None:
        # A group whose parameters all stay on device frees its output at post-backward
        # exactly as the baseline does — it must not be throttled or accounted.
        mgr = FSDPGradOffloadManager(enabled=True)
        param = type("FakeParam", (), {})()
        pg = _fake_param([param])
        param.grad = _FakeDTensor(torch.arange(8, dtype=torch.float32))  # type: ignore[attr-defined]
        mgr._budget_gib = 0.0  # type: ignore[attr-defined]  # unlimited budget: offload everything

        mgr.note_backward()
        mgr.maybe_offload_group(pg)
        mgr.restore()

        mgr.begin_step()
        param.grad = _FakeDTensor(torch.arange(8, dtype=torch.float32))  # type: ignore[attr-defined]
        mgr._offload_bytes = 0  # type: ignore[attr-defined]
        mgr._acc.clear()
        mgr._budget_gib = 2**-30  # type: ignore[attr-defined]  # budget below one param: skip
        mgr.maybe_offload_group(pg)

        assert param.grad is not None  # type: ignore[attr-defined]
        assert not mgr._outstanding and mgr._inflight_bytes == 0

    def test_producer_throttles_until_oldest_event_fires(self) -> None:
        # Regression (run71): an un-throttled producer out-runs the reduce-scatter
        # stream's queue, and the task-held gradient storage pins the group outputs on
        # device until their copies EXECUTE — OOMing the SFA-grad workspace. The
        # producer must sleep-poll the oldest outstanding event (never a blocking wait)
        # until the inflight bytes are under the cap, releasing them FIFO.
        mgr = FSDPGradOffloadManager(enabled=True)
        mgr._max_inflight_bytes = 16  # type: ignore[attr-defined]
        gate = threading.Event()
        pg1 = _fake_param([type("FakeParam", (), {})()])
        pg1.comm_ctx.reduce_scatter_stream = _FakeStream(_GatedQueryEvent(gate))  # type: ignore[attr-defined]
        param1 = pg1.fsdp_params[0].sharded_param
        param1.grad = _FakeDTensor(torch.arange(8, dtype=torch.float32))  # 32 bytes in flight

        mgr.note_backward()
        mgr.maybe_offload_group(pg1)
        assert mgr._inflight_bytes == 32

        # Second group: the throttle must block until group 1's event fires...
        pg2 = _fake_param([type("FakeParam", (), {})()])
        param2 = pg2.fsdp_params[0].sharded_param
        param2.grad = _FakeDTensor(torch.ones(4, dtype=torch.float32))  # 16 bytes
        done = threading.Event()

        def queue_group2() -> None:
            mgr.maybe_offload_group(pg2)
            done.set()

        worker = threading.Thread(target=queue_group2)
        worker.start()
        assert not done.wait(0.05), "producer must not queue past the inflight cap"

        # ...and proceed FIFO once it fires, leaving only group 2's bytes in flight.
        gate.set()
        assert done.wait(10)
        assert mgr._outstanding[0][1] == 16 and mgr._inflight_bytes == 16
        worker.join()

    def test_throttle_timeout_raises(self) -> None:
        # A genuinely wedged stream must fail loudly instead of parking the autograd
        # thread forever.
        mgr = FSDPGradOffloadManager(enabled=True)
        mgr._max_inflight_bytes = 1  # type: ignore[attr-defined]
        mgr._inflight_timeout_s = 0.05  # type: ignore[attr-defined]
        pg1 = _fake_param([type("FakeParam", (), {})()])
        gated = _GatedQueryEvent(threading.Event())
        pg1.comm_ctx.reduce_scatter_stream = _FakeStream(gated)  # type: ignore[attr-defined]
        pg1.fsdp_params[0].sharded_param.grad = _FakeDTensor(torch.arange(8, dtype=torch.float32))

        mgr.note_backward()
        mgr.maybe_offload_group(pg1)

        pg2 = _fake_param([type("FakeParam", (), {})()])
        pg2.fsdp_params[0].sharded_param.grad = _FakeDTensor(torch.ones(4, dtype=torch.float32))
        with pytest.raises(RuntimeError, match="inflight throttle timed out"):
            mgr.maybe_offload_group(pg2)
