"""FSDP sharded-gradient CPU offload with host-side accumulation (fp32).

After every backward of a train step, each FSDP parameter group's freshly reduced fp32
sharded gradients are copied D2H in chunks through a small pinned ring on the FSDP
reduce-scatter stream and released from device, and a worker thread accumulates them into
pinned per-parameter buffers, replaying the same fp32 adds — same order, same operands —
that the device-side accumulation would have performed. Offloaded gradients therefore
never reside on device between microbatches, which removes their sharded-gradient
footprint from the per-step memory peak (13.5-13.9 GiB per rank on GLM-5.3-Flash, 16 dies;
~11.45 GiB of sharded fp32 gradients plus group workspaces). Before clipping,
:meth:`FSDPGradOffloadManager.restore` drains the worker and re-attaches the accumulated
device gradients. Numerical equivalence: with a single microbatch per train step
(``intra_layer_micro_batch == 1``) the path is pure copying and loss, grad norm, clipping,
and optimizer updates match the non-offloaded baseline bit for bit; with more microbatches
the host replays the device accumulation's fp32 adds in the same order, which is
structurally equivalent but not probed bit-identical.

The offloaded set is a **budget**, not necessarily everything: ``XTUNER_FSDP_GRAD_OFFLOAD_BUDGET_GIB``
caps the pinned host bytes per rank (unset or 0 = offload every group). Offloading costs
host memory 1:1 against device memory, and the realized device saving is capped by the
restore: the whole offloaded set is copied back before clipping, so it must fit in the
activation headroom that backward leaves behind (~2.8 GiB per rank measured on
GLM-5.3-Flash, 128K, 16 dies). Host memory is a hard, unswappable budget — pinned pages
show up in no meminfo bucket and in no RSS field — so the budget gate is the only safe way
to bound it. Each offloaded parameter owns exactly one pinned accumulation buffer; the D2H
path itself goes through a shared ring of ``_RING_SLOTS`` chunks, so pinned bytes stay at
``budget + _RING_SLOTS * _RING_CHUNK_BYTES`` instead of the former two full buffers per
parameter.

The D2H copies run on a **dedicated copy stream** owned by the manager, never on the
reduce-scatter stream: queueing them there makes every chunk's completion event wait
behind the whole collective queue — measured ~16 MB/s effective (run73) — which
lockstepped the producer throttle, stretched the step-1 backward ~15x, and OOMed the
allocator. Ordering is an event gate instead (programming-requirements 2.2/3.1): the
producer records the group's reduce-scatter event right after FSDP enqueues it, and each
task's copies wait that event once on the copy stream — never a stream-level wait on
rs_stream, and never a redundant second wait. Releasing the gradient storage makes no
stream wait either: the storage is allocated on the reduce-scatter stream but read by
the copy stream, so the worker calls ``record_stream(copy_stream)`` and drops the task's
reference as soon as the task's copies are issued — the allocator defers reuse of the
block until the copies execute (probe-validated pattern, programming-requirements 2.4).
Measured pinned bandwidth on this host is ~41 GB/s D2H and ~58 GB/s H2D, so the ~0.28 s
per-microbatch D2H overlaps the next microbatch's compute, leaving only the pre-clip
restore H2D (~0.2 s) on the serial path. Stream/buffer ownership: ring slots are written
only by the copy stream and read by the worker only after the chunk's event fires; a
slot is re-armed (host-side ``threading.Event``) only after the worker consumed it and
the copy stream is FIFO, so a re-staged slot can never overwrite an unconsumed copy and
no cross-step device event is needed. Accumulation buffers are written host-side and
read by the restore H2D only after ``_drain()``. HSDP and custom all-reduce-hook groups
materialize gradients on other streams and are left untouched.
The producer additionally throttles **itself**: a queued task holds its group's output
storage until the worker has issued its copy, and an un-throttled producer out-ran the
device by tens of seconds (run71: the pinned-alive set grew to GiBs on ranks whose
backward ran ahead, and the SparseFlashAttentionGrad workspace OOMed on 4 of 16 ranks
while the OFF baseline cleared the same op with <1 GiB to spare). Before queueing a
group the producer sleep-polls the oldest outstanding group event (a non-blocking
``query()``) until the not-yet-executed bytes are under ``_MAX_INFLIGHT_BYTES``. The
wait target is always inside this rank's own already-enqueued prefix — the oldest event
is at most the cap behind the current group, and every rank enqueues groups in the same
order — so the minimum-enqueue-position rank's wait is always already satisfiable and
the throttle cannot participate in a cross-rank rendezvous deadlock, unlike any wait on
copy/slot completion (run65-70). This module keeps its own copy stream instead of
sharing the activation-offload's: that stream lives inside the activation-offload
manager (which may be disabled entirely), and this path's traffic is budget-capped at a
few GiB per step — tens of milliseconds against a ~40 s step — so a second copy
stream's arbitration impact is negligible (programming-requirements 4.4).

Enable with ``XTUNER_FSDP_GRAD_OFFLOAD=1``. Set ``XTUNER_FSDP_GRAD_OFFLOAD_DEBUG=1`` for all-rank
diagnostic logging (device memory at budget-reached / pinned-growth / restore, throttle parks, and
per-parameter first-copy timings; run73 forensics).

Not covered: FSDP groups created outside ``BaseModel._fully_shard`` (e.g. the
``XTUNER_FSDP_FUSE_K`` grouped sharding path) keep device gradients; that stays correct
and only reduces the saving for those groups.
"""

import os
import threading
import time
import traceback
from collections import deque
from queue import SimpleQueue
from typing import Any

import torch
from mmengine.dist import get_rank
from torch.distributed._composable_state import _get_module_state
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
from torch.nn import Parameter

from xtuner.v1.utils import get_logger, log_rank0


_GRAD_OFFLOAD_ENV = "XTUNER_FSDP_GRAD_OFFLOAD"
_BUDGET_ENV = "XTUNER_FSDP_GRAD_OFFLOAD_BUDGET_GIB"
_DEBUG_ENV = "XTUNER_FSDP_GRAD_OFFLOAD_DEBUG"
_ENABLED_VALUES = ("1", "true", "on")
_DISABLED_VALUES = ("", "0", "false", "off")
_RING_CHUNK_BYTES = 32 << 20
_RING_SLOTS = 8
# Ceiling on the selected groups' reduce-scatter output bytes that are enqueued but not
# yet EXECUTED (see the throttle paragraph in the module docstring). A queued task holds
# its group's output storage until the worker has issued its copy, so the accounting uses
# the full group size; 256 MiB ≈ one average GLM-5.3-Flash FSDP group keeps the pile-up
# far below the baseline's own SFA-grad margin while leaving the device a multi-group
# runway.
_MAX_INFLIGHT_BYTES = 256 << 20
_INFLIGHT_POLL_S = 0.001
_INFLIGHT_TIMEOUT_S = 600.0


def _env_flag(env: str) -> bool:
    value = os.environ.get(env, "").strip().lower()
    if value in _ENABLED_VALUES:
        return True
    if value in _DISABLED_VALUES:
        return False
    raise ValueError(f"{env} must be one of {_ENABLED_VALUES + _DISABLED_VALUES}, got {value!r}")


def fsdp_grad_offload_enabled() -> bool:
    """Whether the FSDP fp32 gradient CPU offload is enabled.

    Returns:
        bool: True when ``XTUNER_FSDP_GRAD_OFFLOAD`` selects an enabled value.
    """
    return _env_flag(_GRAD_OFFLOAD_ENV)


def _budget_gib_from_env() -> float:
    """Read the per-rank pinned-host budget for the offloaded gradient set.

    Returns:
        float: Budget in GiB; 0.0 means "offload every group" (no cap).
    """
    raw = os.environ.get(_BUDGET_ENV, "").strip()
    if not raw:
        return 0.0
    value = float(raw)
    if value < 0:
        raise ValueError(f"{_BUDGET_ENV} must be >= 0, got {raw!r}")
    return value


# One worker task per contribution: (param, local device gradient, first_contribution,
# contribution-done event, group reduce-scatter gate event). The worker owns the whole
# D2H path — slot acquisition, gated copies on the dedicated copy stream, and
# accumulation — because the producer (the autograd backward thread) must never block on
# anything that depends on device streams: a blocked producer desynchronizes the per-rank
# backward progress, and the resulting collective rendezvous mismatch deadlocks the HCCL
# channels across ranks (run69).
_Task = tuple[Parameter, torch.Tensor, bool, threading.Event, Any]


class FSDPGradOffloadManager:
    """Offload FSDP sharded fp32 gradients to pinned host memory and accumulate them there.

    The engine arms the manager before every backward of a train step via
    :meth:`note_backward`. While armed, the ``FSDPParamGroup.post_backward`` wrappers
    installed by :func:`install_fsdp_grad_offload` hand every freshly reduced sharded
    gradient to :meth:`maybe_offload_group`, which copies it D2H in chunks through the
    shared pinned ring on the manager's dedicated copy stream (gated by the producing
    group's reduce-scatter event), releases the device storage via ``record_stream``, and
    hands the task to a worker thread that accumulates it into the parameter's pinned
    accumulation buffer. The accumulation replays the same fp32 adds, in the same order,
    the device-side accumulation would have performed. :meth:`restore` drains the worker
    and re-attaches the accumulated device gradients before clipping. Parameters outside
    the host budget keep their device gradients and accumulate exactly as the baseline does.

    Args:
        enabled (bool | None): Force the manager on or off; None resolves the env gate lazily.
        budget_gib (float | None): Per-rank pinned-host budget for the offloaded gradient set;
            None resolves ``XTUNER_FSDP_GRAD_OFFLOAD_BUDGET_GIB`` lazily (0 = offload every group).
    """

    def __init__(self, enabled: bool | None = None, budget_gib: float | None = None) -> None:
        self._enabled = enabled
        self._armed = False
        self._device: torch.device | None = None
        self._device_handle: Any = None
        self._copy_stream: Any = None
        self._acc: dict[Parameter, torch.Tensor] = {}
        # Original sharded-grad shapes: the pinned accumulation buffers are flat, but the
        # re-attached DTensor local must carry the FSDP shard shape again — downstream
        # (optimizer states via zeros_like(param), foreach ops) pairs the grad with
        # shape-matched tensors, and a flat local dies in the first foreach lerp with
        # aclnn error 161002 on every rank (run76).
        self._acc_shapes: dict[Parameter, torch.Size] = {}
        self._stash: dict[Parameter, Any] = {}
        self._inflight: dict[Parameter, threading.Event] = {}
        self._tasks: SimpleQueue[_Task] = SimpleQueue()
        self._worker: threading.Thread | None = None
        self._wrapped_groups: set[int] = set()
        self._worker_error: Exception | None = None
        self._pinned_bytes = 0
        self._pinned_logged_bytes = 0
        self._budget_gib = budget_gib
        self._offload_bytes = 0
        self._budget_logged = False
        self._skipped_params: set[Parameter] = set()
        self._ring_chunk_bytes = _RING_CHUNK_BYTES
        self._ring_slot_bytes = 0
        self._ring: list[torch.Tensor] = []
        self._ring_release: list[threading.Event] = []
        self._ring_cursor = 0
        self._pinned_ring_bytes = 0
        # Producer-side inflight throttle: (stream event, group output bytes) FIFO, oldest
        # first. Bytes leave the accounting only when the event's non-blocking query()
        # reports the group's reduce-scatter has EXECUTED.
        self._outstanding: deque[tuple[Any, int]] = deque()
        self._inflight_bytes = 0
        self._max_inflight_bytes = _MAX_INFLIGHT_BYTES
        self._inflight_timeout_s = _INFLIGHT_TIMEOUT_S
        self._debug_enabled: bool | None = None
        self._first_copy_logged: set[Parameter] = set()
        self._last_throttle_log_s = 0.0

    @property
    def enabled(self) -> bool:
        """bool: Whether the manager is enabled; the env gate is resolved on first access."""
        if self._enabled is None:
            self._enabled = fsdp_grad_offload_enabled()
        return self._enabled

    @property
    def budget_bytes(self) -> int:
        """int: Per-rank pinned-host budget for the offloaded set; 0 means unlimited."""
        if self._budget_gib is None:
            self._budget_gib = _budget_gib_from_env()
        return int(self._budget_gib * 2**30)

    @property
    def debug_enabled(self) -> bool:
        """bool: Whether all-rank diagnostic logging is on (forensics runs only)."""
        if self._debug_enabled is None:
            self._debug_enabled = _env_flag(_DEBUG_ENV)
        return self._debug_enabled

    def note_backward(self) -> None:
        """Arm offloading for the backward the engine is about to run.

        Every backward of the step offloads, because host-side accumulation replaces the
        device-side accumulation for the whole step.
        """
        if not self.enabled:
            return
        self._armed = True

    def begin_step(self) -> None:
        """Drain leftovers from an aborted previous train step and reset accumulation."""
        if not self.enabled:
            return
        self._drain()
        # An aborted step never reached restore(), so its stashed DTensor identities and
        # partial accumulation are stale; dropping the stash makes the next step's first
        # contribution overwrite the accumulation buffers instead of adding to them.
        self._stash.clear()
        self._first_copy_logged.clear()
        self._outstanding.clear()
        self._inflight_bytes = 0

    def maybe_offload_group(self, param_group: FSDPParamGroup) -> None:
        """Offload a parameter group's freshly reduced sharded gradients while armed.

        Args:
            param_group (FSDPParamGroup): The FSDP parameter group whose post-backward just ran.
        """
        if not self.enabled or not self._armed:
            return
        # HSDP and custom all-reduce-hook groups materialize gradients on streams other
        # than the reduce-scatter stream; leave their gradients on device.
        if getattr(param_group, "_is_hsdp", False) or getattr(param_group, "_all_reduce_hook", None) is not None:
            return
        pairs: list[tuple[Parameter, Any, torch.Tensor]] = []
        for fsdp_param in param_group.fsdp_params:
            param = fsdp_param.sharded_param
            grad = param.grad
            local = getattr(grad, "_local_tensor", None)
            # Plain-Tensor gradients (e.g. FSDP-ignored params) have no _local_tensor and
            # stay on device, accumulating exactly as the baseline does.
            if not isinstance(local, torch.Tensor):
                continue
            pairs.append((param, grad, local))
        if not pairs:
            return
        # First pass: run the budget selection for every parameter (it has side effects —
        # pinned accumulation buffers, budget accounting) and remember which pairs to
        # queue. Selection must happen for every group even when nothing is queued.
        selected: list[tuple[Parameter, Any, torch.Tensor, bool]] = []
        for param, grad, local in pairs:
            if not self._select(param, local):
                continue
            # First contribution of the current step = parameter not yet in the stash:
            # the stash holds the identity of everything offloaded since the last
            # restore(), so restore() itself is the accumulation boundary and a fresh
            # backward after it starts over even when begin_step was not called. The
            # stashed DTensor is the identity restored later; its spec
            # (mesh/placements/global shape) is identical every microbatch.
            selected.append((param, grad, local, param not in self._stash))
        if not selected:
            return
        # Bound the not-yet-executed selected bytes BEFORE queueing more: a queued task
        # holds its group's reduce-scatter output storage until the worker has issued its
        # copy, and an un-throttled producer out-runs the device queue by tens of seconds
        # (run71 OOM). The throttle only ever waits on this rank's own already-enqueued
        # prefix, so it cannot deadlock (module docstring).
        self._throttle_inflight()
        # The worker must be alive BEFORE the first task is queued, and the device context
        # must be resolved before any task references it.
        self._ensure_worker()
        rs_stream = param_group.comm_ctx.reduce_scatter_stream
        if self._device is None:
            self._device = pairs[0][2].device
            self._device_handle = param_group.device_handle
            # One dedicated copy stream for the whole D2H path: enqueueing the copies on
            # the reduce-scatter stream instead makes every chunk's completion event wait
            # behind the whole collective queue (run73: ~16 MB/s effective, backward
            # stretched ~15x into a fragmentation OOM).
            self._copy_stream = self._device_handle.Stream(device=self._device)
        # This wrapper runs after FSDP enqueued this group's reduce-scatter, so the event
        # fires once the freshly reduced gradient storage is complete. It is both the
        # copy stream's start gate for every task of this group and the producer-side
        # inflight event — ordering comes from the event gate, never from queue position
        # on rs_stream and never from a stream-level wait.
        gate_event = rs_stream.record_event()
        for param, grad, local, first in selected:
            if first:
                self._stash[param] = grad
                grad._local_tensor = None
            param.grad = None
            # A fresh done event per contribution: _drain() waits on the events in
            # _inflight, which must mean "the most recent queued contribution of every
            # parameter has been accumulated" — a reused event could satisfy the drain
            # with a previous contribution while the newest one is still in the queue.
            # The FIFO queue orders every earlier contribution before this one, so
            # draining on the newest events is sufficient and race-free.
            done = threading.Event()
            self._inflight[param] = done
            # Hand the whole D2H path to the worker and return immediately: the
            # producer must never block on ANYTHING whose completion depends on device
            # progress — not ring slots, not copy events (run65-69), and not even the
            # per-parameter ``done`` of the previous contribution (run70: one rank
            # parked in done.wait stopped enqueueing, the other fifteen then wedged in
            # the recomputed-forward a2a count exchange waiting for this rank's
            # collectives to actually EXECUTE, and the un-executed reduce-scatters
            # this rank's copy sat behind never gathered their peers — a closed loop).
            # The worker releases ``local`` via record_stream(copy_stream) once the
            # task's copies are issued, so only queued-but-unissued tasks pin their
            # group's storage — exactly what the inflight throttle bounds. The only
            # wait the producer ever performs is the bounded inflight poll above,
            # whose target event always lies inside this rank's own enqueued prefix.
            self._tasks.put((param, local, first, done, gate_event))
        # The group's reduce-scatter output is one shared buffer holding every parameter's
        # shard plus per-param padding, so one selected parameter holds the whole storage
        # until the worker issues its copy: charge the buffer's actual storage size, which
        # the per-shard numel sum undercounts by the padding. The gate event recorded
        # above doubles as the producer-side inflight event.
        group_bytes = pairs[0][2].untyped_storage().nbytes()
        self._outstanding.append((gate_event, group_bytes))
        self._inflight_bytes += group_bytes

    def restore(self) -> None:
        """Drain the worker and re-attach the accumulated gradients on device."""
        if not self._stash:
            return
        self._drain()
        assert self._device is not None
        if self.debug_enabled:
            self._log_all(
                f"restore: re-attaching {len(self._stash)} params / {self._offload_bytes / 2**30:.2f} GiB; "
                f"pre-attach {self._device_mem_str()}"
            )
        for param, grad in self._stash.items():
            acc = self._acc[param]
            # Synchronous H2D: the worker must never write the accumulation buffers
            # behind a still-in-flight device read of them (the next step's first
            # contribution is a host-side copy into the same buffer). ~0.2 s serial.
            # .view back into the recorded shard shape: only contiguous shards are
            # offloaded (_select's local.view(-1) gate), so the flat buffer holds the
            # shard's storage order and the view reproduces the original local exactly.
            grad._local_tensor = acc.to(device=self._device).view(self._acc_shapes[param])
            param.grad = grad
        self._stash.clear()
        self._outstanding.clear()
        self._inflight_bytes = 0
        if self.debug_enabled:
            self._log_all(f"restore done: {self._device_mem_str()}")

    def _log_all(self, message: str) -> None:
        get_logger().info(f"[FSDP grad offload][rank{get_rank()}] {message}")

    def _device_mem_str(self) -> str:
        if self._device is None:
            return "device unresolved"
        mod = getattr(torch, self._device.type, None)
        if mod is None or not hasattr(mod, "memory_allocated"):
            return f"device {self._device.type} has no memory stats"
        return (
            f"alloc={mod.memory_allocated(self._device) / 2**30:.2f}GiB "
            f"max_alloc={mod.max_memory_allocated(self._device) / 2**30:.2f}GiB "
            f"reserved={mod.memory_reserved(self._device) / 2**30:.2f}GiB"
        )

    def _throttle_inflight(self) -> None:
        """Sleep-poll the oldest outstanding group event until inflight bytes fit the cap.

        The only wait the producer thread ever performs. The polled event is at most the
        cap behind the group currently being queued and belongs to this rank's own
        already-enqueued reduce-scatter prefix, so the wait always terminates without
        participation from any other rank (the minimum-enqueue-position rank's target is
        quorate by construction) — unlike a wait on copy or slot completion, which can
        close a cross-rank rendezvous loop (run65-70). Raises after
        ``_inflight_timeout_s`` so a genuinely wedged run fails loudly instead of
        parking the autograd thread forever.

        Raises:
            RuntimeError: When no outstanding event fired within ``_inflight_timeout_s``.
        """
        deadline = time.monotonic() + self._inflight_timeout_s
        while self._inflight_bytes > self._max_inflight_bytes:
            if not self._outstanding:
                return
            event, group_bytes = self._outstanding[0]
            if event.query():
                self._outstanding.popleft()
                self._inflight_bytes -= group_bytes
                continue
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"FSDP grad offload inflight throttle timed out after {self._inflight_timeout_s:.0f}s: "
                    f"{self._inflight_bytes / 2**20:.0f} MiB of reduce-scatter outputs have not executed"
                )
            if self.debug_enabled and time.monotonic() - self._last_throttle_log_s > 30:
                self._last_throttle_log_s = time.monotonic()
                self._log_all(
                    f"inflight throttle parked: {self._inflight_bytes / 2**20:.0f} MiB outstanding, "
                    f"oldest group event not fired; {self._device_mem_str()}"
                )
            time.sleep(_INFLIGHT_POLL_S)

    def _drain(self) -> None:
        for done in self._inflight.values():
            done.wait()
        if self._worker_error is not None:
            error, self._worker_error = self._worker_error, None
            raise RuntimeError(f"FSDP grad offload worker failed: {error}") from error

    def _ensure_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._worker_loop, name="fsdp-grad-offload", daemon=True)
            self._worker.start()

    def _worker_loop(self) -> None:
        while True:
            task = self._tasks.get()
            if task is None:
                return
            param, local, first, done, gate_event = task
            try:
                self._offload_contribution(param, local, first, gate_event)
            except Exception as error:  # noqa: BLE001 - surfaced at the next drain, never swallowed
                self._worker_error = error
                # _drain() raises only at the next restore(), which a mid-step OOM can
                # preempt (run71/72): surface the failure the moment it happens, on every
                # rank — the traceback frames would otherwise also leak the task's `local`.
                get_logger().error(
                    f"[FSDP grad offload][rank{get_rank()}] worker task failed: {error!r}\n{traceback.format_exc()}"
                )
            finally:
                done.set()

    def _offload_contribution(self, param: Parameter, local: torch.Tensor, first: bool, gate_event: Any) -> None:
        """Copy one contribution D2H through the ring and accumulate it, on the worker thread.

        The copies run on the manager's dedicated copy stream, gated once by ``gate_event``
        (the producing group's reduce-scatter event). Every wait here is safe: the worker
        holds no other rank's progress hostage, and the producer only ever waits on ``done``
        events whose preconditions are inside this rank's already-enqueued backward prefix,
        so this method can never participate in a cross-rank deadlock.
        """
        flat = local.view(-1)
        total = flat.numel()
        esize = flat.element_size()
        slot_elems = self._ring_slot_bytes // esize
        assert self._device is not None  # resolved before the first task was queued
        assert self._copy_stream is not None
        acc = self._acc[param]
        acc_flat = acc.view(-1)
        offset = 0
        chunks = 0
        start_s = time.monotonic()
        while True:
            count = min(slot_elems, total - offset)
            slot = self._acquire_slot()
            try:
                # Only the worker may block: slot acquisition, the gated copy, and the
                # event sync all happen on this thread. The FIFO queue plus the per-chunk
                # release keep a parameter larger than the whole ring moving (chunk k's
                # slot is freed before chunk k+8 is acquired).
                with torch.device(self._device), self._device_handle.stream(self._copy_stream):
                    if offset == 0:
                        # Start gate (requirements 2.2/3.1): one wait_event against the
                        # producing group's reduce-scatter event — never a stream-level
                        # wait on rs_stream, and never a per-chunk gate (later chunks are
                        # ordered by the copy stream's own FIFO after the first).
                        self._copy_stream.wait_event(gate_event)
                    dst = self._ring[slot][: count * esize].view(acc.dtype)
                    dst.copy_(flat.narrow(0, offset, count), non_blocking=True)
                    event = self._copy_stream.record_event()
                event.synchronize()  # this chunk has landed in the ring
                src = self._ring[slot][: count * esize].view(acc.dtype)
                # Same order, same operands as the device-side accumulation: the first
                # contribution overwrites, every later one adds, contributions arrive
                # in microbatch order through the FIFO queue.
                dst_acc = acc_flat.narrow(0, offset, count)
                if first:
                    dst_acc.copy_(src)
                else:
                    dst_acc.add_(src)
            finally:
                # Release the slot even on failure so a worker error cannot wedge the
                # ring for the next contribution.
                self._ring_release[slot].set()
            offset += count
            chunks += 1
            if offset >= total:
                break
        if first and self.debug_enabled and param not in self._first_copy_logged:
            self._first_copy_logged.add(param)
            self._log_all(
                f"first copy done: {total * esize / 2**20:.1f} MiB in {(time.monotonic() - start_s) * 1000:.0f} ms "
                f"({chunks} chunks)"
            )
        self._release_device_source(local)

    def _release_device_source(self, local: torch.Tensor) -> None:
        # Requirements 2.4: the gradient storage was allocated on the reduce-scatter
        # stream but is read by the copy stream, so ``record_stream`` — never a
        # stream-level wait — tells the allocator to defer reuse of the block until the
        # copies issued above have executed. Dropping the task's reference right after
        # returns the block to the allocator pool as soon as the copies are issued
        # instead of keeping it allocated until they finish (run71's pinned set).
        # CPU tensors (tests) have no cross-stream deferral and record_stream rejects
        # foreign stream objects, so only real device tensors are marked.
        if local.device.type != "cpu":
            local.record_stream(self._copy_stream)

    def _select(self, param: Parameter, local: torch.Tensor) -> bool:
        """Return whether this parameter is (or becomes) part of the offloaded set.

        The budget is spent at the first encounter of each parameter, in backward order,
        and only on parameters that have not been offloaded yet; a rejected parameter is
        remembered so later microbatches skip it without retrying. Which parameters end up
        offloaded does not change the saving: the device footprint drops by exactly the
        total offloaded bytes, and the pre-clip restore adds exactly those bytes back.

        Args:
            param (Parameter): The sharded parameter whose gradient just materialized.
            local (torch.Tensor): Its freshly reduced local gradient shard.

        Returns:
            bool: True when the gradient should be copied to the host.
        """
        if param in self._acc:
            return True
        if param in self._skipped_params:
            return False
        try:
            flat = local.view(-1)
        except RuntimeError:
            # Non-contiguous shards cannot be copied in byte chunks; keep them on device
            # (correctness is unaffected, only the saving shrinks).
            self._skipped_params.add(param)
            return False
        need = flat.numel() * flat.element_size()
        budget = self.budget_bytes
        if budget and self._offload_bytes + need > budget:
            self._skipped_params.add(param)
            if not self._budget_logged:
                self._budget_logged = True
                suffix = f"; {self._device_mem_str()}" if self.debug_enabled else ""
                log_rank0.info(
                    f"[FSDP grad offload] budget reached: {self._offload_bytes / 2**30:.2f} GiB offloaded "
                    f"(cap {budget / 2**30:.2f} GiB); remaining gradients stay on device{suffix}"
                )
            return False
        self._ensure_acc(param, flat)
        self._acc_shapes[param] = local.shape
        return True

    def _ensure_acc(self, param: Parameter, flat: torch.Tensor) -> None:
        if param in self._acc:
            return
        self._acc[param] = torch.empty(flat.shape, dtype=flat.dtype, pin_memory=True)
        need = flat.numel() * flat.element_size()
        self._offload_bytes += need
        self._ensure_ring()
        self._report_pinned_growth(need)

    def _ensure_ring(self) -> None:
        if self._ring:
            return
        # The ring is ALWAYS sized at _RING_CHUNK_BYTES regardless of the first selected
        # parameter's size (run71-75 root cause): a tiny first param (norm weight, 16 B)
        # used to shrink every slot to the 4 KiB floor, and every later 605 MiB parameter
        # then copied in ~155k chunks paying the ~200 us event.synchronize round-trip per
        # 4 KiB chunk — ~30 s per parameter, stretching the backward until the queued
        # tasks' reduce-output storages OOMed the allocator. 8 x 32 MiB = 256 MiB pinned
        # is the documented ring cost (module docstring).
        self._ring_slot_bytes = self._ring_chunk_bytes
        self._ring = [
            torch.empty(self._ring_slot_bytes, dtype=torch.uint8, pin_memory=True) for _ in range(_RING_SLOTS)
        ]
        self._ring_release = [threading.Event() for _ in range(_RING_SLOTS)]
        for release in self._ring_release:
            release.set()
        self._report_pinned_growth(0)

    def _acquire_slot(self) -> int:
        slot = self._ring_cursor
        self._ring_cursor = (slot + 1) % len(self._ring)
        release = self._ring_release[slot]
        # Only the worker calls this, on its single FIFO thread: chunk k waits on the
        # release of the slot chunk k-8 staged into, which chunk k-8's own consumption
        # already set. A wait can therefore never outlive this thread's own progress.
        release.wait()
        release.clear()
        return slot

    def _report_pinned_growth(self, acc_bytes: int) -> None:
        # Pinned host memory is invisible to every meminfo bucket AND to process RSS
        # (run50: 22.9 GiB pinned while Unevictable/mlock stayed at 128 MB), so the only
        # reliable monitor is this self-report. Log at 1-GiB growth steps to keep the
        # first-step ramp to a handful of lines.
        first_ring = acc_bytes == 0 and self._ring and self._pinned_ring_bytes == 0
        if first_ring:
            self._pinned_ring_bytes = len(self._ring) * self._ring_slot_bytes
        self._pinned_bytes += acc_bytes + (self._pinned_ring_bytes if first_ring else 0)
        if self._pinned_bytes - self._pinned_logged_bytes < (1 << 30):
            return
        ring_gib = self._pinned_ring_bytes / 2**30
        suffix = f"; {self._device_mem_str()}" if self.debug_enabled else ""
        log_rank0.info(
            f"[FSDP grad offload] pinned host buffers: {self._pinned_bytes / 2**30:.1f} GiB so far "
            f"(acc + {ring_gib:.2f} GiB chunk ring; invisible to meminfo and RSS monitors){suffix}"
        )
        self._pinned_logged_bytes = self._pinned_bytes


_grad_offload = FSDPGradOffloadManager()


def install_fsdp_grad_offload(target: torch.nn.Module, manager: FSDPGradOffloadManager | None = None) -> int:
    """Wrap every FSDP parameter group under ``target`` so post-backward feeds the manager.

    Args:
        target (torch.nn.Module): The module that was just passed to ``fully_shard``.
        manager (FSDPGradOffloadManager | None): Manager to install; defaults to the module singleton.

    Returns:
        int: Number of parameter groups wrapped by this call.
    """
    mgr = manager if manager is not None else _grad_offload
    if not mgr.enabled:
        return 0
    wrapped = 0
    for module in target.modules():
        state = _get_module_state(module)
        param_group = getattr(state, "_fsdp_param_group", None)
        if not isinstance(param_group, FSDPParamGroup) or id(param_group) in mgr._wrapped_groups:
            continue
        mgr._wrapped_groups.add(id(param_group))
        orig_post_backward = param_group.post_backward

        def post_backward_wrapper(
            *args: Any, _orig: Any = orig_post_backward, _group: FSDPParamGroup = param_group, **kwargs: Any
        ) -> Any:
            result = _orig(*args, **kwargs)
            mgr.maybe_offload_group(_group)
            return result

        param_group.post_backward = post_backward_wrapper  # type: ignore[method-assign]
        wrapped += 1
    if wrapped:
        log_rank0.info(f"[FSDP grad offload] installed on {wrapped} FSDP param groups under {type(target).__name__}")
    return wrapped


def note_grad_offload_backward() -> None:
    """Arm offloading for the backward the engine is about to run."""
    _grad_offload.note_backward()


def begin_grad_offload_step() -> None:
    """Drain leftovers from an aborted previous train step and reset accumulation."""
    _grad_offload.begin_step()


def restore_grad_offload() -> None:
    """Drain the worker and re-attach accumulated gradients on device before clipping."""
    _grad_offload.restore()
