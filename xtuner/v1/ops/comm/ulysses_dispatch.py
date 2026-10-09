# Copyright © 2026 Huawei Technologies Co., Ltd.
"""Ulysses a2a dispatch gate and the deferred-work-fencing machinery it drives.

``xtuner.v1.ops.comm.all_to_all.ulysses_all_to_all`` consults
``a2a_async_enabled()`` (env ``XTUNER_KDA_SP_A2A_ASYNC``, read per call, default
off) and routes batch-1 ``scatter_dim=1`` / ``gather_dim=2`` movements through
``ulysses_scatter_heads_blocking`` when it is on; every other call stays on the
synchronous functional-collectives path. The KDA sequence-parallel lever
``XTUNER_KDA_SP_A2A_DEFER`` (``xtuner.v1.module.attention.kda_lever``) issues
the collectives with ``issue_ulysses_dim1`` and fences each at its consumer
with ``finish_ulysses_*``.

Inline on the compute stream each collective blocks the host for the wire plus
the slowest rank's arrival skew -- and with every rank parked in its own
previous collective the skew cascades: the run112 trace pinned ~10 s/step of
host park inside the issue call alone at EP8/SP16, starving the compute
stream. This module issues the collective with ``async_op=True`` on a per-group
comm stream and defers the fence to the consumption point. The hccl collective
runs on the backend's internal comm stream (a 2-NPU probe showed an event
recorded right after the async enqueue does NOT fence it), so the returned
c10d work handle is the only sound fence: it is polled with
``is_completed() and is_success()`` -- free when the collective already retired
successfully, the common case once the consumer ran other compute -- and
``wait()``-ed otherwise, which parks no longer than the old blocking call would
have and, for a completed-but-failed work, raises the stored backend exception
instead of silently consuming a never-written recv buffer. A per-module
registry keeps the send/recv buffers of outstanding collectives alive so the
caching allocator cannot reuse their blocks while the internal comm stream
still touches them; the registry is pruned at every fence.

Structure (the dispatcher's ``_AsyncDispatch`` pattern, adapted to deferred
consumption):

- ``issue_ulysses_dim1/dim2`` enqueue the collective and return the raw
  received buffer, the work handle and the output shape. They deliberately do
  NOT wait -- waiting at the issue point would re-serialize the compute
  stream. The output glue (permute/reshape) cannot run yet either, so the
  caller applies ``finish_ulysses_*`` (fence + glue) at the consumption point.
- The backward of the autograd wrapper runs the reverse a2a inline: the a2a
  is a pure permutation, so the data gradient is the same all-to-all applied
  to the output grad followed by the inverse reshape, and the consuming
  autograd node is the very next kernel -- fencing the result before returning
  puts the wait exactly at its consumption point.

The movement is the classic-c10d mirror of ``ulysses_all_to_all``
(``ops/comm/all_to_all.py``): scatter dim 1 and gather dim 2 of a batch-1
``[1, A, S, ...]`` tensor (and the exact inverse). The functional-collectives
autograd a2a has no NPU backend and queues inline on the current stream, so
the collective itself runs classic ``dist.all_to_all_single`` under the comm
stream -- same value and layout contract (contiguous output, element
``(a, s)`` = head ``a``'s global token ``s``), making results bit-identical
to the synchronous helpers.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from xtuner.v1.utils import get_device


if get_device() == "npu":
    from torch_npu.contrib import transfer_to_npu  # noqa: F401


_A2A_ASYNC_ENV = "XTUNER_KDA_SP_A2A_ASYNC"

# Send/recv buffers of collectives issued with ``async_op=True`` and not yet
# fenced: (work, send, recv). Held so the caching allocator cannot reuse their
# blocks while the backend's internal comm stream still reads/writes them;
# pruned by ``_wait_a2a`` at every fence point.
#
# Single-thread contract: issue and fence run on the host training thread.
# ``_AsyncUlysses*.backward`` also issues, but it executes on an autograd
# engine worker thread only while the host thread is parked inside
# ``backward()``, so appends and prunes never actually overlap. Pruning is
# still written as snapshot-iterate + ``remove`` (not a whole-list rebuild) so
# an entry appended concurrently by another thread cannot be dropped by the
# prune -- dropping one would free a buffer the comm stream still touches.
# Do not switch to a ``[:] =`` comprehension rebuild without adding a lock.
_PENDING_A2A: list[tuple[dist.Work, torch.Tensor, torch.Tensor]] = []

_COMM_STREAMS: dict[int, torch.cuda.Stream] = {}

_WorkOpt = dist.Work | None


def a2a_async_enabled() -> bool:
    """Whether ``XTUNER_KDA_SP_A2A_ASYNC`` is set to ``"1"`` (read per call)."""
    return os.getenv(_A2A_ASYNC_ENV, "0") == "1"


def _comm_stream(group: dist.ProcessGroup) -> torch.cuda.Stream:
    """Lazily create the dedicated a2a comm stream for ``group``.

    Args:
        group (dist.ProcessGroup): Process group whose a2a traffic the stream
            carries.

    Returns:
        torch.cuda.Stream: The group's comm stream (one per group object).
    """
    stream = _COMM_STREAMS.get(id(group))
    if stream is None:
        stream = torch.cuda.Stream()
        _COMM_STREAMS[id(group)] = stream
    return stream


def _a2a_issue(send: torch.Tensor, group: dist.ProcessGroup) -> tuple[torch.Tensor, _WorkOpt]:
    """Enqueue ``all_to_all_single`` on the comm stream without parking the
    host.

    The received buffer is allocated on the calling stream so its block
    returns to the caller's allocator pool. Neither buffer is readable until
    the returned work handle is fenced via ``_wait_a2a``; both are kept alive
    in ``_PENDING_A2A`` until then.

    Args:
        send (torch.Tensor): Uniformly split send buffer (dim 0 chunked per
            rank), produced on the calling stream.
        group (dist.ProcessGroup): Process group to communicate over.

    Returns:
        tuple[torch.Tensor, dist.Work | None]: The received buffer (same
            shape as ``send``) and the c10d work handle fencing it (``None``
            when the backend ran the collective synchronously).
    """
    recv = torch.empty_like(send)
    stream = _comm_stream(group)
    ready = torch.cuda.Event()
    ready.record(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        stream.wait_event(ready)
        work = dist.all_to_all_single(recv, send, group=group, async_op=True)
    if work is not None:
        _PENDING_A2A.append((work, send, recv))
    return recv, work


def _work_retired(work: dist.Work) -> bool:
    """Whether an issued work has completed *successfully*.

    ``is_success`` exists on the HCCL/cuda work handles this module fences in
    production, but backends without it (the gloo ``_CpuWork`` used by the
    CPU test path) treat completion as success -- their failures only surface
    through ``wait()``, and the fence below always parks on a non-retired
    work, so those backends still raise from ``wait()``.

    Args:
        work (dist.Work): Work handle returned by ``all_to_all_single``.

    Returns:
        bool: ``True`` once the collective retired without error.
    """
    if not work.is_completed():
        return False
    is_success = getattr(work, "is_success", None)
    return is_success() if is_success is not None else True


def _wait_a2a(work: _WorkOpt) -> None:
    """Fence an issued a2a at its consumption point.

    Parks the host only when the collective is still in flight -- the poll is
    free once the backend's internal comm stream retired it. A work that
    completed but *failed* (peer abort, HCCL error) also goes through
    ``wait()`` so the stored exception surfaces here instead of the consumer
    silently reading a never-written recv buffer. Also prunes the
    pending-buffer registry of every collective that completed successfully
    meanwhile; failed entries stay pinned (their buffers may still be touched
    by the comm stream, and their fence will raise shortly anyway).

    Args:
        work (dist.Work | None): Work handle returned by ``_a2a_issue``
            (``None`` means the collective ran synchronously).
    """
    if work is not None and not _work_retired(work):
        work.wait()
    if _PENDING_A2A:
        # Snapshot-iterate + remove (see the _PENDING_A2A contract): entries
        # appended by another thread while this loop runs are not in the
        # snapshot and survive the prune. Tuple identity short-circuits the
        # comparison, so Tensor __eq__ is never triggered.
        for entry in list(_PENDING_A2A):
            if _work_retired(entry[0]):
                _PENDING_A2A.remove(entry)


class _AsyncUlyssesDim1(torch.autograd.Function):
    """Issue-only autograd node for the dim1-scatter Ulysses a2a.

    Forward returns the raw received buffer (``[world, A/sp, S, ...]``
    send-layout) plus the work handle; the caller applies the fence and the
    output glue at the consumption point. Backward runs the reverse a2a
    inline (the collective is a permutation, so the data gradient is the same
    all-to-all on the output grad).
    """

    @staticmethod
    def forward(  # noqa: ANN001
        ctx,
        input: torch.Tensor,
        group: dist.ProcessGroup,
    ) -> tuple[torch.Tensor, _WorkOpt]:
        world = dist.get_world_size(group)
        ctx.group = group
        if world == 1:
            return input, None
        send = input.reshape(world, input.shape[1] // world, *input.shape[2:]).contiguous()
        return _a2a_issue(send, group)

    @staticmethod
    def backward(  # noqa: ANN001
        ctx, grad_output: torch.Tensor, _grad_work: None
    ) -> tuple[torch.Tensor | None, None]:
        world = dist.get_world_size(ctx.group)
        if world == 1:
            return grad_output, None
        grad_recv = grad_output.contiguous()
        grad_send, work = _a2a_issue(grad_recv, ctx.group)
        _wait_a2a(work)
        d_x = grad_send.reshape(1, grad_send.shape[0] * grad_send.shape[1], *grad_send.shape[2:])
        return d_x, None


class _AsyncUlyssesDim2(torch.autograd.Function):
    """Issue-only autograd node for the dim2-scatter Ulysses a2a (the exact
    inverse movement of ``_AsyncUlyssesDim1``)."""

    @staticmethod
    def forward(  # noqa: ANN001
        ctx,
        input: torch.Tensor,
        group: dist.ProcessGroup,
    ) -> tuple[torch.Tensor, _WorkOpt]:
        world = dist.get_world_size(group)
        ctx.group = group
        if world == 1:
            return input, None
        send = input.reshape(1, input.shape[1], world, input.shape[2] // world, *input.shape[3:])
        send = send.permute(2, 0, 1, *range(3, send.dim())).contiguous()
        return _a2a_issue(send, group)

    @staticmethod
    def backward(  # noqa: ANN001
        ctx, grad_output: torch.Tensor, _grad_work: None
    ) -> tuple[torch.Tensor | None, None]:
        world = dist.get_world_size(ctx.group)
        if world == 1:
            return grad_output, None
        grad_recv = grad_output.contiguous()
        grad_send, work = _a2a_issue(grad_recv, ctx.group)
        _wait_a2a(work)
        d_x = grad_send.permute(1, 2, 0, *range(3, grad_send.dim()))
        d_x = d_x.reshape(1, grad_send.shape[2], grad_send.shape[0] * grad_send.shape[3], *grad_send.shape[4:])
        return d_x, None


def issue_ulysses_dim1(x: torch.Tensor, group: dist.ProcessGroup) -> tuple[torch.Tensor, _WorkOpt, torch.Size]:
    """Issue the head-scatter Ulysses a2a without blocking the caller.

    Args:
        x (torch.Tensor): Batch-1 ``[1, A, S/sp, ...]`` input.
        group (dist.ProcessGroup): Sequence-parallel process group.

    Returns:
        tuple[torch.Tensor, dist.Work | None, torch.Size]: The raw received
            buffer ``[world, A/sp, S, ...]`` (unreadable until the work is
            fenced via ``finish_ulysses_dim1``), the c10d work handle, and the
            ``[1, A/sp, S, ...]`` output shape to pass to
            ``finish_ulysses_dim1``.
    """
    world = dist.get_world_size(group)
    out_shape = torch.Size((1, x.shape[1] // world, x.shape[2] * world, *x.shape[3:]))
    if world == 1:
        # Honour the "returned as is" contract: Function.apply materializes a
        # fresh alias even for the world==1 passthrough.
        return x, None, out_shape
    recv, work = _AsyncUlyssesDim1.apply(x, group)
    return recv, work, out_shape


def finish_ulysses_dim1(recv: torch.Tensor, work: _WorkOpt, out_shape: torch.Size) -> torch.Tensor:
    """Consume a ``issue_ulysses_dim1`` result: fence, then apply the glue.

    Args:
        recv (torch.Tensor): Raw received buffer from ``issue_ulysses_dim1``.
        work (dist.Work | None): Work handle returned by the issue call.
        out_shape (torch.Size): Output shape returned by the issue call.

    Returns:
        torch.Tensor: The ``[1, A/sp, S, ...]`` movement output, contiguous,
            safe to read on the current stream.
    """
    _wait_a2a(work)
    out = recv.permute(1, 0, *range(2, recv.dim())).contiguous()
    return out.reshape(out_shape)


def finish_ulysses_dim1_seq_major(recv: torch.Tensor, work: _WorkOpt) -> torch.Tensor:
    """Consume a ``issue_ulysses_dim1`` result directly in sequence-major
    layout with a single copy.

    Equivalent to ``finish_ulysses_dim1(recv, work, out_shape).transpose(1,
    2).contiguous()`` for the 3-D case, but as one permutation copy instead
    of two: ``recv[w, a, l]`` holds head-shard element ``a`` of global token
    ``s = w * (L/sp) + l``, so ``permute(0, 2, 1)`` flattens ``(w, l)`` into
    ``s`` directly and the head-major intermediate never materialises. The
    backward saves the matching second glue copy (the single autograd
    permutation lands the gradient straight in the ``[world, A/sp, L/sp]``
    layout the reverse a2a consumes).

    Args:
        recv (torch.Tensor): Raw received buffer from ``issue_ulysses_dim1``
            (``[world, A/sp, L/sp]``).
        work (dist.Work | None): Work handle returned by the issue call.

    Returns:
        torch.Tensor: The ``[1, S, A/sp]`` movement output (with ``S = world
            * L/sp``), contiguous, safe to read on the current stream.
    """
    _wait_a2a(work)
    out = recv.permute(0, 2, 1).contiguous()
    return out.reshape(1, recv.shape[0] * recv.shape[2], recv.shape[1])


def issue_ulysses_dim2(x: torch.Tensor, group: dist.ProcessGroup) -> tuple[torch.Tensor, _WorkOpt, torch.Size]:
    """Issue the sequence-scatter Ulysses a2a without blocking the caller.

    Args:
        x (torch.Tensor): Batch-1 ``[1, A/sp, S, ...]`` input.
        group (dist.ProcessGroup): Sequence-parallel process group.

    Returns:
        tuple[torch.Tensor, dist.Work | None, torch.Size]: The raw received
            buffer (unreadable until the work is fenced via
            ``finish_ulysses_dim2``), the c10d work handle, and the
            ``[1, A, S/sp, ...]`` output shape for ``finish_ulysses_dim2``.
    """
    world = dist.get_world_size(group)
    out_shape = torch.Size((1, x.shape[1] * world, x.shape[2] // world, *x.shape[3:]))
    if world == 1:
        # Honour the "returned as is" contract: Function.apply materializes a
        # fresh alias even for the world==1 passthrough.
        return x, None, out_shape
    recv, work = _AsyncUlyssesDim2.apply(x, group)
    return recv, work, out_shape


def finish_ulysses_dim2(recv: torch.Tensor, work: _WorkOpt, out_shape: torch.Size) -> torch.Tensor:
    """Consume a ``issue_ulysses_dim2`` result: fence, then apply the glue.

    Args:
        recv (torch.Tensor): Raw received buffer from ``issue_ulysses_dim2``.
        work (dist.Work | None): Work handle returned by the issue call.
        out_shape (torch.Size): Output shape returned by the issue call.

    Returns:
        torch.Tensor: The ``[1, A, S/sp, ...]`` movement output, contiguous,
            safe to read on the current stream.
    """
    _wait_a2a(work)
    out = recv.permute(1, 0, *range(2, recv.dim())).contiguous()
    return out.reshape(out_shape)


def ulysses_scatter_heads_blocking(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Head-scatter Ulysses a2a issued on the comm stream, fenced on return.

    Value- and layout-identical to ``ulysses_all_to_all(x, scatter_dim=1,
    gather_dim=2, ...)``; for ``world_size == 1`` the input is returned as is.

    Args:
        x (torch.Tensor): Batch-1 ``[1, A, S/sp, ...]`` input.
        group (dist.ProcessGroup): Sequence-parallel process group.

    Returns:
        torch.Tensor: The ``[1, A/sp, S, ...]`` output, safe to read on the
            current stream.
    """
    if dist.get_world_size(group) == 1:
        return x
    recv, work, out_shape = issue_ulysses_dim1(x, group)
    return finish_ulysses_dim1(recv, work, out_shape)


def ulysses_scatter_seq_blocking(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Sequence-scatter Ulysses a2a issued on the comm stream, fenced on
    return.

    Value- and layout-identical to ``ulysses_all_to_all(x, scatter_dim=1,
    gather_dim=2, ...)`` on a ``[1, S, H/sp, ...]`` tensor (sequence on dim 1);
    for ``world_size == 1`` the input is returned as is.

    Args:
        x (torch.Tensor): Batch-1 ``[1, A/sp, S, ...]`` input.
        group (dist.ProcessGroup): Sequence-parallel process group.

    Returns:
        torch.Tensor: The ``[1, A, S/sp, ...]`` output, safe to read on the
            current stream.
    """
    if dist.get_world_size(group) == 1:
        return x
    recv, work, out_shape = issue_ulysses_dim2(x, group)
    return finish_ulysses_dim2(recv, work, out_shape)


__all__ = [
    "a2a_async_enabled",
    "issue_ulysses_dim1",
    "issue_ulysses_dim2",
    "finish_ulysses_dim1",
    "finish_ulysses_dim1_seq_major",
    "finish_ulysses_dim2",
    "ulysses_scatter_heads_blocking",
    "ulysses_scatter_seq_blocking",
]
