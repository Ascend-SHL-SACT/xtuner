# Copyright © 2026 Huawei Technologies Co., Ltd.
"""Async Ulysses all-to-all on a dedicated comm stream.

The sequence-parallel KDA path issues six Ulysses a2a per layer per
micro-batch (q/k/v/g/beta forward, core output back). Inline on the compute
stream each collective blocks for the wire plus the slowest rank's arrival
skew — profiling showed ~9.9 s busy / 9.1 s fully exposed per step at
EP8/SP16. This module moves the collective onto a per-group comm stream: the
collective flies under independent compute while the consuming stream only
parks at a ``wait_event`` placed at the consumption point, so a later
micro-batch's projections and a2a issues can overlap the current one's
chunked KDA scan.

Structure (the dispatcher's ``_AsyncDispatch`` pattern, adapted to deferred
consumption):

- ``issue_ulysses_dim1/dim2`` enqueue the collective on the comm stream and
  return the raw received buffer plus the fencing event. They deliberately do
  NOT wait — waiting at the issue point would re-serialize the compute
  stream. The output glue (permute/reshape) cannot run yet either, so the
  caller applies ``finish_ulysses_*`` (wait + glue) at the consumption point.
- The backward of the autograd wrapper runs the reverse a2a inline: the a2a
  is a pure permutation, so the data gradient is the same all-to-all applied
  to the output grad followed by the inverse reshape, and the consuming
  autograd node is the very next kernel — fencing the result before returning
  puts the wait exactly at its consumption point.

The movement is the classic-c10d mirror of ``ulysses_all_to_all``
(``ops/comm/all_to_all.py``): scatter dim 1 and gather dim 2 of a batch-1
``[1, A, S, ...]`` tensor (and the exact inverse). The functional-collectives
autograd a2a has no NPU backend and queues inline on the current stream, so
the collective itself runs classic ``dist.all_to_all_single`` under the comm
stream — same value and layout contract (contiguous output, element
``(a, s)`` = head ``a``'s global token ``s``), making results bit-identical
to the synchronous helpers.
"""

import torch
import torch.distributed as dist

from xtuner.v1.utils import get_device


if get_device() == "npu":
    from torch_npu.contrib import transfer_to_npu  # noqa: F401


_COMM_STREAMS: dict[int, torch.cuda.Stream] = {}


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


def _a2a_issue(send: torch.Tensor, group: dist.ProcessGroup, finished: torch.cuda.Event) -> torch.Tensor:
    """Enqueue ``all_to_all_single`` on the comm stream and record
    ``finished``.

    The received buffer is allocated on the calling stream so its block
    returns to the caller's allocator pool; every read of it is fenced by
    ``finished``, and the send buffer is fenced against comm-side reuse via
    ``record_stream``.

    Args:
        send (torch.Tensor): Uniformly split send buffer (dim 0 chunked per
            rank), produced on the calling stream.
        group (dist.ProcessGroup): Process group to communicate over.
        finished (torch.cuda.Event): Recorded on the comm stream once the
            collective is enqueued-complete.

    Returns:
        torch.Tensor: The received buffer (same shape as ``send``); unreadable
            until ``finished`` is waited on the consuming stream.
    """
    recv = torch.empty_like(send)
    stream = _comm_stream(group)
    ready = torch.cuda.Event()
    ready.record(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        stream.wait_event(ready)
        dist.all_to_all_single(recv, send, group=group)
        send.record_stream(stream)
        finished.record(stream)
    return recv


class _AsyncUlyssesDim1(torch.autograd.Function):
    """Issue-only autograd node for the dim1-scatter Ulysses a2a.

    Forward returns the raw received buffer (``[world, A/sp, S, ...]``
    send-layout); the caller applies the output glue at the consumption point.
    Backward runs the reverse a2a inline (the collective is a permutation, so
    the data gradient is the same all-to-all on the output grad).
    """

    @staticmethod
    def forward(  # noqa: ANN001
        ctx,
        input: torch.Tensor,
        group: dist.ProcessGroup,
        finished: torch.cuda.Event,
    ) -> torch.Tensor:
        world = dist.get_world_size(group)
        ctx.group = group
        if world == 1:
            finished.record(torch.cuda.current_stream())
            return input
        send = input.reshape(world, input.shape[1] // world, *input.shape[2:]).contiguous()
        return _a2a_issue(send, group, finished)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:  # noqa: ANN001
        world = dist.get_world_size(ctx.group)
        if world == 1:
            return grad_output, None, None
        grad_recv = grad_output.contiguous()
        finished = torch.cuda.Event()
        grad_send = _a2a_issue(grad_recv, ctx.group, finished)
        torch.cuda.current_stream().wait_event(finished)
        d_x = grad_send.reshape(1, grad_send.shape[0] * grad_send.shape[1], *grad_send.shape[2:])
        return d_x, None, None


class _AsyncUlyssesDim2(torch.autograd.Function):
    """Issue-only autograd node for the dim2-scatter Ulysses a2a (the exact
    inverse movement of ``_AsyncUlyssesDim1``)."""

    @staticmethod
    def forward(  # noqa: ANN001
        ctx,
        input: torch.Tensor,
        group: dist.ProcessGroup,
        finished: torch.cuda.Event,
    ) -> torch.Tensor:
        world = dist.get_world_size(group)
        ctx.group = group
        if world == 1:
            finished.record(torch.cuda.current_stream())
            return input
        send = input.reshape(1, input.shape[1], world, input.shape[2] // world, *input.shape[3:])
        send = send.permute(2, 0, 1, *range(3, send.dim())).contiguous()
        return _a2a_issue(send, group, finished)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:  # noqa: ANN001
        world = dist.get_world_size(ctx.group)
        if world == 1:
            return grad_output, None, None
        grad_recv = grad_output.contiguous()
        finished = torch.cuda.Event()
        grad_send = _a2a_issue(grad_recv, ctx.group, finished)
        torch.cuda.current_stream().wait_event(finished)
        d_x = grad_send.permute(1, 2, 0, *range(3, grad_send.dim()))
        d_x = d_x.reshape(1, grad_send.shape[2], grad_send.shape[0] * grad_send.shape[3], *grad_send.shape[4:])
        return d_x, None, None


def issue_ulysses_dim1(x: torch.Tensor, group: dist.ProcessGroup) -> tuple[torch.Tensor, torch.cuda.Event, torch.Size]:
    """Issue the head-scatter Ulysses a2a without blocking the caller.

    Args:
        x (torch.Tensor): Batch-1 ``[1, A, S/sp, ...]`` input.
        group (dist.ProcessGroup): Sequence-parallel process group.

    Returns:
        tuple[torch.Tensor, torch.cuda.Event, torch.Size]: The raw received
            buffer ``[world, A/sp, S, ...]`` (unreadable until the event is
            waited), the fencing event, and the ``[1, A/sp, S, ...]`` output
            shape to pass to ``finish_ulysses_dim1``.
    """
    finished = torch.cuda.Event()
    recv = _AsyncUlyssesDim1.apply(x, group, finished)
    world = dist.get_world_size(group)
    out_shape = torch.Size((1, x.shape[1] // world, x.shape[2] * world, *x.shape[3:]))
    return recv, finished, out_shape


def finish_ulysses_dim1(recv: torch.Tensor, event: torch.cuda.Event, out_shape: torch.Size) -> torch.Tensor:
    """Consume a ``issue_ulysses_dim1`` result: wait, then apply the glue.

    Args:
        recv (torch.Tensor): Raw received buffer from ``issue_ulysses_dim1``.
        event (torch.cuda.Event): Its fencing event.
        out_shape (torch.Size): Output shape returned by the issue call.

    Returns:
        torch.Tensor: The ``[1, A/sp, S, ...]`` movement output, contiguous,
            safe to read on the current stream.
    """
    torch.cuda.current_stream().wait_event(event)
    out = recv.permute(1, 0, *range(2, recv.dim())).contiguous()
    return out.reshape(out_shape)


def issue_ulysses_dim2(x: torch.Tensor, group: dist.ProcessGroup) -> tuple[torch.Tensor, torch.cuda.Event, torch.Size]:
    """Issue the sequence-scatter Ulysses a2a without blocking the caller.

    Args:
        x (torch.Tensor): Batch-1 ``[1, A/sp, S, ...]`` input.
        group (dist.ProcessGroup): Sequence-parallel process group.

    Returns:
        tuple[torch.Tensor, torch.cuda.Event, torch.Size]: The raw received
            buffer (unreadable until the event is waited), the fencing event,
            and the ``[1, A, S/sp, ...]`` output shape for
            ``finish_ulysses_dim2``.
    """
    finished = torch.cuda.Event()
    recv = _AsyncUlyssesDim2.apply(x, group, finished)
    world = dist.get_world_size(group)
    out_shape = torch.Size((1, x.shape[1] * world, x.shape[2] // world, *x.shape[3:]))
    return recv, finished, out_shape


def finish_ulysses_dim2(recv: torch.Tensor, event: torch.cuda.Event, out_shape: torch.Size) -> torch.Tensor:
    """Consume a ``issue_ulysses_dim2`` result: wait, then apply the glue.

    Args:
        recv (torch.Tensor): Raw received buffer from ``issue_ulysses_dim2``.
        event (torch.cuda.Event): Its fencing event.
        out_shape (torch.Size): Output shape returned by the issue call.

    Returns:
        torch.Tensor: The ``[1, A, S/sp, ...]`` movement output, contiguous,
            safe to read on the current stream.
    """
    torch.cuda.current_stream().wait_event(event)
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
    recv, event, out_shape = issue_ulysses_dim1(x, group)
    return finish_ulysses_dim1(recv, event, out_shape)


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
    recv, event, out_shape = issue_ulysses_dim2(x, group)
    return finish_ulysses_dim2(recv, event, out_shape)
