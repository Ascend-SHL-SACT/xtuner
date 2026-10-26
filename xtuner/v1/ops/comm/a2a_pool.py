# Copyright © 2026 Huawei Technologies Co., Ltd.
"""Bounded staging pool for the Ulysses a2a received buffers.

The Ulysses sequence-parallel a2a (``xtuner.v1.ops.comm.ulysses_dispatch``) allocates a
fresh ``recv`` buffer per collective (``torch.empty_like(send)``). In the deferred schedule
(``XTUNER_KDA_SP_A2A_DEFER``) each KDA layer issues five head-scatter a2a's back to back, so
up to five of them coexist per layer, and a received buffer keeps its block after its data
is dead because tensor objects that saw it are still alive in the caller frame and in the
autograd graph -- the caching allocator cannot recycle it until they all drop. This module
trades that per-call growth for a bounded free list: a received buffer goes back to the pool
once its consumer is done and is handed to the next collective of the same shape instead of
a new allocation.

Recycle point (the point at which the data is provably dead): the end of the
``finish_ulysses_*`` glue copy, the single reader of a received buffer. Callers get the glue
tensor, never the raw buffer -- except for the degenerate shapes noted below, which are then
not recycled.

The pool is gated behind ``XTUNER_ULYSSES_A2A_POOL`` (read per call, default off) and bounded
by ``XTUNER_ULYSSES_A2A_POOL_CAP_GIB`` (default 1 GiB of free-list bytes): buffers beyond the
cap are dropped instead of pooled, so a pathological shape never pins device memory.
``XTUNER_ULYSSES_A2A_POOL_DEBUG=1`` dumps hit/miss/held-byte counters at interpreter exit.

Soundness. A recycled buffer is still aliased by tensor objects that are alive (the caller
frame keeps the ``issue_ulysses_*`` tuple until its ``finish_ulysses_*``, the autograd node
keeps the forward pair), but never read: the module's contract is that a raw recv is readable
only up to its own glue copy, so recycling past it cannot change a value. Reuse is ordered
against that copy explicitly: ``pool_release`` records an event on the stream that runs the
copy and ``pool_acquire`` makes the caller's current stream wait for it, so the next
collective can never write a buffer before the glue copy that read it has retired -- a
reliance on the comm-stream ready event of ``_a2a_issue`` alone is not enough, because the
glue copy may have been enqueued from a stream context the issuing caller no longer sees.
The glue is not always a copy: with a degenerate permutation (``num_heads // sp_size == 1``,
as in the vision tower's ``[world, 1, L, D]`` received buffers) ``contiguous()`` is the
identity and the caller keeps reading the buffer itself, so ``pool_release`` drops such a
buffer -- detected by its glue output aliasing it -- and the shape stays unpooled.
The backward movements never pool: their received buffer is the data gradient that flows on,
not staging.
"""

from __future__ import annotations

import atexit
import os

import torch


_POOL_ENV = "XTUNER_ULYSSES_A2A_POOL"
_CAP_ENV = "XTUNER_ULYSSES_A2A_POOL_CAP_GIB"
_DEBUG_ENV = "XTUNER_ULYSSES_A2A_POOL_DEBUG"
_DEFAULT_CAP_GIB = 1.0

_PoolKey = tuple[int, torch.dtype, int | None]
# A recycled buffer plus the event marking its consumer's completion (``None`` when the
# device copies synchronously and no ordering is needed, i.e. on CPU).
_PoolEntry = tuple[torch.Tensor, torch.cuda.Event | None]

# Recycled staging buffers, keyed by (numel, dtype, device index).
_FREE: dict[_PoolKey, list[_PoolEntry]] = {}
_FREE_BYTES = 0
# Buffers currently handed out, keyed by ``id`` of the flat tensor a pooled view aliases.
_HELD: dict[int, torch.Tensor] = {}
_STATS = {
    "hit": 0,
    "miss": 0,
    "held_bytes": 0,
    "peak_held_bytes": 0,
    "released": 0,
    "dropped": 0,
    "aliased": 0,
}


def a2a_pool_enabled() -> bool:
    """Whether ``XTUNER_ULYSSES_A2A_POOL`` is set to ``"1"`` (read per call)."""
    return os.getenv(_POOL_ENV, "0") == "1"


def a2a_pool_cap_bytes() -> int:
    """Free-list byte cap from ``XTUNER_ULYSSES_A2A_POOL_CAP_GIB`` (default 1 GiB).

    Returns:
        int: Maximum bytes the free list may hold; ``0`` means unbounded.
    """
    raw = os.getenv(_CAP_ENV, "")
    try:
        gib = float(raw) if raw else _DEFAULT_CAP_GIB
    except ValueError:
        gib = _DEFAULT_CAP_GIB
    return int(gib * (1 << 30))


def pool_acquire(shape: torch.Size, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Take a staging tensor of ``shape``, reusing a recycled buffer when possible.

    Args:
        shape (torch.Size): Shape of the staging tensor to hand out.
        dtype (torch.dtype): Element type of the staging tensor.
        device (torch.device): Device the staging tensor lives on.

    Returns:
        torch.Tensor: A contiguous tensor of exactly ``shape``. It aliases pooled storage
            when a recycled buffer was available, so the caller must not read it after the
            point at which its data is dead (see the module docstring).
    """
    numel = 1
    for dim in shape:
        numel *= int(dim)
    key: _PoolKey = (numel, dtype, device.index)
    bucket = _FREE.get(key)
    if bucket:
        flat, consumed = bucket.pop()
        if consumed is not None:
            # Do not hand the buffer out before its previous consumer's copy retired.
            torch.cuda.current_stream().wait_event(consumed)
        global _FREE_BYTES
        _FREE_BYTES -= flat.numel() * flat.element_size()
        _STATS["hit"] += 1
    else:
        flat = torch.empty(numel, dtype=dtype, device=device)
        _STATS["miss"] += 1
    _HELD[id(flat)] = flat
    _STATS["held_bytes"] += flat.numel() * flat.element_size()
    _STATS["peak_held_bytes"] = max(_STATS["peak_held_bytes"], _STATS["held_bytes"])
    return flat.view(shape)


def is_pooled_staging(t: torch.Tensor) -> bool:
    """Whether ``t`` is a pooled staging view still owned by the pool.

    The check is the identity of the flat tensor ``t`` aliases: pooled views carry it as
    ``_base`` and the pool keeps that flat tensor alive while the view is handed out, so a
    caller's own tensor can never collide with it.

    Args:
        t (torch.Tensor): Tensor to test.

    Returns:
        bool: ``True`` when ``t`` came from :func:`pool_acquire` and is still out.
    """
    base = t._base
    return base is not None and id(base) in _HELD


def pool_release(t: torch.Tensor, consumer: torch.Tensor | None = None) -> None:
    """Return a pooled staging tensor to the free list (no-op for any other tensor).

    Only tensors handed out by :func:`pool_acquire` are recycled: the check is the
    identity of the flat base tensor a pooled view aliases, so an unpooled staging buffer
    or a caller's own tensor is never retained.

    A buffer whose glue output still aliases it is dropped, not recycled: the glue copy of
    ``finish_ulysses_*`` is the identity whenever the permuted dimensions are degenerate
    (``recv`` of shape ``[world, 1, L, ...]``, i.e. ``num_heads // sp_size == 1``, as in the
    vision tower), so the caller keeps reading the buffer for the rest of the layer and
    reusing it would overwrite live data. Such shapes then keep the unpooled behaviour.

    Args:
        t (torch.Tensor): A staging tensor previously returned by :func:`pool_acquire`,
            whose data is dead.
        consumer (torch.Tensor | None): The tensor the glue produced out of ``t``; when it
            aliases ``t`` the buffer is dropped instead of pooled.
    """
    base = t._base
    if base is None or not _HELD:
        return
    flat = _HELD.pop(id(base), None)
    if flat is None:
        return
    nbytes = flat.numel() * flat.element_size()
    _STATS["held_bytes"] -= nbytes
    _STATS["released"] += 1
    if consumer is not None and consumer.data_ptr() == flat.data_ptr():
        _STATS["aliased"] += 1
        return
    cap = a2a_pool_cap_bytes()
    global _FREE_BYTES
    if cap and _FREE_BYTES + nbytes > cap:
        _STATS["dropped"] += 1
        return
    _FREE.setdefault((flat.numel(), flat.dtype, flat.device.index), []).append((flat, _consumer_done(flat)))
    _FREE_BYTES += nbytes


def a2a_pool_stats() -> dict[str, float]:
    """Snapshot of the pool counters (bytes in MiB).

    Returns:
        dict[str, float]: Counters plus the free-list and in-use byte levels.
    """
    mib = 1 << 20
    return {
        "hit": float(_STATS["hit"]),
        "miss": float(_STATS["miss"]),
        "released": float(_STATS["released"]),
        "dropped": float(_STATS["dropped"]),
        "aliased": float(_STATS["aliased"]),
        "held_mib": _STATS["held_bytes"] / mib,
        "peak_held_mib": _STATS["peak_held_bytes"] / mib,
        "free_mib": _FREE_BYTES / mib,
    }


def _consumer_done(flat: torch.Tensor) -> torch.cuda.Event | None:
    """Record the current stream's position so that later reuse of ``flat`` is ordered after it.

    Everything enqueued so far on the current stream -- in particular the glue copy that
    read ``flat`` -- precedes any later reuse of the buffer.

    Args:
        flat (torch.Tensor): The buffer being released.

    Returns:
        torch.cuda.Event | None: The recorded event, or ``None`` on CPU where copies are
            synchronous and no ordering is needed.
    """
    if flat.device.type == "cpu":
        return None
    done = torch.cuda.Event()
    done.record(torch.cuda.current_stream())
    return done


def _dump_stats() -> None:
    if os.getenv(_DEBUG_ENV, "0") != "1":
        return
    stats = a2a_pool_stats()
    print(
        "[a2a_pool] "
        + " ".join(f"{name}={value:.1f}" for name, value in stats.items())
        + f" buffers_free={sum(len(b) for b in _FREE.values())}",
        flush=True,
    )


atexit.register(_dump_stats)


__all__ = [
    "a2a_pool_cap_bytes",
    "a2a_pool_enabled",
    "a2a_pool_stats",
    "is_pooled_staging",
    "pool_acquire",
    "pool_release",
]
