"""D2H-overlap copy stream for the Muon momentum swap (``XTUNER_MUON_SWAP``).

Gated by ``XTUNER_MUON_SWAP`` (default off). When on, the Muon optimizer's
momentum (and the AdamW ``m``/``v`` for non-Muon params) live on pinned CPU;
``h2d_momentum`` stages them to a fresh device temp and ``d2h_momentum`` writes
them back. The D2H runs on a dedicated copy stream (overlapping the AsyncRuntime
event loop's prior-batch AGRS collectives on the default stream); the H2D stays
on the default stream. This mirrors the *recommended* configuration of
``swap_adamw_overlap`` -- whose own docstring states D2H-overlap is "mild"
(reclaimable) while H2D-overlap "fragments hard" because "the cross-stream write
itself taints regardless of alloc stream."

That taint was verified three ways here: a fresh copy-stream-allocated H2D temp
is never reused by the NPU caching allocator and OOMs by step ~4; a
default-allocated H2D temp written on the copy stream corrupts the default
stream's compute read and produces nan on the second optim iteration (whether or
not ``record_stream`` is used); and a persistent buffer with
``storage().resize_(0/full)`` reintroduces the same nan (the host-synchronous
``resize_(0)`` frees storage while the async D2H copy_ read is still in flight,
and cross-batch default-stream allocations reuse the freed storage). So H2D
stays on the default stream (serial, no cross-stream write, no taint, no nan),
and only the D2H -- a cross-stream *read* of a default-allocated block via
``record_stream`` -- goes on the copy stream. That read is the docstring's
explicit *reclaimable* pattern (allocated on default, read on copy): no
fragmentation, no OOM. The D2H overlaps the caller's subsequent AGRS comm.

The cross-step ``d2h_ev`` (re-recorded each D2H, primed at construction) is
waited on the default stream at the next H2D: a step's D2H (copy stream) writes
the pinned-CPU buffer that the next step's H2D (default stream) reads, so the
default stream must wait the prior D2H.

``XTUNER_MUON_SWAP_OVERLAP=0`` falls back to default-stream FIFO copies (no copy
stream, no events) -- the prior serial path, kept for A/B.
"""

from __future__ import annotations

import os

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor

from xtuner.v1.utils import get_torch_device_module


DEVICE_MODULE = get_torch_device_module()


def is_enabled() -> bool:
    """Return whether the Muon momentum swap is on (``XTUNER_MUON_SWAP=1``).

    Returns:
        bool: True when the feature is enabled. Default off.
    """
    return os.environ.get("XTUNER_MUON_SWAP", "0") == "1"


def _use_copy_stream() -> bool:
    """Return whether D2H runs on a dedicated copy stream (default on).

    Returns:
        bool: True (default) for D2H-overlap; False for the default-stream FIFO
        fallback (``XTUNER_MUON_SWAP_OVERLAP=0``).
    """
    return os.environ.get("XTUNER_MUON_SWAP_OVERLAP", "1") == "1"


class _SwapCtx:
    """Lazy copy stream + cross-step event for D2H overlap.

    ``d2h_ev`` is re-recorded on the copy stream at the end of each D2H; the
    next H2D (default stream) waits it, so a step's D2H lands before the next
    step reads the same pinned-CPU buffer. It is primed at construction so the
    first step's wait is a no-op.
    """

    __slots__ = ("copy_s", "d2h_ev")

    def __init__(self) -> None:
        self.copy_s = DEVICE_MODULE.Stream()
        self.d2h_ev = DEVICE_MODULE.Event()
        with DEVICE_MODULE.stream(self.copy_s):
            self.d2h_ev.record()


_SWAP_CTX: _SwapCtx | None = None


def _get_ctx() -> _SwapCtx:
    """Return the lazy swap context, constructing it on first use.

    Returns:
        _SwapCtx: The process-wide swap context holding the copy stream.
    """
    global _SWAP_CTX
    if _SWAP_CTX is None:
        _SWAP_CTX = _SwapCtx()
    return _SWAP_CTX


def init_pinned_state(state: dict, param: Tensor, algo: str) -> None:
    """Replace a parameter's optimizer state buffers with pinned-CPU tensors.

    Called from ``Muon._get_or_initialize_state`` after the default device
    buffers are created, so the default (swap-off) init path is untouched and
    this hook is purely additive. The transient device alloc it replaces is
    once per parameter at init time and negligible.

    Args:
        state (dict): The optimizer state dict for ``param``.
        param (Tensor): The parameter tensor (may be a ``DTensor`` under FSDP2).
        algo (str): ``"muon"`` or ``"adamw"``.
    """
    local = param.to_local() if isinstance(param, DTensor) else param
    state["momentum"] = torch.zeros_like(local, device="cpu").pin_memory()
    if algo == "adamw":
        state["variance"] = torch.zeros_like(local, device="cpu").pin_memory()


def h2d_momentum(cpu_tensors: list[Tensor], *, device: torch.device) -> list[Tensor]:
    """Stage pinned-CPU momentum to fresh device temps on the default stream.

    H2D stays on the default stream (serial with the caller's compute): a
    cross-stream H2D write taints/fragments the temp and corrupts the compute
    read (verified). The leading ``wait_event(d2h_ev)`` orders this step's read
    of the pinned-CPU buffer after the prior step's D2H on the copy stream. With
    ``XTUNER_MUON_SWAP_OVERLAP=0`` this is the same default-stream FIFO path.

    Args:
        cpu_tensors (list[Tensor]): Pinned-CPU momentum tensors.
        device (torch.device): Target device.

    Returns:
        list[Tensor]: Device-resident momentum temps (default-allocated).
    """
    dev_temps = [torch.empty_like(m, device=device) for m in cpu_tensors]
    if not _use_copy_stream():
        for d, m in zip(dev_temps, cpu_tensors):
            d.copy_(m, non_blocking=True)
        return dev_temps
    default_s = DEVICE_MODULE.current_stream()
    ctx = _get_ctx()
    default_s.wait_event(ctx.d2h_ev)
    for d, m in zip(dev_temps, cpu_tensors):
        d.copy_(m, non_blocking=True)
    return dev_temps


def d2h_momentum(dev_tensors: list[Tensor], cpu_tensors: list[Tensor]) -> None:
    """Write device momentum back to pinned CPU on the copy stream.

    A compute-done event is recorded on the default stream, then the D2H runs on
    the copy stream gated by it (so D2H waits the in-place momentum update) and
    ``record_stream`` marks the default-allocated temp -- the docstring's
    reclaimable pattern (allocated on default, read on copy), so it returns to
    the default pool once the copy lands (no fragmentation). ``d2h_ev`` is
    re-recorded for the next step's H2D to wait -- overlapping the caller's
    subsequent comm (AGRS). With ``XTUNER_MUON_SWAP_OVERLAP=0`` the copy is
    same-stream FIFO on the default stream (serial).

    Args:
        dev_tensors (list[Tensor]): Device momentum temps (just updated).
        cpu_tensors (list[Tensor]): Pinned-CPU momentum sinks (written in place).
    """
    if not _use_copy_stream():
        for d, c in zip(dev_tensors, cpu_tensors):
            c.copy_(d, non_blocking=True)
        return
    ctx = _get_ctx()
    compute_ev = DEVICE_MODULE.Event()
    compute_ev.record()
    with DEVICE_MODULE.stream(ctx.copy_s):
        ctx.copy_s.wait_event(compute_ev)
        for d, c in zip(dev_tensors, cpu_tensors):
            c.copy_(d, non_blocking=True)
            d.record_stream(ctx.copy_s)
        ctx.d2h_ev.record()
