"""D2H-overlap copy stream for the Muon momentum swap (``XTUNER_MUON_SWAP``).

Gated by ``XTUNER_MUON_SWAP`` (default off). When on, the Muon optimizer's
momentum (and the AdamW ``m``/``v`` for non-Muon params) live on pinned CPU;
``h2d_momentum`` stages them to a fresh device temp and ``d2h_momentum`` writes
them back. The two stream placements are selected by one env var:

- combo1 (``XTUNER_MUON_SWAP_OVERLAP=1``, default): H2D on the default stream,
  D2H on a dedicated copy stream so it overlaps the caller's subsequent AGRS
  collectives on the default stream.
- combo2 (``XTUNER_MUON_SWAP_OVERLAP=0``): H2D and D2H both on the default
  stream (FIFO, no copy stream, no events) -- the serial path, kept for A/B.

H2D cannot move to the copy stream: a copy-stream write of the device temp
taints the default-stream EMA's read and produces nan on the second optim
iteration, even with a correct done-event sync. This was re-verified on
torch_npu 2.13.0.rc2 (18-run A/B: all four H2D-on-copy configs nan at step 2,
synced == naive == nan) and earlier on an older build (three ways: a fresh
copy-stream-allocated temp is never reused by the caching allocator and OOMs by
step ~4; a default-allocated temp written on the copy stream corrupts the
default stream's compute read and produces nan on the second optim iteration
(whether or not ``record_stream`` is used); and a persistent buffer with
``storage().resize_(0/full)`` reintroduces the same nan (the host-synchronous
``resize_(0)`` frees storage while the async D2H copy_ read is still in flight,
and cross-batch default-stream allocations reuse the freed storage). So H2D
stays on the default stream (serial, no cross-stream write, no taint, no nan),
and only the D2H -- a cross-stream *read* of a default-allocated block via
``record_stream`` -- goes on the copy stream. That read is the reclaimable
pattern (allocated on default, read on copy): no fragmentation, no OOM. The D2H
overlaps the caller's subsequent AGRS comm.

The cross-step ``d2h_ev`` (re-recorded each D2H, primed at construction) is
waited on the default stream at the next H2D: a step's D2H (copy stream) writes
the pinned-CPU buffer that the next step's H2D (default stream) reads, so the
default stream must wait the prior D2H. In combo2 (same default stream) ordering
is automatic and no event is recorded.
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


_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def cpu_momentum_dtype() -> torch.dtype | None:
    """Return the pinned-CPU momentum dtype override, or None to keep the native dtype.

    ``XTUNER_MUON_SWAP_CPU_DTYPE`` is opt-in: unset (default), ``float32``, or
    ``off``/``none``/``disable``/``0``/empty keep the native fp32 swap,
    byte-identical to HEAD. ``bfloat16`` stores the pinned-CPU momentum (and
    AdamW variance) as bf16 -- halving pinned memory and H2D/D2H bandwidth per
    step -- while the device momentum stays fp32 so the EMA still accumulates in
    fp32. The fp32<->bf16 cast is folded into the H2D/D2H ``copy_`` by
    torch_npu's mixed-dtype non_blocking device-side cast (commit e87c8a73c);
    builds without it (e.g. 2.9.x, 2.12.0) fall back to a host-synchronous CPU
    cast -- functionally correct but slower (measured ~20% TGS below the fp32
    path on 2.9.1), so the default stays off. Verified A/B (13B run97-ON vs
    run98-OFF, torch_npu 2.13.0.rc2): step1 loss byte-identical, |dloss|<7e-4
    (bf16 noise), max_memory unchanged, no nan/inf. Requires fp32 master params
    (the XTuner training default: ``BaseModel.fully_shard`` upcasts trainable
    params to fp32 before FSDP2, so the optimizer-visible shard is fp32 even
    with ``param_dtype=bf16``).

    Returns:
        torch.dtype | None: The override dtype, or None to keep native fp32.
    """
    v = os.environ.get("XTUNER_MUON_SWAP_CPU_DTYPE")
    if v is None:
        return None
    vl = v.lower()
    if vl in ("", "off", "none", "disable", "0", "float32"):
        return None
    if vl not in _DTYPE_MAP:
        raise ValueError(
            f"XTUNER_MUON_SWAP_CPU_DTYPE={v!r} not recognized; use bfloat16/float32 or off/none/disable/0 to disable"
        )
    return _DTYPE_MAP[vl]


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
    # XTUNER_MUON_SWAP_CPU_DTYPE=bfloat16 (opt-in): allocate the pinned-CPU state
    # DIRECTLY at the override dtype (halved pinned memory + H2D/D2H bandwidth;
    # no transient full-size fp32 pinned alloc and no double pin). The device
    # temp stays fp32 so the EMA accumulates in fp32, and the fp32<->bf16 cast
    # is folded into the H2D/D2H copy_ by torch_npu's mixed-dtype non_blocking
    # device-side cast (commit e87c8a73c). No-op when the override dtype already
    # matches the native shard dtype; byte-identical to HEAD's zeros_like when
    # the override is None (unset) or float32.
    cpu_dt = cpu_momentum_dtype()
    if cpu_dt is not None and cpu_dt != local.dtype:
        if local.dtype != torch.float32:
            raise ValueError(
                f"XTUNER_MUON_SWAP_CPU_DTYPE={cpu_dt} requires fp32 master "
                f"params (requires_grad=True upcasts to fp32), got {local.dtype}"
            )
    state_dt = local.dtype if cpu_dt is None else cpu_dt
    state["momentum"] = torch.zeros(local.shape, dtype=state_dt, device="cpu").pin_memory()
    if algo == "adamw":
        state["variance"] = torch.zeros(local.shape, dtype=state_dt, device="cpu").pin_memory()


def h2d_momentum(cpu_tensors: list[Tensor], *, device: torch.device) -> list[Tensor]:
    """Stage pinned-CPU momentum to fresh device temps.

    H2D always runs on the default stream (a copy-stream write would taint the
    temp the default-stream EMA reads -- see the module docstring). combo1
    (``OVERLAP=1``, default) waits the prior step's D2H event first; combo2
    (``OVERLAP=0``) is same-stream FIFO and needs no event.

    Args:
        cpu_tensors (list[Tensor]): Pinned-CPU momentum tensors.
        device (torch.device): Target device.

    Returns:
        list[Tensor]: Device-resident momentum temps (default-allocated).
    """
    # Per-tensor device dtype (no env read on this per-step path): an fp32
    # device temp whenever the pinned-CPU momentum was narrowed below fp32 by
    # XTUNER_MUON_SWAP_CPU_DTYPE (under the fp32-master contract only the bf16
    # override does this) -- copy_ below then fires torch_npu's mixed-dtype
    # non_blocking device-side cast (commit e87c8a73c), so the EMA accumulates
    # in fp32 while the H2D DMA moves bf16 (half bandwidth). fp32 CPU momentum
    # (the unset default) keeps the native empty_like path, identical to HEAD.
    dev_temps = [
        torch.empty(m.shape, dtype=torch.float32, device=device)
        if m.dtype != torch.float32
        else torch.empty_like(m, device=device)
        for m in cpu_tensors
    ]
    if not _use_copy_stream():
        # combo2: default-stream FIFO, no events (identical to HEAD when
        # XTUNER_MUON_SWAP_CPU_DTYPE is unset/float32). Same-stream ordering is
        # automatic.
        for d, m in zip(dev_temps, cpu_tensors):
            d.copy_(m, non_blocking=True)
        return dev_temps
    # combo1: H2D on the default stream; wait the prior-step D2H (copy stream)
    # before reading the pinned-CPU buffer (identical to HEAD when
    # XTUNER_MUON_SWAP_CPU_DTYPE is unset/float32).
    default_s = DEVICE_MODULE.current_stream()
    ctx = _get_ctx()
    default_s.wait_event(ctx.d2h_ev)
    for d, m in zip(dev_temps, cpu_tensors):
        d.copy_(m, non_blocking=True)
    return dev_temps


def d2h_momentum(dev_tensors: list[Tensor], cpu_tensors: list[Tensor]) -> None:
    """Write device momentum back to pinned CPU.

    combo1 (``OVERLAP=1``, default): D2H on the copy stream. A compute-done
    event recorded on the default stream gates it (D2H waits the in-place
    momentum update); ``record_stream`` marks the default-allocated temp --
    the reclaimable pattern (allocated on default, read on copy), so it
    returns to the default pool once the copy lands (no fragmentation);
    ``d2h_ev`` is re-recorded for the next step's H2D, overlapping the
    caller's subsequent comm (AGRS). combo2 (``OVERLAP=0``): same-stream FIFO
    on the default stream (serial).

    Args:
        dev_tensors (list[Tensor]): Device momentum temps (just updated).
        cpu_tensors (list[Tensor]): Pinned-CPU momentum sinks (written in place).
    """
    if not _use_copy_stream():
        # combo2: default-stream FIFO, no events (identical to HEAD when
        # XTUNER_MUON_SWAP_CPU_DTYPE is unset/float32).
        for d, c in zip(dev_tensors, cpu_tensors):
            c.copy_(d, non_blocking=True)
        return
    # combo1: D2H on the copy stream (identical to HEAD when
    # XTUNER_MUON_SWAP_CPU_DTYPE is unset/float32). compute_ev orders D2H after the in-place
    # momentum update; record_stream marks the default-allocated temp
    # (reclaimable, no fragmentation); d2h_ev is re-recorded for the next
    # step's H2D -- overlapping the caller's AGRS.
    ctx = _get_ctx()
    compute_ev = DEVICE_MODULE.Event()
    compute_ev.record()
    with DEVICE_MODULE.stream(ctx.copy_s):
        ctx.copy_s.wait_event(compute_ev)
        for d, c in zip(dev_tensors, cpu_tensors):
            c.copy_(d, non_blocking=True)
            d.record_stream(ctx.copy_s)
        ctx.d2h_ev.record()
