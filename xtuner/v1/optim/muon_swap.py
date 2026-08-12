"""Default-stream H2D/D2H for the Muon momentum swap (``XTUNER_MUON_SWAP``).

Gated solely by ``XTUNER_MUON_SWAP`` (default off). When on, the Muon
optimizer's momentum (and the AdamW ``m``/``v`` for non-Muon params) live on
pinned CPU; ``h2d_momentum`` stages them to a fresh per-update device temp and
``d2h_momentum`` writes them back -- both on the *default* compute stream, with
no dedicated copy stream, no events, no ``record_stream``, and no
``synchronize()``.

Why the default stream (not a dedicated copy stream + cross-stream events):
``torch_npu`` cross-stream ``Event`` joins are unreliable -- a
``default_s.wait_event(ev)`` where ``ev`` is recorded on a copy stream does not
reliably order the default stream after the copy stream's work. The H2D copy
lands on the copy stream, the compute that reads the device temp lands on the
default stream, and the join does not hold: the compute may run before the H2D
fills the temp (reads stale/empty memory -> nan params) and the D2H may not
drain before the next step's forward reuses a ``record_stream``'d block (marker
inheritance -> use-after-free -> nan grads). Both failure modes were reproduced
on the 13B run (step-2 optim nan; step-2 backward nan); a full
``synchronize()`` after every stage clears them (8 steps clean) but serializes
host launches and triples step time. Issuing H2D+D2H on the default stream
makes same-stream FIFO order them for free -- the H2D completes before the
compute reads it, the compute before the D2H reads it, and the D2H before the
next step's H2D reads the CPU buffer -- with no events, no ``record_stream``
(so no marker-inheritance race), and no ``synchronize``. This is the same
reliable pattern as ``swap_adamw_overlap``'s default (non-overlap) path, where
H2D+D2H and the ``d2h_ev`` all live on the default stream.

``non_blocking=True`` is kept so the host does not stall on each copy: the copy
is queued on the default stream and the host proceeds to launch the next op;
same-stream FIFO guarantees the downstream compute/comm reads ready data. The
H2D/D2H cost lands on the optim critical path (no within-task copy overlap);
the ``AsyncRuntime`` inter-task yields still overlap other params' comm.
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

    Allocates a device temp per CPU tensor and issues a non-blocking H2D copy on
    the default (caller's) stream. Same-stream FIFO orders the copy before the
    caller's compute, so no event join is needed. Replaces
    ``[m.to(device=..., non_blocking=True) for m in M]``.

    Args:
        cpu_tensors (list[Tensor]): Pinned-CPU momentum tensors.
        device (torch.device): Target device.

    Returns:
        list[Tensor]: Device-resident momentum temps.
    """
    dev_temps = [torch.empty_like(m, device=device) for m in cpu_tensors]
    for d, m in zip(dev_temps, cpu_tensors):
        d.copy_(m, non_blocking=True)
    return dev_temps


def d2h_momentum(dev_tensors: list[Tensor], cpu_tensors: list[Tensor]) -> None:
    """Write device momentum back to pinned CPU on the default stream.

    Issues non-blocking D2H copies on the default (caller's) stream. Same-stream
    FIFO orders the copies after the caller's compute (the in-place momentum
    update just ran on the same stream) and before the next step's H2D, so no
    event join or ``synchronize`` is needed. The device temps carry no
    ``record_stream`` marker, so there is no marker-inheritance reuse race.

    Args:
        dev_tensors (list[Tensor]): Device momentum temps (just updated).
        cpu_tensors (list[Tensor]): Pinned-CPU momentum sinks (written in place).
    """
    for d, c in zip(dev_tensors, cpu_tensors):
        c.copy_(d, non_blocking=True)
