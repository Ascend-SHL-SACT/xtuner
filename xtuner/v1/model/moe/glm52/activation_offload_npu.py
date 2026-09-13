# Copyright (c) OpenMMLab. All rights reserved.
"""Hook-based activation (hidden-state) offload for GLM-5.2 on NPU.

The legacy ``XTUNER_ACTIVATION_OFFLOAD`` path offloads each offload-scope
layer's input hidden states through the ``async_save_on_cpu`` saved-tensors
window. Under the reentrant checkpointing used by GLM-5.2 the window does pack
those hidden states (the checkpoint frame saves its tensor inputs), but the
``OffloadManager`` bookkeeping strands device memory exactly where the step's
allocated peak sits:

- block 0's ``may_npu_tensors`` entry is removed only by the
  ``del_may_npu_tensor`` call for ``block_idx = -1``, which never runs, so the
  first offloaded layer's hidden states stay referenced through the rest of
  backward and into the next step;
- the final block's ``items`` entries are consumed only by a later block's
  prefetch, which does not exist, so the last offloaded layer's hidden states
  stay referenced from their forward until the same layer packs again in the
  next step;
- the final block is also never evacuated during forward (evacuation is
  triggered by the next block's pack), so it is not offloaded at all.

This module replaces the hidden-state path with the park/refill hook pattern
of ``dsa_topk_offload_npu.py``, split for transfer overlap: a forward pre-hook
launches an async D2H copy of the layer input on the no_grad original forward
(the copy hides behind the layer's own compute) and the matching post-hook
releases the device storage once the copy is done; the same pre-hook refills
the storage at the grad-enabled replay entry, which is exactly where the
replay's first op consumes it, while prefetching the previous layer's refill
so its H2D overlaps with the current replay. The refill pops the registry
entry, so no reference can outlive consumption. Integer DSA ids tensors keep
flowing through the legacy window untouched (see
``make_activation_offload_npu_ctx``).

Gated by ``XTUNER_ACTIVATION_OFFLOAD_NPU=1``. While the gate is on, this
mechanism owns the hidden states independently of ``XTUNER_ACTIVATION_OFFLOAD``
(the legacy env then only affects block-index counting and the ids window).
"""

import os
from contextlib import AbstractContextManager
from functools import partial
from typing import Any, Callable

import torch

from xtuner.v1.utils import log_rank0
from xtuner.v1.utils.activation_offload import OffloadManager, SwapTensor


__all__ = [
    "activation_offload_npu_enabled",
    "make_activation_offload_npu_ctx",
    "register_activation_offload_npu_hooks",
]


def activation_offload_npu_enabled() -> bool:
    """Whether XTUNER_ACTIVATION_OFFLOAD_NPU enables the hook-based activation
    offload.

    Returns:
        bool: True when XTUNER_ACTIVATION_OFFLOAD_NPU is set to "1".
    """
    return os.getenv("XTUNER_ACTIVATION_OFFLOAD_NPU", "0") == "1"


def register_activation_offload_npu_hooks(
    *,
    model_layers: torch.nn.ModuleDict,
    first_offload_layer: int,
) -> None:
    """Register the hidden-state park/refill hooks on the raw decoder layers.

    Called once from GLM-5.2 ``_configure_model_specific_layers``, before
    activation checkpointing wraps the layers. The hooks keep the same module
    objects, so both fire inside the checkpoint boundary:

    - a pre-hook on every layer at or after ``first_offload_layer`` launches an
      async D2H copy of the layer-input hidden states into CPU pinned memory
      at the no_grad original forward entry (micro-batch lists park every
      element), and the matching post-hook releases the device storage once
      the copy has completed behind the layer's own compute;
    - the same pre-hook refills the storage at the grad-enabled replay entry,
      right before the replay's first op reads it, and prefetches the previous
      layer's refill so the H2D overlaps with the current layer's replayed
      compute; the pop-based registry makes every other call a no-op miss;
    - the highest offload-scope layer is the first one replayed in backward
      order and has no earlier refill to prefetch it, so its post-hook issues
      the H2D back as soon as the D2H completes: the copy overlaps the
      forward-to-backward gap and the replay entry only waits on its event.

    Layers that never run under activation checkpointing stay inert: their
    original forward keeps grad enabled (no park) and they have no replay (no
    refill).

    Args:
        model_layers (torch.nn.ModuleDict): Decoder layers keyed by str(idx).
        first_offload_layer (int): First layer index whose input hidden states
            are in offload scope (mirrors the legacy window's
            ``layer_idx >= first_k_dense_replace`` scope).
    """
    registered: list[int] = []
    for idx, layer in model_layers.items():
        layer_idx = int(idx)
        if layer_idx < first_offload_layer:
            continue
        layer.register_forward_hook(partial(_park_layer_input_after_forward, layer_idx=layer_idx), with_kwargs=True)
        layer.register_forward_pre_hook(partial(_forward_input_pre_hook, layer_idx=layer_idx), with_kwargs=True)
        registered.append(layer_idx)
    global _chain_head_layer
    _chain_head_layer = max(registered) if registered else None
    log_rank0.info(
        f"Activation offload (NPU) registered on layers {registered}: layer-input hidden states are copied "
        "to CPU pinned memory asynchronously at the no_grad forward entry (launched before the layer computes, "
        "storage released after it) and refilled at replay entry with previous-layer prefetch; while this "
        "gate is on the legacy saved-tensors window no longer captures float tensors."
    )


def make_activation_offload_npu_ctx(
    original_ctx: Callable[[int, list[torch.Tensor]], AbstractContextManager],
) -> Callable[[int, list[torch.Tensor]], AbstractContextManager]:
    """Build the ``_saved_tensors_offload_ctx`` shadow used while the gate is
    on.

    The hook mechanism owns every float (hidden-state) tensor, so the shadow
    filters them out of the legacy window; integer DSA ids tensors listed by
    ``XTUNER_DSA_TOPK_OFFLOAD`` keep flowing through ``original_ctx``
    unchanged.

    Args:
        original_ctx (Callable[[int, list[torch.Tensor]], AbstractContextManager]):
            The bound ``_saved_tensors_offload_ctx`` captured before shadowing.

    Returns:
        Callable[[int, list[torch.Tensor]], AbstractContextManager]: Window
            builder with the same signature as ``original_ctx``.
    """

    def _ctx(block_idx: int, tensors: list[torch.Tensor]) -> AbstractContextManager:
        return original_ctx(block_idx, [tensor for tensor in tensors if not tensor.is_floating_point()])

    return _ctx


# Live offloads keyed by layer index, one SwapTensor per micro-batch; each
# SwapTensor pins its hidden-state tensor alive until the replay-entry refill
# consumes the entry. Layer-index keying (not tensor identity) mirrors
# dsa_topk_offload_npu.py: the reentrant replay passes detach_variable copies,
# so id() does not survive into the replay while the shared storage does.
_layer_offloads: dict[int, list[SwapTensor]] = {}
_offload_streams: dict[int, torch.cuda.Stream] = {}
# The highest offload-scope layer, set at registration. The backward replay
# re-enters layers in reverse order, so this layer's refill is the first one
# and no earlier refill exists to prefetch it: its post-hook issues the H2D
# back instead, or the replay entry would block on a cold copy.
_chain_head_layer: int | None = None


def _park_layer_input_after_forward(
    module: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    _output: object,
    *,
    layer_idx: int,
) -> None:
    # Forward-hook body (with_kwargs). Completes the evacuation that the
    # pre-hook started (training + no_grad = reentrant original forward). By
    # the time this hook runs, the layer has consumed its input hidden states
    # and nothing else can read them until the replay refill: the original
    # forward ran under no_grad so no op saved them, and the caller's
    # reference dies with the layer call. The grad-enabled recompute replay
    # skips this hook; the next pre-hook refills.
    if not module.training or torch.is_grad_enabled():
        return
    _park_finish(layer_idx)


def _forward_input_pre_hook(
    module: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    layer_idx: int,
) -> None:
    # Forward-pre-hook body (with_kwargs). Two roles, told apart by grad mode:
    # - no_grad pass = checkpoint original forward: launch the async D2H copy
    #   of the layer input so it overlaps with the whole layer's compute (the
    #   copy only reads; nothing in the layer mutates its input in place, the
    #   same premise the upstream saved-tensors window relies on).
    # - grad-enabled pass = checkpoint replay, whose first op consumes the
    #   input hidden states immediately, so the replay entry is the
    #   consumption point: refill here (the pop-based registry makes every
    #   other call a no-op miss). If the previous layer's entry was already
    #   prefetched by the last refill, this is only an event wait.
    if not module.training:
        return
    if torch.is_grad_enabled():
        _ensure_hidden_resident(layer_idx)
        return
    hidden_states = args[0] if args else kwargs.get("hidden_states")
    if hidden_states is None:
        return
    _park_launch(layer_idx=layer_idx, hidden_states=hidden_states)


@torch.compiler.disable
def _park_launch(*, layer_idx: int, hidden_states: torch.Tensor | list[torch.Tensor]) -> None:
    # Start parking one layer's input hidden states: asynchronously copy the
    # device storage to the (pinned, cached) CPU buffer on the offload stream,
    # on the checkpoint original forward only (no_grad pre-hook). The device
    # storage stays alive while the layer computes and is released by
    # _park_finish once the copy is done. The tensor objects stay referenced
    # by the checkpoint frames; _ensure_hidden_resident restores their storage
    # before the replay reads them.
    #
    # One layer owns one live offload entry at a time. Reusing it while still
    # active would overwrite the earlier D2H result, so fail loudly if the
    # scheduling contract is violated (mirrors dsa_topk_offload_npu.py).
    if layer_idx in _layer_offloads:
        raise RuntimeError(f"Activation offload for layer {layer_idx} is still active.")

    swap_tensors: list[SwapTensor] = []
    # Register before parking so a mid-loop failure cannot strand a launched
    # copy without a visible owner: the partial entry stays visible to
    # _park_finish and _ensure_hidden_resident, and the next consumption
    # refills whatever was launched.
    _layer_offloads[layer_idx] = swap_tensors
    for mb_idx, hidden in enumerate(hidden_states if isinstance(hidden_states, list) else [hidden_states]):
        # Reuse the activation-offload pinned-buffer cache, keyed per layer and
        # micro-batch slot; shape or dtype drift reallocates.
        key = f"glm52_act_hidden_{layer_idx}_{mb_idx}"
        tensor_cpu = OffloadManager().get_or_create_pin_memory(key, hidden.shape, hidden.dtype)
        swap_tensor = SwapTensor(hidden, key, tensor_cpu=tensor_cpu)
        stream = _offload_stream(hidden.device)
        stream.wait_stream(torch.cuda.current_stream(hidden.device))
        swap_tensor.launch_d2h(stream)
        swap_tensors.append(swap_tensor)


@torch.compiler.disable
def _park_finish(layer_idx: int) -> None:
    # Complete the evacuation _park_launch started: order the compute streams
    # behind the D2H copies (any later reuse of the released storage by
    # compute is therefore stream-ordered after the copies read it) and
    # release the device storage. The registry entry is kept: the replay
    # refill pops it. Swaps whose copy never launched (stat "device") are
    # skipped by wait_d2h_finished, so partial parks finish consistently.
    swap_tensors = _layer_offloads.get(layer_idx)
    if not swap_tensors:
        return
    for swap_tensor in swap_tensors:
        swap_tensor.wait_d2h_finished(_offload_stream(swap_tensor.tensor.device), True)
    if layer_idx == _chain_head_layer:
        # Chain head: the first replayed layer has no earlier refill to
        # prefetch it, so its replay entry would block on a cold H2D. The D2H
        # copy is complete by now (wait_d2h_finished above), so issue the H2D
        # back immediately: it overlaps the forward-to-backward gap and the
        # owning refill degrades to an event wait (stat is no longer "host").
        for swap_tensor in swap_tensors:
            if swap_tensor.stat == "host":
                swap_tensor.prefetch_launch_h2d(_offload_stream(swap_tensor.tensor.device), True)


@torch.compiler.disable
def _ensure_hidden_resident(layer_idx: int) -> None:
    # Restore the device storage of one layer's input hidden states at the
    # replay entry and kick off the previous layer's H2D prefetch. The
    # backward replay re-enters layers in reverse order; each layer's pre-hook
    # pops its own entry, so every other call (original forwards,
    # already-consumed layers, inference) is a no-op miss. Prefetching the
    # previous layer here lets its H2D overlap with this layer's replayed
    # compute; the owning refill then only waits on the copy's event.
    swap_tensors = _layer_offloads.pop(layer_idx, None)
    if swap_tensors is None:
        return

    prev_entry = _layer_offloads.get(layer_idx - 1)
    if prev_entry:
        for swap_tensor in prev_entry:
            if swap_tensor.stat == "host":
                # Async H2D on the offload stream (records h2d_event and
                # keeps the storage alive for the copy via record_stream).
                swap_tensor.prefetch_launch_h2d(_offload_stream(swap_tensor.tensor.device), True)

    for swap_tensor in swap_tensors:
        stream = _offload_stream(swap_tensor.tensor.device)
        working_stream = torch.cuda.current_stream(swap_tensor.tensor.device)
        if swap_tensor.stat != "host":
            # Prefetched (or never evacuated): launch_h2d's waiting path
            # orders the working stream behind the copy's event, or is a
            # no-op on an event that was never recorded.
            swap_tensor.launch_h2d(stream, True, working_stream)
            continue
        # Not prefetched: blocking H2D on the offload stream, same side-stream
        # choreography as dsa_topk_offload_npu.py. Only reachable as a
        # fallback now — the chain head prefetches itself in its post-hook, so
        # a first replayed layer normally takes the event-wait path above.
        stream.wait_stream(working_stream)
        with torch.cuda.stream(stream):
            swap_tensor.launch_h2d(stream, True, stream)
        working_stream.wait_stream(stream)


def _offload_stream(device: torch.device) -> torch.cuda.Stream:
    device_idx = torch.cuda.current_device() if device.index is None else device.index
    if device_idx not in _offload_streams:
        _offload_streams[device_idx] = torch.cuda.Stream(device=device_idx)
    return _offload_streams[device_idx]
