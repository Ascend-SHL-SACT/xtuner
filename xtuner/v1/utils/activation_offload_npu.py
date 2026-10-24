# Copyright (c) OpenMMLab. All rights reserved.
"""Hook-based activation (hidden-state) offload on NPU.

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

Registration is chained: each ``register_activation_offload_npu_hooks`` call
wires one module chain (the GLM-5.2/5.3 text decoder layers, the GLM-5.3-Flash
vision tower's recompute-scope blocks, ...) under its own ``chain_id``, so
several chains coexist -- parking and refilling independently, without
colliding in the shared registry or the pinned-buffer cache.
"""

import os
from collections.abc import Sequence
from contextlib import AbstractContextManager
from functools import partial
from typing import Any, Callable

import torch

from xtuner.v1.utils import log_rank0
from xtuner.v1.utils.activation_offload import OffloadManager, SwapTensor


__all__ = [
    "activation_offload_npu_enabled",
    "make_activation_offload_npu_ctx",
    "offload_stream",
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
    model_layers: torch.nn.ModuleDict | Sequence[torch.nn.Module],
    first_offload_layer: int,
    chain_id: str = "text",
) -> None:
    """Register the hidden-state park/refill hooks on the raw decoder layers.

    Called once per module chain before activation checkpointing wraps its
    layers -- from GLM-5.2 ``_configure_model_specific_layers`` for the text
    decoder layers, and from the GLM-5.3-Flash vision tower for its
    recompute-scope blocks. The hooks keep the same module objects, so both
    fire inside the checkpoint boundary:

    - a pre-hook on every layer at or after ``first_offload_layer`` launches an
      async D2H copy of the layer-input hidden states into CPU pinned memory
      at the no_grad original forward entry (micro-batch lists park every
      element), and the matching post-hook releases the device storage without
      ordering compute behind the copy: the release records the copy stream on
      the tensor so the allocator defers reuse of the block until the copy has
      completed behind the layer's own compute (d2h doc 2.4);
    - the same pre-hook refills the storage at the grad-enabled replay entry,
      right before the replay's first op reads it, and prefetches the previous
      layer's refill so the H2D overlaps with the current layer's replayed
      compute; the pop-based registry makes every other call a no-op miss;
    - the highest *wrapped* offload-scope layer is the first one replayed in
      backward order and has no earlier refill to prefetch it. If it is also
      the highest registered layer, its post-hook issues the H2D back right
      away; otherwise the first registered layer above it -- unwrapped, hence
      running with grad enabled in the original forward -- issues that H2D from
      its own pre-hook. Either way the copy overlaps the forward-to-backward
      gap and the replay entry only waits on its event.

    Layers that never run under activation checkpointing stay inert: their
    original forward keeps grad enabled (no park) and they have no replay (no
    refill). Registering the same modules twice is an error: the second pair of
    hooks would park every layer twice.

    Args:
        model_layers (torch.nn.ModuleDict | Sequence[torch.nn.Module]): Decoder
            layers keyed by str(idx), or a plain sequence whose position is the
            layer index (e.g. a vision tower's ``nn.ModuleList`` blocks).
        first_offload_layer (int): First layer index whose input hidden states
            are in offload scope (mirrors the legacy window's
            ``layer_idx >= first_k_dense_replace`` scope; vision towers pass 0).
        chain_id (str): Registry namespace; each chain parks and refills
            independently and prefixes the pinned-buffer cache keys.
    """
    registered: list[int] = []
    # Duck-typed dispatch: a ModuleDict (or any str->module mapping) iterates (key, module)
    # pairs via .items(); a plain sequence (vision blocks) enumerates (position, module).
    # isinstance(x, Mapping) would NOT catch ModuleDict -- it is not registered with the ABC.
    items = list(model_layers.items() if hasattr(model_layers, "items") else enumerate(model_layers))
    layers = [(int(idx), layer) for idx, layer in items if int(idx) >= first_offload_layer]
    for layer_idx, layer in layers:
        if _has_offload_hooks(layer):
            raise RuntimeError(
                f"Activation offload (NPU) hooks are already registered on layer {layer_idx} of chain "
                f"{chain_id!r}; every module chain must be registered exactly once."
            )
    for layer_idx, layer in layers:
        layer.register_forward_hook(
            partial(_park_layer_input_after_forward, layer_idx=layer_idx, chain_id=chain_id), with_kwargs=True
        )
        layer.register_forward_pre_hook(
            partial(_forward_input_pre_hook, layer_idx=layer_idx, chain_id=chain_id), with_kwargs=True
        )
        registered.append(layer_idx)
    _chain_heads[chain_id] = max(registered) if registered else None
    log_rank0.info(
        f"Activation offload (NPU) registered on chain {chain_id!r} layers {registered}: layer-input hidden states "
        "are copied to CPU pinned memory asynchronously at the no_grad forward entry (launched before the layer "
        "computes, storage released after it) and refilled at replay entry with previous-layer prefetch; while "
        "this gate is on the legacy saved-tensors window no longer captures float tensors."
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


# Live offloads keyed by layer index, one pass per entry and one SwapTensor per
# micro-batch; each SwapTensor pins its hidden-state tensor alive until the
# replay-entry refill consumes it. Layer-index keying (not tensor identity)
# mirrors dsa_topk_offload_npu.py: the reentrant replay passes detach_variable
# copies, so id() does not survive into the replay while the shared storage
# does. Entries are keyed by (chain_id, layer_idx) so coexisting chains -- the
# text decoder layers and a vision tower -- never pop each other's offloads.
#
# A layer can hold several passes at once: under intra_layer_micro_batch > 1
# the vision tower runs once per sample (plus a dummy call for a text-only
# sample), so both passes park a block before either is replayed. Each pass
# keeps its own entry and pinned buffer -- one shared entry would let the second
# pass overwrite the first, whose replay then reads zero-sized storage.
_layer_offloads: dict[tuple[str, int], list[list[SwapTensor]]] = {}
# A layer re-parked past this many in-flight passes is accumulating passes that
# never reach their backward (forward_only calls, aborted steps): drop the
# oldest, which replaces the stale-entry drop of the single-entry registry.
_MAX_INFLIGHT_PASSES = 8
_offload_streams: dict[int, torch.cuda.Stream] = {}
# Per chain: the highest registered layer, set at registration. Only consulted
# when that layer is itself wrapped by activation checkpointing (a model whose
# last layer is the top of the chain, e.g. with MTP layers below the recompute
# cut); its post-hook then issues the H2D back, or the backward replay would
# start on a cold copy. When the top of the chain is *not* wrapped, the first
# registered layer above the wrapped range takes over that prefetch from its
# pre-hook (see _ensure_hidden_resident), because registration cannot know the
# wrap set: the recompute decision depends on the FSDP config, which is not
# available until fully_shard.
_chain_heads: dict[str, int | None] = {}


def _park_layer_input_after_forward(
    module: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    _output: object,
    *,
    layer_idx: int,
    chain_id: str,
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
    _park_finish(chain_id, layer_idx)


def _forward_input_pre_hook(
    module: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    layer_idx: int,
    chain_id: str,
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
    hidden_states = args[0] if args else kwargs.get("hidden_states")
    if torch.is_grad_enabled():
        _ensure_hidden_resident(chain_id, layer_idx, hidden_states)
        return
    if hidden_states is None:
        return
    _park_launch(chain_id=chain_id, layer_idx=layer_idx, hidden_states=hidden_states)


@torch.compiler.disable
def _park_launch(*, chain_id: str, layer_idx: int, hidden_states: torch.Tensor | list[torch.Tensor]) -> None:
    # Start parking one layer's input hidden states: asynchronously copy the
    # device storage to the (pinned, cached) CPU buffer on the offload stream,
    # on the checkpoint original forward only (no_grad pre-hook). The device
    # storage stays alive while the layer computes and is released by
    # _park_finish once the copy is done. The tensor objects stay referenced
    # by the checkpoint frames; _ensure_hidden_resident restores their storage
    # before the replay reads them.
    #
    # Each park takes the next pass index of the layer (see _layer_offloads). A
    # pass that never reaches its backward -- a forward_only pass or a step
    # aborted between forward and backward -- cannot be dropped at the next
    # park without breaking its own replay, so it is bounded instead: past
    # _MAX_INFLIGHT_PASSES the oldest pass is abandoned.
    passes = _layer_offloads.setdefault((chain_id, layer_idx), [])
    if len(passes) >= _MAX_INFLIGHT_PASSES:
        dropped = passes.pop(0)
        log_rank0.warning(
            f"Activation offload (NPU): layer {layer_idx} of chain {chain_id!r} held {_MAX_INFLIGHT_PASSES} "
            f"passes; dropped the oldest ({len(dropped)} parked tensor(s)) whose replay cannot run."
        )
    pass_idx = len(passes)
    swap_tensors: list[SwapTensor] = []
    # Register before parking so a mid-loop failure cannot strand a launched
    # copy without a visible owner: the partial entry stays visible to
    # _park_finish and _ensure_hidden_resident, and the next consumption
    # refills whatever was launched.
    passes.append(swap_tensors)
    for mb_idx, hidden in enumerate(hidden_states if isinstance(hidden_states, list) else [hidden_states]):
        # Reuse the activation-offload pinned-buffer cache, keyed per chain,
        # layer, pass and micro-batch slot; shape or dtype drift reallocates.
        # The pass is part of the key because a second in-flight pass must not
        # write over the copy the first one still has to be refilled from.
        key = f"{chain_id}_act_hidden_{layer_idx}_{pass_idx}_{mb_idx}"
        tensor_cpu = OffloadManager().get_or_create_pin_memory(key, hidden.shape, hidden.dtype)
        swap_tensor = SwapTensor(hidden, key, tensor_cpu=tensor_cpu)
        # d2h doc 2.2: one ordering gate per copy. launch_d2h records an event on
        # the current stream and makes the offload stream wait on it before the
        # copy, so an extra wait_stream here would be a redundant second gate.
        swap_tensor.launch_d2h(offload_stream(hidden.device))
        swap_tensors.append(swap_tensor)


@torch.compiler.disable
def _park_finish(chain_id: str, layer_idx: int) -> None:
    # Complete the evacuation _park_launch started: release the device storage
    # without ordering compute behind the in-flight D2H copies (d2h doc 2.4:
    # record_stream makes the allocator defer reuse of the released block until
    # the offload stream's copy completes, so compute keeps launching and any
    # later allocation simply gets a different block). The registry entry is
    # kept: the replay refill pops it. Swaps whose copy never launched (stat
    # "device") are skipped by wait_d2h_finished, so partial parks finish
    # consistently.
    # Park and finish are the adjacent pre/post hooks of one module invocation
    # on one thread, so the pass this invocation parked is the last one.
    passes = _layer_offloads.get((chain_id, layer_idx))
    if not passes:
        return
    swap_tensors = passes[-1]
    if not swap_tensors:
        return
    for swap_tensor in swap_tensors:
        stream = offload_stream(swap_tensor.tensor.device)
        if swap_tensor.stat == "host":
            swap_tensor.tensor.record_stream(stream)
        swap_tensor.wait_d2h_finished(stream, False)
    if layer_idx == _chain_heads.get(chain_id):
        # Chain head, only reached when the highest registered layer is itself
        # wrapped: the first replayed layer has no earlier refill to prefetch
        # it, so its replay entry would block on a cold H2D. Issue the H2D back
        # immediately: it overlaps the forward-to-backward gap and the owning
        # refill degrades to an event wait (stat is no longer "host"). When the
        # registered head is not wrapped this branch never runs; the first
        # registered layer above the recompute cut prefetches it instead.
        # The D2H may still be in flight, but the prefetch's resize re-extends
        # the storage into a block of its own -- the in-flight D2H's source
        # block stays deferred by record_stream above, so the copies cannot
        # alias.
        for swap_tensor in swap_tensors:
            if swap_tensor.stat == "host":
                swap_tensor.prefetch_launch_h2d(offload_stream(swap_tensor.tensor.device), True)


@torch.compiler.disable
def _ensure_hidden_resident(
    chain_id: str, layer_idx: int, hidden_states: torch.Tensor | list[torch.Tensor] | None
) -> None:
    # Restore the device storage of one layer's input hidden states at the
    # replay entry and kick off the previous layer's H2D prefetch. The
    # backward replay re-enters layers in reverse order; each layer's pre-hook
    # pops its own entry, so every other call (original forwards,
    # already-consumed layers, inference) is a no-op miss.
    #
    # The replay tensor is a detached view of the parked tensor and shares its
    # StorageImpl, whose ``_cdata`` survives the park's ``resize_(0)``, so the
    # pass this replay owns is selected by storage identity: a layer can hold
    # several in-flight passes, and restoring another pass's copy would hand
    # this replay the wrong data.
    #
    # The previous-layer prefetch runs before the miss return on purpose. The
    # highest *wrapped* layer never gets prefetched by a later replay (it is
    # replayed first), so it must be prefetched in the forward pass. When the
    # top of the chain is itself wrapped, its own post-hook does that
    # (_park_finish chain-head); when it is not -- e.g. no MTP layers, where
    # the recompute cut always excludes the last decoder layer -- the first
    # registered layer above the wrapped range is the model's next layer, it
    # runs with grad enabled in the original forward, calls this pre-hook and
    # misses (unwrapped layers never park), and this prefetch is the only
    # callback between the head's park and the backward replay.
    tensors = hidden_states if isinstance(hidden_states, list) else [hidden_states]
    wanted = {tensor.untyped_storage()._cdata for tensor in tensors if isinstance(tensor, torch.Tensor)}
    passes = _layer_offloads.get((chain_id, layer_idx))
    swap_tensors: list[SwapTensor] | None = None
    matched_pass: int | None = None
    if passes:
        for pass_idx, pass_tensors in list(enumerate(passes)):
            hits = [swap for swap in pass_tensors if swap.tensor.untyped_storage()._cdata in wanted]
            if not hits:
                continue
            matched_pass = pass_idx
            swap_tensors = hits
            rest = [swap for swap in pass_tensors if swap not in hits]
            if rest:
                passes[pass_idx] = rest
            else:
                passes.pop(pass_idx)
            break
        if not passes:
            del _layer_offloads[(chain_id, layer_idx)]
    prev_passes = _layer_offloads.get((chain_id, layer_idx - 1))
    if prev_passes:
        # Same pass as the copy this replay owns, so only that pass is brought
        # back early; a miss falls back to every in-flight pass.
        for pass_idx, pass_tensors in enumerate(prev_passes):
            if matched_pass is not None and pass_idx != matched_pass:
                continue
            for swap_tensor in pass_tensors:
                if swap_tensor.stat == "host":
                    # Async H2D on the offload stream (records h2d_event and
                    # keeps the storage alive for the copy via record_stream).
                    swap_tensor.prefetch_launch_h2d(offload_stream(swap_tensor.tensor.device), True)
    if swap_tensors is None:
        return

    for swap_tensor in swap_tensors:
        stream = offload_stream(swap_tensor.tensor.device)
        working_stream = torch.cuda.current_stream(swap_tensor.tensor.device)
        if swap_tensor.stat != "host":
            # Prefetched (or never evacuated): launch_h2d's waiting path
            # orders the working stream behind the copy's event, or is a
            # no-op on an event that was never recorded.
            swap_tensor.launch_h2d(stream, True, working_stream)
            continue
        # Not prefetched: blocking H2D on the offload stream, same side-stream
        # choreography as dsa_topk_offload_npu.py. Only reachable as a
        # fallback now — the highest wrapped layer is prefetched either by its
        # own post-hook or by the pre-hook of the unwrapped layer above it, so
        # a first replayed layer normally takes the event-wait path above.
        # d2h doc 3.5/3.6: re-extend the H2D target block on the working
        # stream, never inside the copy-stream context (cross-stream pool
        # stranding); launch_h2d(flag=False) below then only copies.
        stream.wait_stream(working_stream)
        swap_tensor.tensor.storage().resize_(swap_tensor.storage_size)
        with torch.cuda.stream(stream):
            swap_tensor.launch_h2d(stream, False, stream)
        working_stream.wait_stream(stream)


def offload_stream(device: torch.device) -> torch.cuda.Stream:
    """Get the single offload stream shared by every copy of a device.

    The activation offload and the DSA top-k offload issue all their D2H/H2D
    copies on the same per-device stream, so the two mechanisms' DMA traffic is
    serialized instead of contending on two concurrent streams (d2h doc 4.4).

    Args:
        device (torch.device): Device whose offload stream is requested.

    Returns:
        torch.cuda.Stream: Lazily created, per-device shared offload stream.
    """
    device_idx = torch.cuda.current_device() if device.index is None else device.index
    if device_idx not in _offload_streams:
        _offload_streams[device_idx] = torch.cuda.Stream(device=device_idx)
    return _offload_streams[device_idx]


def _has_offload_hooks(layer: torch.nn.Module) -> bool:
    # Private: idempotence guard for register_activation_offload_npu_hooks. A
    # second registration would stack a second park/refill pair on every layer,
    # double the D2H traffic and let the pop-based registry silently drop one of
    # the two entries, so detect our own partial hooks before wiring again.
    hooks = [*layer._forward_pre_hooks.values(), *layer._forward_hooks.values()]
    ours = (_forward_input_pre_hook, _park_layer_input_after_forward)
    return any(getattr(hook, "func", hook) in ours for hook in hooks)
