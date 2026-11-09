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
from collections import OrderedDict, deque
from collections.abc import Sequence
from contextlib import AbstractContextManager
from functools import partial
from typing import TYPE_CHECKING, Any, Callable

import torch

from xtuner.v1.utils import log_rank0
from xtuner.v1.utils.activation_offload import OffloadManager, SwapTensor


if TYPE_CHECKING:
    from xtuner.v1.ops.attn_imp import AttnOpOutputs


__all__ = [
    "activation_offload_npu_enabled",
    "make_activation_offload_npu_ctx",
    "offload_stream",
    "register_activation_offload_npu_hooks",
    "sfa_stash_key",
    "sfa_stash_pop",
    "sfa_stash_prefetch",
    "sfa_stash_push",
    "vit_flash_attn_skip_recompute",
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


def vit_flash_attn_skip_recompute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: Any,
    max_seqlen_k: Any,
    softmax_scale: float,
    deterministic: bool,
    module: Any,
) -> "AttnOpOutputs":
    """NPU ViT flash attention with the checkpoint recompute skipped.

    Drop-in for ``attn_imp.flash_attention`` for the GLM-5.3-Flash vision tower on NPU (same
    BNSD ``[1, heads, total_seq, dim]`` input views, same ``{"raw_output": [1, T, H, D]}``
    return shape), but the attention runs through ``_VitFlashAttnSkipRecompute``: the no_grad
    checkpoint forward stashes ``(attn_output, softmax_max, softmax_sum)`` as device tensors
    and the grad-enabled replay restores them, so the backward feeds
    ``npu_fusion_attention_grad`` with the exact tensors the forward produced instead of
    recomputing the attention. The stash stays device-resident (see the module note below).
    ``dropout_p`` is fixed at 0.0 -- the caller gates on ``attention_dropout == 0.0``. The
    checkpoint phase is read from the ambient grad mode here (``Function.apply`` forces
    no_grad inside forward, so the Function itself cannot see it) and passed down as the
    ``is_replay`` flag.

    Args:
        q (torch.Tensor): Query, ``[1, num_heads, total_seq, head_dim]`` (BNSD, batch 1).
        k (torch.Tensor): Key, same layout as ``q``.
        v (torch.Tensor): Value, same layout as ``q``.
        cu_seqlens_q (torch.Tensor): Cumulative query sequence ends, length ``n_seqs + 1``.
        cu_seqlens_k (torch.Tensor): Cumulative key sequence ends, length ``n_seqs + 1``.
        max_seqlen_q (Any): Max query sequence length (unused by NFA, kept for signature parity).
        max_seqlen_k (Any): Max key sequence length (unused by NFA, kept for signature parity).
        softmax_scale (float): Softmax scale applied by the kernel.
        deterministic (bool): Deterministic mode; no NPU knob, only forwards into the varlen
            wrapper's warning, same as the production path.
        module (Any): The calling attention module; half of the stash key (with the q shape).

    Returns:
        AttnOpOutputs: ``{"raw_output": [1, total_seq, num_heads, head_dim], "softmax_lse": None}``.
    """
    if q.size(0) != 1:
        raise RuntimeError(f"vit_flash_attn_skip_recompute expects batch size 1, got {q.size(0)}")
    # Same BNSD -> TND view chain as attn_imp.flash_attention, so the kernel sees the identical
    # (strided) layout and the forward stays bitwise with the production path.
    q_tnd = q.transpose(1, 2).squeeze(0)
    k_tnd = k.transpose(1, 2).squeeze(0)
    v_tnd = v.transpose(1, 2).squeeze(0)
    # Same actual_seq building as npu_flash_varlen_attn; the backward's grad call needs them and
    # the Function must not carry device tensors it does not grad through.
    actual_seq_qlen = (
        tuple(cu_seqlens_q[1:].tolist()) if isinstance(cu_seqlens_q, torch.Tensor) else tuple(cu_seqlens_q[1:])
    )
    actual_seq_kvlen = (
        tuple(cu_seqlens_k[1:].tolist()) if isinstance(cu_seqlens_k, torch.Tensor) else tuple(cu_seqlens_k[1:])
    )
    out = _VitFlashAttnSkipRecompute.apply(
        q_tnd,
        k_tnd,
        v_tnd,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        deterministic,
        actual_seq_qlen,
        actual_seq_kvlen,
        module,
        torch.is_grad_enabled(),
    )
    return {"raw_output": out[None], "softmax_lse": None}


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


# ViT flash-attention recompute skip (MindSpeed-MM port,
# fsdp/ops/flash_attn/skip_recompute_flash_attn.py): the checkpointed GLM-5.3-Flash vision
# blocks recompute their attention at the backward replay, a full NFA forward per block per
# pack. _VitFlashAttnSkipRecompute stashes (attn_output, softmax_max, softmax_sum) at the
# no_grad original forward -- the lse pair is exactly what an NFA backward would otherwise
# recompute -- and restores them at the grad-enabled replay, so the backward calls
# npu_fusion_attention_grad directly and the attention recompute disappears.
#
# The checkpoint phase is discriminated by the wrapper, not inside the Function: autograd's
# Function.apply forces no_grad around forward, so torch.is_grad_enabled() inside forward is
# always False and the replay would silently recompute instead of restoring. vit_flash_attn_
# skip_recompute reads the ambient grad mode -- visible there, the reentrant checkpoint runs
# the original forward under no_grad and the replay under grad -- and passes it down as the
# is_replay flag, the same discriminator the park/refill hooks above use at hook level.
#
# The stashed tensors stay device-resident until their replay: the stash depth within one
# checkpoint pass equals the number of tower blocks (24 for GLM-5.3-Flash), so the bytes ride
# the step's peak and parking them to host would add copy work to the launch-bound tower
# loop -- the device-memory cost is the accepted trade for skipping the attention recompute.
# The offload hooks' resize-to-0 CPU park is also unsafe here: a parked tensor refilled
# through storage resize does not survive ctx.save_for_backward.
#
# The stash is keyed by (attention module, q shape) and consumed LIFO: with one
# forward+backward per pack the replay order is the exact reverse of the push order. The
# shape in the key keeps the FSDP-alignment dummy visual forward (a few patches, same
# modules) in its own lane instead of interleaving with the real packs. A no_grad push
# without a matching replay (forward-only probes, or a frozen vision tower whose wrapped
# blocks skip replay) sinks below the live entries and is dropped by the in-flight cap; a
# pop whose shape does not match the replayed q fails loudly instead of handing wrong
# tensors to the grad kernel.
_MAX_INFLIGHT_ATTN_STASH = 32
_attn_stash: dict[tuple[Any, tuple[int, ...]], deque[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}


class _VitFlashAttnSkipRecompute(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: Any,
        max_seqlen_k: Any,
        softmax_scale: float,
        deterministic: bool,
        actual_seq_qlen: tuple[int, ...],
        actual_seq_kvlen: tuple[int, ...],
        module: Any,
        is_replay: bool,
    ) -> torch.Tensor:
        stash = _attn_stash.setdefault((module, tuple(q.shape)), deque())
        if is_replay:
            # Checkpoint replay: restore the entry the no_grad original forward stashed
            # instead of recomputing the attention; the backward's grad call consumes the
            # exact tensors that forward produced. Correctness rests on the trainer pairing
            # every checkpoint forward with its backward before the next one (LIFO): the q
            # restored here is a qkv-proj product recomputed during the replay, so its
            # storage differs from the push-time one and no data_ptr fingerprint can
            # discriminate a legitimate replay from a same-shape cross-graph pop.
            if not stash:
                raise RuntimeError(
                    "ViT skip-recompute stash is empty at the replay entry: either the no_grad "
                    "forward did not stash for this attention module (a frozen vision tower "
                    "replays without a stashing forward; keep the gate off for it), or the block "
                    "runs outside the vision checkpoint (VISION_RECOMPUTE_RATIO < 1.0 "
                    "leaves trailing blocks unwrapped; keep it at 1.0 with the gate on)."
                )
            out, smax, ssum = stash.pop()
            if out.shape != q.shape:
                raise RuntimeError(
                    f"ViT skip-recompute stash misalignment: stashed attn output {tuple(out.shape)} "
                    f"does not match the replayed q {tuple(q.shape)}."
                )
        else:
            # no_grad original forward: compute the attention and stash (output, lse pair).
            from xtuner.v1.ops.flash_attn.npu import npu_flash_varlen_attn

            out, smax, ssum = npu_flash_varlen_attn(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                causal=False,
                deterministic=deterministic,
                return_lse=True,
            )
            if len(stash) >= _MAX_INFLIGHT_ATTN_STASH:
                stash.popleft()
                log_rank0.warning(
                    f"ViT skip-recompute stash held {_MAX_INFLIGHT_ATTN_STASH} in-flight entries for one "
                    "attention module; dropped the oldest, whose replay cannot run (forward-only or "
                    "frozen-vision passes stash without a matching checkpoint replay)."
                )
            stash.append((out, smax, ssum))
        ctx.actual_seq_qlen = actual_seq_qlen
        ctx.actual_seq_kvlen = actual_seq_kvlen
        ctx.scale = softmax_scale
        ctx.save_for_backward(q, k, v, out, smax, ssum)
        return out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        import torch_npu

        q, k, v, attn_output, softmax_max, softmax_sum = ctx.saved_tensors
        # MindSpeed skip_recompute_flash_attn's grad call; inner_precise=0/sparse_mode=0 are
        # both the MindSpeed recipe and the op schema defaults the production forward runs with.
        grad_q, grad_k, grad_v, *_ = torch_npu.npu_fusion_attention_grad(
            q,
            k,
            v,
            grad_output,
            q.shape[1],
            "TND",
            softmax_max=softmax_max,
            softmax_sum=softmax_sum,
            attention_in=attn_output,
            pse=None,
            padding_mask=None,
            atten_mask=None,
            actual_seq_qlen=ctx.actual_seq_qlen,
            actual_seq_kvlen=ctx.actual_seq_kvlen,
            scale_value=ctx.scale,
            keep_prob=1.0,
            inner_precise=0,
            sparse_mode=0,
        )
        return (grad_q, grad_k, grad_v) + (None,) * 10


# ---------------------------------------------------------------------------
# SFA forward-output stash: skip the checkpoint replay's SFA recompute
# ---------------------------------------------------------------------------
# Port of MindSpeed-MM op_replay's "sfa" scope, restricted to the sparse-MLA
# Function path: the no-grad original forward stashes the raw
# npu_sparse_flash_attention kernel outputs and the grad-enabled checkpoint
# replay pops them instead of re-running the kernel. One call stashes ~270
# MiB at the 30B production shape ([8192, 32, 512] bf16 output + two
# [8192, 32] fp32 stats) and the 12 DSA-layer calls per microbatch (11 cycle
# DSA + MTP) share one shape, so the live stash peaks at 12 entries per lane.
# XTUNER_SFA_SKIP_RECOMPUTE (read in npu_sparse_mla) gates the whole feature;
# XTUNER_SFA_STASH_OFFLOAD selects the stash's residency and is read here,
# once, at import.
#
# Device-resident mode keeps the stashed tensors on the NPU until their
# replay. Offload mode parks them in pinned host memory instead, d2h-doc
# choreography throughout: the push issues the D2H right after the kernel on
# the shared offload stream (gate event recorded on the working stream,
# non_blocking copy, record_stream release of the source blocks), the op
# prefetches the H2D refill before its index transforms so the transfer
# overlaps them, and the pop waits on the refill event and returns fresh
# device tensors -- bitwise equal to the originals, same dtype both ways. The
# entry holds no device reference, so offload mode's forward device-memory
# footprint matches the feature-off run. Every copy of a device shares the
# one offload stream, so returning a pinned buffer to the pool while its last
# read or write is still in flight is safe: any later copy touching it is
# FIFO-ordered after it, and no cross-step events are needed. The pool keeps
# each shape family's high-water buffers pinned for reuse: training pads
# every microbatch to one shape, so one family holds ~3.2 GiB per rank
# (cap 16 x ~270 MiB) for the whole run, while an eval-only lane can fill
# its family's cap (~4.2 GiB) with orphan pushes that nothing ever pops.
# That bound is per shape family, not global: variable-length evaluation
# adds a family per distinct length, and pinned host usage is invisible to
# device-side monitoring -- the free-list LRU cap below bounds the number
# of resident families, and dropped buffers only return memory to the
# caching host allocator, which keeps the process's pinned footprint at its
# high-water mark rather than releasing it to the OS.
#
# The replay discriminator is is_checkpoint_replay (the ContextVar
# apply_activation_checkpointing sets), read at the call site outside
# apply(): Function.forward always sees no_grad, and grad mode alone would
# misread an UN-checkpointed grad-enabled forward (recompute_ratio < 1 leaves
# trailing layers unwrapped) as a replay and pop a foreign same-shape entry.
#
# LIFO pairing holds for the production topology, not as a general autograd
# guarantee (the engine is a priority queue; branch order is
# scheduler-dependent). The load-bearing invariants: every DSA call sits
# inside a per-layer reentrant checkpoint (recompute_ratio=1.0; MTP via
# share_weights), each checkpoint's replay+inner backward is atomic, the MTP
# block taps the trunk's final output so the tap node's two consumers force
# the whole MTP branch to finish before any main-chain replay starts, the
# losses are combined into one backward, and intra_layer_micro_batch=1 --
# with IMB>1 the interleaved microbatches' pushes and pops cross-pair
# same-shape entries and overflow the cap, so the gate read in
# npu_sparse_mla force-disables the feature at import when
# INTRA_LAYER_MICRO_BATCH>1. Orphan no-grad pushes (eval, probes) never
# pop: they sink below live entries and are cap-evicted; in offload mode
# they park pinned memory only.
#
# Guards warn and fall back to the kernel recompute instead of raising: the
# recompute is always numerically correct, so a detectable mispair (empty
# lane, shape mismatch, prefetch/pop divergence) degrades to the stock path
# with a rank-0 warning rather than crashing the training run. A miss
# recompute is itself a replay, so the call site does not push its outputs
# -- nothing would ever pop them and they could only mispair with the next
# step's original forward. Both contracts are equally blind to the one
# corruption mode that matters -- a same-shape cross-graph mispair, which
# only the invariants above prevent. The pop's shape check is
# defense-in-depth only (the lane key already fixes both shapes; it cannot
# detect a same-shape mispair), and the module-global shape-keyed stash
# assumes one live same-shape model per process -- two interleaved models
# would cross-pop.
#
# Stashed tensors are the original forward's outputs, so an in-place mutation
# downstream would corrupt them in device-resident mode -- the DSA call site
# only reads them via einsum, the same contract the ViT stash above relies
# on; offload mode's pinned copies freeze the values at park time.

_SFA_STASH_OFFLOAD = os.environ.get("XTUNER_SFA_STASH_OFFLOAD", "0") == "1"
_MAX_INFLIGHT_SFA_STASH = 16
# Pinned free-lists are keyed per exact shape family (see section comment):
# under variable-length evaluation a new family appears per distinct length,
# so the pool below drops the least-recently-used family's free-list -- and
# with it the last references to those buffers -- once more families than
# this are resident. The hot family is always the most recently used, so a
# live lane never loses its buffers to the cap.
_MAX_SFA_PINNED_FAMILIES = 4
_SfaStashKey = tuple[tuple[int, ...], tuple[int, ...]]
_SfaStashEntry = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
_SfaPinnedKey = tuple[tuple[int, ...], torch.dtype]
_sfa_pinned_pool: OrderedDict[_SfaPinnedKey, deque[torch.Tensor]] = OrderedDict()


def _acquire_sfa_pinned(shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
    """Take a pinned host buffer of exactly this shape and dtype from the pool,
    allocating a new one only when the free-list for this shape family is empty.

    Args:
        shape (`torch.Size`): Buffer shape to acquire.
        dtype (`torch.dtype`): Buffer dtype to acquire.

    Returns:
        `torch.Tensor`: A pinned CPU tensor of the requested shape and dtype.
    """
    key = (tuple(shape), dtype)
    free = _sfa_pinned_pool.get(key)
    if free:
        _sfa_pinned_pool.move_to_end(key)
        return free.pop()
    return torch.empty(shape, dtype=dtype, pin_memory=True, device="cpu")


def _release_sfa_pinned(buffer: torch.Tensor) -> None:
    """Return a pinned buffer to its shape family's free-list for reuse.

    Safe while the buffer's last copy is still in flight: all copies share the
    one offload stream, so any later write FIFO-follows that read. The family
    becomes the pool's most recently used; families beyond
    ``_MAX_SFA_PINNED_FAMILIES`` have their free-lists dropped in LRU order.

    Args:
        buffer (`torch.Tensor`): The pinned CPU tensor to recycle.
    """
    key = (tuple(buffer.shape), buffer.dtype)
    _sfa_pinned_pool.setdefault(key, deque()).append(buffer)
    _sfa_pinned_pool.move_to_end(key)
    while len(_sfa_pinned_pool) > _MAX_SFA_PINNED_FAMILIES:
        # popitem drops the LRU family's free-list and with it the last
        # references to those buffers; the caching host allocator reclaims
        # the blocks once any still-in-flight copy has completed -- the same
        # lifecycle the ACT/DSA offloads above use for their pinned staging.
        _sfa_pinned_pool.popitem(last=False)


class _SfaStashParked:
    """One stashed SFA forward parked in pinned host memory.

    Holds only the pinned copies, their shapes/dtypes, and the refill event --
    no device tensor -- so parked entries add zero forward device memory.

    Args:
        attn_output (`torch.Tensor`): Raw kernel attention output.
        softmax_max (`torch.Tensor`): Kernel softmax max statistic.
        softmax_sum (`torch.Tensor`): Kernel softmax sum statistic.
    """

    def __init__(self, attn_output: torch.Tensor, softmax_max: torch.Tensor, softmax_sum: torch.Tensor) -> None:
        outputs = (attn_output, softmax_max, softmax_sum)
        self.device = attn_output.device
        self.shapes = tuple(tuple(t.shape) for t in outputs)
        self.dtypes = tuple(t.dtype for t in outputs)
        stream = offload_stream(self.device)
        # Gate the D2H on the kernel that produced these outputs: record on
        # the working stream (where the kernel was enqueued, FIFO after it)
        # and make the offload stream wait before reading the sources.
        gate = torch.cuda.Event()
        gate.record()
        stream.wait_event(gate)
        self.buffers = [_acquire_sfa_pinned(t.shape, t.dtype) for t in outputs]
        with torch.cuda.stream(stream):
            for buf, t in zip(self.buffers, outputs):
                buf.copy_(t, non_blocking=True)
                # The sources are freed on the working stream once the
                # original forward is done with them; record_stream defers
                # the allocator's reuse of their blocks until this copy has
                # read them.
                t.record_stream(stream)
        self.h2d_event: torch.cuda.Event | None = None
        self.restored: list[torch.Tensor] = []

    def prefetch(self) -> None:
        # Idempotent: at most one in-flight refill per entry.
        if self.h2d_event is not None:
            return
        stream = offload_stream(self.device)
        # Same gate logic as the park: a fresh block handed out by the
        # allocator may carry a prior owner's still-queued kernels, and the
        # H2D write must not overtake them. The fresh tensors are allocated
        # OUTSIDE the offload-stream context on purpose: allocating inside it
        # would home their blocks in the offload stream's pool, where a later
        # offload-stream allocation could reuse them under still-running
        # consumer kernels. On the working stream the blocks return to the
        # working pool on free, and record_stream below tracks the H2D write.
        gate = torch.cuda.Event()
        gate.record()
        stream.wait_event(gate)
        restored = [
            torch.empty(shape, dtype=dtype, device=self.device) for shape, dtype in zip(self.shapes, self.dtypes)
        ]
        with torch.cuda.stream(stream):
            for buf, fresh in zip(self.buffers, restored):
                fresh.copy_(buf, non_blocking=True)
                fresh.record_stream(stream)
            self.h2d_event = torch.cuda.Event()
            self.h2d_event.record()
        self.restored = restored

    def consume(self) -> _SfaStashEntry:
        # No prefetch issued (or it was lost): blocking self-rescue refill --
        # the pop still returns the right values, just without overlap.
        if self.h2d_event is None:
            self.prefetch()
        # Async enqueue on the caller's current (working) stream: consumer
        # kernels queue behind the H2D and run once it completes.
        torch.cuda.current_stream().wait_event(self.h2d_event)
        self.release()
        return (self.restored[0], self.restored[1], self.restored[2])

    def release(self) -> None:
        # FIFO-safe on the shared offload stream (see _release_sfa_pinned);
        # idempotent for consume-then-cap-evict paths that can no longer fire.
        for buffer in self.buffers:
            _release_sfa_pinned(buffer)
        self.buffers = []


_sfa_stash: dict[_SfaStashKey, deque[_SfaStashEntry | _SfaStashParked]] = {}


def sfa_stash_key(q_nope: torch.Tensor, kv: torch.Tensor) -> _SfaStashKey:
    """Lane key for the SFA forward-output stash.

    Args:
        q_nope (`torch.Tensor`): Absorbed queries ``[S, N, Rkv]`` of the call.
        kv (`torch.Tensor`): Prefix-extended KV slice ``[L, 1, Rkv]`` of the call.

    Returns:
        `_SfaStashKey`: The (query shape, kv shape) pair identifying the lane.
    """
    return tuple(q_nope.shape), tuple(kv.shape)


def sfa_stash_push(
    key: _SfaStashKey,
    attn_output: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
) -> None:
    """Stash one SFA forward's kernel outputs for its checkpoint replay.

    In offload mode the entry parks the outputs in pinned host memory and
    holds no device reference; in device-resident mode it holds the raw
    tensors themselves.

    Args:
        key (`_SfaStashKey`): Lane key from :func:`sfa_stash_key`.
        attn_output (`torch.Tensor`): Raw kernel attention output.
        softmax_max (`torch.Tensor`): Kernel softmax max statistic.
        softmax_sum (`torch.Tensor`): Kernel softmax sum statistic.
    """
    lane = _sfa_stash.setdefault(key, deque())
    if len(lane) >= _MAX_INFLIGHT_SFA_STASH:
        evicted = lane.popleft()
        if isinstance(evicted, _SfaStashParked):
            evicted.release()
        log_rank0.warning(
            f"SFA skip-recompute stash held {_MAX_INFLIGHT_SFA_STASH} in-flight "
            f"entries for shape {key}; dropping the oldest (expected only when "
            f"orphan no-grad forwards outlive live graphs)"
        )
    if _SFA_STASH_OFFLOAD:
        lane.append(_SfaStashParked(attn_output, softmax_max, softmax_sum))
    else:
        lane.append((attn_output, softmax_max, softmax_sum))


def sfa_stash_prefetch(key: _SfaStashKey) -> _SfaStashParked | None:
    """Issue the H2D refill for this lane's most recent parked entry.

    Idempotent per entry; the entry handle must be handed back to
    :func:`sfa_stash_pop` so the pop can detect a prefetch that no longer
    matches the lane tail.

    Args:
        key (`_SfaStashKey`): Lane key from :func:`sfa_stash_key`.

    Returns:
        `_SfaStashParked | None`: The prefetched entry, or None when offload
        mode is off or the lane is empty (device-resident mode pops directly).
    """
    if not _SFA_STASH_OFFLOAD:
        return None
    lane = _sfa_stash.get(key)
    if not lane:
        return None
    entry = lane[-1]
    if not isinstance(entry, _SfaStashParked):
        # A foreign plain-tuple entry appended into an offload-mode lane has
        # no refill to issue; return None and the pop's tuple branch returns
        # its values as-is. The replay path must not raise: an assert would
        # also be stripped under ``python -O`` and fail later as AttributeError.
        return None
    entry.prefetch()
    return entry


def sfa_stash_pop(
    key: _SfaStashKey,
    q_nope: torch.Tensor,
    kv: torch.Tensor,
    prefetch: _SfaStashParked | None = None,
) -> _SfaStashEntry | None:
    """Pop the stashed kernel outputs for this replay's lane.

    A miss never raises: the kernel recompute is always numerically correct,
    so the caller falls back to it and training keeps running.

    Args:
        key (`_SfaStashKey`): Lane key from :func:`sfa_stash_key`.
        q_nope (`torch.Tensor`): Absorbed queries ``[S, N, Rkv]`` of the replay call.
        kv (`torch.Tensor`): Prefix-extended KV slice ``[L, 1, Rkv]`` of the replay call.
        prefetch (`_SfaStashParked`, optional): Entry handle returned by
            :func:`sfa_stash_prefetch` for this call. When it differs from the
            lane tail (LIFO violated or the entry was cap-evicted in between),
            the mismatched entry stays parked and the actual tail takes a
            blocking self-rescue refill. On an empty lane it is released so
            its pinned buffers return to the pool instead of leaking.

    Returns:
        `_SfaStashEntry | None`: The stashed ``(attn_output, softmax_max,
        softmax_sum)`` (offload mode: fresh device tensors bitwise equal to
        the originals), or ``None`` when the lane is empty or the tail entry's
        shape does not match this replay call — the caller then reruns the
        kernel instead.
    """
    lane = _sfa_stash.get(key)
    if not lane:
        if prefetch is not None:
            prefetch.release()
        log_rank0.warning(
            "SFA skip-recompute stash is empty at the replay entry: a checkpoint-"
            "replay SFA forward found no stashed original for shape "
            f"{key} (external lane clear or cap eviction). Falling back to the "
            "kernel recompute for this call."
        )
        return None
    entry = lane[-1]
    if isinstance(entry, _SfaStashParked):
        if prefetch is not None and prefetch is not entry:
            log_rank0.warning(
                "SFA stash prefetch does not match the popped entry for lane "
                f"{key}; falling back to a blocking refill (LIFO violated or the "
                "entry was cap-evicted between prefetch and pop)"
            )
        outputs = entry.consume()
    else:
        outputs = entry
    lane.pop()
    if not lane:
        # Prune the drained lane so variable-shape evaluation does not
        # accumulate one empty deque per distinct shape ever seen.
        del _sfa_stash[key]
    # Defense-in-depth only: the lane key already fixes both shapes, so this
    # fires solely on a foreign entry appended directly into the lane; it
    # cannot detect a same-shape cross-graph mispair (see section comment).
    expected = (q_nope.shape[0], q_nope.shape[1], kv.shape[-1])
    if tuple(outputs[0].shape[-3:]) != expected:
        log_rank0.warning(
            "SFA skip-recompute stash misalignment: entry shape "
            f"{tuple(outputs[0].shape)} does not match the replay call "
            f"{expected} for lane {key}. Dropping the foreign entry and "
            "falling back to the kernel recompute for this call."
        )
        return None
    return outputs
