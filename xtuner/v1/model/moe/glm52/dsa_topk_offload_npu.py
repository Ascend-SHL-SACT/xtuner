# Copyright (c) OpenMMLab. All rights reserved.
import os
from functools import partial
from typing import Any, cast

import torch

from xtuner.v1.utils import log_rank0
from xtuner.v1.utils.activation_offload import OffloadManager, SwapTensor

from .activation_offload_npu import _offload_stream
from .decoder_layer import (
    GLM52DenseDecoderLayerMicroBatchOutput,
    GLM52DenseDecoderLayerOutput,
    GLM52MoEDecoderLayerMicroBatchOutput,
    GLM52MoEDecoderLayerOutput,
)
from .dsa_mla import DSAMultiLatentAttention


__all__ = [
    "dsa_topk_offload_npu_enabled",
    "register_dsa_topk_offload_npu_hooks",
]


def dsa_topk_offload_npu_enabled() -> bool:
    """Whether XTUNER_DSA_TOPK_OFFLOAD_NPU enables the GLM-5.2 DSA top-k
    offload.

    Returns:
        bool: True when XTUNER_DSA_TOPK_OFFLOAD_NPU is set to "1".
    """
    return os.getenv("XTUNER_DSA_TOPK_OFFLOAD_NPU", "0") == "1"


def register_dsa_topk_offload_npu_hooks(
    *,
    model_layers: torch.nn.ModuleDict,
    source_layers: tuple[int, ...],
    last_consumers: frozenset[int],
) -> None:
    """Register the DSA top-k offload/refill wiring on the raw decoder layers.

    Called once from GLM-5.2 `_configure_model_specific_layers`, before
    activation checkpointing wraps the layers. The wrapper keeps the same
    module objects, so everything fires inside the checkpoint boundary:

    - a post-hook on each source's last consumer parks the ids' device
      storage on the CPU after the no_grad original forward;
    - each DSA attention's `sparse_mla_func` instance attribute is wrapped so
      the backward replay refills the storage at the moment of consumption
      (right before sparse MLA reads the ids, keeping them evacuated through
      the replayed layer prologue); the first replayed consumer in reverse
      backward order pops the entry, later consumers are no-ops. Each refill
      also prefetches the next source backward will consume, hiding its H2D
      behind the current layer's replayed compute.

    Args:
        model_layers (torch.nn.ModuleDict): Decoder layers keyed by str(idx).
        source_layers (tuple[int, ...]): Per-layer DSA top-k source-layer index.
        last_consumers (frozenset[int]): Layers that last consume each source's
            ids.
    """
    for idx, layer in model_layers.items():
        attn = getattr(layer, "self_attn")
        if not isinstance(attn, DSAMultiLatentAttention):
            raise TypeError(f"DSA top-k NPU offload requires DSAMultiLatentAttention, got {type(attn).__name__}.")
        _wrap_sparse_mla_with_refill(attn, int(idx))

    for layer_idx in sorted(last_consumers):
        model_layers[str(layer_idx)].register_forward_hook(
            partial(_offload_dsa_topk_after_last_use, source_layer_idx=source_layers[layer_idx])
        )
    # The backward replay consumes the sources in reverse consumer order. When
    # a consumer's ids are refilled, the next source to be consumed (the
    # consumer with the next-smaller layer index) is prefetched, so its H2D
    # overlaps with the current layer's replayed compute instead of blocking
    # that layer's consumption point.
    _next_consumed_source.update(
        {consumer: source_layers[prev] for prev, consumer in zip(sorted(last_consumers), sorted(last_consumers)[1:])}
    )

    log_rank0.info(
        "DSA top-k NPU offload registered: parking the ids of source layers "
        f"{sorted({source_layers[layer_idx] for layer_idx in last_consumers})} after their last consumers "
        f"{sorted(last_consumers)}; refill happens at consumption inside sparse_mla_func."
    )
    if os.getenv("XTUNER_DSA_TOPK_OFFLOAD", "0") == "1":
        log_rank0.info(
            "XTUNER_DSA_TOPK_OFFLOAD is also enabled: its saved-tensors window additionally captures the ids as "
            "checkpoint frame inputs on the last consumers and restores them at frame entry. Coexistence relies "
            "on the refill wrapper being the last writer before the sole reader: the blocking consumption-point "
            "refill overwrites the window restore before sparse MLA reads the ids."
        )


# Live offloads keyed by source layer index (like the upstream residency's
# cache.offloaded), one SwapTensor per micro-batch; each SwapTensor pins its
# ids tensor alive until _ensure_dsa_topk_resident consumes the entry. Tensor
# identity cannot be the key: the reentrant checkpoint replay passes
# detach_variable copies of the frame-held ids, so id() does not survive into
# the replay while the shared storage does.
_topk_offloads: dict[int, list[SwapTensor]] = {}
# Consumer layer index -> source index of the next consumer that backward
# will visit (consumers are replayed in reverse order). Filled at
# registration; a consumer absent from the map is the first one replayed and
# has no earlier consumption to prefetch for.
_next_consumed_source: dict[int, int] = {}


def _wrap_sparse_mla_with_refill(attn: DSAMultiLatentAttention, layer_idx: int) -> None:
    # Shadow the attention instance's sparse_mla_func attribute with a shim
    # that refills parked ids right before the original call consumes them
    # (dsa_mla.py source stays untouched). A no-op miss makes the wrapper
    # safe on every forward: original forwards run before anything is parked,
    # and inference never parks. After its own refill, the wrapper also
    # prefetches the next source backward will consume, hiding its H2D behind
    # this layer's replayed compute.
    original = attn.sparse_mla_func
    if getattr(original, "_dsa_topk_refill_wrapper", False):
        return
    source_layer_idx = attn.source_layer_idx

    def _refill_then_call(*args: Any, **kwargs: Any) -> Any:
        _ensure_dsa_topk_resident(source_layer_idx)
        next_source = _next_consumed_source.get(layer_idx)
        if next_source is not None:
            _prefetch_dsa_topk(next_source)
        return original(*args, **kwargs)

    setattr(_refill_then_call, "_dsa_topk_refill_wrapper", True)
    attn.sparse_mla_func = _refill_then_call


def _offload_dsa_topk_after_last_use(
    module: torch.nn.Module,
    _args: tuple[Any, ...],
    output: object,
    *,
    source_layer_idx: int,
) -> None:
    # Forward-hook body. Mirrors the checkpoint detection of the upstream
    # decoder lifecycle hooks (training + no_grad = reentrant original
    # forward). The saved-tensors offload window in _call_decoder_layer only
    # reaches ids that are checkpoint frame inputs on their last consumer: ids
    # whose source is its own last consumer are computed inside the frame and
    # never packed, the final block's ids are never evacuated (no later block
    # pack triggers it), and packed ids are refilled at frame entry rather
    # than at consumption. Park the storage here instead, once the last
    # forward consumer has run; the grad-enabled recompute replay skips this
    # hook and the sparse_mla_func wrapper refills the ids at the moment of
    # consumption.
    if not module.training or torch.is_grad_enabled():
        return
    glm_results = cast(
        GLM52DenseDecoderLayerOutput
        | GLM52DenseDecoderLayerMicroBatchOutput
        | GLM52MoEDecoderLayerOutput
        | GLM52MoEDecoderLayerMicroBatchOutput,
        output,
    )
    _offload_dsa_topk(source_layer_idx=source_layer_idx, topk_ids=glm_results["dsa_topk_ids"])


@torch.compiler.disable
def _offload_dsa_topk(*, source_layer_idx: int, topk_ids: torch.Tensor | list[torch.Tensor]) -> None:
    # Park the device storage of one source layer's DSA top-k ids on the CPU,
    # on the checkpoint original (no_grad) forward only. The ids tensor itself
    # stays referenced by the checkpoint frames; _ensure_dsa_topk_resident
    # restores its storage before any replayed consumer reads it. Like the
    # upstream XTUNER_DSA_TOPK_OFFLOAD residency, this therefore assumes every
    # consumer of an offloaded source runs under activation checkpointing.
    #
    # One source layer owns one live offload entry at a time. Reusing it while
    # still active would overwrite the earlier D2H result, so fail loudly if
    # the scheduling contract is violated (mirrors the upstream residency).
    if source_layer_idx in _topk_offloads:
        raise RuntimeError(f"DSA top-k offload for source layer {source_layer_idx} is still active.")

    swap_tensors: list[SwapTensor] = []
    # Register before parking so a mid-loop failure cannot strand launched
    # copies without a visible owner: the partial entry stays visible to
    # _ensure_dsa_topk_resident, and the next consumption refills whatever was
    # launched (same self-healing contract as an aborted step's miss path). A
    # refill of a launched-but-not-waited entry takes the blocking path on the
    # same offload stream, which is ordered after the in-flight D2H, so a
    # visible entry is always restorable.
    _topk_offloads[source_layer_idx] = swap_tensors
    # Launch every micro-batch's D2H back-to-back on the offload stream, then
    # free the device storages without stalling the compute stream on the
    # in-flight copies: record_stream defers each block's reuse until its copy
    # finishes, so the copies overlap the rest of the forward instead of
    # serializing against it at the park point (a post-hook has no layer body
    # left to cover them, unlike the activation offload's pre-hook launch).
    # Refills run on the same offload stream, which orders them after any copy
    # still in flight.
    for mb_idx, ids in enumerate(topk_ids if isinstance(topk_ids, list) else [topk_ids]):
        # Reuse the activation-offload pinned-buffer cache, keyed per source
        # layer and micro-batch slot (same helper the upstream DSA top-k
        # residency uses); shape or dtype drift reallocates.
        key = f"glm52_dsa_topk_{source_layer_idx}_{mb_idx}"
        tensor_cpu = OffloadManager().get_or_create_pin_memory(key, ids.shape, ids.dtype)
        swap_tensor = SwapTensor(ids, key, tensor_cpu=tensor_cpu)
        # launch_d2h gates the copy on a forward_event recorded here on the
        # compute stream (the muon_swap start-gate pattern); a stream-wide
        # wait_stream would duplicate that gate.
        swap_tensor.launch_d2h(_topk_stream(ids.device))
        swap_tensors.append(swap_tensor)
    for swap_tensor in swap_tensors:
        swap_tensor.tensor.record_stream(_topk_stream(swap_tensor.tensor.device))
        swap_tensor.tensor.storage().resize_(0)


@torch.compiler.disable
def _ensure_dsa_topk_resident(source_layer_idx: int) -> None:
    # Restore the device storage of one source layer's ids if _offload_dsa_topk
    # parked it on the CPU. Called by the sparse_mla_func wrapper right before
    # consumption: the backward replay re-enters layers in reverse order, so a
    # source's last consumer is the first replayed reader of its ids and pops
    # the entry, refilling every micro-batch; later replays of the same source
    # find nothing parked (no-op), as does every original (no_grad) forward.
    swap_tensors = _topk_offloads.pop(source_layer_idx, None)
    if swap_tensors is None:
        return

    for swap_tensor in swap_tensors:
        stream = _topk_stream(swap_tensor.tensor.device)
        working_stream = torch.cuda.current_stream(swap_tensor.tensor.device)
        if swap_tensor.stat != "host":
            # Prefetched by the previous consumption: only the event wait
            # remains (launch_h2d's waiting path is a no-op on an event that
            # was never recorded, so never-evacuated swaps pass through too).
            swap_tensor.launch_h2d(stream, True, working_stream)
            continue
        # Not prefetched (first consumption after an aborted/partial park):
        # blocking H2D on the offload stream, same side-stream choreography
        # as the activation offload: issue the H2D on the offload stream and
        # order the current stream behind it, so the layer body reads
        # refilled storage.
        stream.wait_stream(working_stream)
        with torch.cuda.stream(stream):
            swap_tensor.launch_h2d(stream, True, stream)
        working_stream.wait_stream(stream)


@torch.compiler.disable
def _prefetch_dsa_topk(source_layer_idx: int) -> None:
    # Start the async H2D of the next source's parked ids (stat "host" only:
    # never-evacuated or already-prefetched swaps are skipped). Called by the
    # current source's refill wrapper, one consumer ahead of the actual
    # consumption, so the copy overlaps with the current layer's replayed
    # compute; the owning refill then only waits on the copy's event.
    entry = _topk_offloads.get(source_layer_idx)
    if not entry:
        return
    for swap_tensor in entry:
        if swap_tensor.stat == "host":
            swap_tensor.prefetch_launch_h2d(_topk_stream(swap_tensor.tensor.device), True)


def _topk_stream(device: torch.device) -> torch.cuda.Stream:
    # Shared with the activation offload so the two mechanisms' DMA traffic
    # never runs on two concurrent streams: in the both-ON A/B each mechanism
    # alone is timing-neutral but together they cost ~0.6 s/step, uniformly
    # across steps, and the ids refill prefetch does not touch it — pointing
    # at dual-stream arbitration, not at any refill blocking. Serializing the
    # copies on one stream is harmless (their total transfer time per step is
    # two orders of magnitude below the step time).
    return _offload_stream(device)
