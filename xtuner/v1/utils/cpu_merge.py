"""Gated host-memory merge for save unshard, plus CPU-tensor IPC serialization.

Export ``XTUNER_UNSHARD_CPU_MERGE=1`` in the launch environment (trainer-side processes
only; the vLLM engine is not involved). With the gate unset, callers keep their upstream
code paths untouched. Two halves:

- Final-step merge on host memory (the memory win): in the save unshard
  (``unshard_tensors_for_hf_save``), the gathered chunks already hold the full-tensor byte
  volume on device, and merging on device needs ~2x that on top (cat + narrow/contiguous,
  and the interleaved branch cats twice). For 30B MoE expert weights the full tensor is
  12+ GiB bf16 per bucket, which OOMs a colocated NPU whose free device memory is mostly
  consumed by HCCL buffers and the training state. For EVERY tensor's last unshard step —
  no size threshold, no routing — this module moves the gathered chunks to host memory
  before the merge, so device usage stays bounded by the gathered collective buffer; the
  trade is host-side, ~2x the full tensor in host RAM transiently during the merge. The
  gather schedule stays the upstream one: compatible tensors of a round share one batched
  ``foreach_all_gather`` collective, and only the final merge target moves to host.
  Intermediate steps must stay on device — their output feeds another device collective in
  the next round.
- CPU-tensor IPC serialization (the host-payload channel): host-memory tensors merged by
  ``unshard_tensors_for_hf_save_with_cpu_merge`` cannot ride the default
  ``file_descriptor`` sharing strategy — it pickles an fd number that cannot cross the
  HTTP JSON channel to the rollout process — while the ``file_system`` strategy rebuilds
  by shared-memory filename. ``dump_state_dict_file_system`` performs the
  ``reduce_tensor`` pass AND the pickle under that strategy (``reduce_storage`` reads the
  strategy at reduce time, so the toggle must wrap the reduce loop, not only the dump).
  Device-memory tensors in the same payload reduce through the device IPC path, which is
  independent of the sharing strategy. The vLLM receiver
  (``update_weight_npu_ipc_compat._construct``) rebuilds the short-signature host tensors
  as-is on CPU and lets ``load_weights`` copy them into the device parameters.

Two transport-mechanics optimizations keep the channel cheap (no routing, no thresholds):
the gathered chunks are DMA'd into a reused pinned staging buffer (``_stage_chunks_to_pinned``)
instead of per-chunk ``.to("cpu")`` into fresh pageable tensors, whose cold first-touch pages
run at a measured ~2 GB/s while the reused pinned buffer sustains ~40 GB/s single-device; and
on the lazy RL weight-iteration path (``reusable_staging=True``) the chunks are DMA'd into a
bounded bump pool of ``file_system``-shared segments (``shared_pool_alloc``) whose payloads
serialize metadata-only, with the dim-0 contiguous merge collapsing to a zero-copy view of
the staging buffer. Pool total size is bounded by one bucket's high-water bytes (the cursor
resets at each bucket build), not by the parameter set. All reuses are safe because bucket
iteration is lazy — bucket k+1 is built only after bucket k's transfer completed — and the
IPC transport barriers every rank after the rollout engine consumed bucket k, so a staging
buffer is never overwritten while its previous contents are still being read. The checkpoint
save path keeps ``reusable_staging=False``: its async safetensors writes may still be reading
earlier buckets while the next one is built, so it never touches the shared pool.

The gate is validated for the vLLM rollout backend. The turbomind/LMDeploy layer-batch path
(``iter_layer_batches``) consumes the same unshard without a copy-back to device and has no
corresponding serialization toggle — exporting the gate with backend=turbomind is unverified.
The orchestration reuses the upstream load_spec helpers unchanged; only the final-step merge
target and the serialization strategy differ.
"""

import atexit
import gc
import os
import threading
from typing import Any, BinaryIO

import torch

from xtuner.v1.utils import load_spec as load_spec_module
from xtuner.v1.utils.load_spec import (
    HFSavePlan,
    SaveShardStep,
    _finalize_hf_save_tensor,
    _is_same_process_group,
    _merge_gathered_save_shard,
    _pad_tensor_for_save_shard,
    _take_ready_save_unshard_groups,
)


CPU_MERGE_ENV = "XTUNER_UNSHARD_CPU_MERGE"

# ``set_sharing_strategy`` is process-global (not thread-local). This lock serializes the
# toggles issued by this module so two of its own dump/alloc paths cannot interleave; it
# does NOT make the toggle thread-private — an unrelated thread pickling CPU tensors
# during the toggle window would observe (and reduce under) the toggled strategy. In
# practice both strategies rebuild correctly, so a stray toggle is benign.
_SHARING_STRATEGY_LOCK = threading.Lock()


def cpu_merge_enabled() -> bool:
    """Return whether the save-unshard CPU merge bundle is enabled via the gate env var.

    Returns:
        bool: True when ``XTUNER_UNSHARD_CPU_MERGE`` is exported as ``"1"``.
    """
    return os.environ.get(CPU_MERGE_ENV) == "1"


def dump_state_dict_file_system(data: list[tuple[str, Any]], buf: BinaryIO) -> None:
    """Reduce and pickle tensors under the ``file_system`` sharing strategy.

    ``data`` holds the raw ``(name, tensor)`` items; the ``reduce_tensor`` pass runs HERE,
    inside the strategy toggle, because ``reduce_storage`` reads the sharing strategy at
    reduce time — toggling around a pre-reduced payload would leave host-memory tensors
    on the ``file_descriptor`` strategy, which pickles an fd number that cannot cross the
    HTTP JSON channel to the rollout process, while ``file_system`` rebuilds by
    shared-memory filename (see the module docstring). Device-memory tensors in the same
    payload reduce through the device IPC path, independent of the sharing strategy. The
    previous strategy is restored afterwards; the toggle is process-global, so it is
    serialized under a lock.

    Args:
        data (list[tuple[str, Any]]): Raw ``(name, tensor)`` weight items to reduce and pickle.
        buf (BinaryIO): Writable buffer the pickled payload is dumped into.
    """
    from multiprocessing.reduction import ForkingPickler

    from torch.multiprocessing.reductions import reduce_tensor

    with _SHARING_STRATEGY_LOCK:
        previous_strategy = torch.multiprocessing.get_sharing_strategy()
        torch.multiprocessing.set_sharing_strategy("file_system")
        try:
            reduced = [(name, reduce_tensor(tensor)) for name, tensor in data]
            ForkingPickler(buf).dump(reduced)
        finally:
            torch.multiprocessing.set_sharing_strategy(previous_strategy)


def new_shared_cpu_tensor(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Allocate a CPU tensor born inside a ``file_system``-shared (shm) segment.

    The sharing strategy only matters when a storage is first shared and when it is later
    reduced, so the segment must be created under the ``file_system`` strategy for the
    receiver-side shm rebuild to work. Reusing an already-shared tensor costs no copy: a
    fresh pageable tensor would instead be copied into shm by ``reduce_storage`` on a
    single thread at a measured ~1.5-3 GB/s, which dominates the sync.

    Args:
        shape (tuple[int, ...]): Shape of the tensor to allocate.
        dtype (torch.dtype): Dtype of the tensor to allocate.

    Returns:
        torch.Tensor: A zero-filled CPU tensor backed by a named shared-memory segment.
    """
    with _SHARING_STRATEGY_LOCK:
        previous_strategy = torch.multiprocessing.get_sharing_strategy()
        torch.multiprocessing.set_sharing_strategy("file_system")
        try:
            tensor = torch.empty(shape, dtype=dtype)
            tensor.share_memory_()
        finally:
            torch.multiprocessing.set_sharing_strategy(previous_strategy)
    return tensor


# Reused staging for the device-to-host copy of gathered chunks. Fresh pageable
# destinations are copied through cold first-touch pages at a measured ~2 GB/s; a pinned
# buffer allocated once and reused runs at the full ~40 GB/s single-device DMA rate. With
# all ranks syncing concurrently the node's PCIe/host bandwidth is shared, so the durable
# win is not faster copies but fewer copies — hence the shared pool below, which also
# makes the merged payload zero-copy.
_PINNED_STAGING: dict[str, torch.Tensor] = {}

# Bounded bump pool of file_system-shared segments for one bucket's payloads (chunk
# staging and merge outputs). Segments persist and are reused across buckets/syncs; the
# cursor resets once per bucket build, which is safe on the RL weight-iteration flow
# because bucket iteration is lazy and the IPC transport barriers every rank after the
# rollout engine consumed the bucket (the checkpoint save path never enables the pool).
# Total shared memory is bounded by a single bucket's high-water bytes, not by the
# parameter set.
_SHARED_POOL: list[list] = []


def shared_pool_alloc(nbytes: int) -> torch.Tensor:
    """Allocate a shared 1-D uint8 slice of ``nbytes`` bytes from the bump pool.

    Trainer-side merges and the transport-side payload staging both allocate here; the
    pool cursor resets at the next bucket build (see ``reset_shared_pool``).

    Args:
        nbytes (int): Number of bytes to allocate.

    Returns:
        torch.Tensor: A 1-D uint8 view into a persistent shared segment.
    """
    # The cursor advances to a 16-byte boundary so a later ``.view(dtype)`` never sees a
    # storage_offset that is not divisible by the dtype's itemsize.
    alloc_bytes = (nbytes + 15) & ~15
    for entry in _SHARED_POOL:
        free = entry[0].numel() - entry[1]
        if free >= alloc_bytes:
            view = entry[0][entry[1] : entry[1] + nbytes]
            entry[1] += alloc_bytes
            return view
    segment = new_shared_cpu_tensor((alloc_bytes,), torch.uint8)
    _SHARED_POOL.append([segment, alloc_bytes])
    return segment[:nbytes]


def reset_shared_pool() -> None:
    """Reset the shared pool cursor so the next bucket build reuses every segment.

    Call only when no payload from the previous bucket is still being read (the lazy
    weight-iteration flow guarantees this at the next bucket build).
    """
    for entry in _SHARED_POOL:
        entry[1] = 0


def stage_cpu_state_dict_to_shared(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Copy host payload tensors into shared pool buffers for metadata-only IPC pickling.

    An already-shared storage serializes without a data copy, while a fresh pageable
    payload would instead be single-thread-copied into shared memory at reduce time.
    Device-resident payloads ride the device IPC path, and payloads already living in a
    shared segment (the cpu_merge shared pool) pass through untouched. The pool cursor
    resets at the next bucket build, after this batch's transfer has completed (lazy
    bucket iteration).

    Args:
        state_dict (dict[str, torch.Tensor]): Weight payloads staged for IPC transport.

    Returns:
        dict[str, torch.Tensor]: Payloads keyed by name, with host-resident non-shared
        tensors replaced by shared pool copies.
    """
    staged: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if tensor.device.type != "cpu" or tensor.is_shared():
            staged[name] = tensor
            continue
        shared = shared_pool_alloc(tensor.numel() * tensor.element_size())
        shared = shared.view(tensor.dtype).view(tuple(tensor.shape))
        shared.copy_(tensor)
        staged[name] = shared
    return staged


def _release_shared_pool_at_exit() -> None:
    # torch's file_system strategy unlinks a /dev/shm segment only when its owner-side
    # storage reference is released while the interpreter is still alive. Module-global
    # pool segments survive until teardown, where the finalizers no longer run, leaking
    # the whole pool (hundreds of GiB at 30B). Drop the references explicitly so the
    # segments are unlinked before the shared-memory manager shuts down.
    _SHARED_POOL.clear()
    _PINNED_STAGING.clear()
    gc.collect()


atexit.register(_release_shared_pool_at_exit)


def _pinned_staging(nbytes: int) -> torch.Tensor:
    """Return the reused pinned uint8 staging buffer, growing it when needed.

    Args:
        nbytes (int): Minimum buffer size in bytes.

    Returns:
        torch.Tensor: A pinned 1-D uint8 buffer of at least ``nbytes`` bytes.
    """
    buf = _PINNED_STAGING.get("chunks")
    if buf is None or buf.numel() < nbytes:
        _PINNED_STAGING["chunks"] = buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    return buf


def _stage_chunks_to_pinned(gathered_chunks: list[torch.Tensor]) -> list[torch.Tensor]:
    """Copy accelerator-resident gathered chunks into the reused pinned staging buffer.

    The copies are issued back-to-back as ``non_blocking`` DMA transfers and followed by a
    single device synchronization, so per-chunk launch/sync overhead does not scale with
    the chunk count. The returned views are read by the host merge immediately after this
    call returns and are never retained, which is what makes the buffer reusable for the
    next tensor.

    Args:
        gathered_chunks (list[torch.Tensor]): Gathered chunks living on the accelerator.

    Returns:
        list[torch.Tensor]: Pinned host views holding the same bytes, one per chunk.
    """
    from xtuner.v1.utils import get_torch_device_module

    total_bytes = sum(chunk.numel() * chunk.element_size() for chunk in gathered_chunks)
    staging = _pinned_staging(total_bytes)
    host_chunks: list[torch.Tensor] = []
    offset = 0
    for chunk in gathered_chunks:
        nbytes = chunk.numel() * chunk.element_size()
        view = staging[offset : offset + nbytes].view(chunk.dtype).view(tuple(chunk.shape))
        view.copy_(chunk, non_blocking=True)
        host_chunks.append(view)
        offset += nbytes
    get_torch_device_module().synchronize(gathered_chunks[0].device)
    return host_chunks


def _stage_chunks_to_shared(gathered_chunks: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Copy accelerator-resident gathered chunks into a shared pool staging buffer.

    Same batching as ``_stage_chunks_to_pinned`` (back-to-back copies + one device sync),
    but the destination is a ``file_system``-shared pool buffer: payloads cut from it are
    already shared, so serialization is metadata-only and no transport-side staging copy
    is needed. The pool is reused across buckets and syncs; this is safe because bucket
    iteration is lazy — bucket k+1 is staged only after bucket k's transfer completed
    (see the module docstring).

    Args:
        gathered_chunks (list[torch.Tensor]): Gathered chunks living on the accelerator.

    Returns:
        tuple[torch.Tensor, list[torch.Tensor]]: The staging base (1-D uint8 shared
            buffer) and shared-memory views holding the same bytes, one per chunk.
    """
    from xtuner.v1.utils import get_torch_device_module

    total_bytes = sum(chunk.numel() * chunk.element_size() for chunk in gathered_chunks)
    staging = shared_pool_alloc(total_bytes)
    host_chunks: list[torch.Tensor] = []
    offset = 0
    for chunk in gathered_chunks:
        nbytes = chunk.numel() * chunk.element_size()
        view = staging[offset : offset + nbytes].view(chunk.dtype).view(tuple(chunk.shape))
        view.copy_(chunk, non_blocking=True)
        host_chunks.append(view)
        offset += nbytes
    get_torch_device_module().synchronize(gathered_chunks[0].device)
    return staging, host_chunks


def unshard_tensors_for_hf_save_with_cpu_merge(
    tensors: list[torch.Tensor],
    save_plans: list[HFSavePlan],
    reusable_staging: bool = False,
) -> list[torch.Tensor]:
    """Run the save unshard all-gathers with each tensor's final merge on host memory.

    Drop-in replacement for ``unshard_tensors_for_hf_save`` under the CPU-merge gate.
    Group building, padding, and the merge math are reused from load_spec unchanged; this
    wrapper only derives which tensors are on their last unshard step —
    ``_take_ready_save_unshard_groups`` pops a tensor's head step in place, so an empty
    pending list after the call means the popped step was the final one — and merges
    those on host memory. See the module docstring for the memory rationale.

    Args:
        tensors (list[torch.Tensor]): Local runtime tensors to unshard.
        save_plans (list[HFSavePlan]): HF save plans corresponding to ``tensors``.
        reusable_staging (bool): Whether final-step payloads may live in the reused
            shared staging (zero-copy dim-0 merge view, metadata-only serialize). Only
            safe for consumers that finish reading a bucket before the next one is built;
            the RL weight-iteration flow passes ``True`` (its IPC transport also barriers
            after the rollout engine consumed each bucket), while the checkpoint save
            path keeps ``False``: its async safetensors writes may still be reading
            earlier buckets, so fresh per-call staging is used there.

    Returns:
        list[torch.Tensor]: Tensors after all pending save unshard steps have been
            executed; final-step results are host-memory tensors.
    """
    assert len(tensors) == len(save_plans), "Internal error: save tensor and plan count mismatch"
    if not tensors:
        return []

    if reusable_staging:
        # One bucket's payloads die once its transfer completes; the next build (this
        # call) may therefore reuse every pool segment from byte zero.
        reset_shared_pool()

    # Shallow-copy the list, not the tensors. Entries with no gather work can be returned as-is, while entries
    # that do need all-gather are overwritten in this working list with their gathered tensor.
    tensor_list = list(tensors)

    # Convert each tensor's forward shard history into the save-time work queue. Save must undo shards from
    # inner to outer, so the steps are reversed; preserved shards, such as an EP shard kept local for RL weight
    # sync, are removed from the queue. Their effect is already represented by the plan's output shapes.
    pending_shard_steps_list = [
        [step for step in reversed(save_plan.unshard_steps) if not step.preserved] for save_plan in save_plans
    ]

    while True:
        # One all-gather round per iteration; ``_take_ready_save_unshard_groups`` consumes each tensor's queue
        # head by head (reverse-unshard steps must run one by one) and batches compatible queues into groups.
        # It mutates ``pending_shard_steps_list`` in place, so emptiness afterwards identifies final steps.
        unshard_groups = _take_ready_save_unshard_groups(tensor_list, pending_shard_steps_list)
        if not unshard_groups:
            break

        for unshard_group in unshard_groups:
            merge_on_cpu = [not pending_shard_steps_list[index] for index in unshard_group.tensor_indices]
            gathered_tensors = _foreach_all_gather_save_shards_cpu_merge(
                unshard_group.tensors,
                unshard_group.shard_steps,
                merge_on_cpu,
                reusable_staging,
            )
            for index, gathered_tensor in zip(unshard_group.tensor_indices, gathered_tensors, strict=True):
                tensor_list[index] = gathered_tensor

    return [
        _finalize_hf_save_tensor(tensor, save_plan) for tensor, save_plan in zip(tensor_list, save_plans, strict=True)
    ]


def _foreach_all_gather_save_shards_cpu_merge(
    tensor_list: list[torch.Tensor],
    shard_steps: list[SaveShardStep],
    merge_on_cpu_flags: list[bool],
    reusable_staging: bool,
) -> list[torch.Tensor]:
    assert len(tensor_list) == len(shard_steps), "Internal error: tensor and shard-step count mismatch"
    assert tensor_list, "Internal error: empty save all-gather group"
    group = shard_steps[0].shard.group
    assert all(_is_same_process_group(group, shard_step.shard.group) for shard_step in shard_steps), (
        "Internal error: save all-gather group contains different process groups"
    )
    if os.environ.get("XTUNER_GLM52_CHUNKED_SAVE_AG") == "1" and not reusable_staging:
        # Oversized checkpoint-save groups (the ~12 GiB single-tensor expert bucket) gather
        # via a sliced all_gather_into_tensor loop with page-able staging: the bucket-sized
        # device and page-locked transients drop to one shard (run109/run111 node resets
        # happened exactly at this bucket during the final HF save). Weight-update groups
        # (reusable_staging=True) must keep the shared-pool staging: their payloads reach
        # the rollout engine through zero-copy IPC off that pool, and a page-able payload
        # would add a full extra copy per bucket (run112: sync_weight 56s -> 78s). Lazy
        # import: the chunked module imports this file's helpers.
        from xtuner.v1.rl.weight_update.chunked_save_unshard import try_chunked_save_unshard

        chunked = try_chunked_save_unshard(tensor_list, shard_steps, merge_on_cpu_flags)
        if chunked is not None:
            return chunked
    padded_tensor_list = [
        _pad_tensor_for_save_shard(tensor, shard_step)
        for tensor, shard_step in zip(tensor_list, shard_steps, strict=True)
    ]
    gathered_chunks_list = load_spec_module.foreach_all_gather(padded_tensor_list, group)
    return [
        _merge_gathered_save_shard_cpu_merge(gathered_chunks, shard_step, merge_on_cpu, reusable_staging)
        for gathered_chunks, shard_step, merge_on_cpu in zip(
            gathered_chunks_list, shard_steps, merge_on_cpu_flags, strict=True
        )
    ]


def _merge_gathered_save_shard_shared(
    staging_base: torch.Tensor,
    host_chunks: list[torch.Tensor],
    shard_step: SaveShardStep,
    staged_shared: bool,
) -> torch.Tensor:
    """Host merge that keeps the result inside shared memory, mirroring upstream math.

    Args:
        staging_base (torch.Tensor): The shared staging buffer holding ``host_chunks``
            back to back (1-D uint8); used as the merge output for the zero-copy view.
        host_chunks (list[torch.Tensor]): Host chunks (shared staging views when
            ``staged_shared``).
        shard_step (SaveShardStep): The final save unshard step being merged.
        staged_shared (bool): Whether ``host_chunks`` live in the shared staging buffer,
            which enables the zero-copy dim-0 concatenation view.

    Returns:
        torch.Tensor: Merged tensor, shared-memory backed whenever possible. Unlike the
            upstream merge this keeps narrow results as views (their storage is already
            shared, so serialization and the receiver's ``load_weights`` copy handle
            strides without an extra materialization).
    """
    dim = shard_step.shard.dim
    runtime_dim_size = shard_step.shape_before_shard[dim]
    if (
        staged_shared
        and shard_step.shard.interleave_factor == 1
        and dim == 0
        and all(chunk.is_contiguous() for chunk in host_chunks)
    ):
        # The staging layout is exactly the byte-level concatenation of contiguous dim-0
        # chunks in rank order, so the merged tensor is a zero-copy view of the staging
        # buffer; the runtime trim is a prefix view of the same storage.
        first = host_chunks[0]
        total_bytes = sum(chunk.numel() * chunk.element_size() for chunk in host_chunks)
        cat_shape = (sum(chunk.shape[0] for chunk in host_chunks), *first.shape[1:])
        merged = staging_base[:total_bytes].view(first.dtype).view(cat_shape)
        return merged.narrow(0, 0, runtime_dim_size)
    if shard_step.shard.interleave_factor == 1:
        base_shape = host_chunks[0].shape
        cat_shape = (*base_shape[:dim], sum(chunk.shape[dim] for chunk in host_chunks), *base_shape[dim + 1 :])
        dest_numel = 1
        for size in cat_shape:
            dest_numel *= size
        dest = shared_pool_alloc(dest_numel * host_chunks[0].element_size()).view(host_chunks[0].dtype).view(cat_shape)
        torch.cat(host_chunks, dim=dim, out=dest)
        return dest.narrow(dim, 0, runtime_dim_size)
    # One gathered chunk per rank, interleaved along ``dim`` in run_size blocks.
    interleave_factor = shard_step.shard.interleave_factor
    world_size = len(host_chunks)
    assert runtime_dim_size % (world_size * interleave_factor) == 0, (
        "Internal error: runtime dim is not divisible by the interleave schedule"
    )
    run_size = runtime_dim_size // (world_size * interleave_factor)
    ordered_runs = [
        host_chunks[rank].narrow(dim, run_index * run_size, run_size)
        for run_index in range(interleave_factor)
        for rank in range(world_size)
    ]
    base_shape = host_chunks[0].shape
    cat_shape = (*base_shape[:dim], runtime_dim_size, *base_shape[dim + 1 :])
    dest_numel = 1
    for size in cat_shape:
        dest_numel *= size
    dest = shared_pool_alloc(dest_numel * host_chunks[0].element_size()).view(host_chunks[0].dtype).view(cat_shape)
    torch.cat(ordered_runs, dim=dim, out=dest)
    return dest


def _merge_gathered_save_shard_cpu_merge(
    gathered_chunks: list[torch.Tensor],
    shard_step: SaveShardStep,
    merge_on_cpu: bool,
    reusable_staging: bool,
) -> torch.Tensor:
    # Move the gathered chunks to host memory before the merge (see the module docstring):
    # the chunks are views into one gathered collective buffer, so that buffer is released
    # only once every view dies; the device merge copy the upstream merge would otherwise
    # allocate on top is what this avoids. The merge below then runs on host chunks and
    # returns a host tensor.
    staged_shared = False
    if merge_on_cpu and gathered_chunks and gathered_chunks[0].device.type != "cpu":
        # Chunks live on the accelerator: batch them into a reused staging (back-to-back
        # copies + one sync) instead of per-chunk .to("cpu") into fresh pageable tensors,
        # whose cold first-touch pages run at a measured ~2 GB/s. The shared pool staging
        # also makes the merged payload zero-copy; the pinned staging keeps a
        # function-local lifetime for consumers that may retain the contents longer.
        if reusable_staging:
            staging_base, host_chunks = _stage_chunks_to_shared(gathered_chunks)
            staged_shared = True
        else:
            host_chunks = _stage_chunks_to_pinned(gathered_chunks)
    else:
        # Chunks are already host-resident (non-merge steps, or CPU tensors in tests).
        host_chunks = gathered_chunks
        staging_base = None
    if merge_on_cpu and staged_shared:
        assert staging_base is not None, "Internal error: shared merge without a staging base"
        result = _merge_gathered_save_shard_shared(staging_base, host_chunks, shard_step, staged_shared)
    else:
        result = _merge_gathered_save_shard(host_chunks, shard_step)
    return result
