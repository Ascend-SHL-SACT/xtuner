"""Chunked all-gather with page-able inline staging for oversized HF-checkpoint save groups.

The checkpoint-save path (``reusable_staging=False``) cannot reuse a shared staging pool:
its async safetensors writes may still read
earlier buckets when the next one is built, so the shared pool must not be reset between
buckets and would instead accumulate one staging per bucket. The save path therefore still
runs the fused ``foreach_all_gather``, whose ~12 GiB single-tensor expert bucket allocates
the whole gathered buffer on the device plus a whole-bucket page-locked host staging —
16 ranks doing both at once is the recurring node-killer: run109 and run111 both died
exactly at that bucket during the final HF save, while run110 survived it.

This module reproduces the ``foreach_all_gather_stage_chunked`` staging layout with a
plain page-able host buffer plus a small pinned bounce buffer (the largest slice): the
device transient drops from bucket-sized (~12 GiB) to shard-sized (~0.8 GiB) and the
page-locked footprint to one slice, while the byte layout stays rank-ordered so the
zero-copy ``_merge_gathered_save_shard_shared`` conditions hold unchanged. The staging
frees with the merged payload, so pool-reset semantics are irrelevant here.

Gated by ``XTUNER_GLM52_CHUNKED_SAVE_AG``; only single-tensor dim-0 interleave-1 groups
whose gathered bytes reach ``XTUNER_GLM52_CHUNKED_SAVE_AG_MIN_GB`` (default 4) take the
chunked path. Other groups fall back to the fused gather unchanged.
"""

import os

import torch
import torch.distributed as dist

from xtuner.v1.utils import get_torch_device_module
from xtuner.v1.utils.cpu_merge import _merge_gathered_save_shard_shared
from xtuner.v1.utils.load_spec import SaveShardStep, _is_same_process_group, _pad_tensor_for_save_shard
from xtuner.v1.utils.logger import get_logger


GLM52_CHUNKED_SAVE_AG_ENV = "XTUNER_GLM52_CHUNKED_SAVE_AG"
GLM52_CHUNKED_SAVE_AG_MIN_GB_ENV = "XTUNER_GLM52_CHUNKED_SAVE_AG_MIN_GB"

_DEFAULT_MIN_GB = 4.0

DEVICE_MODULE = get_torch_device_module()


def _gate_active(world_size: int, merge_on_cpu: bool, single_tensor: bool, shard_step: SaveShardStep) -> bool:
    if os.environ.get(GLM52_CHUNKED_SAVE_AG_ENV) != "1":
        return False
    if not merge_on_cpu:
        return False
    if world_size <= 1 or not single_tensor:
        # Nothing to chunk on a single rank; the sliced staging below is defined for
        # exactly one tensor per group (multi-tensor groups defer to the fused path).
        return False
    if shard_step.shard.dim != 0 or shard_step.shard.interleave_factor != 1:
        # The rank-ordered staging below is the byte-level dim-0 concatenation of
        # continuous rank shards; other layouts defer to the fused path. This check must
        # stay ahead of the padding, which only supports these layouts.
        return False
    return True


def _gate_min_bytes() -> int:
    # The runtime-env whitelist forwards unset knobs as set-but-empty strings.
    raw_min_gb = os.environ.get(GLM52_CHUNKED_SAVE_AG_MIN_GB_ENV, "").strip()
    return int((float(raw_min_gb) if raw_min_gb else _DEFAULT_MIN_GB) * 1024**3)


def foreach_all_gather_stage_chunked_paged(
    param: torch.Tensor, group: dist.ProcessGroup | None
) -> tuple[torch.Tensor, list[torch.Tensor], bool]:
    """Gather one padded shard slice-by-slice, staging each slice straight into a page-able
    host buffer.

    Byte-for-byte replica of the ``foreach_all_gather_stage_chunked`` staging layout: the
    slot holds the world_size rank-ordered shard copies back to back, so the shared
    zero-copy merge conditions hold unchanged. The device transient is one shard slice
    (global / world size) and the page-locked footprint is the pinned bounce (the largest
    slice) instead of bucket-sized buffers.

    Args:
        param (torch.Tensor): Exactly one padded local shard tensor, identical shape and
            dtype on every rank of ``group``.
        group (dist.ProcessGroup | None): Process group to gather over (``None`` = world).

    Returns:
        tuple[torch.Tensor, list[torch.Tensor], bool]: ``(staging_base, rank-ordered host
        views, True)``. ``True`` here means the host views are zero-copy views of this
        function's own staging buffer — plain page-able memory, not the file_system
        shared pool; ``_merge_gathered_save_shard_shared`` only relies on that view
        property.
    """
    if group is None:
        group = dist.group.WORLD
    assert group is not None
    dtype = param.dtype
    device = param.device
    shard_numel = param.numel()
    shape = tuple(param.shape)
    world_size = dist.get_world_size(group)
    element_size = param.element_size()
    shard_bytes = shard_numel * element_size

    staging = torch.empty(shard_bytes * world_size, dtype=torch.uint8, device="cpu")
    # world_size slices cap the gathered transient at one shard; tensor_split keeps uneven
    # shards correct (slice sizes differ by at most one element).
    slices = torch.tensor_split(param.flatten(), world_size)
    # The bounce only needs to hold the largest slice, not the whole shard: page-locked
    # pressure is the failure mode this module exists to remove.
    bounce = torch.empty(
        max(piece.numel() for piece in slices), dtype=dtype, device="cpu", pin_memory=device.type != "cpu"
    )
    slice_offsets: list[int] = []
    offset = 0
    for piece in slices:
        slice_offsets.append(offset)
        offset += piece.numel()

    for s_index, piece in enumerate(slices):
        gathered = torch.empty((world_size * piece.numel(),), dtype=dtype, device=device)
        dist.all_gather_into_tensor(gathered, piece, group=group)
        nbytes = piece.numel() * element_size
        for rank_index in range(world_size):
            src = gathered[rank_index * piece.numel() : (rank_index + 1) * piece.numel()]
            if device.type != "cpu":
                # Pinned-bounce D2H: pinned speed without a bucket-sized page-locked buffer.
                bounce[: piece.numel()].copy_(src)
                src_host = bounce[: piece.numel()]
            else:
                src_host = src
            dst_bytes = (rank_index * shard_numel + slice_offsets[s_index]) * element_size
            staging[dst_bytes : dst_bytes + nbytes].view(dtype).copy_(src_host)
    if device.type != "cpu":
        DEVICE_MODULE.synchronize(device)
    host_chunks = [
        staging[rank_index * shard_bytes : (rank_index + 1) * shard_bytes].view(dtype).view(shape)
        for rank_index in range(world_size)
    ]
    get_logger().info(
        "[chunked_save_ag] oversized group chunked: {:.2f} GiB gathered, device transient ~one shard "
        "({:.2f} GiB), page-locked bounce one slice ({:.2f} GiB)",
        shard_bytes * world_size / 1024**3,
        shard_bytes / 1024**3,
        bounce.numel() * element_size / 1024**3,
    )
    return staging, host_chunks, True


def try_chunked_save_unshard(
    tensor_list: list[torch.Tensor],
    shard_steps: list[SaveShardStep],
    merge_on_cpu_flags: list[bool],
) -> list[torch.Tensor] | None:
    """Chunked drop-in for oversized checkpoint-save unshard groups.

    Returns ``None`` when the chunked path does not apply (gate off, small or multi-tensor
    group, non-dim-0 or interleaved shard, device merge) and the caller must use the
    original fused path.

    Args:
        tensor_list (list[torch.Tensor]): FSDP-local shard tensors for one unshard step.
        shard_steps (list[SaveShardStep]): Save work items matching ``tensor_list``.
        merge_on_cpu_flags (list[bool]): Per-tensor merge-on-CPU flags.

    Returns:
        list[torch.Tensor] | None: Merged host tensors, or ``None`` to fall back.
    """
    assert len(tensor_list) == len(shard_steps), "Internal error: tensor and shard-step count mismatch"
    assert tensor_list, "Internal error: empty save all-gather group"
    group = shard_steps[0].shard.group
    assert all(_is_same_process_group(group, shard_step.shard.group) for shard_step in shard_steps), (
        "Internal error: save all-gather group contains different process groups"
    )
    world_size = dist.get_world_size(group)
    if not _gate_active(world_size, all(merge_on_cpu_flags), len(tensor_list) == 1, shard_steps[0]):
        return None
    # Pad before the size check: the padded local shard has rank-invariant numel (the ceil
    # of the shard dim over world_size), so the gathered estimate below — and therefore
    # the chunked-vs-fused decision — is identical on every rank. Estimating from the
    # unpadded tensors could straddle the threshold across ranks and desynchronize the
    # collective counts. The caller re-pads harmlessly when this function returns None.
    padded_tensor_list = [
        _pad_tensor_for_save_shard(tensor, shard_step)
        for tensor, shard_step in zip(tensor_list, shard_steps, strict=True)
    ]
    padded_gathered_bytes = padded_tensor_list[0].numel() * padded_tensor_list[0].element_size() * world_size
    if padded_gathered_bytes < _gate_min_bytes():
        return None
    staging_base, host_chunks, staged_shared = foreach_all_gather_stage_chunked_paged(padded_tensor_list[0], group)
    return [_merge_gathered_save_shard_shared(staging_base, host_chunks, shard_steps[0], staged_shared)]
