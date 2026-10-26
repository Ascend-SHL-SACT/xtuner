# Copyright (c) OpenMMLab. All rights reserved.
"""Vision-tower FSDP unit fusion (``XTUNER_VISION_FSDP_FUSE``).

The GLM-5.3-Flash vision tower shards one FSDP2 unit per ViT block over the *flat
world mesh* (:func:`xtuner.v1.model.compose.glm53.modeling_vision.init_world_mesh`),
so a 512-rank job runs 24 units whose per-block unshard / reshard payload is one
shard of a 16.79 M-parameter block (``count=32800`` elements per rank, i.e. ~65 KB
in bf16 - the job-dyb-infer-07 profile's
``hcom_allGather__612_2_1 count=32800`` is exactly one block's shard).
Those collectives are rendezvous-bound, not bandwidth-bound: a single block's
shard took 80 ms - 2.5 s per op on that job, and the 24 block units' collectives
summed to 59.9 s of the tower's 76.5 s per step at 5% compute coverage.

Grouping K consecutive blocks into a single FSDP2 unit collapses the per-block
collectives into one per group (the mechanism ``XTUNER_FSDP_FUSE_K`` uses for
decoder layers, see :mod:`xtuner.v1.model.fsdp_fuse`), cutting the serial
rendezvous count from ``depth`` to ``ceil(depth / K)`` while the payload - still
only ``K`` blocks' shard per rank - stays negligible.

``XTUNER_VISION_FSDP_FUSE`` unset / ``0`` / any value ``< 2`` = off: the caller
keeps its original per-block sharding loop, byte-identical to HEAD. ``K > 1`` =
fuse K blocks per unit.

Numerics: the forward stays bit-identical (every parameter is still gathered from
the same fp32 shard and cast per parameter), while the backward only matches
statistically - fusing turns 24 per-block bf16 gradient buckets into one per group,
which regroups the reduce-scatter summation. Fusing also keeps the whole tower
resident inside its unshard window (bf16, ~0.81 GB for 24 blocks); measured on 16
dies the per-step ``max_memory`` did not move, because that window sits below the
text-model peak.
"""

import os
from typing import TYPE_CHECKING

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard


if TYPE_CHECKING:
    from xtuner.v1.model.base import BaseModel


def vision_fuse_k() -> int:
    """Read the vision-tower FSDP unit size from ``XTUNER_VISION_FSDP_FUSE``.

    Returns:
        int: blocks per FSDP2 unit; ``0`` when the lever is off (unset, empty,
        ``0``, ``1`` or any negative value), otherwise the requested ``K > 1``.
    """
    raw = os.environ.get("XTUNER_VISION_FSDP_FUSE", "0").strip()
    if not raw:
        return 0
    try:
        fuse_k = int(raw)
    except ValueError as exc:
        raise ValueError(f"XTUNER_VISION_FSDP_FUSE must be an integer >= 0, got {raw!r}") from exc
    return fuse_k if fuse_k > 1 else 0


def vision_block_groups(num_blocks: int, fuse_k: int) -> list[list[int]]:
    """Group ViT block indices into ``fuse_k``-sized FSDP2 units.

    Args:
        num_blocks (int): number of ViT blocks, in forward order.
        fuse_k (int): blocks per FSDP2 unit; must be ``> 1``.

    Returns:
        list[list[int]]: block indices per unit, in forward order. Only the last
        group can be shorter than ``fuse_k``, and it is a single block exactly
        when ``num_blocks % fuse_k == 1``.
    """
    assert fuse_k > 1, "vision_block_groups is the K>1 grouped path only"
    return [list(range(start, min(start + fuse_k, num_blocks))) for start in range(0, num_blocks, fuse_k)]


def shard_vision_blocks(
    model: "BaseModel",
    blocks: nn.ModuleList,
    *,
    mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy,
    offload_policy: CPUOffloadPolicy | None,
    reshard_after_forward: bool,
    fuse_k: int,
) -> None:
    """Shard the ViT blocks as ``fuse_k``-block FSDP2 units.

    The tower still calls every block individually; FSDP2's group forward hooks
    unshard on the group's first block and reshard after its last block, so the
    block-level ``forward`` loop, activation recompute and activation-offload
    hooks are untouched. A trailing group of exactly one block is sharded with the
    model's per-block ``_fully_shard`` so it stays byte-identical to the default
    path. No cross-group forward prefetch is chained: the production setting is a
    single unit (``fuse_k == depth``), and the gain comes from collapsing
    rendezvous rather than from overlap.

    Args:
        model (BaseModel): the owning vision model; provides ``_fully_shard``.
        blocks (nn.ModuleList): ViT blocks in forward order.
        mesh (DeviceMesh): device mesh for sharding.
        mp_policy (MixedPrecisionPolicy): mixed-precision policy.
        offload_policy (CPUOffloadPolicy | None): CPU offload policy.
        reshard_after_forward (bool): reshard the whole group after its forward.
        fuse_k (int): blocks per FSDP2 unit; must be ``> 1``.
    """
    for group in vision_block_groups(len(blocks), fuse_k):
        if len(group) == 1:
            model._fully_shard(
                mesh=mesh,
                mp_policy=mp_policy,
                reshard_after_forward=reshard_after_forward,
                offload_policy=offload_policy,
                module=blocks[group[0]],
            )
        else:
            fully_shard(
                [blocks[i] for i in group],
                mesh=mesh,
                mp_policy=mp_policy,
                reshard_after_forward=reshard_after_forward,
                offload_policy=offload_policy,  # type: ignore[arg-type]  # None disables offload at runtime
            )
