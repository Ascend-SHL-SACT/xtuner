"""Interleaved expert-parallel partition for MoE ``GroupedLinear``.

Gated by ``XTUNER_MOE_INTERLEAVED_EP=1`` (default off, contiguous partition). When
enabled, expert ``e`` is placed on EP rank ``e % ep_size`` with local index
``e // ep_size``, which spreads contiguous expert-id routing skew evenly across EP
groups. Weight placement (``shard_interleaved``) and dispatch (``remap_token_counts``,
``dispatch_sort_key``) share the same permutation so each rank's local experts match
the tokens it receives.
"""

import os

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor


def is_interleaved_ep() -> bool:
    """Return whether interleaved expert partition is enabled."""
    return os.environ.get("XTUNER_MOE_INTERLEAVED_EP", "0") == "1"


def interleaved_partition(n_routed_experts: int, ep_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the interleaved partition permutation.

    Expert ``e`` -> EP rank ``e % ep_size``, local index ``e // ep_size``.

    Args:
        n_routed_experts (int): Total number of routed experts.
        ep_size (int): Expert-parallel world size.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(perm, inv_perm)`` int64 tensors of
        length ``n_routed_experts``. ``perm[i]`` is the global expert id placed at
        storage position ``i``; ``inv_perm[e]`` is the storage position of global
        expert ``e`` (also the sort key ordering tokens by target rank then local
        index for the dispatch permute).
    """
    assert n_routed_experts % ep_size == 0
    npr = n_routed_experts // ep_size
    perm = [(i // npr) + (i % npr) * ep_size for i in range(n_routed_experts)]
    inv = [0] * n_routed_experts
    for i, e in enumerate(perm):
        inv[e] = i
    return (
        torch.tensor(perm, dtype=torch.int64),
        torch.tensor(inv, dtype=torch.int64),
    )


def shard_interleaved(tensor: torch.Tensor, num_routed_experts: int, ep_mesh: DeviceMesh) -> DTensor:
    """Distribute ``tensor`` across ``ep_mesh`` with interleaved ``Shard(0)``.

    Reorders the expert dimension (dim 0) so the standard contiguous ``Shard(0)``
    slice assigns rank ``r`` the interleaved expert set ``{r, r + ep, 2*ep + r, ...}``
    in local order, matching the dispatch routing.

    Args:
        tensor (torch.Tensor): Expert weight of shape ``[E * chunk, ...]`` where dim 0
            groups ``E`` experts of ``chunk`` rows each.
        num_routed_experts (int): Total number of routed experts (``E``).
        ep_mesh (DeviceMesh): Expert-parallel mesh to shard across.

    Returns:
        DTensor: The interleaved-sharded weight.
    """
    ep_size = ep_mesh.size()
    perm, _ = interleaved_partition(num_routed_experts, ep_size)
    grouped = tensor.reshape(num_routed_experts, tensor.shape[0] // num_routed_experts, -1)
    grouped = torch.index_select(grouped, 0, perm)
    grouped = grouped.reshape(tensor.shape)
    return distribute_tensor(grouped, ep_mesh, [Shard(0)])


def remap_token_counts(tokens_per_expert: torch.Tensor, n_routed_experts: int, ep_size: int) -> torch.Tensor:
    """Gather per-expert token counts from expert-id order into storage order.

    The count all-to-all uses an equal-split exchange, so it must operate on a vector
    whose contiguous blocks are each rank's owned (interleaved) experts. This gather
    by ``perm`` reorders ``tokens_per_expert`` so position ``i`` holds the count for
    the expert at storage position ``i``.

    Args:
        tokens_per_expert (torch.Tensor): Per-expert token counts in global expert-id
            order.
        n_routed_experts (int): Total number of routed experts.
        ep_size (int): Expert-parallel world size.

    Returns:
        torch.Tensor: Counts reordered to storage order.
    """
    perm, _ = interleaved_partition(n_routed_experts, ep_size)
    perm = perm.to(tokens_per_expert.device)
    return torch.index_select(tokens_per_expert, 0, perm)


def dispatch_sort_key(topk_ids: torch.Tensor, n_routed_experts: int, ep_size: int) -> torch.Tensor:
    """Map routed expert ids to the storage-position sort key for dispatch permute.

    Sorting tokens by this key groups them by ``(target EP rank, local index)``, the
    contiguous layout the payload all-to-all splits against under interleaved
    partition.

    Args:
        topk_ids (torch.Tensor): Routed expert id per token.
        n_routed_experts (int): Total number of routed experts.
        ep_size (int): Expert-parallel world size.

    Returns:
        torch.Tensor: Sort key (int32) per token.
    """
    _, inv = interleaved_partition(n_routed_experts, ep_size)
    inv = inv.to(topk_ids.device)
    # Flatten to 1-D: NPU aclnnIndexSelect rejects index dimNum > 1.
    flat = torch.index_select(inv, 0, topk_ids.reshape(-1).to(torch.int64))
    return flat.reshape(topk_ids.shape).to(torch.int32)


def shard_expert_tensor(
    tensor: torch.Tensor,
    n_routed_experts: int,
    ep_mesh: DeviceMesh,
) -> DTensor:
    """Shard an expert-stacked tensor across ``ep_mesh`` for expert parallelism.

    When interleaved EP is enabled the expert dimension is permuted to the
    interleaved rank assignment before the contiguous ``Shard(0)`` slice;
    otherwise the tensor is sharded in place (the default contiguous partition).

    Args:
        tensor (torch.Tensor): Expert weight of shape ``[E * chunk, ...]``.
        n_routed_experts (int): Total number of routed experts (``E``).
        ep_mesh (DeviceMesh): Expert-parallel mesh to shard across.

    Returns:
        DTensor: The sharded expert tensor.
    """
    if is_interleaved_ep():
        return shard_interleaved(tensor, n_routed_experts, ep_mesh)
    return distribute_tensor(tensor, ep_mesh, [Shard(0)])


def histc_for_dispatch(
    topk_ids: torch.Tensor,
    n_routed_experts: int,
    ep_size: int,
) -> torch.Tensor:
    """Return per-expert token counts in all-to-all storage order.

    Counts are computed with ``torch.histc`` and, when interleaved EP is enabled,
    gathered into storage order so the equal-split count exchange sends each rank
    the contiguous block of its own (interleaved) experts. When disabled the
    expert-id-order counts are returned unchanged.

    Args:
        topk_ids (torch.Tensor): Routed expert id per token.
        n_routed_experts (int): Total number of routed experts.
        ep_size (int): Expert-parallel world size.

    Returns:
        torch.Tensor: Per-expert token counts in storage order.
    """
    counts = torch.histc(topk_ids, bins=n_routed_experts, min=0, max=n_routed_experts)
    if is_interleaved_ep():
        counts = remap_token_counts(counts, n_routed_experts, ep_size)
    return counts


def sort_key_for_dispatch(
    topk_ids: torch.Tensor,
    n_routed_experts: int,
    ep_size: int,
) -> torch.Tensor:
    """Return the sort key grouping tokens by target EP rank for dispatch.

    When interleaved EP is enabled, tokens are grouped by interleaved rank then
    local index (``dispatch_sort_key``); otherwise tokens are grouped by expert id
    via ``topk_ids`` cast to int32 (the default contiguous partition sort key).

    Args:
        topk_ids (torch.Tensor): Routed expert id per token.
        n_routed_experts (int): Total number of routed experts.
        ep_size (int): Expert-parallel world size.

    Returns:
        torch.Tensor: Sort key (int32) per token.
    """
    if is_interleaved_ep():
        return dispatch_sort_key(topk_ids, n_routed_experts, ep_size)
    return topk_ids.to(torch.int32)
