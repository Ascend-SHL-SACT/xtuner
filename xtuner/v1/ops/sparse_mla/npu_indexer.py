# Copyright (c) OpenMMLab. All rights reserved.
"""NPU DSA top-k indexer backend for GLM-5.2.

Three paths:
  - Single sequence (no packing): BSND layout, kernel handles causal internally.
  - Packed multi-sequence (SP=1): TND layout + global cu_seq_lens, V1 kernel.
  - Packed multi-sequence (SP>1): TND layout + prefix-extended KV slice, V2 kernel.
    KV slice extends before shard_start to include the full segment prefix,
    enabling the indexer to select top-k from the complete segment KV (not just
    the local shard). cu_seq_q and cu_seq_kv have the same segment count but
    different per-segment lengths (V2 supports this natively).

Reference:
  - MindSpeed: mindspeed/core/transformer/experimental_attention_variant/
    dsa_kvallgather_context_parallel.py — get_cu_seqlens_qkv_before_attn
  - CANN 9.1.0: cann_ops_transformer.ops.lightning_indexer (LightningIndexerV2)
"""

import torch
import torch_npu

from xtuner.v1.data_proto import SequenceContext


def npu_dsa_topk_indices(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    seq_ctx: SequenceContext,
    *,
    index_head_dim: int,
    index_topk: int,
) -> torch.Tensor:
    """NPU fused DSA top-k index computation.

    Args:
        q: Indexer query ``[1, S, Ni, Di]``.
        k: Indexer key ``[1, S_g, Di]`` (SP-gathered).
        weights: Per-token weights ``[1, S, Ni]``.
        seq_ctx: Sequence context for packed causal masking.
        index_head_dim: Indexer head dimension (128 for GLM-5.2).
        index_topk: Number of top-k tokens to select (2048 for GLM-5.2).

    Returns:
        ``[S, 1, K]`` int64 tensor. Invalid slots padded with -1.
    """
    query_len = q.shape[1]
    kv_len = k.shape[1]
    topk = min(index_topk, kv_len)

    cu_seq_lens = seq_ctx.cu_seq_lens_q
    num_segments = cu_seq_lens.numel() - 1

    if num_segments > 1:
        return _indexer_tnd_packed(
            q, k, weights, seq_ctx, cu_seq_lens, query_len, kv_len, topk, index_head_dim,
        )
    return _indexer_bsnd_single(q, k, weights, seq_ctx, query_len, kv_len, topk)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_prefix_extended_kv_slice(
    cu_seq_q_global: torch.Tensor,
    shard_start: int,
    shard_end: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Compute prefix-extended KV slice range and local cu_seq_lens.

    Implements MindSpeed's ``get_cu_seqlens_qkv_before_attn`` logic:
    for each segment overlapping the shard, Q overlap is clipped to
    ``[max(seg_start, shard_start), min(seg_end, shard_end)]`` while
    KV overlap starts from ``seg_start`` (including prefix tokens before
    shard_start). When a segment starts before shard_start, kv_start
    is extended backward to that segment's start.

    Returns:
        cu_seq_q_local:  Local Q cu_seq_lens ``[0, q1, q1+q2, ...]``.
        cu_seq_k_local:  Local KV cu_seq_lens (same segment count, possibly
                         different per-segment lengths).
        kv_start:        Start of KV slice in global coordinates.
        kv_end:          End of KV slice (= shard_end).
    """
    seg_boundaries = cu_seq_q_global.tolist()
    kv_start = shard_start
    rank_cu_q = [0]
    rank_cu_kv = [0]
    cur_q = 0
    cur_kv = 0

    for i in range(1, len(seg_boundaries)):
        seg_start = seg_boundaries[i - 1]
        seg_end = seg_boundaries[i]
        if seg_end <= shard_start:
            continue
        if seg_start >= shard_end:
            break

        # Q: clipped to shard range
        q_len = min(seg_end, shard_end) - max(seg_start, shard_start)
        if q_len > 0:
            cur_q += q_len
            rank_cu_q.append(cur_q)

        # KV: from segment start to shard_end (includes prefix)
        kv_len = min(seg_end, shard_end) - seg_start
        if kv_len > 0:
            cur_kv += kv_len
            rank_cu_kv.append(cur_kv)

        # Extend kv_start backward when segment starts before shard
        if seg_start < shard_start:
            kv_start = seg_start

    kv_end = shard_end
    cu_q = torch.tensor(rank_cu_q, dtype=torch.int32, device=device)
    cu_kv = torch.tensor(rank_cu_kv, dtype=torch.int32, device=device)
    return cu_q, cu_kv, kv_start, kv_end


def _local_to_global_indices(
    topk_indices: torch.Tensor,
    cu_seq_q_local: torch.Tensor,
    cu_seq_k_local: torch.Tensor,
    kv_slice_offset: int,
    query_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert per-segment local indices to global indices.

    Both V1 and V2 kernels return per-segment local indices (each segment
    starts from 0). This function converts them to global indices via::

        global = local + kv_seg_start + kv_slice_offset

    where ``kv_seg_start`` is the KV segment's start in the local KV slice
    (from ``cu_seq_k_local[:-1]``), and ``kv_slice_offset`` is the global
    offset of the KV slice (``kv_start`` for SP>1, 0 for SP=1).
    """
    kv_seg_starts = cu_seq_k_local[:-1]
    q_positions = torch.arange(query_len, device=device)
    seg_idx = torch.searchsorted(cu_seq_q_local[1:], q_positions, right=True)
    seg_offsets = kv_seg_starts[seg_idx] + kv_slice_offset

    topk_indices = topk_indices.squeeze(1)  # [S, K]
    valid = topk_indices >= 0
    topk_indices = torch.where(valid, topk_indices + seg_offsets[:, None], topk_indices)
    return topk_indices.unsqueeze(1)  # [S, 1, K]


def _apply_packed_causal_mask(
    topk_indices: torch.Tensor,
    seq_ctx: SequenceContext,
    query_len: int,
    kv_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Filter global indices through packed causal mask.

    Sets indices to -1 for KV positions beyond the query's causal range
    (query at global position ``p`` can only attend to ``[seg_start, p+1)``).
    """
    starts, ends = seq_ctx.packed_causal_query_ranges(query_len, device)
    cols = torch.arange(kv_len, device=device)[None, :]
    packed_mask = (cols >= starts[:, None]) & (cols < ends[:, None])

    safe = topk_indices.clamp(min=0, max=kv_len - 1)
    valid = packed_mask[:, None, :].expand(-1, 1, -1).gather(2, safe)
    return topk_indices.masked_fill(~valid, -1)


def _packed_causal_mask(
    seq_ctx: SequenceContext, query_len: int, kv_len: int, device: torch.device,
) -> torch.Tensor:
    """Build packed causal mask ``[query_len, kv_len]``."""
    starts, ends = seq_ctx.packed_causal_query_ranges(query_len, device)
    cols = torch.arange(kv_len, device=device)[None, :]
    return (cols >= starts[:, None]) & (cols < ends[:, None])


# ---------------------------------------------------------------------------
# Single-sequence path (BSND)
# ---------------------------------------------------------------------------

def _indexer_bsnd_single(
    q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor,
    seq_ctx: SequenceContext, query_len: int, kv_len: int, topk: int,
) -> torch.Tensor:
    """Single-sequence: BSND layout, sparse_mode=3 handles causal."""
    q_bsnd = q.contiguous().to(torch.bfloat16)
    k_bsnd = k.unsqueeze(2).contiguous().to(torch.bfloat16)
    w_bsnd = weights.contiguous().to(torch.bfloat16)

    topk_indices, _ = torch_npu.npu_lightning_indexer(
        q_bsnd, k_bsnd, w_bsnd,
        layout_query="BSND", layout_key="BSND",
        sparse_count=topk, sparse_mode=3, return_value=True,
    )
    topk_indices = topk_indices.squeeze(0).to(torch.int64)  # [S, 1, K]

    # Safety: apply packed causal mask (no-op for single sequence)
    if hasattr(seq_ctx, "packed_causal_query_ranges"):
        topk_indices = _apply_packed_causal_mask(
            topk_indices, seq_ctx, query_len, kv_len, q.device,
        )
    return topk_indices


# ---------------------------------------------------------------------------
# Packed multi-sequence path (TND)
# ---------------------------------------------------------------------------

def _indexer_tnd_packed(
    q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor,
    seq_ctx: SequenceContext, cu_seq_lens: torch.Tensor,
    query_len: int, kv_len: int, topk: int, index_head_dim: int,
) -> torch.Tensor:
    """Packed multi-sequence: TND layout + cu_seq_lens.

    SP=1: V1 ``npu_lightning_indexer`` with global KV and global cu_seq_lens.
    SP>1: V2 ``cann_ops_transformer.lightning_indexer`` with prefix-extended
          KV slice and asymmetric cu_seq_q / cu_seq_kv (same segment count,
          different per-segment lengths — V2 supports this natively).
    """
    device = q.device
    shard_start = getattr(seq_ctx, '_shard_start', 0)

    # BSND → TND: [1, S, N, D] → [S, N, D]
    q_tnd = q.squeeze(0).contiguous().to(torch.bfloat16)       # [S, Ni, Di]
    # V2 requires float32 weights; V1 also accepts float32
    w_tnd = (weights.squeeze(0) * (index_head_dim ** -0.5)).contiguous().to(torch.float32)

    cu_seq_q_global = cu_seq_lens.to(torch.int32).to(device)

    is_sp1 = (shard_start == 0 and query_len == cu_seq_q_global[-1].item())

    if is_sp1:
        # ── SP=1: V1 kernel with global KV ──
        k_tnd = k.squeeze(0).unsqueeze(1).contiguous().to(torch.bfloat16)
        cu_seq_q_local = cu_seq_q_global
        cu_seq_k_local = seq_ctx.cu_seq_lens_k.to(torch.int32).to(device)
        kv_slice_offset = 0

        topk_indices, _ = torch_npu.npu_lightning_indexer(
            q_tnd, k_tnd, w_tnd,
            actual_seq_lengths_query=cu_seq_q_local,
            actual_seq_lengths_key=cu_seq_k_local,
            layout_query="TND", layout_key="TND",
            sparse_count=topk, sparse_mode=3, return_value=True,
        )
    else:
        # ── SP>1: V2 kernel with prefix-extended KV slice ──
        from cann_ops_transformer.ops import lightning_indexer as _li_v2
        from cann_ops_transformer.ops import lightning_indexer_metadata as _li_v2_meta

        shard_end = shard_start + query_len
        cu_seq_q_local, cu_seq_k_local, kv_start, kv_end = _compute_prefix_extended_kv_slice(
            cu_seq_q_global, shard_start, shard_end, device,
        )

        # Slice KV to [kv_start, kv_end] — may be wider than [shard_start, shard_end]
        k_tnd = k.squeeze(0)[kv_start:kv_end].unsqueeze(1).contiguous().to(torch.bfloat16)
        kv_slice_offset = kv_start

        meta = _li_v2_meta(
            num_heads_q=q.shape[2], num_heads_k=1,
            head_dim=index_head_dim, topk=topk,
            cu_seqlens_q=cu_seq_q_local, cu_seqlens_k=cu_seq_k_local,
            batch_size=cu_seq_q_local.numel() - 1,
            max_seqlen_q=query_len, max_seqlen_k=kv_end - kv_start,
            layout_q="TND", layout_k="TND",
            mask_mode=3, cmp_ratio=1,
        )
        topk_indices, _ = _li_v2(
            q_tnd, k_tnd, w_tnd, topk,
            cu_seqlens_q=cu_seq_q_local, cu_seqlens_k=cu_seq_k_local,
            metadata=meta, max_seqlen_q=query_len,
            layout_q="TND", layout_k="TND",
            mask_mode=3, cmp_ratio=1, return_value=0,
        )

    # ── Common: per-segment local → global indices ──
    topk_indices = topk_indices.to(torch.int64)
    topk_indices = _local_to_global_indices(
        topk_indices, cu_seq_q_local, cu_seq_k_local,
        kv_slice_offset, query_len, device,
    )

    # ── Common: packed causal mask filter ──
    return _apply_packed_causal_mask(
        topk_indices, seq_ctx, query_len, kv_len, device,
    )
