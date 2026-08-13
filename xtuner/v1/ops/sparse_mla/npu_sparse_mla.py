# Copyright (c) OpenMMLab. All rights reserved.
"""NPU SparseMLA backend for GLM-5.2 DSA attention.

Wraps ``torch_npu.npu_sparse_flash_attention`` (V1) for fused sparse MLA
on Ascend NPU (910B/910C).

Three paths:
  - Single sequence (no packing): BSND layout.
  - Packed multi-sequence (SP=1): TND layout + global cu_seq_lens.
  - Packed multi-sequence (SP>1): TND layout + prefix-extended KV slice
    with asymmetric cu_seq_q / cu_seq_kv (same segment count, different
    per-segment lengths). KV slice extends before shard_start to include
    the full segment prefix.

Key implementation notes:
  - ``sparse_mode=3`` (RightDownCausal): kernel applies per-segment causal
    mask internally. Requires valid (non -1) indices at training scale;
    -1 is rewritten to query's own global position (self-attention).
  - ``self_pos`` must use GLOBAL position (``arange(S) + shard_start``)
    for correct global→local conversion. Using local position causes
    V1 kernel backward to produce NaN (sparse_mode=3 + multi-segment TND).
  - indices passed to kernel are per-segment local (each segment from 0).

Reference: mindspeed/core/transformer/experimental_attention_variant/dsa_fused.py:426-475
"""

import torch
import torch_npu

from .protocol import SparseMLAOutputs


def npu_sparse_mla(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    scaling: float | None,
    value_dim: int | None = None,
    *,
    seq_ctx=None,
) -> SparseMLAOutputs:
    """NPU fused sparse MLA attention.

    Args:
        q: Absorbed query ``[S, N, Rkv + Dr]``.
        kv: Absorbed key-value ``[S_g, 1, Rkv + Dr]``.
        indices: Top-k sparse indices ``[S, 1, K]``. ``-1`` marks invalid slots.
        scaling: Softmax scale (typically ``qk_head_dim ** -0.5``).
        value_dim: Value dimension (equals ``Rkv`` for absorbed MLA).
        seq_ctx: SequenceContext for packed causal masking.

    Returns:
        SparseMLAOutputs with ``raw_output`` ``[S, N, value_dim]`` and
        ``softmax_lse`` ``[S, N]``.
    """
    seq_len, num_heads, q_dim = q.shape
    kv_len = kv.shape[0]

    if value_dim is None:
        value_dim = q_dim
        rope_dim = 0
    else:
        rope_dim = q_dim - value_dim

    q_nope = q[..., :value_dim]       # [S, N, Rkv]
    q_rope = q[..., value_dim:]       # [S, N, Dr]
    kv_compressed = kv[..., :value_dim]  # [S_g, 1, Rkv]
    k_rope = kv[..., value_dim:]      # [S_g, 1, Dr]

    scale_value = float(scaling) if scaling is not None else (value_dim + rope_dim) ** -0.5

    use_tnd = (
        seq_ctx is not None
        and getattr(seq_ctx, "cu_seq_lens_q", None) is not None
        and seq_ctx.cu_seq_lens_q.numel() > 2
    )

    if use_tnd:
        return _sparse_mla_tnd_packed(
            q_nope, q_rope, kv_compressed, k_rope,
            indices, seq_ctx, seq_len, kv_len, num_heads, scale_value,
        )
    return _sparse_mla_bsnd_single(
        q_nope, q_rope, kv_compressed, k_rope,
        indices, seq_len, kv_len, num_heads, scale_value,
    )


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

    For each segment overlapping ``[shard_start, shard_end)``:
      - Q overlap is clipped to ``[max(seg, shard_start), min(seg, shard_end)]``
      - KV overlap starts from ``seg_start`` (includes prefix before shard_start)
      - When ``seg_start < shard_start``, ``kv_start`` extends backward

    Returns ``(cu_seq_q_local, cu_seq_k_local, kv_start, kv_end)``.
    The two cu_seq tensors have the same segment count but possibly
    different per-segment lengths (KV segments may be longer due to prefix).
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

        q_len = min(seg_end, shard_end) - max(seg_start, shard_start)
        if q_len > 0:
            cur_q += q_len
            rank_cu_q.append(cur_q)

        kv_len_seg = min(seg_end, shard_end) - seg_start
        if kv_len_seg > 0:
            cur_kv += kv_len_seg
            rank_cu_kv.append(cur_kv)

        if seg_start < shard_start:
            kv_start = seg_start

    kv_end = shard_end
    cu_q = torch.tensor(rank_cu_q, dtype=torch.int32, device=device)
    cu_kv = torch.tensor(rank_cu_kv, dtype=torch.int32, device=device)
    return cu_q, cu_kv, kv_start, kv_end


def _rewrite_invalid_indices(
    indices: torch.Tensor,
    shard_start: int,
    device: torch.device,
) -> torch.Tensor:
    """Rewrite -1 (invalid) indices to query's own GLOBAL position.

    ``sparse_mode=3`` kernel requires valid indices at training scale;
    -1 causes aicore exception. The rewritten self-attention position
    is always within causal range and segment boundaries.

    ``self_pos`` must be GLOBAL (``arange(S) + shard_start``) for correct
    global→local conversion downstream. Using local position causes V1
    kernel backward NaN (sparse_mode=3 + multi-segment TND).
    """
    S = indices.shape[0]
    self_pos = torch.arange(S, device=device, dtype=indices.dtype) + shard_start
    return torch.where(
        indices == -1,
        self_pos.unsqueeze(1).unsqueeze(2).expand_as(indices),
        indices,
    ).to(torch.int32)


def _global_to_local_indices(
    sparse_indices: torch.Tensor,
    cu_seq_q_local: torch.Tensor,
    cu_seq_k_local: torch.Tensor,
    kv_slice_offset: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert global indices to per-segment local indices for TND kernel.

    ``sparse_flash_attention`` expects per-segment local indices (each
    segment from 0). The reverse of ``_local_to_global_indices`` in
    the indexer::

        local = global - kv_seg_start - kv_slice_offset

    ``kv_seg_start`` comes from ``cu_seq_k_local[:-1]`` (KV segment starts).
    Segment membership is determined via ``cu_seq_q_local`` boundaries
    (Q and KV have the same segment count).
    """
    kv_seg_starts = cu_seq_k_local[:-1]
    q_positions = torch.arange(seq_len, device=device, dtype=torch.int32)
    seg_idx = torch.searchsorted(cu_seq_q_local[1:], q_positions, right=True)
    global_offsets = kv_seg_starts[seg_idx] + kv_slice_offset

    local_indices = sparse_indices.squeeze(1).clone()
    local_indices = local_indices - global_offsets.unsqueeze(1)
    local_indices = local_indices.clamp(min=0)
    return local_indices.unsqueeze(1).contiguous()


# ---------------------------------------------------------------------------
# Single-sequence path (BSND)
# ---------------------------------------------------------------------------

def _sparse_mla_bsnd_single(
    q_nope, q_rope, kv_compressed, k_rope,
    indices, seq_len, kv_len, num_heads, scale_value,
) -> SparseMLAOutputs:
    """Single-sequence: BSND layout, sparse_mode=3."""
    device = q_nope.device

    # Rewrite -1 to self-attention position
    safe_indices = _rewrite_invalid_indices(indices, 0, device)

    attn_outs = torch_npu.npu_sparse_flash_attention(
        q_nope.unsqueeze(0).contiguous(),         # [1, S, N, Rkv]
        kv_compressed.unsqueeze(0).contiguous(),  # [1, S_g, 1, Rkv]
        kv_compressed.unsqueeze(0).contiguous(),  # value = key
        sparse_indices=safe_indices.unsqueeze(0).contiguous(),
        block_table=None,
        actual_seq_lengths_query=torch.tensor([seq_len], dtype=torch.int32, device=device),
        actual_seq_lengths_kv=torch.tensor([kv_len], dtype=torch.int32, device=device),
        query_rope=q_rope.unsqueeze(0).contiguous(),
        key_rope=k_rope.unsqueeze(0).contiguous(),
        scale_value=scale_value,
        sparse_block_size=1,
        layout_query="BSND",
        layout_kv="BSND",
        sparse_mode=3,
        attention_mode=2,
        return_softmax_lse=True,
    )
    return _parse_attn_outs(attn_outs, seq_len, num_heads, device)


# ---------------------------------------------------------------------------
# Packed multi-sequence path (TND)
# ---------------------------------------------------------------------------

def _sparse_mla_tnd_packed(
    q_nope, q_rope, kv_compressed, k_rope,
    indices, seq_ctx, seq_len, kv_len, num_heads, scale_value,
) -> SparseMLAOutputs:
    """Packed multi-sequence: TND layout + cu_seq_lens.

    SP=1: Full global KV, global cu_seq_lens.
    SP>1: Prefix-extended KV slice with asymmetric cu_seq_q / cu_seq_kv.
          KV slice ``[kv_start, kv_end)`` may extend before ``shard_start``
          to include the full segment prefix (MindSpeed approach).
    """
    device = q_nope.device
    shard_start = getattr(seq_ctx, '_shard_start', 0)

    cu_seq_q_global = seq_ctx.cu_seq_lens_q.to(torch.int32).to(device)
    is_sp1 = (shard_start == 0 and seq_len == cu_seq_q_global[-1].item())

    if is_sp1:
        # ── SP=1: full global KV ──
        key_tnd = kv_compressed.contiguous()          # [T_g, 1, Rkv]
        value_tnd = key_tnd
        key_rope_tnd = k_rope.contiguous()            # [T_g, 1, Dr]
        cu_seq_q_local = cu_seq_q_global
        cu_seq_k_local = seq_ctx.cu_seq_lens_k.to(torch.int32).to(device)
        kv_slice_offset = 0
    else:
        # ── SP>1: prefix-extended KV slice ──
        shard_end = shard_start + seq_len
        cu_seq_q_local, cu_seq_k_local, kv_start, kv_end = _compute_prefix_extended_kv_slice(
            cu_seq_q_global, shard_start, shard_end, device,
        )
        key_tnd = kv_compressed[kv_start:kv_end].contiguous()
        value_tnd = key_tnd
        key_rope_tnd = k_rope[kv_start:kv_end].contiguous()
        kv_slice_offset = kv_start

    # TND tensors (no batch dim)
    query_tnd = q_nope.contiguous()                   # [S, N, Rkv]
    query_rope_tnd = q_rope.contiguous()              # [S, N, Dr]

    # Rewrite -1 → global self position (sparse_mode=3 requires valid indices)
    sparse_indices = _rewrite_invalid_indices(indices, shard_start, device)

    # Convert global indices → per-segment local for TND kernel
    sparse_indices_tnd = _global_to_local_indices(
        sparse_indices, cu_seq_q_local, cu_seq_k_local,
        kv_slice_offset, seq_len, device,
    )

    attn_outs = torch_npu.npu_sparse_flash_attention(
        query_tnd, key_tnd, value_tnd,
        sparse_indices=sparse_indices_tnd,
        block_table=None,
        actual_seq_lengths_query=cu_seq_q_local,
        actual_seq_lengths_kv=cu_seq_k_local,
        query_rope=query_rope_tnd,
        key_rope=key_rope_tnd,
        scale_value=scale_value,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="TND",
        sparse_mode=3,
        attention_mode=2,
        return_softmax_lse=True,
    )
    return _parse_attn_outs(attn_outs, seq_len, num_heads, device)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def _parse_attn_outs(attn_outs, seq_len: int, num_heads: int, device: torch.device) -> SparseMLAOutputs:
    """Parse ``npu_sparse_flash_attention`` outputs into SparseMLAOutputs."""
    if isinstance(attn_outs, torch.Tensor):
        attn_output = attn_outs.contiguous()
        softmax_lse = None
    else:
        attn_output = attn_outs[0].contiguous()
        softmax_max = attn_outs[1] if len(attn_outs) > 1 else None
        softmax_sum = attn_outs[2] if len(attn_outs) > 2 else None
        if softmax_sum is not None and softmax_max is not None:
            lse = softmax_max.float() + torch.log(softmax_sum.float() + 1e-30)
            while lse.dim() > 2:
                lse = lse.squeeze(0)
            softmax_lse = lse[:seq_len, :num_heads].contiguous()
        else:
            softmax_lse = None

    # [1, S, N, Rkv] (BSND) or [S, N, Rkv] (TND)
    raw_output = attn_output.squeeze(0) if attn_output.dim() == 4 else attn_output

    if softmax_lse is None:
        softmax_lse = torch.zeros(seq_len, num_heads, device=device, dtype=torch.float32)

    return SparseMLAOutputs(raw_output=raw_output, softmax_lse=softmax_lse)
