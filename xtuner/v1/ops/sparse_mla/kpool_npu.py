# Copyright © 2026 Huawei Technologies Co., Ltd.
"""NPU KPool indexer top-k backend (``INDEXER_BACKEND=torch_npu``).

Same pool-space protocol as :func:`kpool.torch_kpool_topk_indices` (build pools, score
queries against pool keys, top-k within per-query causal-visibility ranges, expand back to
token ids + tail), with the scoring stage replaced by the pair the pre-merge GLM-5.3
implementation proved end to end at 64K context on 910C:

- on NPU, when the fused lightning indexer kernel is importable, scoring runs in
  ``[BS, BP]`` tiles and the per-head ``[S, NH, P]`` fp32 score tensor never materializes
  (8.6 GB at ``S=16384`` -- the reason the ``torch`` reference path OOMs at that scale);
- otherwise an fp32 query-chunked eager fallback bounds that transient to ``query_chunk_size``
  queries, or to ``_SCORE_CHUNK_ELEMS`` score elements when the caller leaves it unset.

Like the pre-merge implementation, scoring/top-k run **per document segment**: a query only
ranks its own document's pools, so the score/mask/top-k tensors are ``[rows_in_doc,
pools_in_doc]`` instead of ``[S, P]``. Scoring every local query against every gathered pool
(the whole-segment variant) computed the cross-document pairs and then discarded them -- the
128K profile showed that variant spending ~19 s/step in the lightning kernel plus ~15 s/step
in full-width mask/top-k/expand glue, versus a few seconds for the segmented structure.

Below the scoring stage the code mirrors the reference path's ``-inf``-to-``-1`` invalid-slot
convention. Scoring/top-k stay per segment (per-query causal ranges are ragged), but the
validity decision and the pool-id -> token-id expand are batched across the whole shard: the
selected column id is valid exactly when it is below the row's visible-pool prefix (the same
decision the old implementation made by re-gathering the visibility bit -- the fused kernel
stores ``finfo.min``, not ``-inf``, so scores cannot be compared). The expand itself is pure
elementwise arithmetic -- pool ``p`` of document ``d`` (a document-local column id ``c`` in the
slot buffer) covers tokens ``doc_start_d + c * kpool + slot`` -- with no gather at all: random
-read gathers on 910C are latency-bound at ~6 ns/element, and both earlier forms (one
``GatherV3`` per segment over ``[rows, k, kpool]`` offsets; the batched whole-shard
``index_select`` over the per-pool token-base table, 4x the gathered elements) cost 10-100 ms
per call at 128K, ~11-19 s of device time per step. Batched here also means the per-row
broadcast tensors are built once per context and cached (one blocking H2D copy of the
cumulative lengths per micro-batch instead of one per indexer call -- a per-call H2D drains the
deeply queued training stream and serialized host enqueueing against device execution).

The fused scoring stage itself -- the triton kernel, its autograd wrapper / tile picker and the
batched "into" entry -- is ported from MindSpeed-MM
``mindspeed_mm/fsdp/ops/dsa/lightning_indexer_score.py`` and
``mindspeed_mm/fsdp/ops/dsa/triton/lightning_indexer_score.py`` (upstream provenance: Huawei,
Apache-2.0) and lives in the guarded section at the bottom of this file.
"""

from __future__ import annotations

import functools
import os
import weakref

import torch
from torch import Tensor

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.ops.comm import gather_for_sequence_parallel

from .kpool import _visible_tail_tokens, build_pools, kpool_output_width


try:
    import triton
    import triton.language as tl
except ImportError:  # triton-ascend unavailable (CPU CI / reference boxes): eager fallback only
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]

__all__ = ["npu_kpool_topk_indices"]

# Batched indexer scoring (one score launch per document segment into a whole-call score
# buffer + one whole-shard topk) instead of the per-segment pad -> score -> topk -> scatter
# chain. Setting it to ``0`` restores the legacy per-segment path bit for bit.
_BATCHED_SCORE = os.environ.get("XTUNER_GLM53_KPOOL_BATCHED_SCORE", "1") == "1"

# Eager-fallback budget: at most 2**28 fp32 score elements per chunk (~1 GiB), matching the
# pre-merge implementation's ``_score_chunk_elems``.
_SCORE_CHUNK_ELEMS = 1 << 28

# Slot-buffer pad sentinel: far above any per-document pool count (the visible prefix is
# bounded by 2**15 for 128K contexts) so it always fails the ``sel < row_prefix`` validity
# test, yet small enough that the arithmetic expand below (``sel * kpool + doc_start``)
# stays inside int32.
_SEL_PAD = 1 << 26

# Batched-expand broadcast tensors per (context, kpool, query length, device). Keyed weakly by the
# SequenceContext so the entries die with the micro-batch; the build pays exactly one
# blocking H2D copy (the cumulative length list) per context instead of one per indexer
# call -- a per-call H2D drains the deeply queued training stream on every call and
# serialized host enqueueing against device execution (~45 s/step at 128K).
_KPOOL_META_CACHE: weakref.WeakKeyDictionary[
    SequenceContext, dict[tuple[int, int, torch.device], tuple[Tensor, Tensor, Tensor]]
] = weakref.WeakKeyDictionary()


@torch.no_grad()
def npu_kpool_topk_indices(
    q: Tensor,
    k: Tensor,
    gate_scores: Tensor,
    weights: Tensor,
    kpool_ape: Tensor,
    seq_ctx: SequenceContext,
    *,
    index_head_dim: int,
    index_topk: int,
    index_kpool: int = 4,
    always_select_tail: bool = True,
    alignment: int = 512,
    query_chunk_size: int | None = None,
) -> Tensor:
    """NPU KPool indexer top-k: build pools (A), score + top-k per document segment (B),
    expand back to token ids + tail (C).

    Args:
        q (Tensor): Index query, shape ``[1, S, index_n_heads, index_head_dim]``, bf16.
        k (Tensor): Index key, shape ``[1, S, index_head_dim]``, bf16.
        gate_scores (Tensor): ``index_kpool_compress_gate(hidden_states)``, shape
            ``[1, S, index_head_dim]``.
        weights (Tensor): ``weights_proj(hidden_states)``, shape ``[1, S, index_n_heads]``,
            **unscaled** raw projection output (this function applies the
            ``index_n_heads**-0.5 * index_head_dim**-0.5`` scaling).
        kpool_ape (Tensor): ``index_kpool_compress_ape``, shape ``[index_kpool, index_head_dim]``.
        seq_ctx (SequenceContext): Packed-sequence and SP metadata.
        index_head_dim (int): Indexer head dimension (scaling only; the scorers infer dims
            from q/k).
        index_topk (int): Semantic token budget (e.g. 2048); must be divisible by
            ``index_kpool``.
        index_kpool (int): Pool size.
        always_select_tail (bool): Whether to append the incomplete trailing pool.
        alignment (int): Output width alignment of the chosen SparseMLA forward backend
            (``torch_npu`` uses 1 -- the kernel takes any index width).
        query_chunk_size (int | None): Eager-fallback query-chunk bound; ignored by the fused
            kernel, which self-bounds through its tile picker.

    Returns:
        Tensor: ``[S, 1, kpool_output_width(index_topk, index_kpool, alignment)]`` int32.
    """
    assert index_topk % index_kpool == 0
    # SP: queries stay sharded; the key side gathers *before* pooling, because a pool can
    # straddle the shard seam (see build_pools).
    sp_mesh = seq_ctx.sequence_parallel_mesh
    k = gather_for_sequence_parallel(k.squeeze(0), dim=0, sp_mesh=sp_mesh)
    gate_scores = gather_for_sequence_parallel(gate_scores.squeeze(0), dim=0, sp_mesh=sp_mesh)

    pool_key, _, _ = build_pools(k, gate_scores, kpool_ape, seq_ctx, index_kpool=index_kpool)

    q_local = q.squeeze(0).contiguous()  # [S, NH, Di]
    seq_len, device = q_local.shape[0], q_local.device
    width = kpool_output_width(index_topk, index_kpool, alignment)
    out = torch.full((seq_len, width), -1, dtype=torch.int32, device=device)

    select_k = index_topk // index_kpool
    w_all = weights.squeeze(0)

    ranges = _segment_ranges(seq_ctx, seq_len, index_kpool)
    if ranges:
        row_prefix, row_doc_start, slot_offsets = _expand_meta(seq_ctx, seq_len, index_kpool, device)

        # Out-of-place scaling (matches kpool.py's reference path): if a caller ever hands in a
        # shared/cached fp32 weights tensor, ``.float()`` is a no-op alias and an in-place
        # ``mul_`` would silently compound the NH scaling into every consumer.
        w_all_f = w_all.float() * (q_local.shape[1] ** -0.5)
        max_pools = max(r[3] - r[2] for r in ranges)
        if _BATCHED_SCORE and lightning_indexer_score_into is not None and q_local.device.type == "npu":
            if select_k <= max_pools:
                # Batched scoring: one kernel launch per segment into a whole-call fp32 buffer
                # and one whole-shard topk -- no per-segment pad / topk / scatter (see
                # _score_and_topk_batched). The returned ids are document-local column ids
                # exactly like the legacy slot buffer's; where the legacy path held
                # ``_SEL_PAD`` these hold real column ids past the row's visible prefix, which
                # the shared validity test below rejects the same way. The gate is only
                # ``select_k <= max_pools`` (the *largest* segment): rows whose own visible
                # prefix is narrower still keep every real column, because visible columns
                # score above the ``finfo.min`` prefill and every prefill/pad id fails the
                # prefix test (see the ``select_k`` contract in ``_score_and_topk_batched``).
                # Do not tighten this to a per-document minimum -- mixed packs would lose the
                # batched path for no correctness gain.
                sel_all = _score_and_topk_batched(
                    q_local, pool_key, w_all_f, ranges, row_prefix, select_k, index_head_dim
                )
            else:
                # Degenerate workload (every document shorter than ``index_topk`` tokens, e.g.
                # SFT over short samples): each row's visible prefix is below ``select_k``, so
                # the legacy per-segment ``topk(min(select_k, num_pools))`` returns *every*
                # real pool column and leaves the rest at ``_SEL_PAD``. That buffer is exactly
                # the column-id arange masked by the row prefix -- scoring, the key pad, the
                # per-segment pad/topk chain and the batched score buffer are all dead work
                # here, so build it directly. Values, per-slot validity and the slot count the
                # downstream expand consumes are identical to the legacy buffer; only the order
                # of the surviving ids inside a row differs (column order vs score order),
                # which the gather consumer sums over insensitively.
                col_ids = torch.arange(select_k, device=device, dtype=torch.int32)
                sel_all = torch.where(col_ids.unsqueeze(0) < row_prefix.unsqueeze(-1), col_ids.unsqueeze(0), _SEL_PAD)
        else:
            # ``select_k``-wide slot buffer, written per segment (only the first
            # ``min(select_k, num_pools)`` columns); pad columns hold ``_SEL_PAD``, which is never a
            # valid pool id, so the batched validity test below rejects them without a per-segment
            # mask. Everything after the loop -- validity, the arithmetic token expand and the
            # output write -- is then one broadcast per call instead of one chain per segment.
            sel_all = torch.full((seq_len, select_k), _SEL_PAD, dtype=torch.int32, device=device)
            p_arange = torch.arange(max_pools, device=device)
            for row_lo, row_hi, pool_lo, pool_hi, _doc_start in ranges:
                num_pools = pool_hi - pool_lo
                # A query sees exactly the complete pools formed before it inside its own
                # document, i.e. the prefix ``[0, (local_pos + 1) // kpool)`` of the segment's
                # pool slice -- the pool-space analogue of ``packed_causal_query_ranges``.
                # ``row_prefix`` is the per-row prefix bound; slicing it (and ``p_arange``)
                # keeps the per-segment mask down to a single compare.
                visible = p_arange[:num_pools] < row_prefix[row_lo:row_hi].unsqueeze(-1)

                index_scores = _score_pools(
                    q_local[row_lo:row_hi],
                    pool_key[pool_lo:pool_hi],
                    w_all_f[row_lo:row_hi],
                    index_head_dim,
                    query_chunk_size,
                    visible=visible,
                )
                selected = index_scores.topk(min(select_k, num_pools), dim=-1).indices
                sel_all[row_lo:row_hi, : selected.shape[-1]] = selected

        # Pool-id -> token-id expand, batched over the whole shard and *gather-free*: pool
        # ``p`` of document ``d`` -- a document-local column id ``c`` in ``sel_all`` --
        # covers tokens ``doc_start_d + c * kpool + slot``, so the token ids are pure
        # elementwise arithmetic on broadcast tensors. No ``index_select`` at all:
        # random-read gathers on 910C are latency-bound at ~6 ns/element, and both earlier
        # forms (the per-segment ``[rows, k, kpool]`` offset gather; the batched whole-shard
        # gather over the per-pool token-base table, which moved 4x the elements) cost
        # 10-100 ms per call at 128K. Validity is ``sel < row_prefix`` -- the selected
        # document-local pool id is below the row's visible-pool prefix exactly when that
        # pool was visible (the fused kernel stores ``finfo.min``, not ``-inf``, so the old
        # implementation re-gathered the visibility bit instead of comparing scores; the
        # column-id comparison is the same decision without the gather). Pad columns hold
        # ``_SEL_PAD`` and fail the test; the incomplete trailing pool of a document is
        # never below the prefix either, so its out-of-document arithmetic slots are always
        # masked back to -1, matching ``pool_index``'s padding.
        valid = sel_all < row_prefix.unsqueeze(-1)
        sel_all.mul_(index_kpool).add_(row_doc_start.unsqueeze(-1))
        tok = sel_all.unsqueeze(-1) + slot_offsets
        tok = tok.masked_fill_(~valid.unsqueeze(-1), -1)
        out[:, :index_topk] = tok.view(seq_len, index_topk)

    if always_select_tail:
        out[:, index_topk : index_topk + index_kpool - 1] = _visible_tail_tokens(seq_ctx, seq_len, index_kpool, device)

    return out.unsqueeze(1)


def _segment_ranges(
    seq_ctx: SequenceContext, query_len: int, index_kpool: int
) -> list[tuple[int, int, int, int, int]]:
    """Intersect the local query shard with the packed documents, in pool space.

    Args:
        seq_ctx (SequenceContext): Packed-sequence and SP metadata; ``cu_seq_lens_q`` is
            global and ``shard_start`` locates this rank's queries inside it.
        query_len (int): Number of local queries on this rank.
        index_kpool (int): Pool size.

    Returns:
        list[tuple[int, int, int, int, int]]: ``(row_lo, row_hi, pool_lo, pool_hi,
        doc_start)`` per intersecting document -- shard-row range, global-pool-id range, and
        the document's first global token id -- in document order. Documents tile the shard,
        so the row ranges are contiguous and cover ``[0, query_len)``.
    """
    # The host-side cumulative-length list is precomputed once on SequenceContext
    # (``cu_seq_lens_q_list``); reading it here instead of ``cu_seq_lens_q.tolist()``
    # avoids a device->host sync on every indexer call -- at 128K the sync drained the
    # stream behind the freshly enqueued SP-gather + build_pools work (~26 ms per call,
    # 192 calls/step, ~5 s of device idle per step).
    cu = [int(v) for v in seq_ctx.cu_seq_lens_q_list]
    shard_start = int(seq_ctx.shard_start)
    shard_end = shard_start + query_len

    ranges: list[tuple[int, int, int, int, int]] = []
    pool_lo = 0
    for doc_start, doc_end in zip(cu[:-1], cu[1:]):
        num_pools = -(-(doc_end - doc_start) // index_kpool)
        row_lo, row_hi = max(doc_start, shard_start), min(doc_end, shard_end)
        if row_hi > row_lo:
            ranges.append((row_lo - shard_start, row_hi - shard_start, pool_lo, pool_lo + num_pools, doc_start))
        pool_lo += num_pools
    return ranges


def _expand_meta(
    seq_ctx: SequenceContext, query_len: int, index_kpool: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    """Build (or fetch from cache) the batched-expand broadcast tensors.

    Args:
        seq_ctx (SequenceContext): Packed-sequence and SP metadata; ``cu_seq_lens_q_list``
            is the host-side cumulative length list.
        query_len (int): Number of local queries on this rank.
        index_kpool (int): Pool size.
        device (torch.device): Target device.

    Returns:
        tuple[Tensor, Tensor, Tensor]: ``(row_prefix, row_doc_start, slot_offsets)`` -- the
        row's causal visible-pool count ``[query_len]`` int32, the row's document first
        global token id ``[query_len]`` int32, and the in-pool slot offsets ``[kpool]``
        int32. All pure device tensors built from one H2D copy of the cumulative lengths.
    """
    cache = _KPOOL_META_CACHE.get(seq_ctx)
    if cache is None:
        cache = {}
        _KPOOL_META_CACHE[seq_ctx] = cache
    # device joins the key (as in kpool.py's _POOL_INDEX_CACHE): SequenceContext.to() mutates
    # the context in place, so a moved context must not hit entries built on the old device.
    key = (index_kpool, query_len, torch.device(device))
    hit = cache.get(key)
    if hit is not None:
        return hit

    cu_host = [int(v) for v in seq_ctx.cu_seq_lens_q_list]
    cu = torch.tensor(cu_host, dtype=torch.int64, device=device)

    # Row side: the document each sharded query row belongs to and its causal visible-pool
    # count -- the ``[0, (local_pos + 1) // kpool)`` pool slice of the document. Pool ``p``
    # of that document (a document-local column id ``c`` in the slot buffer) covers tokens
    # ``doc_start_d + c * kpool + slot``, so these two row vectors plus the slot offsets are
    # all the expand needs.
    row_tok = torch.arange(query_len, device=device, dtype=torch.int64) + int(seq_ctx.shard_start)
    doc_of_row = torch.searchsorted(cu, row_tok, right=True) - 1
    row_doc_start = cu[doc_of_row].to(torch.int32)
    row_prefix = ((row_tok - cu[doc_of_row] + 1) // index_kpool).to(torch.int32)
    slot_offsets = torch.arange(index_kpool, device=device, dtype=torch.int32)

    meta = (row_prefix, row_doc_start, slot_offsets)
    cache[key] = meta
    return meta


def _score_and_topk_batched(
    q_local: Tensor,
    pool_key: Tensor,
    w_all_f: Tensor,
    ranges: list[tuple[int, int, int, int, int]],
    row_prefix: Tensor,
    select_k: int,
    index_head_dim: int,
) -> Tensor:
    """Score every document segment with one kernel launch each, then topk the shard once.

    Replaces the per-segment pad -> score -> topk -> scatter chain with: one ``finfo.min``
    prefill of a whole-call ``[S, W]`` fp32 score buffer (``W`` = max segment pool count plus
    one tile of slack, so every segment's padded pool axis fits its rows), one whole-call
    visibility compare, one score launch per segment (``lightning_indexer_score_into`` writes
    strided rows), and a single batched ``topk``. Padded pool columns read past a document's
    real keys are zero-keyed and past every row's visible prefix, so they score as invalid;
    the batched ``topk`` may therefore select pad-column ids for rows with fewer visible pools
    than ``select_k``, and the caller's ``sel < row_prefix`` test rejects them exactly like
    the legacy ``_SEL_PAD`` slot padding.

    Args:
        q_local (Tensor): Query rows ``[S, NH, D]``, contiguous.
        pool_key (Tensor): Whole-call pool keys ``[P_total, D]``, contiguous.
        w_all_f (Tensor): Per-head weights ``[S, NH]``, already ``NH**-0.5``-scaled.
        ranges (list[tuple[int, int, int, int, int]]): ``(row_lo, row_hi, pool_lo, pool_hi,
            doc_start)`` per segment, from ``_segment_ranges``.
        row_prefix (Tensor): Per-row visible-pool prefix ``[S]`` int32.
        select_k (int): Slot width (``index_topk // index_kpool``); must not exceed the largest
            segment's pool count so the batched topk width matches the legacy slot buffer.
        index_head_dim (int): Indexer head dimension (scale only).

    Returns:
        Tensor: ``[S, min(select_k, max_pools)]`` int64 document-local pool ids, sorted by
        score descending -- the same content the legacy per-segment ``topk`` produced.
    """
    device = q_local.device
    seq_len = q_local.shape[0]
    if lightning_indexer_score_into is None:  # caller gates on the fused kernel being importable
        raise RuntimeError("batched kpool scoring requires the fused lightning indexer kernel")
    max_pools = max(r[3] - r[2] for r in ranges)
    width = max_pools + _MAX_BLOCK_P
    # Transient memory scales as S_local x (largest-document pools + one tile of slack) for
    # *every* row, not each row's own document size: ~2.2 GB fp32 scores_all + ~0.55 GB bool
    # visible_all per call at 128K/SP=8 (allocator-reused across the 192 calls/step; validated
    # by run145). The legacy per-segment path instead peaks at the largest rows_d x pools_d
    # segment -- budget accordingly when planning 256K / SP-off / heterogeneous packs.
    scores_all = torch.full(
        (seq_len, width), float(torch.finfo(torch.float32).min), dtype=torch.float32, device=device
    )
    # One whole-call visibility compare instead of one per segment: column ``c`` is visible to
    # row ``r`` iff ``c < row_prefix[r]`` -- columns past a document's real pools are past its
    # rows' prefixes too, i.e. invalid (the legacy path padded the mask with zeros instead).
    visible_all = torch.arange(width, device=device).unsqueeze(0) < row_prefix.unsqueeze(-1)
    # One tail pad for the whole-call key buffer: every segment launches with its pool axis
    # padded to whole tiles, and the kernel's pool-axis boundary_check does not reliably clamp
    # the MTE read (see lightning_indexer_score.forward), so the last segment must have real
    # rows to read into. Padded keys are zero; their columns are invalid per the mask above.
    pool_key_pad = torch.nn.functional.pad(pool_key, (0, 0, 0, _MAX_BLOCK_P - 1))
    scale = index_head_dim**-0.5
    for row_lo, row_hi, pool_lo, pool_hi, _doc_start in ranges:
        lightning_indexer_score_into(
            q_local[row_lo:row_hi],
            pool_key_pad[pool_lo:],
            w_all_f[row_lo:row_hi],
            visible_all[row_lo:row_hi],
            scores_all[row_lo:row_hi],
            pool_hi - pool_lo,
            scale,
        )
    return scores_all.topk(min(select_k, max_pools), dim=-1).indices


def _score_pools(
    q_seg: Tensor,
    pool_key_seg: Tensor,
    weights_seg: Tensor,
    index_head_dim: int,
    query_chunk_size: int | None,
    *,
    visible: Tensor,
) -> Tensor:
    """Score one segment's queries against its own pools.

    Args:
        q_seg (Tensor): Query rows ``[R, NH, D]``.
        pool_key_seg (Tensor): The segment's pool keys ``[P_s, D]``.
        weights_seg (Tensor): Per-head weights, already ``NH**-0.5``-scaled, ``[R, NH]``.
        index_head_dim (int): Indexer head dimension (scale only).
        query_chunk_size (int | None): Eager-fallback query-chunk bound.
        visible (Tensor): Causal visibility mask ``[R, P_s]`` bool.

    Returns:
        Tensor: Scores ``[R, P_s]`` float32; invisible pools hold ``finfo.min`` (fused path)
        or ``-inf`` (eager path).
    """
    scale = index_head_dim**-0.5
    if lightning_indexer_score is not None and q_seg.device.type == "npu":
        # Fused path: relu(dot * scale) is accumulated per [BS, BP] tile with the weighted
        # head sum, fp32 throughout, and the visibility mask is fused into the store.
        return lightning_indexer_score(
            q_seg.unsqueeze(0),  # [1, R, NH, D]
            pool_key_seg.unsqueeze(0),  # [1, P_s, D]
            weights_seg.unsqueeze(0),  # [1, R, NH]
            visible.unsqueeze(0),  # [1, R, P_s]
            scale=scale,
        ).squeeze(0)

    # Eager fallback (CPU / no triton-ascend): chunk over the query axis so the transient
    # [chunk, NH, P] fp32 score tensor stays bounded. Per-row math is identical to the
    # reference path's unchunked einsum pair.
    rows, num_pools = q_seg.shape[0], pool_key_seg.shape[0]
    chunk = query_chunk_size
    if chunk is None:
        chunk = max(1, _SCORE_CHUNK_ELEMS // max(1, q_seg.shape[1] * num_pools))
    chunk = max(1, min(int(chunk), rows))

    pool_key_f = pool_key_seg.float()
    index_scores = torch.empty((rows, num_pools), dtype=torch.float32, device=q_seg.device)
    for s0 in range(0, rows, chunk):
        s1 = s0 + chunk
        scores = torch.relu(torch.einsum("shd,pd->shp", q_seg[s0:s1].float(), pool_key_f) * scale)
        index_scores[s0:s1] = torch.einsum("shp,sh->sp", scores, weights_seg[s0:s1])
    return index_scores.masked_fill(~visible, float("-inf"))


# ---------------------------------------------------------------------------
# Fused lightning-indexer scoring (triton): the kernel, autograd wrapper and tile picker
# ported from MindSpeed-MM (see the module docstring for provenance). Forward-only -- the
# DSA indexer runs under torch.no_grad(). Guarded on the triton import at the top of the
# file: without triton-ascend (pure-CPU CI / reference boxes) the ``else`` sentinels keep
# the eager fallback in ``_score_pools``.
if triton is not None:

    @triton.jit
    def _lightning_indexer_score_kernel(
        q_ptr,  # [B, S, NH, D]
        k_ptr,  # [B, >=P, D]
        w_ptr,  # [B, S, NH]
        valid_ptr,  # [B, S, W]   bool / int8, row stride V_STRIDE (W >= P)
        out_ptr,  # [B, S, W]   float32, row stride O_STRIDE (W >= P)
        S,
        P,
        scale,
        neg_inf,
        V_STRIDE,
        O_STRIDE,
        NH: tl.constexpr,
        D: tl.constexpr,
        BS: tl.constexpr,
        BP: tl.constexpr,
        HAS_VALID: tl.constexpr,
    ):
        """Fused DSA (lightning) indexer scoring.

        Computes, without ever materialising the per-head score tensor::

            out[b, s, p] = sum_h w[b, s, h] * relu(scale * sum_d q[b, s, h, d] * k[b, p, d])

        and, when ``HAS_VALID``, replaces entries whose candidate mask is False with
        ``neg_inf`` so the caller can feed the result straight into ``topk``.

        The eager version keeps ``[B, S, NH, P]`` scores alive between the two
        matmuls; here that tensor only ever exists as a ``[BS, BP]`` tile on chip.

        Args:
            q_ptr: Indexer queries ``[B, S, NH, D]``.
            k_ptr: Compressed k-pool keys ``[B, >=P, D]``; only rows ``[0, P)`` are read, so the
                caller may pass a longer buffer (batched scoring pads the whole-call key buffer
                once instead of per segment).
            w_ptr: Per-head weights (already ``NH**-0.5``-scaled) ``[B, S, NH]``.
            valid_ptr: Candidate visibility mask ``[B, S, W]`` (``W >= P``), row stride
                ``V_STRIDE``.
            out_ptr: Output scores ``[B, S, W]`` float32, row stride ``O_STRIDE``.
            S: Sequence length.
            P: Padded pool count (the logical pool-axis extent of this launch).
            scale: Softmax scale (usually ``D**-0.5``).
            neg_inf: ``float32`` finfo.min written to invalid candidates.
            V_STRIDE: Row stride of ``valid_ptr``; ``P`` when the mask is contiguous.
            O_STRIDE: Row stride of ``out_ptr``; ``P`` when the output is contiguous. Strided
                callers must keep the batch axis at 1 (the batch offset below still assumes a
                contiguous ``[B, S, P]`` layout).
            NH: Number of indexer heads.
            D: Head dim.
            BS: Query-axis tile.
            BP: Pool-axis tile.
            HAS_VALID: Whether ``valid_ptr`` carries a mask.
        """
        pid_s = tl.program_id(0)
        pid_p = tl.program_id(1)
        pid_b = tl.program_id(2)

        s0 = pid_s * BS
        p0 = pid_p * BP

        q_batch_off = pid_b * S * NH * D
        w_batch_off = pid_b * S * NH
        k_batch_off = pid_b * P * D
        o_batch_off = pid_b * S * P

        # k tile is reused by every head, so it is loaded once and kept on chip.
        p_k = tl.make_block_ptr(k_ptr + k_batch_off, (P, D), (D, 1), (p0, 0), (BP, D), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kt = tl.trans(b_k)

        acc = tl.zeros((BS, BP), dtype=tl.float32)
        for i_h in range(NH):
            # q[b, s0:s0+BS, i_h, :] -- token stride is NH * D because the head axis
            # sits between the sequence and feature axes.
            p_q = tl.make_block_ptr(q_ptr + q_batch_off + i_h * D, (S, D), (NH * D, 1), (s0, 0), (BS, D), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1))

            # The D-wide reduction has to be fp32: the result feeds a topk, and a
            # bfloat16 accumulator would reorder the selected pools. It already is --
            # `tl.dot` returns fp32 for bfloat16 operands and the cube accumulates in
            # fp32, so this is exact to ~1e-7 relative against an fp32 matmul of the
            # same values (a true bfloat16 accumulation is off by ~2e-2). Upcasting
            # b_q / b_kt to fp32 first changes nothing but costs ~2.5x -- do not.
            b_s = tl.dot(b_q, b_kt)
            b_s = tl.maximum(b_s * scale, 0.0)

            p_w = tl.make_block_ptr(w_ptr + w_batch_off + i_h, (S,), (NH,), (s0,), (BS,), (0,))
            b_w = tl.load(p_w, boundary_check=(0,)).to(tl.float32)

            acc += b_w[:, None] * b_s

        if HAS_VALID:
            p_v = tl.make_block_ptr(valid_ptr + o_batch_off, (S, P), (V_STRIDE, 1), (s0, p0), (BS, BP), (1, 0))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            acc = tl.where(b_v != 0, acc, neg_inf)

        p_o = tl.make_block_ptr(out_ptr + o_batch_off, (S, P), (O_STRIDE, 1), (s0, p0), (BS, BP), (1, 0))
        tl.store(p_o, acc, boundary_check=(0, 1))

    # tl.dot needs at least one cube tile on both the M and N axes.
    _MIN_TILE = 16
    _MAX_BLOCK_S = 64
    # BP = 512 faults the device (npuSynchronizeDevice error), not just a UB overflow.
    _MAX_BLOCK_P = 256
    # Largest BS * BP that still compiles on arch32 (UB = 192 KiB).
    _MAX_TILE_ELEMS = 16384
    # Ascend910_9382 has 48 vector cores; keep them busy on short sequences.
    _MIN_PROGRAMS = 48

    @functools.lru_cache(maxsize=None)
    def _is_arch35() -> bool:
        """Whether the current device is an Ascend 910_95 / 950 (arch 3.5)."""
        try:
            import torch_npu

            name = torch_npu.npu.get_device_name()
            return "Ascend910_95" in name or "Ascend950" in name
        except Exception:  # MindSpeed ports this probe with the same blanket catch
            return False

    @functools.lru_cache(maxsize=4096)
    def _pick_tiles(seq_len: int, num_pools: int, batch_size: int = 1) -> tuple[int, int]:
        """Pick ``(BS, BP)`` for one program.

        Measured on Ascend910_9382 (UB 192 KiB -- the compiler reports the budget as
        ``1572864`` *bits*): ``BS * BP <= 16384`` is the point where the BiShengIR
        pipeline still fits the live tiles on chip; the next step up, (128, 256),
        fails in ConvertLinalgRToBinary. ``BP = 512`` faults the device outright, so
        it is capped at ``_MAX_BLOCK_P``.

        Within that budget a larger BP is always better -- q is re-read once per
        pool block, so widening BP directly cuts global memory traffic. A tile sweep
        at S=5304, P=1326 gave 7.0 / 2.6 / 1.9 / 1.6 / 1.26 ms for
        (64,64) / (64,128) / (32,256) / (128,128) / (64,256), i.e. BP dominates and
        (64, 256) wins. That last one needs the 16384 budget: the earlier 8192 cap
        forced (32, 256) and cost ~1.5x.

        Neither axis goes below ``_MIN_TILE``: they are the M and N axes of
        ``tl.dot``, which degenerates below one cube tile. A ragged sequence tail is
        left to ``boundary_check``; a ragged pool tail is padded away by the caller
        (see the note in ``forward``).

        Args:
            seq_len (int): Query-axis length ``S``.
            num_pools (int): Pool-axis length ``P``.
            batch_size (int): Batch axis length ``B``.

        Returns:
            tuple[int, int]: ``(BS, BP)`` tile shape.
        """
        block_p = _MAX_BLOCK_P
        while block_p > _MIN_TILE and block_p >= num_pools * 2:
            block_p //= 2
        block_s = _MAX_BLOCK_S
        while block_s > _MIN_TILE and block_s >= seq_len * 2:
            block_s //= 2
        while block_s > _MIN_TILE and block_s * block_p > _MAX_TILE_ELEMS:
            block_s //= 2
        # Short sequences otherwise launch fewer programs than there are vector
        # cores and leave most of the device idle.
        n_p = -(-num_pools // block_p)
        while block_s > _MIN_TILE and batch_size * (-(-seq_len // block_s)) * n_p < _MIN_PROGRAMS:
            block_s //= 2
        return block_s, block_p

    class LightningIndexerScoreFunction(torch.autograd.Function):
        """Fused scoring stage of the DeepSeek sparse-attention (DSA) indexer.

        Forward contract (MindSpeed parity)::

            q:                [B, S, NH, D]   indexer queries (after wq_b)
            pool_keys:        [B, P, D]       compressed k-pool keys
            weights:          [B, S, NH]      per-head weights, already NH**-0.5-scaled
            valid_candidates: [B, S, P] bool  optional; masked-out pools become neg_inf
            scale:            float           softmax_scale, usually D**-0.5

            index_scores:     [B, S, P]       float32

        The indexer runs under ``torch.no_grad`` in the model, so no backward is
        provided -- calling backward raises.
        """

        @staticmethod
        def forward(
            ctx,
            q: torch.Tensor,
            pool_keys: torch.Tensor,
            weights: torch.Tensor,
            valid_candidates: torch.Tensor | None = None,
            scale: float | None = None,
        ) -> torch.Tensor:
            """Score every query against every compressed k-pool candidate.

            Args:
                ctx (Any): Autograd context.
                q (torch.Tensor): Indexer queries ``[B, S, NH, D]``.
                pool_keys (torch.Tensor): Compressed k-pool keys ``[B, P, D]``.
                weights (torch.Tensor): Per-head weights ``[B, S, NH]``, already
                    scaled by ``NH**-0.5``.
                valid_candidates (torch.Tensor | None): Candidate visibility mask
                    ``[B, S, P]`` bool; masked-out entries become ``neg_inf``.
                scale (float | None): Softmax scale; defaults to ``D**-0.5``.

            Returns:
                torch.Tensor: Index scores ``[B, S, P]`` float32.
            """
            if q.dim() != 4:
                raise ValueError(f"q must be [B, S, NH, D], got {tuple(q.shape)}")
            if pool_keys.dim() != 3:
                raise ValueError(f"pool_keys must be [B, P, D], got {tuple(pool_keys.shape)}")
            if weights.dim() != 3:
                raise ValueError(f"weights must be [B, S, NH], got {tuple(weights.shape)}")

            batch_size, seq_len, num_heads, head_dim = q.shape
            num_pools = pool_keys.shape[1]
            if pool_keys.shape[0] != batch_size or pool_keys.shape[2] != head_dim:
                raise ValueError(f"pool_keys {tuple(pool_keys.shape)} does not match q {tuple(q.shape)}")
            if weights.shape[:2] != (batch_size, seq_len) or weights.shape[2] != num_heads:
                raise ValueError(f"weights {tuple(weights.shape)} does not match q {tuple(q.shape)}")
            if head_dim & (head_dim - 1):
                raise ValueError(f"head_dim must be a power of two, got {head_dim}")

            if _is_arch35():
                raise NotImplementedError("this op is not supported in this platform")

            if scale is None:
                scale = head_dim**-0.5

            # tl.dot wants both operands in the same dtype. It is enough to make them
            # agree -- the instruction multiplies in the input dtype but accumulates
            # the D-wide reduction in fp32 regardless (see the note in the kernel), so
            # promoting further would only cost bandwidth.
            common_dtype = torch.promote_types(q.dtype, pool_keys.dtype)
            q = q.to(common_dtype).contiguous()
            pool_keys = pool_keys.to(common_dtype).contiguous()
            weights = weights.contiguous()

            has_valid = valid_candidates is not None
            if valid_candidates is not None:
                if tuple(valid_candidates.shape) != (batch_size, seq_len, num_pools):
                    raise ValueError(f"valid_candidates must be [B, S, P], got {tuple(valid_candidates.shape)}")
                valid_candidates = valid_candidates.contiguous()
            else:
                valid_candidates = q  # unused placeholder, kernel guards on HAS_VALID

            if batch_size * seq_len * num_pools == 0:
                return torch.empty((batch_size, seq_len, num_pools), dtype=torch.float32, device=q.device)

            block_s, block_p = _pick_tiles(seq_len, num_pools, batch_size)

            # `boundary_check` does not reliably clamp the MTE read on the pool axis:
            # some ragged tails (e.g. P=642 with BP=256) fault with an aicore MTE
            # error while others of the same shape class do not, i.e. it depends on
            # where the over-read lands. Pad the pool axis instead so every tile is
            # whole. Padded keys are zero, padded candidates invalid, and the extra
            # columns are sliced off before returning.
            pad = (-num_pools) % block_p
            pools_padded = num_pools + pad
            if pad:
                pool_keys = torch.nn.functional.pad(pool_keys, (0, 0, 0, pad))
                if has_valid:
                    valid_candidates = torch.nn.functional.pad(valid_candidates, (0, pad))

            out = torch.empty((batch_size, seq_len, pools_padded), dtype=torch.float32, device=q.device)
            grid = (triton.cdiv(seq_len, block_s), pools_padded // block_p, batch_size)
            _lightning_indexer_score_kernel[grid](
                q,
                pool_keys,
                weights,
                valid_candidates,
                out,
                seq_len,
                pools_padded,
                float(scale),
                float(torch.finfo(torch.float32).min),
                pools_padded,
                pools_padded,
                num_heads,
                head_dim,
                block_s,
                block_p,
                has_valid,
            )
            return out[:, :, :num_pools] if pad else out

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor) -> None:
            raise RuntimeError("lightning_indexer_score is forward-only; the DSA indexer runs under torch.no_grad().")

    def lightning_indexer_score(
        q: torch.Tensor,
        pool_keys: torch.Tensor,
        weights: torch.Tensor,
        valid_candidates: torch.Tensor | None = None,
        scale: float | None = None,
    ) -> torch.Tensor:
        """Functional entry; mirrors MindSpeed-MM's ``lightning_indexer_score``.

        Args:
            q (torch.Tensor): Indexer queries ``[B, S, NH, D]``.
            pool_keys (torch.Tensor): Compressed k-pool keys ``[B, P, D]``.
            weights (torch.Tensor): Per-head weights ``[B, S, NH]``, already scaled
                by ``NH**-0.5``.
            valid_candidates (torch.Tensor | None): Candidate visibility mask
                ``[B, S, P]`` bool; masked-out entries become ``neg_inf``.
            scale (float | None): Softmax scale; defaults to ``D**-0.5``.

        Returns:
            torch.Tensor: Index scores ``[B, S, P]`` float32.
        """
        return LightningIndexerScoreFunction.apply(q, pool_keys, weights, valid_candidates, scale)

    def lightning_indexer_score_into(
        q: torch.Tensor,
        pool_keys: torch.Tensor,
        weights: torch.Tensor,
        valid: torch.Tensor,
        out: torch.Tensor,
        num_pools: int,
        scale: float,
    ) -> None:
        """Launch the fused scoring kernel straight into caller-provided (possibly strided) buffers.

        Batched-path entry for :func:`npu_kpool_topk_indices`: instead of allocating a fresh
        contiguous ``[R, P_padded]`` score tensor per document segment, the kernel writes into a
        whole-call ``[S, W]`` score buffer and reads visibility from a whole-call mask, so the
        per-segment pad / topk / scatter glue collapses to one kernel launch per segment plus
        one batched ``topk`` over the whole buffer. Math is identical to
        :func:`lightning_indexer_score` -- same kernel, same tile picker, only the
        output/visibility addressing differs.

        Args:
            q (torch.Tensor): Query rows ``[R, NH, D]``, contiguous (a row slice of the
                whole-call query), same dtype as ``pool_keys``.
            pool_keys (torch.Tensor): Pool keys ``[P_avail, D]``, contiguous, with ``P_avail >=
                num_pools`` padded up to the tile grid: only rows ``[0, num_pools + pad)`` are
                read, so the caller may pass a longer slice tail (rows past the document's real
                pools are zero-padded by the caller once per call).
            weights (torch.Tensor): Per-head weights ``[R, NH]``, already ``NH**-0.5``-scaled,
                contiguous.
            valid (torch.Tensor): Visibility mask ``[R, W]`` bool whose row stride may exceed
                ``num_pools``; columns past the row's visible prefix must be False.
            out (torch.Tensor): Score buffer ``[R, W]`` float32 sharing ``valid``'s row stride;
                columns ``[num_pools, num_pools + pad)`` are written ``finfo.min`` (invalid), the
                rest of the row keeps whatever the caller prefilled.
            num_pools (int): The segment's real pool count ``P``.
            scale (float): Softmax scale (usually ``D**-0.5``).
        """
        rows, num_heads, head_dim = q.shape
        # Same platform guard as the autograd forward: the tile budget (_MAX_TILE_ELEMS /
        # _MAX_BLOCK_P) is tuned to arch32's 192 KiB UB and is explicitly not supported upstream
        # on Ascend910_95 / Ascend950.
        if _is_arch35():
            raise NotImplementedError("this op is not supported in this platform")
        if rows == 0 or num_pools <= 0:
            return
        block_s, block_p = _pick_tiles(rows, num_pools)
        # Same pool-axis reasoning as the autograd forward above: boundary_check does not reliably
        # clamp the MTE read on the pool axis, so the launch extent is padded to whole tiles and
        # the key buffer must carry real rows that far (the caller's one-shot tail pad).
        pad = (-num_pools) % block_p
        pools_padded = num_pools + pad
        grid = (triton.cdiv(rows, block_s), pools_padded // block_p, 1)
        _lightning_indexer_score_kernel[grid](
            q,
            pool_keys,
            weights,
            valid,
            out,
            rows,
            pools_padded,
            float(scale),
            float(torch.finfo(torch.float32).min),
            valid.stride(0),
            out.stride(0),
            num_heads,
            head_dim,
            block_s,
            block_p,
            True,
        )
else:
    _MAX_BLOCK_P = 256
    lightning_indexer_score = None  # type: ignore[assignment]
    lightning_indexer_score_into = None  # type: ignore[assignment]
