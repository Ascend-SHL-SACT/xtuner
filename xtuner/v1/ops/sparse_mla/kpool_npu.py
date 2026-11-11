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
import math
import os
import weakref

import numpy as np
import torch
from torch import Tensor

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.ops.comm import gather_for_sequence_parallel

from . import kpool
from .kpool import _visible_tail_tokens, build_pool_index, build_pools, kpool_output_width
from .sequence_parallel_indexer import get_indexer_sp_partition, redistribute_indexer_queries


try:
    import triton
    import triton.language as tl
except ImportError:  # triton-ascend unavailable (CPU CI / reference boxes): eager fallback only
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]

__all__ = ["npu_kpool_topk_indices", "pack_visible_tail_enabled", "pack_visible_tail_into"]

# Batched indexer scoring (one score launch per document segment into a whole-call score
# buffer + one whole-shard topk) instead of the per-segment pad -> score -> topk -> scatter
# chain. Setting it to ``0`` restores the legacy per-segment path bit for bit.
_BATCHED_SCORE = os.environ.get("XTUNER_GLM53_KPOOL_BATCHED_SCORE", "1") == "1"

# Causal-extent cropping for the batched scorer. A row only ever reads scores below its own
# visible prefix, and the fused kernel stores ``finfo.min`` -- the buffer prefill value -- at
# invisible cells, so per row chunk everything at or past the chunk's widest prefix is
# skippable: the topk / prefill / visibility extents stop there, and the score kernel
# early-exits programs whose whole row tile sits past its rows' widest prefix (they would
# only compute and store finfo.min). Caller-visible pool ids are unchanged (rows shorter
# than ``k`` still pick implementation-defined pad columns, masked to -1 by the caller).
# The win is dominated by the score-kernel row-tile skip and grows with prefix skew
# (device micro-bench 2026-11-11, 512K-doc shards: e2e x7.9 early rank / x1.7 mid /
# x1.3 late rank; topk crop [1024,131072] 4.70 -> 1.96 ms averaged over a causal chunk
# sweep, though the strided ``[:, :w_topk]`` view pays a 1.04-1.19x stride penalty vs a
# contiguous buffer at equal width). ``0`` restores the full-width rectangle bit for bit.
_CAUSAL_CROP = os.environ.get("XTUNER_KPOOL_CAUSAL_CROP", "1") == "1"

# Span-dense packed index rows (built by ``pack_visible_tail_into`` below): each row dense
# with distinct visible tokens inside its causal read span, -1 only beyond it. The default
# ``0`` keeps the legacy -1-padded layout bit for bit.
_PACK_VISIBLE_TAIL = os.environ.get("XTUNER_GLM53_KPOOL_PACK_VISIBLE_TAIL", "0") == "1"


def _parse_row_chunk(raw: str) -> int:
    """Parse ``XTUNER_KPOOL_SCORE_ROW_CHUNK``; ``<= 0`` selects the whole-shard
    layout.

    Args:
        raw (str): Raw environment value.

    Returns:
        int: Positive row-chunk bound, or ``0`` for the whole-shard layout.
    """
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"XTUNER_KPOOL_SCORE_ROW_CHUNK={raw!r} is not a valid int") from exc
    return value if value > 0 else 0


# Row-chunk bound for the batched scoring path. The whole-call ``[S, W]`` fp32 score buffer,
# the bool visibility buffer and the batched ``topk`` workspace all scale with ``S_local x W``
# (``W`` = the largest document's pool count): a ~512K pack of two ~262K documents at SP=32
# holds a [16384, 65792] buffer pair (~5.0 GiB) plus a ~2.2 GiB ``topk`` workspace in ONE call
# (measured on 910C), which device-OOMs against ~53 GiB of resident state (job
# dyb-q35-xtuner-09, step-1 backward replay, 3.30 GiB free). Chunking the query rows caps
# every per-call transient at ``row_chunk x W`` independent of the shard length, pack length
# and balance redistribution (measured ~0.45 GiB at the ``row_chunk=1024`` default, 16x down
# from the whole-shard layout); per-row
# scores and topk are unchanged, so the caller-visible output is identical -- rows with fewer
# visible pools than ``select_k`` pick pad columns from an implementation-defined set in both
# layouts, and the caller's ``sel < row_prefix`` test masks them to ``-1`` either way.
# Chunking also lifts the fused score kernel's launch-grid cap (triton-ascend rejects
# ``coreDim = row-tiles x pool-tiles > 65535``, which bounds the whole-shard layout at
# ~262K-token documents for a 16K-row segment): a chunk launch carries ``row_chunk / 64`` row
# tiles, extending the ceiling to ~4M-token documents at ``row_chunk=1024``. ``0`` restores
# the whole-shard single-topk layout bit for bit.
#
# The default 1024 also keeps the per-call transient small enough that the step-1 backward
# watermark leaves room for HCCL's lazily grown per-communicator buffer pools at SP=32 x 512K
# (job dyb-q35-xtuner-12): every GiB shaved off the peak widens the free window the clip-time
# collectives allocate from.
_SCORE_ROW_CHUNK = _parse_row_chunk(os.environ.get("XTUNER_KPOOL_SCORE_ROW_CHUNK", "1024"))

# SP work-balancing overrides for the indexer scorer (see ``npu_kpool_topk_indices``). The
# launch scripts export both; an unset ``XTUNER_INDEXER_SP_WORK_RATIO`` selects the
# auto-calibrated fixed work below.
_ENV_BALANCE_SP: bool | None = (
    None if (raw := os.environ.get("XTUNER_INDEXER_BALANCE_SP")) is None else raw.lower() in ("1", "true", "yes", "on")
)
_ENV_WORK_RATIO: float | None = None
if (raw := os.environ.get("XTUNER_INDEXER_SP_WORK_RATIO")) is not None and raw.strip():
    try:
        _ENV_WORK_RATIO = float(raw)
    except ValueError as exc:
        raise ValueError(f"XTUNER_INDEXER_SP_WORK_RATIO={raw!r} is not a valid float") from exc

# Auto-calibrated per-query fixed work for balanced SP scoring, in absolute pool units. The
# batched scorer's per-launch cost scales with rows x chunk-end visible pools, while the
# position-independent per-row side (TopK, expand, P2P, launch amortization) corresponds to
# ~4915 pools; an absolute count stays valid across pack lengths where a fixed ratio would not.
_SP_FIXED_WORK_POOLS = 4915

# Eager-fallback budget: at most 2**28 fp32 score elements per chunk (~1 GiB), matching the
# pre-merge implementation's ``_score_chunk_elems``.
_SCORE_CHUNK_ELEMS = 1 << 28

# Slot-buffer pad sentinel: far above any per-document pool count (the visible prefix is
# bounded by 2**15 for 128K contexts) so it always fails the ``sel < row_prefix`` validity
# test, yet small enough that the arithmetic expand below (``sel * kpool + doc_start``)
# stays inside int32.
_SEL_PAD = 1 << 26

# Batched-expand broadcast tensors per (context, kpool, query length, shard start, device).
# Keyed weakly by the SequenceContext so the entries die with the micro-batch; the build pays
# exactly one blocking H2D copy (the cumulative length list) per context instead of one per
# indexer call -- a per-call H2D drains the deeply queued training stream on every call and
# serialized host enqueueing against device execution (~45 s/step at 128K). The shard start
# joins the key so the balanced-SP scoring shard (a different start per micro-batch, but the
# same across all kpool layers of one micro-batch) reuses the original context's cache entry
# instead of rebuilding -- and re-copying -- the device side 24x per step.
_KPOOL_META_CACHE: weakref.WeakKeyDictionary[
    SequenceContext, dict[tuple[int, int, torch.device, int], tuple[Tensor, Tensor, Tensor, list[int]]]
] = weakref.WeakKeyDictionary()

# ``build_pool_index`` is a pure function of the packed-sequence lengths, so its result is
# reused once per (context, pool size, seq_len, device) instead of rebuilt on every indexer
# call: at 128K the live path rebuilds it 384 times per training step (24 kpool layers x 16
# micro-batches) against only ~32 distinct inputs, and every rebuild is a ~2 ms serial AiCPU
# host-op chain (SearchSorted, IndexPut[N,4], Cumsum, ...). Keyed weakly by the SequenceContext
# so entries die with the micro-batch; ``device`` joins the key because ``SequenceContext.to()``
# mutates the context in place (same contract as ``_KPOOL_META_CACHE`` above).
# ``XTUNER_SPARSE_MLA_POOL_INDEX_CACHE`` (default on, ``0`` off) is the A/B lever. The cached
# int64 tensor is shared read-only: ``build_pools`` only reads it (the mask/clamp/gather chain
# is out-of-place) and hands its callers a fresh ``.to(torch.int32)`` copy, so in-place edits
# of that copy cannot corrupt the cache.
_POOL_INDEX_CACHE: weakref.WeakKeyDictionary[SequenceContext, dict[tuple[int, int, torch.device], Tensor]] = (
    weakref.WeakKeyDictionary()
)


def pack_visible_tail_enabled() -> bool:
    """Whether span-dense packed index rows are enabled.

    Returns:
        bool: True when ``XTUNER_GLM53_KPOOL_PACK_VISIBLE_TAIL=1`` (default off). The
            GLM-5.2 per-token ``npu_dsa_topk_indices`` producer pads -1 inside the
            read span and must not run while this gate is on (it raises instead).
    """
    return _PACK_VISIBLE_TAIL


def pack_visible_tail_into(
    out: Tensor,
    sel_all: Tensor,
    row_prefix: Tensor,
    row_doc_start: Tensor,
    slot_offsets: Tensor,
    tail: Tensor,
    *,
    index_topk: int,
    index_kpool: int,
) -> None:
    """Expand selected pool ids into span-dense packed rows, written in place
    into ``out``.

    The kernel cannot take the reference path's -1-padded layout as is: its backward
    reads an in-span -1 slot as a raw negative index and raises an AICORE exception,
    which is why ``_rewrite_invalid_indices`` rewrites -1 to the query's own global
    position by default. That rewrite makes rows duplicate-laden, and the kernel
    accumulates duplicate slots per-slot (multiset softmax, not dedup): the forward
    deviates from the reference "invalid slot contributes nothing" semantics, and with
    duplicates present the backward dkv is no longer the gradient of the forward
    (cross-row shared positions are corrupted).

    The layout built here is the only form that is both exact and safe: each row is
    DENSE with distinct visible tokens inside the kernel's causal read span
    ``[0, min(qpos + 1, width))`` and holds -1 ONLY BEYOND it. The kernel's forward
    skips out-of-span -1 slots and its backward never reads them (the span rule,
    verified on 910C for forward and backward; MindSpeed-MM's glm5_next packs the
    same layout upstream).

    Row shape (width = index_topk, tail token count ``t = (local_pos + 1) % kpool``):

    - short rows (visible tokens ``p + 1 <= width``): ``[0..p]`` + -1 padding;
    - long rows: the kpool-expanded selected pools truncated to ``width - t`` slots,
      then the ``t``-token document tail ending at the query itself -- exactly full.

    Duplicates are impossible by construction: selected pool ids are all below
    ``row_prefix`` while the tail spans the first ``t`` tokens of pool ``row_prefix``.

    Pure tensor ops only (no SequenceContext, no torch_npu), so the pack is
    unit-testable on CPU.

    Args:
        out (Tensor): ``[S, index_topk]`` int32 pre-filled with -1; the packed rows
            are written in place.
        sel_all (Tensor): ``[S, index_topk // index_kpool]`` int32 per-row selected
            pool ids (document-local columns); short rows may hold out-of-range
            pad values, which this function never reads.
        row_prefix (Tensor): ``[S]`` int32 number of complete visible pools per row.
        row_doc_start (Tensor): ``[S]`` int32 global id of each row's first token.
        slot_offsets (Tensor): ``[index_kpool]`` int32 intra-pool slot offsets.
        tail (Tensor): ``[S, index_kpool - 1]`` int32 visible tail token global ids,
            left-aligned and -1-padded (``_visible_tail_tokens`` output; ``[S, 0]``
            when ``index_kpool == 1``).
        index_topk (int): Semantic token budget; equals the packed row width.
        index_kpool (int): Pool size.
    """
    seq_len = out.shape[0]
    device = out.device
    select_k = index_topk // index_kpool
    assert out.shape[1] == index_topk

    # Visible tail token count t = (local_pos + 1) % kpool, recovered from the -1
    # padding of the tail tensor.
    tail_count = (tail >= 0).sum(-1, dtype=torch.int32)
    # History length in tokens: short rows carry ``prefix * kpool == p + 1 - t``
    # visible tokens; long rows truncate the pool expand to ``index_topk - t`` so
    # the tail fits and the row is exactly full. minimum() picks per row -- no
    # branch needed.
    history_len = torch.minimum(row_prefix * index_kpool, index_topk - tail_count)

    # Short rows are rebuilt as the column arange: topk guarantees the valid pool-id
    # SET equals ``[0, row_prefix)`` but not the order, and the canonical order
    # keeps rows comparable to the reference expand. Slots at or past ``row_prefix``
    # are zeroed -- they sit at/after ``history_len`` and never win the where below.
    # Long rows keep the topk pool ids (all valid, all distinct). The where result
    # is a fresh tensor, so the in-place mul_/add_ are safe.
    long_row = row_prefix >= select_k
    col_ids = torch.arange(select_k, device=device, dtype=torch.int32)
    short_ids = col_ids * (col_ids.unsqueeze(0) < row_prefix.unsqueeze(-1))
    compact_sel = torch.where(long_row.unsqueeze(-1), sel_all, short_ids).to(torch.int32)
    compact_sel.mul_(index_kpool).add_(row_doc_start.unsqueeze(-1))
    hist_tok = (compact_sel.unsqueeze(-1) + slot_offsets).view(seq_len, index_topk)

    # Column classification via ``past_history = pos - history_len``: negative =
    # history segment; ``[0, t)`` = tail segment (``tail[:, :1]`` is the first tail
    # token when ``t > 0`` and a -1 placeholder when ``t == 0``, where ``is_tail``
    # is all-False so nothing lands). Remaining columns keep the pre-filled -1,
    # beyond the causal read span.
    past_history = torch.arange(index_topk, device=device, dtype=torch.int32).unsqueeze(0) - history_len.unsqueeze(-1)
    is_history = past_history < 0
    is_tail = (past_history >= 0) & (past_history < tail_count.unsqueeze(-1))
    # ``out`` doubles as the where "else" input and the output: elementwise, no
    # cross-element dependency; the aliasing is verified on NPU.
    torch.where(is_history, hist_tok, out, out=out)
    if tail.shape[1]:
        torch.where(is_tail, tail[:, :1] + past_history, out, out=out)


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
    balance_sp: bool = False,
    sp_full_pool_work_ratio: float | None = None,
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
        always_select_tail (bool): Whether to append the incomplete trailing pool. Required
            under ``XTUNER_GLM53_KPOOL_PACK_VISIBLE_TAIL=1``.
        alignment (int): Output width alignment of the chosen SparseMLA forward backend
            (``torch_npu`` uses 1 -- the kernel takes any index width).
        query_chunk_size (int | None): Eager-fallback query-chunk bound; ignored by the fused
            kernel, which self-bounds through its tile picker.
        balance_sp (bool): Redistribute query scoring work with neighbor P2P (same
            protocol as the TileLang backends), then restore attention ownership
            before token expansion and tail selection.
        sp_full_pool_work_ratio (float): Fixed per-query selection/launch estimate relative to
            the full pool width. Kept for call-site parity with the TileLang backends; this
            backend derives the fixed work from ``_SP_FIXED_WORK_POOLS`` (capped at half the
            pool width) unless ``XTUNER_INDEXER_SP_WORK_RATIO`` sets an explicit ratio.

    Returns:
        Tensor: ``[S, 1, kpool_output_width(index_topk, index_kpool, alignment)]`` int32;
            with ``XTUNER_GLM53_KPOOL_PACK_VISIBLE_TAIL=1`` the width is ``index_topk`` and each
            row is dense with distinct visible tokens inside its causal read span, holding -1
            only beyond it (see :func:`pack_visible_tail_into`).
    """
    assert index_topk % index_kpool == 0
    pack = pack_visible_tail_enabled()
    if pack and not always_select_tail:
        # Without the visible tail the pack cannot fill the causal read span, and the
        # kernel's backward reads in-span -1 slots as raw negative indices.
        raise ValueError("XTUNER_GLM53_KPOOL_PACK_VISIBLE_TAIL=1 requires always_select_tail=True")
    # SP: queries stay sharded; the key side gathers *before* pooling, because a pool can
    # straddle the shard seam (see build_pools).
    sp_mesh = seq_ctx.sequence_parallel_mesh
    k = gather_for_sequence_parallel(k.squeeze(0), dim=0, sp_mesh=sp_mesh)
    gate_scores = gather_for_sequence_parallel(gate_scores.squeeze(0), dim=0, sp_mesh=sp_mesh)

    pool_key, _, _ = build_pools(k, gate_scores, kpool_ape, seq_ctx, index_kpool=index_kpool)

    q_local = q.squeeze(0).contiguous()  # [S, NH, Di]
    seq_len, device = q_local.shape[0], q_local.device
    # Pack gate: rows shrink to index_topk (the tail moves inside the row), so the
    # separate tail columns and the alignment no longer apply.
    width = index_topk if pack else kpool_output_width(index_topk, index_kpool, alignment)
    out = torch.full((seq_len, width), -1, dtype=torch.int32, device=device)

    select_k = index_topk // index_kpool
    w_all = weights.squeeze(0)
    # Out-of-place scaling (matches kpool.py's reference path): ``.float()`` on an fp32 input
    # is an alias, so an in-place ``mul_`` would compound the NH scaling into every consumer.
    w_all_f = w_all.float() * (q_local.shape[1] ** -0.5)

    if _ENV_BALANCE_SP is not None:
        balance_sp = _ENV_BALANCE_SP
    if _ENV_WORK_RATIO is not None:
        if not math.isfinite(_ENV_WORK_RATIO) or _ENV_WORK_RATIO < 0:
            raise ValueError("XTUNER_INDEXER_SP_WORK_RATIO must be finite and nonnegative")
        per_query_work = round(pool_key.shape[0] * _ENV_WORK_RATIO)
    else:
        # Auto default: the calibrated absolute fixed work, capped at half the pool width
        # (the per-query average visible-pool count). Short packs take the cap and recover
        # the legacy half-width ratio; full-length packs take the calibrated constant.
        per_query_work = min(_SP_FIXED_WORK_POOLS, pool_key.shape[0] // 2)

    # SP work balancing: the causal score rectangle grows with the query's position, so under
    # a uniform shard the last rank scores ~the full pool width while the first scores almost
    # nothing, and the step is bound by the slowest rank. Equalize the estimated scoring work,
    # score under the balanced ownership, then ship the selected pool ids back before the
    # token expand -- attention itself keeps its uniform shard.
    # The scoring shard shares the context's global cu lengths; under balance only the shard
    # start moves (into the redistributed ownership). Passing the start explicitly instead of
    # through a ``seq_ctx.copy()`` keeps the meta/segment builds on the original context's
    # caches -- a copied context would miss the weakref caches and rebuild (with a blocking
    # H2D of the cu lengths) on every one of the 384 calls/step.
    score_shard_start: int | None = None
    score_len = seq_len
    partition = None
    if balance_sp:
        # Same query alignment as the TileLang "original" scorer: identical packed shapes
        # must derive identical partitions on every backend.
        index_n_heads = q_local.shape[1]
        query_alignment = 128 // index_n_heads if 128 % index_n_heads == 0 else 1
        partition = get_indexer_sp_partition(
            seq_ctx,
            index_kpool,
            query_alignment,
            per_query_work,
            query_len=seq_len,
        )
        if partition is not None:
            assert sp_mesh is not None
            q_local, w_all_f = redistribute_indexer_queries(
                (q_local, w_all_f), partition.attention, partition.indexer, sp_mesh
            )
            score_shard_start = partition.indexer[sp_mesh.get_local_rank()]
            score_len = q_local.shape[0]

    ranges = _segment_ranges(seq_ctx, score_len, index_kpool, shard_start=score_shard_start)
    if ranges:
        row_prefix, row_doc_start, slot_offsets, prefix_host = _expand_meta(
            seq_ctx, score_len, index_kpool, device, shard_start=score_shard_start
        )

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
                    q_local,
                    pool_key,
                    w_all_f,
                    ranges,
                    row_prefix,
                    prefix_host,
                    select_k,
                    index_head_dim,
                    row_chunk=_SCORE_ROW_CHUNK,
                ).to(torch.int32)
                # ``topk`` hands back int64 ids; left as int64 they poison the shared expand
                # below -- ``tok`` becomes a 2.1 GB int64 buffer and ``masked_fill_`` (no
                # int64 AICORE kernel on 910C) drops to the AI_CPU at ~40 GB/s, ~56 ms per
                # call, ~20 s per training step at 128K (run16 step-9 profile). The ids and
                # ``_SEL_PAD`` (1<<26) are int32-safe by design; the degenerate/legacy
                # buffers below are already int32, so this keeps every branch on one dtype.
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
            sel_all = torch.full((score_len, select_k), _SEL_PAD, dtype=torch.int32, device=device)
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

        if partition is not None:
            # Ship the selected pool ids back to the attention owner; column ids are
            # ownership-independent, but the expand below must use the attention shard's
            # own row prefix / document starts.
            (sel_all,) = redistribute_indexer_queries((sel_all,), partition.indexer, partition.attention, sp_mesh)
            row_prefix, row_doc_start, slot_offsets, _prefix_host = _expand_meta(seq_ctx, seq_len, index_kpool, device)

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
        if pack:
            # Span-dense packed rows (see :func:`pack_visible_tail_into`): dense with
            # distinct visible tokens inside each row's causal read span, -1 only
            # beyond it. The tail is packed inside the row, so no separate tail
            # columns exist.
            pack_visible_tail_into(
                out,
                sel_all,
                row_prefix,
                row_doc_start,
                slot_offsets,
                _visible_tail_tokens(seq_ctx, seq_len, index_kpool, device),
                index_topk=index_topk,
                index_kpool=index_kpool,
            )
            return out.unsqueeze(1)

        valid = sel_all < row_prefix.unsqueeze(-1)
        sel_all.mul_(index_kpool).add_(row_doc_start.unsqueeze(-1))
        tok = sel_all.unsqueeze(-1) + slot_offsets
        tok = tok.masked_fill_(~valid.unsqueeze(-1), -1)
        out[:, :index_topk] = tok.view(seq_len, index_topk)

    if always_select_tail:
        out[:, index_topk : index_topk + index_kpool - 1] = _visible_tail_tokens(seq_ctx, seq_len, index_kpool, device)

    return out.unsqueeze(1)


def _pool_index_cache_on() -> bool:
    return os.environ.get("XTUNER_SPARSE_MLA_POOL_INDEX_CACHE", "1") == "1"


def _cached_pool_index(seq_ctx: SequenceContext, seq_len: int, index_kpool: int, device: torch.device) -> Tensor:
    """``build_pool_index`` behind the per-context pool-index cache
    (``_POOL_INDEX_CACHE``).

    Args:
        seq_ctx (SequenceContext): Packed-sequence metadata (document boundaries).
        seq_len (int): Whole-sequence token count (pools always cover the whole sequence).
        index_kpool (int): Pool size.
        device (torch.device): Target device for the built tensor.

    Returns:
        Tensor: ``pool_index``, shape ``[P, index_kpool]`` int64 -- the shared read-only
        cached tensor on a hit, a freshly built one otherwise.
    """
    if not _pool_index_cache_on():
        return build_pool_index(seq_ctx, seq_len, index_kpool, device)
    cache = _POOL_INDEX_CACHE.get(seq_ctx)
    if cache is None:
        cache = {}
        _POOL_INDEX_CACHE[seq_ctx] = cache
    key = (index_kpool, seq_len, torch.device(device))
    hit = cache.get(key)
    if hit is not None:
        return hit
    pool_index = build_pool_index(seq_ctx, seq_len, index_kpool, device)
    cache[key] = pool_index
    return pool_index


# Wiring: install the per-context cache over ``kpool.build_pool_index`` (whose signature stays
# untouched) so every caller -- ``build_pools`` (both the reference and the NPU backend) and
# direct calls -- reuses it, matching the pre-merge behavior where the cache lived inside
# ``kpool.build_pool_index`` itself. ``_cached_pool_index`` resolves the plain
# ``build_pool_index`` name above -- the original pure builder captured by the from-import
# before this rebind -- so the wrapper does not recurse into itself.
kpool.build_pool_index = _cached_pool_index


def _segment_ranges(
    seq_ctx: SequenceContext, query_len: int, index_kpool: int, shard_start: int | None = None
) -> list[tuple[int, int, int, int, int]]:
    """Intersect the local query shard with the packed documents, in pool
    space.

    Args:
        seq_ctx (SequenceContext): Packed-sequence and SP metadata; ``cu_seq_lens_q`` is
            global and ``shard_start`` locates this rank's queries inside it.
        query_len (int): Number of local queries on this rank.
        index_kpool (int): Pool size.
        shard_start (int | None): First global token id of the shard; ``None`` uses the
            context's attention shard (``seq_ctx.shard_start``). Balanced-SP scoring passes
            the redistributed start while keeping the context's global cu lengths.

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
    if shard_start is None:
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
    seq_ctx: SequenceContext, query_len: int, index_kpool: int, device: torch.device, shard_start: int | None = None
) -> tuple[Tensor, Tensor, Tensor, list[int]]:
    """Build (or fetch from cache) the batched-expand broadcast tensors.

    Args:
        seq_ctx (SequenceContext): Packed-sequence and SP metadata; ``cu_seq_lens_q_list``
            is the host-side cumulative length list.
        query_len (int): Number of local queries on this rank.
        index_kpool (int): Pool size.
        device (torch.device): Target device.
        shard_start (int | None): First global token id of the shard; ``None`` uses the
            context's attention shard (``seq_ctx.shard_start``). Balanced-SP scoring passes
            the redistributed start, so its per-micro-batch shard reuses this context's
            cached entry across all kpool layers instead of rebuilding per call.

    Returns:
        tuple[Tensor, Tensor, Tensor, list[int]]: ``(row_prefix, row_doc_start, slot_offsets,
        prefix_host)`` -- the row's causal visible-pool count ``[query_len]`` int32, the row's
        document first global token id ``[query_len]`` int32, the in-pool slot offsets
        ``[kpool]`` int32, and ``row_prefix`` as plain host ints. All pure device tensors are
        built from one H2D copy of the cumulative lengths; the host list is derived from the
        same cumulative lengths without a device readback, so per-segment launch bounding in
        :func:`_score_and_topk_batched` never syncs the training stream.
    """
    cache = _KPOOL_META_CACHE.get(seq_ctx)
    if cache is None:
        cache = {}
        _KPOOL_META_CACHE[seq_ctx] = cache
    if shard_start is None:
        shard_start = int(seq_ctx.shard_start)
    # device joins the key (as in _POOL_INDEX_CACHE above): SequenceContext.to() mutates
    # the context in place, so a moved context must not hit entries built on the old device.
    # The shard start joins it so balanced-SP scoring shards share this cache without
    # colliding with the attention shard's entry.
    key = (index_kpool, query_len, torch.device(device), shard_start)
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
    row_tok = torch.arange(query_len, device=device, dtype=torch.int64) + shard_start
    doc_of_row = torch.searchsorted(cu, row_tok, right=True) - 1
    row_doc_start = cu[doc_of_row].to(torch.int32)
    row_prefix = ((row_tok - cu[doc_of_row] + 1) // index_kpool).to(torch.int32)
    slot_offsets = torch.arange(index_kpool, device=device, dtype=torch.int32)

    # Host mirror of ``row_prefix`` (monotone within each document): the same arithmetic as
    # the tensor above, kept as plain ints so the batched scorer can size per-segment launches
    # from the cumulative lengths alone -- a device readback here would park the host behind
    # the stream (see _segment_ranges' note on the ~26 ms stall). Vectorized over numpy: the
    # balanced-SP path builds this for a second (redistributed) context on every call, and a
    # per-row python loop would land that cost directly on the step's host path.
    tok = np.arange(shard_start, shard_start + query_len, dtype=np.int64)
    cu_arr = np.asarray(cu_host, dtype=np.int64)
    doc = np.searchsorted(cu_arr, tok, side="right") - 1
    prefix_host = ((tok - cu_arr[doc] + 1) // index_kpool).tolist()

    meta = (row_prefix, row_doc_start, slot_offsets, prefix_host)
    cache[key] = meta
    return meta


def _score_and_topk_batched(
    q_local: Tensor,
    pool_key: Tensor,
    w_all_f: Tensor,
    ranges: list[tuple[int, int, int, int, int]],
    row_prefix: Tensor,
    prefix_host: list[int],
    select_k: int,
    index_head_dim: int,
    row_chunk: int = 0,
) -> Tensor:
    """Score every document segment with one kernel launch each, then topk the
    shard once.

    Replaces the per-segment pad -> score -> topk -> scatter chain with: one ``finfo.min``
    prefill of a whole-call ``[S, W]`` fp32 score buffer (``W`` = max segment pool count plus
    one tile of slack, so every segment's padded pool axis fits its rows), one whole-call
    visibility compare, one score launch per segment (``lightning_indexer_score_into`` writes
    strided rows), and a single batched ``topk``. Each launch's pool axis is clamped to the
    segment's widest row prefix, rounded up to a whole tile: the kernel computes every
    ``[BS, BP]`` tile it is launched over and applies causal visibility only at the store, so
    an unclamped launch burns full-tile compute on causally-invisible pool columns (a 128K
    single-document pack scores ~9.6 s/step that way, run18). Skipped columns keep their
    ``finfo.min`` prefill, so the ``topk`` sees exactly the scores the unclamped layout
    produced. Padded pool columns read past a document's real keys are zero-keyed and past
    every row's visible prefix, so they score as invalid; the batched ``topk`` may therefore
    select pad-column ids for rows with fewer visible pools than ``select_k``, and the
    caller's ``sel < row_prefix`` test rejects them exactly like the legacy ``_SEL_PAD`` slot
    padding.

    Args:
        q_local (Tensor): Query rows ``[S, NH, D]``, contiguous.
        pool_key (Tensor): Whole-call pool keys ``[P_total, D]``, contiguous.
        w_all_f (Tensor): Per-head weights ``[S, NH]``, already ``NH**-0.5``-scaled.
        ranges (list[tuple[int, int, int, int, int]]): ``(row_lo, row_hi, pool_lo, pool_hi,
            doc_start)`` per segment, from ``_segment_ranges``.
        row_prefix (Tensor): Per-row visible-pool prefix ``[S]`` int32.
        prefix_host (list[int]): ``row_prefix`` as plain ints; bounds each launch's pool axis
            without a device readback.
        select_k (int): Slot width (``index_topk // index_kpool``); must not exceed the largest
            segment's pool count so the batched topk width matches the legacy slot buffer.
        index_head_dim (int): Indexer head dimension (scale only).
        row_chunk (int): Query-row chunk bound; ``0`` (default) keeps the whole-shard layout.
            A positive bound below ``S`` sizes the score/visibility buffers and the ``topk``
            workspace to ``row_chunk`` rows instead of the whole shard, capping the call's
            transients at ~``row_chunk x width`` independent of shard length, pack length and
            balance redistribution. Per-row scores and per-row ``topk`` do not depend on other
            rows, so the returned ids carry the same content the whole-shard layout produces;
            rows with fewer visible pools than ``select_k`` pick pad columns from an
            implementation-defined set in either layout, which the caller's ``sel < row_prefix``
            test masks to ``-1``.

    Returns:
        Tensor: ``[S, min(select_k, max_pools)]`` int64 document-local pool ids, sorted by
        score descending -- the same content the legacy per-segment ``topk`` produced.

    With ``XTUNER_KPOOL_CAUSAL_CROP=1`` (default) the per-chunk prefill / visibility / topk
    extents and the fused kernel's row tiles stop at each chunk's widest visible prefix
    (``=0`` keeps the full-width rectangle bit for bit): rows only read scores below their
    own prefixes and invisible cells hold the ``finfo.min`` prefill either way, so the
    selected pool-id sets are unchanged -- rows with fewer visible pools than ``select_k``
    still pick pad columns from an implementation-defined set, which the caller's
    ``sel < row_prefix`` test masks to ``-1`` exactly as before.
    """
    device = q_local.device
    seq_len = q_local.shape[0]
    if lightning_indexer_score_into is None:  # caller gates on the fused kernel being importable
        raise RuntimeError("batched kpool scoring requires the fused lightning indexer kernel")
    max_pools = max(r[3] - r[2] for r in ranges)
    width = max_pools + _MAX_BLOCK_P
    k = min(select_k, max_pools)
    # One tail pad for the whole-call key buffer: every segment launches with its pool axis
    # padded to whole tiles, and the kernel's pool-axis boundary_check does not reliably clamp
    # the MTE read (see lightning_indexer_score.forward), so the last segment must have real
    # rows to read into. Padded keys are zero; their columns are invalid per the mask above.
    pool_key_pad = torch.nn.functional.pad(pool_key, (0, 0, 0, _MAX_BLOCK_P - 1))
    scale = index_head_dim**-0.5
    if 0 < row_chunk < seq_len:
        # Chunked layout: two [row_chunk, width] buffers plus a topk workspace proportional to
        # row_chunk, allocated once and reused per chunk, so per-call transients stay at
        # ~row_chunk x width regardless of S_local, pack length and balance redistribution
        # (the whole-shard layout below peaks at S_local x width plus its topk workspace). Row
        # content is identical to the whole-shard layout: a row's scores depend only on that
        # row, the chunked launch extent is clamped on the overlap's last row and columns past
        # it are causally invisible to every overlap row (they keep the finfo.min prefill in
        # both layouts), and topk reduces along the last dim independently per row.
        finfo_min = float(torch.finfo(torch.float32).min)
        topk_ids = torch.empty((seq_len, k), dtype=torch.int64, device=device)
        scores_buf = torch.empty((row_chunk, width), dtype=torch.float32, device=device)
        visible_buf = torch.empty((row_chunk, width), dtype=torch.bool, device=device)
        arange_p = torch.arange(width, device=device)
        for c_lo in range(0, seq_len, row_chunk):
            c_hi = min(c_lo + row_chunk, seq_len)
            # Causal crop (see _CAUSAL_CROP): every row's real scores sit below its own
            # visible prefix, so the chunk's widest prefix bounds all of them. row_prefix is
            # monotone inside a document and resets across documents, so the bound is the
            # chunk MAX, not its last row. Extents: ``w_vis`` covers every launch's padded
            # pool axis (each launch's ``w`` is clamped to its own chunk-clamped last-row
            # prefix, so ``pools_padded <= w_vis`` always holds), and ``w_topk = max(w_vis,
            # k)`` keeps the topk legal when the chunk's rows are narrower than ``k`` slots.
            # Cells past these extents are never read by this chunk: the topk stops at
            # ``w_topk``, and both buffers are re-prefilled over ``[:w_topk]`` next chunk.
            if _CAUSAL_CROP:
                w_vis = -(-max(prefix_host[c_lo:c_hi]) // _MAX_BLOCK_P) * _MAX_BLOCK_P
                w_topk = max(w_vis, k)
            else:
                w_vis = width
                w_topk = width
            scores = scores_buf[: c_hi - c_lo]
            scores[:, :w_topk].fill_(finfo_min)
            visible = visible_buf[: c_hi - c_lo]
            torch.lt(arange_p[:w_vis].unsqueeze(0), row_prefix[c_lo:c_hi].unsqueeze(-1), out=visible[:, :w_vis])
            for row_lo, row_hi, pool_lo, pool_hi, _doc_start in ranges:
                r_lo = max(row_lo, c_lo)
                r_hi = min(row_hi, c_hi)
                if r_lo >= r_hi:
                    continue
                w = min(pool_hi - pool_lo, -(-prefix_host[r_hi - 1] // _MAX_BLOCK_P) * _MAX_BLOCK_P)
                if w <= 0:
                    continue
                lightning_indexer_score_into(
                    q_local[r_lo:r_hi],
                    pool_key_pad[pool_lo:],
                    w_all_f[r_lo:r_hi],
                    visible[r_lo - c_lo : r_hi - c_lo, :w],
                    scores[r_lo - c_lo : r_hi - c_lo, :w],
                    w,
                    scale,
                    row_prefix=row_prefix[r_lo:r_hi] if _CAUSAL_CROP else None,
                )
            topk_ids[c_lo:c_hi] = scores[:, :w_topk].topk(k, dim=-1).indices
        return topk_ids
    # Whole-shard layout. The causal-crop bound is the shard-wide MAX visible prefix (a row
    # never reads at or past its own prefix, so no column >= the widest prefix contributes a
    # real score), rounded up to a whole tile for the same launch-extent reason as the
    # per-segment ``w`` below and k-floored to keep the topk legal. Every launch's
    # ``pools_padded`` stays inside it (each ``w <= round256(prefix_host[segment-last]) <=
    # w_vis by the same argument as the chunked layout), and cells past ``w_topk`` are
    # finfo.min in both arms (invisible cells hold exactly that), so cropping the buffer
    # widths is bit-equivalent -- rows shorter than ``k`` still pick implementation-defined
    # pad columns, which the caller's ``sel < row_prefix`` test masks to -1.
    if _CAUSAL_CROP:
        w_vis = -(-max(prefix_host) // _MAX_BLOCK_P) * _MAX_BLOCK_P
        w_topk = max(w_vis, k)
    else:
        w_vis = width
        w_topk = width
    # Transient memory scales as S_local x (largest-document pools + one tile of slack) for
    # *every* row, not each row's own document size: ~2.2 GB fp32 scores_all + ~0.55 GB bool
    # visible_all per call at 128K/SP=8 (allocator-reused across the 192 calls/step; validated
    # by run145). The legacy per-segment path instead peaks at the largest rows_d x pools_d
    # segment -- budget accordingly when planning 256K / SP-off / heterogeneous packs. Under
    # balanced-SP scoring S_local itself grows on the early ranks (128K single document,
    # ratio 0.5, SP=8: rank 0 scores ~27K rows, 1.66x the uniform 16K, i.e. ~3.6 GB fp32 +
    # 0.9 GB bool per call) -- size the headroom from the balanced row counts, not the
    # uniform ones.
    scores_all = torch.full(
        (seq_len, w_topk), float(torch.finfo(torch.float32).min), dtype=torch.float32, device=device
    )
    # One whole-call visibility compare instead of one per segment: column ``c`` is visible to
    # row ``r`` iff ``c < row_prefix[r]`` -- columns past a document's real pools are past its
    # rows' prefixes too, i.e. invalid (the legacy path padded the mask with zeros instead).
    visible_all = torch.arange(w_vis, device=device).unsqueeze(0) < row_prefix.unsqueeze(-1)
    for row_lo, row_hi, pool_lo, pool_hi, _doc_start in ranges:
        # ``row_prefix`` is monotone within a document, so the segment's last row carries the
        # widest visible prefix; rounding it up to a whole tile matches the kernel's
        # tile-granular launch extent (``into`` pads ``num_pools`` the same way).
        w = min(pool_hi - pool_lo, -(-prefix_host[row_hi - 1] // _MAX_BLOCK_P) * _MAX_BLOCK_P)
        if w <= 0:
            continue
        lightning_indexer_score_into(
            q_local[row_lo:row_hi],
            pool_key_pad[pool_lo:],
            w_all_f[row_lo:row_hi],
            visible_all[row_lo:row_hi, :w],
            scores_all[row_lo:row_hi, :w],
            w,
            scale,
            row_prefix=row_prefix[row_lo:row_hi] if _CAUSAL_CROP else None,
        )
    return scores_all.topk(k, dim=-1).indices


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
        rp_ptr,  # [S] int32 per-row visible-pool prefix; unused placeholder when HAS_RP is False
        V_STRIDE,
        O_STRIDE,
        NH: tl.constexpr,
        D: tl.constexpr,
        BS: tl.constexpr,
        BP: tl.constexpr,
        HAS_VALID: tl.constexpr,
        HAS_RP: tl.constexpr,
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
            rp_ptr: Optional per-row visible-pool prefix ``[S]`` int32 -- column ``p`` is
                visible to row ``s`` iff ``p < rp_ptr[s]``. Only read when ``HAS_RP``; any
                tensor is an acceptable placeholder otherwise.
            V_STRIDE: Row stride of ``valid_ptr``; ``P`` when the mask is contiguous.
            O_STRIDE: Row stride of ``out_ptr``; ``P`` when the output is contiguous. Strided
                callers must keep the batch axis at 1 (the batch offset below still assumes a
                contiguous ``[B, S, P]`` layout).
            NH: Number of indexer heads.
            D: Head dim.
            BS: Query-axis tile.
            BP: Pool-axis tile.
            HAS_VALID: Whether ``valid_ptr`` carries a mask.
            HAS_RP: Whether ``rp_ptr`` carries row prefixes (enables the row-tile skip).
        """
        pid_s = tl.program_id(0)
        pid_p = tl.program_id(1)
        pid_b = tl.program_id(2)

        s0 = pid_s * BS
        p0 = pid_p * BP

        if HAS_RP:
            # Causal row-tile skip (see _CAUSAL_CROP): a program whose whole row tile sits at
            # or past the tile's widest visible prefix would only compute tiles that the
            # store then overwrites with ``neg_inf`` -- which is the caller's buffer prefill
            # value -- so skipping both the compute and the store is bit-identical. The
            # bound is the MAX prefix over the tile's rows (prefixes are monotone within a
            # launch -- one document segment -- but the max keeps the skip safe under any
            # ordering); ``other=0`` cannot inflate it because prefixes are non-negative.
            offs_rp = s0 + tl.arange(0, BS)
            b_rp = tl.load(rp_ptr + offs_rp, mask=offs_rp < S, other=0)
            if p0 >= tl.max(b_rp):
                return

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
                valid_candidates,  # unused placeholder, kernel guards on HAS_RP
                pools_padded,
                pools_padded,
                num_heads,
                head_dim,
                block_s,
                block_p,
                has_valid,
                False,
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
        """Functional entry; mirrors MindSpeed-MM's
        ``lightning_indexer_score``.

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
        row_prefix: Tensor | None = None,
    ) -> None:
        """Launch the fused scoring kernel straight into caller-provided
        (possibly strided) buffers.

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
            row_prefix (Tensor | None): Optional per-row visible-pool prefix ``[R]`` int32,
                aligned with ``q``'s rows (a row slice of the whole-call ``row_prefix``).
                When given, the kernel early-exits programs whose whole row tile sits at or
                past the tile's max prefix -- those tiles would only compute and store
                ``finfo.min``, the caller's prefill value, so the skip is bit-identical.
                Rows within one launch must belong to one document segment so prefixes are
                monotone (the batched scorer's per-segment launches qualify); the kernel
                takes the tile MAX anyway, so any ordering stays correct.
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
            q if row_prefix is None else row_prefix,  # placeholder, kernel guards on HAS_RP
            valid.stride(0),
            out.stride(0),
            num_heads,
            head_dim,
            block_s,
            block_p,
            True,
            row_prefix is not None,
        )
else:
    _MAX_BLOCK_P = 256
    lightning_indexer_score = None  # type: ignore[assignment]
    lightning_indexer_score_into = None  # type: ignore[assignment]
