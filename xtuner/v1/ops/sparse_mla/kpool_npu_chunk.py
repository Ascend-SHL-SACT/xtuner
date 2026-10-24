"""Row-block wrapper for the batched kpool indexer scorer (``kpool_npu._score_and_topk_batched``).

The whole-call batched scorer allocates one ``[S_local, W]`` fp32 score buffer plus a
``[S_local, W]`` bool visibility buffer for the *entire* local query shard, where ``W`` is the
largest packed document's pool count (plus one tile). Transient memory therefore scales with
``S_local x W`` even though every row only ever scores its own causal prefix: at 512K/SP=64 a
single call holds ~8 GiB fp32 + ~2 GiB bool, which by itself keeps 512K from running at SP=32.

:func:`score_and_topk_batched_chunked` walks the shard in row blocks and scores each block with
the same fused kernel into its own block-sized buffer, whose pool axis is clamped to that
block's own visible-pool prefix. Columns outside the clamp are causally invisible to every row
of the block and hold the ``finfo.min`` prefill on the whole-call path too (the kernel applies
the visibility mask at the store), so the score multiset per row -- and therefore the selected
content -- is identical to the whole-call path once the caller's ``sel < row_prefix`` validity
test runs.

The one difference the narrower block buffer can produce is *which* of several exactly tied
equal-scoring columns survives the ``topk`` boundary: ``torch.topk`` orders equal values by an
implementation detail that depends on the number of columns, so a tie sitting on the
``select_k``-th rank may keep a different column than the whole-call width did. The kept columns
all carry the same score, the downstream gather sums over them order-insensitively, and a real
pack of continuous bf16 scores has no exact ties, so this is a tie-order, not a value,
difference.

Gated by ``XTUNER_GLM53_KPOOL_SCORE_CHUNK_ROWS`` in ``kpool_npu``: ``0`` (default) keeps the
whole-call buffer, a positive row count enables this path.
"""

from __future__ import annotations

import torch
from torch import Tensor


__all__ = ["score_and_topk_batched_chunked"]

# Pool-axis clamp: align + slack, deliberately larger than the fused kernel's own
# ``_MAX_BLOCK_P`` (256) tile so a clamped block's pool axis never cuts a column the kernel
# would write. Kept as local constants instead of importing ``kpool_npu._MAX_BLOCK_P``, which
# only exists behind that module's triton import guard.
_POOL_AXIS_ALIGN = 512
_POOL_AXIS_SLACK = 512


def score_and_topk_batched_chunked(
    q_local: Tensor,
    pool_key: Tensor,
    w_all_f: Tensor,
    ranges: list[tuple[int, int, int, int, int]],
    row_prefix: Tensor,
    prefix_host: list[int],
    select_k: int,
    index_head_dim: int,
    chunk_rows: int,
) -> Tensor:
    """Score and topk the query shard in row blocks with one block-sized scratch buffer.

    Args:
        q_local (Tensor): Query rows ``[S, NH, D]``, contiguous.
        pool_key (Tensor): Whole-call pool keys ``[P_total, D]``, contiguous.
        w_all_f (Tensor): Per-head weights ``[S, NH]``, already ``NH**-0.5``-scaled.
        ranges (list[tuple[int, int, int, int, int]]): ``(row_lo, row_hi, pool_lo, pool_hi,
            doc_start)`` per document segment, from ``kpool_npu._segment_ranges``.
        row_prefix (Tensor): Per-row visible-pool prefix ``[S]`` int32.
        prefix_host (list[int]): ``row_prefix`` as plain ints (no device readback).
        select_k (int): Slot width (``index_topk // index_kpool``).
        index_head_dim (int): Indexer head dimension (scale only).
        chunk_rows (int): Query rows per block; ``<= 0`` or ``>= S`` uses one whole-call block.

    Returns:
        Tensor: ``[S, select_k]`` int32 document-local pool ids sorted by score descending,
        with ``_SEL_PAD`` wherever a block's clamped pool axis returned fewer columns; after
        the caller's ``sel < row_prefix`` test this carries the same selected scores as
        ``_score_and_topk_batched(...).to(torch.int32)``, identical ids apart from exact ties
        on the ``select_k``-th rank (see the module docstring).
    """
    # Call-time import: ``kpool_npu`` imports this module from inside its scoring branch, so a
    # module-level import here would be circular.
    from .kpool_npu import _MAX_BLOCK_P, _SEL_PAD, _score_and_topk_batched

    score_len = q_local.shape[0]
    if chunk_rows <= 0 or chunk_rows >= score_len:
        return _score_and_topk_batched(
            q_local, pool_key, w_all_f, ranges, row_prefix, prefix_host, select_k, index_head_dim
        ).to(torch.int32)

    # One tail pad for the whole call (the kernel's pool-axis MTE read needs real rows past the
    # last pool): padding inside every block would copy the entire key buffer once per block.
    pool_key_padded = torch.nn.functional.pad(pool_key, (0, 0, 0, _MAX_BLOCK_P - 1))
    sel_all = torch.full((score_len, select_k), _SEL_PAD, dtype=torch.int32, device=q_local.device)
    seg = 0
    for lo in range(0, score_len, chunk_rows):
        hi = min(lo + chunk_rows, score_len)
        # Clamp this block's pool axis to its widest row prefix (a block may straddle a
        # document boundary, where the prefix resets, so take the block maximum rather than
        # the last row's prefix).
        pool_bound = -(-max(prefix_host[lo:hi]) // _POOL_AXIS_ALIGN) * _POOL_AXIS_ALIGN + _POOL_AXIS_SLACK
        # ``ranges`` is sorted and non-overlapping (documents tile the shard), so the segments
        # touching this block are a contiguous slice starting at the first one that has not
        # ended yet: walk the cursor forward instead of rescanning every segment per block.
        while seg < len(ranges) and ranges[seg][1] <= lo:
            seg += 1
        blocks: list[tuple[int, int, int, int, int]] = []
        for j in range(seg, len(ranges)):
            row_lo, row_hi, pool_lo, pool_hi, doc_start = ranges[j]
            if row_lo >= hi:
                break
            blk_lo, blk_hi = max(row_lo, lo), min(row_hi, hi)
            if blk_lo >= blk_hi:
                continue
            blocks.append((blk_lo - lo, blk_hi - lo, pool_lo, min(pool_hi, pool_lo + pool_bound), doc_start))
        if not blocks:
            continue
        sel = _score_and_topk_batched(
            q_local[lo:hi],
            pool_key,
            w_all_f[lo:hi],
            blocks,
            row_prefix[lo:hi],
            prefix_host[lo:hi],
            select_k,
            index_head_dim,
            pool_key_padded,
        ).to(torch.int32)
        # A block whose clamped pool count fell below ``select_k`` returns fewer than
        # ``select_k`` columns; the ``_SEL_PAD`` prefill fills the rest. Rows never lose a
        # real column there: a row's visible columns are at most its own prefix, which the
        # clamp covers, and they outrank every invisible column.
        n = min(sel.shape[1], select_k)
        sel_all[lo:hi, :n] = sel[:, :n]
    return sel_all
