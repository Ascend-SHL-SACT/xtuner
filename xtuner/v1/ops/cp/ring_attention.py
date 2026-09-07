# Copyright (c) OpenMMLab. All rights reserved.
"""Ring-attention context parallelism for absorbed DSA-MLA (GLM-5.2).

ONE implementation -- the genuine ring: each hop's latent KV chunk is
consumed by its own ``npu_sparse_flash_attention`` kernel call
(``sparse_mode=0``, BSND), and the hop for chunk ``j+1`` is POSTED
(``step_async``) before chunk ``j``'s kernel is enqueued, so the P2P transfer
overlaps the per-chunk compute (``work.wait()`` at the top of the next step is
the only completion fence -- a ``torch.npu.Event`` recorded after ``irecv``
fires at QUEUE time, not landing time, and produced nans). A per-query
count-correction subtracts the zero-dummy rows' contribution (score ``0`` ->
weight ``1`` each) from the softmax normaliser; it rides on the fp32 merge
weights (folding the RAW chunk output with ``z_true * rescale`` is
fp32-exactly folding the corrected output with ``z_true``), so the
online-softmax merge keeps a single fp32 accumulator and never materialises an
fp32 chunk temporary.

DSA top-k has strong locality, so of the ``cp_size`` chunks on the ring only
~1-2 per rank hold real keys; the remaining all-dummy steps skip the kernel
entirely (host-decided from the precomputed ``max_counts`` -- one D2H sync per
unique top-k, cached across layers/recompute/backward). The P2P rotation count
is a UNIFORM host bound from the packed segment lengths (:func:`_ring_reach`),
so every ``isend`` always has a matching ``irecv`` and no per-layer collective
is needed.

Backward is Formulation B (MindSpeed's dense-ring trick adapted to sparse):
each chunk's ``npu_sparse_flash_attention_grad`` sees the MERGED corrected
stats, which makes its per-token ``d_s`` the exact gradient of the merged
output; ``dq`` accumulates across chunks, the per-chunk ``d_kv`` is scatter-
added into a global fp32 buffer by global position, and one ``reduce_scatter``
+ local reduce finishes the cross-rank KV grad (HCCL ``all_reduce`` deadlocks
on the CP subgroup under CANN 9.2; ``reduce_scatter`` does not).

The former ``XTUNER_CP_RING_MODE="chimera"`` path (blocking P2P relay that
collected the whole prefix-extended KV slice and ran ONE fused TND kernel --
numerically a KV-allgather variant) has been REMOVED per the ring charter: the
CP line stays a genuine ring, no gather-all implementation exists.

Layout (public entry last): :class:`RingP2P` transport -> shared constants /
:func:`_reduce_scatter_reduce` -> ``forward_true`` / ``backward_true`` (+
bucketed row-compaction machinery) -> :class:`RingAttentionCP`, the single
autograd entry wired from ``dsa_mla`` behind ``XTUNER_CP_RING=1``.

Validated on Ascend 910B3 / CANN 9.2.
"""

from __future__ import annotations

import os
import weakref
from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.distributed as dist

from xtuner.v1.utils.activation_offload import OffloadManager, SwapTensor


if TYPE_CHECKING:
    from xtuner.v1.data_proto.sequence_context import SequenceContext


# GLM-5.2 absorbed-MLA latent dims: kv_lora_rank (Rkv) + qk_rope (Dr).
Rkv = 512
Dr = 64


class RingP2P:
    """One chunk rotation per step around the CP ring.

    Args:
        cp_size: Ring size (context-parallel degree).
        cp_rank: This rank's position in the ring.
        group: Process group backing the ring (an HCCL subgroup). ``None``
            uses the default process group.
    """

    def __init__(self, cp_size: int, cp_rank: int, group: "dist.ProcessGroup | None" = None) -> None:
        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.group = group
        next_local = (cp_rank + 1) % cp_size
        prev_local = (cp_rank - 1) % cp_size
        # ``dist.isend``/``irecv`` take GLOBAL ranks for ``dst``/``src``, but the
        # ring topology is group-local (``0..cp_size-1``). For a real CP subgroup
        # (e.g. ranks ``{12,13,14,15}`` of a 16-rank world) the local rank is not
        # its global rank; passing local-as-global makes ``irecv`` raise
        # "Global rank N is not part of group". Convert here. ``group=None``
        # (default == world) keeps local == global.
        if group is None:
            self.next_rank = next_local
            self.prev_rank = prev_local
        else:
            self.next_rank = dist.get_global_rank(group, next_local)
            self.prev_rank = dist.get_global_rank(group, prev_local)

    def step(self, send_buf: torch.Tensor, recv_buf: torch.Tensor) -> None:
        """Synchronously send ``send_buf`` and fill ``recv_buf`` (blocking).

        Each hop blocks until both send and recv complete (``work.wait``), so
        ``recv_buf`` is valid to read immediately on return.

        Args:
            send_buf: Chunk to send to ``next_rank``.
            recv_buf: Buffer to receive from ``prev_rank`` (overwritten).
        """
        send_buf = send_buf.contiguous()
        recv_buf = recv_buf.contiguous()
        even = self.cp_rank % 2 == 0
        if even:
            send_work = dist.isend(send_buf, dst=self.next_rank, group=self.group)
            recv_work = dist.irecv(recv_buf, src=self.prev_rank, group=self.group)
        else:
            recv_work = dist.irecv(recv_buf, src=self.prev_rank, group=self.group)
            send_work = dist.isend(send_buf, dst=self.next_rank, group=self.group)
        assert send_work is not None and recv_work is not None
        send_work.wait()
        recv_work.wait()

    def step_async(self, send_buf: torch.Tensor, recv_buf: torch.Tensor) -> tuple[dist.Work, dist.Work]:
        """Post the send/recv for one hop without waiting (overlap-enabled).

        Identical parity stagger and buffers as :meth:`step`, but returns the
        two ``dist.Work`` handles immediately. The caller must fence the
        receive with :meth:`wait_all` (``work.wait()``) before reading
        ``recv_buf`` -- that wait is the completion gate; nothing else (no
        ``torch.npu.Event``) reliably observes HCCL P2P landing on this stack.
        Between post and fence, compute on the *current* buffer may run in
        parallel: torch_npu enqueues the P2P op with an event-wait on the
        compute stream at post time, so the hop overlaps work already enqueued
        ahead of it without racing the buffers it sends.

        Args:
            send_buf: Chunk to send to ``next_rank`` (read until send work
                completes; do not mutate before the fence).
            recv_buf: Buffer to receive from ``prev_rank`` (overwritten; do
                not read before the fence).

        Returns:
            ``(send_work, recv_work)`` handles; fence both with
            :meth:`wait_all` before consuming ``recv_buf``.
        """
        send_buf = send_buf.contiguous()
        recv_buf = recv_buf.contiguous()
        even = self.cp_rank % 2 == 0
        if even:
            send_work = dist.isend(send_buf, dst=self.next_rank, group=self.group)
            recv_work = dist.irecv(recv_buf, src=self.prev_rank, group=self.group)
        else:
            recv_work = dist.irecv(recv_buf, src=self.prev_rank, group=self.group)
            send_work = dist.isend(send_buf, dst=self.next_rank, group=self.group)
        assert send_work is not None and recv_work is not None
        return send_work, recv_work

    @staticmethod
    def wait_all(works: "tuple[dist.Work, dist.Work] | None") -> None:
        """Block until all posted works in ``works`` complete.

        Args:
            works: Work handles from :meth:`step_async` (``None`` is a no-op,
                so loop heads can fence unconditionally).
        """
        if works is None:
            return
        for work in works:
            work.wait()


def _reduce_scatter_reduce(
    tensor: torch.Tensor,
    cp_group: "dist.ProcessGroup | None",
) -> torch.Tensor:
    """Reduce-scatter the d_kv partial; each rank gets its own slice summed.

    Unlike :func:`_all_gather_reduce` (which materialises a world-sized
    gathered buffer then sums locally), ``reduce_scatter_tensor`` fuses the
    reduction into the collective and delivers each rank only its own chunk of
    the summed result -- ``[chunk_len, ...]`` vs ``[world*chunk_len, ...]``.
    For the 75 MB d_kv buffer this is a ~9 MB output (8x less HCCL traffic and
    no separate elementwise sum). The eager ``reduce_scatter_tensor`` uses the
    same HCCL subgroup as the (working) ``all_gather``; if it deadlocks like
    ``all_reduce``, fall back to ``all_gather``.

    Args:
        tensor: ``[cp_size * chunk_len, ...]`` this rank's partial d_kv.
        cp_group: HCCL subgroup of the CP ring.

    Returns:
        ``[chunk_len, ...]``: this rank's (cp_rank-th) slice of the summed
        d_kv, i.e. ``dkv_local``.
    """
    world = dist.get_world_size(cp_group)
    chunk_len = tensor.shape[0] // world
    output = torch.empty(chunk_len, *tensor.shape[1:], device=tensor.device, dtype=tensor.dtype)
    dist.reduce_scatter_tensor(output, tensor.contiguous(), group=cp_group)
    return output


# Per-real-chunk backward payload: (ring step j, global row start of the
# chunk, the saved [S, 1, Rkv + Dr] bf16 KV clone). ``j`` re-locates the
# compacted indices in the (cached) ``all_idx`` list at backward time.
_ChunkPayload = tuple[int, int, torch.Tensor]


def forward_true(
    q_states: torch.Tensor,
    kv_local: torch.Tensor,
    topk_indices: torch.Tensor,
    scale: float,
    cp_group: "dist.ProcessGroup | None",
    cp_size: int,
    cp_rank: int,
    seq_ctx: "SequenceContext",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[_ChunkPayload]]:
    """True-ring forward: per-chunk P2P rotation overlapped with per-chunk kernels.

    Ring step ``j`` holds global chunk ``(cp_rank - j) % cp_size`` (``j = 0``
    is the local chunk). The hop to step ``j + 1`` is POSTED before chunk
    ``j``'s kernel is enqueued, so the transfer overlaps the compute; the
    landing is fenced by ``work.wait()`` at the top of step ``j + 1``
    (:meth:`RingP2P.step_async`). Each real chunk runs its own
    ``sparse_mode=0`` BSND kernel with the compacted indices and the zero-row
    dummy band, is count-corrected, and merged into running online-softmax
    statistics. All-dummy steps skip the kernel (host-decided from the cached
    ``max_counts``) but still rotate, so the hop count stays uniform across
    ranks (:func:`_ring_reach` -- pure host, no collective).

    Real-chunk KV is cloned into the returned payload so the backward
    iterates chunks locally (no P2P re-rotation). Under reentrant
    activation checkpointing the whole pass would normally be REPLAYED inside
    the backward (P2P + kernels + fold again) -- the replay is what pushes the
    backward window over this box's pool ceiling (run281/284/285: reserved
    55.4-55.5 against a ~55.0 plateau ceiling, dead on the 2.5 GiB aclnn
    workspace). :func:`_stash_put` / :func:`_stash_take` eliminate the replay:
    AC pass 1 (running in the forward, outside the autograd engine) stashes
    the fold RESULT (output + stats + KV payload) asynchronously to pinned
    CPU, and the replay (running INSIDE the engine, identified by
    :func:`_in_backward_graph`) restores it and returns immediately -- no P2P,
    no kernels, no fold in the backward.

    Args:
        q_states: ``[S, N, Rkv + Dr]`` local query (absorbed q_nope ++ q_pe).
        kv_local: ``[S, 1, Rkv + Dr]`` local latent KV (rings around the group).
        topk_indices: ``[S, 1, K]`` global indices (``-1`` invalid), int32.
        scale: Softmax scale.
        cp_group: HCCL subgroup of the CP ring.
        cp_size: Context-parallel degree.
        cp_rank: This rank's position in the CP ring.
        seq_ctx: SequenceContext (packed ``cu_seq_lens_q_list`` + shard start).

    Returns:
        ``(out, smax, ssum, payload)``: merged corrected output ``[S, N, Rkv]``
        in the MODEL dtype (the fold accumulates in FLOAT32 -- what the CP4
        oracle requires, see :func:`_merge_stats` -- and the final cast reuses
        the last chunk's kernel output buffer), merged ``[S, N]`` float32
        softmax_max / dummy-excluded softmax_sum, and the per-real-chunk
        backward payload.
    """
    seq_len = q_states.shape[0]
    # Recompute-skip: in the reentrant-AC replay (running INSIDE the autograd
    # engine), a stash entry for THIS top-k tensor means AC pass 1 already
    # computed everything -- restore it and return without touching P2P /
    # kernels / fold (the replay is what overflowed the backward window at 256K).
    if _recompute_skip_enabled() and _in_backward_graph():
        stashed = _stash_take(topk_indices)
        if stashed is not None:
            return stashed
    if _recompute_skip_enabled():
        # Free the PREVIOUS layer's stashed GPU storage BEFORE this layer
        # allocates anything: a stashed ``out_final`` (1.07 GiB at 256K) whose
        # D2H landed a whole layer ago must not stay alive into this layer's
        # fold window -- that overlap (prev out + 2 raws + tile scratch) was
        # ~+1 GiB on the forward high-water and kept run286/287 one workspace
        # allocation under the ceiling. The deferred-in-``_stash_put`` flush
        # fires too late (at this layer's END); its D2H wait is a no-op here
        # exactly as there (a full layer of stream time has elapsed).
        _flush_fold_pending()
    shard_start = int(getattr(seq_ctx, "_shard_start", 0))
    # The ring maps chunk ``c`` to step ``(cp_rank - c) % cp_size``, i.e. rank
    # r's queries are the global block ``[r*S, (r+1)*S)``. True only under
    # uniform CP sharding; assert instead of silently mis-attending.
    assert shard_start == cp_rank * seq_len, (
        f"true ring requires uniform CP shards: shard_start={shard_start}, cp_rank={cp_rank}, S={seq_len}"
    )
    q_nope = q_states[..., :Rkv]
    q_rope = q_states[..., Rkv:]
    overlap = _overlap_enabled()
    all_idx = _remap_all_chunks(topk_indices, seq_len, cp_size, cp_rank)
    reach = _ring_reach(seq_ctx, seq_len, cp_size)
    # Sufficiency check of the host bound (free -- the widths are cached host
    # ints): no rank may hold real top-k slots past the rotation bound, else
    # attention silently drops keys.
    beyond = [j for j in range(reach + 1, cp_size) if all_idx[j][2] != 0]
    assert not beyond, (
        f"real top-k beyond host reach bound at steps {beyond} (reach={reach}, "
        f"S={seq_len}, cp_size={cp_size}): segment-length bound violated"
    )
    ring = RingP2P(cp_size, cp_rank, cp_group)
    # CLONE the local chunk: the ping-pong swap makes the send buffer a later
    # recv target (recv writes in place), so an alias of kv_local would let a
    # hop silently mutate the caller's key_states. The ring owns its buffers.
    cur = kv_local.contiguous().clone()
    buf = torch.empty_like(cur)
    works: "tuple[dist.Work, dist.Work] | None" = None
    merged_out: torch.Tensor | None = None
    merged_smax: torch.Tensor | None = None
    merged_ssum: torch.Tensor | None = None
    last_raw: torch.Tensor | None = None  # most recent chunk's kernel output
    # buffer -- dead right after the LAST fold, so it absorbs the final
    # fp32->bf16 cast (see below) instead of allocating a fresh tensor.
    # ``reach == 1`` (the 256K/143K norm) defers the fold: both raw chunk
    # outputs are kept as bf16 and merged tile-by-tile, so the 2 GiB fp32
    # accumulator never exists -- the exact-fold transient that put true ring
    # past this box's pool ceiling (see :func:`_fold_two_raws`).
    # ``XTUNER_CP_DEFER_FOLD=0`` forces the online accumulator (bit-identical
    # results; only the memory/launch shape changes).
    defer_fold = reach == 1 and _defer_fold_enabled()
    pend_raw: torch.Tensor | None = None
    pend_res: torch.Tensor | None = None
    pend_smax: torch.Tensor | None = None
    pend_ssum: torch.Tensor | None = None
    # Bucketed row-compaction state (see :class:`_ChunkPlan`). Pass 2 attaches
    # a plan to AT MOST ONE ring step and never to step 0 (see
    # :func:`_remap_all_chunks`), so under ``defer_fold`` the planned chunk is
    # the last real one and its fold is a pure after-loop step: the deferred
    # pair stays raw-in-place and the peak never exceeds the non-bucketed
    # (which it already better: the second raw is ``[m]`` rows, not ``[S]``).
    sub_raw: torch.Tensor | None = None
    sub_res: torch.Tensor | None = None
    sub_smax: torch.Tensor | None = None
    sub_ssum: torch.Tensor | None = None
    sub_rows: torch.Tensor | None = None
    payload: list[_ChunkPayload] = []
    for j in range(reach + 1):
        if j > 0:
            RingP2P.wait_all(works)  # hop j landed: buf (now cur) holds chunk (cp_rank - j) % cp_size
            works = None
        if j < reach:
            works = ring.step_async(cur, buf)
            if not overlap:
                RingP2P.wait_all(works)  # bring-up fallback: blocking hop
                works = None
        local_idx, ndummy, k, plan = all_idx[j]
        if k > 0:
            assert local_idx is not None and ndummy is not None  # k > 0 <=> pass 2 built it
            use_plan = plan is not None and defer_fold
            if use_plan:
                assert sub_rows is None  # at most one planned step (Pass 2 invariant)
                rows = plan.rows  # type: ignore[union-attr]  # use_plan <=> plan is not None
                # Active-row compaction: the kernel runs on the count>0
                # rows only -- the dropped rows' exact contribution is a
                # zero output and a 1e-30 weight (zero-KV dummies), so the
                # fold math is unchanged (see _bucket_enabled block comment).
                out_j, smax_j, ssum_j, rescale_j = _fused_chunk_attention(
                    q_nope.index_select(0, rows),
                    q_rope.index_select(0, rows),
                    cur,
                    plan.idx,  # type: ignore[union-attr]
                    plan.ndummy,  # type: ignore[union-attr]
                    scale,
                    qlen=plan.qlens[0:1],  # type: ignore[union-attr]
                )
                sub_raw, sub_res, sub_smax, sub_ssum, sub_rows = out_j, rescale_j, smax_j, ssum_j, rows
                payload.append((j, ((cp_rank - j) % cp_size) * seq_len, cur.clone()))
                if j < reach:
                    cur, buf = buf, cur
                continue
            out_j, smax_j, ssum_j, rescale_j = _fused_chunk_attention(q_nope, q_rope, cur, local_idx, ndummy, scale)
            if defer_fold:
                if pend_raw is None:
                    pend_raw, pend_res, pend_smax, pend_ssum = out_j, rescale_j, smax_j, ssum_j
                else:
                    assert merged_out is None  # reach == 1: at most two real chunks
                    assert pend_res is not None and pend_smax is not None and pend_ssum is not None
                    merged_out, merged_smax, merged_ssum = _fold_two_raws(
                        pend_raw,
                        pend_res,
                        pend_smax,
                        pend_ssum,
                        out_j,
                        rescale_j,
                        smax_j,
                        ssum_j,
                    )
                    pend_raw = None  # merged out ALIASES pend_raw's buffer
                    del out_j  # raw1 is dead once merged; do not hold its 1 GiB to stash/return
            else:
                last_raw = out_j  # rebinding frees the PREVIOUS raw (already folded)
                if merged_out is None:
                    # fp32 accumulator; the mixed bf16 x fp32 mul promotes bit-exactly
                    # (bf16 -> fp32 is lossless), so this equals the old
                    # ``out_j.float().mul_(rescale)`` sans cast copy.
                    merged_out = torch.mul(out_j, rescale_j.unsqueeze(-1))
                    merged_smax, merged_ssum = smax_j, ssum_j
                else:
                    assert merged_smax is not None and merged_ssum is not None
                    merged_out, merged_smax, merged_ssum = _merge_stats(
                        merged_out, merged_smax, merged_ssum, out_j, smax_j, ssum_j, rescale_j
                    )
            payload.append((j, ((cp_rank - j) % cp_size) * seq_len, cur.clone()))
        if j < reach:
            cur, buf = buf, cur
    out_final: torch.Tensor
    if defer_fold:
        if merged_out is None:
            if sub_raw is not None:
                assert sub_res is not None and sub_smax is not None and sub_ssum is not None
                assert sub_rows is not None
                if pend_raw is not None:
                    # Planned last chunk + full-width pend chunk 0: the subset
                    # twin of the pair fold (bit-identical on covered rows).
                    assert pend_res is not None and pend_smax is not None and pend_ssum is not None
                    merged_out, merged_smax, merged_ssum = _fold_pair_subset(
                        pend_raw,
                        pend_res,
                        pend_smax,
                        pend_ssum,
                        sub_raw,
                        sub_res,
                        sub_smax,
                        sub_ssum,
                        sub_rows,
                    )
                    pend_raw = None  # merged out ALIASES pend_raw's buffer
                else:
                    # Only the planned chunk was real (rank-0-style steps with
                    # a local... none: Pass 2 never plans step 0 -- reachable
                    # when step 0 is all-dummy and step 1 is planned).
                    merged_out, merged_smax, merged_ssum = _fold_subset_single(
                        sub_raw, sub_res, sub_smax, sub_ssum, sub_rows, seq_len
                    )
            elif pend_raw is not None:
                # Exactly one real chunk (common: rank 0 sees only its local): the
                # deferred fold is the bare rescale, in place, tile by tile.
                assert pend_res is not None and pend_smax is not None and pend_ssum is not None
                merged_out = _fold_one_raw(pend_raw, pend_res)
                merged_smax, merged_ssum = pend_smax, pend_ssum
            else:
                assert False, "k == 0 at every ring step"
        assert merged_out is not None and merged_smax is not None and merged_ssum is not None
        out_final = merged_out
    else:
        assert merged_out is not None and merged_smax is not None and merged_ssum is not None
        # Cast the fp32 accumulator back to the model dtype by OVERWRITING the
        # last chunk's kernel output buffer (dead once the final fold consumed
        # it) instead of allocating a fresh bf16 tensor: the new tensor would
        # stack on the live accumulator and lift the forward peak from 3.15 to
        # 4.2 GiB at 256K (the delta that pushed run280's step-2 backward
        # re-unshard over the budget). ``copy_`` rounds identically to
        # ``.to()`` (single fp32 -> bf16 cast of the same values).
        assert last_raw is not None and last_raw is not merged_out  # mul/fold never alias
        last_raw.copy_(merged_out)
        out_final = last_raw
    if _recompute_skip_enabled() and not _in_backward_graph():
        # Outside the autograd engine = reentrant-AC pass 1: stash the result
        # so the replay (:func:`_stash_take`) can skip the ring.
        _stash_put(topk_indices, out_final, merged_smax, merged_ssum, payload)
    return out_final, merged_smax, merged_ssum, payload


def backward_true(
    q_states: torch.Tensor,
    topk_indices: torch.Tensor,
    out_merged: torch.Tensor,
    smax_merged: torch.Tensor,
    ssum_merged: torch.Tensor,
    grad_out: torch.Tensor,
    payload: list[_ChunkPayload],
    scale: float,
    cp_group: "dist.ProcessGroup | None",
    cp_size: int,
    cp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Formulation-B backward: per-chunk grad calls with MERGED stats, row-tiled.

    Each real chunk's ``npu_sparse_flash_attention_grad`` (``sparse_mode=0``,
    BSND) is fed the MERGED corrected ``(smax, ssum)`` and merged corrected
    output, so its ``d_s = exp(s - smax_m) / z_merged * (g.v - <g, out_m>)`` is
    the EXACT gradient of the merged output w.r.t. this chunk's scores (the
    MindSpeed dense-ring trick adapted to sparse -- validated single- and
    multi-chunk by ``tests/ops/test_cp_formulation_b.py``). The query rows are
    split into :func:`_grad_tiles` tiles (exact per row) to keep the grad-op
    workspace inside the pool. ``dq`` accumulates over chunks (q is shared),
    each chunk's ``d_kv`` scatter-adds into a global
    fp32 buffer by row start, and one ``reduce_scatter`` finishes the cross-
    rank sum (no P2P re-rotation: the forward saved the real chunks).

    Args:
        q_states: ``[S, N, Rkv + Dr]`` local query (forward input).
        topk_indices: ``[S, 1, K]`` global indices (``-1`` invalid) --
            re-locates the compacted indices via the (cached) remap.
        out_merged: ``[S, N, Rkv]`` merged corrected output in the model dtype
            (the dispatcher saves the bf16 return tensor -- the grad kernel
            consumes out in model dtype; ``.to(dtype)`` below is then a view).
        smax_merged: ``[S, N]`` float32 merged softmax_max.
        ssum_merged: ``[S, N]`` float32 merged dummy-excluded softmax_sum.
        grad_out: ``[S, N, Rkv]`` incoming gradient.
        payload: Per-real-chunk ``(j, global_start, kv)`` from the forward.
        scale: Softmax scale.
        cp_group: HCCL subgroup of the CP ring.
        cp_size: Context-parallel degree.
        cp_rank: This rank's position in the CP ring.

    Returns:
        ``(dq, dkv_local)``: ``[S, N, Rkv + Dr]`` query grad and
        ``[S, 1, Rkv + Dr]`` this rank's latent-KV grad, native dtype.
    """
    import torch_npu  # type: ignore[import-untyped]  # local: only the fused-kernel path needs the NPU runtime

    seq_len = q_states.shape[0]
    dtype = q_states.dtype
    device = q_states.device
    q_nope = q_states[..., :Rkv].unsqueeze(0).contiguous()
    q_rope = q_states[..., Rkv:].unsqueeze(0).contiguous()
    all_idx = _remap_all_chunks(topk_indices, seq_len, cp_size, cp_rank)
    _qlen, _kvlen = _seq_tensors(seq_len, device)
    # A row-compacted chunk skips the count-0 rows: their dq contribution is
    # EXACTLY zero (all their index slots point at zero-KV rows), so the
    # skipped writes stay correct on a zero-initialised accumulator instead of
    # the empty-plus-first-chunk-copy of the full-width path (~2 ms memset).
    has_plan = any(entry[3] is not None for entry in all_idx)
    # Grad-op stats in kernel-native [B, kv_heads, S, N] = [1, 1, S, N] fp32
    # (same layout the forward read out of the fwd kernel).
    smax_native = smax_merged.unsqueeze(0).unsqueeze(0).float().contiguous()
    ssum_native = ssum_merged.unsqueeze(0).unsqueeze(0).float().contiguous()
    out_bf = out_merged.unsqueeze(0).to(dtype).contiguous()  # [1, S, N, Rkv]
    grad_out_bf = grad_out.unsqueeze(0).contiguous().to(dtype)
    dq_init = torch.zeros if has_plan else torch.empty
    dq_nope = dq_init(seq_len, q_nope.shape[2], Rkv, device=device, dtype=dtype)
    dq_rope = dq_init(seq_len, q_nope.shape[2], Dr, device=device, dtype=dtype)
    grad_kv_full = torch.zeros(cp_size * seq_len, Rkv + Dr, device=device, dtype=torch.float32)
    assert payload, "true-ring backward requires at least one real chunk"  # was: dq-buffer None guard
    n_tile = _grad_tiles()
    for c, (j, gstart, chunk_kv) in enumerate(payload):
        local_idx, _ndummy, k, plan = all_idx[j]
        assert k > 0 and local_idx is not None  # payload only carries real chunks
        kv_pad, krope_pad = _grad_op_inputs(chunk_kv, device, dtype)
        kvp = kv_pad.unsqueeze(0).contiguous()  # [1, S + Z, 1, Rkv] -- key AND value (MLA)
        krp = krope_pad.unsqueeze(0).contiguous()
        li = local_idx.unsqueeze(0)  # [1, S, 1, k]
        first_chunk = c == 0
        if plan is not None:
            # Row-compacted grad: one kernel per (tile of the) active rows.
            # Per-row independence (merged stats fed in; KV side read-only)
            # makes the subset split EXACT; the count-0 rows' dq is exactly 0
            # (zero-KV indices) and their dkv lands only in the sliced-off
            # dummy band, so skipping them is mathematically lossless.
            for t in range(plan.tiles):
                t0 = t * plan.m // plan.tiles
                t1 = (t + 1) * plan.m // plan.tiles
                if t1 <= t0:  # m < tiles: degenerate tile, rows covered elsewhere
                    continue
                rs = plan.rows[t0:t1]
                dq, dk, dv, dqr, dkr = torch_npu.npu_sparse_flash_attention_grad(
                    q_nope.index_select(1, rs),
                    kvp,
                    kvp,
                    # [m, 1, k]: rows are dim 0 (rs = plan.rows[t0:t1]), NO leading
                    # batch dim before the unsqueeze (slicing dim 1 here silently
                    # clipped the size-1 head dim instead of the rows).
                    plan.idx[t0:t1].unsqueeze(0).contiguous(),
                    grad_out_bf.index_select(1, rs),
                    out_bf.index_select(1, rs),
                    smax_native.index_select(2, rs),  # [1, 1, S, N]: the S dim is 2 (rs are global ROW ids)
                    ssum_native.index_select(2, rs),
                    scale,
                    1,  # sparse_block_size
                    query_rope=q_rope.index_select(1, rs),
                    key_rope=krp,
                    actual_seq_qlen=plan.qlens[1 + t : 2 + t],
                    actual_seq_kvlen=_kvlen,
                    layout="BSND",
                    sparse_mode=0,
                    attention_mode=2,
                )
                dkv = torch.cat([dk + dv, dkr], dim=-1)  # [1, S + Z, 1, Rkv + Dr]
                grad_kv_full[gstart : gstart + seq_len].add_(dkv.squeeze(0).squeeze(1)[:seq_len].float())
                dq_nope.index_add_(0, rs, dq.squeeze(0))
                dq_rope.index_add_(0, rs, dqr.squeeze(0))
            continue
        # Row-tile the grad kernel (:func:`_grad_tiles`): per-row independence
        # (merged stats fed in, KV side read-only) makes the split EXACT, while
        # the workspace request drops to ``seq_len / n_tile`` rows' worth.
        for t in range(n_tile):
            t0 = t * seq_len // n_tile
            t1 = (t + 1) * seq_len // n_tile
            if t1 <= t0:  # seq_len < n_tile: degenerate tile, rows covered elsewhere
                continue
            full_tile = t0 == 0 and t1 == seq_len
            _q_t = (q_nope, q_rope) if full_tile else (q_nope[:, t0:t1], q_rope[:, t0:t1])
            _g_t = grad_out_bf if full_tile else grad_out_bf[:, t0:t1]
            _o_t = out_bf if full_tile else out_bf[:, t0:t1]
            _sm_t = smax_native if full_tile else smax_native[:, :, t0:t1]
            _ss_t = ssum_native if full_tile else ssum_native[:, :, t0:t1]
            _li_t = li if full_tile else li[:, t0:t1]
            _ql_t, _ = _seq_tensors(t1 - t0, device)
            dq, dk, dv, dqr, dkr = torch_npu.npu_sparse_flash_attention_grad(
                _q_t[0],
                kvp,
                kvp,
                _li_t.contiguous(),
                _g_t,
                _o_t,
                _sm_t,
                _ss_t,
                scale,
                1,  # sparse_block_size
                query_rope=_q_t[1],
                key_rope=krp,
                actual_seq_qlen=_ql_t,
                actual_seq_kvlen=_kvlen,
                layout="BSND",
                sparse_mode=0,
                attention_mode=2,
            )
            # MLA: kv_pad serves as BOTH key and value -> nope grad = dk + dv,
            # rope grad = dkr. The dummy zero rows [S:] receive gradient (their
            # p = exp(-smax_m)/ssum_true != 0) but are sliced off -- they are not
            # real tokens. The KV side is tiled-invariant, so every tile's dkv
            # contributes to the SAME full-chunk rows and all are added.
            dkv = torch.cat([dk + dv, dkr], dim=-1)  # [1, S + Z, 1, Rkv + Dr]
            grad_kv_full[gstart : gstart + seq_len].add_(dkv.squeeze(0).squeeze(1)[:seq_len].float())
            # dq accumulates in the MODEL dtype (bf16): the returned dq is cast
            # to bf16 anyway, and a fp32 [S, N, Rkv] buffer costs ~2.1 GiB each
            # at 256K -- two live fp32 chunk accumulators blow the ring's
            # backward workspace budget (run286/287/288 pool-forensics scale).
            # The sum of two bf16 terms adds ~1 ulp (~1.6% rel) vs an fp32 sum,
            # far inside the 8e-2 oracle tolerance; formulation-B tests pin this.
            _dst = (dq_nope, dq_rope)
            if first_chunk:
                _dst[0][t0:t1].copy_(dq.squeeze(0))
                _dst[1][t0:t1].copy_(dqr.squeeze(0))
            else:
                _dst[0][t0:t1].add_(dq.squeeze(0))
                _dst[1][t0:t1].add_(dqr.squeeze(0))
    dq_full = torch.cat([dq_nope, dq_rope], dim=-1)  # [S, N, Rkv + Dr] bf16
    if cp_size > 1:
        dkv_local = _reduce_scatter_reduce(grad_kv_full, cp_group)
    else:
        dkv_local = grad_kv_full[cp_rank * seq_len : (cp_rank + 1) * seq_len]
    dkv_local = dkv_local.unsqueeze(1).to(dtype)  # [S, 1, Rkv + Dr]
    return dq_full, dkv_local


# ---------------------------------------------------------------------------
# Count-bucketed row compaction (Lever A, run306-validated).
#
# Mechanism (profile-established, run303/304/306): the per-chunk BSND
# ``sparse_mode=0`` kernel pays a fixed seed (fwd ~4.4 ms, grad ~9.2 ms per
# tile) PLUS ~1.63 ns per (row x column) slot, and the width k is the MAX
# count over rows (amax padding). A neighbor chunk at 256K carries real keys on
# only ~9-470 of 16384 rows (measured ``n_active``) yet still runs full-
# S x k dummy tails on 99.5% of its rows. Splitting into multiple narrower
# buckets is a NET LOSS under the measured seed/rate (the count distribution
# inside a chunk is too flat -- the extra 4.4/9.2 ms seeds outweigh the width
# saved), so the shipped plan is the degenerate one-bucket form: run the chunk
# kernel over ONLY the active rows (count > 0 under the per-chunk descending
# row order), which removes the width term without adding a kernel and lets
# the grad op drop from ``n_tile`` tiles to one when the row count collapses.
# Excluded rows are mathematically dead: a count-0 row attends only to zero-KV
# dummies -> out contribution 0 and corrected ssum clamps to 1e-30 (softmax
# weight ~1e-30), and its dq is exactly 0 (all its K/V/rope rows are zero),
# so dropping them is exact up to the 1e-30 clamp (validated by the CP4
# oracle). The ring structure is untouched: same hops, same P2P buffers, same
# ``work.wait()`` fences, same payload / reduce-scatter.
#
# Gate: ``XTUNER_CP_RING_BUCKET=1`` (default OFF -- byte-identical to the
# pre-bucket path when off). ``XTUNER_CP_RING_BUCKET_N=force`` skips the
# predicted-time model and shrinks every chunk that has inactive rows
# (``auto`` default = shrink only when the model predicts a win). The model
# constants are env-calibratable via ``XTUNER_CP_RING_BUCKET_COST=<seed_f_ms>,
# <seed_g_ms>,<rate_ns_per_col>``.
# ---------------------------------------------------------------------------


def _bucket_enabled() -> bool:
    """Whether count-bucketed row compaction is active (env ``XTUNER_CP_RING_BUCKET``)."""
    return os.environ.get("XTUNER_CP_RING_BUCKET", "0") == "1"


def _bucket_force() -> bool:
    """Whether to shrink regardless of the predicted-time model (``XTUNER_CP_RING_BUCKET_N=force``)."""
    return os.environ.get("XTUNER_CP_RING_BUCKET_N", "auto").strip().lower() == "force"


def _bucket_cost_model() -> "tuple[float, float, float]":
    """Kernel cost model ``(seed_fwd_s, seed_grad_s, rate_s_per_col)``.

    Defaults fitted on run303 (256K trace): a per-kernel fixed seed of 4.4 ms
    (fwd) / 9.2 ms (per grad tile) plus 1.63 ns per query-row x index-column
    slot. ``XTUNER_CP_RING_BUCKET_COST="4.4,9.2,1.63"`` overrides (ms, ms, ns).

    Returns:
        ``(seed_fwd_s, seed_grad_s, rate_s_per_col)`` in SECONDS.
    """
    raw = os.environ.get("XTUNER_CP_RING_BUCKET_COST", "")
    if raw:
        try:
            sf, sg, rate = (float(x) for x in raw.split(","))
            if sf > 0 and sg > 0 and rate > 0:
                return sf * 1e-3, sg * 1e-3, rate * 1e-9
        except ValueError:
            pass
    return 4.4e-3, 9.2e-3, 1.63e-9


class _ChunkPlan(NamedTuple):
    """Prebuilt row-compaction plan for one ring chunk (built in Pass 2).

    The chunk kernel runs over ``rows`` (``count > 0`` rows in descending-count
    order) at the unchanged compact width ``k``; everything the forward and
    backward hot paths need is prebuilt here so the ring loops never touch
    host syncs (the per-tile ``qlen`` slices are async H2D views of
    :attr:`qlens`, kept alive by the cached plan).
    """

    rows: torch.Tensor  # [m] int64 global row indices
    idx: torch.Tensor  # [m, 1, k] int32 (row-gathered local_idx; width == k)
    ndummy: torch.Tensor  # [m] float32
    qlens: torch.Tensor  # [1 + tiles] int32 device: [m, tile sizes...]
    tiles: int  # grad row-tiles for this plan (host int; 1 when m is small)
    m: int  # active row count (host int)
    ms_pin: torch.Tensor  # pinned staging keep-alive (async H2D safety)


def _pick_bucket_plan(seq_len: int, k: int, na: int, n_tile: int) -> bool:
    """Host-only decision: shrink the chunk kernel to its ``na`` active rows.

    Compares the run303-fitted cost model of one full ``S x k`` call against
    one ``na x k`` call (with the grad tile count collapsing to 1 when
    ``na <= S // 4``, the measured small-M launch regime): shrinking saves
    ``rate * k * (S - na)`` columns twice (fwd + grad) plus, when the tile
    collapses, ``(n_tile - 1) * seed_grad`` -- it never adds a kernel, so the
    model cannot regress like multi-bucket splitting does (see the block
    comment above).

    Args:
        seq_len: Full query rows ``S``.
        k: Compact width (chunk amax count).
        na: Active-row count (``count > 0``), host int.
        n_tile: Grad row tiles today (:func:`_grad_tiles`).

    Returns:
        ``True`` when the plan should drop the inactive rows.
    """
    if na >= seq_len:
        return False  # nothing to drop -> byte-identical full-width path
    if _bucket_force():
        return True
    seed_f, seed_g, rate = _bucket_cost_model()
    nt_new = 1 if na <= seq_len // 4 else n_tile
    full = seed_f + seed_g * n_tile + rate * seq_len * k * 2.0
    shrink = seed_f + seed_g * nt_new + rate * na * k * 2.0
    return shrink < full


def _overlap_enabled() -> bool:
    """Whether the P2P hop overlaps the per-chunk kernel (bring-up fallback).

    ``XTUNER_CP_RING_OVERLAP=0`` degrades the ring to post-then-immediately-
    wait (pure blocking, equivalent to the old synchronous hop) so a suspected
    irecv-buffer race can be isolated without touching the ring structure.
    """
    return os.environ.get("XTUNER_CP_RING_OVERLAP", "1") != "0"


def _remap_cache_enabled() -> bool:
    """Whether the per-topk remap cache is enabled.

    The DSA top-k is shared across layers (``index_topk_freq``: some source
    layers compute it, the rest reuse the SAME tensor), and activation
    checkpointing recompute + the backward both reuse the forward's top-k
    tensor. The full remap (``max_counts`` Pass 1 + the per-chunk
    ``local_idx`` / ``ndummy`` tensors of Pass 2) is a PURE function of
    (top-k, chunk_len, cp_size, cp_rank), and the latter three are constant
    per process, so the same top-k tensor always yields the same ``all_idx``.
    Caching lets the sharing layers + recompute + backward skip Pass 1 (the
    ``scatter_add`` + ``.tolist`` D2H sync -- the host stall that gates
    ring-kernel launches) AND Pass 2 (``_remap_one_chunk`` device compute).

    The cached value holds device tensors, so a missed eviction would leak
    ~134 MB/step and OOM within steps; ``weakref.finalize`` evicts the entry
    the instant the top-k tensor is collected (at the step boundary, after the
    backward frees ``ctx``). A monotonic token guards against id-reuse: a late
    finalizer for a freed top-k (whose id was reused by a new one) checks the
    token and leaves the newer entry intact.
    """
    return os.environ.get("XTUNER_CP_REMAP_CACHE", "1") != "0"


# id(topk) -> (weakref.ref(topk), token, chunk_len, cp_size, cp_rank, all_idx).
# The token guards against id-reuse: a late finalizer for a freed top-k whose
# id was reused by a newer tensor checks the token and skips eviction of the
# newer entry.
_REMAP_CACHE: dict[
    int,
    tuple[
        weakref.ref,
        int,
        int,
        int,
        int,
        list[tuple[torch.Tensor | None, torch.Tensor | None, int, _ChunkPlan | None]],
    ],
] = {}
_REMAP_TOKEN = 0  # monotonic; incremented on each cache insert


def _remap_cache_evict(key: int, token: int) -> None:
    """Finalizer callback: drop a cache entry once its top-k tensor is freed.

    The token check prevents a stale finalizer (freed top-k whose id was reused
    by a newer tensor) from evicting the newer entry.
    """
    entry = _REMAP_CACHE.get(key)
    if entry is not None and entry[1] == token:
        del _REMAP_CACHE[key]


# Extra zero KV rows appended to every chunk (dims ``Rkv``/``Dr`` are imported
# from :mod:`ring_attention` -- single source of truth). Out-of-chunk / invalid
# indices are spread across these rows (``chunk_len + slot % _ZERO_ROWS``)
# instead of all landing on a single dummy row. The fused grad-op accumulates
# ``d_kv`` writes per row, so concentrating ``cp_size-1``/``cp_size`` of all
# slots on one row makes the backward ~6-7x slower (write contention);
# spreading across ``_ZERO_ROWS`` rows drops it to the random-index baseline.
# The count-correction is unchanged -- every dummy contributes ``exp(0) = 1``
# regardless of which zero row it points at.
_ZERO_ROWS = 64

# Cached constant seq-length tensors. ``npu_sparse_flash_attention[_grad]``
# takes ``actual_seq_{qlen,kvlen}`` as [1]-element int32 device tensors;
# building one per call via ``torch.tensor([int], device=device)`` is a
# synchronous host->device scalar transfer (~10 ms each on Ascend) that
# serialises the host ~26 ms per BWD grad-op call -- 5x the kernel itself.
# seq_len + _ZERO_ROWS are constant per process (uniform CP), so the pair is
# built once.
_SEQ_TENSORS: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}


def _seq_tensors(seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Cached ``(qlen, kvlen)`` int32 device tensors for the fused kernel.

    Args:
        seq_len: Per-rank query length (= chunk length for uniform CP).
        device: NPU device.

    Returns:
        ``(qlen, kvlen)``: ``[1]`` int32 tensors ``[seq_len]`` and
        ``[seq_len + _ZERO_ROWS]``, built once and reused across all calls.
    """
    key = (seq_len, str(device))
    t = _SEQ_TENSORS.get(key)
    if t is None:
        t = (
            torch.tensor([seq_len], dtype=torch.int32, device=device),
            torch.tensor([seq_len + _ZERO_ROWS], dtype=torch.int32, device=device),
        )
        _SEQ_TENSORS[key] = t
    return t


def _ring_reach(seq_ctx: "SequenceContext", seq_len: int, cp_size: int) -> int:
    """Uniform P2P hop bound from the packed segment lengths (pure host, 0 sync).

    A segment straddling the ring boundary extends any rank's causal KV need
    back by at most its own length, so the global max segment length bounds
    every rank's reach: ``ceil(max_seg / seq_len)`` past chunks suffice for
    ALL ranks. The count is uniform across the ring (``isend`` <-> ``irecv``
    match -> no deadlock), computed locally from ``cu_seq_lens_q_list``
    (CPU-resident, identical on every rank) -- NO collective, because a
    per-layer collective would lazily create a communicator whose CCL buffer
    can OOM mid-step on tight-memory configs (see
    ``backups/gradreduce_warmup_handoff/HANDOFF.md``). Over-estimation only
    appends all-dummy steps (kernel skipped locally); capped at ``cp_size - 1``
    because the ring physically wraps past that.

    Args:
        seq_ctx: SequenceContext (packed ``cu_seq_lens_q_list``).
        seq_len: Per-rank chunk length ``S``.
        cp_size: Context-parallel degree.

    Returns:
        Number of P2P hops past the local chunk (``0`` = local only).
    """
    if cp_size <= 1:
        return 0
    cu = seq_ctx.cu_seq_lens_q_list
    max_seg = 0
    for i in range(1, len(cu)):
        max_seg = max(max_seg, int(cu[i]) - int(cu[i - 1]))
    return min((max_seg + seq_len - 1) // seq_len, cp_size - 1)


def _remap_one_chunk(
    topk_global: torch.Tensor,
    chunk_start: int,
    chunk_len: int,
    max_count: int,
    n_active: int = 0,
) -> "tuple[torch.Tensor, torch.Tensor, int, _ChunkPlan | None]":
    """Remap global top-k indices to compacted chunk-local + zero-KV dummy pad.

    In-chunk real indices are converted to chunk-local offsets and packed
    (compacted) to ``[0, max_count)``; out-of-chunk / ``-1`` slots are routed
    across the appended zero-KV rows (indices
    ``[chunk_len, chunk_len + _ZERO_ROWS)``) as a dummy tail. The fused
    kernel's time scales with the index width ``K`` (measured on 910B3:
    ``K=2048`` is ~150x slower than ``K=128`` per chunk), so compacting the
    ~120 reals out of the full ``K=2048`` top-k to ``max_count`` is what keeps
    the per-chunk kernel fast. The compaction uses a stable argsort of the
    in-chunk mask (reals to the head, dummies to the tail); a cumsum prefix-sum
    scatter cannot do this cleanly (dummy cumsum positions collide with real
    positions, overwriting them).

    The count-correction (see :func:`_fused_chunk_attention`) cancels the
    dummy normaliser exactly regardless of how many dummies are present: each
    dummy KV row is zero, so it contributes ``exp(-smax)`` to ``ssum`` and
    thus ``1`` to ``z``; the kernel processes every slot independently (no
    dedup -- verified on 910B3), so ``ndummy = max_count - count`` matches the
    dummy slot count and the corrected ``ssum_true`` is exact (fp32).

    The all-dummy chunks (``max_count == 0``) early-return, so the ~15
    all-dummy chunks per ring step skip the kernel entirely. The full
    ``all_idx`` list is memoised by the ``topk_global`` tensor object
    (:data:`_REMAP_CACHE`): the DSA top-k is shared across layers (and reused
    by activation-checkpointing recompute + the backward), so the remap cost
    is paid once per step and amortised over the layer count.

    When ``n_active`` (Pass-1 count of rows with ``count > 0``) is below
    ``chunk_len`` and the cost model approves it (:func:`_pick_bucket_plan`,
    gate :func:`_bucket_enabled`), a :class:`_ChunkPlan` is additionally
    prebuilt: the ``n_active`` leading rows of the descending-count order
    (exactly the active ones) plus their gathered indices, so the hot loops
    can run the chunk kernel over the active rows only -- the inactive rows
    contribute exactly a zero output and a 1e-30-clamped weight (see the
    block comment above :func:`_bucket_enabled`), so dropping them is
    oracle-exact.

    Args:
        topk_global: ``[S, 1, K]`` int32/int64 global indices, ``-1`` invalid.
        chunk_start: Global offset of the chunk currently held.
        chunk_len: Length of the chunk (= ``S`` for uniform CP).
        max_count: Precomputed max real count for this chunk (host int).
        n_active: Precomputed active-row count for this chunk (host int;
            ``0`` disables plan building -- the bucket-gate-off path).

    Returns:
        ``(local_idx, ndummy, k, plan)``: ``local_idx`` is ``[S, 1, k]`` int32
        (reals packed to ``[0, count)`` as chunk-local offsets, dummy tail
        ``[count, max_count)`` spread across the zero-KV rows); ``ndummy`` is
        ``[S]`` float32 ``= max_count - count`` per query; ``k`` is
        ``max_count`` (the compact width), or ``0`` for an all-dummy chunk
        (``max_count == 0``) so the caller skips the kernel; ``plan`` is the
        row-compaction plan or ``None`` (full-width path, byte-identical).
    """
    k = max_count
    if max_count == 0:
        empty = topk_global.new_zeros(*topk_global.shape[:-1], 0, dtype=torch.int32)
        ndummy = torch.zeros(topk_global.shape[0], device=topk_global.device, dtype=torch.float32)
        return empty, ndummy, 0, None
    end = chunk_start + chunk_len
    in_chunk = (topk_global >= chunk_start) & (topk_global < end) & (topk_global != -1)
    counts = in_chunk.sum(dim=-1, dtype=torch.int32).squeeze(-1)  # [S]
    device = topk_global.device
    # Compact the in-chunk reals to ``[0, max_count)`` (see docstring for the
    # width argument). The dummy tail points into the appended zero-KV ROW BAND
    # spread by slot index (``chunk_len + slot % _ZERO_ROWS``), NOT at one
    # shared row: concentrating all dummies on a single row makes the fused
    # grad-op's per-row ``d_kv`` accumulation a write-contention hotspot (~6-7x
    # backward slowdown, measured); spreading across ``_ZERO_ROWS`` zero rows
    # costs nothing (all rows are zeros -> identical math).
    local_all = (topk_global - chunk_start).clamp(min=0).to(torch.int32)  # [S,1,K]
    num_slots = topk_global.shape[-1]
    dummy_rows = (chunk_len + torch.arange(num_slots, device=device, dtype=torch.int32) % _ZERO_ROWS).view(
        1, 1, num_slots
    )  # [1,1,K]
    local_src = torch.where(in_chunk, local_all, dummy_rows)  # [S,1,K]: reals + spread dummies
    # Stable argsort of the in-chunk mask packs reals to the head and dummies
    # to the tail. The mask MUST be cast to float32: on Ascend 910B3 the
    # ``[ArgSort]`` kernel cannot support int32/int64 on AiCore, so an int
    # mask dispatches to AiCpu (~45 ms per ``[S,1,K]`` call -- the dominant
    # compaction cost on the CP-on profile). A float32 mask dispatches to
    # AiCore (~0.16 ms -- 281x faster); the order among equal keys (the many
    # dummy ``0.0``s) differs from the int order, but the kernel is
    # permutation-invariant over the reals, so the real SET in ``[0, count)``
    # is identical and correctness holds.
    order = torch.argsort(in_chunk.float(), dim=-1, descending=True)  # reals first
    local = local_src.gather(-1, order)[..., :k]  # [S,1,k]: reals then dummy tail
    ndummy = (k - counts).to(torch.float32)  # [S] = max_count - count
    local = local.contiguous()
    plan: _ChunkPlan | None = None
    if (
        _bucket_enabled()
        and 0 < n_active < local.shape[0]
        and _pick_bucket_plan(local.shape[0], k, n_active, _grad_tiles())
    ):
        plan = _build_chunk_plan(counts, local, k, n_active)
    return local, ndummy, k, plan


def _build_chunk_plan(
    counts: torch.Tensor,
    local: torch.Tensor,
    k: int,
    n_active: int,
) -> _ChunkPlan:
    """Prebuild the active-row compaction plan for one chunk (Pass 2, off hot loop).

    Args:
        counts: ``[S]`` int32 real-key count per query for this chunk.
        local: ``[S, 1, k]`` int32 compacted chunk-local indices.
        k: Compact width (chunk amax count).
        n_active: Host count of rows with ``counts > 0`` (``< S``; rows with
            ``count == 0`` carry no real key anywhere in this chunk).

    Returns:
        The :class:`_ChunkPlan` (descending-count order makes the active rows
        exactly the first ``n_active`` entries; tile ``qlen`` tensors are
        async-H2D views of one pinned transfer -- never ``torch.tensor(int,
        device=...)``, which costs a ~10 ms host sync per build on Ascend).
    """
    seq_len = counts.shape[0]
    device = counts.device
    order = torch.argsort(
        counts.float(), dim=0, descending=True
    )  # [S] int64, AiCore (float32 cast: int argsort is the AiCpu trap)
    rows = order[:n_active].contiguous()
    idx = local.index_select(0, rows).contiguous()  # [m,1,k]
    ndummy = (k - counts.index_select(0, rows)).to(torch.float32)
    tiles = 1 if n_active <= seq_len // 4 else _grad_tiles()
    sizes = [n_active] + [(t + 1) * n_active // tiles - t * n_active // tiles for t in range(tiles)]
    ms_pin = torch.tensor(sizes, dtype=torch.int32).pin_memory()
    ms = torch.empty(len(sizes), dtype=torch.int32, device=device)
    ms.copy_(ms_pin, non_blocking=True)
    return _ChunkPlan(rows=rows, idx=idx, ndummy=ndummy, qlens=ms, tiles=tiles, m=n_active, ms_pin=ms_pin)


def _remap_max_counts(
    topk_global: torch.Tensor,
    chunk_len: int,
    cp_size: int,
    cp_rank: int,
) -> "tuple[list[int], list[int]]":
    """Pass 1: max real index count per ring step (one scatter_add + one D2H).

    Returns:
        ``(max_counts, n_active)`` -- both ``[cp_size]`` host ints indexed by
        ring step ``j``. ``max_counts[j]`` is the max over queries of the
        number of this rank's top-k slots landing in ring step ``j``'s chunk
        (``0`` = all-dummy, Pass 2 skips it). ``n_active[j]`` is the number of
        queries with at least one real slot in that chunk (the row-compaction
        target of :func:`_pick_bucket_plan`); it is ``0`` (unused) when the
        bucket gate is off and only the plain ``amax`` row was transferred.
    """
    seq_len = topk_global.shape[0]
    device = topk_global.device
    valid = topk_global != -1  # [S,1,K]
    chunk_id = topk_global.clamp(min=0).div(chunk_len, rounding_mode="floor")  # [S,1,K]
    counts_abs = torch.zeros(seq_len, cp_size, dtype=torch.int32, device=device)
    # ``scatter_add_`` REQUIRES an int64 index tensor; the production DSA top-k
    # is int32 (kept narrow to halve cache residency), so convert explicitly --
    # an int32 index raises "scatter_add only supports int64 indexing".
    counts_abs.scatter_add_(1, chunk_id.squeeze(1).to(torch.long), valid.squeeze(1).to(torch.int32))
    if not _bucket_enabled():
        max_counts_abs = counts_abs.amax(dim=0).tolist()  # [cp_size] -- one D2H sync
        return [int(max_counts_abs[(cp_rank - j) % cp_size]) for j in range(cp_size)], [0] * cp_size
    # The bucket gate widens the SAME single D2H with the ``n_active`` row
    # (counts <= 2048 and n_active <= 16384 are exact in fp32; the float32 cast
    # rides AiCore -- the int-reduction trap documented in _remap_one_chunk).
    amax_row = counts_abs.amax(dim=0).to(torch.float32)
    na_row = (counts_abs > 0).sum(dim=0).to(torch.float32)
    wide = torch.stack([amax_row, na_row]).tolist()  # ONE D2H sync ([2, cp_size])
    out = [int(wide[0][(cp_rank - j) % cp_size]) for j in range(cp_size)]
    na_abs = wide[1]
    n_active = [min(int(na_abs[(cp_rank - j) % cp_size]), seq_len) for j in range(cp_size)]
    return out, n_active


def _remap_all_chunks(
    topk_global: torch.Tensor,
    chunk_len: int,
    cp_size: int,
    cp_rank: int,
    max_counts: list[int] | None = None,
) -> "list[tuple[torch.Tensor | None, torch.Tensor | None, int, _ChunkPlan | None]]":
    """Pass 2: build compacted indices for ONLY the real (non-dummy) chunks.

    Memoised by the ``topk_global`` tensor object (:data:`_REMAP_CACHE`): the
    DSA top-k is shared across layers (and reused by activation-checkpointing
    recompute + the backward), so the same tensor maps to the same index
    tensors. A cache hit skips Pass 1 (the ``scatter_add`` + ``.tolist`` D2H
    sync) AND Pass 2 entirely.

    Args:
        topk_global: ``[S, 1, K]`` int32 global indices, ``-1`` invalid.
        chunk_len: Length of one chunk (= ``S`` for uniform CP).
        cp_size: Context-parallel degree.
        cp_rank: This rank's position in the CP ring.
        max_counts: Optional precomputed Pass-1 counts (fresh calls pass
            ``None`` to compute them here; the ring driver already computed
            them for its reach assertion).

    Returns:
        List of ``cp_size`` ``(local_idx, ndummy, k, plan)`` tuples indexed by
        ring step ``j`` (step ``j`` holds global chunk ``(cp_rank - j) %
        cp_size``); all-dummy steps carry ``(None, None, 0, None)`` sentinels
        and non-compacted steps carry ``plan=None`` (full-width path).
    """
    if _remap_cache_enabled():
        key = id(topk_global)
        entry = _REMAP_CACHE.get(key)
        if entry is not None:
            wref, _tok, cl, cs, cr, all_idx = entry
            if wref() is not None and cl == chunk_len and cs == cp_size and cr == cp_rank:
                return all_idx  # cache hit: skip Pass 1 + Pass 2
            # Dead top-k (freed at step boundary) or param change -> evict.
            _REMAP_CACHE.pop(key, None)
    if max_counts is None:
        max_counts, n_active = _remap_max_counts(topk_global, chunk_len, cp_size, cp_rank)
    else:
        n_active = [0] * cp_size
    starts = [((cp_rank - j) % cp_size) * chunk_len for j in range(cp_size)]
    # Bucket-gate invariant (forward_true's after-loop fold relies on it): at
    # most one ring step gets a plan, and never step 0 (the deferred pair fold
    # stays raw-in-place into step 0's full-width buffer). When several steps
    # would qualify, bucket NOTHING (cost-model loss anyway: each extra plan
    # adds a fold partner, and the first step's raw must keep [S] rows).
    plan_j = -1
    if _bucket_enabled():
        cands = [
            j
            for j in range(cp_size)
            if max_counts[j] > 0 and _pick_bucket_plan(chunk_len, max_counts[j], n_active[j], _grad_tiles())
        ]
        if len(cands) == 1 and cands[0] > 0:
            plan_j = cands[0]
    # All-dummy chunks (max_count == 0) are skipped entirely -- no
    # new_zeros/zeros placeholder ops, no range-check. The DSA top-k has strong
    # locality (~1 real chunk per ring step), so ~15 of 16 chunks are
    # all-dummy; building their placeholders was ~900 wasted host-launched
    # ops/step. The ring driver's ``if k == 0: skip`` never touches the None
    # sentinels, so None is safe.
    out: list[tuple[torch.Tensor | None, torch.Tensor | None, int, _ChunkPlan | None]] = []
    for j in range(cp_size):
        if max_counts[j] == 0:
            out.append((None, None, 0, None))
        else:
            out.append(
                _remap_one_chunk(topk_global, starts[j], chunk_len, max_counts[j], n_active[j] if j == plan_j else 0)
            )
    if _remap_cache_enabled():
        global _REMAP_TOKEN
        _REMAP_TOKEN += 1
        token = _REMAP_TOKEN
        key = id(topk_global)
        _REMAP_CACHE[key] = (
            weakref.ref(topk_global),
            token,
            chunk_len,
            cp_size,
            cp_rank,
            out,
        )
        weakref.finalize(topk_global, _remap_cache_evict, key, token)
    return out


def _fused_chunk_attention(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    chunk_kv: torch.Tensor,
    local_idx: torch.Tensor,
    ndummy: torch.Tensor,
    scale: float,
    qlen: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the fused sparse kernel on one chunk + dummy pad + count-correction.

    Args:
        q_nope: ``[S, N, Rkv]`` absorbed query (no-rope latent part); a row
            SUBSET (``S <=`` chunk rows) when ``qlen`` is given (bucketed
            row-compaction call, see :class:`_ChunkPlan`).
        q_rope: ``[S, N, Dr]`` query rope (same rows as ``q_nope``).
        chunk_kv: ``[chunk_len, 1, Rkv + Dr]`` latent KV (kv_compressed ++
            k_pe) -- always the FULL chunk (permutation- and subset-invariant
            on the KV side).
        local_idx: ``[S, 1, k]`` int32 chunk-local indices for the query rows
            passed, dummies in the appended zero-row band
            ``[chunk_len, chunk_len + _ZERO_ROWS)``.
        ndummy: ``[S]`` float32 dummy count per query row passed.
        scale: Softmax scale.
        qlen: Optional ``[1]`` int32 device tensor with the query-row count
            (subset calls); ``None`` = full chunk, uses the cached constant.

    Returns:
        ``(out_raw, smax_sn, ssum_true_sn, rescale_sn)``: ``[S, N, Rkv]``
        UNCORRECTED output in the kernel-native (model) dtype, ``[S, N]``
        float32 kernel softmax_max (squeezed), ``[S, N]`` float32 corrected
        (dummy-excluded) softmax_sum (squeezed), and the ``[S, N]`` float32
        output rescale ``ssum / ssum_true``. The correction is folded into the
        merge weights instead of the output buffer: with the true normaliser
        part ``z_true = ssum_true * exp`` we have raw ``z_raw = ssum * exp =
        z_true * rescale``, so folding ``out_raw`` with ``z_true * rescale``
        (or ``rescale`` alone for the first chunk) is fp32-exactly the
        corrected fold -- with no ``[S, N, Rkv]`` fp32 temporary per chunk
        (2.1 GiB each at 256K). The merged ``(smax, ssum_true)`` over chunks
        reconstructs the exact merged normaliser ``z_merged =
        ssum_merged * exp(smax_merged)``.
    """
    seq_len, num_heads, _ = q_nope.shape
    kv_len = chunk_kv.shape[0]
    # Uniform CP: chunk length == query length, so the cached _kvlen
    # (seq_len + _ZERO_ROWS) matches kv_pad's first dim exactly. Bucketed
    # subset calls pass fewer query rows than the (unchanged) full chunk.
    assert kv_len >= seq_len and (qlen is not None or kv_len == seq_len)
    device = chunk_kv.device
    dtype = chunk_kv.dtype
    import torch_npu  # type: ignore[import-untyped]  # local: only the fused-kernel path needs the NPU runtime

    kv_comp = chunk_kv[..., :Rkv]
    k_rope = chunk_kv[..., Rkv : Rkv + Dr]
    kv_pad = torch.cat([kv_comp, torch.zeros(_ZERO_ROWS, 1, Rkv, device=device, dtype=dtype)], dim=0)
    krope_pad = torch.cat([k_rope, torch.zeros(_ZERO_ROWS, 1, Dr, device=device, dtype=dtype)], dim=0)
    if qlen is None:
        _qlen, _kvlen = _seq_tensors(seq_len, device)
    else:
        _qlen = qlen
        _, _kvlen = _seq_tensors(kv_len, device)
    out = torch_npu.npu_sparse_flash_attention(
        q_nope.unsqueeze(0).contiguous(),
        kv_pad.unsqueeze(0).contiguous(),
        kv_pad.unsqueeze(0).contiguous(),
        sparse_indices=local_idx.unsqueeze(0).contiguous(),
        block_table=None,
        actual_seq_lengths_query=_qlen,
        actual_seq_lengths_kv=_kvlen,
        query_rope=q_rope.unsqueeze(0).contiguous(),
        key_rope=krope_pad.unsqueeze(0).contiguous(),
        scale_value=scale,
        sparse_block_size=1,
        layout_query="BSND",
        layout_kv="BSND",
        sparse_mode=0,
        attention_mode=2,
        return_softmax_lse=True,
    )
    attn = out[0].squeeze(0)  # [S, N, Rkv] kernel-native dtype, RAW -- the
    # dummy correction rides on the merge weights (rescale below), so no fp32
    # copy of the kernel output is ever materialised.
    # Kernel softmax stats are [B, kv_heads, S, N//kv_heads] = [1,1,S,N] for
    # MLA (kv_heads=1). Squeeze to [S, N] for the merge math.
    smax_sn = out[1].squeeze(0).squeeze(0).float()  # [S, N]
    ssum_sn = out[2].squeeze(0).squeeze(0).float()  # [S, N]
    # Dummy-exclusion correction in OVERFLOW-FREE form. The raw normaliser is
    # ``z = exp(smax) * ssum`` and each dummy adds ``exp(0 - smax) * exp(smax)
    # = 1``, so ``ssum_true = ssum - ndummy * exp(-smax)`` and the output
    # rescale ``z / z_true`` equals ``ssum / ssum_true`` -- never forming
    # ``exp(smax)`` (which overflows fp32 once the row max exceeds ~88). Safe
    # to rely on: every query row holds dummies with score 0, so the kernel's
    # smax (max over all compacted slots) is >= 0 and ``exp(-smax) <= 1``.
    ssum_true_sn = (ssum_sn - ndummy.unsqueeze(-1) * torch.exp(-smax_sn)).clamp_min(min=1e-30)  # [S, N]
    rescale_sn = ssum_sn / ssum_true_sn  # [S, N] fp32 -- folded into the
    # merge weight (z_true * rescale == z_raw) by _merge_stats / the driver's
    # first-chunk fold, keeping the accumulator fold fp32-exact.
    return attn, smax_sn, ssum_true_sn, rescale_sn


def _merge_stats(
    prev_out: torch.Tensor,
    prev_smax: torch.Tensor,
    prev_ssum: torch.Tensor,
    cur_out: torch.Tensor,
    cur_smax: torch.Tensor,
    cur_ssum: torch.Tensor,
    cur_rescale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Online-softmax merge of two (out, smax, ssum) triples.

    Merges chunk statistics into running merged statistics. ``z = ssum *
    exp(smax)`` so the merged normaliser is ``z_prev + z_cur``; the output is
    the z-weighted average ``(z_prev*out_prev + z_cur*out_cur) / z_merged``.
    ``cur_out`` is the kernel's RAW (uncorrected) output, so its weight carries
    the count correction: ``z_raw * ... = z_true * cur_rescale`` (``ssum =
    ssum_true * rescale``), which folds the dummy exclusion into the fp32
    weight product instead of the ``[S, N, Rkv]`` output buffer -- fp32-exact
    with no fp32 temporary per chunk.

    Args:
        prev_out: ``[S, N, Rkv]`` float32 running merged output.
        prev_smax: ``[S, N]`` running merged softmax_max.
        prev_ssum: ``[S, N]`` running merged (dummy-excluded) softmax_sum.
        cur_out: ``[S, N, Rkv]`` current chunk RAW output (kernel dtype).
        cur_smax: ``[S, N]`` current chunk raw softmax_max.
        cur_ssum: ``[S, N]`` current chunk corrected softmax_sum.
        cur_rescale: ``[S, N]`` float32 ``ssum / ssum_true`` output correction.

    Returns:
        ``(new_out, new_smax, new_ssum)``: merged ``[S, N, Rkv]`` float32
        output and ``[S, N]`` merged softmax stats. ``new_out`` IS ``prev_out``
        (folded in-place). Both stat buffers are raw tensors inside the custom
        Function (no autograd history), and the driver's first-chunk output
        buffer has no other owner, so the in-place fold is safe. The
        accumulator stays FLOAT32 end to end: a bf16 fold drifts ~1 ulp
        (~0.8%) per chunk and the Formulation-B backward amplifies it through
        cancellation (caught by the CP4 oracle). The bf16 kernel output is
        promoted elementwise in place by ``addcmul_`` -- zero allocation,
        bit-identical to an explicit ``.float()`` cast (verified on 910B3).
    """
    new_smax = torch.maximum(prev_smax, cur_smax)  # [S, N]
    prev_w = prev_ssum * torch.exp(prev_smax - new_smax)  # [S, N] (= z_prev / exp(new_smax))
    cur_w = cur_ssum * torch.exp(cur_smax - new_smax)  # [S, N]
    new_ssum = prev_w + cur_w  # [S, N]
    prev_out.mul_(prev_w.unsqueeze(-1))
    # cur_out (kernel dtype) x fp32 weight: the fused correction (cur_w *
    # cur_rescale) is fp32 [S, N], so the mixed-dtype addcmul_ computes in the
    # accumulator's fp32 and allocates nothing.
    prev_out.addcmul_(cur_out, (cur_w * cur_rescale).unsqueeze(-1))
    prev_out.div_(new_ssum.unsqueeze(-1))
    return prev_out, new_smax, new_ssum


# Value-dim tile count for the deferred folds below: 512 -> 4 tiles of
# [S, N, 128] fp32 = 0.5 GiB at 256K.
_FOLD_TILES = 4


def _defer_fold_enabled() -> bool:
    """Whether the ``reach == 1`` deferred fold is active (see :func:`_fold_two_raws`)."""
    return os.environ.get("XTUNER_CP_DEFER_FOLD", "1") != "0"


def _grad_tiles() -> int:
    """Row-tiles per backward grad kernel call (env ``XTUNER_CP_RING_GRAD_TILES``, ``1`` = off).

    The ``npu_sparse_flash_attention_grad`` workspace scales with the query-row
    count (an fp32 score-grad buffer of ``[M, N, k]``): one whole-chunk call at
    256K (M = 16384) asks the NPUWorkspaceAllocator for 2.55 GiB. The ring's
    deferred-fold forward high-water (+2 GiB -- the raw pair is irreducible
    without breaking the bit-exact fp32 fold order) ratchets torch
    reserved to ~55.5 GB, and the workspace pool is then starved by exactly one
    allocation -- runs 286/287/288 all died on this one call. Attention grads
    are EXACT per query row (the merged stats are fed in; the KV side is only
    read, its grads accumulate), so splitting rows into tiles costs two extra
    kernel launches and halves the workspace request (~1.28 GiB, which fits).
    """
    try:
        n = int(os.environ.get("XTUNER_CP_RING_GRAD_TILES", "2"))
    except ValueError:
        n = 2
    return max(1, n)


def _fold_one_raw(raw0: torch.Tensor, res0: torch.Tensor) -> torch.Tensor:
    """Fold a single real chunk in place, tile by tile, without an fp32 accumulator.

    The ring's most common case at 256K (``cp_size=16``, short packed segments)
    is ``reach == 1`` with only the local chunk holding real top-k slots. The
    online path would still build the 2 GiB fp32 accumulator (``mul``) and cast
    it back; with exactly one chunk the fold is just ``raw0 * res0``, so each
    value-dim tile computes its fp32 product and writes straight back into
    ``raw0``'s bf16 buffer -- the accumulator never exists. Elementwise fp32
    op on the same inputs in the same order, so this is BIT-IDENTICAL to the
    online first-chunk fold plus the final ``copy_`` cast.

    Args:
        raw0: ``[S, N, Rkv]`` kernel-dtype RAW chunk output (becomes the folded
            output in place).
        res0: ``[S, N]`` float32 dummy-exclusion rescale of ``raw0``.

    Returns:
        ``raw0`` (now holding the corrected output in the model dtype).
    """
    per_tile = (Rkv + _FOLD_TILES - 1) // _FOLD_TILES
    for t in range(_FOLD_TILES):
        sl = slice(t * per_tile, min((t + 1) * per_tile, Rkv))
        raw0[..., sl].copy_(raw0[..., sl].mul(res0.unsqueeze(-1)))
    return raw0


def _fold_two_raws(
    raw0: torch.Tensor,
    res0: torch.Tensor,
    smax0: torch.Tensor,
    ssum0: torch.Tensor,
    raw1: torch.Tensor,
    res1: torch.Tensor,
    smax1: torch.Tensor,
    ssum1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deferred two-chunk merge: no persistent fp32 accumulator (``reach == 1``).

    The online fold keeps a FLOAT32 ``[S, N, Rkv]`` accumulator (2 GiB at 256K)
    alive from the first real chunk's fold until the last -- and it is the
    transient that pushes this box past its
    pool ceiling (run281/283/284 forensics). With at most two chunks the
    accumulator is avoidable entirely: both RAW bf16 outputs are held instead
    (1 GiB each), and the merge runs value-dim tile by tile, each tile
    performing EXACTLY the online fp32 op sequence (``mul res0`` -> ``mul
    prev_w`` -> ``addcmul_`` the corrected ``raw1`` -> ``div_`` new_ssum`) on a
    0.5 GiB scratch and writing the bf16 result back into ``raw0``'s buffer.
    Elementwise arithmetic with the same inputs in the same order means the
    result is BIT-IDENTICAL to the online two-chunk fold, so the CP4 oracle
    numerics are unchanged. Fold peak drops 3.2 -> 2.5 GiB; the 2 GiB odd size
    class never enters the pool.

    Args:
        raw0: ``[S, N, Rkv]`` kernel-dtype RAW output of the first real chunk
            (becomes the merged output in place).
        res0: ``[S, N]`` float32 rescale of ``raw0``.
        smax0: ``[S, N]`` raw softmax_max of ``raw0``.
        ssum0: ``[S, N]`` corrected softmax_sum of ``raw0``.
        raw1: ``[S, N, Rkv]`` kernel-dtype RAW output of the second real chunk.
        res1: ``[S, N]`` float32 rescale of ``raw1``.
        smax1: ``[S, N]`` raw softmax_max of ``raw1``.
        ssum1: ``[S, N]`` corrected softmax_sum of ``raw1``.

    Returns:
        ``(raw0, new_smax, new_ssum)``: merged output (model dtype, aliased
        into ``raw0``'s buffer) and merged ``[S, N]`` float32 stats.
    """
    new_smax = torch.maximum(smax0, smax1)  # [S, N]
    prev_w = ssum0 * torch.exp(smax0 - new_smax)  # [S, N]
    cur_w = ssum1 * torch.exp(smax1 - new_smax)  # [S, N]
    new_ssum = prev_w + cur_w  # [S, N]
    per_tile = (Rkv + _FOLD_TILES - 1) // _FOLD_TILES
    for t in range(_FOLD_TILES):
        sl = slice(t * per_tile, min((t + 1) * per_tile, Rkv))
        acc = raw0[..., sl].mul(res0.unsqueeze(-1))  # fp32 tile (promoted product)
        acc.mul_(prev_w.unsqueeze(-1))
        acc.addcmul_(raw1[..., sl], (cur_w * res1).unsqueeze(-1))
        acc.div_(new_ssum.unsqueeze(-1))
        raw0[..., sl].copy_(acc)
    return raw0, new_smax, new_ssum


def _fold_pair_subset(
    raw0: torch.Tensor,
    res0: torch.Tensor,
    smax0: torch.Tensor,
    ssum0: torch.Tensor,
    raw1: torch.Tensor,
    res1: torch.Tensor,
    smax1: torch.Tensor,
    ssum1: torch.Tensor,
    rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-chunk deferred fold where the SECOND chunk covers only ``rows`` (bucketed).

    Row-compaction twin of :func:`_fold_two_raws`: ``raw1``/stats hold the
    ``m`` active rows of chunk 1 (:class:`_ChunkPlan`), ``raw0`` the full
    ``[S]`` rows of chunk 0. Rows in ``rows`` get EXACTLY the
    :func:`_fold_two_raws` fp32 expression (same op order, promoted
    elementwise) folded in place into ``raw0``'s buffer; rows outside it saw
    only chunk 0, so they keep the bare rescale -- which is what the
    full-width fold computes for them up to the 1e-30-clamped empty
    contribution of their dummy rows (dropped here; the CP4 oracle covers the
    drift, see the block comment on :func:`_bucket_enabled`). Peak is the
    pair plus one ``[m, N, Rkv // _FOLD_TILES]`` fp32 tile -- never larger
    than the non-bucketed fold; the stat tensors are the deferred pend's
    single-owner buffers, so they are updated in place.

    Args:
        raw0: ``[S, N, Rkv]`` kernel-dtype RAW output of the first (full)
            chunk (becomes the merged output in place).
        res0: ``[S, N]`` float32 rescale of ``raw0``.
        smax0: ``[S, N]`` raw softmax_max of ``raw0`` (updated in place).
        ssum0: ``[S, N]`` corrected softmax_sum of ``raw0`` (updated in place).
        raw1: ``[m, N, Rkv]`` kernel-dtype RAW output of the second chunk.
        res1: ``[m, N]`` float32 rescale of ``raw1``.
        smax1: ``[m, N]`` raw softmax_max of ``raw1``.
        ssum1: ``[m, N]`` corrected softmax_sum of ``raw1``.
        rows: ``[m]`` int64 global row indices covered by ``raw1``.

    Returns:
        ``(raw0, smax0, ssum0)``: merged output (aliased) and merged stats.
    """
    r_smax0 = smax0.index_select(0, rows)  # [m, N]
    r_ssum0 = ssum0.index_select(0, rows)
    new_smax = torch.maximum(r_smax0, smax1)  # [m, N]
    prev_w = r_ssum0 * torch.exp(r_smax0 - new_smax)
    cur_w = ssum1 * torch.exp(smax1 - new_smax)
    new_ssum = prev_w + cur_w  # [m, N]
    smax0.index_copy_(0, rows, new_smax)
    ssum0.index_copy_(0, rows, new_ssum)
    r_res0 = res0.index_select(0, rows)  # [m, N] -- pre-fold chunk-0 rescale of the subset
    per_tile = (Rkv + _FOLD_TILES - 1) // _FOLD_TILES
    for t in range(_FOLD_TILES):
        sl = slice(t * per_tile, min((t + 1) * per_tile, Rkv))
        a0 = raw0[..., sl].index_select(0, rows).float()  # [m, N, ts] fp32, pre-fold chunk-0 part
        raw0[..., sl].copy_(raw0[..., sl].mul(res0.unsqueeze(-1)))  # single-chunk rescale for ALL
        # rows, in place (out-of-place mul + copy_: bf16 self x fp32 arg is not
        # a valid in-place promotion -- same form as _fold_one_raw, and the
        # rows overwritten below never read this intermediate).
        a0.mul_(r_res0.unsqueeze(-1))  # the _fold_two_raws op order exactly:
        a0.mul_(prev_w.unsqueeze(-1))  # raw -> mul res0 -> mul prev_w (fp32, bit-identical)
        a0.addcmul_(raw1[..., sl], (cur_w * res1).unsqueeze(-1))
        a0.div_(new_ssum.unsqueeze(-1))
        raw0[..., sl].index_copy_(0, rows, a0.to(raw0.dtype))
    return raw0, smax0, ssum0


def _fold_subset_single(
    raw0: torch.Tensor,
    res0: torch.Tensor,
    smax0: torch.Tensor,
    ssum0: torch.Tensor,
    rows: torch.Tensor,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fold the only real chunk when it covers a row subset (bucketed degenerate case).

    Output rows outside ``rows`` keep today's all-dummy-fold values exactly
    (zero out, ``smax`` 0, clamped ``ssum`` 1e-30 -- their whole kernel
    contribution was that clamp), reconstructed device-side with no host
    sync; rows inside get the plain ``raw0 * res0`` rescale.

    Args:
        raw0: ``[m, N, Rkv]`` kernel-dtype RAW output of the only real chunk.
        res0: ``[m, N]`` float32 rescale.
        smax0: ``[m, N]`` raw softmax_max.
        ssum0: ``[m, N]`` corrected softmax_sum.
        rows: ``[m]`` int64 global row indices.
        seq_len: Full query rows ``S``.

    Returns:
        ``(out, merged_smax, merged_ssum)`` in the full ``[S, ...]`` shapes.
    """
    out = torch.zeros(seq_len, raw0.shape[1], raw0.shape[2], dtype=raw0.dtype, device=raw0.device)
    merged_smax = torch.zeros(seq_len, smax0.shape[1], dtype=torch.float32, device=smax0.device)
    merged_ssum = torch.zeros_like(merged_smax)
    per_tile = (Rkv + _FOLD_TILES - 1) // _FOLD_TILES
    for t in range(_FOLD_TILES):
        sl = slice(t * per_tile, min((t + 1) * per_tile, Rkv))
        out[..., sl].index_copy_(0, rows, raw0[..., sl].mul(res0.unsqueeze(-1)).to(raw0.dtype))
    merged_smax.index_copy_(0, rows, smax0)
    merged_ssum.index_copy_(0, rows, ssum0)
    merged_ssum.masked_fill_(merged_ssum == 0, 1e-30)  # never-attended rows: today's clamp
    return out, merged_smax, merged_ssum


# ---------------------------------------------------------------------------
# Recompute-skip stash.
#
# Reentrant activation checkpointing replays every checkpointed region INSIDE
# the backward: true ring would re-run P2P + per-chunk kernels + fold a second
# time in the window where grads, aclnn workspaces and H2D re-unpacks are
# densest -- the replay, not the forward, is what overflowed the pool ceiling
# (run281/284/285 died with reserved 55.4-55.5 GB).
# The stash eliminates the replay: AC pass 1 (a forward pass running OUTSIDE
# the autograd engine) copies the fold result to pinned CPU asynchronously --
# the same ``SwapTensor`` dance the activation offload uses every step, with
# its stream-ordering guarantees -- and frees the GPU storage; the replay
# (running INSIDE the engine, see :func:`_in_backward_graph`) pops the entry,
# H2Ds the bytes back into the SAME tensor objects and returns, so the
# backward sees bit-identical activations without running a single ring op.
# Grad mode is NOT a usable discriminator: ``Function.apply`` runs the user
# forward with grad disabled in both passes on this torch_npu stack.
# GPU cost: zero during the
# fwd->bwd gap (storage is resize_0'd); host cost: ~1.1 GiB pinned per live
# layer (pinned buffers are pooled and reused across steps).
#
# LIFO ordering per top-k object: DSA top-k tensors are SHARED across the
# ``index_topk_freq`` layers that consume them, so one stack holds several
# layer results; pass-1 pushes in forward layer order and the backward
# recomputes in reverse, so ``pop()`` hands each replay exactly its own entry.
# A miss (CP4-style direct grad-enabled call, replay with the stash already
# drained) falls back to full recomputation -- the pre-stash behaviour, always
# correct.
# ---------------------------------------------------------------------------


def _recompute_skip_enabled() -> bool:
    """Whether the recompute-skip stash is active (env ``XTUNER_CP_RECOMPUTE_SKIP``)."""
    return os.environ.get("XTUNER_CP_RECOMPUTE_SKIP", "1") != "0"


def _in_backward_graph() -> bool:
    """Whether the current code runs inside an autograd-engine backward graph.

    Reentrant activation checkpointing replays the region INSIDE ``backward``,
    so the engine's graph-task id (``-1`` outside) -- not the grad mode --
    distinguishes AC pass 1 from the replay: ``torch.autograd.Function.apply``
    executes the user ``forward`` with grad mode DISABLED in BOTH passes, so
    ``torch.is_grad_enabled()`` reads identically ``False`` inside the ring
    forward and cannot gate the stash (verified on this box's torch_npu 2.13).
    """
    return int(torch._C._current_graph_task_id()) >= 0


class _FoldStash:
    """SwapTensor-wrapped fold result parked between AC pass 1 and the replay."""

    def __init__(self) -> None:
        self.out: SwapTensor | None = None
        self.smax: SwapTensor | None = None
        self.ssum: SwapTensor | None = None
        self.payload: list[tuple[int, int, SwapTensor]] = []


# id(topk) -> (weakref.ref(topk), token, LIFO stack of stashes). Token guards
# id-reuse exactly like ``_REMAP_CACHE``; the finalizer drops the whole entry
# when the top-k tensor dies (step boundary), releasing its pinned slots and
# (already resize_0'd) GPU shells.
_FOLD_STASH: dict[int, tuple[weakref.ref, int, list[_FoldStash]]] = {}
_FOLD_STASH_TOKEN = 0
# Swaps whose GPU storage is waiting to be resize_0'd: freed at the NEXT
# forward ENTRY (or the first restore), by which point the D2H has had a full
# layer (~s) to finish, so ``wait_d2h_finished``'s stream waits are no-ops --
# and crucially BEFORE the next layer's fold window allocates its 2 raws, so a
# stashed 1.07 GiB ``out_final`` never overlaps the incoming fold peak
# (run286/287 died one aclnn workspace short of exactly this overlap).
_FOLD_PENDING: list[SwapTensor] = []
# Pinned-CPU buffer slots (into OffloadManager.get_or_create_pin_memory),
# recycled between steps: allocate on stash, return on restore.
_FOLD_FREE_SLOTS: list[int] = []
_FOLD_SLOT_COUNTER = 0
_FOLD_COPY_STREAM: "torch.cuda.Stream | None" = None


def _fold_copy_stream() -> torch.cuda.Stream:
    """Lazily-created dedicated copy stream for the fold stash."""
    global _FOLD_COPY_STREAM
    if _FOLD_COPY_STREAM is None:
        _FOLD_COPY_STREAM = torch.cuda.Stream()
    return _FOLD_COPY_STREAM


def _flush_fold_pending() -> None:
    """Free the GPU storage of previously-stashed swaps (their D2H is long done)."""
    stream = _fold_copy_stream()
    for st in _FOLD_PENDING:
        st.wait_d2h_finished(stream, True)
    _FOLD_PENDING.clear()


def _swap_offload(tensor: torch.Tensor, name: str) -> SwapTensor:
    """Copy ``tensor`` to a pooled pinned buffer asynchronously (AC pass 1 only)."""
    global _FOLD_SLOT_COUNTER
    if _FOLD_FREE_SLOTS:
        slot = _FOLD_FREE_SLOTS.pop()
    else:
        slot = _FOLD_SLOT_COUNTER
        _FOLD_SLOT_COUNTER += 1
    key = f"ringfold_{slot}_{name}"
    cpu = OffloadManager().get_or_create_pin_memory(key, tensor.shape, tensor.dtype)
    swap = SwapTensor(tensor, key, tensor_cpu=cpu)
    stream = _fold_copy_stream()
    stream.wait_stream(torch.cuda.current_stream())  # after the producing kernels
    swap.launch_d2h(stream)
    _FOLD_PENDING.append(swap)  # GPU storage freed by the next flush
    return swap


def _stash_put(
    topk_indices: torch.Tensor,
    out: torch.Tensor,
    smax: torch.Tensor,
    ssum: torch.Tensor,
    payload: list[_ChunkPayload],
) -> None:
    """Park an AC pass-1 (forward, outside the engine) fold result for the replay."""
    global _FOLD_STASH_TOKEN
    _flush_fold_pending()  # previous layer's D2H is a full layer old: no-op waits
    key = id(topk_indices)
    entry = _FOLD_STASH.get(key)
    if entry is None or entry[0]() is not topk_indices:
        _FOLD_STASH_TOKEN += 1
        entry = (weakref.ref(topk_indices), _FOLD_STASH_TOKEN, [])
        _FOLD_STASH[key] = entry
        weakref.finalize(topk_indices, _stash_evict, key, _FOLD_STASH_TOKEN)
    stash = _FoldStash()
    stash.out = _swap_offload(out, "out")
    stash.smax = _swap_offload(smax, "smax")
    stash.ssum = _swap_offload(ssum, "ssum")
    for j, gs, kv in payload:
        stash.payload.append((j, gs, _swap_offload(kv, "kv")))
    entry[2].append(stash)


def _stash_evict(key: int, token: int) -> None:
    """Finalizer: drop the whole stash entry once its top-k tensor is freed."""
    entry = _FOLD_STASH.get(key)
    if entry is not None and entry[1] == token:
        for stash in entry[2]:
            _FOLD_FREE_SLOTS.extend(int(s.key.split("_")[1]) for s in _stash_swaps(stash))
        _FOLD_STASH.pop(key, None)


def _stash_swaps(stash: _FoldStash) -> list[SwapTensor]:
    """All SwapTensors of one stashed fold result (out/smax/ssum + payload KV)."""
    assert stash.out is not None and stash.smax is not None and stash.ssum is not None
    return [stash.out, stash.smax, stash.ssum, *[s for _, _, s in stash.payload]]


def _stash_take(
    topk_indices: torch.Tensor,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[_ChunkPayload]] | None":
    """Pop the most-recent stash for ``topk_indices`` and H2D it back (replay pass).

    Returns the restored ``(out, smax, ssum, payload)`` -- the SAME tensor
    objects AC pass 1 computed, refilled bit-identically from pinned CPU -- or
    ``None`` when nothing is stashed (fall back to full recomputation).
    """
    key = id(topk_indices)
    entry = _FOLD_STASH.get(key)
    # Identity check on the weakref target, mirroring ``_stash_put`` /
    # ``_remap_all_chunks``: an entry whose referent died before its evict
    # finalizer ran must NOT hand its stash to an id-reusing tensor (a wrong
    # restore is silently bad activations -- strictly worse than a miss).
    if entry is None or entry[0]() is not topk_indices:
        return None
    if not entry[2]:
        return None
    stash = entry[2].pop()
    _flush_fold_pending()  # order the current stream behind every D2H (pinned
    # buffers must hold final bytes before the H2D below reads them) + free the
    # last forward layer's GPU storage before we re-materialise one.
    stream = _fold_copy_stream()
    working = torch.cuda.current_stream()
    stream.wait_stream(working)

    def restore(swap: SwapTensor) -> torch.Tensor:
        swap.launch_h2d(stream, True, working)  # resize_ + copy_ on the working stream
        assert swap.tensor is not None
        return swap.tensor

    out = restore(stash.out) if stash.out is not None else None
    smax = restore(stash.smax) if stash.smax is not None else None
    ssum = restore(stash.ssum) if stash.ssum is not None else None
    assert out is not None and smax is not None and ssum is not None  # _stash_put always fills
    payload = [(j, gs, restore(s)) for j, gs, s in stash.payload]
    _FOLD_FREE_SLOTS.extend(int(s.key.split("_")[1]) for s in _stash_swaps(stash))
    return out, smax, ssum, payload


def _parse_lse(smax: torch.Tensor, ssum: torch.Tensor, seq_len: int, num_heads: int) -> torch.Tensor:
    """Reconstruct the merged logsumexp from (smax, ssum) stats.

    ``lse = smax + log(ssum)``; the result is reshaped to ``[seq_len,
    num_heads]`` so callers can pass either the merged ``[S, N]`` stats or the
    kernel-native ``[1, 1, S, N]`` / ``[1, S, N]`` layout.

    Args:
        smax: Merged softmax_max (any leading dims, ``S*N`` elements).
        ssum: Merged (dummy-excluded) softmax_sum (same shape as ``smax``).
        seq_len: Local sequence length ``S``.
        num_heads: Number of attention heads ``N``.

    Returns:
        ``[seq_len, num_heads]`` float32 merged logsumexp.
    """
    return (smax + torch.log(ssum + 1e-30)).reshape(seq_len, num_heads)


def _grad_op_inputs(
    chunk_kv: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the padded KV tensors for one chunk's grad-op call.

    Args:
        chunk_kv: ``[S, 1, Rkv + Dr]`` latent KV for this chunk.
        device: NPU device.
        dtype: model dtype (bf16).

    Returns:
        ``(kv_pad, krope_pad)``: ``[S+_ZERO_ROWS, 1, Rkv]`` and
        ``[S+_ZERO_ROWS, 1, Dr]`` padded with ``_ZERO_ROWS`` zero dummy rows.
    """
    kv_comp = chunk_kv[..., :Rkv]
    k_rope = chunk_kv[..., Rkv : Rkv + Dr]
    kv_pad = torch.cat([kv_comp, torch.zeros(_ZERO_ROWS, 1, Rkv, device=device, dtype=dtype)], dim=0)
    krope_pad = torch.cat([k_rope, torch.zeros(_ZERO_ROWS, 1, Dr, device=device, dtype=dtype)], dim=0)
    return kv_pad, krope_pad


def _cp_ring_enabled(seq_ctx: "SequenceContext") -> bool:
    """Whether context-parallel ring attention is active for this batch.

    Requires ``XTUNER_CP_RING=1`` and a sequence-parallel mesh of size > 1.
    The SP axis is reused as the CP ring group (CP replaces the SP all-gather
    of the attention KV with a P2P ring rotation; the small indexer key stays
    all-gathered so the DSA top-k remains global). Default OFF keeps the
    byte-identical all-gather path.

    Args:
        seq_ctx: Sequence context carrying the sequence-parallel mesh.

    Returns:
        True if the CP ring path should be used instead of the SP gather.
    """
    if os.environ.get("XTUNER_CP_RING", "0") != "1":
        return False
    mesh = seq_ctx.sequence_parallel_mesh
    return mesh is not None and mesh.size() > 1


class RingAttentionCP(torch.autograd.Function):
    """Ring-attention over the CP group (genuine per-chunk overlapped ring).

    The single autograd entry: delegates to :func:`forward_true` /
    :func:`backward_true` in this module (per-chunk ``sparse_mode=0`` kernels
    + online-softmax merge, P2P/compute overlap); the forward reads the merged
    activations exactly once and hands the per-hop KV payload through
    ``save_for_backward`` so reentrant activation checkpointing can free it.

    The forward output (``raw_output``) and ``softmax_lse`` match the non-CP
    gather's ``SparseMLAOutputs`` shapes and dtypes, so the rest of the
    DSA-MLA layer is unchanged.
    """

    @staticmethod
    def forward(
        ctx,
        q_states: torch.Tensor,
        kv_local: torch.Tensor,
        topk_indices: torch.Tensor,
        scale: float,
        cp_group: "dist.ProcessGroup | None",
        cp_size: int,
        cp_rank: int,
        seq_ctx: "SequenceContext",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the true-ring forward.

        Args:
            q_states: ``[S, N, Rkv + Dr]`` local query.
            kv_local: ``[S, 1, Rkv + Dr]`` local latent KV.
            topk_indices: ``[S, 1, K]`` global indices (``-1`` invalid).
            scale (float): Softmax scale.
            cp_group: HCCL subgroup of the CP ring.
            cp_size (int): Context-parallel degree.
            cp_rank (int): This rank's position in the CP ring.
            seq_ctx: SequenceContext (packed cu_seq + shard start).

        Returns:
            ``(raw_output, softmax_lse)``: ``[S, N, Rkv]`` native-dtype output
            and ``[S, N]`` float32 log-sum-exp (detached; ``softmax_lse`` is not
            on the training grad path).
        """
        out_m, smax_m, ssum_m, payload = forward_true(
            q_states, kv_local, topk_indices, scale, cp_group, cp_size, cp_rank, seq_ctx
        )
        seq_len, num_heads, _ = q_states.shape
        lse = _parse_lse(smax_m, ssum_m, seq_len, num_heads)
        # ``forward_true`` folds the online-softmax merge in FLOAT32
        # (a bf16 accumulator drifts ~1 ulp per chunk and the Formulation-B
        # backward amplifies it -- CP4 oracle). At ``reach == 1`` (the 256K
        # norm) the two chunks merge straight out of their bf16 RAW buffers
        # tile-by-tile -- bit-identical, no 2 GiB fp32 accumulator; for
        # larger reach an fp32 accumulator is used and the final cast
        # reuses the last chunk's kernel buffer. Either way the output is
        # already model dtype, so the ``.to`` below is a view.
        out_ret = out_m.to(q_states.dtype)
        # The per-hop KV clones ride save_for_backward, NOT a plain ctx
        # attribute: reentrant activation checkpointing only frees packed
        # (saved) tensors, so a ctx attribute would pin 2 chunks x ~19 MB
        # per ring layer (~1.2 GiB across the step at 256K) as dead weight
        # -- the recomputed pass saves FRESH clones for the real backward.
        # The host ints (ring step, global row start) go on ctx.
        ctx.save_for_backward(q_states, topk_indices, out_ret, smax_m, ssum_m, *[p[2] for p in payload])
        ctx.payload_meta = [(p[0], p[1]) for p in payload]
        ctx.scale = scale
        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        ctx.cp_rank = cp_rank
        return out_ret, lse.detach()

    @staticmethod
    def backward(
        ctx,
        grad_out: torch.Tensor,
        grad_lse: "torch.Tensor | None",
    ) -> tuple:
        """True-ring backward (Formulation B)."""
        saved = ctx.saved_tensors
        # Rebuild the (step, global_start, kv) payload from the tensors
        # appended after the 5 common saved slots (the forward packs the
        # KV clones into save_for_backward so AC can free them).
        payload = [(j, gs, saved[5 + i]) for i, (j, gs) in enumerate(ctx.payload_meta)]
        dq, dkv_local = backward_true(
            saved[0],
            saved[1],
            saved[2],
            saved[3],
            saved[4],
            grad_out,
            payload,
            ctx.scale,
            ctx.cp_group,
            ctx.cp_size,
            ctx.cp_rank,
        )
        # 8 inputs: q_states, kv_local, topk_indices, scale, cp_group, cp_size,
        # cp_rank, seq_ctx. Only q_states and kv_local are differentiable.
        return dq, dkv_local, None, None, None, None, None, None
