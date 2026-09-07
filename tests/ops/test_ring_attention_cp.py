# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for context-parallel ring attention (``RingAttentionCP``).

All tests are NPU end-to-end: ``RingAttentionCP.apply`` across a CP4 ring vs
an exact pytorch reference built from the all-gathered KV with the same
global top-k (forward: output + lse; backward: dq + dkv). The reference is
the exact sparse-attention math; the genuine ring masks ``-1`` and excludes
dummies by count-correction, so it matches the reference within the bf16
oracle tolerance. The test constructs a REAL ``SequenceContext`` (the
production input) so the shard-start assert and the ``cu_seq``-driven ring
reach run on the production path.

Uses torch's ``DistributedTestBase`` (a ``MultiProcessTestCase`` that spawns
the NPU ranks itself — invoke directly, NOT under ``torchrun``; torchrun
double-spawns and collides on the HCCL port)::

    XTUNER_TEST_WORLD_SIZE=4 python -m pytest tests/ops/test_ring_attention_cp.py -x --noconftest

Every compare runs the genuine ring (per-chunk kernels + online-softmax
merge, both ``XTUNER_CP_RING_OVERLAP`` settings) against the SAME reference
-- the removed ``chimera`` gather-all relay variant has no tests left.
``TestRingAttentionCPCheckpointed`` additionally runs the true ring under
reentrant activation checkpointing, where the grad-disabled pass 1 stashes the
fold result and the backward recompute must hit the stash (recompute-skip).

``--noconftest`` skips ``tests/conftest.py`` (its eager ``xtuner._testing``
import pulls the float8/triton stack — a pre-existing collection-time issue
unrelated to this op). Dims match ``tests/module/attention/test_npu_sparse_mla_accuracy.py``
(``DIM=576`` — the kernel's hard-coded ``qk_head_dim=512`` + ``rope_dim=64``).
The optional-import guards below only keep this file *collectable* on a
CPU-only runner; every test still requires an NPU (skipped otherwise).
"""

import os

import pytest
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.ops.cp.ring_attention import _FOLD_STASH, _REMAP_CACHE, Dr, RingAttentionCP, RingP2P, Rkv


# Optional NPU runtime + torch dist-test harness: tests skip when either is
# absent (so this file is collectable on a CPU-only CI runner that lacks
# torch_npu / expecttest).
try:
    import torch_npu  # noqa: F401  registers torch.npu / transfers cuda->npu

    _HAS_TORCH_NPU = True
except ModuleNotFoundError:
    _HAS_TORCH_NPU = False
try:
    from torch.testing._internal.common_distributed import DistributedTestBase

    _HAS_DIST_BASE = True
except ModuleNotFoundError:  # expecttest / torch test utils missing on CPU CI
    _HAS_DIST_BASE = False

    class DistributedTestBase:  # type: ignore[no-redef]  fallback base so classes collect
        pass


# GLM-5.2 absorbed-MLA latent dims: kv_lora_rank (512) + qk_rope (64). The fused
# kernel hard-codes qk_head_dim=512, so Rkv/Dr are fixed.
DIM = Rkv + Dr  # 576
SCALE = DIM**-0.5  # matches tests/module/attention/test_npu_sparse_mla_accuracy.py

# Correctness-gate dims (small; this is a math gate, not scale).
S_LOCAL = 4  # per-rank seq len
N_HEADS = 64  # GLM-5.2 num_heads (MHA, num_kv_heads == num_heads)
TOPK = 4  # DSA top-k

# bf16 tolerances (the ring math is exact in float32; bf16 drift comes from the
# fused kernel + the bf16 KV rotation).
FWD_ATOL, FWD_RTOL = 5e-2, 5e-2
BWD_ATOL, BWD_RTOL = 8e-2, 8e-2  # grads accumulate over the K-chunks -> slightly looser


def _npu_available() -> bool:
    return _HAS_TORCH_NPU and torch.npu.is_available()


def _sync() -> None:
    if _npu_available():
        torch.npu.synchronize()


def _init_ring(self) -> tuple:
    """create_pg + a real CP subgroup (mimics ``mesh.get_group()``), memoized.

    HCCL P2P ``isend``/``irecv`` do not transfer data when given the
    ``dist.group.WORLD`` *sentinel* (verified on 910B3 / CANN 9.2 — the ring
    silently fails to rotate, so each rank attends only to its own shard).
    A *real* subgroup — exactly what ``cp_mesh.get_group()`` returns in
    training — works, as does ``group=None`` (default). We build a real
    subgroup here so the test reproduces the production group path.

    Memoized on ``self`` because the overlap-loop variants re-enter the same
    rank process; a second ``create_pg`` raises "initialize the default
    process group twice!". All ranks hit the cache on the same call (the
    loop structure is rank-uniform), so the one-time ``new_group`` collective
    stays matched.
    """
    cached = getattr(self, "_ring_pg", None)
    if cached is None:
        self._set_device()
        self.create_pg("cuda")
        ws = dist.get_world_size()
        rank = dist.get_rank()
        cp_group = dist.new_group(ranks=list(range(ws)))
        device = torch.device("cuda", rank % torch.npu.device_count())
        cached = (cp_group, ws, rank, device)
        self._ring_pg = cached
    return cached


def _make_seq_ctx(rank: int, ws: int, device: torch.device) -> SequenceContext:
    """A REAL SequenceContext for one packed segment of ``ws * S_LOCAL`` tokens.

    Mirrors production CP: the batch is a single packed segment spanning the
    whole global sequence; each rank's shard starts at ``rank * S_LOCAL``
    (what ``split_for_sequence_parallel`` produces). ``RingAttentionCP`` reads
    ``cu_seq_lens_q`` (global segment boundaries -> ring reach) and
    ``_shard_start`` (the shard-start assert + global index rewrite).
    """
    return SequenceContext(
        input_ids=None,
        cu_seq_lens_q=torch.tensor([0, ws * S_LOCAL], dtype=torch.int32),
        cu_seq_lens_k=torch.tensor([0, ws * S_LOCAL], dtype=torch.int32),
        max_length_q=S_LOCAL,
        max_length_k=ws * S_LOCAL,
        device=device,
        shard_start=rank * S_LOCAL,
        shard_size=S_LOCAL,
    )


def _make_inputs(rank: int, ws: int, device: torch.device, dtype: torch.dtype):
    """Per-rank q, kv_local, topk (causal recent-token global indices, int32).

    Query at local position i has global position p = rank*S_LOCAL + i and
    selects the TOPK most-recent tokens [p, p-1, ..., p-TOPK+1] clamped to -1
    when out of range. This mixes in-chunk and out-of-chunk (and -1 dummy)
    indices on every rank, exercising the per-chunk remap + count-correction.
    int32 matches the production DSA top-k dtype.
    """
    torch.manual_seed(rank + 1)
    q = torch.randn(S_LOCAL, N_HEADS, DIM, device=device, dtype=dtype)
    kv_local = torch.randn(S_LOCAL, 1, DIM, device=device, dtype=dtype)
    g0 = rank * S_LOCAL
    topk = torch.empty(S_LOCAL, 1, TOPK, device=device, dtype=torch.int32)
    for i in range(S_LOCAL):
        p = g0 + i
        for k in range(TOPK):
            t = p - k
            topk[i, 0, k] = t if t >= 0 else -1
    return q, kv_local, topk


# Multi-segment packing: 16 global tokens in 4 packed segments (lengths 3/4/4/5).
# Segments straddle rank shards (rank 1 straddles the [3,7)/[7,11) boundary), so
# the ring's reach (rotate past chunks back to the segment start), the
# segment-local top-k remap, and the per-chunk skip are all exercised — none of
# which the single-segment tests above reach (there kv_start == 0 everywhere).
CU_MULTI = [0, 3, 7, 11, 16]


def _make_multi_seg_seq_ctx(rank: int, ws: int, device: torch.device) -> SequenceContext:
    """REAL SequenceContext with 4 packed segments straddling the CP shards.

    Mirrors production variable-length packing: each segment is an independent
    sample (attention is segment-local), and rank shards (``rank * S_LOCAL``)
    cut across segment boundaries, so the ring reach extension (rotating past
    chunks back to the segment start) is required.

    Args:
        rank (int): This rank's position in the CP ring.
        ws (int): CP ring size (shard = ``rank * S_LOCAL``).
        device: NPU device.

    Returns:
        The packed SequenceContext for this rank's shard.
    """
    cu = torch.tensor(CU_MULTI, dtype=torch.int32)
    max_k = max(b - a for a, b in zip(CU_MULTI, CU_MULTI[1:]))
    return SequenceContext(
        input_ids=None,
        cu_seq_lens_q=cu,
        cu_seq_lens_k=cu.clone(),
        max_length_q=S_LOCAL,
        max_length_k=max_k,
        device=device,
        shard_start=rank * S_LOCAL,
        shard_size=S_LOCAL,
    )


def _make_multi_seg_inputs(rank: int, ws: int, device: torch.device, dtype: torch.dtype):
    """Per-rank q/kv/topk with SEGMENT-local causal recent-token top-k.

    Production DSA top-k never crosses a packed-segment boundary (each sample
    attends within itself), so entry ``k`` of query ``p`` in segment ``[s, e)``
    is ``p - k`` when ``>= s`` else ``-1``. This exercises ``-1`` padding
    (short segments) and — on straddling ranks — in-segment indices below
    ``shard_start`` that only the rotated-in past chunks cover.

    Args:
        rank (int): This rank's position in the CP ring.
        ws (int): CP ring size.
        device: NPU device.
        dtype: Model dtype (bf16).

    Returns:
        ``(q, kv_local, topk)``: ``[S_LOCAL, N, DIM]`` bf16 query, ``[S_LOCAL,
        1, DIM]`` bf16 latent KV, ``[S_LOCAL, 1, TOPK]`` int32 global indices.
    """
    torch.manual_seed(rank + 7)
    q = torch.randn(S_LOCAL, N_HEADS, DIM, device=device, dtype=dtype)
    kv_local = torch.randn(S_LOCAL, 1, DIM, device=device, dtype=dtype)
    g0 = rank * S_LOCAL
    topk = torch.full((S_LOCAL, 1, TOPK), -1, dtype=torch.int32, device=device)
    for i in range(S_LOCAL):
        p = g0 + i
        seg_start = max(s for s in CU_MULTI if s <= p)
        for k in range(TOPK):
            if p - k >= seg_start:
                topk[i, 0, k] = p - k
    return q, kv_local, topk


def _gather_kv(kv_local: torch.Tensor, group) -> torch.Tensor:
    """All-gather the per-rank KV shards into the global ascending KV."""
    ws = dist.get_world_size(group)
    kv_list = [torch.empty_like(kv_local) for _ in range(ws)]
    dist.all_gather(kv_list, kv_local.contiguous(), group=group)
    return torch.cat(kv_list, dim=0)


def _pytorch_forward(q: torch.Tensor, kv_full: torch.Tensor, topk: torch.Tensor, scale: float):
    """Exact pytorch full-attention reference (all-gathered KV, same global top-k)."""
    ind = topk.squeeze(1)
    safe = ind.clamp(min=0).to(torch.long)
    valid = ind != -1
    gkv = kv_full[:, 0, :][safe].float()  # [S, K, DIM]
    qf = q.float()
    s = torch.einsum("snd,skd->snk", qf, gkv) * scale
    s = s.masked_fill(~valid[:, None, :], float("-inf"))
    out = torch.einsum("snk,skd->snd", torch.softmax(s, -1), gkv[..., :Rkv])
    lse = torch.logsumexp(s, -1)
    return out, lse


def _pytorch_backward(q, kv_full, topk, out, lse, g_out, g_lse, scale, ws, rank, group):
    """Exact pytorch backward reference (all-gathered KV).

    The cross-rank d_kv sum uses ``all_gather`` + local sum (NOT ``all_reduce``:
    eager HCCL ``all_reduce`` deadlocks on CP subgroups under CANN 9.2 — the
    same reason production routes the ring's d_kv through reduce-scatter).
    """
    ind = topk.squeeze(1)
    safe = ind.clamp(min=0).to(torch.long)
    valid = ind != -1
    gkv = kv_full[:, 0, :][safe].float()
    qf = q.float()
    s = torch.einsum("snd,skd->snk", qf, gkv) * scale
    s = s.masked_fill(~valid[:, None, :], float("-inf"))
    p = torch.exp(s - lse[:, :, None])
    gv = torch.einsum("snd,skd->snk", g_out, gkv[..., :Rkv])
    odg = (g_out * out).sum(-1)
    d_s = p * (gv - odg[:, :, None] + g_lse[:, :, None])
    dq = torch.einsum("snk,skd->snd", d_s, gkv) * scale
    dk = torch.einsum("snk,snd->skd", d_s, qf) * scale
    dv = torch.einsum("snk,snd->skd", p, g_out)
    dkv = dk.clone()
    dkv[..., :Rkv] += dv
    dkv_full = torch.zeros(kv_full.shape[0], Rkv + Dr, device=q.device, dtype=torch.float32)
    dkv_full.index_add_(0, safe.reshape(-1), dkv.reshape(-1, Rkv + Dr))
    if ws > 1:
        parts = [torch.empty_like(dkv_full) for _ in range(ws)]
        dist.all_gather(parts, dkv_full.contiguous(), group=group)
        total = torch.zeros_like(dkv_full)
        for part in parts:
            total += part
        dkv_full = total  # fresh buffer: dkv_full already holds own partial -> no double count
    return dq, dkv_full[rank * S_LOCAL : (rank + 1) * S_LOCAL].unsqueeze(1)


@pytest.mark.skipif(not (_npu_available() and _HAS_DIST_BASE), reason="requires NPU + DistributedTestBase")
@pytest.mark.gpu
class TestRingAttentionCPForward(DistributedTestBase):
    """Ring-attention forward (CP4) vs the pytorch full-attention reference."""

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "4"))

    def _set_device(self) -> None:
        if self.rank >= 0:
            torch.npu.set_device(self.rank % torch.npu.device_count())

    def _check_forward(self, overlap: int = 1) -> None:
        os.environ["XTUNER_CP_RING_OVERLAP"] = str(overlap)
        cp_group, ws, rank, device = _init_ring(self)

        q, kv_local, topk = _make_inputs(rank, ws, device, torch.bfloat16)
        seq_ctx = _make_seq_ctx(rank, ws, device)

        out, lse = RingAttentionCP.apply(q, kv_local, topk, SCALE, cp_group, ws, rank, seq_ctx)

        # Reference: all-gather KV, run the exact pytorch sparse attention.
        # NB: RingAttentionCP.forward no longer mutates kv_local (the ring
        # buffers are cloned), so gathering after the ring is safe.
        kv_full = _gather_kv(kv_local, cp_group)
        out_ref, lse_ref = _pytorch_forward(q, kv_full, topk, SCALE)
        _sync()

        torch.testing.assert_close(out.float(), out_ref, atol=FWD_ATOL, rtol=FWD_RTOL)
        torch.testing.assert_close(lse.float(), lse_ref, atol=FWD_ATOL, rtol=FWD_RTOL)

    def test_forward_matches_pytorch_reference(self):
        """True ring (per-chunk kernels + online-softmax merge), overlap ON and OFF.

        The recent-token top-k makes every step past the local chunk a mix of
        real and all-dummy steps (kernels skipped), so this exercises both the
        merge numerics and the host-decided skip on the ring's P2P rotation.
        """
        for overlap in (1, 0):
            self._check_forward(overlap)
            dist.barrier()

    def _check_finite(self) -> None:
        cp_group, ws, rank, device = _init_ring(self)

        q, kv_local, topk = _make_inputs(rank, ws, device, torch.bfloat16)
        seq_ctx = _make_seq_ctx(rank, ws, device)
        out, lse = RingAttentionCP.apply(q, kv_local, topk, SCALE, cp_group, ws, rank, seq_ctx)
        _sync()

        assert torch.isfinite(out).all(), "forward output has nan/inf"
        assert torch.isfinite(lse).all(), "forward lse has nan/inf"

    def test_forward_is_finite(self):
        self._check_finite()


@pytest.mark.skipif(not (_npu_available() and _HAS_DIST_BASE), reason="requires NPU + DistributedTestBase")
@pytest.mark.gpu
class TestRingAttentionCPBackward(DistributedTestBase):
    """Ring-attention backward (CP4, exact pytorch recompute) vs pytorch reference.

    The lse is detached in the forward (metrics-only, matching the non-CP path's
    ``mla.py:640`` detach and real training, where the lse is not in the loss),
    so ``grad_lse = 0`` and the reference uses ``g_lse = 0`` (the standard
    attention backward with no lse-grad path).
    """

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "4"))

    def _set_device(self) -> None:
        if self.rank >= 0:
            torch.npu.set_device(self.rank % torch.npu.device_count())

    def _check_backward(self, overlap: int = 1) -> None:
        os.environ["XTUNER_CP_RING_OVERLAP"] = str(overlap)
        cp_group, ws, rank, device = _init_ring(self)

        q, kv_local, topk = _make_inputs(rank, ws, device, torch.bfloat16)
        seq_ctx = _make_seq_ctx(rank, ws, device)
        # Use the SAME pytorch forward (out, lse) for both paths so the backward
        # math is isolated from bf16 forward drift (mirrors the probe's MATH gate).
        kv_full = _gather_kv(kv_local, cp_group)
        out_ref, lse_ref = _pytorch_forward(q, kv_full, topk, SCALE)

        g_out = torch.randn_like(out_ref, dtype=torch.float32)
        # lse is detached in the forward (metrics-only), so grad_lse is zero --
        # the reference uses g_lse = 0 (the standard backward, no lse-grad path,
        # matching non-CP and real training where the lse is not in the loss).
        g_lse = torch.zeros(S_LOCAL, N_HEADS, device=device, dtype=torch.float32)

        dq_ref, dkv_ref = _pytorch_backward(
            q, kv_full, topk, out_ref, lse_ref, g_out, g_lse, SCALE, ws, rank, cp_group
        )

        # End-to-end: kernel forward + autograd backward.
        q2 = q.clone().requires_grad_(True)
        kv2 = kv_local.clone().requires_grad_(True)
        out_cp, _lse = RingAttentionCP.apply(q2, kv2, topk, SCALE, cp_group, ws, rank, seq_ctx)
        # lse is detached -> not in the loss (only the merged output is consumed).
        loss = (out_cp.float() * g_out).sum()
        loss.backward()
        _sync()

        torch.testing.assert_close(q2.grad.float(), dq_ref, atol=BWD_ATOL, rtol=BWD_RTOL)
        torch.testing.assert_close(kv2.grad.float(), dkv_ref, atol=BWD_ATOL, rtol=BWD_RTOL)

    def test_backward_matches_pytorch_reference(self):
        """Formulation B across a real CP4 P2P rotation, overlap ON and OFF."""
        for overlap in (1, 0):
            self._check_backward(overlap)
            dist.barrier()

    def _check_grads_finite(self) -> None:
        cp_group, ws, rank, device = _init_ring(self)

        q, kv_local, topk = _make_inputs(rank, ws, device, torch.bfloat16)
        seq_ctx = _make_seq_ctx(rank, ws, device)
        q.requires_grad_(True)
        kv_local.requires_grad_(True)
        out, _lse = RingAttentionCP.apply(q, kv_local, topk, SCALE, cp_group, ws, rank, seq_ctx)
        # lse is detached (metrics-only); only the merged output is in the loss.
        loss = out.float().sum()
        loss.backward()
        _sync()

        assert torch.isfinite(q.grad).all(), "dq has nan/inf"
        assert torch.isfinite(kv_local.grad).all(), "dkv has nan/inf"

    def test_gradients_are_finite(self):
        self._check_grads_finite()

    def test_backward_grad_tile_invariance(self):
        """Row-tiling the grad kernel (``XTUNER_CP_RING_GRAD_TILES``) must not move
        the oracle: the attention grad is exact per query row, so tiles=1 (off),
        2 (production default -- the 256K workspace fit) and 3 (uneven splits on
        ``S_LOCAL=4`` -> tiles of 1/1/2) all match the pytorch reference."""
        for tiles in ("1", "2", "3"):
            os.environ["XTUNER_CP_RING_GRAD_TILES"] = tiles
            try:
                self._check_backward()
            finally:
                os.environ.pop("XTUNER_CP_RING_GRAD_TILES", None)
            dist.barrier()


@pytest.mark.skipif(not (_npu_available() and _HAS_DIST_BASE), reason="requires NPU + DistributedTestBase")
@pytest.mark.gpu
class TestRingAttentionCPCheckpointed(DistributedTestBase):
    """Reentrant-AC recompute-skip stash (CP4): the replay must HIT the fold
    stash, restore bit-identical activations, and match the pytorch reference.

    The classes above call ``RingAttentionCP.apply`` directly, so the stash is
    never POPULATED (only an AC pass 1 stashes) -- the mechanism that keeps
    true ring under the 256K pool ceiling (run281/284/285: the backward-window
    ring REPLAY, not the forward fold, overflowed the allocator) would go
    completely untested. Training wraps every layer in reentrant activation
    checkpointing: pass 1 (running outside the autograd engine -- grad mode is
    NOT usable as the discriminator, ``Function.apply`` runs the user forward
    with grad disabled in both passes) stashes ``(out, smax, ssum, payload)``
    to pinned CPU, the backward recompute (running inside the engine) pops it
    and returns without running a single ring op. These tests pin that
    contract -- including the pass-detection itself (a stash entry only appears
    if pass 1 really was the forward pass), the LIFO pairing when DSA shares
    one top-k object across regions, and the ``XTUNER_CP_RECOMPUTE_SKIP=0``
    full-replay fallback.
    """

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "4"))

    def _set_device(self) -> None:
        if self.rank >= 0:
            torch.npu.set_device(self.rank % torch.npu.device_count())

    def _stack_depth(self, topk: torch.Tensor) -> int:
        """Live stash entries for THIS exact top-k tensor (0 when none/id reused)."""
        entry = _FOLD_STASH.get(id(topk))
        return 0 if entry is None or entry[0]() is not topk else len(entry[2])

    def _make_grad_out(self, device: torch.device, seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        return torch.randn(S_LOCAL, N_HEADS, Rkv, device=device, dtype=torch.float32)

    def _reference(self, q, kv_local, topk, cp_group, ws, rank, device, seed):
        """Exact pytorch fwd+bwd reference (mirrors ``_check_backward``)."""
        kv_full = _gather_kv(kv_local, cp_group)
        out_ref, lse_ref = _pytorch_forward(q, kv_full, topk, SCALE)
        g_out = self._make_grad_out(device, seed)
        g_lse = torch.zeros(S_LOCAL, N_HEADS, device=device, dtype=torch.float32)
        dq_ref, dkv_ref = _pytorch_backward(
            q, kv_full, topk, out_ref, lse_ref, g_out, g_lse, SCALE, ws, rank, cp_group
        )
        return g_out, dq_ref, dkv_ref

    def test_checkpointed_backward_matches_reference(self):
        os.environ["XTUNER_CP_RECOMPUTE_SKIP"] = "1"
        cp_group, ws, rank, device = _init_ring(self)

        q, kv_local, topk = _make_inputs(rank, ws, device, torch.bfloat16)
        seq_ctx = _make_seq_ctx(rank, ws, device)
        g_out, dq_ref, dkv_ref = self._reference(q, kv_local, topk, cp_group, ws, rank, device, 1234)

        q2 = q.clone().requires_grad_(True)
        kv2 = kv_local.clone().requires_grad_(True)

        def region(q_in, kv_in):
            return RingAttentionCP.apply(q_in, kv_in, topk, SCALE, cp_group, ws, rank, seq_ctx)[0]

        out = checkpoint(region, q2, kv2, use_reentrant=True)
        # AC pass 1 (grad DISABLED) must have stashed exactly one fold result.
        assert self._stack_depth(topk) == 1, "grad-disabled pass did not stash"
        loss = (out.float() * g_out).sum()  # out's bytes die at the next stash flush
        loss.backward()
        _sync()
        # Only _stash_take pops: an empty stack proves the recompute HIT the
        # stash (a miss would replay the ring and leave depth == 1).
        assert self._stack_depth(topk) == 0, "recompute pass did not restore the stash"
        torch.testing.assert_close(q2.grad.float(), dq_ref, atol=BWD_ATOL, rtol=BWD_RTOL)
        torch.testing.assert_close(kv2.grad.float(), dkv_ref, atol=BWD_ATOL, rtol=BWD_RTOL)

    def test_shared_topk_lifo_pairing(self):
        """One top-k shared by TWO checkpoint regions (DSA ``index_topk_freq``
        layer sharing): pass 1 pushes in forward order, backward recomputes in
        REVERSE, so LIFO must hand each replay its own activations. Region B is
        chained downstream of A (its query carries a zero-valued gradient link
        through A's output), so the engine visits B first deterministically; a
        FIFO pairing would restore A's activations into B's autograd node and
        the grads would miss the per-region references.
        """
        os.environ["XTUNER_CP_RECOMPUTE_SKIP"] = "1"
        cp_group, ws, rank, device = _init_ring(self)

        qA, kvA, topk = _make_inputs(rank, ws, device, torch.bfloat16)
        torch.manual_seed(rank + 50)  # second "layer": same top-k object, new q/kv
        qB = torch.randn(S_LOCAL, N_HEADS, DIM, device=device, dtype=torch.bfloat16)
        kvB = torch.randn(S_LOCAL, 1, DIM, device=device, dtype=torch.bfloat16)
        seq_ctx = _make_seq_ctx(rank, ws, device)
        gA, dqA_ref, dkvA_ref = self._reference(qA, kvA, topk, cp_group, ws, rank, device, 11)
        gB, dqB_ref, dkvB_ref = self._reference(qB, kvB, topk, cp_group, ws, rank, device, 22)

        qA2, kvA2 = qA.clone().requires_grad_(True), kvA.clone().requires_grad_(True)
        qB2, kvB2 = qB.clone().requires_grad_(True), kvB.clone().requires_grad_(True)

        def region_a(q_in, kv_in):
            return RingAttentionCP.apply(q_in, kv_in, topk, SCALE, cp_group, ws, rank, seq_ctx)[0]

        def region_b(q_in, kv_in, link):
            # Zero-valued gradient link: numerically identity (exact bf16
            # zeros), but makes B strictly downstream of A in the graph.
            q_eff = q_in + link[..., :1].to(q_in.dtype) * 0.0
            return RingAttentionCP.apply(q_eff, kv_in, topk, SCALE, cp_group, ws, rank, seq_ctx)[0]

        outA = checkpoint(region_a, qA2, kvA2, use_reentrant=True)
        ta = outA.float()  # consumed before B's stash can free outA's storage
        assert self._stack_depth(topk) == 1
        outB = checkpoint(region_b, qB2, kvB2, ta, use_reentrant=True)
        assert self._stack_depth(topk) == 2, "second region did not stash onto the shared stack"
        loss = (ta * gA).sum() + (outB.float() * gB).sum()
        loss.backward()
        _sync()
        assert self._stack_depth(topk) == 0, "both recomputes must pop the shared LIFO stack"
        torch.testing.assert_close(qA2.grad.float(), dqA_ref, atol=BWD_ATOL, rtol=BWD_RTOL)
        torch.testing.assert_close(kvA2.grad.float(), dkvA_ref, atol=BWD_ATOL, rtol=BWD_RTOL)
        torch.testing.assert_close(qB2.grad.float(), dqB_ref, atol=BWD_ATOL, rtol=BWD_RTOL)
        torch.testing.assert_close(kvB2.grad.float(), dkvB_ref, atol=BWD_ATOL, rtol=BWD_RTOL)

    def test_checkpointed_fallback_without_stash(self):
        """``XTUNER_CP_RECOMPUTE_SKIP=0``: nothing is stashed, the recompute
        replays the full ring, and the grads must STILL match the reference --
        the pre-stash behaviour stays a correct escape hatch (and an id-collision
        drain mid-replay takes exactly this path).
        """
        os.environ["XTUNER_CP_RECOMPUTE_SKIP"] = "0"
        try:
            cp_group, ws, rank, device = _init_ring(self)
            q, kv_local, topk = _make_inputs(rank, ws, device, torch.bfloat16)
            seq_ctx = _make_seq_ctx(rank, ws, device)
            g_out, dq_ref, dkv_ref = self._reference(q, kv_local, topk, cp_group, ws, rank, device, 1234)

            q2 = q.clone().requires_grad_(True)
            kv2 = kv_local.clone().requires_grad_(True)

            def region(q_in, kv_in):
                return RingAttentionCP.apply(q_in, kv_in, topk, SCALE, cp_group, ws, rank, seq_ctx)[0]

            out = checkpoint(region, q2, kv2, use_reentrant=True)
            assert self._stack_depth(topk) == 0, "stash disabled but pass 1 stashed"
            loss = (out.float() * g_out).sum()
            loss.backward()
            _sync()
            assert self._stack_depth(topk) == 0
            torch.testing.assert_close(q2.grad.float(), dq_ref, atol=BWD_ATOL, rtol=BWD_RTOL)
            torch.testing.assert_close(kv2.grad.float(), dkv_ref, atol=BWD_ATOL, rtol=BWD_RTOL)
        finally:
            os.environ.pop("XTUNER_CP_RECOMPUTE_SKIP", None)


@pytest.mark.skipif(not (_npu_available() and _HAS_DIST_BASE), reason="requires NPU + DistributedTestBase")
@pytest.mark.gpu
class TestRingAttentionCPMultiSegment(DistributedTestBase):
    """Ring attention (CP4) vs reference on MULTI-segment packed batches.

    Production batches pack variable-length samples whose boundaries straddle
    rank shards, so the ring reach must extend back to the segment start
    (``kv_start < shard_start``) and the segment-local top-k must be remapped
    per chunk. The single-segment classes above never exercise that extension.
    """

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "4"))

    def _set_device(self) -> None:
        if self.rank >= 0:
            torch.npu.set_device(self.rank % torch.npu.device_count())

    def _check_multi_seg_forward(self, overlap: int = 1) -> None:
        os.environ["XTUNER_CP_RING_OVERLAP"] = str(overlap)
        cp_group, ws, rank, device = _init_ring(self)

        q, kv_local, topk = _make_multi_seg_inputs(rank, ws, device, torch.bfloat16)
        seq_ctx = _make_multi_seg_seq_ctx(rank, ws, device)

        out, lse = RingAttentionCP.apply(q, kv_local, topk, SCALE, cp_group, ws, rank, seq_ctx)

        kv_full = _gather_kv(kv_local, cp_group)
        out_ref, lse_ref = _pytorch_forward(q, kv_full, topk, SCALE)
        _sync()

        torch.testing.assert_close(out.float(), out_ref, atol=FWD_ATOL, rtol=FWD_RTOL)
        torch.testing.assert_close(lse.float(), lse_ref, atol=FWD_ATOL, rtol=FWD_RTOL)

    def test_multi_seg_forward_matches_pytorch_reference(self):
        """True ring on straddling segments: host reach bound + per-chunk skip.

        ``_ring_reach`` (global max segment length, pure host) caps the hops at
        2 of 3; the forward asserts no real top-k lives past the bound, so the
        packed segment structure is what the skip path is judged against.
        """
        for overlap in (1, 0):
            self._check_multi_seg_forward(overlap)
            dist.barrier()

    def _check_multi_seg_backward(self, overlap: int = 1) -> None:
        os.environ["XTUNER_CP_RING_OVERLAP"] = str(overlap)
        cp_group, ws, rank, device = _init_ring(self)

        q, kv_local, topk = _make_multi_seg_inputs(rank, ws, device, torch.bfloat16)
        seq_ctx = _make_multi_seg_seq_ctx(rank, ws, device)

        kv_full = _gather_kv(kv_local, cp_group)
        out_ref, lse_ref = _pytorch_forward(q, kv_full, topk, SCALE)
        g_out = torch.randn_like(out_ref, dtype=torch.float32)
        g_lse = torch.zeros(S_LOCAL, N_HEADS, device=device, dtype=torch.float32)
        dq_ref, dkv_ref = _pytorch_backward(
            q, kv_full, topk, out_ref, lse_ref, g_out, g_lse, SCALE, ws, rank, cp_group
        )

        q2 = q.clone().requires_grad_(True)
        kv2 = kv_local.clone().requires_grad_(True)
        out_cp, _lse = RingAttentionCP.apply(q2, kv2, topk, SCALE, cp_group, ws, rank, seq_ctx)
        loss = (out_cp.float() * g_out).sum()
        loss.backward()
        _sync()

        torch.testing.assert_close(q2.grad.float(), dq_ref, atol=BWD_ATOL, rtol=BWD_RTOL)
        torch.testing.assert_close(kv2.grad.float(), dkv_ref, atol=BWD_ATOL, rtol=BWD_RTOL)

    def test_multi_seg_backward_matches_pytorch_reference(self):
        """Formulation B + reduce-scatter on straddling segments, overlap ON/OFF."""
        for overlap in (1, 0):
            self._check_multi_seg_backward(overlap)
            dist.barrier()


# ---------------------------------------------------------------------------
# Bucketed row-compaction fixtures. Both layouts keep max segment length ==
# S_LOCAL so the ring reach is 1 (the deferred-fold path -- the 256K norm), and
# use segment-local causal recent top-k. Because the segments straddle the
# shard boundaries by 1/2 tokens, EVERY rank's step 0 (local) chunk is fully
# active (each row references itself -> na == S -> never a plan candidate),
# EXACTLY one neighbour step (j == 1) is sparse (only the segment-head rows
# reach back), and the remaining steps are empty -- precisely the Pass-2
# single-candidate shape under which the row-compaction plan attaches (and it
# must never attach to step 0).
# ---------------------------------------------------------------------------
CU_SKEW_1 = [0, 1, 5, 9, 13, 16]  # neighbour m == 1; its single row has count == k (ndummy 0)
CU_SKEW_2 = [0, 2, 6, 10, 14, 16]  # neighbour m == 2; tiles == 2 -> qlens == [2, 1, 1]


def _make_skew_seq_ctx(rank: int, ws: int, device: torch.device, cu: list) -> SequenceContext:
    """REAL SequenceContext for a skew layout (max segment length == S_LOCAL)."""
    seg_max = max(b - a for a, b in zip(cu, cu[1:]))
    assert seg_max == S_LOCAL, "skew fixtures must pin reach == 1"
    return SequenceContext(
        input_ids=None,
        cu_seq_lens_q=torch.tensor(cu, dtype=torch.int32),
        cu_seq_lens_k=torch.tensor(cu, dtype=torch.int32),
        max_length_q=S_LOCAL,
        max_length_k=seg_max,
        device=device,
        shard_start=rank * S_LOCAL,
        shard_size=S_LOCAL,
    )


def _make_skew_inputs(rank: int, ws: int, device: torch.device, dtype, cu: list, seed: int):
    """Segment-local causal recent top-k over a skew layout (int32 globals)."""
    torch.manual_seed(seed + rank)
    q = torch.randn(S_LOCAL, N_HEADS, DIM, device=device, dtype=dtype)
    kv_local = torch.randn(S_LOCAL, 1, DIM, device=device, dtype=dtype)
    g0 = rank * S_LOCAL
    topk = torch.full((S_LOCAL, 1, TOPK), -1, dtype=torch.int32, device=device)
    for i in range(S_LOCAL):
        p = g0 + i
        seg_start = max(c for c in cu if c <= p)
        for k in range(TOPK):
            if p - k >= seg_start:
                topk[i, 0, k] = p - k
    return q, kv_local, topk


def _cached_all_idx(topk: torch.Tensor):
    """The cached Pass-2 list for THIS live top-k tensor (None on miss)."""
    entry = _REMAP_CACHE.get(id(topk))
    if entry is None or entry[0]() is not topk:
        return None
    return entry[5]


@pytest.mark.skipif(not (_npu_available() and _HAS_DIST_BASE), reason="requires NPU + DistributedTestBase")
@pytest.mark.gpu
class TestRingAttentionCPBucketed(DistributedTestBase):
    """Count-bucketed row compaction (``XTUNER_CP_RING_BUCKET=1``, CP4).

    Fwd + bwd vs the same pytorch reference the un-bucketed classes use, on
    the skew fixtures (single sparse neighbour step -> the plan attaches at
    step 1 only). Whitebox pin: the cached Pass-2 list must carry exactly one
    plan (rank 0: none) matching the fixture shape, and the fwd/bwd must stay
    inside the 8e-2 oracle tolerance (dkv with the row-magnitude-scaled atol
    of ``_assert_dkv_row_scaled``) -- the shrink is not bit-identical (the
    fold stats shift per the module docstring) but is oracle-exact. The
    ``BUCKET=0`` control pins the fixture itself, so any drift is attributable
    to the plan path alone. Each check runs TWO iterations (second reusing the
    SAME top-k object) so a plan leaked across a step boundary (stale cached
    tensors / wrong reuse) would move the second iteration's grads.
    """

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "4"))

    def _set_device(self) -> None:
        if self.rank >= 0:
            torch.npu.set_device(self.rank % torch.npu.device_count())

    def _assert_dkv_row_scaled(self, got: torch.Tensor, ref: torch.Tensor) -> None:
        """dkv check with a row-magnitude-scaled atol.

        The fused grad kernel rounds each dkv row to the bf16 grid of the
        ROW's magnitude (~ulp(40) == 0.25), so a near-zero element inside a
        large row carries one-ulp absolute noise the flat 8e-2 atol cannot
        absorb. The skew fixtures make this unavoidable: single-real rows
        have p == 1 exactly, so their dkv rows reach |20-40| by construction.
        The ``BUCKET=0`` control shows the identical drift, i.e. it is
        kernel quantization -- not the plan path. Systematic errors (dropped
        or swapped rows) are O(row magnitude) and still violate this bound.
        """
        atol = BWD_ATOL + (2.0**-8) * ref.abs().amax(dim=-1, keepdim=True)
        diff = (got.float() - ref).abs()
        bad = torch.nonzero(diff > atol + BWD_RTOL * ref.abs())
        assert bad.numel() == 0, f"dkv out of row-scaled tolerance at {bad[:4].tolist()}"

    def _assert_plan_shape(self, topk, bucket: str, rank: int, expect_m: int) -> None:
        all_idx = _cached_all_idx(topk)
        assert all_idx is not None, "remap cache disabled or evicted mid-check"
        plans = [(j, e[3]) for j, e in enumerate(all_idx) if e[3] is not None]
        if bucket == "0" or rank == 0:
            assert plans == [], f"unexpected plan on rank {rank} (bucket={bucket}): {[j for j, _ in plans]}"
            return
        assert [j for j, _ in plans] == [1], f"expected exactly one plan at step 1: {plans}"
        plan = plans[0][1]
        assert plan.m == expect_m and plan.rows.tolist() == list(range(expect_m))
        assert plan.idx.shape[0] == expect_m and plan.idx.dtype == torch.int32
        assert plan.ndummy.shape == (expect_m,)
        if expect_m == 2:
            tiles = 2 if int(os.environ.get("XTUNER_CP_RING_GRAD_TILES", "2")) == 2 else 1
            assert plan.tiles == tiles, (plan.tiles, tiles)
            assert plan.qlens.tolist() == ([2, 1, 1] if tiles == 2 else [2, 2])

    def _check_bucketed(self, cu: list, bucket: str, tiles: str, expect_m: int) -> None:
        os.environ["XTUNER_CP_RING_BUCKET"] = bucket
        os.environ["XTUNER_CP_RING_GRAD_TILES"] = tiles
        os.environ.pop("XTUNER_CP_RING_BUCKET_N", None)
        os.environ.pop("XTUNER_CP_RING_BUCKET_COST", None)
        try:
            cp_group, ws, rank, device = _init_ring(self)
            q, kv_local, topk = _make_skew_inputs(rank, ws, device, torch.bfloat16, cu, 4000)
            seq_ctx = _make_skew_seq_ctx(rank, ws, device, cu)
            kv_full = _gather_kv(kv_local, cp_group)
            out_ref, lse_ref = _pytorch_forward(q, kv_full, topk, SCALE)
            torch.manual_seed(5000)
            g_out = torch.randn_like(out_ref, dtype=torch.float32)
            g_lse = torch.zeros(S_LOCAL, N_HEADS, device=device, dtype=torch.float32)
            dq_ref, dkv_ref = _pytorch_backward(
                q, kv_full, topk, out_ref, lse_ref, g_out, g_lse, SCALE, ws, rank, cp_group
            )

            q2 = q.clone().requires_grad_(True)
            kv2 = kv_local.clone().requires_grad_(True)
            out_cp, lse_cp = RingAttentionCP.apply(q2, kv2, topk, SCALE, cp_group, ws, rank, seq_ctx)
            self._assert_plan_shape(topk, bucket, rank, expect_m)
            torch.testing.assert_close(out_cp.float(), out_ref, atol=FWD_ATOL, rtol=FWD_RTOL)
            torch.testing.assert_close(lse_cp.float(), lse_ref, atol=FWD_ATOL, rtol=FWD_RTOL)
            loss = (out_cp.float() * g_out).sum()
            loss.backward()
            _sync()
            torch.testing.assert_close(q2.grad.float(), dq_ref, atol=BWD_ATOL, rtol=BWD_RTOL)
            self._assert_dkv_row_scaled(kv2.grad, dkv_ref)

            # Iteration 2: a NEW top-k object (new data -> new reference) run
            # back-to-back; a stale plan/cache leak would move the grads.
            q, kv_local, topk = _make_skew_inputs(rank, ws, device, torch.bfloat16, cu, 4100)
            kv_full = _gather_kv(kv_local, cp_group)
            out_ref, lse_ref = _pytorch_forward(q, kv_full, topk, SCALE)
            dq_ref, dkv_ref = _pytorch_backward(
                q, kv_full, topk, out_ref, lse_ref, g_out, g_lse, SCALE, ws, rank, cp_group
            )
            q3 = q.clone().requires_grad_(True)
            kv3 = kv_local.clone().requires_grad_(True)
            out3, _ = RingAttentionCP.apply(q3, kv3, topk, SCALE, cp_group, ws, rank, seq_ctx)
            self._assert_plan_shape(topk, bucket, rank, expect_m)
            (out3.float() * g_out).sum().backward()
            _sync()
            torch.testing.assert_close(q3.grad.float(), dq_ref, atol=BWD_ATOL, rtol=BWD_RTOL)
            self._assert_dkv_row_scaled(kv3.grad, dkv_ref)
        finally:
            os.environ.pop("XTUNER_CP_RING_BUCKET", None)
            os.environ.pop("XTUNER_CP_RING_GRAD_TILES", None)
            dist.barrier()

    def test_bucket_off_baseline_on_skew_fixtures(self):
        """The fixtures themselves are oracle-clean on the un-bucketed path."""
        for cu, m in ((CU_SKEW_1, 1), (CU_SKEW_2, 2)):
            self._check_bucketed(cu, "0", "2", m)

    def test_bucket_m1_tile_collapse(self):
        """CU_SKEW_1: m == 1 <= S//4 -> the plan collapses grad tiles to 1.

        GRAD_TILES is then irrelevant (tiles == 1 either way) -- run one
        setting; the reference decides the fwd subset-fold and the single
        grad-tile path.
        """
        self._check_bucketed(CU_SKEW_1, "1", "1", 1)
        self._check_bucketed(CU_SKEW_1, "1", "2", 1)

    def test_bucket_m2_grad_tiles(self):
        """CU_SKEW_2: m == 2 > S//4 -> tiles follow GRAD_TILES (1 vs uneven 2)."""
        for tiles in ("1", "2"):
            self._check_bucketed(CU_SKEW_2, "1", tiles, 2)


@pytest.mark.skipif(not (_npu_available() and _HAS_DIST_BASE), reason="requires NPU + DistributedTestBase")
@pytest.mark.gpu
class TestRingP2PGlobalRank(DistributedTestBase):
    """Regression: ``RingP2P`` must convert group-local ranks to GLOBAL ranks.

    ``dist.isend``/``irecv`` take *global* ranks for ``dst``/``src``, but the
    ring topology is group-local (``0..cp_size-1``). The production CP mesh is
    a DeviceMesh submesh that does NOT start at global rank 0 — e.g. the SP4
    submesh for the last data-parallel group is ranks ``{12,13,14,15}`` of a
    16-rank world, so local ``0`` != global ``12``. Passing local-as-global
    there makes ``irecv`` raise ``"Global rank N is not part of group"`` (the
    13B CP4 smoke hit exactly this). This test builds a 2-rank subgroup
    ``{ws-2, ws-1}`` inside a ``ws>=4`` world — local ``0/1`` != global
    ``ws-2/ws-1`` — so the unfixed code crashes and the fixed code rotates
    correctly. (The other NPU classes use ``range(ws)`` where local==global,
    which hid the bug.)
    """

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "4"))

    def _set_device(self) -> None:
        if self.rank >= 0:
            torch.npu.set_device(self.rank % torch.npu.device_count())

    def test_ring_rotates_with_offset_subgroup(self):
        self._set_device()
        self.create_pg("cuda")
        ws = dist.get_world_size()
        rank = dist.get_rank()
        device = torch.device("cuda", rank % torch.npu.device_count())

        if ws < 4:
            self.skipTest("needs ws>=4 so the offset subgroup excludes low global ranks")
        # Top-2 global ranks -> cp_size 2; local 0/1 map to global ws-2/ws-1
        # (local 1 as global 1 is NOT in {ws-2, ws-1} for ws>=4 -> the crash).
        ring_ranks = [ws - 2, ws - 1]
        cp_group = dist.new_group(ranks=ring_ranks)
        dist.barrier()  # collective: every rank must reach this

        if rank not in ring_ranks:
            return  # stand-by rank (e.g. 0, 1 for ws=4)

        cp_rank = rank - ring_ranks[0]  # global ws-2 -> 0, global ws-1 -> 1
        cp_size = len(ring_ranks)
        ring = RingP2P(cp_size, cp_rank, cp_group)

        send_buf = torch.tensor([float(rank)], device=device, dtype=torch.float32)
        recv_buf = torch.empty_like(send_buf)
        ring.step(send_buf, recv_buf)
        _sync()

        # A size-2 ring: each rank's next == prev == the single neighbor, so
        # after one step each rank holds the OTHER rank's token.
        other = ring_ranks[1 - cp_rank]
        torch.testing.assert_close(recv_buf, torch.tensor([float(other)], device=device))
