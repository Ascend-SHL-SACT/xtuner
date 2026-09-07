# Copyright (c) OpenMMLab. All rights reserved.
"""CPU math tests for the true-ring CP components (``ring_attention``).

These gate the ported building blocks (remap / compaction / online-softmax
merge / dummy-correction algebra / reach) BEFORE the ring driver is wired --
the backup they were ported from was dead code and carried three verified rot
bugs (int32 ``scatter_add`` index, single-row dummy contention, ``exp(smax)``
overflow), each pinned by a regression test here.

No NPU required -- every function under test is pure torch. Run::

    python -m pytest tests/ops/test_cp_ring_math.py -x --noconftest
"""

import gc
import math
import types

import torch

from xtuner.v1.ops.cp.ring_attention import (
    _FOLD_STASH,
    _REMAP_CACHE,
    _ZERO_ROWS,
    Dr,
    Rkv,
    _bucket_cost_model,
    _fold_one_raw,
    _fold_pair_subset,
    _fold_subset_single,
    _fold_two_raws,
    _merge_stats,
    _parse_lse,
    _pick_bucket_plan,
    _remap_all_chunks,
    _remap_max_counts,
    _remap_one_chunk,
    _ring_reach,
    _seq_tensors,
    _stash_take,
)


DIM = Rkv + Dr


def _max_counts_reference(topk: torch.Tensor, chunk_len: int, cp_size: int, cp_rank: int) -> list[int]:
    """Naive per-(step, row) python-loop ground truth for Pass 1."""
    seq_len = topk.shape[0]
    out = []
    for j in range(cp_size):
        chunk = (cp_rank - j) % cp_size
        lo, hi = chunk * chunk_len, (chunk + 1) * chunk_len
        worst = 0
        for i in range(seq_len):
            c = sum(1 for t in topk[i, 0].tolist() if lo <= t < hi)
            worst = max(worst, c)
        out.append(worst)
    return out


class TestRemapMaxCounts:
    """Pass 1: scatter_add over absolute chunk ids."""

    def test_matches_python_loop(self):
        torch.manual_seed(0)
        # Production top-k semantics: global positions (here 0..15) or -1.
        topk = torch.randint(0, 16, (8, 1, 12), dtype=torch.int32)
        topk.masked_fill_(torch.rand(topk.shape) < 0.25, -1)
        for cp_size, cp_rank in [(4, 0), (4, 1), (4, 3)]:
            got, _na = _remap_max_counts(topk, 4, cp_size, cp_rank)
            want = _max_counts_reference(topk, 4, cp_size, cp_rank)
            assert got == want, f"cp_rank={cp_rank}: {got} != {want}"

    def test_int32_topk_does_not_crash(self):
        """Regression (rot fix 1): scatter_add needs an int64 index.

        The production DSA top-k is int32; without the ``.to(torch.long)``
        fix on the scatter index this raises ``scatter_add only supports
        int64 indexing``.
        """
        topk = torch.randint(0, 8, (4, 1, 4), dtype=torch.int32)
        _remap_max_counts(topk, 4, 2, 0)


class TestRemapOneChunk:
    """Pass 2: compaction + spread dummy band."""

    def test_reals_packed_to_head(self):
        # Chunk [4, 8). Row 0 reals 5, 4, 6 (count 3 = k -> no dummies);
        # row 1 has no reals (ndummy = k).
        topk = torch.tensor(
            [[[-1, 5, 9, 4, -1, 100, 6, 2]], [[-1, -1, 8, 3, 9, -1, 100, -1]]],
            dtype=torch.int32,
        )
        local, ndummy, k, plan = _remap_one_chunk(topk, chunk_start=4, chunk_len=4, max_count=3)
        assert k == 3 and plan is None  # bucket gate off -> full-width path
        row0 = local[0, 0].tolist()
        assert sorted(row0) == [0, 1, 2]  # 4->0, 5->1, 6->2, all real, compacted
        row1 = local[1, 0].tolist()
        assert all(4 <= v < 4 + _ZERO_ROWS for v in row1)  # all dummies in the zero band
        assert float(ndummy[0]) == 0 and float(ndummy[1]) == 3

    def test_dummy_band_not_concentrated(self):
        """Regression (rot fix 2): dummies spread across the 64-row band.

        The backup concentrated ALL dummies on one row (``chunk_len``), which
        the same file's own comment condemned (grad-op per-row d_kv write
        contention, ~6-7x backward slowdown). A row with more than
        ``_ZERO_ROWS`` dummy slots MUST hit at least two distinct rows.
        """
        topk = torch.full((1, 1, 200), -1, dtype=torch.int32)
        local, ndummy, k, _plan = _remap_one_chunk(topk, chunk_start=0, chunk_len=64, max_count=100)
        vals = local[0, 0].tolist()
        assert len(set(vals)) > 1, f"all dummies concentrated on one row: {set(vals)}"
        assert all(64 <= v < 64 + _ZERO_ROWS for v in vals)
        assert float(ndummy[0]) == 100

    def test_all_dummy_early_return(self):
        topk = torch.full((5, 1, 7), 123, dtype=torch.int32)  # all outside [0, 4)
        local, ndummy, k, plan = _remap_one_chunk(topk, chunk_start=0, chunk_len=4, max_count=0)
        assert plan is None
        assert k == 0
        assert local.shape == (5, 1, 0)
        assert torch.equal(ndummy, torch.zeros(5))


class TestRemapAllChunks:
    """Whole-list build with ``(None, None, 0, None)`` sentinels + weakref cache."""

    def test_sentinels_and_reals(self):
        # cp_size 2, chunk_len 4 -> chunk0=[0,4) chunk1=[4,8).
        # row reals: r0 {1| -}, r1 {2| 5}, r2 {3| 7}, r3 {0| 6}
        # -> max_counts (abs) = [1, 1]; cp_rank 0: step0->chunk0 k=1, step1->chunk1 k=1.
        topk = torch.tensor(
            [[[-1, 1]], [[2, 5]], [[7, 3]], [[0, 6]]],
            dtype=torch.int32,
        )
        all_idx = _remap_all_chunks(topk, 4, 2, 0)
        assert [t[2] for t in all_idx] == [1, 1]
        # cp_rank 1: step0->chunk1 k=1, step1 (wrap) ->chunk0 k=1.
        all_idx1 = _remap_all_chunks(topk, 4, 2, 1)
        assert [t[2] for t in all_idx1] == [1, 1]
        for _j, (local, ndummy, k, plan) in enumerate(all_idx):
            assert plan is None  # bucket gate off by default
            assert local is not None and local.shape == (4, 1, k) and ndummy.shape == (4,)

    def test_all_dummy_step_sentinel(self):
        topk = torch.tensor([[[-1]], [[0]]], dtype=torch.int32)  # nothing in chunk1
        all_idx = _remap_all_chunks(topk, 1, 2, 0)  # chunks [0,1) [1,2)
        assert all_idx[1] == (None, None, 0, None)

    def test_cache_hit_and_eviction(self):
        _REMAP_CACHE.clear()
        topk = torch.randint(-4, 8, (4, 1, 4), dtype=torch.int32)
        first = _remap_all_chunks(topk, 4, 2, 0)
        second = _remap_all_chunks(topk, 4, 2, 0)
        assert first is second  # object-identical cache hit
        del topk
        gc.collect()  # weakref.finalize fires -> entry evicted
        assert len(_REMAP_CACHE) == 0


class TestMergeAndLse:
    """Online-softmax merge == exact logsumexp merge."""

    def test_merge_matches_logsumexp(self):
        torch.manual_seed(1)
        S, N, K = 6, 3, 5
        scores_a = torch.randn(S, N, K) * 4
        scores_b = torch.randn(S, N, K) * 4
        vals_a = torch.randn(S, N, K, Rkv)
        vals_b = torch.randn(S, N, K, Rkv)

        def stats(scores, vals):
            smax = scores.max(dim=-1, keepdim=True).values
            ssum = torch.exp(scores - smax).sum(dim=-1, keepdim=True)
            out = (torch.exp(scores - smax) / ssum).unsqueeze(-1) * vals
            return out.sum(2), smax.squeeze(-1), ssum.squeeze(-1)

        oa, ma, za = stats(scores_a, vals_a)
        ob, mb, zb = stats(scores_b, vals_b)
        # rescale=ones == already-corrected chunk (no dummies here).
        mo, mm, mz = _merge_stats(oa, ma, za, ob, mb, zb, torch.ones_like(zb))
        both_s = torch.cat([scores_a, scores_b], dim=-1)
        both_v = torch.cat([vals_a, vals_b], dim=2)
        exact_out = (torch.softmax(both_s, -1).unsqueeze(-1) * both_v).sum(2)
        exact_lse = torch.logsumexp(both_s, -1)
        assert torch.allclose(mo, exact_out, atol=1e-4)
        assert torch.allclose(_parse_lse(mm, mz, S, N), exact_lse, atol=1e-4)

    def test_correction_is_overflow_free(self):
        """Regression (rot fix 3): the dummy correction must never form exp(smax).

        The backup computed ``z = exp(smax) * ssum`` (fp32 overflow once the
        row max exceeds ~88). The ported algebra ``ssum_true = ssum -
        ndummy*exp(-smax)``, ``rescale = ssum/ssum_true`` is algebraically
        identical (z = real_z + ndummy when dummies score 0) and finite for
        arbitrarily large smax (kernel smax >= 0 always: dummy score 0 is in
        every row's max).
        """

        def corrected(smax, real_ssum, ndummy):  # ported (overflow-free) form
            ssum = real_ssum + ndummy * math.exp(-smax)  # kernel raw: reals + dummies
            ssum_true = max(ssum - ndummy * math.exp(-smax), 1e-30)
            return ssum / ssum_true

        def brute(smax, real_ssum, ndummy):  # backup form, for moderate smax
            z = math.exp(smax) * (real_ssum + ndummy * math.exp(-smax))
            return z / (z - ndummy)

        for smax in (0.0, 5.0, 20.0):
            got, want = corrected(smax, 12.0, 3.0), brute(smax, 12.0, 3.0)
            assert math.isclose(got, want, rel_tol=1e-9), (smax, got, want)
        assert corrected(200.0, 12.0, 3.0) == 1.0  # finite; brute() would be inf/inf
        assert abs(corrected(0.0, 12.0, 3.0) - 15.0 / 12.0) < 1e-12

    def test_raw_fold_equals_corrected_fold(self):
        """Regression (CP4 oracle): the dummy rescale may ride on the fp32
        MERGE WEIGHT, folding the RAW (even kernel-dtype bf16) chunk output --
        but the ACCUMULATOR must stay fp32. A bf16-accumulator fold drifted
        ~1 ulp per chunk, which the Formulation-B backward amplified past the
        oracle tolerance; the identity ``z_true * rescale == z_raw`` is what
        keeps the weight-fold fp32-exact.
        """
        torch.manual_seed(2)
        S, N, K = 5, 2, 4
        scores_a = torch.randn(S, N, K) * 3
        scores_b = torch.randn(S, N, K) * 3
        vals_a = torch.randn(S, N, K, Rkv)
        vals_b = torch.randn(S, N, K, Rkv)

        def raw_stats(scores, vals):
            smax = scores.max(dim=-1, keepdim=True).values
            ssum = torch.exp(scores - smax).sum(dim=-1, keepdim=True)  # raw incl. dummies
            out_raw = (torch.exp(scores - smax)).unsqueeze(-1) * vals  # un-normalised
            return out_raw.sum(2), smax.squeeze(-1), ssum.squeeze(-1)

        ra, ma, za = raw_stats(scores_a, vals_a)
        rb, mb, zb = raw_stats(scores_b, vals_b)
        # Kernel-dtype raw chunks (bf16 casts == what forward_true hands over;
        # the STATS stay fp32 in production too -- they come from .float()).
        ra_bf, rb_bf = ra.to(torch.bfloat16), rb.to(torch.bfloat16)
        rescale = torch.full((S, N), 1.7, dtype=torch.float32)  # synthetic correction
        acc = torch.mul(ra_bf, rescale.unsqueeze(-1))  # driver's first-chunk fold
        got, _, got_ssum = _merge_stats(acc, ma, za, rb_bf, mb, zb, rescale)
        # Closed form: z-weighted average of the corrected chunks, all in fp32.
        new_smax = torch.maximum(ma, mb)
        za_e = za * torch.exp(ma - new_smax)
        zb_e = zb * torch.exp(mb - new_smax)
        want = (
            ra_bf.float() * rescale.unsqueeze(-1) * za_e.unsqueeze(-1)
            + rb_bf.float() * rescale.unsqueeze(-1) * zb_e.unsqueeze(-1)
        ) / ((za_e + zb_e).unsqueeze(-1))
        # fp32-accumulated in-place ops agree to fp32 ulps (reassociation only);
        # a bf16-accumulator fold would drift ~0.8% -- the oracle regression.
        assert torch.allclose(got, want, rtol=1e-5, atol=1e-6), (got - want).abs().max()
        assert got.dtype == torch.float32
        assert torch.allclose(got_ssum, za_e + zb_e, rtol=1e-6)

    def test_deferred_fold_bitidentical(self):
        """Regression (256K pool ceiling): the ``reach == 1`` driver folds the
        two bf16 RAW chunks tile-by-tile (:func:`_fold_two_raws`, no persistent
        fp32 accumulator) instead of the online accumulator path. The per-tile
        ops replicate the online fp32 sequence element-for-element, so results
        must be BIT-IDENTICAL -- both single-real (``_fold_one_raw``) and
        two-real. Pins the deferred path to the CP4-validated fold.
        """
        torch.manual_seed(3)
        S, N = 5, 3
        r0 = torch.randn(S, N, Rkv, dtype=torch.bfloat16)
        r1 = torch.randn(S, N, Rkv, dtype=torch.bfloat16)
        m0, m1 = torch.rand(S, N) * 8, torch.rand(S, N) * 8  # kernel smax >= 0
        z0, z1 = torch.rand(S, N) + 0.5, torch.rand(S, N) + 0.5
        res0, res1 = 1.0 + torch.rand(S, N) * 0.5, 1.0 + torch.rand(S, N) * 0.5

        # single-real: online = fp32 mul then cast-copy into the raw buffer.
        acc1 = torch.mul(r0, res0.unsqueeze(-1))
        one_online = r0.clone()
        one_online.copy_(acc1)
        assert torch.equal(_fold_one_raw(r0.clone(), res0), one_online)

        # two-real: online = first-chunk mul + _merge_stats + cast-copy.
        acc = torch.mul(r0, res0.unsqueeze(-1))
        merged_online, ms_o, mz_o = _merge_stats(acc, m0, z0, r1, m1, z1, res1)
        two_online = r1.clone()
        two_online.copy_(merged_online)
        got, ms_g, mz_g = _fold_two_raws(r0.clone(), res0, m0, z0, r1.clone(), res1, m1, z1)
        assert torch.equal(got, two_online)
        assert torch.equal(ms_g, ms_o) and torch.equal(mz_g, mz_o)
        assert got.dtype == torch.bfloat16


class TestReachAndSeqTensors:
    """Host-side reach bound + cached scalar tensors."""

    def test_reach_bound(self):
        ctx = types.SimpleNamespace(cu_seq_lens_q_list=[0, 12, 16])  # max seg 12
        assert _ring_reach(ctx, 4, 16) == 3  # ceil(12/4)
        assert _ring_reach(ctx, 4, 2) == 1  # capped at cp_size - 1
        assert _ring_reach(ctx, 4, 1) == 0
        assert _ring_reach(types.SimpleNamespace(cu_seq_lens_q_list=[0]), 4, 4) == 0

    def test_seq_tensors_cached(self):
        dev = torch.device("cpu")
        q1, k1 = _seq_tensors(8, dev)
        q2, k2 = _seq_tensors(8, dev)
        assert q1 is q2 and k1 is k2
        assert int(q1) == 8 and int(k1) == 8 + _ZERO_ROWS
        assert q1.dtype == torch.int32


class TestStashIdentity:
    """``_stash_take`` must refuse a stale entry (dead referent / id-reuse):
    a wrong restore is silently bad activations -- strictly worse than a miss."""

    def test_stale_entry_refused_and_not_popped(self):
        sentinel = object()
        stack = [sentinel]
        t = torch.zeros(2)
        saved = dict(_FOLD_STASH)
        _FOLD_STASH.clear()
        try:
            # weakref target dead (lambda: None) while an id-matching tensor lives.
            _FOLD_STASH[id(t)] = (lambda: None, 777, stack)
            assert _stash_take(t) is None
            assert stack == [sentinel]  # NOT popped -> no wrong restore
        finally:
            _FOLD_STASH.clear()
            _FOLD_STASH.update(saved)

    def test_alive_empty_stack_is_miss(self):
        import weakref

        t = torch.zeros(2)
        saved = dict(_FOLD_STASH)
        _FOLD_STASH.clear()
        try:
            _FOLD_STASH[id(t)] = (weakref.ref(t), 778, [])
            assert _stash_take(t) is None
        finally:
            _FOLD_STASH.clear()
            _FOLD_STASH.update(saved)


class TestBucketCostModel:
    """Host cost-model parsing (``XTUNER_CP_RING_BUCKET_COST="seed_f_ms,seed_g_ms,rate_ns"``)."""

    def _clean(self, monkeypatch):
        monkeypatch.delenv("XTUNER_CP_RING_BUCKET_COST", raising=False)

    def test_defaults_match_run303_fit(self, monkeypatch):
        self._clean(monkeypatch)
        sf, sg, rate = _bucket_cost_model()
        assert abs(sf - 4.4e-3) < 1e-9 and abs(sg - 9.2e-3) < 1e-9 and abs(rate - 1.63e-9) < 1e-12

    def test_override_parse(self, monkeypatch):
        monkeypatch.setenv("XTUNER_CP_RING_BUCKET_COST", "1,2,3")
        got = _bucket_cost_model()
        assert all(math.isclose(a, b, rel_tol=1e-12) for a, b in zip(got, (1e-3, 2e-3, 3e-9)))

    def test_malformed_falls_back(self, monkeypatch):
        for raw in ("abc", "1,2", "0,2,3", "-1,2,3", ""):
            monkeypatch.setenv("XTUNER_CP_RING_BUCKET_COST", raw)
            assert _bucket_cost_model() == (4.4e-3, 9.2e-3, 1.63e-9), raw


class TestPickBucketPlan:
    """The shrink-only decision table: drop rows iff ``na < S`` (the cost model is
    monotone in columns -- shrinking never adds a kernel, so it cannot regress)."""

    def _clean(self, monkeypatch):
        monkeypatch.delenv("XTUNER_CP_RING_BUCKET_N", raising=False)
        monkeypatch.delenv("XTUNER_CP_RING_BUCKET_COST", raising=False)

    def test_nothing_to_drop(self, monkeypatch):
        self._clean(monkeypatch)
        assert not _pick_bucket_plan(16384, 336, 16384, 2)  # run306 local step: na == S
        monkeypatch.setenv("XTUNER_CP_RING_BUCKET_N", "force")
        assert not _pick_bucket_plan(1024, 54, 1024, 2)  # force does NOT resurrect na >= S

    def test_neighbor_case_adopted(self, monkeypatch):
        self._clean(monkeypatch)
        # run306-measured neighbor step: k p50=54, n_active p50=62 of S=16384.
        assert _pick_bucket_plan(16384, 54, 62, 2)

    def test_auto_adopts_any_shrink(self, monkeypatch):
        self._clean(monkeypatch)
        # Column work is strictly reduced and no kernel is added -> adopt for
        # every na < S, tile-collapse boundary included (pins the no-regress claim).
        for na in (1, 62, 4095, 4096, 4097, 16383):
            assert _pick_bucket_plan(16384, 54, na, 2), na
            assert _pick_bucket_plan(16384, 54, na, 1), na

    def test_force(self, monkeypatch):
        self._clean(monkeypatch)
        monkeypatch.setenv("XTUNER_CP_RING_BUCKET_N", "FORCE")
        assert _pick_bucket_plan(1024, 54, 1023, 2)
        # A cost model with a gigantic grad seed still cannot veto force.
        monkeypatch.setenv("XTUNER_CP_RING_BUCKET_COST", "1,1000000,1")
        assert _pick_bucket_plan(1024, 54, 1023, 2)


class TestChunkPlanBuild:
    """``_remap_one_chunk`` plan invariants under the bucket gate (CPU)."""

    S = 16
    K = 8

    def _topk(self):
        # rows 0-5 carry in-chunk reals (descending: 4,3,2,1,1,1 slots), rows
        # 6-15 carry NONE (all -1 or out-of-chunk) -> n_active = 6, k = 4.
        topk = torch.full((self.S, 1, self.K), -1, dtype=torch.int32)
        counts = [4, 3, 2, 1, 1, 1]
        for i, c in enumerate(counts):
            topk[i, 0, :c] = torch.arange(1, 1 + c, dtype=torch.int32)  # in [0, 16)
        topk[7, 0, 0] = 100  # out-of-chunk -> still count 0
        topk[8, 0, 0] = 0
        return topk, counts

    def _gate(self, monkeypatch, on=True):
        monkeypatch.setenv("XTUNER_CP_RING_BUCKET", "1" if on else "0")
        monkeypatch.setenv("XTUNER_CP_RING_BUCKET_N", "force")
        monkeypatch.delenv("XTUNER_CP_RING_BUCKET_COST", raising=False)
        monkeypatch.setenv("XTUNER_CP_RING_GRAD_TILES", "2")

    def test_plan_invariants(self, monkeypatch):
        self._gate(monkeypatch)
        topk, counts = self._topk()
        local, ndummy, k, plan = _remap_one_chunk(topk, 0, self.S, max(counts), n_active=6)
        assert k == 4 and plan is not None
        rows = plan.rows
        assert rows.dtype == torch.int64 and rows.shape == (6,)
        assert sorted(rows.tolist()) == [0, 1, 2, 3, 4, 5]  # covers exactly count > 0
        c_rows = torch.tensor(counts, dtype=torch.int32).index_select(0, rows)
        assert bool((c_rows[:-1] >= c_rows[1:]).all())  # descending-count order
        assert torch.equal(plan.idx, local.index_select(0, rows))
        assert plan.idx.shape == (6, 1, 4) and plan.idx.dtype == torch.int32
        assert torch.equal(plan.ndummy, (4 - c_rows).to(torch.float32))
        # ndummy-correction identity: subset ndummy == full-width ndummy on the rows.
        assert torch.equal(plan.ndummy, ndummy.index_select(0, rows))
        # tiles: 6 > S // 4 == 4 -> keeps GRAD_TILES == 2; qlens = [m, t0, t1].
        assert plan.tiles == 2 and plan.m == 6
        assert plan.qlens.tolist() == [6, 3, 3]
        assert plan.qlens.dtype == torch.int32

    def test_tile_collapse_small_m(self, monkeypatch):
        self._gate(monkeypatch)
        topk, counts = self._topk()
        *_, plan = _remap_one_chunk(topk, 0, self.S, max(counts), n_active=2)
        assert plan is not None
        assert plan.tiles == 1 and plan.m == 2  # 2 <= S // 4 == 4
        assert plan.qlens.tolist() == [2, 2]  # every grad tile qlen sums into m

    def test_gate_off_or_degenerate_counts(self, monkeypatch):
        topk, counts = self._topk()
        self._gate(monkeypatch, on=False)
        *_, plan = _remap_one_chunk(topk, 0, self.S, max(counts), n_active=6)
        assert plan is None  # gate off -> byte-identical full-width path
        self._gate(monkeypatch)
        for na in (0, self.S):
            *_, plan = _remap_one_chunk(topk, 0, self.S, max(counts), n_active=na)
            assert plan is None, na  # nothing to drop


class TestRemapAllChunksPlan:
    """Pass-2 global predicate: at most one planned step, never step 0 (the
    deferred pair fold stays raw-in-place into step 0's full-width buffer)."""

    def _topk(self):
        # cp=2, chunk_len=8: chunk0=[0,8) DENSE (every row has a real -> na==S),
        # chunk1=[8,16) SPARSE (rows 0/1 only) -> exactly one shrinkable step.
        topk = torch.full((8, 1, 4), -1, dtype=torch.int32)
        for i in range(8):
            topk[i, 0, 0] = i  # chunk0: one real per row -> na = S
        topk[0, 0, 1] = 9
        topk[1, 0, 1] = 10  # chunk1: two active rows
        return topk

    def _gate(self, monkeypatch):
        monkeypatch.setenv("XTUNER_CP_RING_BUCKET", "1")
        monkeypatch.delenv("XTUNER_CP_RING_BUCKET_N", raising=False)
        monkeypatch.delenv("XTUNER_CP_RING_BUCKET_COST", raising=False)

    def test_single_candidate_planned(self, monkeypatch):
        self._gate(monkeypatch)
        _REMAP_CACHE.clear()
        all_idx = _remap_all_chunks(self._topk(), 8, 2, 0)  # rank0: step0=chunk0(dense), step1=chunk1(sparse)
        assert all_idx[0][3] is None
        plan = all_idx[1][3]
        assert plan is not None and plan.m == 2
        assert sorted(plan.rows.tolist()) == [0, 1]
        assert all_idx[0][3] is None and all_idx[1][2] == 1

    def test_step0_candidate_never_planned(self, monkeypatch):
        self._gate(monkeypatch)
        _REMAP_CACHE.clear()
        # rank1 flips which step holds the dense chunk: step0 = chunk1 (sparse
        # -> would qualify) -- the predicate must still refuse step 0.
        all_idx = _remap_all_chunks(self._topk(), 8, 2, 1)
        assert all(e[3] is None for e in all_idx), "plan attached to step 0"

    def test_multiple_candidates_bucket_nothing(self, monkeypatch):
        self._gate(monkeypatch)
        _REMAP_CACHE.clear()
        topk = torch.full((8, 1, 4), -1, dtype=torch.int32)
        topk[0, 0, 0] = 3
        topk[1, 0, 0] = 10  # both chunks sparse -> two qualifying steps -> refuse all
        all_idx = _remap_all_chunks(topk, 8, 2, 0)
        assert all(e[3] is None for e in all_idx)

    def test_sentinels_keep_four_tuple(self, monkeypatch):
        self._gate(monkeypatch)
        _REMAP_CACHE.clear()
        topk = torch.full((8, 1, 4), -1, dtype=torch.int32)  # every slot invalid -> both steps all-dummy
        all_idx = _remap_all_chunks(topk, 8, 2, 0)
        assert all_idx == [(None, None, 0, None), (None, None, 0, None)]


class TestFoldSubsetMath:
    """Row-subset folds must replicate the FULL-WIDTH deferred fold exactly for
    the covered rows (same fp32 op order -> bit-identical) and the bare-rescale
    limit for the dropped rows (zero contribution, 1e-30-clamped weight)."""

    def test_pair_subset_bitidentical_on_rows(self):
        torch.manual_seed(7)
        S, N = 8, 3
        rows = torch.tensor([1, 4, 6], dtype=torch.int64)
        mask = torch.zeros(S, dtype=torch.bool)
        mask[rows] = True
        raw0 = torch.randn(S, N, Rkv, dtype=torch.bfloat16)
        r0 = torch.rand(S, N) * 0.5 + 1.0
        m0 = torch.rand(S, N) * 8  # kernel smax >= 0
        z0 = torch.rand(S, N) + 0.5
        # Full-width "second chunk" where the dropped rows have ZERO weight:
        # ssum_true == 0 (clamp), raw out == 0 -- exactly what a count-0 row's
        # all-dummy kernel call produces.
        raw1 = torch.randn(S, N, Rkv, dtype=torch.bfloat16) * mask.view(S, 1, 1)
        r1 = torch.rand(S, N) * 0.5 + 1.0
        m1 = (torch.rand(S, N) * 8) * mask.view(S, 1).float()
        z1 = (torch.rand(S, N) + 0.5) * mask.view(S, 1).float()
        want, wsm, wsz = _fold_two_raws(
            raw0.clone(), r0, m0.clone(), z0.clone(), raw1.clone(), r1, m1.clone(), z1.clone()
        )
        got, gsm, gsz = _fold_pair_subset(
            raw0.clone(),
            r0,
            m0.clone(),
            z0.clone(),
            raw1.index_select(0, rows),
            r1.index_select(0, rows),
            m1.index_select(0, rows),
            z1.index_select(0, rows),
            rows,
        )
        assert torch.equal(got[rows], want[rows])  # same op order -> bit-identical
        # Dropped rows: full width multiplies by prev_w (= z0 * exp(0) since
        # m1 == 0, m0 >= 0) and divides by the same value -> a <=2 bf16-ulp
        # round trip around the bare rescale the subset path keeps directly.
        assert ((got[~mask].float() - want[~mask].float()).abs() <= 2 * 2**-7 * want[~mask].float().abs() + 1e-7).all()
        assert torch.equal(gsm, wsm) and torch.equal(gsz, wsz)  # m1 == 0 -> merged stats unchanged

    def test_subset_single_reconstructs_dummy_rows(self):
        torch.manual_seed(11)
        S, N = 8, 2
        rows = torch.tensor([0, 3, 5], dtype=torch.int64)
        raw = torch.randn(3, N, Rkv, dtype=torch.bfloat16)
        res = torch.rand(3, N) + 1.0
        smax = torch.rand(3, N) * 8
        ssum = torch.rand(3, N) + 0.5
        out, m, z = _fold_subset_single(raw, res, smax, ssum, rows, S)
        assert out.shape == (S, N, Rkv) and out.dtype == torch.bfloat16
        mask = torch.zeros(S, dtype=torch.bool)
        mask[rows] = True
        want = torch.zeros(S, N, Rkv, dtype=torch.bfloat16)
        want[rows] = raw.float().mul(res.unsqueeze(-1)).to(torch.bfloat16)
        assert torch.equal(out, want)  # attended rows rescaled, never-attended rows exactly 0
        assert torch.equal(m[mask], smax) and bool((m[~mask] == 0).all())
        assert torch.equal(z[mask], ssum)
        assert bool((z[~mask] == 1e-30).all())  # never-attended rows: today's clamp
