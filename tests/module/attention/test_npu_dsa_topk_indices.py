"""NPU DSA top-k indices 精度单元测试。

通过 XTuner 的 dsa_topk_indices API 对比 npu 后端与 torch 基线，
验证 npu_lightning_indexer 的 top-k indices 正确性。

精度阈值与 test_dsa_mla.py 一致。

测试矩阵:
  1. 单序列 top-k 集合一致性
  2. 打包多序列 causal 边界
  3. 大 topk (全选)
  4. 不同 head 数量
  5. 索引范围合法性
  6. Causal 约束 (不选未来位置)
  7. 输出形状与 dtype
  8. 打包序列跨序列泄漏
  9. topk=seq_len/2 高重叠

PACK / SP 场景补充:
  10. PACK + SP=1: 完整序列, 打包多段
  11. PACK + SP=2: SP 切分 query, 全局 KV
  12. NOPACK + SP=1: 单序列无打包
  13. NOPACK + SP=2: 单序列 SP 切分
"""
import os
import sys

import pytest
import torch
import torch_npu

sys.path.insert(0, "/weight/jschen/code/xtuner")
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.ops.sparse_mla import dsa_topk_indices

# 注册 slow marker (大规模测试, 默认跳过)
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with -k 'not slow')")

DEVICE = "npu:0"
INDEX_HEAD_DIM = 128

# 精度阈值 — 收紧以检测 SP>1 场景下的累积偏差
MATCH_RATE_THRESHOLD = 95.0        # top-k 集合重叠率 (原80%, 收紧至95%)
PACKED_MATCH_RATE_THRESHOLD = 90.0  # 打包多序列重叠率 (原70%, 收紧至90%)
SP_PACKED_MATCH_RATE_THRESHOLD = 85.0  # SP>1 打包多序列重叠率 (新增, SP8场景)


def _make_seq_ctx(seq_lens, device, shard_start=0, shard_size=None):
    """Build a SequenceContext for packed sequences.

    Args:
        seq_lens: list of sequence lengths (e.g. [256] or [128, 256, 192])
        shard_start: SP shard start offset (0 for SP=1 or SP=2 rank0)
        shard_size: SP shard size (total_len for SP=1, total_len//2 for SP=2)
    """
    cu_seq = torch.tensor([0] + list(seq_lens), dtype=torch.int32, device=device)
    cu_seq = torch.cumsum(cu_seq, dim=0).to(torch.int32)
    total = sum(seq_lens)

    if shard_size is None:
        shard_size = total

    input_ids = torch.arange(total, device=device).unsqueeze(0).long()

    return SequenceContext(
        input_ids=input_ids,
        cu_seq_lens_q=cu_seq,
        cu_seq_lens_k=cu_seq,
        max_length_q=max(seq_lens),
        max_length_k=max(seq_lens),
        device=device,
        shard_start=shard_start,
        shard_size=shard_size,
    )


def _make_inputs(seq_lens, num_index_heads, head_dim, device, dtype=torch.bfloat16, seed=42,
                 shard_start=0, shard_size=None):
    """Build q, k, weights for the DSA indexer.

    Returns:
        q: [B=1, S_local, Ni, Di]  (S_local = shard_size for SP>1)
        k: [B=1, S_g, Di]          (S_g = total_len, always global)
        weights: [B=1, S_local, Ni]
        seq_ctx: SequenceContext
    """
    torch.manual_seed(seed)
    total_len = sum(seq_lens)

    if shard_size is None:
        shard_size = total_len

    # q and weights are LOCAL (shard), k is GLOBAL
    q = torch.randn(1, shard_size, num_index_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(1, total_len, head_dim, dtype=dtype, device=device)
    weights = torch.randn(1, shard_size, num_index_heads, dtype=dtype, device=device)
    weights = torch.softmax(weights.float(), dim=1).to(dtype)

    seq_ctx = _make_seq_ctx(seq_lens, device, shard_start=shard_start, shard_size=shard_size)

    return q, k, weights, seq_ctx


def _make_sp_shard(seq_lens, sp_rank, sp_size, device):
    """Compute shard_start and shard_size for SP split.

    For SP=2 with total=576: rank0 gets [0:288], rank1 gets [288:576]
    (padded to even split if needed).
    """
    total = sum(seq_lens)
    # Pad to multiple of sp_size
    padded = ((total + sp_size - 1) // sp_size) * sp_size
    shard_size = padded // sp_size
    shard_start = sp_rank * shard_size
    return shard_start, shard_size


def _compute_match_rate(ref_indices, npu_indices, seq_ctx, kv_len):
    """Compare topk indices: per-query set overlap + causal validity.

    match_rate = |ref ∩ npu| / |npu| (NPU 索引的命中率)

    For SP>1: ref_indices/npu_indices are local [shard_size, 1, K],
    but packed_causal_query_ranges returns global ranges.
    We adjust by using local positions (i in [0, S)) and checking
    global index validity via causal_valid.

    Returns:
        (match_rate, violations, stats_str)
    """
    S = ref_indices.shape[0]
    K = ref_indices.shape[-1]
    shard_start = getattr(seq_ctx, '_shard_start', 0)

    starts, ends = seq_ctx.packed_causal_query_ranges(S, ref_indices.device)
    cols = torch.arange(kv_len, device=ref_indices.device)[None, :]
    causal_valid = (cols >= starts[:, None]) & (cols < ends[:, None])  # [S, kv_len]

    total_match = 0
    total_npu_slots = 0
    total_ref_slots = 0
    violations = 0

    for i in range(S):
        ref_set = set()
        npu_set = set()

        for j in range(K):
            ri = ref_indices[i, 0, j].item()
            ni = npu_indices[i, 0, j].item()

            if ri >= 0:
                ref_set.add(ri)
                total_ref_slots += 1
            if ni >= 0:
                npu_set.add(ni)
                total_npu_slots += 1
                if ni < kv_len and not causal_valid[i, ni].item():
                    violations += 1

        overlap = npu_set & ref_set
        total_match += len(overlap)

    match_rate = total_match / max(total_npu_slots, 1) * 100
    ref_coverage = total_match / max(total_ref_slots, 1) * 100

    # per-segment match rate 分解 (定位哪个段出错)
    # For SP>1: starts/ends are global, but indices are local [0, S).
    # Convert global segment boundaries to local by subtracting shard_start.
    seg_details = []
    starts_list, ends_list = seq_ctx.packed_causal_query_ranges(S, ref_indices.device)
    starts_list = starts_list.tolist()
    ends_list = ends_list.tolist()
    seg_starts = sorted(set(starts_list))
    for ss in seg_starts:
        seg_end = ends_list[starts_list.index(ss)]
        # Convert to local indices for indexing into ref/npu_indices
        local_start = max(0, ss - shard_start)
        local_end = min(S, seg_end - shard_start)
        if local_start >= local_end:
            continue
        seg_match = 0
        seg_total = 0
        for i in range(local_start, local_end):
            for j in range(K):
                ni = npu_indices[i, 0, j].item()
                ri = ref_indices[i, 0, j].item()
                if ni >= 0:
                    seg_total += 1
                    if ri == ni:
                        seg_match += 1
        seg_mr = seg_match / max(seg_total, 1) * 100
        seg_details.append(f"seg[{local_start}:{local_end}](g:{ss}:{seg_end})={seg_mr:.1f}%({seg_match}/{seg_total})")

    seg_str = " | ".join(seg_details)

    stats_str = (f"match_rate={match_rate:.1f}% (npu_hit={total_match}/{total_npu_slots}), "
                 f"ref_coverage={ref_coverage:.1f}% (ref_hit={total_match}/{total_ref_slots}), "
                 f"violations={violations}, "
                 f"per_seg=[{seg_str}]")
    return match_rate, violations, stats_str


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires NPU environment")
class TestNPUDSATopKIndices:

    def test_single_sequence_topk(self):
        """单序列: NPU 与 torch 基线的 top-k indices 集合应高重叠。"""
        seq_lens = [256]
        topk = 128
        q, k, weights, seq_ctx = _make_inputs(seq_lens, num_index_heads=4, head_dim=INDEX_HEAD_DIM, device=DEVICE)

        ref = dsa_topk_indices(q, k, weights, seq_ctx,
                               index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="torch")
        npu = dsa_topk_indices(q, k, weights, seq_ctx,
                               index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="npu")

        assert ref.shape == npu.shape, f"shape mismatch: {ref.shape} vs {npu.shape}"
        assert ref.dtype == npu.dtype, f"dtype mismatch: {ref.dtype} vs {npu.dtype}"

        match_rate, violations, stats = _compute_match_rate(ref, npu, seq_ctx, sum(seq_lens))
        print(f"  {stats}", flush=True)
        assert violations == 0, f"causal violations: {violations}"
        assert match_rate >= MATCH_RATE_THRESHOLD, f"match rate too low: {match_rate:.1f}%"

    def test_packed_sequences(self):
        """打包多序列: causal 边界正确, top-k 集合高重叠。"""
        seq_lens = [128, 256, 192]
        topk = 128
        q, k, weights, seq_ctx = _make_inputs(seq_lens, num_index_heads=4, head_dim=INDEX_HEAD_DIM, device=DEVICE)

        ref = dsa_topk_indices(q, k, weights, seq_ctx,
                               index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="torch")
        npu = dsa_topk_indices(q, k, weights, seq_ctx,
                               index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="npu")

        assert ref.shape == npu.shape
        match_rate, violations, stats = _compute_match_rate(ref, npu, seq_ctx, sum(seq_lens))
        print(f"  {stats}", flush=True)
        assert violations == 0, f"causal violations: {violations}"
        assert match_rate >= PACKED_MATCH_RATE_THRESHOLD, f"match rate too low: {match_rate:.1f}%"

    def test_large_topk(self):
        """topk = seq_len (全选): 所有有效位置都应被选中。"""
        seq_lens = [256]
        topk = 256
        q, k, weights, seq_ctx = _make_inputs(seq_lens, num_index_heads=4, head_dim=INDEX_HEAD_DIM, device=DEVICE)

        ref = dsa_topk_indices(q, k, weights, seq_ctx,
                               index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="torch")
        npu = dsa_topk_indices(q, k, weights, seq_ctx,
                               index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="npu")

        match_rate, violations, stats = _compute_match_rate(ref, npu, seq_ctx, sum(seq_lens))
        print(f"  {stats}", flush=True)
        assert violations == 0, f"causal violations: {violations}"
        assert match_rate >= 90.0, f"match rate too low: {match_rate:.1f}%"

    def test_various_head_counts(self):
        """不同 index head 数量: 结果应稳定。"""
        seq_lens = [256]
        topk = 128
        for num_heads in [1, 2, 8, 16]:
            q, k, weights, seq_ctx = _make_inputs(
                seq_lens, num_index_heads=num_heads, head_dim=INDEX_HEAD_DIM, device=DEVICE)
            ref = dsa_topk_indices(q, k, weights, seq_ctx,
                                   index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="torch")
            npu = dsa_topk_indices(q, k, weights, seq_ctx,
                                   index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="npu")
            match_rate, violations, stats = _compute_match_rate(ref, npu, seq_ctx, sum(seq_lens))
            print(f"  {stats}", flush=True)
            assert violations == 0, f"heads={num_heads}: causal violations: {violations}"
            assert match_rate >= MATCH_RATE_THRESHOLD, f"heads={num_heads}: match rate too low: {match_rate:.1f}%"

    def test_indices_within_range(self):
        """所有非 -1 的索引应在 [0, kv_len) 范围内。"""
        seq_lens = [256]
        topk = 128
        q, k, weights, seq_ctx = _make_inputs(seq_lens, num_index_heads=4, head_dim=INDEX_HEAD_DIM, device=DEVICE)

        npu = dsa_topk_indices(q, k, weights, seq_ctx,
                               index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="npu")

        kv_len = sum(seq_lens)
        valid_mask = npu >= 0
        valid_indices = npu[valid_mask]
        assert valid_indices.min() >= 0, f"negative valid index: {valid_indices.min()}"
        assert valid_indices.max() < kv_len, f"index out of range: {valid_indices.max()} >= {kv_len}"

    def test_causal_constraint(self):
        """每个 query 的 top-k 索引不应包含未来位置。"""
        seq_lens = [256]
        topk = 128
        q, k, weights, seq_ctx = _make_inputs(seq_lens, num_index_heads=4, head_dim=INDEX_HEAD_DIM, device=DEVICE)

        npu = dsa_topk_indices(q, k, weights, seq_ctx,
                               index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="npu")

        S = npu.shape[0]
        kv_len = sum(seq_lens)
        starts, ends = seq_ctx.packed_causal_query_ranges(S, DEVICE)
        cols = torch.arange(kv_len, device=DEVICE)[None, :]
        causal_valid = (cols >= starts[:, None]) & (cols < ends[:, None])

        violations = 0
        for i in range(S):
            for j in range(npu.shape[-1]):
                idx = npu[i, 0, j].item()
                if idx >= 0 and not causal_valid[i, idx].item():
                    violations += 1

        assert violations == 0, f"causal violations: {violations}"

    def test_output_shape_dtype(self):
        """输出形状 [S, 1, K], dtype int64。"""
        seq_lens = [256]
        topk = 128
        q, k, weights, seq_ctx = _make_inputs(seq_lens, num_index_heads=4, head_dim=INDEX_HEAD_DIM, device=DEVICE)

        npu = dsa_topk_indices(q, k, weights, seq_ctx,
                               index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="npu")

        S = sum(seq_lens)
        assert npu.shape == (S, 1, topk), f"shape mismatch: {npu.shape} != ({S}, 1, {topk})"
        assert npu.dtype == torch.int64, f"dtype mismatch: {npu.dtype} != int64"

    def test_packed_causal_boundaries(self):
        """打包序列的边界: 序列 0 的 query 不应选到序列 1 的 KV。"""
        seq_lens = [128, 128]
        topk = 128
        q, k, weights, seq_ctx = _make_inputs(seq_lens, num_index_heads=4, head_dim=INDEX_HEAD_DIM, device=DEVICE)

        npu = dsa_topk_indices(q, k, weights, seq_ctx,
                               index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="npu")

        S = sum(seq_lens)
        starts, ends = seq_ctx.packed_causal_query_ranges(S, DEVICE)

        cross_seq_violations = 0
        for i in range(S):
            seq_start = starts[i].item()
            seq_end = ends[i].item()
            for j in range(npu.shape[-1]):
                idx = npu[i, 0, j].item()
                if idx >= 0 and (idx < seq_start or idx >= seq_end):
                    cross_seq_violations += 1

        assert cross_seq_violations == 0, f"cross-sequence violations: {cross_seq_violations}"

    def test_indexer_scores_consistency(self):
        """topk=seq_len/2: top-k 集合高重叠。"""
        seq_lens = [256]
        topk = 128
        num_heads = 4
        q, k, weights, seq_ctx = _make_inputs(seq_lens, num_index_heads=num_heads,
                                               head_dim=INDEX_HEAD_DIM, device=DEVICE)

        ref = dsa_topk_indices(q, k, weights, seq_ctx,
                               index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="torch")
        npu = dsa_topk_indices(q, k, weights, seq_ctx,
                               index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="npu")

        match_rate, violations, stats = _compute_match_rate(ref, npu, seq_ctx, sum(seq_lens))
        print(f"  {stats}", flush=True)
        assert violations == 0, f"causal violations: {violations}"
        assert match_rate >= 99.0, f"match rate too low: {match_rate:.1f}%"


# ── PACK / SP 场景补充测试 ─────────────────────────────────────────────────

@pytest.mark.skipif(not torch.npu.is_available(), reason="requires NPU environment")
class TestNPUDSATopKIndicesPackSP:
    """PACK / NOPACK × SP=1 / SP=2 场景的精度对齐测试。"""

    # --- PACK + SP=1 ---

    def test_pack_sp1_single_seq(self):
        """PACK + SP=1: 单序列打包 (seq_lens=[256]), 完整 query + 完整 KV。"""
        seq_lens = [256]
        topk = 128
        q, k, weights, seq_ctx = _make_inputs(seq_lens, num_index_heads=4, head_dim=INDEX_HEAD_DIM, device=DEVICE,
                                               shard_start=0, shard_size=sum(seq_lens))
        ref = dsa_topk_indices(q, k, weights, seq_ctx,
                               index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="torch")
        npu = dsa_topk_indices(q, k, weights, seq_ctx,
                               index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="npu")
        match_rate, violations, stats = _compute_match_rate(ref, npu, seq_ctx, sum(seq_lens))
        print(f"  {stats}", flush=True)
        assert violations == 0, f"causal violations: {violations}"
        assert match_rate >= MATCH_RATE_THRESHOLD, f"match rate too low: {match_rate:.1f}%"

    def test_pack_sp1_multi_seq(self):
        """PACK + SP=1: 多序列打包 (seq_lens=[128, 256, 192]), 完整 query + 完整 KV。

        这是 pack_sp1 训练的场景。npu_lightning_indexer 用 sparse_mode=3 做全局
        rightDownCausal, 但打包多序列时应做 packed causal (段间隔离)。
        """
        seq_lens = [128, 256, 192]
        topk = 128
        q, k, weights, seq_ctx = _make_inputs(seq_lens, num_index_heads=4, head_dim=INDEX_HEAD_DIM, device=DEVICE,
                                               shard_start=0, shard_size=sum(seq_lens))
        ref = dsa_topk_indices(q, k, weights, seq_ctx,
                               index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="torch")
        npu = dsa_topk_indices(q, k, weights, seq_ctx,
                               index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="npu")
        match_rate, violations, stats = _compute_match_rate(ref, npu, seq_ctx, sum(seq_lens))
        print(f"  {stats}", flush=True)
        assert violations == 0, f"causal violations: {violations}"
        assert match_rate >= PACKED_MATCH_RATE_THRESHOLD, f"match rate too low: {match_rate:.1f}%"

    # --- PACK + SP=2 ---

    def test_pack_sp2_rank0(self):
        """PACK + SP=2 rank0: 打包多序列, query 切分前半, KV 全局 gather。

        这是 pack_sp2 训练的场景。SP=2 时 query 是 local shard (前半),
        kv 是 global (完整)。seq_ctx 的 shard_start=0, shard_size=total//2。
        """
        seq_lens = [128, 256, 192]  # total=576
        topk = 128
        total = sum(seq_lens)
        shard_start, shard_size = _make_sp_shard(seq_lens, sp_rank=0, sp_size=2, device=DEVICE)

        q, k, weights, seq_ctx = _make_inputs(seq_lens, num_index_heads=4, head_dim=INDEX_HEAD_DIM, device=DEVICE,
                                               shard_start=shard_start, shard_size=shard_size)
        ref = dsa_topk_indices(q, k, weights, seq_ctx,
                               index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="torch")
        npu = dsa_topk_indices(q, k, weights, seq_ctx,
                               index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="npu")
        match_rate, violations, stats = _compute_match_rate(ref, npu, seq_ctx, total)
        print(f"  {stats}", flush=True)
        assert violations == 0, f"causal violations: {violations}"
        assert match_rate >= PACKED_MATCH_RATE_THRESHOLD, f"match rate too low: {match_rate:.1f}%"

    def test_pack_sp2_rank1(self):
        """PACK + SP=2 rank1: 打包多序列, query 切分后半, KV 全局 gather。"""
        seq_lens = [128, 256, 192]  # total=576
        topk = 128
        total = sum(seq_lens)
        shard_start, shard_size = _make_sp_shard(seq_lens, sp_rank=1, sp_size=2, device=DEVICE)

        q, k, weights, seq_ctx = _make_inputs(seq_lens, num_index_heads=4, head_dim=INDEX_HEAD_DIM, device=DEVICE,
                                               shard_start=shard_start, shard_size=shard_size)
        ref = dsa_topk_indices(q, k, weights, seq_ctx,
                               index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="torch")
        npu = dsa_topk_indices(q, k, weights, seq_ctx,
                               index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="npu")
        match_rate, violations, stats = _compute_match_rate(ref, npu, seq_ctx, total)
        print(f"  {stats}", flush=True)
        assert violations == 0, f"causal violations: {violations}"
        assert match_rate >= PACKED_MATCH_RATE_THRESHOLD, f"match rate too low: {match_rate:.1f}%"

    # --- NOPACK + SP=1 ---

    def test_nopack_sp1(self):
        """NOPACK + SP=1: 单序列无打包, 完整 query + 完整 KV。"""
        seq_lens = [256]
        topk = 128
        q, k, weights, seq_ctx = _make_inputs(seq_lens, num_index_heads=4, head_dim=INDEX_HEAD_DIM, device=DEVICE,
                                               shard_start=0, shard_size=sum(seq_lens))
        ref = dsa_topk_indices(q, k, weights, seq_ctx,
                               index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="torch")
        npu = dsa_topk_indices(q, k, weights, seq_ctx,
                               index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="npu")
        match_rate, violations, stats = _compute_match_rate(ref, npu, seq_ctx, sum(seq_lens))
        print(f"  {stats}", flush=True)
        assert violations == 0, f"causal violations: {violations}"
        assert match_rate >= MATCH_RATE_THRESHOLD, f"match rate too low: {match_rate:.1f}%"

    # --- NOPACK + SP=2 ---

    def test_nopack_sp2_rank0(self):
        """NOPACK + SP=2 rank0: 单序列无打包, query 切分前半, KV 全局。"""
        seq_lens = [256]
        topk = 128
        total = sum(seq_lens)
        shard_start, shard_size = _make_sp_shard(seq_lens, sp_rank=0, sp_size=2, device=DEVICE)

        q, k, weights, seq_ctx = _make_inputs(seq_lens, num_index_heads=4, head_dim=INDEX_HEAD_DIM, device=DEVICE,
                                               shard_start=shard_start, shard_size=shard_size)
        ref = dsa_topk_indices(q, k, weights, seq_ctx,
                               index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="torch")
        npu = dsa_topk_indices(q, k, weights, seq_ctx,
                               index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="npu")
        match_rate, violations, stats = _compute_match_rate(ref, npu, seq_ctx, total)
        print(f"  {stats}", flush=True)
        assert violations == 0, f"causal violations: {violations}"
        assert match_rate >= MATCH_RATE_THRESHOLD, f"match rate too low: {match_rate:.1f}%"

    def test_nopack_sp2_rank1(self):
        """NOPACK + SP=2 rank1: 单序列无打包, query 切分后半, KV 全局。"""
        seq_lens = [256]
        topk = 128
        total = sum(seq_lens)
        shard_start, shard_size = _make_sp_shard(seq_lens, sp_rank=1, sp_size=2, device=DEVICE)

        q, k, weights, seq_ctx = _make_inputs(seq_lens, num_index_heads=4, head_dim=INDEX_HEAD_DIM, device=DEVICE,
                                               shard_start=shard_start, shard_size=shard_size)
        ref = dsa_topk_indices(q, k, weights, seq_ctx,
                               index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="torch")
        npu = dsa_topk_indices(q, k, weights, seq_ctx,
                               index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="npu")
        match_rate, violations, stats = _compute_match_rate(ref, npu, seq_ctx, total)
        print(f"  {stats}", flush=True)
        assert violations == 0, f"causal violations: {violations}"
        assert match_rate >= MATCH_RATE_THRESHOLD, f"match rate too low: {match_rate:.1f}%"

    # --- 大规模 PACK + SP 场景 (接近训练规模) ---

    @pytest.mark.slow
    def test_pack_sp1_large(self):
        """PACK + SP=1 大规模: seq_lens=[2048], topk=2048 (训练规模)。"""
        seq_lens = [2048]
        topk = 2048
        q, k, weights, seq_ctx = _make_inputs(seq_lens, num_index_heads=4, head_dim=INDEX_HEAD_DIM, device=DEVICE,
                                               shard_start=0, shard_size=sum(seq_lens))
        ref = dsa_topk_indices(q, k, weights, seq_ctx,
                               index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="torch")
        npu = dsa_topk_indices(q, k, weights, seq_ctx,
                               index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="npu")
        match_rate, violations, stats = _compute_match_rate(ref, npu, seq_ctx, sum(seq_lens))
        print(f"  {stats}", flush=True)
        assert violations == 0, f"causal violations: {violations}"
        assert match_rate >= MATCH_RATE_THRESHOLD, f"match rate too low: {match_rate:.1f}%"

    @pytest.mark.slow
    def test_pack_sp1_multi_seq_large(self):
        """PACK + SP=1 多序列大规模: seq_lens=[1024, 1024], topk=2048。

        模拟训练中 pack_max_length=4096 的打包场景。
        """
        seq_lens = [1024, 1024]
        topk = 2048
        q, k, weights, seq_ctx = _make_inputs(seq_lens, num_index_heads=4, head_dim=INDEX_HEAD_DIM, device=DEVICE,
                                               shard_start=0, shard_size=sum(seq_lens))
        ref = dsa_topk_indices(q, k, weights, seq_ctx,
                               index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="torch")
        npu = dsa_topk_indices(q, k, weights, seq_ctx,
                               index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="npu")
        match_rate, violations, stats = _compute_match_rate(ref, npu, seq_ctx, sum(seq_lens))
        print(f"  {stats}", flush=True)
        assert violations == 0, f"causal violations: {violations}"
        assert match_rate >= PACKED_MATCH_RATE_THRESHOLD, f"match rate too low: {match_rate:.1f}%"


# ── SP8 场景测试 (复现训练配置 EP8SP8PACK16K) ──────────────────────────────

@pytest.mark.skipif(not torch.npu.is_available(), reason="requires NPU environment")
class TestNPUDSATopKIndicesSP8:
    """SP8 场景的 top-k indices 精度对齐测试。

    复现训练中的 EP8SP8PACK16K 配置: query 被 SP 切成 8 份,
    KV 全局 gather, 每个 rank 只处理 1/8 的 query。
    使用接近训练规模的 seq_len 和 topk。
    """

    SP8_SEQ_LENS = [512, 512, 512, 512]  # total=2048, 4段打包
    SP8_TOPK = 512

    def _run_sp8_test(self, sp_rank, seq_lens=None, topk=None):
        seq_lens = seq_lens or self.SP8_SEQ_LENS
        topk = topk or self.SP8_TOPK
        total = sum(seq_lens)
        shard_start, shard_size = _make_sp_shard(seq_lens, sp_rank=sp_rank, sp_size=8, device=DEVICE)

        q, k, weights, seq_ctx = _make_inputs(
            seq_lens, num_index_heads=4, head_dim=INDEX_HEAD_DIM, device=DEVICE,
            shard_start=shard_start, shard_size=shard_size,
        )
        ref = dsa_topk_indices(q, k, weights, seq_ctx,
                               index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="torch")
        npu = dsa_topk_indices(q, k, weights, seq_ctx,
                               index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="npu")

        assert ref.shape == npu.shape, f"shape mismatch: {ref.shape} vs {npu.shape}"
        match_rate, violations, stats = _compute_match_rate(ref, npu, seq_ctx, total)
        print(f"  [SP8 rank{sp_rank}] {stats}", flush=True)
        assert violations == 0, f"[SP8 rank{sp_rank}] causal violations: {violations}"
        assert match_rate >= SP_PACKED_MATCH_RATE_THRESHOLD, \
            f"[SP8 rank{sp_rank}] match rate too low: {match_rate:.1f}% (threshold={SP_PACKED_MATCH_RATE_THRESHOLD}%)"
        return match_rate, violations

    def test_sp8_rank0(self):
        self._run_sp8_test(0)

    def test_sp8_rank1(self):
        self._run_sp8_test(1)

    def test_sp8_rank3(self):
        self._run_sp8_test(3)

    def test_sp8_rank4(self):
        self._run_sp8_test(4)

    def test_sp8_rank7(self):
        self._run_sp8_test(7)

    def test_sp8_all_ranks(self):
        rates = []
        for r in range(8):
            mr, v = self._run_sp8_test(r)
            rates.append(mr)
        avg = sum(rates) / len(rates)
        worst = min(rates)
        print(f"  [SP8 summary] avg_match={avg:.1f}%, worst={worst:.1f}%, "
              f"per_rank={['%.1f' % r for r in rates]}", flush=True)
        assert avg >= SP_PACKED_MATCH_RATE_THRESHOLD, \
            f"SP8 avg match rate too low: {avg:.1f}%"

    @pytest.mark.slow
    def test_sp8_large_packed(self):
        seq_lens = [4096, 4096]
        topk = 2048
        rates = []
        for r in [0, 3, 7]:
            mr, v = self._run_sp8_test(r, seq_lens=seq_lens, topk=topk)
            rates.append(mr)
        avg = sum(rates) / len(rates)
        print(f"  [SP8 large] avg_match={avg:.1f}%, tested_ranks=[0,3,7]", flush=True)
        assert avg >= SP_PACKED_MATCH_RATE_THRESHOLD, \
            f"SP8 large avg match rate too low: {avg:.1f}%"

    def test_sp8_cross_segment_boundary(self):
        seq_lens = [256] * 8
        topk = 128
        rates = []
        for r in range(8):
            mr, v = self._run_sp8_test(r, seq_lens=seq_lens, topk=topk)
            rates.append(mr)
        avg = sum(rates) / len(rates)
        print(f"  [SP8 boundary] avg_match={avg:.1f}%", flush=True)
        assert avg >= SP_PACKED_MATCH_RATE_THRESHOLD, \
            f"SP8 boundary avg match rate too low: {avg:.1f}%"


# ── 独立运行入口 ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
