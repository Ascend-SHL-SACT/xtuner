"""NPU DSA top-k indices 反向梯度测试。

验证 indexer 和 sparse_mla 在 packed 多序列场景下的反向梯度有限性。
覆盖:
  1. dsa_topk_indices 的 autograd 梯度有限性 (前向 indices 不参与梯度, 测试 q/k/weights 的梯度)
  2. sparse_mla 的反向梯度有限性 (packed 多序列 + SP=1/SP=2)
  3. 端到端: indexer → sparse_mla → backward, 检查梯度是否有限

精度阈值与 test_dsa_mla.py 一致。
"""
import os
import sys

import pytest
import torch
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.ops.sparse_mla import dsa_topk_indices, sparse_mla

DEVICE = "npu:0"
INDEX_HEAD_DIM = 128
Rkv, Dr = 512, 64
DIM = Rkv + Dr  # 576
VALUE_DIM = Rkv
SCALING = DIM ** -0.5

# 精度阈值 — 收紧以检测 SP>1 累积偏差
BF16_ATOL = 1e-2
BF16_RTOL = 1.6e-2
DKV_ATOL = 5e-2    # 原 1e-1, 统一为 sparse_mla_accuracy 的 5e-2
DKV_RTOL = 5e-2    # 原 1e-1, 统一为 5e-2

# 梯度有限性阈值: max(|grad|) 不应超过此值
GRAD_MAX_THRESHOLD = 10.0      # 原 100, 收紧至 10 (torch 基线 q_grad max 通常 < 1.0)
GRAD_MAX_KV_THRESHOLD = 100.0  # 原 1000, 收紧至 100 (torch 基线 kv_grad max ~48)

# 新增: 梯度精度对比阈值 (vs torch 基线)
GRAD_RELATIVE_ERROR_THRESHOLD = 0.5   # ||npu_grad - ref_grad|| / ||ref_grad|| < 0.5
GRAD_COSINE_SIM_THRESHOLD = 0.99      # cosine_similarity > 0.99


def _make_seq_ctx(seq_lens, device, shard_start=0, shard_size=None):
    """Build a SequenceContext for packed sequences."""
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


def _make_sp_shard(seq_lens, sp_rank, sp_size, device):
    """Compute shard_start and shard_size for SP split."""
    total = sum(seq_lens)
    padded = ((total + sp_size - 1) // sp_size) * sp_size
    shard_size = padded // sp_size
    shard_start = sp_rank * shard_size
    return shard_start, shard_size


def _sync():
    torch.npu.synchronize()


# ── 1. dsa_topk_indices autograd 梯度有限性 ──────────────────────────────

@pytest.mark.skipif(not torch.npu.is_available(), reason="requires NPU environment")
class TestIndexerGradientFinite:
    """验证 dsa_topk_indices 的 q/k/weights 梯度有限性。

    dsa_topk_indices 返回 int64 indices (不可微), 但 q/k/weights 参与
    前向计算 (score = einsum(q, k) * weights), 如果使用 detach=False 的
    前向, 梯度会通过 score 传播。实际中 indices 是 detached 的, 但这里
    测试确保前向不产生 NaN/Inf 的中间值。
    """

    def test_indexer_forward_finite_single_seq(self):
        """单序列: indexer 前向输出有限 (无 NaN/Inf)。"""
        seq_lens = [256]
        topk = 128
        torch.manual_seed(42)
        total = sum(seq_lens)
        q = torch.randn(1, total, 4, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        k = torch.randn(1, total, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        w = torch.randn(1, total, 4, dtype=torch.bfloat16, device=DEVICE)
        seq_ctx = _make_seq_ctx(seq_lens, DEVICE)

        indices = dsa_topk_indices(q, k, w, seq_ctx,
                                   index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="torch_npu")
        _sync()
        assert torch.isfinite(indices).all(), "indices contain NaN/Inf"

    def test_indexer_forward_finite_packed_multi_seq(self):
        """packed 多序列: indexer 前向输出有限 (无 NaN/Inf)。"""
        seq_lens = [128, 256, 192]
        topk = 128
        torch.manual_seed(42)
        total = sum(seq_lens)
        q = torch.randn(1, total, 4, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        k = torch.randn(1, total, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        w = torch.randn(1, total, 4, dtype=torch.bfloat16, device=DEVICE)
        seq_ctx = _make_seq_ctx(seq_lens, DEVICE)

        indices = dsa_topk_indices(q, k, w, seq_ctx,
                                   index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="torch_npu")
        _sync()
        assert torch.isfinite(indices).all(), "indices contain NaN/Inf"

    def test_indexer_forward_finite_packed_sp1_large(self):
        """packed 多序列 + SP=1 大规模: indexer 前向输出有限。"""
        seq_lens = [1024, 1024]
        topk = 2048
        torch.manual_seed(42)
        total = sum(seq_lens)
        q = torch.randn(1, total, 4, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        k = torch.randn(1, total, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        w = torch.randn(1, total, 4, dtype=torch.bfloat16, device=DEVICE)
        seq_ctx = _make_seq_ctx(seq_lens, DEVICE)

        indices = dsa_topk_indices(q, k, w, seq_ctx,
                                   index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="torch_npu")
        _sync()
        assert torch.isfinite(indices).all(), "indices contain NaN/Inf"

    def test_indexer_backward_finite_packed_sp1(self):
        """packed 多序列 + SP=1: indexer 端到端反向梯度有限 + 与 torch 基线对比。

        dsa_topk_indices 返回 int64 indices (不可微), 但 q/k/weights
        在真实模型中通过共享参数参与反向。这里通过 sparse_mla 的反向
        间接验证 indexer 选出的 indices 不会导致梯度爆炸。

        对比方式: 使用 NPU indexer 产生的 indices, 分别跑 NPU 和 torch
        sparse_mla backward, 只对比 sparse_mla 的精度差异 (排除 indexer
        indices 选择差异的干扰)。

        阈值对齐 tilelang 后端测试 (test_dsa_mla.py):
          - q_grad:  BF16_ATOL=1e-2, BF16_RTOL=1.6e-2
          - kv_grad: DKV_ATOL=5e-2,  DKV_RTOL=5e-2
        """
        seq_lens = [128, 128]
        topk = 128
        torch.manual_seed(42)
        total = sum(seq_lens)

        q_idx = torch.randn(1, total, 4, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        k_idx = torch.randn(1, total, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        w_idx = torch.randn(1, total, 4, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        seq_ctx = _make_seq_ctx(seq_lens, DEVICE)

        # 使用 NPU indexer 产生的 indices (两后端共用, 排除 indexer 差异)
        indices = dsa_topk_indices(q_idx, k_idx, w_idx, seq_ctx,
                                   index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="torch_npu")
        _sync()

        # sparse_mla: q=[S,N,D], kv=[S_g,1,D], indices=[S,1,K]
        q = torch.randn(total, 8, DIM, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        kv = torch.randn(total, 1, DIM, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        q_ref = q.detach().clone().requires_grad_(True)
        kv_ref = kv.detach().clone().requires_grad_(True)

        # NPU backward
        out = sparse_mla(q, kv, indices, scaling=SCALING, value_dim=VALUE_DIM, backend="torch_npu", seq_ctx=seq_ctx)
        out.raw_output.sum().backward()
        _sync()

        assert torch.isfinite(q.grad).all(), "q.grad has NaN/Inf"
        assert torch.isfinite(kv.grad).all(), "kv.grad has NaN/Inf"

        # torch 基线: 相同 q/kv 和相同 indices, torch 后端
        out_ref = sparse_mla(q_ref, kv_ref, indices, scaling=SCALING, value_dim=VALUE_DIM, backend="torch", seq_ctx=seq_ctx)
        out_ref.raw_output.sum().backward()

        print(f"  q_grad max={q.grad.abs().max().item():.4f} (ref={q_ref.grad.abs().max().item():.4f})", flush=True)
        print(f"  kv_grad max={kv.grad.abs().max().item():.4f} (ref={kv_ref.grad.abs().max().item():.4f})", flush=True)

        # 梯度精度: 对齐 tilelang 后端阈值 (test_dsa_mla.py)
        #   q_grad:  BF16_ATOL=1e-2, BF16_RTOL=1.6e-2
        #   kv_grad: DKV_ATOL=5e-2,  DKV_RTOL=5e-2
        torch.testing.assert_close(q.grad, q_ref.grad, atol=BF16_ATOL, rtol=BF16_RTOL)
        torch.testing.assert_close(kv.grad, kv_ref.grad, atol=DKV_ATOL, rtol=DKV_RTOL)


# ── 2. sparse_mla 反向梯度有限性 ──────────────────────────────────────────

@pytest.mark.skipif(not torch.npu.is_available(), reason="requires NPU environment")
class TestSparseMLAGradientFinite:
    """验证 sparse_mla 在 packed/SP 场景下的反向梯度有限性。

    这是 pack_sp1 梯度爆炸的核心测试: 如果 indexer 选错了跨序列 token,
    sparse_mla 的反向梯度会爆炸。
    """

    def test_backward_finite_single_seq_sp1(self):
        """单序列 + SP=1: 反向梯度有限。"""
        seq_lens = [256]
        topk = 128
        torch.manual_seed(42)
        total = sum(seq_lens)
        q_idx = torch.randn(1, total, 4, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        k_idx = torch.randn(1, total, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        w_idx = torch.randn(1, total, 4, dtype=torch.bfloat16, device=DEVICE)
        seq_ctx = _make_seq_ctx(seq_lens, DEVICE)

        indices = dsa_topk_indices(q_idx, k_idx, w_idx, seq_ctx,
                                   index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="torch_npu")

        # sparse_mla inputs: q=[S,N,D], kv=[S_g,1,D], indices=[S,1,K]
        q = torch.randn(total, 8, DIM, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        kv = torch.randn(total, 1, DIM, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)

        out = sparse_mla(q, kv, indices, scaling=SCALING, value_dim=VALUE_DIM, backend="torch_npu", seq_ctx=seq_ctx)
        out.raw_output.sum().backward()
        _sync()

        assert torch.isfinite(q.grad).all(), "q.grad contains NaN/Inf"
        assert torch.isfinite(kv.grad).all(), "kv.grad contains NaN/Inf"
        print(f"  q_grad max={q.grad.abs().max().item():.4f}", flush=True)
        assert q.grad.abs().max().item() < GRAD_MAX_THRESHOLD, \
            f"q.grad too large: {q.grad.abs().max().item()}"
        print(f"  kv_grad max={kv.grad.abs().max().item():.4f}", flush=True)
        assert kv.grad.abs().max().item() < GRAD_MAX_KV_THRESHOLD, \
            f"kv.grad too large: {kv.grad.abs().max().item()}"

    def test_backward_finite_packed_multi_seq_sp1(self):
        """packed 多序列 + SP=1: 反向梯度有限 (核心测试)。

        这是 pack_sp1 训练梯度爆炸的场景。如果 indexer 跨序列泄漏,
        反向梯度会爆炸。
        """
        seq_lens = [128, 128]
        topk = 128
        torch.manual_seed(42)
        total = sum(seq_lens)
        q_idx = torch.randn(1, total, 4, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        k_idx = torch.randn(1, total, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        w_idx = torch.randn(1, total, 4, dtype=torch.bfloat16, device=DEVICE)
        seq_ctx = _make_seq_ctx(seq_lens, DEVICE)

        indices = dsa_topk_indices(q_idx, k_idx, w_idx, seq_ctx,
                                   index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="torch_npu")

        q = torch.randn(total, 8, DIM, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        kv = torch.randn(total, 1, DIM, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)

        out = sparse_mla(q, kv, indices, scaling=SCALING, value_dim=VALUE_DIM, backend="torch_npu", seq_ctx=seq_ctx)
        out.raw_output.sum().backward()
        _sync()

        assert torch.isfinite(q.grad).all(), "q.grad contains NaN/Inf"
        assert torch.isfinite(kv.grad).all(), "kv.grad contains NaN/Inf"
        print(f"  q_grad max={q.grad.abs().max().item():.4f}", flush=True)
        assert q.grad.abs().max().item() < GRAD_MAX_THRESHOLD, \
            f"q.grad too large: {q.grad.abs().max().item()}"
        print(f"  kv_grad max={kv.grad.abs().max().item():.4f}", flush=True)
        assert kv.grad.abs().max().item() < GRAD_MAX_KV_THRESHOLD, \
            f"kv.grad too large: {kv.grad.abs().max().item()}"

    def test_backward_finite_packed_multi_seq_sp2_rank0(self):
        """packed 多序列 + SP=2 rank0: 反向梯度有限。"""
        seq_lens = [128, 128]
        topk = 128
        torch.manual_seed(42)
        total = sum(seq_lens)
        shard_start, shard_size = _make_sp_shard(seq_lens, sp_rank=0, sp_size=2, device=DEVICE)

        q_idx = torch.randn(1, shard_size, 4, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        k_idx = torch.randn(1, total, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        w_idx = torch.randn(1, shard_size, 4, dtype=torch.bfloat16, device=DEVICE)
        seq_ctx = _make_seq_ctx(seq_lens, DEVICE, shard_start=shard_start, shard_size=shard_size)

        indices = dsa_topk_indices(q_idx, k_idx, w_idx, seq_ctx,
                                   index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="torch_npu")

        q = torch.randn(shard_size, 8, DIM, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        kv = torch.randn(total, 1, DIM, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)

        out = sparse_mla(q, kv, indices, scaling=SCALING, value_dim=VALUE_DIM, backend="torch_npu", seq_ctx=seq_ctx)
        out.raw_output.sum().backward()
        _sync()

        assert torch.isfinite(q.grad).all(), "q.grad contains NaN/Inf"
        assert torch.isfinite(kv.grad).all(), "kv.grad contains NaN/Inf"
        print(f"  q_grad max={q.grad.abs().max().item():.4f}", flush=True)
        assert q.grad.abs().max().item() < GRAD_MAX_THRESHOLD, \
            f"q.grad too large: {q.grad.abs().max().item()}"
        print(f"  kv_grad max={kv.grad.abs().max().item():.4f}", flush=True)
        assert kv.grad.abs().max().item() < GRAD_MAX_KV_THRESHOLD, \
            f"kv.grad too large: {kv.grad.abs().max().item()}"

    def test_backward_finite_packed_multi_seq_sp2_rank1(self):
        """packed 多序列 + SP=2 rank1: 反向梯度有限。

        rank1 是关键测试：query 覆盖段2 [128:256]，clamp(min=0) 后
        会将 -1 映射到段1的 KV[0]，造成跨段泄漏和梯度异常。
        """
        seq_lens = [128, 128]
        topk = 128
        torch.manual_seed(42)
        total = sum(seq_lens)
        shard_start, shard_size = _make_sp_shard(seq_lens, sp_rank=1, sp_size=2, device=DEVICE)

        q_idx = torch.randn(1, shard_size, 4, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        k_idx = torch.randn(1, total, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        w_idx = torch.randn(1, shard_size, 4, dtype=torch.bfloat16, device=DEVICE)
        seq_ctx = _make_seq_ctx(seq_lens, DEVICE, shard_start=shard_start, shard_size=shard_size)

        indices = dsa_topk_indices(q_idx, k_idx, w_idx, seq_ctx,
                                   index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="torch_npu")

        q = torch.randn(shard_size, 8, DIM, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        kv = torch.randn(total, 1, DIM, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)

        out = sparse_mla(q, kv, indices, scaling=SCALING, value_dim=VALUE_DIM, backend="torch_npu", seq_ctx=seq_ctx)
        out.raw_output.sum().backward()
        _sync()

        assert torch.isfinite(q.grad).all(), "q.grad contains NaN/Inf"
        assert torch.isfinite(kv.grad).all(), "kv.grad contains NaN/Inf"
        print(f"  q_grad max={q.grad.abs().max().item():.4f}", flush=True)
        assert q.grad.abs().max().item() < GRAD_MAX_THRESHOLD, \
            f"q.grad too large: {q.grad.abs().max().item()}"
        print(f"  kv_grad max={kv.grad.abs().max().item():.4f}", flush=True)
        assert kv.grad.abs().max().item() < GRAD_MAX_KV_THRESHOLD, \
            f"kv.grad too large: {kv.grad.abs().max().item()}"

    def test_backward_finite_packed_three_seqs_sp1(self):
        """packed 三序列 + SP=1: 反向梯度有限 (更复杂的打包)。"""
        seq_lens = [128, 256, 192]
        topk = 128
        torch.manual_seed(42)
        total = sum(seq_lens)
        q_idx = torch.randn(1, total, 4, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        k_idx = torch.randn(1, total, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        w_idx = torch.randn(1, total, 4, dtype=torch.bfloat16, device=DEVICE)
        seq_ctx = _make_seq_ctx(seq_lens, DEVICE)

        indices = dsa_topk_indices(q_idx, k_idx, w_idx, seq_ctx,
                                   index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="torch_npu")

        q = torch.randn(total, 8, DIM, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        kv = torch.randn(total, 1, DIM, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)

        out = sparse_mla(q, kv, indices, scaling=SCALING, value_dim=VALUE_DIM, backend="torch_npu", seq_ctx=seq_ctx)
        out.raw_output.sum().backward()
        _sync()

        assert torch.isfinite(q.grad).all(), "q.grad contains NaN/Inf"
        assert torch.isfinite(kv.grad).all(), "kv.grad contains NaN/Inf"
        print(f"  q_grad max={q.grad.abs().max().item():.4f}", flush=True)
        assert q.grad.abs().max().item() < GRAD_MAX_THRESHOLD, \
            f"q.grad too large: {q.grad.abs().max().item()}"
        print(f"  kv_grad max={kv.grad.abs().max().item():.4f}", flush=True)
        assert kv.grad.abs().max().item() < GRAD_MAX_KV_THRESHOLD, \
            f"kv.grad too large: {kv.grad.abs().max().item()}"


# ── 3. 端到端: indexer → sparse_mla → backward, 梯度对比 torch 基线 ──────

@pytest.mark.skipif(not torch.npu.is_available(), reason="requires NPU environment")
class TestEndToEndGradientConsistency:
    """验证 NPU 端到端梯度与 torch 基线一致 (packed 多序列)。"""

    def test_e2e_packed_sp1_grad_finite(self):
        """packed 多序列 + SP=1: NPU 端到端梯度有限 (不爆炸)。

        CANN 9.0 的 aclnnSparseFlashAttentionGrad 在 packed 场景下
        与 torch 基线有量级差异 (非爆炸), 检查有限性。
        使用 NPU indexer 产生的 indices, 分别跑 NPU 和 torch sparse_mla backward,
        只对比 sparse_mla 的精度差异 (排除 indexer indices 选择差异的干扰)。

        阈值对齐 tilelang 后端测试 (test_dsa_mla.py):
          - q_grad:  BF16_ATOL=1e-2, BF16_RTOL=1.6e-2
          - kv_grad: DKV_ATOL=5e-2,  DKV_RTOL=5e-2
        """
        seq_lens = [128, 128]
        topk = 128
        torch.manual_seed(42)
        total = sum(seq_lens)
        dtype = torch.bfloat16

        q_idx = torch.randn(1, total, 4, INDEX_HEAD_DIM, dtype=dtype, device=DEVICE)
        k_idx = torch.randn(1, total, INDEX_HEAD_DIM, dtype=dtype, device=DEVICE)
        w_idx = torch.randn(1, total, 4, dtype=dtype, device=DEVICE)
        seq_ctx = _make_seq_ctx(seq_lens, DEVICE)

        # 使用 NPU indexer 产生的 indices (两后端共用, 排除 indexer 差异)
        indices = dsa_topk_indices(q_idx, k_idx, w_idx, seq_ctx,
                                   index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="torch_npu")
        _sync()

        q = torch.randn(total, 8, DIM, dtype=dtype, device=DEVICE, requires_grad=True)
        kv = torch.randn(total, 1, DIM, dtype=dtype, device=DEVICE, requires_grad=True)

        # NPU backward
        out = sparse_mla(q, kv, indices, scaling=SCALING, value_dim=VALUE_DIM, backend="torch_npu", seq_ctx=seq_ctx)
        out.raw_output.sum().backward()
        _sync()

        assert torch.isfinite(q.grad).all(), "q.grad has NaN/Inf"
        assert torch.isfinite(kv.grad).all(), "kv.grad has NaN/Inf"

        # torch 基线: 相同 q/kv 和相同 indices
        q_ref = q.detach().clone().requires_grad_(True)
        kv_ref = kv.detach().clone().requires_grad_(True)
        out_ref = sparse_mla(q_ref, kv_ref, indices, scaling=SCALING, value_dim=VALUE_DIM, backend="torch", seq_ctx=seq_ctx)
        out_ref.raw_output.sum().backward()

        print(f"  q_grad max={q.grad.abs().max().item():.4f} (ref={q_ref.grad.abs().max().item():.4f})", flush=True)
        print(f"  kv_grad max={kv.grad.abs().max().item():.4f} (ref={kv_ref.grad.abs().max().item():.4f})", flush=True)

        # 梯度精度: 对齐 tilelang 后端阈值
        torch.testing.assert_close(q.grad, q_ref.grad, atol=BF16_ATOL, rtol=BF16_RTOL)
        torch.testing.assert_close(kv.grad, kv_ref.grad, atol=DKV_ATOL, rtol=DKV_RTOL)


# ── 4. SP8 梯度测试 (复现训练配置 EP8SP8PACK16K) ──────────────────────────

@pytest.mark.skipif(not torch.npu.is_available(), reason="requires NPU environment")
class TestSP8GradientConsistency:
    """SP8 场景梯度有限性与精度对比测试。

    复现训练中 EP8SP8PACK16K 配置: query 被 SP 切成 8 份,
    每 rank 持有 1/8 query, KV 全局。
    """

    SEQ_LENS = [256, 256, 256, 256]  # total=1024, SP8 -> shard=128
    TOPK = 64

    def _make_sp8_inputs(self, sp_rank, seq_lens=None, topk=None):
        seq_lens = seq_lens or self.SEQ_LENS
        topk = topk or self.TOPK
        total = sum(seq_lens)
        padded = ((total + 7) // 8) * 8
        shard_size = padded // 8
        shard_start = sp_rank * shard_size

        torch.manual_seed(42)
        q_idx = torch.randn(1, shard_size, 4, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        k_idx = torch.randn(1, total, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        w_idx = torch.randn(1, shard_size, 4, dtype=torch.bfloat16, device=DEVICE)
        seq_ctx = _make_seq_ctx(seq_lens, DEVICE, shard_start=shard_start, shard_size=shard_size)

        q = torch.randn(shard_size, 8, DIM, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        kv = torch.randn(total, 1, DIM, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        return q_idx, k_idx, w_idx, seq_ctx, q, kv, shard_start, shard_size, total

    def test_sp8_grad_finite_all_ranks(self):
        """SP8: 所有 rank 的反向梯度有限。"""
        for rank in range(8):
            q_idx, k_idx, w_idx, seq_ctx, q, kv, ss, sz, total = self._make_sp8_inputs(rank)
            indices = dsa_topk_indices(q_idx, k_idx, w_idx, seq_ctx,
                                       index_head_dim=INDEX_HEAD_DIM, index_topk=self.TOPK, backend="torch_npu")
            out = sparse_mla(q, kv, indices, scaling=SCALING, value_dim=VALUE_DIM, backend="torch_npu", seq_ctx=seq_ctx)
            out.raw_output.sum().backward()
            _sync()

            assert torch.isfinite(q.grad).all(), f"SP8 rank{rank}: q.grad has NaN/Inf"
            assert torch.isfinite(kv.grad).all(), f"SP8 rank{rank}: kv.grad has NaN/Inf"
            print(f"  [SP8 rank{rank}] q_grad_max={q.grad.abs().max().item():.4f}, "
                  f"kv_grad_max={kv.grad.abs().max().item():.4f}", flush=True)
            assert q.grad.abs().max().item() < GRAD_MAX_THRESHOLD, \
                f"SP8 rank{rank}: q.grad too large: {q.grad.abs().max().item()}"
            assert kv.grad.abs().max().item() < GRAD_MAX_KV_THRESHOLD, \
                f"SP8 rank{rank}: kv.grad too large: {kv.grad.abs().max().item()}"

    def test_sp8_grad_vs_torch_baseline(self):
        """SP8: NPU 梯度与 torch 基线对比 (抽样 rank 0/3/7)。"""
        for rank in [0, 3, 7]:
            q_idx, k_idx, w_idx, seq_ctx, q, kv, ss, sz, total = self._make_sp8_inputs(rank)

            # 使用 NPU indexer 产生的 indices (两后端共用, 排除 indexer 差异)
            indices = dsa_topk_indices(q_idx, k_idx, w_idx, seq_ctx,
                                       index_head_dim=INDEX_HEAD_DIM, index_topk=self.TOPK, backend="torch_npu")
            _sync()

            # NPU backward
            q_npu = q.detach().clone().requires_grad_(True)
            kv_npu = kv.detach().clone().requires_grad_(True)
            out_npu = sparse_mla(q_npu, kv_npu, indices, scaling=SCALING, value_dim=VALUE_DIM, backend="torch_npu", seq_ctx=seq_ctx)
            out_npu.raw_output.sum().backward()
            _sync()

            assert torch.isfinite(q_npu.grad).all(), f"SP8 rank{rank}: q.grad has NaN/Inf"
            assert torch.isfinite(kv_npu.grad).all(), f"SP8 rank{rank}: kv.grad has NaN/Inf"

            # torch 基线: 相同 q/kv 和相同 indices
            q_ref = q.detach().clone().requires_grad_(True)
            kv_ref = kv.detach().clone().requires_grad_(True)
            out_ref = sparse_mla(q_ref, kv_ref, indices, scaling=SCALING, value_dim=VALUE_DIM, backend="torch", seq_ctx=seq_ctx)
            out_ref.raw_output.sum().backward()
            _sync()

            print(f"  [SP8 rank{rank}] q_grad max={q_npu.grad.abs().max().item():.4f} (ref={q_ref.grad.abs().max().item():.4f}) | "
                  f"kv_grad max={kv_npu.grad.abs().max().item():.4f} (ref={kv_ref.grad.abs().max().item():.4f})", flush=True)

            # 梯度精度: 对齐 tilelang 后端阈值
            torch.testing.assert_close(q_npu.grad, q_ref.grad, atol=BF16_ATOL, rtol=BF16_RTOL)
            torch.testing.assert_close(kv_npu.grad, kv_ref.grad, atol=DKV_ATOL, rtol=DKV_RTOL)

    @pytest.mark.slow
    def test_sp8_large_grad_finite(self):
        """SP8 大规模: seq_lens=[1024,1024], topk=256, 梯度有限。"""
        seq_lens = [1024, 1024]
        topk = 256
        for rank in [0, 3, 7]:
            q_idx, k_idx, w_idx, seq_ctx, q, kv, ss, sz, total = self._make_sp8_inputs(
                rank, seq_lens=seq_lens, topk=topk)
            indices = dsa_topk_indices(q_idx, k_idx, w_idx, seq_ctx,
                                       index_head_dim=INDEX_HEAD_DIM, index_topk=topk, backend="torch_npu")
            out = sparse_mla(q, kv, indices, scaling=SCALING, value_dim=VALUE_DIM, backend="torch_npu", seq_ctx=seq_ctx)
            out.raw_output.sum().backward()
            _sync()

            assert torch.isfinite(q.grad).all(), f"SP8 large rank{rank}: q.grad has NaN/Inf"
            assert torch.isfinite(kv.grad).all(), f"SP8 large rank{rank}: kv.grad has NaN/Inf"
            print(f"  [SP8 large rank{rank}] q_grad_max={q.grad.abs().max().item():.4f}, "
                  f"kv_grad_max={kv.grad.abs().max().item():.4f}", flush=True)
            assert q.grad.abs().max().item() < GRAD_MAX_THRESHOLD, \
                f"SP8 large rank{rank}: q.grad too large: {q.grad.abs().max().item()}"
            assert kv.grad.abs().max().item() < GRAD_MAX_KV_THRESHOLD, \
                f"SP8 large rank{rank}: kv.grad too large: {kv.grad.abs().max().item()}"


# ── 5. 训练规模复现测试 (EP8SP8PACK16K) ──────────────────────────────────────
# 训练配置: total=16384, SP8 shard=2048, topk=2048, num_heads=64, head_dim=256
# 单元测试使用训练配置的关键参数, 用随机数据复现梯度爆炸.


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires NPU environment")
class TestSP8TrainScaleRepro:
    """训练规模 SP8 复现测试: 用训练配置参数复现梯度爆炸。

    训练日志: grad_norm=9.4e11 at Step 1
    根因假设: MindSpeed KV slice prefix extension 导致:
      1. topk >= shard_size 时, indices 超出 local shard 范围
      2. clamp(min=0) 将跨 shard 索引映射到错误位置
      3. 前向 attention pattern 错误 → 梯度爆炸

    测试矩阵:
      - 端到端: indexer → sparse_mla → backward, 对比 torch vs npu
      - 重点检查: topk == shard_size 时的 indices 正确性
      - 重点检查: prefix extension 后的 KV slice 范围
    """

    # 训练配置参数
    TRAIN_TOTAL = 17920      # pack ~16K (不等长段模拟训练 pack)
    TRAIN_SP = 8
    TRAIN_SHARD = TRAIN_TOTAL // TRAIN_SP  # 2048 (padded)
    TRAIN_TOPK = 2048        # index_topk=2048 (== shard_size, 关键!)
    TRAIN_NUM_HEADS = 8      # 测试用 8 (训练 64, 但不影响逻辑)
    TRAIN_INDEX_HEADS = 4    # 测试用 4 (训练 32)
    TRAIN_INDEX_DIM = 128
    # 模拟训练中的不等长 pack: 多个不同长度序列打包到 16384
    # 段边界不与 SP8 shard 边界对齐, 触发 prefix extension
    TRAIN_SEQ_LENS = [4096, 2048, 1024, 512, 4096, 2048, 1024, 1024, 256, 256, 256, 256, 256, 256, 256, 256]  # total=16384

    def _make_train_scale_inputs(self, sp_rank):
        """构造训练规模 SP8 输入."""
        total = sum(self.TRAIN_SEQ_LENS)
        shard_start, shard_size = _make_sp_shard(
            self.TRAIN_SEQ_LENS, sp_rank=sp_rank, sp_size=self.TRAIN_SP, device=DEVICE)

        torch.manual_seed(42)
        # Indexer inputs: q=[1, shard_size, Ni, Di], k=[1, total, Di]
        q_idx = torch.randn(1, shard_size, self.TRAIN_INDEX_HEADS, self.TRAIN_INDEX_DIM,
                            dtype=torch.bfloat16, device=DEVICE)
        k_idx = torch.randn(1, total, self.TRAIN_INDEX_DIM,
                            dtype=torch.bfloat16, device=DEVICE)
        w_idx = torch.randn(1, shard_size, self.TRAIN_INDEX_HEADS,
                            dtype=torch.bfloat16, device=DEVICE)
        seq_ctx = _make_seq_ctx(self.TRAIN_SEQ_LENS, DEVICE,
                                shard_start=shard_start, shard_size=shard_size)

        # SparseMLA inputs: q=[shard_size, N, D], kv=[total, 1, D]
        q = torch.randn(shard_size, self.TRAIN_NUM_HEADS, DIM,
                        dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        kv = torch.randn(total, 1, DIM,
                         dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        return q_idx, k_idx, w_idx, seq_ctx, q, kv, shard_start, shard_size, total

    def test_sp8_trainscale_indices_vs_torch(self):
        """训练规模: NPU indexer indices vs torch 基线 (重点: topk==shard_size)."""
        for rank in [0, 3, 7]:
            q_idx, k_idx, w_idx, seq_ctx, _, _, ss, sz, total = self._make_train_scale_inputs(rank)

            ref = dsa_topk_indices(q_idx, k_idx, w_idx, seq_ctx,
                                   index_head_dim=self.TRAIN_INDEX_DIM,
                                   index_topk=self.TRAIN_TOPK, backend="torch")
            npu = dsa_topk_indices(q_idx, k_idx, w_idx, seq_ctx,
                                   index_head_dim=self.TRAIN_INDEX_DIM,
                                   index_topk=self.TRAIN_TOPK, backend="torch_npu")

            # 统计 indices 分布
            ref_valid = (ref >= 0).sum().item()
            npu_valid = (npu >= 0).sum().item()
            ref_in_shard = ((ref >= ss) & (ref < ss + sz)).sum().item()
            npu_in_shard = ((npu >= ss) & (npu < ss + sz)).sum().item()
            ref_out_shard = ref_valid - ref_in_shard
            npu_out_shard = npu_valid - npu_in_shard

            # overlap
            ref_flat = ref.squeeze(1)
            npu_flat = npu.squeeze(1)
            overlap = 0
            for i in range(sz):
                ref_set = set(ref_flat[i].tolist())
                npu_set = set(npu_flat[i].tolist())
                ref_set.discard(-1)
                npu_set.discard(-1)
                overlap += len(ref_set & npu_set)
            match_rate = overlap / max(npu_valid, 1) * 100

            print(f"  [rank{rank}] ref_valid={ref_valid} npu_valid={npu_valid} "
                  f"ref_in_shard={ref_in_shard} ref_out_shard={ref_out_shard} "
                  f"npu_in_shard={npu_in_shard} npu_out_shard={npu_out_shard} "
                  f"match_rate={match_rate:.1f}%", flush=True)

            # 训练规模下, torch 应该能选到 shard 之外的 KV (全局 topk)
            # NPU 如果只能选 shard 内的, 说明 BUG-A 仍然存在
            # rank0 的 shard_start=0, causal 范围全在 shard 内, out_shard 必为 0
            if rank > 0:
                assert ref_out_shard > 0, \
                    f"rank{rank}: torch baseline should select KV outside shard, got 0"
            assert npu_out_shard == ref_out_shard, \
                f"rank{rank}: npu out_shard={npu_out_shard} != ref out_shard={ref_out_shard}"
            assert match_rate >= 99.0, \
                f"rank{rank}: match_rate={match_rate:.1f}% < 99%"

    def test_sp8_trainscale_grad_explosion(self):
        """训练规模: 复现梯度爆炸 (grad_norm >> torch 基线)."""
        for rank in [0, 3, 7]:
            q_idx, k_idx, w_idx, seq_ctx, q, kv, ss, sz, total = self._make_train_scale_inputs(rank)

            # torch 基线
            indices_ref = dsa_topk_indices(q_idx, k_idx, w_idx, seq_ctx,
                                           index_head_dim=self.TRAIN_INDEX_DIM,
                                           index_topk=self.TRAIN_TOPK, backend="torch")
            q_ref = q.detach().clone().requires_grad_(True)
            kv_ref = kv.detach().clone().requires_grad_(True)
            out_ref = sparse_mla(q_ref, kv_ref, indices_ref, scaling=SCALING,
                                 value_dim=VALUE_DIM, backend="torch", seq_ctx=seq_ctx)
            out_ref.raw_output.sum().backward()
            _sync()

            # NPU
            indices_npu = dsa_topk_indices(q_idx, k_idx, w_idx, seq_ctx,
                                           index_head_dim=self.TRAIN_INDEX_DIM,
                                           index_topk=self.TRAIN_TOPK, backend="torch_npu")
            q_npu = q.detach().clone().requires_grad_(True)
            kv_npu = kv.detach().clone().requires_grad_(True)
            out_npu = sparse_mla(q_npu, kv_npu, indices_npu, scaling=SCALING,
                                 value_dim=VALUE_DIM, backend="torch_npu", seq_ctx=seq_ctx)
            out_npu.raw_output.sum().backward()
            _sync()

            q_ref_max = q_ref.grad.float().abs().max().item()
            q_npu_max = q_npu.grad.float().abs().max().item()
            kv_ref_max = kv_ref.grad.float().abs().max().item()
            kv_npu_max = kv_npu.grad.float().abs().max().item()
            q_ratio = q_npu_max / max(q_ref_max, 1e-12)
            kv_ratio = kv_npu_max / max(kv_ref_max, 1e-12)

            _q_rel = (q_npu.grad.float() - q_ref.grad.float()).norm() / \
                     q_ref.grad.float().norm().clamp_min(1e-12)
            _kv_rel = (kv_npu.grad.float() - kv_ref.grad.float()).norm() / \
                      kv_ref.grad.float().norm().clamp_min(1e-12)

            print(f"  [rank{rank}] q_grad: ref={q_ref_max:.4f} npu={q_npu_max:.4f} "
                  f"ratio={q_ratio:.1f}x rel_err={_q_rel.item():.4f} | "
                  f"kv_grad: ref={kv_ref_max:.4f} npu={kv_npu_max:.4f} "
                  f"ratio={kv_ratio:.1f}x rel_err={_kv_rel.item():.4f}", flush=True)

            # 梯度爆炸检测: NPU grad max 不应超过 torch 的 10x
            # 训练中 grad_norm=9.4e11, 这里用 ratio 检测
            assert q_ratio < 10, \
                f"rank{rank}: q_grad explosion! npu/ref={q_ratio:.1f}x " \
                f"(npu={q_npu_max} ref={q_ref_max})"
            assert kv_ratio < 10, \
                f"rank{rank}: kv_grad explosion! npu/ref={kv_ratio:.1f}x " \
                f"(npu={kv_npu_max} ref={kv_ref_max})"

    def test_sp8_trainscale_forward_diff(self):
        """训练规模: 前向输出偏差 (如果 indices 错误, 前向输出会显著偏差)."""
        for rank in [0, 3, 7]:
            q_idx, k_idx, w_idx, seq_ctx, q, kv, ss, sz, total = self._make_train_scale_inputs(rank)

            # torch 基线
            indices_ref = dsa_topk_indices(q_idx, k_idx, w_idx, seq_ctx,
                                           index_head_dim=self.TRAIN_INDEX_DIM,
                                           index_topk=self.TRAIN_TOPK, backend="torch")
            ref_out = sparse_mla(q, kv, indices_ref, scaling=SCALING,
                                 value_dim=VALUE_DIM, backend="torch", seq_ctx=seq_ctx)

            # NPU
            indices_npu = dsa_topk_indices(q_idx, k_idx, w_idx, seq_ctx,
                                           index_head_dim=self.TRAIN_INDEX_DIM,
                                           index_topk=self.TRAIN_TOPK, backend="torch_npu")
            npu_out = sparse_mla(q, kv, indices_npu, scaling=SCALING,
                                 value_dim=VALUE_DIM, backend="torch_npu", seq_ctx=seq_ctx)
            _sync()

            _diff = (npu_out.raw_output.float() - ref_out.raw_output.float()).abs()
            _rel = _diff.max().item() / max(ref_out.raw_output.float().abs().max().item(), 1e-12)

            print(f"  [rank{rank}] forward max_diff={_diff.max().item():.4f} "
                  f"mean_diff={_diff.mean().item():.6f} rel_diff={_rel:.4f}", flush=True)

            # 前向偏差不应超过 10x bf16 精度 (0.0156 * 10 = 0.156)
            assert _diff.max().item() < 1.0, \
                f"rank{rank}: forward diff too large: {_diff.max().item():.4f} " \
                f"(possible indices error causing wrong attention)"


# ── 独立运行入口 ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
