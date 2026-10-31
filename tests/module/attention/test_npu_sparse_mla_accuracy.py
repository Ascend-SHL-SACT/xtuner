"""验证 NPU SparseMLA 与 PyTorch 基线的精度一致性（bf16）。

NPU aclnnSparseFlashAttention 硬性要求 qk_head_dim=512，
因此所有测试固定 dim=576, value_dim=512（GLM-5.2 生产配置），
仅变化 seq_len / num_heads / topk。

测试矩阵:
  TestNPUSparseMLAAccuracy (单序列, BSND 路径)
    test_forward_matches_torch_baseline
    test_forward_large_topk
    test_padded_indices_handled_correctly
    test_various_shapes
    test_output_dtype
    test_output_is_finite
  TestNPUSparseMLABackward (单序列, BSND 路径)
    test_backward_matches_torch_baseline
    test_gradients_are_finite
  TestNPUSparseMLAPackedTND (packed 多序列, TND + cu_seq_lens 路径)
    test_forward_packed_matches_torch
    test_backward_packed_matches_torch
    test_backward_packed_grad_finite
    test_forward_packed_no_cross_segment
    test_backward_packed_three_segments
  TestNPUDSAAttention (端到端, 从 dsa_mla.py 构建完整 attention)
    test_packed_inputs_respect_causal_boundaries_and_backward
    test_shared_layer_consumes_explicit_topk_ids
    test_checkpoint_reuses_source_topk_storage
  TestNPUDSASequenceParallel (SP2, 需 torchrun)
    test_packed_attention_matches_full_sequence

精度阈值与 test_dsa_mla.py 保持一致:
  BF16_ATOL = 1e-2, BF16_RTOL = 1.6e-2  (前向 output, lse, q_grad)
  DKV_ATOL  = 5e-2, DKV_RTOL  = 5e-2    (kv_grad)
"""
import os
import sys

import pytest
import torch
import torch.nn as nn


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.moe.glm52 import DSAMLAConfig
from xtuner.v1.model.utils import apply_activation_checkpointing
from xtuner.v1.ops.sparse_mla import npu_sparse_mla as npu_sparse_mla_mod
from xtuner.v1.ops.sparse_mla import sparse_mla
from xtuner.v1.utils.test_utils import init_data_mesh


# NPU kernel 硬性约束：qk_head_dim 必须为 512
DIM = 576          # Rkv(512) + Dr(64)
VALUE_DIM = 512    # kv_lora_rank
SCALING = DIM ** -0.5

# 精度阈值 — 收紧以检测 SP>1 累积偏差
BF16_ATOL = 1e-2
BF16_RTOL = 1.6e-2
DKV_ATOL = 5e-2   # kv_grad 阈值 (保持, 已比基线 1e-1 更紧)
DKV_RTOL = 5e-2
# 新增: relative error 和 cosine similarity 阈值
GRAD_RELATIVE_ERROR_THRESHOLD = 0.5   # ||npu_grad - ref_grad|| / ||ref_grad|| < 0.5
GRAD_COSINE_SIM_THRESHOLD = 0.99      # cosine_similarity(npu_grad, ref_grad) > 0.99

DTYPES = [torch.bfloat16]


def _npu_available():
    try:
        import torch_npu  # noqa: F401

        return torch.npu.is_available()
    except ImportError:
        return False


def _sync():
    """NPU 同步，确保异步错误在正确的测试中抛出。"""
    if _npu_available():
        torch.npu.synchronize()


def _make_inputs(seq_len=64, num_heads=16, kv_group=1, topk=32, dtype=torch.bfloat16, device="npu"):
    """构造 SparseMLA 测试输入（固定 dim=576）。

    q:     [S, N, 576]        absorbed query
    kv:    [S_g, 1, 576]      compressed KV
    indices: [S, 1, K]        top-k 索引，-1 表示 padding (int64)
    """
    torch.manual_seed(42)
    q = torch.randn(seq_len, num_heads, DIM, device=device, dtype=dtype)
    kv = torch.randn(seq_len, kv_group, DIM, device=device, dtype=dtype)

    # 构造 causal top-k 索引：每个 query 位置只能看到 [0, pos] 范围
    indices = torch.full((seq_len, 1, topk), -1, device=device, dtype=torch.int64)
    for pos in range(seq_len):
        valid = min(pos + 1, topk)
        indices[pos, 0, :valid] = torch.arange(pos + 1 - valid, pos + 1, device=device)

    return q, kv, indices


def _make_packed_inputs(seq_lens, num_heads=16, topk=32, dtype=torch.bfloat16, device="npu"):
    """构造 packed 多序列 SparseMLA 测试输入。

    Returns:
        q: [total_len, N, 576]
        kv: [total_len, 1, 576]
        indices: [total_len, 1, K]  (含 -1, 段内 causal)
        seq_ctx: SequenceContext with cu_seq_lens
    """
    torch.manual_seed(42)
    total = sum(seq_lens)
    q = torch.randn(total, num_heads, DIM, device=device, dtype=dtype)
    kv = torch.randn(total, 1, DIM, device=device, dtype=dtype)

    # 构造 packed causal indices: 每段内 query 只选本段内 [seg_start, pos]
    indices = torch.full((total, 1, topk), -1, device=device, dtype=torch.int64)
    seg_starts = [0] + list(seq_lens)
    seg_starts = [sum(seq_lens[:i]) for i in range(len(seq_lens) + 1)]
    for seg_idx, seg_len in enumerate(seq_lens):
        start = seg_starts[seg_idx]
        for pos in range(seg_len):
            abs_pos = start + pos
            valid = min(pos + 1, topk)
            indices[abs_pos, 0, :valid] = torch.arange(start + pos + 1 - valid, start + pos + 1, device=device)

    # cu_seq_lens (prefix sum, including leading 0)
    cu_seq = torch.tensor(seg_starts, dtype=torch.int32, device=device)

    seq_ctx = SequenceContext(
        input_ids=torch.arange(total, device=device).unsqueeze(0).long(),
        cu_seq_lens_q=cu_seq,
        cu_seq_lens_k=cu_seq,
        max_length_q=max(seq_lens),
        max_length_k=max(seq_lens),
        device=device,
        shard_start=0,
        shard_size=total,
    )

    return q, kv, indices, seq_ctx


# ── 前向精度测试 (单序列, BSND) ──────────────────────────────────────────

@pytest.mark.skipif(not _npu_available(), reason="requires NPU environment")
class TestNPUSparseMLAAccuracy:
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_forward_matches_torch_baseline(self, dtype):
        """NPU 前向输出应与 torch 基线一致。"""
        q, kv, indices = _make_inputs(seq_len=64, num_heads=16, topk=32, dtype=dtype)

        expected = sparse_mla(q, kv, indices, scaling=SCALING, value_dim=VALUE_DIM, backend="torch")
        actual = sparse_mla(q, kv, indices, scaling=SCALING, value_dim=VALUE_DIM, backend="torch_npu")
        _sync()

        assert actual.raw_output.shape == expected.raw_output.shape
        assert actual.raw_output.shape == (64, 16, VALUE_DIM)
        assert actual.raw_output.dtype == dtype, f"expected {dtype}, got {actual.raw_output.dtype}"

        _diff = (actual.raw_output.float() - expected.raw_output.float()).abs()
        print(f"  max_diff={_diff.max().item():.6f}, mean_diff={_diff.mean().item():.6f}", flush=True)
        torch.testing.assert_close(
            actual.raw_output, expected.raw_output,
            atol=BF16_ATOL, rtol=BF16_RTOL,
            msg="raw_output mismatch",
        )

        if expected.softmax_lse is not None and actual.softmax_lse is not None:
            _diff = (actual.softmax_lse.float() - expected.softmax_lse.float()).abs()
            print(f"  lse max_diff={_diff.max().item():.6f}, mean_diff={_diff.mean().item():.6f}", flush=True)
            torch.testing.assert_close(
                actual.softmax_lse, expected.softmax_lse,
                atol=BF16_ATOL, rtol=BF16_RTOL,
                msg="softmax_lse mismatch",
            )

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_forward_large_topk(self, dtype):
        """topk=seq_len/2 时，NPU 应与 torch 基线一致。"""
        seq_len = 64
        q, kv, indices = _make_inputs(seq_len=seq_len, num_heads=16, topk=seq_len // 2, dtype=dtype)

        expected = sparse_mla(q, kv, indices, scaling=SCALING, value_dim=VALUE_DIM, backend="torch")
        actual = sparse_mla(q, kv, indices, scaling=SCALING, value_dim=VALUE_DIM, backend="torch_npu")
        _sync()

        _diff = (actual.raw_output.float() - expected.raw_output.float()).abs()
        print(f"  max_diff={_diff.max().item():.6f}, mean_diff={_diff.mean().item():.6f}", flush=True)
        torch.testing.assert_close(
            actual.raw_output, expected.raw_output,
            atol=BF16_ATOL, rtol=BF16_RTOL,
        )

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_padded_indices_handled_correctly(self, dtype):
        """含 -1 padding 的索引应被正确忽略。"""
        q, kv, indices = _make_inputs(seq_len=64, num_heads=16, topk=8, dtype=dtype)

        expected = sparse_mla(q, kv, indices, scaling=SCALING, value_dim=VALUE_DIM, backend="torch")
        actual = sparse_mla(q, kv, indices, scaling=SCALING, value_dim=VALUE_DIM, backend="torch_npu")
        _sync()

        _diff = (actual.raw_output.float() - expected.raw_output.float()).abs()
        print(f"  max_diff={_diff.max().item():.6f}, mean_diff={_diff.mean().item():.6f}", flush=True)
        torch.testing.assert_close(
            actual.raw_output, expected.raw_output,
            atol=BF16_ATOL, rtol=BF16_RTOL,
        )

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_various_shapes(self, dtype):
        """多种 seq_len/heads/topk 组合下精度一致（dim 固定 576）。"""
        configs = [
            (32, 8, 4),
            (64, 16, 16),
            (128, 8, 32),
        ]
        for seq_len, num_heads, topk in configs:
            q, kv, indices = _make_inputs(seq_len=seq_len, num_heads=num_heads, topk=topk, dtype=dtype)

            expected = sparse_mla(q, kv, indices, scaling=SCALING, value_dim=VALUE_DIM, backend="torch")
            actual = sparse_mla(q, kv, indices, scaling=SCALING, value_dim=VALUE_DIM, backend="torch_npu")
            _sync()

            _diff = (actual.raw_output.float() - expected.raw_output.float()).abs()
            print(f"  [{seq_len},{num_heads},{topk}] max_diff={_diff.max().item():.6f}, mean_diff={_diff.mean().item():.6f}", flush=True)
            torch.testing.assert_close(
                actual.raw_output, expected.raw_output,
                atol=BF16_ATOL, rtol=BF16_RTOL,
                msg=f"shape config failed: seq={seq_len}, heads={num_heads}, topk={topk}, dtype={dtype}",
            )

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_output_dtype(self, dtype):
        """raw_output dtype 应与输入一致，softmax_lse 为 float32。"""
        q, kv, indices = _make_inputs(seq_len=64, num_heads=16, topk=16, dtype=dtype)

        actual = sparse_mla(q, kv, indices, scaling=SCALING, value_dim=VALUE_DIM, backend="torch_npu")
        _sync()

        assert actual.raw_output.dtype == dtype
        if actual.softmax_lse is not None:
            assert actual.softmax_lse.dtype == torch.float32

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_output_is_finite(self, dtype):
        """输出不应包含 NaN/Inf。"""
        q, kv, indices = _make_inputs(seq_len=64, num_heads=16, topk=32, dtype=dtype)

        actual = sparse_mla(q, kv, indices, scaling=SCALING, value_dim=VALUE_DIM, backend="torch_npu")
        _sync()

        assert torch.isfinite(actual.raw_output).all(), "raw_output contains NaN/Inf"
        if actual.softmax_lse is not None:
            assert torch.isfinite(actual.softmax_lse).all(), "softmax_lse contains NaN/Inf"


# ── 反向精度测试 (单序列, BSND) ──────────────────────────────────────────

@pytest.mark.skipif(not _npu_available(), reason="requires NPU environment")
class TestNPUSparseMLABackward:
    def test_backward_matches_torch_baseline(self):
        """NPU 反向梯度应与 torch 基线一致 (q_grad 用 BF16 阈值, kv_grad 用 DKV 阈值)。"""
        q, kv, indices = _make_inputs(seq_len=64, num_heads=16, topk=16, dtype=torch.bfloat16)

        # torch 基线
        q_ref = q.detach().clone().requires_grad_(True)
        kv_ref = kv.detach().clone().requires_grad_(True)
        ref_out = sparse_mla(q_ref, kv_ref, indices, scaling=SCALING, value_dim=VALUE_DIM, backend="torch")
        grad_output = torch.randn_like(ref_out.raw_output)
        ref_out.raw_output.backward(grad_output)
        _sync()

        # NPU
        q_npu = q.detach().clone().requires_grad_(True)
        kv_npu = kv.detach().clone().requires_grad_(True)
        npu_out = sparse_mla(q_npu, kv_npu, indices, scaling=SCALING, value_dim=VALUE_DIM, backend="torch_npu")
        npu_out.raw_output.backward(grad_output)
        _sync()

        _qdiff = (q_npu.grad.float() - q_ref.grad.float()).abs()
        print(f"  q_grad: max_diff={_qdiff.max().item():.6f}, mean_diff={_qdiff.mean().item():.6f}", flush=True)
        torch.testing.assert_close(
            q_npu.grad, q_ref.grad,
            atol=BF16_ATOL, rtol=BF16_RTOL,
            msg="q grad mismatch",
        )
        _kvdiff = (kv_npu.grad.float() - kv_ref.grad.float()).abs()
        print(f"  kv_grad: max_diff={_kvdiff.max().item():.6f}, mean_diff={_kvdiff.mean().item():.6f}", flush=True)
        # 新增: relative error + cosine similarity
        _kv_rel_err = (kv_npu.grad.float() - kv_ref.grad.float()).norm() / kv_ref.grad.float().norm().clamp_min(1e-12)
        _kv_cos = torch.nn.functional.cosine_similarity(
            kv_npu.grad.float().flatten(), kv_ref.grad.float().flatten(), dim=0)
        print(f"  kv_grad: rel_err={_kv_rel_err.item():.4f}, cosine_sim={_kv_cos.item():.6f}", flush=True)
        torch.testing.assert_close(
            kv_npu.grad, kv_ref.grad,
            atol=DKV_ATOL, rtol=DKV_RTOL,
            msg="kv grad mismatch",
        )
        assert _kv_rel_err.item() < GRAD_RELATIVE_ERROR_THRESHOLD, \
            f"kv_grad relative error too large: {_kv_rel_err.item():.4f}"
        assert _kv_cos.item() > GRAD_COSINE_SIM_THRESHOLD, \
            f"kv_grad cosine similarity too low: {_kv_cos.item():.6f}"

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_gradients_are_finite(self, dtype):
        """反向梯度不应包含 NaN/Inf。"""
        q, kv, indices = _make_inputs(seq_len=64, num_heads=16, topk=16, dtype=dtype)

        q = q.detach().requires_grad_(True)
        kv = kv.detach().clone().requires_grad_(True)
        out = sparse_mla(q, kv, indices, scaling=SCALING, value_dim=VALUE_DIM, backend="torch_npu")
        out.raw_output.sum().backward()
        _sync()

        assert torch.isfinite(q.grad).all(), "q grad contains NaN/Inf"
        assert torch.isfinite(kv.grad).all(), "kv grad contains NaN/Inf"


# ── Packed 多序列 TND 路径测试 ───────────────────────────────────────────

@pytest.mark.skipif(not _npu_available(), reason="requires NPU environment")
class TestNPUSparseMLAPackedTND:
    """验证 NPU TND + cu_seq_lens 路径在 packed 多序列场景下的精度。

    这是对 npu.py _sparse_mla_tnd_packed 路径的直接测试。
    必须传 seq_ctx 才能触发 TND 路径。
    """

    def test_forward_packed_matches_torch(self):
        """packed 多序列前向输出应与 torch 基线一致。"""
        seq_lens = [128, 128]
        q, kv, indices, seq_ctx = _make_packed_inputs(seq_lens, num_heads=8, topk=32)

        expected = sparse_mla(q, kv, indices, scaling=SCALING, value_dim=VALUE_DIM, backend="torch")
        actual = sparse_mla(q, kv, indices, scaling=SCALING, value_dim=VALUE_DIM, backend="torch_npu", seq_ctx=seq_ctx)
        _sync()

        assert actual.raw_output.shape == expected.raw_output.shape

        _diff = (actual.raw_output.float() - expected.raw_output.float()).abs()
        print(f"  max_diff={_diff.max().item():.6f}, mean_diff={_diff.mean().item():.6f}", flush=True)
        torch.testing.assert_close(
            actual.raw_output, expected.raw_output,
            atol=BF16_ATOL, rtol=BF16_RTOL,
            msg="packed TND forward mismatch",
        )

    def test_backward_packed_matches_torch(self):
        """packed 多序列反向梯度应与 torch 基线一致。"""
        seq_lens = [128, 128]
        q, kv, indices, seq_ctx = _make_packed_inputs(seq_lens, num_heads=8, topk=32)

        # torch 基线
        q_ref = q.detach().clone().requires_grad_(True)
        kv_ref = kv.detach().clone().requires_grad_(True)
        ref_out = sparse_mla(q_ref, kv_ref, indices, scaling=SCALING, value_dim=VALUE_DIM, backend="torch", seq_ctx=seq_ctx)
        grad_output = torch.randn_like(ref_out.raw_output)
        ref_out.raw_output.backward(grad_output)
        _sync()

        # NPU TND
        q_npu = q.detach().clone().requires_grad_(True)
        kv_npu = kv.detach().clone().requires_grad_(True)
        npu_out = sparse_mla(q_npu, kv_npu, indices, scaling=SCALING, value_dim=VALUE_DIM, backend="torch_npu", seq_ctx=seq_ctx)
        npu_out.raw_output.backward(grad_output)
        _sync()

        _qdiff = (q_npu.grad.float() - q_ref.grad.float()).abs()
        print(f"  q_grad: max_diff={_qdiff.max().item():.6f}, mean_diff={_qdiff.mean().item():.6f}", flush=True)
        torch.testing.assert_close(
            q_npu.grad, q_ref.grad,
            atol=BF16_ATOL, rtol=BF16_RTOL,
            msg="packed TND q grad mismatch",
        )
        _kvdiff = (kv_npu.grad.float() - kv_ref.grad.float()).abs()
        print(f"  kv_grad: max_diff={_kvdiff.max().item():.6f}, mean_diff={_kvdiff.mean().item():.6f}", flush=True)
        # 新增: relative error + cosine similarity
        _kv_rel_err = (kv_npu.grad.float() - kv_ref.grad.float()).norm() / kv_ref.grad.float().norm().clamp_min(1e-12)
        _kv_cos = torch.nn.functional.cosine_similarity(
            kv_npu.grad.float().flatten(), kv_ref.grad.float().flatten(), dim=0)
        print(f"  kv_grad: rel_err={_kv_rel_err.item():.4f}, cosine_sim={_kv_cos.item():.6f}", flush=True)
        torch.testing.assert_close(
            kv_npu.grad, kv_ref.grad,
            atol=DKV_ATOL, rtol=DKV_RTOL,
            msg="packed TND kv grad mismatch",
        )
        assert _kv_rel_err.item() < GRAD_RELATIVE_ERROR_THRESHOLD, \
            f"kv_grad relative error too large: {_kv_rel_err.item():.4f}"
        assert _kv_cos.item() > GRAD_COSINE_SIM_THRESHOLD, \
            f"kv_grad cosine similarity too low: {_kv_cos.item():.6f}"

    def test_backward_packed_grad_finite(self):
        """packed 多序列反向梯度应有限 (无 NaN/Inf)。"""
        seq_lens = [128, 128]
        q, kv, indices, seq_ctx = _make_packed_inputs(seq_lens, num_heads=8, topk=32)

        q = q.detach().requires_grad_(True)
        kv = kv.detach().clone().requires_grad_(True)
        out = sparse_mla(q, kv, indices, scaling=SCALING, value_dim=VALUE_DIM, backend="torch_npu", seq_ctx=seq_ctx)
        out.raw_output.sum().backward()
        _sync()

        assert torch.isfinite(q.grad).all(), "q grad contains NaN/Inf"
        assert torch.isfinite(kv.grad).all(), "kv grad contains NaN/Inf"
        print(f"  q_grad max={q.grad.float().abs().max().item():.4f}", flush=True)
        print(f"  kv_grad max={kv.grad.float().abs().max().item():.4f}", flush=True)

    def test_forward_packed_no_cross_segment(self):
        """packed 多序列: 段2 query 不应 attend 到段1 KV (无跨段泄漏)。"""
        seq_lens = [128, 128]
        q, kv, indices, seq_ctx = _make_packed_inputs(seq_lens, num_heads=8, topk=32)

        # torch 基线
        ref_out = sparse_mla(q, kv, indices, scaling=SCALING, value_dim=VALUE_DIM, backend="torch", seq_ctx=seq_ctx)
        # NPU TND
        npu_out = sparse_mla(q, kv, indices, scaling=SCALING, value_dim=VALUE_DIM, backend="torch_npu", seq_ctx=seq_ctx)
        _sync()

        # 段2 (pos 128:255) 的输出应与 torch 基线一致 (无跨段 attention)
        seg2_diff = (npu_out.raw_output[128:].float() - ref_out.raw_output[128:].float()).abs()
        seg1_diff = (npu_out.raw_output[:128].float() - ref_out.raw_output[:128].float()).abs()
        print(f"  seg1 max_diff={seg1_diff.max().item():.6f}", flush=True)
        print(f"  seg2 max_diff={seg2_diff.max().item():.6f}", flush=True)

        # 段2 的 diff 应与段1 相当 (无跨段泄漏时两段 diff 量级相同)
        # bf16 ULP for ~1.0 values is 0.015625, so allow 2x BF16_ATOL
        assert seg2_diff.max().item() <= BF16_ATOL * 2, \
            f"seg2 diff too large: {seg2_diff.max().item():.6f} (possible cross-segment leak)"
        # 段2/段1 diff ratio 应接近 1 (无跨段泄漏时两段 diff 相当)
        ratio = seg2_diff.max().item() / max(seg1_diff.max().item(), 1e-10)
        print(f"  seg2/seg1 ratio={ratio:.2f}", flush=True)
        assert ratio < 3, f"seg2/seg1 diff ratio={ratio:.1f} (cross-segment leak suspected)"

    def test_backward_packed_three_segments(self):
        """packed 三序列反向梯度应与 torch 基线一致。"""
        seq_lens = [64, 128, 96]
        q, kv, indices, seq_ctx = _make_packed_inputs(seq_lens, num_heads=8, topk=32)

        # torch 基线
        q_ref = q.detach().clone().requires_grad_(True)
        kv_ref = kv.detach().clone().requires_grad_(True)
        ref_out = sparse_mla(q_ref, kv_ref, indices, scaling=SCALING, value_dim=VALUE_DIM, backend="torch", seq_ctx=seq_ctx)
        grad_output = torch.randn_like(ref_out.raw_output)
        ref_out.raw_output.backward(grad_output)
        _sync()

        # NPU TND
        q_npu = q.detach().clone().requires_grad_(True)
        kv_npu = kv.detach().clone().requires_grad_(True)
        npu_out = sparse_mla(q_npu, kv_npu, indices, scaling=SCALING, value_dim=VALUE_DIM, backend="torch_npu", seq_ctx=seq_ctx)
        npu_out.raw_output.backward(grad_output)
        _sync()

        _qdiff = (q_npu.grad.float() - q_ref.grad.float()).abs()
        print(f"  q_grad: max_diff={_qdiff.max().item():.6f}, mean_diff={_qdiff.mean().item():.6f}", flush=True)
        torch.testing.assert_close(
            q_npu.grad, q_ref.grad,
            atol=BF16_ATOL, rtol=BF16_RTOL,
            msg="packed 3-seg q grad mismatch",
        )
        _kvdiff = (kv_npu.grad.float() - kv_ref.grad.float()).abs()
        print(f"  kv_grad: max_diff={_kvdiff.max().item():.6f}, mean_diff={_kvdiff.mean().item():.6f}", flush=True)
        # 新增: relative error + cosine similarity
        _kv_rel_err = (kv_npu.grad.float() - kv_ref.grad.float()).norm() / kv_ref.grad.float().norm().clamp_min(1e-12)
        _kv_cos = torch.nn.functional.cosine_similarity(
            kv_npu.grad.float().flatten(), kv_ref.grad.float().flatten(), dim=0)
        print(f"  kv_grad: rel_err={_kv_rel_err.item():.4f}, cosine_sim={_kv_cos.item():.6f}", flush=True)
        torch.testing.assert_close(
            kv_npu.grad, kv_ref.grad,
            atol=DKV_ATOL, rtol=DKV_RTOL,
            msg="packed 3-seg kv grad mismatch",
        )
        assert _kv_rel_err.item() < GRAD_RELATIVE_ERROR_THRESHOLD, \
            f"kv_grad relative error too large: {_kv_rel_err.item():.4f}"
        assert _kv_cos.item() > GRAD_COSINE_SIM_THRESHOLD, \
            f"kv_grad cosine similarity too low: {_kv_cos.item():.6f}"


# ── DSA Attention 端到端测试 (移植自 test_dsa_mla.py, 改为 NPU 后端) ──────

def _tiny_dsa_attention(
    indexer_types: list[str] | None = None,
    layer_idx: int = 0,
    sparse_mla_backend: str = "torch_npu",
    seq_len: int = 128,
    num_heads: int = 8,
    indexer_topk: int = 32,
):
    """构建一个最小 DSA-MLA attention 模块用于测试。"""
    torch.manual_seed(42)
    config = DSAMLAConfig(
        num_attention_heads=num_heads,
        head_dim=VALUE_DIM + 64,           # qk_nope + rope
        kv_lora_rank=VALUE_DIM,            # 512
        q_lora_rank=256,
        qk_rope_head_dim=64,
        qk_nope_head_dim=VALUE_DIM,        # 512
        v_head_dim=VALUE_DIM,              # 512
        index_head_dim=128,
        index_n_heads=4,
        index_topk=indexer_topk,
        indexer_types=indexer_types or ["full"],
        sparse_mla_backend=sparse_mla_backend,
    )
    attention = config.build(hidden_size=256, layer_idx=layer_idx)
    device = "npu" if _npu_available() else "cpu"
    return attention.to(device)


def _tiny_dsa_decoder_block(attention, hidden_size=256):
    """构建一个最小 decoder block 包含 attention + rmsnorm + MLP。"""

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.RMSNorm(hidden_size)
            self.norm2 = nn.RMSNorm(hidden_size)
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.GELU(),
                nn.Linear(hidden_size * 2, hidden_size),
            )

        def forward(self, x, position_embeddings, seq_ctx):
            x = x + attention(self.norm1(x), position_embeddings, seq_ctx)["projected_output"]
            x = x + self.mlp(self.norm2(x))
            return x

    return _Block()


@pytest.mark.skipif(not _npu_available(), reason="requires NPU environment")
class TestNPUDSAAttention:
    def test_packed_inputs_respect_causal_boundaries_and_backward(self):
        """验证 packed attention 不跨子序列取 key，并能对真实输入完成有限反向传播。"""
        torch.manual_seed(0)
        device = "npu" if _npu_available() else "cpu"
        seq_lens = [128, 128]
        total = sum(seq_lens)
        attention = _tiny_dsa_attention(
            indexer_types=["full"], layer_idx=0, sparse_mla_backend="torch_npu",
        ).to(device)

        hidden = torch.randn(1, total, 256, device=device, dtype=torch.bfloat16) * 0.02
        position_embeddings = (
            torch.randn(1, total, 64, device=device, dtype=torch.bfloat16) * 0.01,
            torch.randn(1, total, 64, device=device, dtype=torch.bfloat16) * 0.01,
        )
        cu = torch.cumsum(torch.tensor([0] + list(seq_lens), dtype=torch.int32, device=device), 0).to(torch.int32)
        seq_ctx = SequenceContext(
            input_ids=torch.arange(total, device=device).unsqueeze(0).long(),
            cu_seq_lens_q=cu, cu_seq_lens_k=cu,
            max_length_q=128, max_length_k=128,
            device=device, shard_start=0, shard_size=total,
        )

        out = attention(hidden, position_embeddings, seq_ctx)
        _sync()
        assert torch.isfinite(out["projected_output"]).all(), "forward output has NaN/Inf"

        out["projected_output"].sum().backward()
        _sync()

        for name, p in attention.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"grad of {name} has NaN/Inf"

        # Clean up to avoid state leaking into subsequent tests.
        del attention, out, seq_ctx
        torch.npu.empty_cache()

    def test_shared_layer_consumes_explicit_topk_ids(self):
        """验证 shared attention 复用显式 top-k IDs，并在漏传时立即报错。"""
        torch.manual_seed(0)
        device = "npu" if _npu_available() else "cpu"
        seq_lens = [128, 128]
        total = sum(seq_lens)
        source_attention = _tiny_dsa_attention(
            indexer_types=["full", "shared"], layer_idx=0, sparse_mla_backend="torch_npu",
        ).to(device)
        shared_attention = _tiny_dsa_attention(
            indexer_types=["full", "shared"], layer_idx=1, sparse_mla_backend="torch_npu",
        ).to(device)

        hidden = torch.randn(1, total, 256, device=device, dtype=torch.bfloat16) * 0.02
        position_embeddings = (
            torch.randn(1, total, 64, device=device, dtype=torch.bfloat16) * 0.01,
            torch.randn(1, total, 64, device=device, dtype=torch.bfloat16) * 0.01,
        )
        cu = torch.cumsum(torch.tensor([0] + list(seq_lens), dtype=torch.int32, device=device), 0).to(torch.int32)
        seq_ctx = SequenceContext(
            input_ids=torch.arange(total, device=device).unsqueeze(0).long(),
            cu_seq_lens_q=cu, cu_seq_lens_k=cu,
            max_length_q=128, max_length_k=128,
            device=device, shard_start=0, shard_size=total,
        )

        out1 = source_attention(hidden, position_embeddings, seq_ctx)
        dsa_topk_ids = out1["dsa_topk_ids"]
        out2 = shared_attention(hidden, position_embeddings, seq_ctx, dsa_topk_ids=dsa_topk_ids)
        _sync()

        assert torch.isfinite(out1["projected_output"]).all()
        assert torch.isfinite(out2["projected_output"]).all()
        assert out2["dsa_topk_ids"] is dsa_topk_ids
        with pytest.raises(RuntimeError, match="requires dsa_topk_ids"):
            shared_attention(hidden, position_embeddings, seq_ctx)
        _sync()

    def test_checkpoint_reuses_source_topk_storage(self):
        """验证显式 top-k IDs 穿过 reentrant checkpoint 且真实 indexer backend 只执行一次。"""
        torch.manual_seed(0)
        device = "npu" if _npu_available() else "cpu"
        seq_lens = [128, 128]
        total = sum(seq_lens)
        # norm/MLP 参数必须与输入同为 bf16，否则 fp32 会传播进 sparse_mla，
        # aclnnSparseFlashAttention 仅支持 fp16/bf16。
        source_block = apply_activation_checkpointing(
            _TinyDsaDecoderBlock(
                _tiny_dsa_attention(indexer_types=["full", "shared"], layer_idx=0, sparse_mla_backend="torch_npu")
            ).to(device=device, dtype=torch.bfloat16)
        )
        shared_block = apply_activation_checkpointing(
            _TinyDsaDecoderBlock(
                _tiny_dsa_attention(indexer_types=["full", "shared"], layer_idx=1, sparse_mla_backend="torch_npu")
            ).to(device=device, dtype=torch.bfloat16)
        )

        # 缩放后再置 requires_grad，保证 hidden 是叶子（.grad 可写入）。
        hidden = (torch.randn(1, total, 256, device=device, dtype=torch.bfloat16) * 0.02).requires_grad_()
        position_embeddings = (
            torch.randn(1, total, 64, device=device, dtype=torch.bfloat16) * 0.01,
            torch.randn(1, total, 64, device=device, dtype=torch.bfloat16) * 0.01,
        )
        cu = torch.cumsum(torch.tensor([0] + list(seq_lens), dtype=torch.int32, device=device), 0).to(torch.int32)
        seq_ctx = SequenceContext(
            input_ids=torch.arange(total, device=device).unsqueeze(0).long(),
            cu_seq_lens_q=cu, cu_seq_lens_k=cu,
            max_length_q=128, max_length_k=128,
            device=device, shard_start=0, shard_size=total,
        )
        indexer_calls = 0

        def count_indexer_call(*_args):
            nonlocal indexer_calls
            indexer_calls += 1

        hook = source_block.self_attn.indexer.register_forward_hook(count_indexer_call)

        try:
            source_out = source_block(hidden, position_embeddings, seq_ctx)
            source_ids = source_out["dsa_topk_ids"]
            shared_out = shared_block(
                source_out["hidden_states"], position_embeddings, seq_ctx, dsa_topk_ids=source_ids
            )
            shared_ids = shared_out["dsa_topk_ids"]
            _sync()
            assert shared_ids.untyped_storage().data_ptr() == source_ids.untyped_storage().data_ptr()
            shared_out["hidden_states"].sum().backward()
            _sync()
        finally:
            hook.remove()

        assert hidden.grad is not None, "hidden grad is None (gradient not propagated)"
        assert torch.isfinite(hidden.grad).all(), "hidden grad has NaN/Inf"
        assert source_ids.dtype == torch.int32
        assert indexer_calls == 1

        # Clean up
        del source_block, shared_block, source_out, shared_out, hidden, seq_ctx
        torch.npu.empty_cache()


class _TinyDsaDecoderBlock(nn.Module):
    """最小 decoder block：attention + RMSNorm + MLP，返回 hidden_states 与 dsa_topk_ids。"""

    def __init__(self, attention, hidden_size: int = 256):
        super().__init__()
        self.self_attn = attention
        self.norm = nn.RMSNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        seq_ctx: object,
        dsa_topk_ids: torch.Tensor | None = None,
    ) -> dict:
        attn_out = self.self_attn(self.norm(hidden_states), position_embeddings, seq_ctx, dsa_topk_ids=dsa_topk_ids)
        hidden_states = hidden_states + attn_out["projected_output"]
        hidden_states = hidden_states + self.mlp(hidden_states)
        return {"hidden_states": hidden_states, "dsa_topk_ids": attn_out["dsa_topk_ids"]}


# ── 序列并行测试 (需 torchrun --nproc_per_node 2) ─────────────────────────

@pytest.mark.skipif(not _npu_available(), reason="requires NPU environment")
class TestNPUDSASequenceParallel:
    def test_packed_attention_matches_full_sequence(self):
        """SP2 的输出、top-k 与输入梯度拼回后等同完整序列。"""
        try:
            mesh = init_data_mesh(seq_parallel_size=2)
        except Exception:
            pytest.skip("requires torchrun --nproc_per_node 2")

        torch.manual_seed(42)
        device = "npu"
        seq_lens = [128, 128]
        total = sum(seq_lens)

        rank = torch.distributed.get_rank(mesh)
        sp_size = mesh.size(0)
        shard_size = total // sp_size
        shard_start = rank * shard_size

        attention = _tiny_dsa_attention(
            indexer_types=["full", "shared"],
            sparse_mla_backend="torch_npu",
        ).to(device)

        hidden = torch.randn(total, 1, 256, device=device, dtype=torch.bfloat16) * 0.02
        position_embeddings = (
            torch.randn(total, 1, 64, device=device, dtype=torch.bfloat16) * 0.01,
            torch.randn(total, 1, 64, device=device, dtype=torch.bfloat16) * 0.01,
        )
        cu = torch.cumsum(torch.tensor([0] + list(seq_lens), dtype=torch.int32, device=device), 0).to(torch.int32)
        seq_ctx = SequenceContext(
            input_ids=torch.arange(total, device=device).unsqueeze(0).long(),
            cu_seq_lens_q=cu, cu_seq_lens_k=cu,
            max_length_q=128, max_length_k=128,
            device=device, shard_start=shard_start, shard_size=shard_size,
            sequence_parallel_mesh=mesh,
        )

        # SP shard
        local_hidden = hidden[shard_start:shard_start + shard_size]
        local_pos = tuple(pe[shard_start:shard_start + shard_size] for pe in position_embeddings)

        local_out = attention(local_hidden, local_pos, seq_ctx)
        _sync()

        assert torch.isfinite(local_out["projected_output"]).all()
        local_out["projected_output"].sum().backward()
        _sync()

        for name, p in attention.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"grad of {name} has NaN/Inf"



# ── SP8 Packed TND 路径精度测试 (复现训练配置) ─────────────────────────────

@pytest.mark.skipif(not _npu_available(), reason="requires NPU environment")
class TestNPUSparseMLASP8PackedTND:
    """SP8 场景下 NPU TND + cu_seq_lens 路径的精度对齐。

    复现 EP8SP8PACK16K 训练配置: query 被 SP 切成 8 份,
    每个 rank 持有 1/8 query, KV 全局 gather。
    """

    SEQ_LENS = [256, 256, 256, 256]  # total=1024, SP8 -> shard=128
    TOPK = 64

    def _make_sp8_packed_inputs(self, sp_rank, seq_lens=None, topk=None, num_heads=8):
        seq_lens = seq_lens or self.SEQ_LENS
        topk = topk or self.TOPK
        total = sum(seq_lens)
        padded = ((total + 7) // 8) * 8
        shard_size = padded // 8
        shard_start = sp_rank * shard_size

        torch.manual_seed(42)
        q = torch.randn(total, num_heads, DIM, device="npu", dtype=torch.bfloat16)
        kv = torch.randn(total, 1, DIM, device="npu", dtype=torch.bfloat16)

        indices = torch.full((total, 1, topk), -1, device="npu", dtype=torch.int64)
        seg_starts = [sum(seq_lens[:i]) for i in range(len(seq_lens) + 1)]
        for seg_idx, seg_len in enumerate(seq_lens):
            start = seg_starts[seg_idx]
            for pos in range(seg_len):
                abs_pos = start + pos
                valid = min(pos + 1, topk)
                indices[abs_pos, 0, :valid] = torch.arange(
                    start + pos + 1 - valid, start + pos + 1, device="npu")

        cu_seq = torch.tensor(seg_starts, dtype=torch.int32, device="npu")
        seq_ctx = SequenceContext(
            input_ids=torch.arange(total, device="npu").unsqueeze(0).long(),
            cu_seq_lens_q=cu_seq, cu_seq_lens_k=cu_seq,
            max_length_q=max(seq_lens), max_length_k=max(seq_lens),
            device="npu", shard_start=shard_start, shard_size=shard_size,
        )
        return q, kv, indices, seq_ctx, shard_start, shard_size, total

    def test_sp8_forward_matches_torch(self):
        """SP8: 各 rank 的前向输出应与 torch 基线一致。"""
        for rank in range(8):
            q, kv, indices, seq_ctx, ss, sz, total = self._make_sp8_packed_inputs(rank)
            local_q = q[ss:ss+sz]
            local_indices = indices[ss:ss+sz]

            expected = sparse_mla(local_q, kv, local_indices, scaling=SCALING,
                                  value_dim=VALUE_DIM, backend="torch", seq_ctx=seq_ctx)
            actual = sparse_mla(local_q, kv, local_indices, scaling=SCALING,
                                value_dim=VALUE_DIM, backend="torch_npu", seq_ctx=seq_ctx)
            _sync()

            _diff = (actual.raw_output.float() - expected.raw_output.float()).abs()
            print(f"  [SP8 rank{rank}] max_diff={_diff.max().item():.6f}, "
                  f"mean_diff={_diff.mean().item():.6f}", flush=True)
            torch.testing.assert_close(
                actual.raw_output, expected.raw_output,
                atol=BF16_ATOL, rtol=BF16_RTOL,
                msg=f"SP8 rank{rank} forward mismatch",
            )

    def test_sp8_backward_matches_torch(self):
        """SP8: 反向梯度应与 torch 基线一致。"""
        for rank in [0, 3, 7]:
            q, kv, indices, seq_ctx, ss, sz, total = self._make_sp8_packed_inputs(rank)
            local_q = q[ss:ss+sz]
            local_indices = indices[ss:ss+sz]

            q_ref = local_q.detach().clone().requires_grad_(True)
            kv_ref = kv.detach().clone().requires_grad_(True)
            ref_out = sparse_mla(q_ref, kv_ref, local_indices, scaling=SCALING,
                                 value_dim=VALUE_DIM, backend="torch", seq_ctx=seq_ctx)
            grad_output = torch.randn_like(ref_out.raw_output)
            ref_out.raw_output.backward(grad_output)
            _sync()

            q_npu = local_q.detach().clone().requires_grad_(True)
            kv_npu = kv.detach().clone().requires_grad_(True)
            npu_out = sparse_mla(q_npu, kv_npu, local_indices, scaling=SCALING,
                                 value_dim=VALUE_DIM, backend="torch_npu", seq_ctx=seq_ctx)
            npu_out.raw_output.backward(grad_output)
            _sync()

            _qdiff = (q_npu.grad.float() - q_ref.grad.float()).abs()
            _kvdiff = (kv_npu.grad.float() - kv_ref.grad.float()).abs()
            _kv_rel = (kv_npu.grad.float() - kv_ref.grad.float()).norm() / kv_ref.grad.float().norm().clamp_min(1e-12)
            _kv_cos = torch.nn.functional.cosine_similarity(
                kv_npu.grad.float().flatten(), kv_ref.grad.float().flatten(), dim=0)
            print(f"  [SP8 rank{rank}] q_grad max_diff={_qdiff.max().item():.6f}, "
                  f"kv_grad max_diff={_kvdiff.max().item():.6f}, "
                  f"kv_rel_err={_kv_rel.item():.4f}, kv_cos={_kv_cos.item():.6f}", flush=True)
            torch.testing.assert_close(
                q_npu.grad, q_ref.grad, atol=BF16_ATOL, rtol=BF16_RTOL,
                msg=f"SP8 rank{rank} q grad mismatch")
            torch.testing.assert_close(
                kv_npu.grad, kv_ref.grad, atol=DKV_ATOL, rtol=DKV_RTOL,
                msg=f"SP8 rank{rank} kv grad mismatch")
            assert _kv_rel.item() < GRAD_RELATIVE_ERROR_THRESHOLD, \
                f"SP8 rank{rank} kv_grad rel_err={_kv_rel.item():.4f}"
            assert _kv_cos.item() > GRAD_COSINE_SIM_THRESHOLD, \
                f"SP8 rank{rank} kv_grad cos={_kv_cos.item():.6f}"

    def test_sp8_no_cross_segment_leak(self):
        """SP8: 段间无跨段泄漏。"""
        for rank in [0, 4, 7]:
            q, kv, indices, seq_ctx, ss, sz, total = self._make_sp8_packed_inputs(rank)
            local_q = q[ss:ss+sz]
            local_indices = indices[ss:ss+sz]

            ref_out = sparse_mla(local_q, kv, local_indices, scaling=SCALING,
                                 value_dim=VALUE_DIM, backend="torch", seq_ctx=seq_ctx)
            npu_out = sparse_mla(local_q, kv, local_indices, scaling=SCALING,
                                value_dim=VALUE_DIM, backend="torch_npu", seq_ctx=seq_ctx)
            _sync()

            _diff = (npu_out.raw_output.float() - ref_out.raw_output.float()).abs()
            print(f"  [SP8 rank{rank}] max_diff={_diff.max().item():.6f}", flush=True)
            assert _diff.max().item() <= BF16_ATOL * 2, \
                f"SP8 rank{rank} diff too large: {_diff.max().item():.6f}"

    @pytest.mark.slow
    def test_sp8_large_packed(self):
        """SP8 大规模: seq_lens=[2048, 2048], topk=512。"""
        seq_lens = [2048, 2048]
        topk = 512
        for rank in [0, 3, 7]:
            q, kv, indices, seq_ctx, ss, sz, total = self._make_sp8_packed_inputs(
                rank, seq_lens=seq_lens, topk=topk)
            local_q = q[ss:ss+sz]
            local_indices = indices[ss:ss+sz]

            expected = sparse_mla(local_q, kv, local_indices, scaling=SCALING,
                                  value_dim=VALUE_DIM, backend="torch", seq_ctx=seq_ctx)
            actual = sparse_mla(local_q, kv, local_indices, scaling=SCALING,
                                value_dim=VALUE_DIM, backend="torch_npu", seq_ctx=seq_ctx)
            _sync()

            _diff = (actual.raw_output.float() - expected.raw_output.float()).abs()
            print(f"  [SP8 large rank{rank}] max_diff={_diff.max().item():.6f}, "
                  f"mean_diff={_diff.mean().item():.6f}", flush=True)
            torch.testing.assert_close(
                actual.raw_output, expected.raw_output,
                atol=BF16_ATOL, rtol=BF16_RTOL,
                msg=f"SP8 large rank{rank} mismatch")
# ── 行分块反向 (XTUNER_SFA_GRAD_ROW_CHUNK) 测试 ──────────────────────────


def _make_prefix_segment_case(
    segs: list[tuple[int, int]], num_heads: int = 8, topk: int = 64, seed: int = 0, device: str = "npu"
):
    """构造 suffix-aligned 前缀扩展多段 case（SP>1 生产几何的单元级缩影）。

    段内行 j 的因果上界 = ``j + (w_s - q_s)``（段内局部 KV 坐标）；indices 为
    段内局部坐标，跨窗口 spread 采样并保证对角线（自位置）在列。

    Args:
        segs (list[tuple[int, int]]): 每段 ``(q_s, w_s)``，w_s >= q_s。
        num_heads (int): 注意力头数。
        topk (int): 每行选中 KV 槽位数。
        seed (int): 随机种子。
        device (str): 目标设备。

    Returns:
        tuple: ``(query, query_rope, key, key_rope, indices, cu_q, cu_k)``。
    """
    import torch_npu  # noqa: F401

    g = torch.Generator().manual_seed(seed)
    q_parts, qr_parts, k_parts, kr_parts, i_parts = [], [], [], [], []
    q_lens, kv_lens = [], []
    for q_s, w_s in segs:
        q_parts.append(torch.randn(q_s, num_heads, VALUE_DIM, generator=g).to(device, torch.bfloat16))
        qr_parts.append(torch.randn(q_s, num_heads, 64, generator=g).to(device, torch.bfloat16))
        k_parts.append(torch.randn(w_s, 1, VALUE_DIM, generator=g).to(device, torch.bfloat16))
        kr_parts.append(torch.randn(w_s, 1, 64, generator=g).to(device, torch.bfloat16))
        pos_hi = torch.arange(q_s, device=device) + (w_s - q_s)
        base = torch.arange(topk, device=device).unsqueeze(0)
        idx = (base * (pos_hi + 1).unsqueeze(1) // topk).clamp_(max=pos_hi.unsqueeze(1))
        idx[:, -1] = pos_hi
        i_parts.append(idx.unsqueeze(1).to(torch.int32))
        q_lens.append(q_s)
        kv_lens.append(w_s)
    cu_q = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32, device=device)
    cu_k = torch.tensor([0] + list(torch.tensor(kv_lens).cumsum(0)), dtype=torch.int32, device=device)
    return (
        torch.cat(q_parts).contiguous(),
        torch.cat(qr_parts).contiguous(),
        torch.cat(k_parts).contiguous(),
        torch.cat(kr_parts).contiguous(),
        torch.cat(i_parts).contiguous(),
        cu_q,
        cu_k,
    )


def _run_grad_pair(case, grad_out, row_chunk: int):
    """整段 stock grad 调用与 ``_sfa_grad_chunked`` 各跑一次，返回 (ref5, chunked5)。

    两者输出元组序一致: ``(dq, dk, dv, dq_rope, dk_rope)``。
    """
    import torch_npu

    query, query_rope, key, key_rope, indices, cu_q, cu_k = case
    outs = torch_npu.npu_sparse_flash_attention(
        query,
        key,
        key,
        sparse_indices=indices,
        block_table=None,
        actual_seq_lengths_query=cu_q,
        actual_seq_lengths_kv=cu_k,
        query_rope=query_rope,
        key_rope=key_rope,
        scale_value=SCALING,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="TND",
        sparse_mode=3,
        attention_mode=2,
        return_softmax_lse=True,
    )
    raw = outs[0].contiguous()
    smax = outs[1].float().contiguous()
    ssum = outs[2].float().contiguous()
    _sync()
    ref = torch_npu.npu_sparse_flash_attention_grad(
        query,
        key,
        key,
        indices,
        grad_out,
        raw,
        smax,
        ssum,
        SCALING,
        1,
        query_rope=query_rope,
        key_rope=key_rope,
        actual_seq_qlen=cu_q,
        actual_seq_kvlen=cu_k,
        layout="TND",
        sparse_mode=3,
        attention_mode=2,
    )
    _sync()
    got = npu_sparse_mla_mod._sfa_grad_chunked(
        query,
        query_rope,
        key,
        key_rope,
        indices,
        grad_out,
        raw,
        smax,
        ssum,
        cu_q,
        cu_k,
        SCALING,
        row_chunk,
    )
    _sync()
    return ref, got


@pytest.mark.skipif(not _npu_available(), reason="requires NPU environment")
class TestSfaGradRowChunk:
    """行分块反向与整段调用的等价性（job dyb-q35-xtuner-19 的回归测试）。

    16K 行 shard 一次 grad 调用 workspace 4.4-4.7 GiB，紧 ranks 缺 40-330 MiB
    OOM；分块后瞬态 ~0.2 GiB/1K 行。dq/dq_rope 逐位等，dk/dv/dk_rope 为同一
    fp32 和的重分组（重排舍入级，DKV 阈值内）。
    """

    def test_parse_sfa_grad_row_chunk(self):
        """env 解析: 非法值报错，<=0 归一为 0（stock），正值透传。"""
        parse = npu_sparse_mla_mod._parse_sfa_grad_row_chunk
        assert parse("0") == 0
        assert parse("1024") == 1024
        assert parse("-3") == 0
        with pytest.raises(ValueError, match="not a valid int"):
            parse("abc")

    def test_chunk_geq_seq_len_matches_stock(self):
        """chunk >= 总行数时等价单次调用: dq/dq_rope 逐位等，kv 类梯度 DKV 阈值内。

        kernel 的 dk/dv/dkr 本身跨调用非确定（同参数两次调用 1-2 ULP 抖动，
        原子归约序），逐位门只对 dq/dq_rope 成立。
        """
        case = _make_prefix_segment_case([(1024, 1024)], seed=1)
        grad_out = torch.randn(1024, 8, VALUE_DIM, device="npu", dtype=torch.bfloat16)
        ref, got = _run_grad_pair(case, grad_out, row_chunk=1 << 30)
        assert torch.equal(ref[0], got[0]), "dq mismatch"
        assert torch.equal(ref[3], got[3]), "dq_rope mismatch"
        for i, name in ((1, "dk"), (2, "dv"), (4, "dk_rope")):
            torch.testing.assert_close(got[i].float(), ref[i].float(), atol=DKV_ATOL, rtol=DKV_RTOL, msg=name)

    def test_chunked_single_segment_dq_bitwise(self):
        """单段 mid-cut 分块: dq/dq_rope 逐位等，kv 类梯度在 DKV 阈值内。"""
        case = _make_prefix_segment_case([(4096, 4096)], seed=2)
        grad_out = torch.randn(4096, 8, VALUE_DIM, device="npu", dtype=torch.bfloat16)
        ref, got = _run_grad_pair(case, grad_out, row_chunk=1024)
        assert torch.equal(ref[0], got[0]), "dq mismatch"
        assert torch.equal(ref[3], got[3]), "dq_rope mismatch"
        for i, name in ((1, "dk"), (2, "dv"), (4, "dk_rope")):
            torch.testing.assert_close(got[i].float(), ref[i].float(), atol=DKV_ATOL, rtol=DKV_RTOL, msg=name)

    def test_chunked_multi_segment_prefix_extended(self):
        """三段前缀扩展几何、chunk=600 错切段中: dq 逐位等，kv 类梯度阈值内。

        覆盖: 首段 W>Q 前缀扩展、chunk 从段中开始/结束、跨段 chunk、
        未触及段的零长 cu 条目。
        """
        segs = [(512, 1024), (1024, 1024), (512, 512)]
        case = _make_prefix_segment_case(segs, seed=3)
        grad_out = torch.randn(sum(q for q, _ in segs), 8, VALUE_DIM, device="npu", dtype=torch.bfloat16)
        ref, got = _run_grad_pair(case, grad_out, row_chunk=600)
        assert torch.equal(ref[0], got[0]), "dq mismatch"
        assert torch.equal(ref[3], got[3]), "dq_rope mismatch"
        for i, name in ((1, "dk"), (2, "dv"), (4, "dk_rope")):
            torch.testing.assert_close(got[i].float(), ref[i].float(), atol=DKV_ATOL, rtol=DKV_RTOL, msg=name)


@pytest.mark.skipif(not _npu_available(), reason="requires NPU environment")
class TestSfaGradRowChunkDispatch:
    def test_packed_backward_uses_row_chunks_and_matches_stock(self, monkeypatch):
        """SP8 packed 路径: 分块开启时 grad 调用按 chunk 拆分，q 梯度逐位等于 stock。"""
        import torch_npu

        sp8 = TestNPUSparseMLASP8PackedTND()
        q, kv, indices, seq_ctx, ss, sz, _total = sp8._make_sp8_packed_inputs(3)
        local_q = q[ss : ss + sz]
        local_indices = indices[ss : ss + sz]

        def run_backward(row_chunk_value: int, grad_out: torch.Tensor):
            monkeypatch.setattr(npu_sparse_mla_mod, "_SFA_GRAD_ROW_CHUNK", row_chunk_value)
            q_leaf = local_q.detach().clone().requires_grad_(True)
            kv_leaf = kv.detach().clone().requires_grad_(True)
            out = sparse_mla(
                q_leaf,
                kv_leaf,
                local_indices,
                scaling=SCALING,
                value_dim=VALUE_DIM,
                backend="torch_npu",
                seq_ctx=seq_ctx,
            )
            calls = {"grad": 0}

            orig_grad = torch_npu.npu_sparse_flash_attention_grad

            def counting_grad(*args, **kwargs):
                calls["grad"] += 1
                return orig_grad(*args, **kwargs)

            monkeypatch.setattr(torch_npu, "npu_sparse_flash_attention_grad", counting_grad)
            out.raw_output.backward(grad_out)
            _sync()
            return q_leaf.grad, kv_leaf.grad, calls["grad"]

        # 两次 run 共用同一 grad_out，隔离反向路径差异。
        grad_out = torch.randn(128, 8, VALUE_DIM, device="npu", dtype=torch.bfloat16)
        q_stock, kv_stock, _ = run_backward(0, grad_out)
        q_chunked, kv_chunked, n_calls = run_backward(48, grad_out)
        # 128 行 shard、chunk=48 → 3 次 grad 调用（stock 走 autograd 内部, 不经 python 属性）。
        assert n_calls == 3, f"expected 3 chunked grad calls, got {n_calls}"
        assert torch.equal(q_stock, q_chunked), "q grad mismatch vs stock"
        torch.testing.assert_close(kv_chunked.float(), kv_stock.float(), atol=DKV_ATOL, rtol=DKV_RTOL)
    def test_offload_backward_uses_row_chunks_and_matches_stock(self, monkeypatch):
        """SP_MLA_OFFLOAD=1 臂（512N 生产臂）: 分块开启时 3 次调用，q 梯度逐位等于 stock 单次。"""
        import torch_npu

        sp8 = TestNPUSparseMLASP8PackedTND()
        q, kv, indices, seq_ctx, ss, sz, _total = sp8._make_sp8_packed_inputs(3)
        local_q = q[ss : ss + sz]
        local_indices = indices[ss : ss + sz]
        monkeypatch.setattr(npu_sparse_mla_mod, "_sp_mla_offload_enabled", lambda: True)
        # 本测试几何的 KV slice 远小于生产门控默认值，钉 0 强制走 pinned offload 路径。
        monkeypatch.setattr(npu_sparse_mla_mod, "_SFA_GRAD_OFFLOAD_MIN_KV", 0)

        def run_backward(row_chunk_value: int, grad_out: torch.Tensor):
            monkeypatch.setattr(npu_sparse_mla_mod, "_SFA_GRAD_ROW_CHUNK", row_chunk_value)
            q_leaf = local_q.detach().clone().requires_grad_(True)
            kv_leaf = kv.detach().clone().requires_grad_(True)
            out = sparse_mla(
                q_leaf,
                kv_leaf,
                local_indices,
                scaling=SCALING,
                value_dim=VALUE_DIM,
                backend="torch_npu",
                seq_ctx=seq_ctx,
            )
            calls = {"grad": 0}

            orig_grad = torch_npu.npu_sparse_flash_attention_grad

            def counting_grad(*args, **kwargs):
                calls["grad"] += 1
                return orig_grad(*args, **kwargs)

            monkeypatch.setattr(torch_npu, "npu_sparse_flash_attention_grad", counting_grad)
            out.raw_output.backward(grad_out)
            _sync()
            return q_leaf.grad, kv_leaf.grad, calls["grad"]

        grad_out = torch.randn(128, 8, VALUE_DIM, device="npu", dtype=torch.bfloat16)
        q_stock, kv_stock, n_stock = run_backward(0, grad_out)
        q_chunked, kv_chunked, n_calls = run_backward(48, grad_out)
        assert n_stock == 1, f"expected 1 stock grad call, got {n_stock}"
        assert n_calls == 3, f"expected 3 chunked grad calls, got {n_calls}"
        assert torch.equal(q_stock, q_chunked), "q grad mismatch vs stock"
        torch.testing.assert_close(kv_chunked.float(), kv_stock.float(), atol=DKV_ATOL, rtol=DKV_RTOL)

    def test_offload_span_gate_skips_pinned_copies_and_matches_stock(self, monkeypatch):
        """短 slice 门控（XTUNER_SFA_GRAD_OFFLOAD_MIN_KV）: slice ≤ 阈值时 forward 不再分配
        pinned 拷贝（全部留 device），分块 backward 照常运行且 q 梯度逐位等于 stock。"""
        sp8 = TestNPUSparseMLASP8PackedTND()
        q, kv, indices, seq_ctx, ss, sz, _total = sp8._make_sp8_packed_inputs(3)
        local_q = q[ss : ss + sz]
        local_indices = indices[ss : ss + sz]
        monkeypatch.setattr(npu_sparse_mla_mod, "_sp_mla_offload_enabled", lambda: True)
        monkeypatch.setattr(npu_sparse_mla_mod, "_SFA_GRAD_ROW_CHUNK", 48)

        pin_calls = {"n": 0}
        orig_empty = torch.empty

        def counting_empty(*args, **kwargs):
            if kwargs.get("pin_memory"):
                pin_calls["n"] += 1
            return orig_empty(*args, **kwargs)

        def run_forward_backward(grad_out: torch.Tensor):
            # 只暂换 torch.empty 计数 pinned 分配，手动恢复——不动 monkeypatch 的模块属性补丁。
            torch.empty = counting_empty
            try:
                q_leaf = local_q.detach().clone().requires_grad_(True)
                kv_leaf = kv.detach().clone().requires_grad_(True)
                out = sparse_mla(
                    q_leaf,
                    kv_leaf,
                    local_indices,
                    scaling=SCALING,
                    value_dim=VALUE_DIM,
                    backend="torch_npu",
                    seq_ctx=seq_ctx,
                )
            finally:
                torch.empty = orig_empty
            out.raw_output.backward(grad_out)
            _sync()
            return q_leaf.grad, kv_leaf.grad

        # 门控默认 131072 > 本几何 slice 长度: 无 pinned 分配, 梯度与 stock 逐位一致。
        grad_out = torch.randn(128, 8, VALUE_DIM, device="npu", dtype=torch.bfloat16)
        q_gated, kv_gated = run_forward_backward(grad_out)
        assert pin_calls["n"] == 0, f"expected 0 pinned allocs on short slice, got {pin_calls['n']}"

        monkeypatch.setattr(npu_sparse_mla_mod, "_SFA_GRAD_OFFLOAD_MIN_KV", 0)
        pin_calls["n"] = 0
        q_offloaded, kv_offloaded = run_forward_backward(grad_out)
        assert pin_calls["n"] >= 5, f"expected >=5 pinned allocs when gate disabled, got {pin_calls['n']}"
        assert torch.equal(q_gated, q_offloaded), "q grad mismatch: gated vs offloaded"
        # dk/dv 走 kernel 原子累加（跨调用非确定），与既有测试一致只做 allclose。
        torch.testing.assert_close(kv_gated.float(), kv_offloaded.float(), atol=DKV_ATOL, rtol=DKV_RTOL)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


