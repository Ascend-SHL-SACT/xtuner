"""NPU GMM (Grouped MatMul) 算子单元测试。

验证 mindspeed npu_gmm 在 CANN 9.0 上的前向+反向正确性。
覆盖 XTuner MoE 训练中的实际使用场景。

测试矩阵:
  1. 基础前向+反向 (小规模)
  2. 多 expert 分组
  3. bf16 精度对比 (vs torch matmul)
  4. 大规模 (训练场景: S*N=32768, E=128)
  5. 梯度有限性 (无 NaN/Inf)
  6. 梯度正确性 (vs torch 手动反向)
  7. requires_grad=False 的权重 (FSDP2 场景)
  8. 连续多次前向+反向 (模拟 recompute)
"""
import os
import sys
import pytest
import torch
import torch_npu

sys.path.insert(0, "/weight/jschen/code/xtuner")
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

from mindspeed.ops.gmm import npu_gmm

DEVICE = "npu:0"
DTYPE = torch.bfloat16


def _torch_gemm_ref(x, w, batch_sizes):
    """torch 参考实现: 逐 expert matmul."""
    outputs = []
    offset = 0
    for i, bs in enumerate(batch_sizes.tolist()):
        xi = x[offset:offset + bs]
        wi = w[i]
        out_i = torch.matmul(xi, wi)
        outputs.append(out_i)
        offset += bs
    return torch.cat(outputs, dim=0)


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires NPU")
class TestNPUGMM:

    def test_basic_forward_backward(self):
        """基础前向+反向: 2 expert, 各 4 token."""
        x = torch.randn(8, 16, dtype=DTYPE, device=DEVICE, requires_grad=True)
        w = torch.randn(2, 16, 32, dtype=DTYPE, device=DEVICE, requires_grad=True)
        batch_sizes = torch.tensor([4, 4], dtype=torch.int32, device=DEVICE)
        group_list = torch.cumsum(batch_sizes, dim=0)

        out = npu_gmm(x, w, bias=None, group_list=group_list, group_type=0)
        assert out.shape == (8, 32)
        assert torch.isfinite(out).all()

        out.sum().backward()
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(w.grad).all()

    def test_multi_expert(self):
        """多 expert: 8 expert, 各 16 token."""
        E, tokens_per_expert, D_in, D_out = 8, 16, 64, 128
        x = torch.randn(E * tokens_per_expert, D_in, dtype=DTYPE, device=DEVICE, requires_grad=True)
        w = torch.randn(E, D_in, D_out, dtype=DTYPE, device=DEVICE, requires_grad=True)
        batch_sizes = torch.tensor([tokens_per_expert] * E, dtype=torch.int32, device=DEVICE)
        group_list = torch.cumsum(batch_sizes, dim=0)

        out = npu_gmm(x, w, bias=None, group_list=group_list, group_type=0)
        assert out.shape == (E * tokens_per_expert, D_out)
        assert torch.isfinite(out).all()

        out.sum().backward()
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(w.grad).all()

    def test_forward_accuracy_vs_torch(self):
        """前向精度: NPU vs torch 逐 expert matmul."""
        E, tokens_per_expert, D_in, D_out = 4, 32, 128, 256
        torch.manual_seed(42)
        x = torch.randn(E * tokens_per_expert, D_in, dtype=DTYPE, device=DEVICE)
        w = torch.randn(E, D_in, D_out, dtype=DTYPE, device=DEVICE)
        batch_sizes = torch.tensor([tokens_per_expert] * E, dtype=torch.int32, device=DEVICE)
        group_list = torch.cumsum(batch_sizes, dim=0)

        npu_out = npu_gmm(x, w, bias=None, group_list=group_list, group_type=0)
        ref_out = _torch_gemm_ref(x, w, batch_sizes)

        max_diff = (npu_out.float() - ref_out.float()).abs().max().item()
        rel_err = (npu_out.float() - ref_out.float()).abs().mean().item() / max(ref_out.float().abs().mean().item(), 1e-8) * 100
        assert rel_err < 1.0, f"rel_err={rel_err:.4f}% too high, max_diff={max_diff:.6f}"

    def test_large_scale(self):
        """大规模 (训练场景): 128 expert, 总 token=32768."""
        E, total_tokens, D_in, D_out = 128, 32768, 512, 256
        tokens_per_expert = total_tokens // E
        x = torch.randn(total_tokens, D_in, dtype=DTYPE, device=DEVICE, requires_grad=True)
        w = torch.randn(E, D_in, D_out, dtype=DTYPE, device=DEVICE, requires_grad=True)
        batch_sizes = torch.tensor([tokens_per_expert] * E, dtype=torch.int32, device=DEVICE)
        group_list = torch.cumsum(batch_sizes, dim=0)

        out = npu_gmm(x, w, bias=None, group_list=group_list, group_type=0)
        assert out.shape == (total_tokens, D_out)
        assert torch.isfinite(out).all()

        out.sum().backward()
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(w.grad).all()

    def test_gradients_finite(self):
        """梯度有限性: 无 NaN/Inf."""
        E, tokens_per_expert, D_in, D_out = 4, 64, 128, 256
        x = torch.randn(E * tokens_per_expert, D_in, dtype=DTYPE, device=DEVICE, requires_grad=True)
        w = torch.randn(E, D_in, D_out, dtype=DTYPE, device=DEVICE, requires_grad=True)
        batch_sizes = torch.tensor([tokens_per_expert] * E, dtype=torch.int32, device=DEVICE)
        group_list = torch.cumsum(batch_sizes, dim=0)

        out = npu_gmm(x, w, bias=None, group_list=group_list, group_type=0)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)

        assert torch.isfinite(x.grad).all(), "x.grad has NaN/Inf"
        assert torch.isfinite(w.grad).all(), "w.grad has NaN/Inf"

    def test_gradient_correctness_vs_torch(self):
        """梯度正确性: NPU vs torch 手动反向."""
        E, tokens_per_expert, D_in, D_out = 2, 16, 32, 64
        torch.manual_seed(42)
        x_ref = torch.randn(E * tokens_per_expert, D_in, dtype=torch.float32, device=DEVICE, requires_grad=True)
        w_ref = torch.randn(E, D_in, D_out, dtype=torch.float32, device=DEVICE, requires_grad=True)
        batch_sizes = torch.tensor([tokens_per_expert] * E, dtype=torch.int32, device=DEVICE)

        # torch 参考
        ref_out = _torch_gemm_ref(x_ref, w_ref, batch_sizes)
        grad_out = torch.randn_like(ref_out)
        ref_out.backward(grad_out)

        # NPU GMM (bf16)
        x_npu = x_ref.detach().to(DTYPE).clone().requires_grad_(True)
        w_npu = w_ref.detach().to(DTYPE).clone().requires_grad_(True)
        group_list = torch.cumsum(batch_sizes, dim=0)
        npu_out = npu_gmm(x_npu, w_npu, bias=None, group_list=group_list, group_type=0)
        npu_out.backward(grad_out.to(DTYPE))

        # 对比梯度 (bf16 vs fp32, 允许较大误差)
        x_diff = (x_npu.grad.float() - x_ref.grad).abs()
        w_diff = (w_npu.grad.float() - w_ref.grad).abs()
        x_rel = x_diff.mean().item() / max(x_ref.grad.abs().mean().item(), 1e-8) * 100
        w_rel = w_diff.mean().item() / max(w_ref.grad.abs().mean().item(), 1e-8) * 100
        assert x_rel < 20.0, f"x_grad rel_err={x_rel:.2f}% too high"
        assert w_rel < 20.0, f"w_grad rel_err={w_rel:.2f}% too high"

    def test_weight_no_grad(self):
        """权重 requires_grad=False (FSDP2 reduce-scatter 场景)."""
        E, tokens_per_expert, D_in, D_out = 4, 32, 64, 128
        x = torch.randn(E * tokens_per_expert, D_in, dtype=DTYPE, device=DEVICE, requires_grad=True)
        w = torch.randn(E, D_in, D_out, dtype=DTYPE, device=DEVICE, requires_grad=False)
        batch_sizes = torch.tensor([tokens_per_expert] * E, dtype=torch.int32, device=DEVICE)
        group_list = torch.cumsum(batch_sizes, dim=0)

        out = npu_gmm(x, w, bias=None, group_list=group_list, group_type=0)
        out.sum().backward()
        assert torch.isfinite(x.grad).all()

    def test_consecutive_forward_backward(self):
        """连续多次前向+反向 (模拟 recompute 重算)."""
        E, tokens_per_expert, D_in, D_out = 4, 32, 64, 128

        for iteration in range(3):
            x = torch.randn(E * tokens_per_expert, D_in, dtype=DTYPE, device=DEVICE, requires_grad=True)
            w = torch.randn(E, D_in, D_out, dtype=DTYPE, device=DEVICE, requires_grad=True)
            batch_sizes = torch.tensor([tokens_per_expert] * E, dtype=torch.int32, device=DEVICE)
            group_list = torch.cumsum(batch_sizes, dim=0)

            out = npu_gmm(x, w, bias=None, group_list=group_list, group_type=0)
            assert torch.isfinite(out).all(), f"iter {iteration}: forward NaN/Inf"

            out.sum().backward()
            assert torch.isfinite(x.grad).all(), f"iter {iteration}: x.grad NaN/Inf"
            assert torch.isfinite(w.grad).all(), f"iter {iteration}: w.grad NaN/Inf"

    def test_output_shape_dtype(self):
        """输出形状和 dtype."""
        E, tokens_per_expert, D_in, D_out = 4, 16, 32, 64
        x = torch.randn(E * tokens_per_expert, D_in, dtype=DTYPE, device=DEVICE)
        w = torch.randn(E, D_in, D_out, dtype=DTYPE, device=DEVICE)
        batch_sizes = torch.tensor([tokens_per_expert] * E, dtype=torch.int32, device=DEVICE)
        group_list = torch.cumsum(batch_sizes, dim=0)

        out = npu_gmm(x, w, bias=None, group_list=group_list, group_type=0)
        assert out.shape == (E * tokens_per_expert, D_out)
        assert out.dtype == DTYPE

    def test_uneven_split(self):
        """不均匀分组 (模拟 MoE load imbalance)."""
        x = torch.randn(100, 64, dtype=DTYPE, device=DEVICE, requires_grad=True)
        w = torch.randn(3, 64, 128, dtype=DTYPE, device=DEVICE, requires_grad=True)
        batch_sizes = torch.tensor([10, 50, 40], dtype=torch.int32, device=DEVICE)
        group_list = torch.cumsum(batch_sizes, dim=0)

        out = npu_gmm(x, w, bias=None, group_list=group_list, group_type=0)
        assert out.shape == (100, 128)
        assert torch.isfinite(out).all()

        out.sum().backward()
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(w.grad).all()
