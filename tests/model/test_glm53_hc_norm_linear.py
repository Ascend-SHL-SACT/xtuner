"""GLM-5.3 mHC 融合 norm+linear（mhc_norm_linear triton）契约、对拍与 grid 封顶回归测试。

``mhc_norm_linear`` 用恒等式 ``mixes = (x @ w.T) * rstd``（无权重 RMSNorm 的 per-row
标量 rstd 提出投影外）替换 eager 的 fp32 物化链；N 不是 1024 倍数（rstd 核
BLOCK_D=1024 硬约束，K/D tile 128/512 随之满足）或设备/dtype/连续性不满足时返回 None
交回 eager 链。梯度由 autograd.Function ``_MhcNormLinearFn`` 手写反传给出：
d_x = (dm@w)*inv - x*(<g,x>)*inv^3/N，d_w = (dm*inv).T @ x。

TestMhcNormLinearServable
    test_cpu_inputs_return_none: CPU 张量恒返回 None（调用方回退 eager 链）。
    test_width_gate_requires_multiple_of_1024: N=512/768（虽整除 256/128）-> None。
    test_shape_dtype_and_contiguity_gates: w dtype/宽度失配、x/w 非连续、T=0 -> None。
    test_env_flag_off_forces_fallback: MHC_NORM_LINEAR_FUSED=False 时 NPU 上也返回 None。
    test_valid_bf16_and_fp16_pairs_are_served: N=1024 的 bf16/fp16 同 dtype 对 -> [T, MIX] fp32。
TestMhcNormLinearNumeric
    test_forward_matches_fp32_math_reference: 前向 vs (x*rstd) @ w.T 的 fp32 数学参考，
        maxabs < 5e-5（实测 ~5e-6 @ mixes std ~0.6）。
    test_backward_matches_autograd_reference: 手写反传 d_x/d_w vs eager fp32 链 autograd，
        范数相对差 bf16 ulp 级（实测 ~2.6e-3，dm 进 MMA 前转 bf16 所致）。
    test_backward_honors_needs_input_grad: 只开 x 或只开 w 梯度时另一路返回 None 且结果一致。
TestMhcNormLinearPersistentGrid
    test_forward_matches_fp32_reference_persistent_grid: 发射 grid 封顶 _MAX_PROGS 后
        tokens 超发射程序数（4096）与生产宽度 n=8192 的前向对拍——漏处理的行/T-tile 输出来自
        torch.empty 未初始化内存，对拍必炸，对拍本身即覆盖完整性检测。
    test_backward_matches_autograd_reference_persistent_grid: 同形状手写反传对拍；核内
        跨步循环的多轮迭代路径由 n=8192 形状覆盖。
TestMhcNormLinearSplitParity
    test_split_matches_fused_bitwise: split 拆核（fwd/bwd，L0C 握手消除）与 fused mix 核
        torch.equal 逐位对拍，T=64/1536（后者覆盖持久 grid 多轮），并单开 x / 单开 w 梯度
        覆盖 d_x-only / d_w-only 分支。

注：原 ``TestMhcNormLinearIntegration``（Glm53HyperConnection 整模块对拍）随 #95 合并
删除该类而作废——decoder 层（mhc.py）现直接接线 mhc_fused，无独立模块边界可对拍。
"""

import pytest
import torch

import xtuner.v1.ops.mhc_fused as hnl
from xtuner.v1.ops.mhc_fused import mhc_norm_linear
from xtuner.v1.utils.device import get_device


DEVICE = get_device()
_requires_accelerator = pytest.mark.skipif(DEVICE != "npu", reason="requires an NPU (fused triton kernels)")


def _mhc_args(tokens: int, n: int, mix: int, dtype: torch.dtype = torch.bfloat16) -> tuple[torch.Tensor, torch.Tensor]:
    """生产量级入参：x ~ N(0, 4)、w ~ N(0, 4e-4)（mixes std ~0.6）。"""
    torch.manual_seed(0)
    x2d = torch.randn(tokens, n, device=DEVICE, dtype=dtype) * 2
    w = torch.randn(mix, n, device=DEVICE, dtype=dtype) * 0.02
    return x2d, w


def _mhc_reference(x2d: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """fp32 数学参考：mixes = (x2d * rstd_row) @ w.T，rstd = 1/sqrt(mean(x^2) + eps)。"""
    xf = x2d.float()
    rstd = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (xf * rstd) @ w.float().T


def _assert_grad_close(g_f: torch.Tensor, g_r: torch.Tensor, name: str) -> None:
    """bf16 梯度对拍：范数相对差 < 0.02 且 max/std < 0.1（实测 ~2.6e-3 / ~3.4e-2）。"""
    g_f32, g_r32 = g_f.float(), g_r.float()
    abs_diff = (g_f32 - g_r32).abs()
    norm_rel = (abs_diff.norm() / g_r32.norm().clamp_min(1e-8)).item()
    max_over_std = (abs_diff.max() / g_r32.std().clamp_min(1e-3)).item()
    assert norm_rel < 0.02 and max_over_std < 0.1, (name, norm_rel, max_over_std)


class TestMhcNormLinearServable:
    """servable 判定：不可服务时必须返回 None 交回 eager 链。"""

    def test_cpu_inputs_return_none(self) -> None:
        """CPU 张量恒返回 None（无论形状/dtype 是否满足其余条件）。"""
        x2d = torch.randn(4, 1024, dtype=torch.bfloat16, device="cpu")
        w = torch.randn(24, 1024, dtype=torch.bfloat16, device="cpu")
        assert mhc_norm_linear(x2d, w, 1e-5) is None

    @_requires_accelerator
    @pytest.mark.gpu
    def test_width_gate_requires_multiple_of_1024(self) -> None:
        """N 非 1024 倍数（512/768，即使整除 256/128）-> None（rstd 核 BLOCK_D=1024 硬约束）。"""
        for n in (512, 768):
            x2d = torch.randn(4, n, device=DEVICE, dtype=torch.bfloat16)
            w = torch.randn(24, n, device=DEVICE, dtype=torch.bfloat16)
            assert mhc_norm_linear(x2d, w, 1e-5) is None, n

    @_requires_accelerator
    @pytest.mark.gpu
    def test_shape_dtype_and_contiguity_gates(self) -> None:
        """dtype 失配 / w.shape[1] 失配 / x 或 w 非连续 / T=0 -> None。"""
        x2d = torch.randn(4, 1024, device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn(24, 1024, device=DEVICE, dtype=torch.bfloat16)
        assert mhc_norm_linear(x2d, w.float(), 1e-5) is None
        assert mhc_norm_linear(x2d, torch.randn(24, 2048, device=DEVICE, dtype=torch.bfloat16), 1e-5) is None
        assert mhc_norm_linear(torch.randn(4, 2048, device=DEVICE, dtype=torch.bfloat16)[:, :1024], w, 1e-5) is None
        assert mhc_norm_linear(x2d, torch.randn(24, 2048, device=DEVICE, dtype=torch.bfloat16)[:, :1024], 1e-5) is None
        assert mhc_norm_linear(torch.empty(0, 1024, device=DEVICE, dtype=torch.bfloat16), w, 1e-5) is None

    @_requires_accelerator
    @pytest.mark.gpu
    def test_env_flag_off_forces_fallback(self, monkeypatch) -> None:
        """MHC_NORM_LINEAR_FUSED=False（env XTUNER_MHC_NORM_LINEAR_FUSED=0 的导入期效果）-> None。"""
        x2d, w = _mhc_args(4, 1024, 24)
        monkeypatch.setattr(hnl, "MHC_NORM_LINEAR_FUSED", False)
        assert mhc_norm_linear(x2d, w, 1e-5) is None
        monkeypatch.setattr(hnl, "MHC_NORM_LINEAR_FUSED", True)
        assert mhc_norm_linear(x2d, w, 1e-5) is not None

    @_requires_accelerator
    @pytest.mark.gpu
    def test_valid_bf16_and_fp16_pairs_are_served(self) -> None:
        """N=1024 的 bf16 / fp16 同 dtype 对被服务：返回 [T, MIX] fp32。"""
        for dtype in (torch.bfloat16, torch.float16):
            x2d, w = _mhc_args(33, 1024, 24, dtype=dtype)
            y = mhc_norm_linear(x2d, w, 1e-5)
            assert y is not None, dtype
            assert y.dtype == torch.float32 and y.shape == (33, 24), dtype


class TestMhcNormLinearNumeric:
    """fused 前向 / 手写反传 vs fp32 数学参考的对拍（NPU）。"""

    @_requires_accelerator
    @pytest.mark.gpu
    @pytest.mark.parametrize(("tokens", "n", "mix"), [(33, 1024, 24), (64, 2048, 24), (1, 1024, 4), (130, 1024, 24)])
    def test_forward_matches_fp32_math_reference(self, tokens: int, n: int, mix: int) -> None:
        """前向 == (x*rstd) @ w.T 的 fp32 参考（T 非块倍数 / MIX 非 2 幂 / N 跨 rstd 块）。"""
        x2d, w = _mhc_args(tokens, n, mix)
        y = mhc_norm_linear(x2d, w, 1e-5)
        assert y is not None
        assert y.dtype == torch.float32 and y.shape == (tokens, mix)
        ref = _mhc_reference(x2d.cpu(), w.cpu(), 1e-5)
        abs_diff = (y.cpu() - ref).abs()
        assert abs_diff.max().item() < 5e-5, abs_diff.max().item()
        norm_rel = (abs_diff.norm() / ref.norm()).item()
        assert norm_rel < 2e-5, norm_rel

    @_requires_accelerator
    @pytest.mark.gpu
    @pytest.mark.parametrize(("tokens", "n", "mix"), [(33, 1024, 24), (1, 1024, 4), (130, 1024, 24)])
    def test_backward_matches_autograd_reference(self, tokens: int, n: int, mix: int) -> None:
        """手写反传 vs eager fp32 链 autograd：d_x/d_w 范数相对差 bf16 ulp 级。"""
        x2d, w = _mhc_args(tokens, n, mix)
        x_leaf = x2d.detach().clone().requires_grad_(True)
        w_leaf = w.detach().clone().requires_grad_(True)
        y = mhc_norm_linear(x_leaf, w_leaf, 1e-5)
        assert y is not None
        dm = torch.randn_like(y)
        y.backward(dm)
        assert x_leaf.grad is not None and w_leaf.grad is not None
        x_r = x2d.detach().clone().requires_grad_(True)
        w_r = w.detach().clone().requires_grad_(True)
        _mhc_reference(x_r, w_r, 1e-5).backward(dm)
        assert x_r.grad is not None and w_r.grad is not None
        _assert_grad_close(x_leaf.grad, x_r.grad, "d_x")
        _assert_grad_close(w_leaf.grad, w_r.grad, "d_w")

    @_requires_accelerator
    @pytest.mark.gpu
    def test_backward_honors_needs_input_grad(self) -> None:
        """只开 x（w 不要求梯度）或只开 w 时另一路梯度为 None，开启路与参考一致。"""
        x2d, w = _mhc_args(33, 1024, 24)
        dm = torch.randn(33, 24, device=DEVICE, dtype=torch.float32)

        x_only = x2d.detach().clone().requires_grad_(True)
        y = mhc_norm_linear(x_only, w.detach().clone(), 1e-5)
        assert y is not None
        y.backward(dm)
        x_r = x2d.detach().clone().requires_grad_(True)
        w_r = w.detach().clone().requires_grad_(True)
        _mhc_reference(x_r, w_r, 1e-5).backward(dm)
        assert x_only.grad is not None and x_r.grad is not None
        _assert_grad_close(x_only.grad, x_r.grad, "d_x")

        w_only = w.detach().clone().requires_grad_(True)
        y2 = mhc_norm_linear(x2d.detach().clone(), w_only, 1e-5)
        assert y2 is not None
        y2.backward(dm)
        w_r2 = w.detach().clone().requires_grad_(True)
        _mhc_reference(x2d.detach().clone(), w_r2, 1e-5).backward(dm)
        assert w_only.grad is not None and w_r2.grad is not None
        _assert_grad_close(w_only.grad, w_r2.grad, "d_w")


class TestMhcNormLinearPersistentGrid:
    """持久化 grid 契约回归：发射 grid 封顶 _MAX_PROGS + 核内跨步循环恢复全覆盖。

    TRITON_ALL_BLOCKS_PARALLEL=1 时 Ascend triton launcher 把 blockNum 钳到物理核数；
    发射 grid 封顶后每程序 ``for blk in range(pid, total, nprogs)`` 跨步循环恢复
    全覆盖（每行 / 每 T-tile / 每 D-tile 归约都在单程序内按原顺序完成，数学不变）。
    输出张量均为 torch.empty：若某行 / T-tile / D-tile 被漏处理，其输出停留在
    未初始化内存上，对 fp32 参考的对拍必然爆掉——对拍本身即覆盖完整性检测，无需
    专门钩子。
    """

    @_requires_accelerator
    @pytest.mark.gpu
    @pytest.mark.parametrize(
        ("tokens", "n", "mix", "max_tol", "norm_tol"),
        [
            (1536, 2048, 24, 5e-5, 2e-5),  # T-tile=24 恰好整除：每程序恰好 1 轮的边界
            (1537, 2048, 24, 5e-5, 2e-5),  # T-tile=25：尾 tile 部分行
            (4096, 2048, 24, 5e-5, 2e-5),  # T-tile=64 > 发射 cap 48：pid 0-15 两轮的多轮边界
            (2048, 8192, 24, 2e-4, 5e-5),  # 生产宽度 n=8192：fwd grid=32 单轮（dfn 由 bwd 覆盖）
            (4096, 8192, 24, 2e-4, 5e-5),  # 生产宽度 x T-tile=64 超发射程序数：fwd/dx/rstd 多轮
        ],
    )
    def test_forward_matches_fp32_reference_persistent_grid(
        self, tokens: int, n: int, mix: int, max_tol: float, norm_tol: float
    ) -> None:
        """tokens 超帽 / 边界 / 生产宽度下前向 vs fp32 数学参考（漏行 = 未初始化内存必炸）。

        实测 maxabs：n<=2048 ~7e-6、n=8192 ~3e-5（fp32 求和顺序差随 sum|terms|~N*w_std 增长），
        阈值留 >6x 余量。
        """
        x2d, w = _mhc_args(tokens, n, mix)
        y = mhc_norm_linear(x2d, w, 1e-5)
        assert y is not None, (tokens, n)
        assert y.dtype == torch.float32 and y.shape == (tokens, mix)
        ref = _mhc_reference(x2d.cpu(), w.cpu(), 1e-5)
        abs_diff = (y.cpu() - ref).abs()
        assert abs_diff.max().item() < max_tol, abs_diff.max().item()
        norm_rel = (abs_diff.norm() / ref.norm()).item()
        assert norm_rel < norm_tol, norm_rel

    @_requires_accelerator
    @pytest.mark.gpu
    @pytest.mark.parametrize(
        ("tokens", "n", "mix"),
        [(1536, 2048, 24), (1537, 2048, 24), (4096, 2048, 24), (2048, 8192, 24), (4096, 8192, 24)],
    )
    def test_backward_matches_autograd_reference_persistent_grid(self, tokens: int, n: int, mix: int) -> None:
        """同形状手写反传 vs eager fp32 链 autograd；核内跨步多轮路径由生产宽度形状覆盖。

        实测 norm_rel ~2.6e-3 / max~std ~4e-2（dm 进 MMA 前转 bf16 主导），沿用既有对拍阈值。
        """
        x2d, w = _mhc_args(tokens, n, mix)
        x_leaf = x2d.detach().clone().requires_grad_(True)
        w_leaf = w.detach().clone().requires_grad_(True)
        y = mhc_norm_linear(x_leaf, w_leaf, 1e-5)
        assert y is not None, (tokens, n)
        dm = torch.randn_like(y)
        y.backward(dm)
        assert x_leaf.grad is not None and w_leaf.grad is not None
        x_r = x2d.detach().clone().requires_grad_(True)
        w_r = w.detach().clone().requires_grad_(True)
        _mhc_reference(x_r, w_r, 1e-5).backward(dm)
        assert x_r.grad is not None and w_r.grad is not None
        _assert_grad_close(x_leaf.grad, x_r.grad, "d_x")
        _assert_grad_close(w_leaf.grad, w_r.grad, "d_w")


class TestMhcNormLinearSplitParity:
    """split 拆核（fwd/bwd，L0C 握手消除）与 fused mix 核的逐位对拍回归。

    split 的逐位等价依赖两条编译器级不变量（tl.dot 的 K 归约序不受输出 tile 宽度
    影响、tl.sum 树不受 g 的布局来源影响），实证仅覆盖 MIX=24 bf16 生产家族；
    本测试把该假设以 torch.equal（逐位）钉进 CI——数值容差对拍（~2.6e-3）抓不住
    归约序漂移。T 恰为 _T_TILE=64 倍数时走 split，flag 全关走 fused；单开 x /
    单开 w 梯度覆盖 d_x-only / d_w-only 分支（现有 needs_input_grad 测试只用
    tokens=33，走的是 fused fallback）。
    """

    @_requires_accelerator
    @pytest.mark.gpu
    @pytest.mark.parametrize(("tokens", "n"), [(64, 1024), (1536, 8192)])
    def test_split_matches_fused_bitwise(self, tokens: int, n: int, monkeypatch) -> None:
        """split vs fused 前向与双向梯度逐位一致；needs_input_grad 单侧分支亦逐位一致。"""
        x2d, w = _mhc_args(tokens, n, 24)
        dm = torch.randn(tokens, 24, device=DEVICE, dtype=torch.float32)

        def _run(
            split_fwd: bool, split_bwd: bool, grad_x: bool, grad_w: bool
        ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
            x_leaf = x2d.detach().clone().requires_grad_(grad_x)
            w_leaf = w.detach().clone().requires_grad_(grad_w)
            monkeypatch.setattr(hnl, "MHC_NORM_LINEAR_SPLIT", split_fwd)
            monkeypatch.setattr(hnl, "MHC_NORM_LINEAR_BWD_SPLIT", split_bwd)
            y = mhc_norm_linear(x_leaf, w_leaf, 1e-5)
            assert y is not None
            y.backward(dm)
            return y, x_leaf.grad, w_leaf.grad

        # 双侧梯度：split vs fused 前向 + d_x + d_w 全逐位
        y_f, d_x_f, d_w_f = _run(False, False, True, True)
        y_s, d_x_s, d_w_s = _run(True, True, True, True)
        assert torch.equal(y_s, y_f)
        assert d_x_s is not None and d_x_f is not None and torch.equal(d_x_s, d_x_f)
        assert d_w_s is not None and d_w_f is not None and torch.equal(d_w_s, d_w_f)
        # d_x-only（w 不要求梯度）
        _, d_x_s_only, _ = _run(True, True, True, False)
        _, d_x_f_only, _ = _run(False, False, True, False)
        assert d_x_s_only is not None and d_x_f_only is not None and torch.equal(d_x_s_only, d_x_f_only)
        # d_w-only（x 不要求梯度）
        _, _, d_w_s_only = _run(True, True, False, True)
        _, _, d_w_f_only = _run(False, False, False, True)
        assert d_w_s_only is not None and d_w_f_only is not None and torch.equal(d_w_s_only, d_w_f_only)
