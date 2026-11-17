"""mHC hc_combine 双模式(FP32 对齐 / HEAD 复刻)的数值回归测试,见 xtuner/v1/ops/mhc_fused.py。

回归背景:修复前(commit 1ea0413a 之前的 8246fc21)triton hc_combine kernel 无条件把
fp32 的 post/comb 载入值经 _bf16_round 重新舍入到 bf16 网格(复刻 eager 链的 bf16 语义),
fp32 梯度 d_post/d_comb 的累加结果也被强制 bf16 round-trip 后存储。而生产上 sinkhorn
mixes 本身是 fp32 产出的,这层多余的重舍入使 backward 的 d_post/d_comb 相对 fp64 参考
的最大相对误差达到 ~2.9e-3 / ~2.4e-3,bf16 损伤重新进入 sinkhorn mixes,表现为训练
grad_norm 不稳。修复(commit 1ea0413a)引入模块开关 ``HC_COMBINE_TRITON_FP32``:默认
fp32 模式下 post/comb 从载入、每个 fwd/bwd 乘积、直到 d_post/d_comb 的 fp32 存储全程
保持 fp32,与 native cann_ops_transformer mhc_post 端到端对齐,d_post/d_comb 相对误差
降到 ~1.5e-7;开关置 False 则逐位恢复 HEAD 复刻语义。

TestHcCombineFp32Math
    test_fp32_mode_grads_match_fp64_reference    核心回归:d_post/d_comb 相对 fp64 参考 <1e-6(旧实现 ~2.4e-3,必失败)
    test_head_mode_bitwise_matches_eager         HEAD 模式 fwd 与 _combine_eager 逐位相等(torch.equal)
    test_bf16_post_comb_forward_backward         bf16 post/comb 入参路径前向+反向可用(形状/dtype/有限值)
    test_native_mhc_post_cross_consistency       (cann_ops 可用时)fp32 模式与 native mhc_post 梯度交叉一致 <1e-6

误差度量说明:rel = max|actual-ref| / max|ref|(参考幅值归一的全局最大相对误差)。
bf16 输出元的逐元素相对误差在近零抵消位元上会无界放大,该度量才能给出稳定阈值;
fp32 梯度在修复后实测 ~1.5e-7,旧实现因 bf16 存储往返被抬到 ~2.4e-3,分离清晰。
"""

from contextlib import contextmanager

import pytest
import torch

import xtuner.v1.ops.mhc_fused as mhc


try:
    import torch_npu  # noqa: F401

    _NPU_AVAILABLE = torch.npu.is_available()  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - CPU-only environments
    _NPU_AVAILABLE = False

try:
    from cann_ops_transformer.ops.mhc.mhc_post import mhc_post as _native_mhc_post
except Exception:  # pragma: no cover - wrapper package absent
    _native_mhc_post = None

DEV = "npu" if _NPU_AVAILABLE else "cpu"
_PROD_HC = 4


@contextmanager
def _hc_combine_mode(fp32: bool):
    """临时切换 triton hc_combine 精度模式;退出时恢复原值并复位 persistent-grid probe。

    kernel 的 FP32_MATH 是 launch 时读取的模块变量,_TRITON_MAX_PROGS 缓存的 NaN 哨兵
    probe 结果与模式绑定,切换后必须置 None 强制重测。
    """
    old = mhc.HC_COMBINE_TRITON_FP32
    mhc.HC_COMBINE_TRITON_FP32 = fp32
    mhc._TRITON_MAX_PROGS[0] = None
    try:
        yield
    finally:
        mhc.HC_COMBINE_TRITON_FP32 = old
        mhc._TRITON_MAX_PROGS[0] = None


def _combine_triton(post, comb, out, res):
    """直通 triton ``_HcCombineFunction.apply``(显式解析 persistent-grid probe)。

    绕开 :func:`mhc.hc_combine` 的 native 分发,保证被测对象始终是 triton kernel ——
    回归目标是 triton 实现,native 路径由 test_native_mhc_post_cross_consistency 单独覆盖。
    """
    mhc._resolve_max_progs(res.device)
    return mhc._HcCombineFunction.apply(post, comb, out, res)


def _prod_contract_inputs(B, T, HC, D, seed):
    """生产契约输入:post/comb fp32,comb 为 padded [B, T, HC, 8] 缓冲的前 HC 列切片
    (fold-consistent 视图,COMB_ST_K=8,与 sinkhorn 前向产出一致),out/res bf16 连续。"""
    torch.manual_seed(seed)
    post = torch.randn(B, T, HC, device=DEV, dtype=torch.float32)
    comb_padded = torch.randn(B, T, HC, 8, device=DEV, dtype=torch.float32)
    comb_padded[:, :, :, HC:] = float("-inf")
    comb = comb_padded[:, :, :, :HC]
    out = torch.randn(B, T, D, device=DEV, dtype=torch.bfloat16)
    res = torch.randn(B, T, HC, D, device=DEV, dtype=torch.bfloat16)
    return post, comb, out, res


def _fp64_reference(post, comb, out, res, grad):
    """fp64 参考(数学定义见模块 docstring 的 Combine 条目)。

    全部在 CPU 上做:先搬运再 cast float64。NPU 不支持 double,在 NPU 上 ``.double()``
    会走 float 替换路径,实测会把参考值污染回 bf16 网格(误差恰为一个 bf16 半 ULP),
    绝不能在卡上 cast。bf16/fp32 -> float64 的 cast 本身无精度损失,故参考是 kernel
    输入的精确数学值。
    """
    post64 = post.detach().to("cpu").to(torch.float64)
    comb64 = comb.detach().to("cpu").to(torch.float64)
    out64 = out.detach().to("cpu").to(torch.float64)
    res64 = res.detach().to("cpu").to(torch.float64)
    g64 = grad.detach().to("cpu").to(torch.float64)
    new_ref = post64.unsqueeze(-1) * out64.unsqueeze(-2) + torch.einsum("tkn,tkd->tnd", comb64[0], res64[0]).unsqueeze(0)
    d_post_ref = torch.einsum("tnd,td->tn", g64[0], out64[0]).unsqueeze(0)
    d_comb_ref = torch.einsum("tkd,tnd->tkn", res64[0], g64[0]).unsqueeze(0)
    d_out_ref = torch.einsum("tn,tnd->td", post64[0], g64[0]).unsqueeze(0)
    d_res_ref = torch.einsum("tkn,tnd->tkd", comb64[0], g64[0]).unsqueeze(0)
    return {
        "new": new_ref,
        "d_post": d_post_ref,
        "d_comb": d_comb_ref,
        "d_out": d_out_ref,
        "d_res": d_res_ref,
    }


def _rel_err(actual, ref):
    """全局归一最大相对误差 max|actual-ref| / max|ref|(度量定义见模块 docstring)。"""
    actual = actual.detach().to("cpu").float()
    ref = ref.detach().to("cpu").float()
    return ((actual - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()


def _grid_inputs(B, T, D, seed):
    """小有效位网格输入:post/comb 取 9-bit 有效位(不在 bf16 网格上,HEAD 载入舍入
    必生效);out/res 取 <=5-bit 有效位 bf16。

    该构造下所有乘积(<=14 bit)与四项求和(指数跨度 <=18 bit)在 fp32 中**任意累加
    顺序均精确**,因此 HEAD 复刻模式与 eager 链的逐位相等不依赖 NPU matmul 的内部
    累加顺序,断言确定性成立;同时 mul/matmul 的 bf16 舍入点仍被真正行使(非整数
    恒等舍入),错误的舍入位置会破坏逐位相等。而 fp32 模式不做这三处舍入,前向在
    ~半数元素上与 HEAD 模式可区分,可同时验证开关确实切换了行为。
    """
    gen = torch.Generator().manual_seed(seed)

    def fp32_9bit(shape, e_max):
        j = torch.randint(256, 512, shape, generator=gen).float() / 512.0  # [0.5, 1), 9 bit
        e = torch.randint(0, e_max + 1, shape, generator=gen).float()
        sign = torch.where(torch.rand(shape, generator=gen) < 0.5, -1.0, 1.0)
        return sign * j * torch.exp2(e)

    def bf16_5bit(shape, e_min, e_max):
        j = torch.randint(8, 32, shape, generator=gen).float() / 32.0  # [0.25, 1), <=5 bit
        e = torch.randint(e_min, e_max + 1, shape, generator=gen).float()
        sign = torch.where(torch.rand(shape, generator=gen) < 0.5, -1.0, 1.0)
        return (sign * j * torch.exp2(e)).to(torch.bfloat16)

    post = fp32_9bit((B, T, _PROD_HC), 1).to(DEV)
    comb_padded = torch.empty(B, T, _PROD_HC, 8, dtype=torch.float32)
    comb_padded[:, :, :, :_PROD_HC] = fp32_9bit((B, T, _PROD_HC, _PROD_HC), 1)
    comb_padded[:, :, :, _PROD_HC:] = float("-inf")
    comb = comb_padded[:, :, :, :_PROD_HC].to(DEV)
    out = bf16_5bit((B, T, D), -1, 2).to(DEV)
    res = bf16_5bit((B, T, _PROD_HC, D), -1, 2).to(DEV)
    return post, comb, out, res


@pytest.mark.gpu
@pytest.mark.skipif(not _NPU_AVAILABLE, reason="需要 NPU(fused triton 路径仅 NPU 可用)")
class TestHcCombineFp32Math:
    def test_fp32_mode_grads_match_fp64_reference(self):
        """核心回归:fp32 模式(默认)下 fwd new 与四个梯度对 fp64 参考的误差。

        d_post/d_comb max-rel < 1e-6 是回归断言:修复前 bf16 重舍入 + bf16 存储
        往返使其停在 ~2.9e-3 / ~2.4e-3(一个 bf16 半 ULP 的量级),该断言在旧实现
        上必然失败;修复后为 fp32 累加噪声 ~1.5e-7。new/d_out/d_res 本身是 bf16
        输出(bf16 半 ULP ~2e-3),阈值放宽到 5e-3。
        """
        B, T, HC, D = 1, 4096, _PROD_HC, 5120
        torch.manual_seed(0)
        post, comb, out, res = _prod_contract_inputs(B, T, HC, D, seed=0)
        with _hc_combine_mode(fp32=True):
            p = post.clone().requires_grad_(True)
            c = comb.clone().requires_grad_(True)
            o = out.clone().requires_grad_(True)
            r = res.clone().requires_grad_(True)
            new = _combine_triton(p, c, o, r)
            assert new.shape == res.shape, f"new 形状 {tuple(new.shape)} != res 形状 {tuple(res.shape)}"
            assert new.dtype == torch.bfloat16, f"new dtype {new.dtype} != bfloat16"
            grad = torch.randn(B, T, HC, D, device=DEV, dtype=torch.bfloat16)
            new.backward(grad)
        assert p.grad is not None and c.grad is not None and o.grad is not None and r.grad is not None
        assert p.grad.dtype == torch.float32, f"d_post dtype {p.grad.dtype} != float32(post 为 fp32)"
        assert c.grad.dtype == torch.float32, f"d_comb dtype {c.grad.dtype} != float32(comb 为 fp32)"

        refs = _fp64_reference(post, comb, out, res, grad)
        errs = {name: _rel_err(act, refs[name]) for name, act in
                {"new": new, "d_post": p.grad, "d_comb": c.grad, "d_out": o.grad, "d_res": r.grad}.items()}
        diag = ", ".join(f"{k}={v:.3e}" for k, v in errs.items())

        # 回归断言:fp32 的 d_post/d_comb 必须达到 fp32 累加精度;旧实现 ~2.9e-3/~2.4e-3
        assert errs["d_post"] < 1e-6, (
            f"d_post 相对 fp64 参考的最大相对误差 {errs['d_post']:.3e} >= 1e-6"
            f"(修复前基线 ~2.9e-3,fp32 模式应为 fp32 累加噪声 ~1.5e-7);全部误差 [{diag}];"
            f"度量: max|actual-ref| / max|ref|, d_post maxabs ref={refs['d_post'].abs().max():.1f}"
        )
        assert errs["d_comb"] < 1e-6, (
            f"d_comb 相对 fp64 参考的最大相对误差 {errs['d_comb']:.3e} >= 1e-6"
            f"(修复前基线 ~2.4e-3,fp32 模式应为 fp32 累加噪声 ~1.6e-7);全部误差 [{diag}]"
        )
        # bf16 输出元:new / d_out / d_res(半 ULP ~2e-3 量级)
        assert errs["new"] < 5e-3, f"new 相对 fp64 参考误差 {errs['new']:.3e} >= 5e-3;全部误差 [{diag}]"
        assert errs["d_out"] < 5e-3, f"d_out 相对 fp64 参考误差 {errs['d_out']:.3e} >= 5e-3;全部误差 [{diag}]"
        assert errs["d_res"] < 5e-3, f"d_res 相对 fp64 参考误差 {errs['d_res']:.3e} >= 5e-3;全部误差 [{diag}]"

    def test_head_mode_bitwise_matches_eager(self):
        """HEAD 保护:HC_COMBINE_TRITON_FP32=False 时 fwd 与 _combine_eager 逐位相等。

        用网格输入(见 _grid_inputs):所有中间值 fp32 精确、任意累加顺序不变,
        torch.equal 的成立不依赖 NPU matmul 内部顺序;同时三个 bf16 舍入点(载入、
        mul、matmul、final add)全部被行使,舍入位置错误会立刻打破逐位相等。
        随机连续值输入在当前 CANN 栈上存在 ~1e-7 概率的 1-ULP 边界差异(eager
        matmul 内部累加顺序所致,与 triton 实现无关),不适合做逐位断言。
        并验证开关确实切换行为:同输入下 fp32 模式前向与 HEAD 模式显著不同。
        """
        B, T, D = 1, 2048, 5120
        post, comb, out, res = _grid_inputs(B, T, D, seed=11)
        with _hc_combine_mode(fp32=False):
            new_head = _combine_triton(post, comb, out, res)
        with _hc_combine_mode(fp32=True):
            new_fp32 = _combine_triton(post, comb, out, res)
        eager = mhc._combine_eager(post, comb, out, res)

        if not torch.equal(new_head, eager):
            mismatch = (new_head != eager)
            n = mismatch.sum().item()
            idx = mismatch.nonzero()[0].tolist()
            raise AssertionError(
                f"HEAD 模式(HC_COMBINE_TRITON_FP32=False)前向与 _combine_eager 不再逐位相等: "
                f"{n}/{new_head.numel()} 个元素不一致, 首个不一致点 {idx}: "
                f"head={new_head[tuple(idx)].item()}, eager={eager[tuple(idx)].item()}"
            )
        assert torch.equal(new_head, eager), "HEAD 模式前向必须与 _combine_eager 逐位相等"

        differ_frac = (new_head != new_fp32).sum().item() / new_head.numel()
        assert differ_frac >= 0.01, (
            f"HC_COMBINE_TRITON_FP32 开关未生效: 同一网格输入下 fp32 模式与 HEAD 模式前向仅 "
            f"{differ_frac:.4%} 元素不同(应 >=1%), fp32 模式的 mul/matmul 无 bf16 舍入点未体现"
        )

    def test_bf16_post_comb_forward_backward(self):
        """bf16 post/comb 入参路径(unit-test dtype 契约)可正常前向+反向。

        只要求形状/dtype/有限值,不做精度阈值;native mhc_post 拒收 bf16,
        该路径必须落在 triton kernel 上。
        """
        B, T, HC, D = 1, 512, _PROD_HC, 5120
        torch.manual_seed(3)
        post_b = torch.randn(B, T, HC, device=DEV, dtype=torch.bfloat16).requires_grad_(True)
        comb_b = torch.randn(B, T, HC, HC, device=DEV, dtype=torch.bfloat16).requires_grad_(True)
        out = torch.randn(B, T, D, device=DEV, dtype=torch.bfloat16)
        res = torch.randn(B, T, HC, D, device=DEV, dtype=torch.bfloat16)
        with _hc_combine_mode(fp32=True):
            new = mhc.hc_combine(post_b, comb_b, out, res)
        assert new.shape == res.shape, f"bf16 路径 new 形状 {tuple(new.shape)} != {tuple(res.shape)}"
        assert new.dtype == torch.bfloat16, f"bf16 路径 new dtype {new.dtype} != bfloat16"
        assert torch.isfinite(new).all().item(), "bf16 路径前向出现 NaN/Inf"

        grad = torch.randn_like(new)
        new.backward(grad)
        assert post_b.grad is not None and comb_b.grad is not None, "bf16 路径 post/comb 梯度缺失"
        assert post_b.grad.shape == post_b.shape, f"d_post 形状 {tuple(post_b.grad.shape)} != {tuple(post_b.shape)}"
        assert comb_b.grad.shape == comb_b.shape, f"d_comb 形状 {tuple(comb_b.grad.shape)} != {tuple(comb_b.shape)}"
        assert post_b.grad.dtype == torch.bfloat16 and comb_b.grad.dtype == torch.bfloat16
        assert torch.isfinite(post_b.grad).all().item(), "bf16 路径 d_post 出现 NaN/Inf"
        assert torch.isfinite(comb_b.grad).all().item(), "bf16 路径 d_comb 出现 NaN/Inf"

    @pytest.mark.skipif(_native_mhc_post is None, reason="cann_ops_transformer mhc_post 不可用")
    def test_native_mhc_post_cross_consistency(self):
        """交叉一致:fp32 模式 triton 与 native mhc_post 的 d_post/d_comb 相对误差 <1e-6。

        native aclnn op 消费 fp32 sinkhorn mixes 全程不重舍入,是 fp32 模式的对齐目标;
        二者是两套独立实现,各自的 fp32 累加噪声 ~1e-7,交叉差应 <1e-6。
        """
        B, T, HC, D = 1, 4096, _PROD_HC, 5120
        torch.manual_seed(0)
        post, comb, out, res = _prod_contract_inputs(B, T, HC, D, seed=0)
        grad = torch.randn(B, T, HC, D, device=DEV, dtype=torch.bfloat16)

        with _hc_combine_mode(fp32=True):
            p1 = post.clone().requires_grad_(True)
            c1 = comb.clone().requires_grad_(True)
            o1 = out.clone().requires_grad_(True)
            r1 = res.clone().requires_grad_(True)
            _combine_triton(p1, c1, o1, r1).backward(grad)

        p2 = post.clone().requires_grad_(True)
        c2 = comb.clone().requires_grad_(True)
        o2 = out.clone().requires_grad_(True)
        r2 = res.clone().requires_grad_(True)
        _native_mhc_post(r2, c2, o2, p2).backward(grad)

        rel_post = _rel_err(p2.grad, p1.grad)
        rel_comb = _rel_err(c2.grad, c1.grad)
        assert rel_post < 1e-6, (
            f"native mhc_post 与 triton fp32 模式的 d_post 相对误差 {rel_post:.3e} >= 1e-6 "
            f"(两实现应同为 fp32 端到端, 交叉差为各自累加噪声 ~2e-7)"
        )
        assert rel_comb < 1e-6, (
            f"native mhc_post 与 triton fp32 模式的 d_comb 相对误差 {rel_comb:.3e} >= 1e-6 "
            f"(两实现应同为 fp32 端到端, 交叉差为各自累加噪声 ~2.5e-7)"
        )
