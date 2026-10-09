# Copyright © 2026 Huawei Technologies Co., Ltd.
"""KDA SP 延迟 fence 调度（``_KDA_SP_A2A_DEFER``，env ``XTUNER_KDA_SP_A2A_DEFER``）的等价/顺序/契约测试。

被测对象：``xtuner/v1/module/attention/kda.py`` 的 ``KimiDeltaAttention.forward_for_sp``
（Ulysses SP，wave6 契约）。该函数按模块常量 ``_KDA_SP_A2A_DEFER`` 在两种调度间选择，两种调度
发出**同样**的 6 次前向 Ulysses a2a（q/k/v/g/beta/out，各自独立 ``issue_ulysses_dim1``），
collective 数值与胶水一致，只有等待点不同，因此输出/梯度应逐位一致：

- DEFER=1（默认，wave6 defer 调度）：q/k/v/g/beta 五发依序尽早在各自投影后 issue（g 为
  ``f_b_proj(f_a_proj)`` 的 transpose；beta 为 ``b_proj.float().transpose``——**sigmoid 不前移**，
  a2a 输入是 sigmoid 前的 fp32 raw 值）；随后 q/k/v 依次经 ``_finish_short_conv``
  （``finish_ulysses_dim1`` 围栏 + ``transpose(1,2).contiguous`` + weight/bias 按 sp_rank chunk +
  conv），g/beta 在消费点围栏（beta 再 ``transpose(1,2).sigmoid()``）；gate → core 之后 issue out
  一发，``_gate_output`` 之后在 o_norm 消费点围栏。
- DEFER=0：blocking 调度——``_sp_short_conv`` 的 ``_all_to_all_conv_pre_{q,k,v}`` / ``_all_to_all_g``
  / ``_all_to_all_beta`` / ``_all_to_all_out`` 每个 a2a issue 后立即 fence（逐 collective 阻塞），
  同样 6 发。

collective parity：两种调度均为前向 6 次 a2a、含反向 12 次（反向经
``ulysses_dispatch._AsyncUlyssesDim1.backward`` 的反向 a2a，q/k/v/g/beta/out 各一次）。

测试环境：纯 CPU / gloo（FileStore 初始化，无 NPU、无 GPU 标记）。``ulysses_dispatch._a2a_issue``
在 worker 内替换为同步 ``dist.all_to_all_single`` stub（返回恒已完成的 dummy work，并捕获 send
缓冲），使 issue/fence 的**调用顺序**与 **send 布局**可见且无悬挂风险（monkeypatch 模块 globals
即同时覆盖前向 issue 与反向反向 a2a）；设备侧内核（``causal_conv1d`` / ``chunk_kda`` /
``fused_recurrent_kda`` / ``fused_kda_gate`` / ``_gate_param``）在 ``kda`` 模块层替换为确定性的
可微 CPU stand-in（因果深度卷积、线性门、``q+k+v+g*beta`` 核），o_norm/o_proj 走真实 CPU eager
路径（npu_backend 的 rms_norm_gated 对 CPU 张量有严格 fp32 eager 回退）。等价验收标准为
``torch.equal``：同一调度内核序列 + 纯置换 a2a，单线程 fp32 下确定。

偏差说明（有意为之）：
- 阻塞基线的集合通信走 ``ulysses_dispatch.ulysses_all_to_all``；env 未设时它是 functional
  collectives 路径，不走 ``dist.all_to_all_single``。为使事件序/计数可与 defer 在同一原语点对拍，
  顺序/等价/含反向计数用例在阻塞侧显式设 ``XTUNER_KDA_SP_A2A_ASYNC=1``（即文档所述的
  ``ulysses_scatter_heads_blocking`` = issue+finish 背靠背），并用 env 未设的 functional 路径
  单独断言前向数量同为 6。
- 分支选择（flag handling）用例在主进程 1-rank gloo 组内跑（world==1 时 a2a 原语早退、不触碰
  CUDA 流/事件），阻塞侧 recorder 委托给 ``ulysses_scatter_heads_blocking``（world==1 恒等返回）。

TestKdaSpDeferScheduleEquivalence（2-rank gloo spawn）
    test_defer_and_blocking_bitwise_equal: DEFER=1 与 DEFER=0（ASYNC=1 与 functional 两条参考
        路径）的 projected_output/raw_output、对 hidden_states 与全部参数的梯度、mock conv/core
        内核的输入与输出全部 ``torch.equal``。
TestKdaSpDeferFenceOrdering（2-rank gloo spawn）
    test_deferred_fences_at_consumption: DEFER=1 完整事件序列 == 五连发（q/k/v/g/beta）→
        (fence, conv)×3（``_finish_short_conv``）→ fence(g) → fence(beta) → gate → core →
        issue(out) → gate_output → fence(out) → o_norm。
    test_blocking_fences_per_call: DEFER=0（ASYNC=1 路由）事件序列 == 每对 (issue, fence) 相邻
        （发出即围栏），与消费计算交错。
TestKdaSpCollectiveParity（2-rank gloo spawn）
    test_six_a2a_per_forward_on_both_schedules: DEFER=1 与 DEFER=0（ASYNC=1 与 functional 路由）
        每次前向均恰 6 次 a2a（q/k/v/g/beta/out）。
    test_fwd_bwd_a2a_parity_is_six_and_six: 含反向时 DEFER=1 与 DEFER=0（ASYNC=1）均 12 次
        （前向 6 + 反向 6）。
TestKdaSpDeferSendLayout（2-rank gloo spawn）
    test_six_independent_sends_match_projections: stub 捕获的 6 发 send 分别等于 q/k/v 投影
        transpose、g 投影 transpose、b_proj fp32 transpose、core 输出的 world 分片（无 qkv 合并）。
    test_beta_a2a_input_is_pre_sigmoid_fp32: beta 一发的 a2a 输入是 sigmoid **前**的 fp32
        transpose（不落在 (0, 1)），而 kernel 消费的 beta 已在 a2a 之后 sigmoid。
TestKdaSpFinishShortConv（2-rank gloo spawn）
    test_finish_short_conv_matches_blocking_post_fence_recipe: ``_finish_short_conv`` 的 fence 后
        尾段（token-major 转置 + weight/bias 按 sp_rank chunk + conv）与阻塞路径逐位一致。
TestKdaSpDeferFlagHandling（主进程 1-rank gloo）
    test_flag_selects_schedule_branches: 模块常量 True → 6 issue + 6 fence、0 次阻塞 wrapper；
        False → 反之（6 次 wrapper）。
TestKdaSpDeferEnvDefault
    test_env_default_resolves_to_deferred: 子进程验证 env 未设 → 常量为 True；"0" → False。
"""

import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

import xtuner.v1.module.attention.kda as kda_module
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.module.attention import kda_lever
from xtuner.v1.module.attention.attn_outputs import AttnOutputs
from xtuner.v1.ops.comm import all_to_all as all_to_all_module
from xtuner.v1.ops.comm import ulysses_dispatch


_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_PROBE_CODE = "import xtuner.v1.module.attention.kda_lever as m; print(m.SP_A2A_DEFER)"

# 用例尺寸：head_dim=16 使 mock 核可行（真实 chunk_kda 要求 128）；S_loc=8、world=2 → 全局 16。
_HIDDEN = 64
_HEADS = 4
_HEAD_DIM = 16
_CONV_W = 4
_WORLD = 2
_S_LOC = 8
_PROJ = _HEADS * _HEAD_DIM

# import 期捕获真实原语（recorder 委托目标）。
_REAL_ISSUE_DIM1 = kda_lever.issue_ulysses_dim1
_REAL_FINISH_DIM1 = kda_lever.finish_ulysses_dim1
_REAL_FINISH_SEQ_MAJOR = kda_lever.finish_ulysses_dim1_seq_major
_REAL_WAIT_A2A = ulysses_dispatch._wait_a2a
_REAL_C10D_A2A = dist.all_to_all_single
_REAL_FUNCTIONAL_A2A = all_to_all_module._all_to_all_single_autograd

# DEFER=1（wave6）的期望事件序列：五连发（q/k/v/g/beta 各一发）→ (fence, conv)×3（q/k/v 依次经
# _finish_short_conv：先 fence 再 conv）→ fence(g)、fence(beta) → gate → core → issue(out) →
# gate_output → fence(out) → o_norm。out 一发插在 _gate_output 之前、围栏在 o_norm 消费点，
# 是延迟调度的行为增量。
_DEFERRED_LOG: list[str] = [
    "issue",
    "issue",
    "issue",
    "issue",
    "issue",
    "fence",
    "conv",
    "fence",
    "conv",
    "fence",
    "conv",
    "fence",
    "fence",
    "gate",
    "core",
    "issue",
    "gate_output",
    "fence",
    "o_norm",
]

# DEFER=0（ASYNC=1 路由）的期望事件序列：每对 (issue, fence) 相邻（发出即围栏），与消费计算交错。
_BLOCKING_LOG: list[str] = [
    "issue",
    "fence",
    "conv",
    "issue",
    "fence",
    "conv",
    "issue",
    "fence",
    "conv",
    "issue",
    "fence",
    "issue",
    "fence",
    "gate",
    "core",
    "issue",
    "fence",
    "gate_output",
    "o_norm",
]


class _CpuWork:
    """同步 a2a stub 返回的 ``dist.Work`` dummy：恒已完成，只记录轮询次数。"""

    def __init__(self) -> None:
        self.is_completed_calls = 0

    def is_completed(self) -> bool:
        self.is_completed_calls += 1
        return True


# ---- worker/主进程共享的记录（用例内重置） ----
_LOG: list[str] = []
_A2A_MECHANISM: list[str] = []
_A2A_SENDS: list[torch.Tensor] = []
_A2A_RECVS: list[torch.Tensor] = []
_CORE_INPUTS: list[tuple[torch.Tensor, ...]] = []
_CORE_OUTPUTS: list[torch.Tensor] = []
_CONV_INPUTS: list[torch.Tensor] = []
_CONV_WEIGHTS: list[torch.Tensor] = []
_CONV_BIASES: list[torch.Tensor | None] = []
_CONV_OUTPUTS: list[torch.Tensor] = []


def _reset_records() -> None:
    _LOG.clear()
    _A2A_MECHANISM.clear()
    _A2A_SENDS.clear()
    _A2A_RECVS.clear()
    _CORE_INPUTS.clear()
    _CORE_OUTPUTS.clear()
    _CONV_INPUTS.clear()
    _CONV_WEIGHTS.clear()
    _CONV_BIASES.clear()
    _CONV_OUTPUTS.clear()


# ---- 确定性 CPU stand-in 内核（可微、逐位确定） ----


def _mock_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    backend: str | None = None,
    cu_seqlens: object | None = None,
    **kwargs: object,
) -> tuple[torch.Tensor, None]:
    """因果深度卷积 stand-in：x 为 (B, L, C) token-major，weight 为 (C, W)。"""
    del backend, cu_seqlens, kwargs
    _LOG.append("conv")
    _CONV_INPUTS.append(x.detach().clone())
    _CONV_WEIGHTS.append(weight.detach().clone())
    _CONV_BIASES.append(bias.detach().clone() if bias is not None else None)
    win = weight.shape[1]
    padded = nn.functional.pad(x, (0, 0, win - 1, 0))
    out = torch.zeros_like(x)
    for i in range(win):
        out = out + padded[:, i : i + x.shape[1], :] * weight[:, i]
    if bias is not None:
        out = out + bias
    if activation == "silu":
        out = out * torch.sigmoid(out)
    _CONV_OUTPUTS.append(out.detach().clone())
    return out, None


def _mock_fused_kda_gate(
    g_raw: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
) -> torch.Tensor:
    """门 stand-in：``g * exp(a_log) + dt_bias/2``（逐头广播，可微，确定性）。"""
    del lower_bound
    _LOG.append("gate")
    heads = a_log.shape[0]
    dim = g_raw.shape[-1]
    scale = torch.exp(a_log).reshape(1, 1, heads, 1).to(g_raw.dtype)
    bias = (dt_bias if dt_bias is not None else torch.zeros(heads * dim)).reshape(1, 1, heads, dim).to(g_raw.dtype)
    return g_raw * scale + 0.5 * bias


def _mock_kda_core(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    **kwargs: object,
) -> tuple[torch.Tensor, None]:
    """KDA core stand-in：``q + k + v + g * beta``（形状同 q，可微，记录全部输入与输出）。"""
    del kwargs
    _LOG.append("core")
    _CORE_INPUTS.append(
        (q.detach().clone(), k.detach().clone(), v.detach().clone(), g.detach().clone(), beta.detach().clone())
    )
    out = q + k + v + g * beta.unsqueeze(-1)
    _CORE_OUTPUTS.append(out.detach().clone())
    return out, None


def _mock_gate_param(param: torch.Tensor) -> torch.Tensor:
    """``_gate_param`` stand-in：CPU 张量直接 fp32 化（跳过 DTensor materialize/缓存）。"""
    return param.float()


# ---- recorder 委托 ----


def _sync_a2a_issue(send: torch.Tensor, group: dist.ProcessGroup) -> tuple[torch.Tensor, _CpuWork]:
    """``ulysses_dispatch._a2a_issue`` 的同步 stub：gloo 上直接 all_to_all_single（无流/事件）。

    捕获 send 缓冲（``_AsyncUlyssesDim1`` reshape 后的 ``(world, dim/world, ...)`` 布局）供 send
    布局断言使用。前向 issue 与反向（``_AsyncUlyssesDim1.backward``）都经模块 globals 解析到此处，
    因此 a2a 计数天然覆盖 fwd+bwd。
    """
    _LOG.append("issue")
    _A2A_SENDS.append(send.detach().clone())
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    # 同步 stub：all_to_all_single 返回时 recv 已就绪，可直接快照。
    _A2A_RECVS.append(recv.detach().clone())
    return recv, _CpuWork()


def _recording_wait_a2a(work: object) -> None:
    """``ulysses_dispatch._wait_a2a`` 的 recorder：每个围栏点记一条 "fence"。"""
    _LOG.append("fence")
    _REAL_WAIT_A2A(work)


def _counting_c10d_a2a(*args: object, **kwargs: object) -> object:
    _A2A_MECHANISM.append("c10d")
    return _REAL_C10D_A2A(*args, **kwargs)


def _counting_functional_a2a(*args: object, **kwargs: object) -> object:
    _A2A_MECHANISM.append("functional")
    return _REAL_FUNCTIONAL_A2A(*args, **kwargs)


def _install_worker_stubs() -> None:
    """spawn 子进程内的 CPU 化补丁（子进程随用例退出，无需还原）。"""
    torch.set_num_threads(1)
    os.environ.pop("XTUNER_KDA_SP_A2A_ASYNC", None)
    ulysses_dispatch._a2a_issue = _sync_a2a_issue  # type: ignore[assignment]
    ulysses_dispatch._wait_a2a = _recording_wait_a2a  # type: ignore[assignment]
    kda_module.causal_conv1d = _mock_causal_conv1d  # type: ignore[assignment]
    kda_module.chunk_kda = _mock_kda_core  # type: ignore[assignment]
    kda_module.fused_recurrent_kda = _mock_kda_core  # type: ignore[assignment]
    kda_module.fused_kda_gate = _mock_fused_kda_gate  # type: ignore[assignment]
    kda_module._gate_param = _mock_gate_param  # type: ignore[assignment]
    dist.all_to_all_single = _counting_c10d_a2a  # type: ignore[assignment]
    all_to_all_module._all_to_all_single_autograd = _counting_functional_a2a  # type: ignore[assignment]
    _reset_records()


class _RecordingNorm(nn.Module):
    """o_norm 包装：记录消费事件后委托真实模块（实例属性替换须保持 Module 类型）。"""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor, g: torch.Tensor, **kwargs: object) -> torch.Tensor:
        _LOG.append("o_norm")
        return self.inner(x, g, **kwargs)


def _install_consumer_recorders(module: kda_module.KimiDeltaAttention) -> None:
    """记录 a2a 消费者（``_gate_output`` 与 ``o_norm``）的调用时刻。"""
    real_gate_output = module._gate_output

    def _rec_gate_output(hidden_states: torch.Tensor) -> torch.Tensor:
        _LOG.append("gate_output")
        return real_gate_output(hidden_states)

    module._gate_output = _rec_gate_output  # type: ignore[method-assign]
    module.o_norm = _RecordingNorm(module.o_norm)  # type: ignore[assignment]


# ---- fixtures ----


def _build_kda_layer() -> kda_module.KimiDeltaAttention:
    """构建 fp32（默认 dtype）的最小 KDA 层（head_dim=16 使 mock 核可行）。"""
    torch.manual_seed(7)
    return kda_module.KimiDeltaAttention(
        hidden_size=_HIDDEN,
        num_heads=_HEADS,
        head_dim=_HEAD_DIM,
        conv_kernel_size=_CONV_W,
    )


def _sp_fixtures(
    rank: int, world: int
) -> tuple[kda_module.KimiDeltaAttention, DeviceMesh, SequenceContext, torch.Tensor]:
    """每个 rank 一份的同构 KDA 层、SP mesh、SequenceContext（from_input_ids + split）与本地切片。"""
    mesh = init_device_mesh("cpu", (world,))
    module = _build_kda_layer()
    s_glob = _S_LOC * world
    torch.manual_seed(11)
    hidden_global = torch.randn(1, s_glob, _HIDDEN, dtype=torch.float32)
    hidden_local = hidden_global[:, rank * _S_LOC : (rank + 1) * _S_LOC, :].contiguous()
    ids_global = torch.randint(0, 100, (1, s_glob))
    seq_ctx = SequenceContext.from_input_ids((ids_global,), sp_mesh=mesh, device="cpu").split(mesh)
    assert seq_ctx.input_ids.shape[1] == _S_LOC, seq_ctx.input_ids.shape
    assert seq_ctx.sp_rank == rank
    assert seq_ctx.cu_seq_lens_q.tolist() == [0, s_glob]
    return module, mesh, seq_ctx, hidden_local


def _forward_grad(
    module: kda_module.KimiDeltaAttention, hidden_local: torch.Tensor, seq_ctx: SequenceContext, go: torch.Tensor
) -> tuple[AttnOutputs, list[torch.Tensor]]:
    """单次前向 + 对 hidden_states 与全部参数的梯度（一次 autograd.grad，按 [hidden, *params] 序）。

    默认 ``allow_unused=False``：任何参数未被计算图使用都会直接报错，本身即是契约检查。
    """
    hidden = hidden_local.detach().clone().requires_grad_(True)
    out = module.forward_for_sp(hidden, seq_ctx)
    grads = torch.autograd.grad((out["projected_output"] * go).sum(), [hidden, *module.parameters()])
    return out, grads


class _RunSnapshot(TypedDict):
    """单次调度运行的输出/梯度与 mock 内核输入输出快照（逐位对比的最小单元）。"""

    projected: torch.Tensor
    raw: torch.Tensor
    grads: list[torch.Tensor]
    conv_in: list[torch.Tensor]
    conv_out: list[torch.Tensor]
    conv_w: list[torch.Tensor]
    conv_b: list[torch.Tensor | None]
    core_in: list[tuple[torch.Tensor, ...]]
    core_out: list[torch.Tensor]


def _run_schedule(
    module: kda_module.KimiDeltaAttention,
    hidden_local: torch.Tensor,
    seq_ctx: SequenceContext,
    go: torch.Tensor,
    *,
    defer: bool,
    async_env: str | None,
) -> _RunSnapshot:
    """以给定调度跑一次前向+梯度并快照（``async_env=None`` 表示 pop 掉 ASYNC env）。"""
    kda_lever.SP_A2A_DEFER = defer
    if async_env is None:
        os.environ.pop("XTUNER_KDA_SP_A2A_ASYNC", None)
    else:
        os.environ["XTUNER_KDA_SP_A2A_ASYNC"] = async_env
    _reset_records()
    out, grads = _forward_grad(module, hidden_local, seq_ctx, go)
    return {
        "projected": out["projected_output"],
        "raw": out["raw_output"],
        "grads": grads,
        "conv_in": list(_CONV_INPUTS),
        "conv_out": list(_CONV_OUTPUTS),
        "conv_w": list(_CONV_WEIGHTS),
        "conv_b": list(_CONV_BIASES),
        "core_in": list(_CORE_INPUTS),
        "core_out": list(_CORE_OUTPUTS),
    }


def _equal_maybe_none(a: torch.Tensor | None, b: torch.Tensor | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return bool(torch.equal(a, b))


def _assert_bitwise_same(a: _RunSnapshot, b: _RunSnapshot, tag: str) -> None:
    """两调度快照（输出/全部叶子梯度/mock 内核输入输出）逐位一致。"""
    assert torch.equal(a["projected"], b["projected"]), f"{tag}: projected_output 不逐位一致"
    assert torch.equal(a["raw"], b["raw"]), f"{tag}: raw_output 不逐位一致"
    assert len(a["grads"]) == len(b["grads"]), f"{tag}: 叶子数不一致"
    for i, (ga, gb) in enumerate(zip(a["grads"], b["grads"])):
        assert torch.equal(ga, gb), f"{tag}: 叶子梯度 #{i} 不逐位一致"
    for i, (xa, xb) in enumerate(zip(a["conv_in"], b["conv_in"])):
        assert torch.equal(xa, xb), f"{tag}: conv 输入 #{i} 不逐位一致"
    for i, (xa, xb) in enumerate(zip(a["conv_w"], b["conv_w"])):
        assert torch.equal(xa, xb), f"{tag}: conv weight #{i} 不逐位一致"
    for i, (xa, xb) in enumerate(zip(a["conv_b"], b["conv_b"])):
        assert _equal_maybe_none(xa, xb), f"{tag}: conv bias #{i} 不逐位一致"
    for i, (xa, xb) in enumerate(zip(a["conv_out"], b["conv_out"])):
        assert torch.equal(xa, xb), f"{tag}: conv 输出 #{i} 不逐位一致"
    for i, (ta, tb) in enumerate(zip(a["core_in"], b["core_in"])):
        for j, (xa, xb) in enumerate(zip(ta, tb)):
            assert torch.equal(xa, xb), f"{tag}: core 输入 #{i}.{j} 不逐位一致"
    for i, (xa, xb) in enumerate(zip(a["core_out"], b["core_out"])):
        assert torch.equal(xa, xb), f"{tag}: core 输出 #{i} 不逐位一致"


# ---- spawn harness ----


def _worker(rank: int, world: int, store_path: str, case: str) -> None:
    dist.init_process_group("gloo", store=dist.FileStore(store_path, world), rank=rank, world_size=world)
    _install_worker_stubs()
    try:
        _CASES[case](rank, world)
    finally:
        dist.destroy_process_group()


def _run_spawned(case: str, tmp_path: Path) -> None:
    # NPU 盒上 transfer_to_npu 会把 gloo init 重映射到 hccl；与上一轮异步 communicator 拆除
    # 竞争，故前后各留缓冲（与 tests/model/test_glm53_kda_sp_async.py 同一约定）。
    time.sleep(2.0)
    mp.spawn(_worker, args=(_WORLD, str(tmp_path / "gloo_store"), case), nprocs=_WORLD, join=True)
    time.sleep(2.0)


# ---- 2-rank 用例 ----


def _case_equivalence(rank: int, world: int) -> None:
    """DEFER=1 vs DEFER=0：输出、全部叶子梯度、mock 内核输入/输出逐位一致（两条 blocking 参考路径）。"""
    module, _, seq_ctx, hidden_local = _sp_fixtures(rank, world)
    torch.manual_seed(13)
    go = torch.randn(1, _S_LOC, _HIDDEN, dtype=torch.float32)

    defer_run = _run_schedule(module, hidden_local, seq_ctx, go, defer=True, async_env=None)
    blocking_async_run = _run_schedule(module, hidden_local, seq_ctx, go, defer=False, async_env="1")
    blocking_functional_run = _run_schedule(module, hidden_local, seq_ctx, go, defer=False, async_env=None)

    assert defer_run["projected"].shape == (1, _S_LOC, _HIDDEN)
    assert defer_run["raw"].shape == (1, _S_LOC, _HIDDEN)
    # 非退化哨兵：mock 内核/真实 a2a 全链产出非零有限值，防止"全零对全零"的空等价。
    assert torch.isfinite(defer_run["projected"]).all()
    assert defer_run["projected"].abs().sum() > 0
    assert defer_run["grads"][0].abs().sum() > 0
    for i, g in enumerate(defer_run["grads"]):
        assert torch.isfinite(g).all(), f"defer 梯度 #{i} 含非有限值"

    # 每次前向恰 3 次 conv（q/k/v）与 1 次 core。
    for snap in (defer_run, blocking_async_run, blocking_functional_run):
        assert len(snap["conv_in"]) == 3 and len(snap["core_in"]) == 1, (len(snap["conv_in"]), len(snap["core_in"]))

    _assert_bitwise_same(defer_run, blocking_async_run, "defer vs blocking(ASYNC=1)")
    _assert_bitwise_same(defer_run, blocking_functional_run, "defer vs blocking(functional)")


def _case_ordering_defer(rank: int, world: int) -> None:
    """DEFER=1：五连发先于第一 fence；q/k/v 依次 fence+conv；out 一发在 gate_output 后、o_norm 前 fence。"""
    module, _, seq_ctx, hidden_local = _sp_fixtures(rank, world)
    _install_consumer_recorders(module)
    kda_lever.SP_A2A_DEFER = True
    os.environ.pop("XTUNER_KDA_SP_A2A_ASYNC", None)
    _reset_records()
    module.forward_for_sp(hidden_local, seq_ctx)
    assert _LOG == _DEFERRED_LOG, f"deferred 事件序列不符: {_LOG}"


def _case_ordering_blocking(rank: int, world: int) -> None:
    """DEFER=0（ASYNC=1 路由）：每次调用发出即围栏，与消费计算交错。"""
    module, _, seq_ctx, hidden_local = _sp_fixtures(rank, world)
    _install_consumer_recorders(module)
    kda_lever.SP_A2A_DEFER = False
    os.environ["XTUNER_KDA_SP_A2A_ASYNC"] = "1"
    _reset_records()
    module.forward_for_sp(hidden_local, seq_ctx)
    assert _LOG == _BLOCKING_LOG, f"blocking 事件序列不符: {_LOG}"


def _case_parity(rank: int, world: int) -> None:
    """每次前向的 a2a 数：defer 与 blocking（ASYNC=1 / functional 路由）均 6 次（q/k/v/g/beta/out）。"""
    module, _, seq_ctx, hidden_local = _sp_fixtures(rank, world)

    kda_lever.SP_A2A_DEFER = True
    os.environ.pop("XTUNER_KDA_SP_A2A_ASYNC", None)
    _reset_records()
    module.forward_for_sp(hidden_local, seq_ctx)
    assert _A2A_MECHANISM.count("c10d") == 6, _A2A_MECHANISM
    assert _A2A_MECHANISM.count("functional") == 0, _A2A_MECHANISM

    kda_lever.SP_A2A_DEFER = False
    os.environ["XTUNER_KDA_SP_A2A_ASYNC"] = "1"
    _reset_records()
    module.forward_for_sp(hidden_local, seq_ctx)
    assert _A2A_MECHANISM.count("c10d") == 6, _A2A_MECHANISM
    assert _A2A_MECHANISM.count("functional") == 0, _A2A_MECHANISM

    os.environ.pop("XTUNER_KDA_SP_A2A_ASYNC", None)
    _reset_records()
    module.forward_for_sp(hidden_local, seq_ctx)
    assert _A2A_MECHANISM.count("functional") == 6, _A2A_MECHANISM
    assert _A2A_MECHANISM.count("c10d") == 0, _A2A_MECHANISM


def _case_parity_fwd_bwd(rank: int, world: int) -> None:
    """前向+反向的 a2a 总数：defer 与 blocking（ASYNC=1）均 12 次（前向 6 + 反向 6）。"""
    module, _, seq_ctx, hidden_local = _sp_fixtures(rank, world)
    torch.manual_seed(13)
    go = torch.randn(1, _S_LOC, _HIDDEN, dtype=torch.float32)

    kda_lever.SP_A2A_DEFER = True
    os.environ.pop("XTUNER_KDA_SP_A2A_ASYNC", None)
    _reset_records()
    _forward_grad(module, hidden_local, seq_ctx, go)
    assert _A2A_MECHANISM.count("c10d") == 12, _A2A_MECHANISM  # 前向 6（q/k/v/g/beta/out）+ 反向 6

    kda_lever.SP_A2A_DEFER = False
    os.environ["XTUNER_KDA_SP_A2A_ASYNC"] = "1"
    _reset_records()
    _forward_grad(module, hidden_local, seq_ctx, go)
    assert _A2A_MECHANISM.count("c10d") == 12, _A2A_MECHANISM  # 前向 6 + 反向 6


def _case_send_layout(rank: int, world: int) -> None:
    """DEFER=1 的 6 发独立 a2a：send 各等于 q/k/v/g/beta/out 投影的 world 分片（无 qkv 合并）。"""
    module, _, seq_ctx, hidden_local = _sp_fixtures(rank, world)

    kda_lever.SP_A2A_DEFER = True
    os.environ.pop("XTUNER_KDA_SP_A2A_ASYNC", None)
    _reset_records()
    module.forward_for_sp(hidden_local, seq_ctx)

    # 仅前向：q/k/v/g/beta/out 共 6 发（stub 收到的 send 已被 _AsyncUlyssesDim1 reshape 成
    # (world, dim/world, ...)）。
    assert len(_A2A_SENDS) == 6, len(_A2A_SENDS)
    with torch.no_grad():
        exp_q = module.q_proj(hidden_local).transpose(1, 2).reshape(world, _PROJ // world, _S_LOC)
        exp_k = module.k_proj(hidden_local).transpose(1, 2).reshape(world, _PROJ // world, _S_LOC)
        exp_v = module.v_proj(hidden_local).transpose(1, 2).reshape(world, _PROJ // world, _S_LOC)
        exp_g = module.f_b_proj(module.f_a_proj(hidden_local)).transpose(1, 2).reshape(world, _PROJ // world, _S_LOC)
        exp_beta = module.b_proj(hidden_local).float().transpose(1, 2).reshape(world, _HEADS // world, _S_LOC)
    exp_out = _CORE_OUTPUTS[-1].reshape(world, _S_LOC, _HEADS // world, _HEAD_DIM)

    assert _A2A_SENDS[0].shape == (world, _PROJ // world, _S_LOC), _A2A_SENDS[0].shape
    assert _A2A_SENDS[3].shape == (world, _PROJ // world, _S_LOC), _A2A_SENDS[3].shape
    assert _A2A_SENDS[4].shape == (world, _HEADS // world, _S_LOC), _A2A_SENDS[4].shape
    assert _A2A_SENDS[5].shape == (world, _S_LOC, _HEADS // world, _HEAD_DIM), _A2A_SENDS[5].shape
    assert torch.equal(_A2A_SENDS[0], exp_q), "q send 不等于 q 投影 transpose 的 world 分片"
    assert torch.equal(_A2A_SENDS[1], exp_k), "k send 不等于 k 投影 transpose 的 world 分片"
    assert torch.equal(_A2A_SENDS[2], exp_v), "v send 不等于 v 投影 transpose 的 world 分片"
    assert torch.equal(_A2A_SENDS[3], exp_g), "g send 不等于 g 投影 transpose 的 world 分片"
    assert torch.equal(_A2A_SENDS[4], exp_beta), "beta send 不等于 b_proj fp32 transpose 的 world 分片"
    assert torch.equal(_A2A_SENDS[5], exp_out), "out send 不等于 core 输出的 world 分块"


def _case_beta_pre_sigmoid(rank: int, world: int) -> None:
    """DEFER=1 的 beta 发：a2a 输入是 sigmoid **前**的 fp32 transpose（非 (0,1) 有界），sigmoid 在围栏后。"""
    module, _, seq_ctx, hidden_local = _sp_fixtures(rank, world)
    with torch.no_grad():
        module.b_proj.weight.mul_(50.0)  # 放大 raw 值，使"未过 sigmoid"可由越界值直接证得

    kda_lever.SP_A2A_DEFER = True
    os.environ.pop("XTUNER_KDA_SP_A2A_ASYNC", None)
    _reset_records()
    module.forward_for_sp(hidden_local, seq_ctx)

    assert len(_A2A_SENDS) == 6, len(_A2A_SENDS)
    beta_send = _A2A_SENDS[4]
    assert beta_send.shape == (world, _HEADS // world, _S_LOC), beta_send.shape
    assert beta_send.dtype == torch.float32, beta_send.dtype
    with torch.no_grad():
        beta_raw = module.b_proj(hidden_local).float()
    exp_beta = beta_raw.transpose(1, 2).reshape(world, _HEADS // world, _S_LOC)
    assert torch.equal(beta_send, exp_beta), "beta send 不等于 sigmoid 前的 b_proj fp32 输出分块"
    # 反转 wave7 断言：a2a 输入未过 sigmoid，因此不被压进 (0, 1)。
    assert bool((beta_send.abs() > 1.0).any()), "beta send 疑似已过 sigmoid（幅值被压进 (0, 1)）"
    # 而 kernel 消费的 beta 是 a2a 结果在围栏后做的 sigmoid（逐位：finish 布局 + sigmoid）。
    beta_recv = _A2A_RECVS[4]
    finished = beta_recv.permute(1, 0, 2).reshape(1, _HEADS // world, _S_LOC * world).transpose(1, 2)
    core_beta = _CORE_INPUTS[-1][4]
    assert core_beta.shape == (1, _S_LOC * world, _HEADS // world), core_beta.shape
    assert torch.equal(core_beta, finished.sigmoid()), "kernel 消费的 beta 不是 a2a 结果的 sigmoid"
    # sigmoid 值域守卫（闭区间：fp32 下大幅值 raw 会饱和到恰 1.0）。
    assert bool((core_beta >= 0).all()) and bool((core_beta <= 1).all()), "kernel 消费的 beta 越出 [0, 1]"


def _case_finish_short_conv(rank: int, world: int) -> None:
    """``_finish_short_conv`` 的围栏后尾段（token-major 转置 + weight/bias 按 sp_rank chunk + conv）。

    参照为阻塞路径 ``_sp_short_conv`` 的同一配方：对同一 a2a gather 结果，两种调度喂给 conv 的
    x/weight/bias 与 conv 输出必须逐位一致。
    """
    module, _, seq_ctx, hidden_local = _sp_fixtures(rank, world)
    proj = _HEADS * _HEAD_DIM

    kda_lever.SP_A2A_DEFER = False
    os.environ["XTUNER_KDA_SP_A2A_ASYNC"] = "1"
    _reset_records()
    module.forward_for_sp(hidden_local, seq_ctx)
    ref_x = list(_CONV_INPUTS)
    ref_w = list(_CONV_WEIGHTS)
    ref_b = list(_CONV_BIASES)
    ref_out = list(_CONV_OUTPUTS)
    assert len(ref_x) == 3 and len(ref_out) == 3, (len(ref_x), len(ref_out))

    kda_lever.SP_A2A_DEFER = True
    os.environ.pop("XTUNER_KDA_SP_A2A_ASYNC", None)
    _reset_records()
    module.forward_for_sp(hidden_local, seq_ctx)
    assert len(_CONV_INPUTS) == 3 and len(_CONV_OUTPUTS) == 3, (len(_CONV_INPUTS), len(_CONV_OUTPUTS))

    for i, conv in enumerate((module.q_conv1d, module.k_conv1d, module.v_conv1d)):
        x, w, b, out = _CONV_INPUTS[i], _CONV_WEIGHTS[i], _CONV_BIASES[i], _CONV_OUTPUTS[i]
        # gather 后的 token-major 全局序列、按 sp_rank 切的通道分片。
        assert x.shape == (1, _S_LOC * world, proj // world), x.shape
        assert torch.equal(x, ref_x[i]), f"conv 输入 #{i} 与阻塞路径尾段不一致"
        assert torch.equal(w, ref_w[i]), f"conv weight #{i} 与阻塞路径尾段不一致"
        assert _equal_maybe_none(b, ref_b[i]), f"conv bias #{i} 与阻塞路径尾段不一致"
        assert torch.equal(out, ref_out[i]), f"conv 输出 #{i} 与阻塞路径尾段不一致"
        # weight/bias 显式按 sp_rank chunk（对照完整 materialize 结果）。
        exp_w, exp_b = conv.materialize_weight_bias()
        assert torch.equal(w, exp_w.chunk(world, dim=0)[rank]), f"conv weight #{i} 未按 sp_rank chunk"
        assert _equal_maybe_none(b, exp_b.chunk(world, dim=0)[rank] if exp_b is not None else None), (
            f"conv bias #{i} 未按 sp_rank chunk"
        )


_CASES: dict[str, Callable[[int, int], None]] = {
    "equivalence": _case_equivalence,
    "ordering_defer": _case_ordering_defer,
    "ordering_blocking": _case_ordering_blocking,
    "parity": _case_parity,
    "parity_fwd_bwd": _case_parity_fwd_bwd,
    "send_layout": _case_send_layout,
    "beta_pre_sigmoid": _case_beta_pre_sigmoid,
    "finish_short_conv": _case_finish_short_conv,
}


# ---- 主进程（1-rank gloo）flag/分支选择 ----

_FLAG_ISSUES: list[str] = []
_FLAG_FENCES: list[str] = []
_FLAG_BLOCKING: list[str] = []


def _recording_issue(x: torch.Tensor, group: dist.ProcessGroup) -> tuple[torch.Tensor, object, torch.Size]:
    _FLAG_ISSUES.append("issue")
    return _REAL_ISSUE_DIM1(x, group)


def _recording_finish(recv: torch.Tensor, work: object, out_shape: torch.Size) -> torch.Tensor:
    _FLAG_FENCES.append("finish")
    return _REAL_FINISH_DIM1(recv, work, out_shape)


def _recording_finish_seq_major(recv: torch.Tensor, work: object) -> torch.Tensor:
    # DIRECT_CONV=1（默认）时 q/k/v 的 _finish_short_conv 走单拷贝 seq-major finish。
    _FLAG_FENCES.append("finish_seq")
    return _REAL_FINISH_SEQ_MAJOR(recv, work)


def _make_blocking_recorder(tag: str) -> Callable[[torch.Tensor, int, int, DeviceMesh], torch.Tensor]:
    def _record(x: torch.Tensor, scatter_dim: int, gather_dim: int, mesh: DeviceMesh) -> torch.Tensor:
        _FLAG_BLOCKING.append(tag)
        # world==1 时为恒等返回：分支选择用例只关心哪个调用点被触发。
        return ulysses_dispatch.ulysses_scatter_heads_blocking(x, mesh.get_group())

    return _record


_BLOCKING_WRAPPERS: tuple[tuple[str, str], ...] = (
    ("q", "_all_to_all_conv_pre_q"),
    ("k", "_all_to_all_conv_pre_k"),
    ("v", "_all_to_all_conv_pre_v"),
    ("g", "_all_to_all_g"),
    ("beta", "_all_to_all_beta"),
    ("out", "_all_to_all_out"),
)


def _install_flag_test_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kda_module, "causal_conv1d", _mock_causal_conv1d)
    monkeypatch.setattr(kda_module, "chunk_kda", _mock_kda_core)
    monkeypatch.setattr(kda_module, "fused_recurrent_kda", _mock_kda_core)
    monkeypatch.setattr(kda_module, "fused_kda_gate", _mock_fused_kda_gate)
    monkeypatch.setattr(kda_module, "_gate_param", _mock_gate_param)
    monkeypatch.setattr(kda_lever, "issue_ulysses_dim1", _recording_issue)
    monkeypatch.setattr(kda_lever, "finish_ulysses_dim1", _recording_finish)
    monkeypatch.setattr(kda_lever, "finish_ulysses_dim1_seq_major", _recording_finish_seq_major)
    for tag, name in _BLOCKING_WRAPPERS:
        monkeypatch.setattr(kda_module, name, _make_blocking_recorder(tag))


def _module_const_with_env(env_value: str | None) -> bool:
    """在干净子进程中以给定 env 值 import kda，返回其 ``_KDA_SP_A2A_DEFER`` 常量。"""
    env = dict(os.environ)
    env.pop("XTUNER_KDA_SP_A2A_DEFER", None)
    if env_value is not None:
        env["XTUNER_KDA_SP_A2A_DEFER"] = env_value
    env["PYTHONPATH"] = str(_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _ENV_PROBE_CODE],
        env=env,
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    return lines[-1].strip() == "True"


class TestKdaSpDeferScheduleEquivalence:
    """DEFER=1 与 DEFER=0 的逐位等价（2-rank gloo，mock 设备内核 + 真实 a2a 移动）。"""

    def test_defer_and_blocking_bitwise_equal(self, tmp_path: Path) -> None:
        _run_spawned("equivalence", tmp_path)


class TestKdaSpDeferFenceOrdering:
    """延迟 fence 的行为增量：fence 移到消费点、5 发连发后逐个消费（2-rank gloo）。"""

    def test_deferred_fences_at_consumption(self, tmp_path: Path) -> None:
        _run_spawned("ordering_defer", tmp_path)

    def test_blocking_fences_per_call(self, tmp_path: Path) -> None:
        _run_spawned("ordering_blocking", tmp_path)


class TestKdaSpCollectiveParity:
    """每前向 a2a 数：defer 与 blocking 均 6（q/k/v/g/beta/out）；含反向均 12（6+6）。"""

    def test_six_a2a_per_forward_on_both_schedules(self, tmp_path: Path) -> None:
        _run_spawned("parity", tmp_path)

    def test_fwd_bwd_a2a_parity_is_six_and_six(self, tmp_path: Path) -> None:
        _run_spawned("parity_fwd_bwd", tmp_path)


class TestKdaSpDeferSendLayout:
    """wave6 的 6 发独立 send 布局与 beta 的 sigmoid 后置（2-rank gloo）。"""

    def test_six_independent_sends_match_projections(self, tmp_path: Path) -> None:
        _run_spawned("send_layout", tmp_path)

    def test_beta_a2a_input_is_pre_sigmoid_fp32(self, tmp_path: Path) -> None:
        _run_spawned("beta_pre_sigmoid", tmp_path)


class TestKdaSpFinishShortConv:
    """``_finish_short_conv``（fence + 转置 + sp_rank chunk + conv）与阻塞路径尾段逐位一致（2-rank gloo）。"""

    def test_finish_short_conv_matches_blocking_post_fence_recipe(self, tmp_path: Path) -> None:
        _run_spawned("finish_short_conv", tmp_path)


class TestKdaSpDeferFlagHandling:
    """模块常量 ``_KDA_SP_A2A_DEFER`` 的分支选择（主进程 1-rank gloo，world==1 a2a 早退）。"""

    def test_flag_selects_schedule_branches(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        dist.init_process_group("gloo", store=dist.FileStore(str(tmp_path / "gloo_store"), 1), rank=0, world_size=1)
        try:
            _install_flag_test_stubs(monkeypatch)
            _reset_records()
            module = _build_kda_layer()
            mesh = init_device_mesh("cpu", (1,))
            ids = torch.randint(0, 100, (1, _S_LOC))
            seq_ctx = SequenceContext.from_input_ids((ids,), sp_mesh=mesh, device="cpu").split(mesh)
            hidden = torch.randn(1, _S_LOC, _HIDDEN, dtype=torch.float32)

            monkeypatch.setattr(kda_lever, "SP_A2A_DEFER", True)
            monkeypatch.setattr(kda_lever, "SP_A2A_DIRECT_CONV", True)
            _FLAG_ISSUES.clear()
            _FLAG_FENCES.clear()
            _FLAG_BLOCKING.clear()
            out = module.forward_for_sp(hidden, seq_ctx)
            assert out["projected_output"].shape == (1, _S_LOC, _HIDDEN)
            assert len(_FLAG_ISSUES) == 6, _FLAG_ISSUES  # q / k / v / g / beta / out
            # q/k/v 经 _finish_short_conv（DIRECT_CONV=1 → seq-major），g/beta/out 经 dim1 finish。
            assert sorted(_FLAG_FENCES) == sorted(["finish_seq"] * 3 + ["finish"] * 3), _FLAG_FENCES
            assert _FLAG_BLOCKING == [], _FLAG_BLOCKING

            monkeypatch.setattr(kda_lever, "SP_A2A_DEFER", False)
            _FLAG_ISSUES.clear()
            _FLAG_FENCES.clear()
            _FLAG_BLOCKING.clear()
            out = module.forward_for_sp(hidden, seq_ctx)
            assert out["projected_output"].shape == (1, _S_LOC, _HIDDEN)
            assert _FLAG_ISSUES == [], _FLAG_ISSUES
            assert _FLAG_FENCES == [], _FLAG_FENCES
            assert sorted(_FLAG_BLOCKING) == ["beta", "g", "k", "out", "q", "v"], _FLAG_BLOCKING
        finally:
            dist.destroy_process_group()


class TestKdaSpDeferEnvDefault:
    """env 解析契约：未设默认延迟调度，``XTUNER_KDA_SP_A2A_DEFER=0`` 恢复阻塞调度。"""

    def test_env_default_resolves_to_deferred(self) -> None:
        assert kda_lever.SP_A2A_DEFER == (os.environ.get("XTUNER_KDA_SP_A2A_DEFER", "1") == "1")
        assert _module_const_with_env(None) is True, "env 未设应解析为延迟调度"
        assert _module_const_with_env("0") is False, "XTUNER_KDA_SP_A2A_DEFER=0 应解析为阻塞调度"
