# Copyright © 2026 Huawei Technologies Co., Ltd.
"""KDA Ulysses SP 复制头分支（``sp_size > num_heads``）的等价测试。

被测对象：``xtuner/v1/module/attention/kda.py`` 的
``KimiDeltaAttention._forward_for_ulysses_sp_replicated``（经 ``forward_for_sp`` 自动路由）。

背景：GLM-5.3-Flash KDA ``num_heads=64``，``sp_size=96/128`` 时无法按头切分（原
``num_heads % sp_size`` 断言拦截）。复制头分支沿用已验证的 GLM-5.3 vision SP 方案
（``modeling_vision.Glm53VisionAttention``）：每个头以 ``repeat_interleave`` 填充
``replica = sp_size / num_heads`` 个 a2a 槽位（槽 s 携带头 ``s // replica``），每个 rank 对
自己的一个头槽跑全序列 KDA core，输出 a2a 后按 ``::replica`` 只保留每个头的首份拷贝
（重复槽位在反向收到零梯度，每个头恰好计一次）。

验收：与非 SP 全序列参考（同一模块、无 SP mesh、mock 内核确定性 fp32）对比——

- 输出（raw/projected）逐位一致（a2a 是纯置换，内核 mock 按元素/全序列确定性计算）；
- mock core 的输入逐位等于参考对应头切片（复制/a2a 管线精确性的定点检查）；
- mock conv 收到的 weight 逐位等于参考全量 weight 的对应头块；
- hidden 梯度切片、跨 rank 求和后的参数梯度与参考 ``assert_close``（加法结合序不同，非逐位）；
- 每次前向恰 6 次 a2a（q/k/v/g/beta/out，与头切分路径相同）；
- ``num_heads`` 与 ``sp_size`` 互不整除时 ``ValueError``。

测试环境：纯 CPU / gloo（FileStore spawn，无 NPU、无 GPU 标记）；设备侧内核
（``causal_conv1d`` / ``chunk_kda`` / ``fused_recurrent_kda`` / ``fused_kda_gate`` /
``_gate_param``）在 ``kda`` 模块层替换为确定性可微 CPU stand-in，o_norm/o_proj 走真实 eager 路径
（与 ``test_glm53_kda_sp_defer.py`` 同一约定）。
"""

import os
import time
from pathlib import Path
from typing import Sequence, TypedDict

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

import xtuner.v1.module.attention.kda as kda_module
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.module.attention.attn_outputs import AttnOutputs
from xtuner.v1.ops.comm import all_to_all as all_to_all_module


# 用例尺寸：world=4、num_heads=2 → replica=2（sp_size=4 > num_heads=2，复制头路径）；
# head_dim=16 使 mock 核可行（真实 chunk_kda 要求 128）；S_loc=8 → 全局 32。
_HIDDEN = 32
_HEADS = 2
_HEAD_DIM = 16
_CONV_W = 4
_WORLD = 4
_S_LOC = 8
_REPLICA = _WORLD // _HEADS

_REAL_C10D_A2A = dist.all_to_all_single
_REAL_FUNCTIONAL_A2A = all_to_all_module._all_to_all_single_autograd

_A2A_MECHANISM: list[str] = []
_CORE_INPUTS: list[tuple[torch.Tensor, ...]] = []
_CONV_WEIGHTS: list[torch.Tensor] = []


def _reset_records() -> None:
    _A2A_MECHANISM.clear()
    _CORE_INPUTS.clear()
    _CONV_WEIGHTS.clear()


# ---- 确定性 CPU stand-in 内核（可微、逐位确定；与 test_glm53_kda_sp_defer.py 同构） ----


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
    _CONV_WEIGHTS.append(weight.detach().clone())
    win = weight.shape[1]
    padded = nn.functional.pad(x, (0, 0, win - 1, 0))
    out = torch.zeros_like(x)
    for i in range(win):
        out = out + padded[:, i : i + x.shape[1], :] * weight[:, i]
    if bias is not None:
        out = out + bias
    if activation == "silu":
        out = out * torch.sigmoid(out)
    return out, None


def _mock_fused_kda_gate(
    g_raw: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
) -> torch.Tensor:
    """门 stand-in：``g * exp(a_log) + dt_bias/2``（逐头广播，可微，确定性）。"""
    del lower_bound
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
    """KDA core stand-in：``q + k + v + g * beta``（形状同 q，可微，记录全部输入）。"""
    del kwargs
    _CORE_INPUTS.append((q.detach().clone(), k.detach().clone(), v.detach().clone(), g.detach().clone()))
    return q + k + v + g * beta.unsqueeze(-1), None


def _mock_gate_param(param: torch.Tensor) -> torch.Tensor:
    """``_gate_param`` stand-in：CPU 张量直接 fp32 化（跳过 DTensor materialize/缓存）。"""
    return param.float()


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
    kda_module.causal_conv1d = _mock_causal_conv1d  # type: ignore[assignment]
    kda_module.chunk_kda = _mock_kda_core  # type: ignore[assignment]
    kda_module.fused_recurrent_kda = _mock_kda_core  # type: ignore[assignment]
    kda_module.fused_kda_gate = _mock_fused_kda_gate  # type: ignore[assignment]
    kda_module._gate_param = _mock_gate_param  # type: ignore[assignment]
    dist.all_to_all_single = _counting_c10d_a2a  # type: ignore[assignment]
    all_to_all_module._all_to_all_single_autograd = _counting_functional_a2a  # type: ignore[assignment]
    _reset_records()


# ---- fixtures ----


def _build_kda_layer(num_heads: int = _HEADS) -> kda_module.KimiDeltaAttention:
    """构建 fp32（默认 dtype）的最小 KDA 层（head_dim=16 使 mock 核可行）。"""
    torch.manual_seed(7)
    return kda_module.KimiDeltaAttention(
        hidden_size=_HIDDEN,
        num_heads=num_heads,
        head_dim=_HEAD_DIM,
        conv_kernel_size=_CONV_W,
    )


def _global_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """全序列输入与 doc 偏移（各 rank 同种子重生成，免 broadcast）。"""
    s_glob = _S_LOC * _WORLD
    torch.manual_seed(11)
    hidden_global = torch.randn(1, s_glob, _HIDDEN, dtype=torch.float32)
    ids_global = torch.randint(0, 100, (1, s_glob))
    return hidden_global, ids_global


def _sp_ctx(mesh: DeviceMesh, ids_global: torch.Tensor) -> SequenceContext:
    ctx = SequenceContext.from_input_ids((ids_global,), sp_mesh=mesh, device="cpu").split(mesh)
    assert ctx.sp_rank == dist.get_rank()
    return ctx


class _RunSnapshot(TypedDict):
    """单次运行的输出/梯度与 mock 内核输入快照（参考与 SP 两侧同构）。"""

    projected: torch.Tensor
    raw: torch.Tensor
    grads: list[torch.Tensor]
    core_in: tuple[torch.Tensor, ...]
    conv_w: torch.Tensor
    a2a_count: int


def _snapshot(out: AttnOutputs, grads: Sequence[torch.Tensor]) -> _RunSnapshot:
    return {
        "projected": out["projected_output"].detach().clone(),
        "raw": out["raw_output"].detach().clone(),
        "grads": [g.detach().clone() for g in grads],
        "core_in": tuple(t.detach().clone() for t in _CORE_INPUTS[0]),
        "conv_w": _CONV_WEIGHTS[0].detach().clone(),
        "a2a_count": len(_A2A_MECHANISM),
    }


def _global_go() -> torch.Tensor:
    """共享上游梯度（固定种子，参考与 SP 两侧、各 rank 用同一全局量的对应切片）。"""
    torch.manual_seed(13)
    return torch.randn(1, _S_LOC * _WORLD, _HIDDEN, dtype=torch.float32)


def _run_sp(
    module: kda_module.KimiDeltaAttention, hidden_local: torch.Tensor, ctx: SequenceContext, go_global: torch.Tensor
) -> _RunSnapshot:
    """复制头 SP 前向 + 对 hidden 与全部参数的梯度（一次 autograd.grad，[hidden, *params] 序）。"""
    _reset_records()
    go = go_global[:, dist.get_rank() * _S_LOC : (dist.get_rank() + 1) * _S_LOC, :]
    hidden = hidden_local.detach().clone().requires_grad_(True)
    out = module.forward_for_sp(hidden, ctx)
    grads = torch.autograd.grad((out["projected_output"] * go).sum(), [hidden, *module.parameters()])
    return _snapshot(out, grads)


def _run_reference(
    module: kda_module.KimiDeltaAttention,
    hidden_global: torch.Tensor,
    ids_global: torch.Tensor,
    go_global: torch.Tensor,
) -> _RunSnapshot:
    """非 SP 全序列参考（同一模块种子重建、无 SP mesh、同一 mock 内核）。"""
    _reset_records()
    ctx = SequenceContext.from_input_ids((ids_global,), sp_mesh=None, device="cpu")
    s_glob = _S_LOC * _WORLD
    assert ctx.cu_seq_lens_q.tolist() == [0, s_glob]
    hidden = hidden_global.detach().clone().requires_grad_(True)
    out = module.forward(hidden, ctx)
    grads = torch.autograd.grad((out["projected_output"] * go_global).sum(), [hidden, *module.parameters()])
    return _snapshot(out, grads)


# ---- 用例体（spawn worker 内执行） ----


def _case_matches_reference(rank: int, world: int) -> None:
    assert world == _WORLD and _HEADS * _REPLICA == world
    mesh = init_device_mesh("cpu", (world,))
    module = _build_kda_layer()
    module_ref = _build_kda_layer()  # 同种子 → 参数逐位一致
    hidden_global, ids_global = _global_inputs()
    hidden_local = hidden_global[:, rank * _S_LOC : (rank + 1) * _S_LOC, :].contiguous()
    ctx = _sp_ctx(mesh, ids_global)

    go_global = _global_go()
    ref = _run_reference(module_ref, hidden_global, ids_global, go_global)
    sp = _run_sp(module, hidden_local, ctx, go_global)

    # 非退化哨兵。
    assert torch.isfinite(sp["projected"]).all() and sp["projected"].abs().sum() > 0
    # 每次前向恰 6 次 a2a（q/k/v/g/beta/out，与头切分路径相同）；参考无 a2a。
    assert sp["a2a_count"] == 6, sp["a2a_count"]
    assert ref["a2a_count"] == 0

    # 输出：本 rank 的局部切片与参考对应 token 段逐位一致（纯置换 + 确定性 mock）。
    assert torch.equal(sp["projected"], ref["projected"][:, rank * _S_LOC : (rank + 1) * _S_LOC, :])
    assert torch.equal(sp["raw"], ref["raw"][:, rank * _S_LOC : (rank + 1) * _S_LOC, :])

    # 复制/a2a 管线定点检查：rank r 的 core 输入 = 参考对应头（r // replica）的全序列值。
    # （SP 侧 (1, L, 1, D)，参考切片 (1, L, D)，squeeze 后逐位比。）
    head = rank // _REPLICA
    assert len(sp["core_in"]) == 4
    for i, name in enumerate(("q", "k", "v", "g")):
        assert torch.equal(sp["core_in"][i][0, :, 0, :], ref["core_in"][i][0, :, head, :]), (
            f"core 输入 {name} 与参考头切片不一致"
        )
    # mock conv 收到的 weight = 参考全量 weight 的对应头块。
    d = _HEAD_DIM
    assert torch.equal(sp["conv_w"], ref["conv_w"][head * d : (head + 1) * d, :])

    # 梯度：hidden 切片 + 跨 rank 求和的参数梯度（加法结合序与参考不同 → assert_close）。
    torch.testing.assert_close(
        sp["grads"][0], ref["grads"][0][:, rank * _S_LOC : (rank + 1) * _S_LOC, :], rtol=1e-5, atol=1e-6
    )
    for i, g in enumerate(sp["grads"][1:]):
        parts = [torch.empty_like(g) for _ in range(world)]
        dist.all_gather(parts, g)
        total = torch.stack(parts).sum(dim=0)
        torch.testing.assert_close(total, ref["grads"][i + 1], rtol=1e-5, atol=1e-6, msg=f"参数梯度 #{i}")


def _case_error_on_indivisible(rank: int, world: int) -> None:
    """num_heads=3 与 world=4 互不整除 → ValueError（两头都除不尽）。"""
    del rank
    mesh = init_device_mesh("cpu", (world,))
    module = _build_kda_layer(num_heads=3)
    hidden_global, ids_global = _global_inputs()
    r = dist.get_rank()
    hidden_local = hidden_global[:, r * _S_LOC : (r + 1) * _S_LOC, :].contiguous()
    ctx = _sp_ctx(mesh, ids_global)
    with pytest.raises(ValueError, match="replicated-head"):
        module.forward_for_sp(hidden_local, ctx)


# ---- spawn harness（与 test_glm53_kda_sp_defer.py 同一约定） ----


def _worker(rank: int, world: int, store_path: str, case: str) -> None:
    dist.init_process_group("gloo", store=dist.FileStore(store_path, world), rank=rank, world_size=world)
    _install_worker_stubs()
    try:
        if case == "matches_reference":
            _case_matches_reference(rank, world)
        elif case == "error_on_indivisible":
            _case_error_on_indivisible(rank, world)
        else:
            raise AssertionError(f"unknown case {case}")
    finally:
        dist.destroy_process_group()


def _run_spawned(case: str, tmp_path: Path) -> None:
    # NPU 盒上 transfer_to_npu 会把 gloo init 重映射到 hccl；与异步 communicator 拆除竞争，
    # 故前后各留缓冲（与 tests/model/test_glm53_kda_sp_defer.py 同一约定）。
    time.sleep(2.0)
    mp.spawn(_worker, args=(_WORLD, str(tmp_path / "gloo_store"), case), nprocs=_WORLD, join=True)
    time.sleep(2.0)


class TestKdaSpReplicatedHead:
    """复制头 Ulysses SP（sp_size > num_heads）与非 SP 参考的等价（4-rank gloo spawn）。"""

    def test_matches_non_sp_reference(self, tmp_path: Path) -> None:
        _run_spawned("matches_reference", tmp_path)

    def test_error_on_indivisible(self, tmp_path: Path) -> None:
        _run_spawned("error_on_indivisible", tmp_path)
