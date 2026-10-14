# Copyright © 2026 Huawei Technologies Co., Ltd.
"""Ulysses a2a 原语（``xtuner/v1/ops/comm/ulysses_dispatch.py``）与 a2a 异步门控分发
（``xtuner/v1/ops/comm/all_to_all.py::ulysses_all_to_all``）的契约测试。

全部在纯 CPU / gloo 上运行（CI 无 NPU）；被测契约与 2-proc hccl 真机冒烟验证过的结论一致：

Work-handle API（``ulysses_dispatch``）：

- ``ulysses_scatter_heads_blocking``：``[1, A, S/sp, ...] -> [1, A/sp, S, ...]``，输出元素
  ``(h, s)`` 等于全局张量 ``G``（``[1, A, S, ...]``）的 ``G[0, rank*(A/sp)+h, s]`` —— rank r
  的输出恰是 ``G`` 的头切片 ``[r*A/sp, (r+1)*A/sp)`` over 全局序列；``world_size == 1`` 时
  原样返回输入对象。
- ``ulysses_scatter_seq_blocking``：逆运动 ``[1, A/sp, S, ...] -> [1, A, S/sp, ...]``，输出
  ``out[0, h, s] == G2[0, h, rank*(S/sp)+s]``；dim1∘dim2 复合 == 恒等（纯置换的自逆性）。
- ``issue_ulysses_dim1/dim2`` + ``finish_ulysses_dim1/dim2``：拆分使用与 blocking 逐位一致
  （含两连发 issue 后逐个 finish）；issue 返回 ``(raw_recv, work, out_shape)``，recv 形状
  ``[world, A/sp, S_loc, ...]``（send 布局），``out_shape == (1, A/sp, S, ...)``，work 是
  c10d ``Work`` 句柄（同步后端可为 ``None``）；finish 先 fence 再做 permute(1,0)+reshape
  胶水（两方向共用同一段胶水代码）。
- 延迟 fence（work 契约）：issue 不等通信，fence 推迟到消费点 ``_wait_a2a`` —— 只轮询
  ``is_completed()``，仅当返回 False 才 ``wait()``；finish 与 backward（反向 a2a）都走这段
  接线。真 async 用例在 gloo 上真跑 ``all_to_all_single(async_op=True)``：issue 把 ready
  事件 record 到当前流、comm 流 ``wait_event`` 接住，wrapper work 计数并委托真实句柄，
  fence 后数据逐位等于阻塞参考，且 ``_PENDING_A2A``（outstanding send/recv 缓冲注册表）
  在 fence 后被剪枝为空。
- backward：纯置换的伴随 = 同一 a2a 作用于输出梯度 + 逆 reshape；blocking 与 deferred 两种
  前向形态的 backward 均与手写全伴随逐位一致；backward 内部的反向 a2a 同样走
  issue -> ``_wait_a2a``（每个已发 work 都被 ``is_completed()`` 轮询）；dim2 的伴随 ==
  dim1 movement 作用于输出梯度。
- 值契约：dim1 与 ``all_to_all.py`` 的 ``ulysses_all_to_all(x, scatter_dim=1, gather_dim=2,
  mesh=...)``（functional collectives，gloo 可用）逐位一致。

异步门控分发（``all_to_all.ulysses_all_to_all``）：

- 路由（``TestDispatchRouting``）：门控 env ``XTUNER_KDA_SP_A2A_ASYNC`` 在**每次调用时**读取。
  未开启（unset / "0" / "" / "junk"）或调用形态不匹配（batch > 1、scatter_dim != 1、
  gather_dim != 2）一律转发同步 functional-collectives 实现；env == "1" 且 batch-1 + (1, 2)
  时改调 ``ulysses_scatter_heads_blocking(input, mesh.get_group())``。用例以 monkeypatch
  的记录器断言「走了哪条路」，不触达真实通信（``_MeshStub`` 的 ``size() == 1`` 让同步路径的
  movedim/tensor_split/cat 退化为输入的连续副本）。
- 值契约 world==1（``TestDispatchValueWorldOne``）：主进程内 1-rank gloo 组，env 关（同步
  functional collectives 路径）与 env 开（async 的 world==1 早退）都逐位返回输入值。
- 值契约 2-rank（``TestDispatchValueTwoRank``）：沿用 spawn harness，各 rank 以同一 seed
  构造同一全局张量 ``G`` 后按序列切片作本地输入，断言 dispatch(env on) 与 dispatch(env off)
  逐位一致，且都等于全局切片语义 ``out[0, h, s] == G[0, rank*(A/sp)+h, s]``（3-D beta 形状
  同断言），并核实 env 开的 batch-1 (1, 2) 调用确实经由真实的 ``ulysses_scatter_heads_blocking``。
- 接线（``TestKdaImportWiring``）：``kda.py`` import 的 ``ulysses_all_to_all`` 来自
  ``all_to_all``；本 box 无 ``fla``，kda **模块本身**可 import（构造 KDA 模块才抛
  ImportError，用例不构造）。

测试环境约束：``ulysses_dispatch`` 的流/事件是 ``torch.cuda.*``（NPU 经 transfer_to_npu
映射），纯 CPU 进程不可用。默认用例把 ``_a2a_issue`` 替换为同步 ``dist.all_to_all_single``
stub（返回 ``_CpuWork`` dummy），流/事件/``torch.cuda.stream`` 替换为记录调用链的 dummy，
从而在 2-rank gloo（FileStore 初始化，无端口竞争）多进程上验证数学与胶水正确性；真 async
用例只替换 comm-stream 部分、保留真实 ``_a2a_issue``（gloo 的 ``async_op=True`` 返回真实
``Work``，由 wrapper 计数委托、断言 fence 后数据）。world==1 分支在主 pytest 进程内以
1-rank gloo 组验证。所有 rank 用同一 seed 构造同一全局张量后按 rank 切片作本地输入，切片
语义参考独立于任何 a2a 机制，能捕获跨 rank 交错次序错误。

TestUlyssesDim1
    test_blocking_matches_global_semantics_and_sync_helper: 头散列 blocking 与全局切片语义
        参考、``ulysses_all_to_all`` 逐位一致；3-D（beta 形状）同断言；头轴不整除 world 时
        issue 抛 RuntimeError。
    test_issue_finish_split_and_double_issue: issue/finish 拆分 == blocking；out_shape/recv
        形状契约；两连发 issue 后逐个 finish 与单发一致；issue 返回 work、finish 轮询
        ``is_completed()``（同步 stub 恒已完成 -> 不 ``wait()``）。
    test_real_async_a2a_deferred_fence: 真 async a2a（gloo ``async_op=True``）：ready 事件
        从当前流接到 comm 流，finish fence 后数据 == 阻塞参考，work 被轮询、
        ``_PENDING_A2A`` fence 后剪枝为空。
TestUlyssesDim2
    test_seq_scatter_matches_global_semantics_and_roundtrip: 序列散列与全局切片语义参考逐位
        一致；dim1∘dim2 == 恒等；序列轴不整除 world 时 issue 抛 RuntimeError。
TestUlyssesBackward
    test_dim1_backward_matches_hand_adjoint: blocking 与 deferred 的 backward 均等于手写全
        伴随（reshape -> permute -> a2a -> reshape）；反向 a2a 的 work 被 fence 轮询。
    test_dim2_backward_matches_hand_adjoint: dim2 backward 等于手写全伴随，且等于 dim1
        movement 作用于输出梯度（互逆结构的交叉验证）。
TestUlyssesWorldOne
    test_world_one_returns_input_unchanged: blocking 原样返回输入对象；issue/finish 数值
        恒等、work 为 ``None`` 且 out_shape == 输入形状。
TestUlyssesFinishGlue
    test_finish_glue_layout_contract: finish 胶水 ``out[h, c*S_loc+t] == recv[c, h, t]`` 的
        逐元素索引断言（4-D 与 3-D），输出连续。
    test_finish_waits_on_issue_work_and_rejects_bad_shape: finish 只在 ``is_completed()``
        为 False 时 ``wait()``（completed=True -> 0 次 wait；False -> 恰 1 次）；out_shape
        与 recv 元素数不符时抛 RuntimeError。
"""

import os
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from xtuner.v1.ops.comm import all_to_all, ulysses_dispatch
from xtuner.v1.ops.comm.all_to_all import ulysses_all_to_all


# 4-D 用例的全局尺寸：heads / 全局 seq / head dim（2-rank 下 a_loc=4, s_loc=8）
_HEADS = 8
_SEQ = 16
_HDIM = 6

# 被测门控 env 的文档化名称（按文档契约断言，而非引用实现内的常量）。
_ENV = "XTUNER_KDA_SP_A2A_ASYNC"

# 真实（未被 stub 的）``_a2a_issue``：真 async 用例在 worker 内包一层计数 wrapper 复用它。
_REAL_A2A_ISSUE = ulysses_dispatch._a2a_issue

_Call = tuple[str, object]


class _CpuEvent:
    """``torch.cuda.Event`` 的 CPU dummy：只记录是否被 record（真 async 用例用）。"""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.recorded = False

    def record(self, stream: object | None = None) -> None:
        self.recorded = True
        _recorded_events.append(self)

    def wait(self, stream: object | None = None) -> None:
        return None

    def synchronize(self) -> None:
        return None


class _CpuStream:
    """``torch.cuda.Stream`` 的 CPU dummy：把 wait_event 的实参记入 ``_waited_events``。"""

    def wait_event(self, event: object) -> None:
        _waited_events.append(event)

    def record_event(self, event: object) -> None:
        return None


class _StreamCtx:
    """``torch.cuda.stream`` 的 CPU dummy：进入时原样返回 dummy 流（无设备可切换）。"""

    def __init__(self, stream: object) -> None:
        self.stream = stream

    def __enter__(self) -> object:
        return self.stream

    def __exit__(self, *args: object) -> None:
        return None


class _CpuWork:
    """``dist.Work`` 的 CPU dummy：``is_completed()`` 可配置，``wait()`` 只计数。

    Attributes:
        completed (bool): ``is_completed()`` 的返回值。
        is_completed_calls (int): ``is_completed()`` 被调用次数（fence 轮询接线断言）。
        wait_calls (int): ``wait()`` 被调用次数（仅允许在 not completed 时发生）。
    """

    def __init__(self, completed: bool = True) -> None:
        self.completed = completed
        self.is_completed_calls = 0
        self.wait_calls = 0

    def is_completed(self) -> bool:
        self.is_completed_calls += 1
        return self.completed

    def wait(self) -> None:
        self.wait_calls += 1


class _CpuWrappedWork(_CpuWork):
    """包裹真实 ``dist.Work`` 的计数 dummy：``is_completed`` / ``wait`` 委托真实句柄。"""

    def __init__(self, real: dist.Work) -> None:
        super().__init__(completed=False)
        self._real = real

    def is_completed(self) -> bool:
        self.is_completed_calls += 1
        self.completed = self._real.is_completed()
        return self.completed

    def wait(self) -> None:
        self.wait_calls += 1
        self._real.wait()


class _MeshStub:
    """路由用例的最小 mesh 替身：``get_group()`` 返回固定哨兵、``size() == 1``，
    不触达真实进程组，并让同步路径的 movedim/tensor_split/cat 退化为输入的连续副本。"""

    def __init__(self) -> None:
        self.group: object = object()

    def get_group(self) -> object:
        return self.group

    def size(self) -> int:
        return 1


class _RouteResult(NamedTuple):
    """一次路由探测的结果：分发函数返回值、调用记录、两个哨兵与所用 mesh 替身。"""

    out: torch.Tensor
    calls: list[_Call]
    sync_sentinel: torch.Tensor
    async_sentinel: torch.Tensor
    mesh: _MeshStub


_waited_events: list[object] = []
_recorded_events: list[_CpuEvent] = []
_issued_works: list[_CpuWork] = []
# 同步 stub 的 work 完成态（用例内可改为 False 以断言 ``wait()`` 分支）。
_WORK_COMPLETED: list[bool] = [True]


def _sync_a2a_issue(send: torch.Tensor, group: dist.ProcessGroup) -> tuple[torch.Tensor, _CpuWork]:
    """``ulysses_dispatch._a2a_issue`` 的同步 stub：gloo 上直接 all_to_all_single。"""
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    work = _CpuWork(completed=_WORK_COMPLETED[0])
    _issued_works.append(work)
    return recv, work


def _real_async_a2a_issue(send: torch.Tensor, group: dist.ProcessGroup) -> tuple[torch.Tensor, _CpuWrappedWork]:
    """真 async stub：调用真实 ``_a2a_issue``（gloo ``async_op=True``），work 包一层计数。"""
    recv, real_work = _REAL_A2A_ISSUE(send, group)
    assert real_work is not None, "gloo async_op=True 必须返回 Work 句柄"
    work = _CpuWrappedWork(real_work)
    _issued_works.append(work)
    return recv, work


def _reset_stub_state() -> None:
    """清空 stub 记录（worker 进程每用例重入，主进程用例间需还原）。"""
    _waited_events.clear()
    _recorded_events.clear()
    _issued_works.clear()
    _WORK_COMPLETED[0] = True
    ulysses_dispatch._PENDING_A2A.clear()


def _install_cpu_stubs(
    monkeypatch: pytest.MonkeyPatch, a2a_issue: Callable[[torch.Tensor, dist.ProcessGroup], object] = _sync_a2a_issue
) -> None:
    """主 pytest 进程内用 monkeypatch 安装 CPU stub（测试结束自动还原）。"""
    _reset_stub_state()
    monkeypatch.setattr(torch.cuda, "Event", _CpuEvent)
    monkeypatch.setattr(torch.cuda, "Stream", _CpuStream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args, **kwargs: _CpuStream())
    monkeypatch.setattr(torch.cuda, "stream", _StreamCtx)
    monkeypatch.setattr(ulysses_dispatch, "_a2a_issue", a2a_issue)


def _install_worker_cpu_stubs(
    a2a_issue: Callable[[torch.Tensor, dist.ProcessGroup], object] = _sync_a2a_issue,
) -> None:
    """spawn 子进程内直接改写（子进程随用例退出，无需还原）。"""
    _reset_stub_state()
    torch.cuda.Event = _CpuEvent  # type: ignore[assignment]
    torch.cuda.Stream = _CpuStream  # type: ignore[assignment]
    torch.cuda.current_stream = lambda *args, **kwargs: _CpuStream()  # type: ignore[assignment]
    torch.cuda.stream = _StreamCtx  # type: ignore[assignment]
    ulysses_dispatch._a2a_issue = a2a_issue  # type: ignore[assignment]


def _mesh_and_group(world: int) -> tuple[DeviceMesh, dist.ProcessGroup]:
    mesh = init_device_mesh("cpu", (world,))
    return mesh, mesh.get_group()


def _run_spawned(case: str, world: int, tmp_path: Path) -> None:
    store_path = str(tmp_path / "gloo_store")
    mp.spawn(_worker, args=(world, store_path, case), nprocs=world, join=True)


def _worker(rank: int, world: int, store_path: str, case: str) -> None:
    dist.init_process_group("gloo", store=dist.FileStore(store_path, world), rank=rank, world_size=world)
    install = _REAL_ASYNC_CASES.get(case, _install_worker_cpu_stubs)
    install()
    try:
        _CASES[case](rank, world)
    finally:
        dist.destroy_process_group()


def _case_dim1_forward(rank: int, world: int) -> None:
    """blocking 头散列：全局切片语义 + ulysses_all_to_all 交叉验证 + 3-D + 整除断言。"""
    _, group = _mesh_and_group(world)
    a_loc, s_loc = _HEADS // world, _SEQ // world

    torch.manual_seed(7)
    g4 = torch.randn(1, _HEADS, _SEQ, _HDIM)
    local = g4[:, :, rank * s_loc : (rank + 1) * s_loc, :].contiguous()
    got = ulysses_dispatch.ulysses_scatter_heads_blocking(local, group)
    expected = g4[:, rank * a_loc : (rank + 1) * a_loc, :, :]
    assert got.shape == expected.shape, (got.shape, expected.shape)
    assert torch.equal(expected, got), "dim1 blocking 违背全局切片语义"
    mesh, _ = _mesh_and_group(world)
    helper = ulysses_all_to_all(local, scatter_dim=1, gather_dim=2, mesh=mesh)
    assert torch.equal(helper, got), "dim1 blocking 与 ulysses_all_to_all 不一致"

    g3 = torch.randn(1, _HEADS, _SEQ)
    local3 = g3[:, :, rank * s_loc : (rank + 1) * s_loc].contiguous()
    got3 = ulysses_dispatch.ulysses_scatter_heads_blocking(local3, group)
    assert torch.equal(g3[:, rank * a_loc : (rank + 1) * a_loc], got3), "3-D dim1 mismatch"

    bad = torch.randn(1, _HEADS + 1, s_loc, _HDIM)
    with pytest.raises(RuntimeError):
        ulysses_dispatch.issue_ulysses_dim1(bad, group)


def _case_dim1_deferred(rank: int, world: int) -> None:
    """issue/finish 拆分：== blocking、形状契约、双 issue、work fence 接线。"""
    _, group = _mesh_and_group(world)
    a_loc, s_loc = _HEADS // world, _SEQ // world

    torch.manual_seed(11)
    local = torch.randn(1, _HEADS, s_loc, _HDIM)
    blocking = ulysses_dispatch.ulysses_scatter_heads_blocking(local, group)

    recv1, work1, out_shape1 = ulysses_dispatch.issue_ulysses_dim1(local, group)
    recv2, work2, out_shape2 = ulysses_dispatch.issue_ulysses_dim1(local, group)
    assert out_shape1 == torch.Size((1, a_loc, _SEQ, _HDIM)), out_shape1
    assert out_shape2 == out_shape1
    assert tuple(recv1.shape) == (world, a_loc, s_loc, _HDIM), recv1.shape
    assert work1 is not None and work2 is not None, "issue 未返回 work 句柄"

    fin1 = ulysses_dispatch.finish_ulysses_dim1(recv1, work1, out_shape1)
    fin2 = ulysses_dispatch.finish_ulysses_dim1(recv2, work2, out_shape2)
    assert torch.equal(blocking, fin1), "deferred(1st) != blocking"
    assert torch.equal(blocking, fin2), "deferred(2nd) != blocking"
    assert fin1.is_contiguous()
    # 延迟 fence 接线：finish 轮询 is_completed()；同步 stub 恒已完成，wait 只能是 0 次。
    assert work1.is_completed_calls >= 1 and work2.is_completed_calls >= 1, "finish 未轮询 work"
    assert work1.wait_calls == 0 and work2.wait_calls == 0, "已完成的 work 不应被 wait"


def _case_dim1_real_async(rank: int, world: int) -> None:
    """真 async a2a（gloo ``async_op=True``）：fence 接线 + fence 后数据 == 阻塞参考。"""
    _, group = _mesh_and_group(world)
    a_loc, s_loc = _HEADS // world, _SEQ // world

    torch.manual_seed(23)
    g4 = torch.randn(1, _HEADS, _SEQ, _HDIM)
    local = g4[:, :, rank * s_loc : (rank + 1) * s_loc, :].contiguous()
    # 手写阻塞参考：经典 c10d 同步 a2a + 与 finish 胶水同款的 permute+reshape。
    send = local.reshape(world, _HEADS // world, s_loc, _HDIM).contiguous()
    ref_recv = torch.empty_like(send)
    dist.all_to_all_single(ref_recv, send, group=group)
    expected = ref_recv.permute(1, 0, 2, 3).reshape(1, a_loc, _SEQ, _HDIM)

    recv, work, out_shape = ulysses_dispatch.issue_ulysses_dim1(local, group)
    assert out_shape == torch.Size((1, a_loc, _SEQ, _HDIM)), out_shape
    assert work is not None, "真 async a2a 必须返回 work 句柄"
    # issue 的流接线：ready 事件 record 到当前流，comm 流 wait_event 接住同一个事件。
    assert _waited_events and _recorded_events, "issue 未接 ready 事件"
    assert _waited_events[0] in _recorded_events, "comm 流 wait 的不是 issue record 的事件"

    fin = ulysses_dispatch.finish_ulysses_dim1(recv, work, out_shape)
    assert torch.equal(expected, fin), "真 async a2a fence 后数据 != 阻塞参考"
    assert work.is_completed_calls >= 1, "finish 未轮询 work.is_completed()"
    assert work.wait_calls <= 1, "wait() 至多在 not completed 时发生一次"
    # outstanding send/recv 缓冲注册表在 fence 后应被剪枝，不再滞留引用。
    assert not ulysses_dispatch._PENDING_A2A, "fence 后 _PENDING_A2A 未剪枝"


def _case_dim2_forward(rank: int, world: int) -> None:
    """blocking 序列散列：全局切片语义 + dim1∘dim2 恒等 + 整除断言。"""
    _, group = _mesh_and_group(world)
    a_loc, s_loc = _HEADS // world, _SEQ // world

    torch.manual_seed(13)
    g2 = torch.randn(1, _HEADS, _SEQ, _HDIM)
    y_in = g2[:, rank * a_loc : (rank + 1) * a_loc, :, :].contiguous()
    got = ulysses_dispatch.ulysses_scatter_seq_blocking(y_in, group)
    expected = g2[:, :, rank * s_loc : (rank + 1) * s_loc, :]
    assert got.shape == expected.shape, (got.shape, expected.shape)
    assert torch.equal(expected, got), "dim2 blocking 违背全局切片语义"

    dim1_in = torch.randn(1, _HEADS, s_loc, _HDIM)
    roundtrip = ulysses_dispatch.ulysses_scatter_seq_blocking(
        ulysses_dispatch.ulysses_scatter_heads_blocking(dim1_in, group), group
    )
    assert torch.equal(roundtrip, dim1_in), "dim1∘dim2 复合 != 恒等"

    bad = torch.randn(1, a_loc, _SEQ + 1, _HDIM)
    with pytest.raises(RuntimeError):
        ulysses_dispatch.issue_ulysses_dim2(bad, group)


def _case_backward_dim1(rank: int, world: int) -> None:
    """dim1 backward == 手写全伴随（blocking 与 deferred 两种前向形态）。"""
    _, group = _mesh_and_group(world)
    a_loc, s_loc = _HEADS // world, _SEQ // world

    torch.manual_seed(17)
    local = torch.randn(1, _HEADS, s_loc, _HDIM)
    go = torch.randn(1, a_loc, _SEQ, _HDIM)
    # 手写全伴随：[1, a_loc, S, D] 的梯度 -> permute 回 send 布局 -> a2a -> 逆 reshape
    gr = go.reshape(a_loc, world, s_loc, _HDIM).permute(1, 0, 2, 3).contiguous()
    gx = torch.empty_like(gr)
    dist.all_to_all_single(gx, gr, group=group)
    expected = gx.reshape(1, world * a_loc, s_loc, _HDIM)

    x = local.detach().clone().requires_grad_(True)
    y = ulysses_dispatch.ulysses_scatter_heads_blocking(x, group)
    (grad_blocking,) = torch.autograd.grad((y * go).sum(), (x,))
    assert torch.equal(expected, grad_blocking), "blocking backward != 手写伴随"

    xd = local.detach().clone().requires_grad_(True)
    recv, work, out_shape = ulysses_dispatch.issue_ulysses_dim1(xd, group)
    y_deferred = ulysses_dispatch.finish_ulysses_dim1(recv, work, out_shape)
    (grad_deferred,) = torch.autograd.grad((y_deferred * go).sum(), (xd,))
    assert torch.equal(expected, grad_deferred), "deferred backward != 手写伴随"
    # backward 内部的反向 a2a 也走 issue -> _wait_a2a：每个已发 work 都被 is_completed 轮询。
    assert all(w.is_completed_calls >= 1 for w in _issued_works), "反向 a2a work 未被 fence 轮询"


def _case_backward_dim2(rank: int, world: int) -> None:
    """dim2 backward == 手写全伴随 == dim1 movement 作用于输出梯度。"""
    _, group = _mesh_and_group(world)
    a_loc, s_loc = _HEADS // world, _SEQ // world

    torch.manual_seed(19)
    y_in = torch.randn(1, a_loc, _SEQ, _HDIM)
    go = torch.randn(1, _HEADS, s_loc, _HDIM)
    # 手写全伴随：[1, A, S/sp, D] 的梯度 -> permute 回 send 布局 -> a2a -> 逆 permute+reshape
    d_raw = go.reshape(1, world, a_loc, s_loc, _HDIM).permute(1, 0, 2, 3, 4).contiguous()
    gx = torch.empty_like(d_raw)
    dist.all_to_all_single(gx, d_raw, group=group)
    expected = gx.permute(1, 2, 0, 3, 4).reshape(1, a_loc, _SEQ, _HDIM)

    y = y_in.detach().clone().requires_grad_(True)
    out = ulysses_dispatch.ulysses_scatter_seq_blocking(y, group)
    (grad,) = torch.autograd.grad((out * go).sum(), (y,))
    assert torch.equal(expected, grad), "dim2 backward != 手写伴随"

    cross = ulysses_dispatch.ulysses_scatter_heads_blocking(go, group)
    assert torch.equal(cross, grad), "dim2 backward != dim1(go_grad)"
    assert all(w.is_completed_calls >= 1 for w in _issued_works), "反向 a2a work 未被 fence 轮询"


def _case_env_on_matches_env_off(rank: int, world: int) -> None:
    """dispatch(env on) == dispatch(env off) == 全局切片语义（4-D 与 3-D beta 形状）。"""
    mesh, _ = _mesh_and_group(world)
    a_loc, s_loc = _HEADS // world, _SEQ // world

    torch.manual_seed(7)
    g4 = torch.randn(1, _HEADS, _SEQ, _HDIM)
    g3 = torch.randn(1, _HEADS, _SEQ)
    local4 = g4[:, :, rank * s_loc : (rank + 1) * s_loc, :].contiguous()
    local3 = g3[:, :, rank * s_loc : (rank + 1) * s_loc].contiguous()

    # 核实 env 开的 batch-1 (1, 2) 调用确实经由真实的 ulysses_scatter_heads_blocking。
    # spy 挂在 all_to_all 命名空间上：gate 读的是它 import 时拷贝的那份绑定。
    real_blocking = ulysses_dispatch.ulysses_scatter_heads_blocking
    seen: list[int] = []

    def _spy(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        seen.append(dist.get_world_size(group))
        return real_blocking(x, group)

    all_to_all.ulysses_scatter_heads_blocking = _spy  # type: ignore[assignment]

    os.environ[_ENV] = "1"
    on4 = all_to_all.ulysses_all_to_all(local4, scatter_dim=1, gather_dim=2, mesh=mesh)
    on3 = all_to_all.ulysses_all_to_all(local3, scatter_dim=1, gather_dim=2, mesh=mesh)
    os.environ[_ENV] = "0"
    off4 = all_to_all.ulysses_all_to_all(local4, scatter_dim=1, gather_dim=2, mesh=mesh)
    off3 = all_to_all.ulysses_all_to_all(local3, scatter_dim=1, gather_dim=2, mesh=mesh)
    os.environ.pop(_ENV, None)
    # 非匹配形态（batch > 1 / 其他维度）env 开时的回退只做路由级验证（TestDispatchRouting）：
    # batch-2 的值级回退检查不可行——同步助手 ulysses_all_to_all(x, scatter_dim=1) 在 B>1 时
    # movedim 后的 send buffer 非连续，c10d functional op 直接抛 "Expected input.is_contiguous()"
    # （预先存在的助手限制，与分发逻辑无关；KDA 调用点恒为 batch-1，不受影响）。

    expected4 = g4[:, rank * a_loc : (rank + 1) * a_loc, :, :]
    expected3 = g3[:, rank * a_loc : (rank + 1) * a_loc, :]
    assert seen == [world, world], "env 开的 batch-1 (1,2) 调用未走 ulysses_scatter_heads_blocking"
    assert on4.shape == expected4.shape and off4.shape == expected4.shape, (on4.shape, off4.shape)
    assert on3.shape == expected3.shape and off3.shape == expected3.shape, (on3.shape, off3.shape)
    assert torch.equal(on4, off4), "dispatch(env on) 与 dispatch(env off) 不逐位一致 (4-D)"
    assert torch.equal(on3, off3), "dispatch(env on) 与 dispatch(env off) 不逐位一致 (3-D)"
    assert torch.equal(on4, expected4), "dispatch(env on) 违背全局切片语义 (4-D)"
    assert torch.equal(off4, expected4), "dispatch(env off) 违背全局切片语义 (4-D)"
    assert torch.equal(on3, expected3), "dispatch(env on) 违背全局切片语义 (3-D)"
    assert torch.equal(off3, expected3), "dispatch(env off) 违背全局切片语义 (3-D)"
    assert on4.is_contiguous() and on3.is_contiguous(), "async 路径输出应连续"


_CASES: dict[str, Callable[[int, int], None]] = {
    "dim1_forward": _case_dim1_forward,
    "dim1_deferred": _case_dim1_deferred,
    "dim1_real_async": _case_dim1_real_async,
    "dim2_forward": _case_dim2_forward,
    "backward_dim1": _case_backward_dim1,
    "backward_dim2": _case_backward_dim2,
    "env_on_matches_env_off": _case_env_on_matches_env_off,
}

# 真 async 用例保留真实 ``_a2a_issue``，只替换 comm-stream 部分（事件/流/stream 上下文）。
_REAL_ASYNC_CASES: dict[str, Callable[[], None]] = {
    "dim1_real_async": lambda: _install_worker_cpu_stubs(_real_async_a2a_issue),
}


def _patch_route_recorders(monkeypatch: pytest.MonkeyPatch) -> tuple[list[_Call], torch.Tensor, torch.Tensor]:
    """把两个路由目标替换为记录器。

    同步路径 inline 于 ``all_to_all.ulysses_all_to_all``（``_all_to_all_single_autograd`` +
    movedim/tensor_split/cat），故同步记录器 patch 在 ``all_to_all._all_to_all_single_autograd``
    上、原样返回输入（``_MeshStub.size() == 1`` 让后续 movedim/tensor_split/cat 退化为输入
    的连续副本）；async 目标 ``ulysses_scatter_heads_blocking`` 是 ``all_to_all`` 在 import 时
    从 ``ulysses_dispatch`` 拷进命名空间的名字，patch 在 ``all_to_all`` 上才能截到 gate 的读取。

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 的 monkeypatch 夹具。

    Returns:
        tuple[list[_Call], torch.Tensor, torch.Tensor]: 调用记录列表、同步哨兵、async 哨兵。
    """
    calls: list[_Call] = []
    sync_sentinel = torch.tensor([1.0])
    async_sentinel = torch.tensor([2.0])

    def fake_sync(
        inp: torch.Tensor,
        *,
        group: object,
        input_split_sizes: object,
        output_split_sizes: object,
    ) -> torch.Tensor:
        calls.append(("sync", inp, input_split_sizes, output_split_sizes))
        return inp

    def fake_async(inp: torch.Tensor, group: object) -> torch.Tensor:
        calls.append(("async", inp, group))
        return async_sentinel

    monkeypatch.setattr(all_to_all, "_all_to_all_single_autograd", fake_sync)
    monkeypatch.setattr(all_to_all, "ulysses_scatter_heads_blocking", fake_async)
    return calls, sync_sentinel, async_sentinel


def _route(
    monkeypatch: pytest.MonkeyPatch,
    env: str | None,
    x: torch.Tensor,
    scatter_dim: int = 1,
    gather_dim: int = 2,
) -> _RouteResult:
    """以给定 env 经 ``all_to_all.ulysses_all_to_all`` 路由一次。

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 的 monkeypatch 夹具。
        env (str | None): 门控 env 的取值；``None`` 表示删除该变量（未设置）。
        x (torch.Tensor): 输入张量。
        scatter_dim (int): scatter 维度。
        gather_dim (int): gather 维度。

    Returns:
        _RouteResult: 输出、调用记录、两个哨兵与所用 mesh 替身。
    """
    calls, sync_sentinel, async_sentinel = _patch_route_recorders(monkeypatch)
    mesh = _MeshStub()
    if env is None:
        monkeypatch.delenv(_ENV, raising=False)
    else:
        monkeypatch.setenv(_ENV, env)
    out = all_to_all.ulysses_all_to_all(x, scatter_dim, gather_dim, mesh)
    return _RouteResult(out, calls, sync_sentinel, async_sentinel, mesh)


class TestUlyssesDim1:
    """头散列（dim1）blocking 与 deferred 拆分契约（2-rank gloo）。"""

    def test_blocking_matches_global_semantics_and_sync_helper(self, tmp_path) -> None:
        _run_spawned("dim1_forward", 2, tmp_path)

    def test_issue_finish_split_and_double_issue(self, tmp_path) -> None:
        _run_spawned("dim1_deferred", 2, tmp_path)

    def test_real_async_a2a_deferred_fence(self, tmp_path) -> None:
        _run_spawned("dim1_real_async", 2, tmp_path)


class TestUlyssesDim2:
    """序列散列（dim2）blocking 契约与自逆复合（2-rank gloo）。"""

    def test_seq_scatter_matches_global_semantics_and_roundtrip(self, tmp_path) -> None:
        _run_spawned("dim2_forward", 2, tmp_path)


class TestUlyssesBackward:
    """两种 movement 的 backward = 同一 a2a 的全伴随（2-rank gloo）。"""

    def test_dim1_backward_matches_hand_adjoint(self, tmp_path) -> None:
        _run_spawned("backward_dim1", 2, tmp_path)

    def test_dim2_backward_matches_hand_adjoint(self, tmp_path) -> None:
        _run_spawned("backward_dim2", 2, tmp_path)


class TestUlyssesWorldOne:
    """world_size == 1 早退分支（主进程内 1-rank gloo 组）。"""

    def test_world_one_returns_input_unchanged(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_cpu_stubs(monkeypatch)
        dist.init_process_group("gloo", store=dist.FileStore(str(tmp_path / "gloo_store"), 1), rank=0, world_size=1)
        try:
            group = dist.group.WORLD
            x = torch.randn(1, 4, 6, 3)
            assert ulysses_dispatch.ulysses_scatter_heads_blocking(x, group) is x
            assert ulysses_dispatch.ulysses_scatter_seq_blocking(x, group) is x

            recv, work, out_shape = ulysses_dispatch.issue_ulysses_dim1(x, group)
            assert out_shape == torch.Size(x.shape), out_shape
            assert recv is x, "world==1 issue 应原样返回输入对象"
            assert work is None, "world==1 issue 不产生 work"
            out = ulysses_dispatch.finish_ulysses_dim1(recv, work, out_shape)
            assert out.shape == x.shape
            assert torch.equal(out, x)

            recv2, work2, out_shape2 = ulysses_dispatch.issue_ulysses_dim2(x, group)
            assert recv2 is x and work2 is None and out_shape2 == torch.Size(x.shape)
            out2 = ulysses_dispatch.finish_ulysses_dim2(recv2, work2, out_shape2)
            assert torch.equal(out2, x)
        finally:
            dist.destroy_process_group()


class TestUlyssesFinishGlue:
    """finish 胶水（permute(1,0)+reshape）与 work fence 分支的单进程契约，无进程组。"""

    def test_finish_glue_layout_contract(self, monkeypatch) -> None:
        _install_cpu_stubs(monkeypatch)
        world, a_loc, s_loc, d = 2, 3, 4, 2
        recv = torch.arange(world * a_loc * s_loc * d, dtype=torch.float32).reshape(world, a_loc, s_loc, d)
        out = ulysses_dispatch.finish_ulysses_dim1(recv, _CpuWork(), torch.Size((1, a_loc, world * s_loc, d)))
        assert out.is_contiguous()
        for c in range(world):
            for h in range(a_loc):
                for t in range(s_loc):
                    assert torch.equal(out[0, h, c * s_loc + t], recv[c, h, t])

        recv3 = torch.arange(world * a_loc * s_loc, dtype=torch.float32).reshape(world, a_loc, s_loc)
        out3 = ulysses_dispatch.finish_ulysses_dim1(recv3, _CpuWork(), torch.Size((1, a_loc, world * s_loc)))
        for c in range(world):
            for h in range(a_loc):
                for t in range(s_loc):
                    assert torch.equal(out3[0, h, c * s_loc + t], recv3[c, h, t])

    def test_finish_waits_on_issue_work_and_rejects_bad_shape(self, monkeypatch) -> None:
        _install_cpu_stubs(monkeypatch)
        recv = torch.randn(2, 3, 4, 2)

        completed = _CpuWork(completed=True)
        out = ulysses_dispatch.finish_ulysses_dim1(recv, completed, torch.Size((1, 3, 8, 2)))
        assert out.shape == (1, 3, 8, 2)
        assert completed.is_completed_calls == 1, "finish 未轮询 work.is_completed()"
        assert completed.wait_calls == 0, "已完成（is_completed()==True）的 work 不应被 wait"

        pending = _CpuWork(completed=False)
        out2 = ulysses_dispatch.finish_ulysses_dim1(recv, pending, torch.Size((1, 3, 8, 2)))
        assert out2.shape == (1, 3, 8, 2)
        assert pending.is_completed_calls == 1
        assert pending.wait_calls == 1, "未完成（is_completed()==False）的 work 必须被 wait"

        with pytest.raises(RuntimeError):
            ulysses_dispatch.finish_ulysses_dim1(recv, _CpuWork(), torch.Size((1, 3, 9, 2)))


class TestDispatchRouting:
    """门控 env + 调用形态 → 同步 / async 路径的路由契约（记录器替身，无真实通信）。"""

    def test_env_unset_routes_to_sync(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """env 未设置 → 同步路径被调，async 目标未被调。"""
        x = torch.randn(1, 4, 6)
        res = _route(monkeypatch, None, x)
        assert any(c[0] == "sync" for c in res.calls), "env 未设置应走同步路径"
        assert not any(c[0] == "async" for c in res.calls), "env 未设置不应走 async 路径"

    @pytest.mark.parametrize("env", ["0", "", "junk", "true"])
    def test_env_non_one_values_route_to_sync(self, monkeypatch: pytest.MonkeyPatch, env: str) -> None:
        """env 存在但不等于 "1" → 同步路径。"""
        x = torch.randn(1, 4, 6)
        res = _route(monkeypatch, env, x)
        assert any(c[0] == "sync" for c in res.calls), f"env={env!r} 应走同步路径"
        assert not any(c[0] == "async" for c in res.calls), f"env={env!r} 不应走 async 路径"

    def test_env_on_matching_call_routes_to_async(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """env == "1" + batch-1 + (1, 2) → async 目标以 (input, mesh.get_group()) 被调。"""
        x = torch.randn(1, 4, 6, 3)
        res = _route(monkeypatch, "1", x)
        assert res.out is res.async_sentinel
        assert res.calls == [("async", x, res.mesh.group)]

    @pytest.mark.parametrize(("scatter_dim", "gather_dim"), [(2, 2), (1, 1), (0, 2), (1, 3)])
    def test_env_on_non_matching_dims_route_to_sync(
        self, monkeypatch: pytest.MonkeyPatch, scatter_dim: int, gather_dim: int
    ) -> None:
        """env == "1" 但 scatter/gather 维度不是 (1, 2) → 同步路径。"""
        x = torch.randn(1, 4, 6, 3)
        res = _route(monkeypatch, "1", x, scatter_dim=scatter_dim, gather_dim=gather_dim)
        assert any(c[0] == "sync" for c in res.calls), f"({scatter_dim},{gather_dim}) 应走同步路径"
        assert not any(c[0] == "async" for c in res.calls), f"({scatter_dim},{gather_dim}) 不应走 async 路径"

    def test_env_on_batch_two_routes_to_sync(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """env == "1" 但 batch > 1 → 同步路径。"""
        x = torch.randn(2, 4, 6)
        res = _route(monkeypatch, "1", x)
        assert any(c[0] == "sync" for c in res.calls), "batch>1 应走同步路径"
        assert not any(c[0] == "async" for c in res.calls), "batch>1 不应走 async 路径"

    def test_env_is_read_at_call_time(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """env 在每次调用时读取：两次调用之间翻转 env 即切换路径，无需重新 import。"""
        x = torch.randn(1, 4, 6)
        first = _route(monkeypatch, "0", x)
        assert not any(c[0] == "async" for c in first.calls), "env=0 应走同步路径"
        second = _route(monkeypatch, "1", x)
        assert second.out is second.async_sentinel
        third = _route(monkeypatch, "0", x)
        assert not any(c[0] == "async" for c in third.calls), "env=0 应走同步路径"


class TestDispatchValueWorldOne:
    """world_size == 1（主进程内 1-rank gloo 组）：env 开/关均逐位返回输入值。"""

    def test_env_off_and_on_return_input_values(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        dist.init_process_group("gloo", store=dist.FileStore(str(tmp_path / "gloo_store"), 1), rank=0, world_size=1)
        try:
            mesh = init_device_mesh("cpu", (1,))
            x4 = torch.randn(1, 4, 6, 3)
            x3 = torch.randn(1, 4, 6)

            monkeypatch.delenv(_ENV, raising=False)
            off4 = all_to_all.ulysses_all_to_all(x4, scatter_dim=1, gather_dim=2, mesh=mesh)
            off3 = all_to_all.ulysses_all_to_all(x3, scatter_dim=1, gather_dim=2, mesh=mesh)
            monkeypatch.setenv(_ENV, "1")
            on4 = all_to_all.ulysses_all_to_all(x4, scatter_dim=1, gather_dim=2, mesh=mesh)
            on3 = all_to_all.ulysses_all_to_all(x3, scatter_dim=1, gather_dim=2, mesh=mesh)
        finally:
            dist.destroy_process_group()

        for x, off, on in ((x4, off4, on4), (x3, off3, on3)):
            assert off.shape == x.shape, (off.shape, x.shape)
            assert on.shape == x.shape, (on.shape, x.shape)
            assert torch.equal(off, x), "env 关（同步路径）未逐位返回输入值"
            assert torch.equal(on, x), "env 开（async world==1 早退）未逐位返回输入值"
        # env 开且 world==1 时 async 早退原样返回输入对象；同步路径返回的是新张量。
        assert on4 is x4 and on3 is x3


class TestDispatchValueTwoRank:
    """2-rank gloo spawn：env 开/关逐位一致，且都等于全局切片语义参考（值 + 布局）。"""

    def test_env_on_matches_env_off_and_global_semantics(self, tmp_path: Path) -> None:
        _run_spawned("env_on_matches_env_off", 2, tmp_path)


class TestKdaImportWiring:
    """``kda.py`` 对 ``all_to_all`` 的 import 接线（不构造 KDA 模块——那需要 ``fla``）。"""

    def test_kda_imports_dispatch_function(self) -> None:
        # import 本身不得抛异常（本 box 无 fla；kda 的模块级 import 不依赖 fla）。
        import xtuner.v1.module.attention.kda as kda_module

        assert kda_module.ulysses_all_to_all.__module__ == "xtuner.v1.ops.comm.all_to_all"
        assert kda_module.ulysses_all_to_all is all_to_all.ulysses_all_to_all
