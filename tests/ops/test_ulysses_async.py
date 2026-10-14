# Copyright © 2026 Huawei Technologies Co., Ltd.
"""异步 Ulysses all-to-all 原语（``xtuner/v1/ops/comm/ulysses_async.py``）的契约测试。

全部在纯 CPU / gloo 上运行（CI 无 NPU）；被测契约与 2-proc hccl 真机冒烟验证过的结论一致：

- ``ulysses_scatter_heads_blocking``：``[1, A, S/sp, ...] -> [1, A/sp, S, ...]``，输出元素
  ``(h, s)`` 等于全局张量 ``G``（``[1, A, S, ...]``）的 ``G[0, rank*(A/sp)+h, s]`` —— rank r
  的输出恰是 ``G`` 的头切片 ``[r*A/sp, (r+1)*A/sp)`` over 全局序列；``world_size == 1`` 时
  原样返回输入对象。
- ``ulysses_scatter_seq_blocking``：逆运动 ``[1, A/sp, S, ...] -> [1, A, S/sp, ...]``，输出
  ``out[0, h, s] == G2[0, h, rank*(S/sp)+s]``；dim1∘dim2 复合 == 恒等（纯置换的自逆性）。
- ``issue_ulysses_dim1/dim2`` + ``finish_ulysses_dim1/dim2``：拆分使用与 blocking 逐位一致
  （含两连发 issue 后逐个 finish）；issue 返回 ``(raw_recv, event, out_shape)``，recv 形状
  ``[world, A/sp, S_loc, ...]``（send 布局），``out_shape == (1, A/sp, S, ...)``；finish 先在
  当前流 wait 事件再做 permute(1,0)+reshape 胶水（两方向共用同一段胶水代码）。
- backward：纯置换的伴随 = 同一 a2a 作用于输出梯度 + 逆 reshape；blocking 与 deferred 两种
  前向形态的 backward 均与手写全伴随逐位一致；dim2 的伴随 == dim1 movement 作用于输出梯度。
- 值契约：dim1 与 ``all_to_all.py`` 的 ``ulysses_all_to_all(x, scatter_dim=1, gather_dim=2,
  mesh=...)``（functional collectives，gloo 可用）逐位一致。

测试环境约束：``ulysses_async`` 的流/事件是 ``torch.cuda.*``（NPU 经 transfer_to_npu 映射），
纯 CPU 进程不可用。harness 把 ``_a2a_issue``（comm stream 部分）替换为同步
``dist.all_to_all_single`` stub、事件/流替换为记录调用链的 dummy（同步语义下 wait 是 no-op，
但事件从 issue 到 finish 的接线仍被断言），从而在 2-rank gloo（FileStore 初始化，无端口
竞争）多进程上验证数学与胶水正确性；world==1 分支在主 pytest 进程内以 1-rank gloo 组验证。
所有 rank 用同一 seed 构造同一全局张量后按 rank 切片作本地输入，切片语义参考独立于任何 a2a
机制，能捕获跨 rank 交错次序错误。

TestUlyssesAsyncDim1
    test_blocking_matches_global_semantics_and_sync_helper: 头散列 blocking 与全局切片语义
        参考、``ulysses_all_to_all`` 逐位一致；3-D（beta 形状）同断言；头轴不整除 world 时
        issue 抛 RuntimeError。
    test_issue_finish_split_and_double_issue: issue/finish 拆分 == blocking；out_shape/recv
        形状契约；两连发 issue 后逐个 finish 与单发一致；事件被 issue 记录、被 finish 等待。
TestUlyssesAsyncDim2
    test_seq_scatter_matches_global_semantics_and_roundtrip: 序列散列与全局切片语义参考逐位
        一致；dim1∘dim2 == 恒等；序列轴不整除 world 时 issue 抛 RuntimeError。
TestUlyssesAsyncBackward
    test_dim1_backward_matches_hand_adjoint: blocking 与 deferred 的 backward 均等于手写全
        伴随（reshape -> permute -> a2a -> reshape）。
    test_dim2_backward_matches_hand_adjoint: dim2 backward 等于手写全伴随，且等于 dim1
        movement 作用于输出梯度（互逆结构的交叉验证）。
TestUlyssesAsyncWorldOne
    test_world_one_returns_input_unchanged: blocking 原样返回输入对象；issue/finish 恒等且
        out_shape == 输入形状。
TestUlyssesAsyncFinishGlue
    test_finish_glue_layout_contract: finish 胶水 ``out[h, c*S_loc+t] == recv[c, h, t]`` 的
        逐元素索引断言（4-D 与 3-D），输出连续。
    test_finish_waits_on_issue_event_and_rejects_bad_shape: finish 在当前流上 wait 的正是
        传入的事件对象；out_shape 与 recv 元素数不符时抛 RuntimeError。
"""

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from xtuner.v1.ops.comm import ulysses_async
from xtuner.v1.ops.comm.all_to_all import ulysses_all_to_all


# 4-D 用例的全局尺寸：heads / 全局 seq / head dim（2-rank 下 a_loc=4, s_loc=8）
_HEADS = 8
_SEQ = 16
_HDIM = 6


class _CpuEvent:
    """``torch.cuda.Event`` 的 CPU dummy：只记录是否被 record。"""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.recorded = False

    def record(self, stream: object | None = None) -> None:
        self.recorded = True

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


_waited_events: list[object] = []


def _sync_a2a_issue(send: torch.Tensor, group: dist.ProcessGroup, finished: object) -> torch.Tensor:
    """``ulysses_async._a2a_issue`` 的同步 stub：gloo 上直接 all_to_all_single。"""
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    finished.record(None)
    return recv


def _install_cpu_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """主 pytest 进程内用 monkeypatch 安装 CPU stub（测试结束自动还原）。"""
    _waited_events.clear()
    monkeypatch.setattr(torch.cuda, "Event", _CpuEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args, **kwargs: _CpuStream())
    monkeypatch.setattr(ulysses_async, "_a2a_issue", _sync_a2a_issue)


def _install_worker_cpu_stubs() -> None:
    """spawn 子进程内直接改写（子进程随用例退出，无需还原）。"""
    _waited_events.clear()
    torch.cuda.Event = _CpuEvent  # type: ignore[assignment]
    torch.cuda.current_stream = lambda *args, **kwargs: _CpuStream()  # type: ignore[assignment]
    ulysses_async._a2a_issue = _sync_a2a_issue  # type: ignore[assignment]


def _mesh_and_group(world: int) -> tuple[DeviceMesh, dist.ProcessGroup]:
    mesh = init_device_mesh("cpu", (world,))
    return mesh, mesh.get_group()


def _run_spawned(case: str, world: int, tmp_path: Path) -> None:
    store_path = str(tmp_path / "gloo_store")
    mp.spawn(_worker, args=(world, store_path, case), nprocs=world, join=True)


def _worker(rank: int, world: int, store_path: str, case: str) -> None:
    dist.init_process_group("gloo", store=dist.FileStore(store_path, world), rank=rank, world_size=world)
    _install_worker_cpu_stubs()
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
    got = ulysses_async.ulysses_scatter_heads_blocking(local, group)
    expected = g4[:, rank * a_loc : (rank + 1) * a_loc, :, :]
    assert got.shape == expected.shape, (got.shape, expected.shape)
    assert torch.equal(expected, got), "dim1 blocking 违背全局切片语义"
    mesh, _ = _mesh_and_group(world)
    helper = ulysses_all_to_all(local, scatter_dim=1, gather_dim=2, mesh=mesh)
    assert torch.equal(helper, got), "dim1 blocking 与 ulysses_all_to_all 不一致"

    g3 = torch.randn(1, _HEADS, _SEQ)
    local3 = g3[:, :, rank * s_loc : (rank + 1) * s_loc].contiguous()
    got3 = ulysses_async.ulysses_scatter_heads_blocking(local3, group)
    assert torch.equal(g3[:, rank * a_loc : (rank + 1) * a_loc], got3), "3-D dim1 mismatch"

    bad = torch.randn(1, _HEADS + 1, s_loc, _HDIM)
    with pytest.raises(RuntimeError):
        ulysses_async.issue_ulysses_dim1(bad, group)


def _case_dim1_deferred(rank: int, world: int) -> None:
    """issue/finish 拆分：== blocking、形状契约、双 issue、事件接线。"""
    _, group = _mesh_and_group(world)
    a_loc, s_loc = _HEADS // world, _SEQ // world

    torch.manual_seed(11)
    local = torch.randn(1, _HEADS, s_loc, _HDIM)
    blocking = ulysses_async.ulysses_scatter_heads_blocking(local, group)

    recv1, event1, out_shape1 = ulysses_async.issue_ulysses_dim1(local, group)
    recv2, event2, out_shape2 = ulysses_async.issue_ulysses_dim1(local, group)
    assert out_shape1 == torch.Size((1, a_loc, _SEQ, _HDIM)), out_shape1
    assert out_shape2 == out_shape1
    assert tuple(recv1.shape) == (world, a_loc, s_loc, _HDIM), recv1.shape
    assert event1.recorded and event2.recorded, "issue 未记录 fencing 事件"

    fin1 = ulysses_async.finish_ulysses_dim1(recv1, event1, out_shape1)
    fin2 = ulysses_async.finish_ulysses_dim1(recv2, event2, out_shape2)
    assert torch.equal(blocking, fin1), "deferred(1st) != blocking"
    assert torch.equal(blocking, fin2), "deferred(2nd) != blocking"
    assert fin1.is_contiguous()
    assert _waited_events[-2:] == [event1, event2], "finish 未按序 wait issue 返回的事件"


def _case_dim2_forward(rank: int, world: int) -> None:
    """blocking 序列散列：全局切片语义 + dim1∘dim2 恒等 + 整除断言。"""
    _, group = _mesh_and_group(world)
    a_loc, s_loc = _HEADS // world, _SEQ // world

    torch.manual_seed(13)
    g2 = torch.randn(1, _HEADS, _SEQ, _HDIM)
    y_in = g2[:, rank * a_loc : (rank + 1) * a_loc, :, :].contiguous()
    got = ulysses_async.ulysses_scatter_seq_blocking(y_in, group)
    expected = g2[:, :, rank * s_loc : (rank + 1) * s_loc, :]
    assert got.shape == expected.shape, (got.shape, expected.shape)
    assert torch.equal(expected, got), "dim2 blocking 违背全局切片语义"

    dim1_in = torch.randn(1, _HEADS, s_loc, _HDIM)
    roundtrip = ulysses_async.ulysses_scatter_seq_blocking(
        ulysses_async.ulysses_scatter_heads_blocking(dim1_in, group), group
    )
    assert torch.equal(roundtrip, dim1_in), "dim1∘dim2 复合 != 恒等"

    bad = torch.randn(1, a_loc, _SEQ + 1, _HDIM)
    with pytest.raises(RuntimeError):
        ulysses_async.issue_ulysses_dim2(bad, group)


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
    y = ulysses_async.ulysses_scatter_heads_blocking(x, group)
    (grad_blocking,) = torch.autograd.grad((y * go).sum(), (x,))
    assert torch.equal(expected, grad_blocking), "blocking backward != 手写伴随"

    xd = local.detach().clone().requires_grad_(True)
    recv, event, out_shape = ulysses_async.issue_ulysses_dim1(xd, group)
    y_deferred = ulysses_async.finish_ulysses_dim1(recv, event, out_shape)
    (grad_deferred,) = torch.autograd.grad((y_deferred * go).sum(), (xd,))
    assert torch.equal(expected, grad_deferred), "deferred backward != 手写伴随"


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
    out = ulysses_async.ulysses_scatter_seq_blocking(y, group)
    (grad,) = torch.autograd.grad((out * go).sum(), (y,))
    assert torch.equal(expected, grad), "dim2 backward != 手写伴随"

    cross = ulysses_async.ulysses_scatter_heads_blocking(go, group)
    assert torch.equal(cross, grad), "dim2 backward != dim1(go_grad)"


_CASES: dict[str, Callable[[int, int], None]] = {
    "dim1_forward": _case_dim1_forward,
    "dim1_deferred": _case_dim1_deferred,
    "dim2_forward": _case_dim2_forward,
    "backward_dim1": _case_backward_dim1,
    "backward_dim2": _case_backward_dim2,
}


class TestUlyssesAsyncDim1:
    """头散列（dim1）blocking 与 deferred 拆分契约（2-rank gloo）。"""

    def test_blocking_matches_global_semantics_and_sync_helper(self, tmp_path) -> None:
        _run_spawned("dim1_forward", 2, tmp_path)

    def test_issue_finish_split_and_double_issue(self, tmp_path) -> None:
        _run_spawned("dim1_deferred", 2, tmp_path)


class TestUlyssesAsyncDim2:
    """序列散列（dim2）blocking 契约与自逆复合（2-rank gloo）。"""

    def test_seq_scatter_matches_global_semantics_and_roundtrip(self, tmp_path) -> None:
        _run_spawned("dim2_forward", 2, tmp_path)


class TestUlyssesAsyncBackward:
    """两种 movement 的 backward = 同一 a2a 的全伴随（2-rank gloo）。"""

    def test_dim1_backward_matches_hand_adjoint(self, tmp_path) -> None:
        _run_spawned("backward_dim1", 2, tmp_path)

    def test_dim2_backward_matches_hand_adjoint(self, tmp_path) -> None:
        _run_spawned("backward_dim2", 2, tmp_path)


class TestUlyssesAsyncWorldOne:
    """world_size == 1 早退分支（主进程内 1-rank gloo 组）。"""

    def test_world_one_returns_input_unchanged(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_cpu_stubs(monkeypatch)
        dist.init_process_group("gloo", store=dist.FileStore(str(tmp_path / "gloo_store"), 1), rank=0, world_size=1)
        try:
            group = dist.group.WORLD
            x = torch.randn(1, 4, 6, 3)
            assert ulysses_async.ulysses_scatter_heads_blocking(x, group) is x
            assert ulysses_async.ulysses_scatter_seq_blocking(x, group) is x

            recv, event, out_shape = ulysses_async.issue_ulysses_dim1(x, group)
            assert out_shape == torch.Size(x.shape), out_shape
            assert event.recorded
            out = ulysses_async.finish_ulysses_dim1(recv, event, out_shape)
            assert out.shape == x.shape
            assert torch.equal(out, x)
        finally:
            dist.destroy_process_group()


class TestUlyssesAsyncFinishGlue:
    """finish 胶水（permute(1,0)+reshape）的单进程索引契约，无进程组。"""

    def test_finish_glue_layout_contract(self, monkeypatch) -> None:
        _install_cpu_stubs(monkeypatch)
        world, a_loc, s_loc, d = 2, 3, 4, 2
        recv = torch.arange(world * a_loc * s_loc * d, dtype=torch.float32).reshape(world, a_loc, s_loc, d)
        out = ulysses_async.finish_ulysses_dim1(recv, _CpuEvent(), torch.Size((1, a_loc, world * s_loc, d)))
        assert out.is_contiguous()
        for c in range(world):
            for h in range(a_loc):
                for t in range(s_loc):
                    assert torch.equal(out[0, h, c * s_loc + t], recv[c, h, t])

        recv3 = torch.arange(world * a_loc * s_loc, dtype=torch.float32).reshape(world, a_loc, s_loc)
        out3 = ulysses_async.finish_ulysses_dim1(recv3, _CpuEvent(), torch.Size((1, a_loc, world * s_loc)))
        for c in range(world):
            for h in range(a_loc):
                for t in range(s_loc):
                    assert torch.equal(out3[0, h, c * s_loc + t], recv3[c, h, t])

    def test_finish_waits_on_issue_event_and_rejects_bad_shape(self, monkeypatch) -> None:
        _install_cpu_stubs(monkeypatch)
        recv = torch.randn(2, 3, 4, 2)
        event = torch.cuda.Event()
        out = ulysses_async.finish_ulysses_dim1(recv, event, torch.Size((1, 3, 8, 2)))
        assert _waited_events == [event], "finish 未在当前流 wait 传入的事件"
        assert out.shape == (1, 3, 8, 2)
        with pytest.raises(RuntimeError):
            ulysses_async.finish_ulysses_dim1(recv, torch.cuda.Event(), torch.Size((1, 3, 9, 2)))
