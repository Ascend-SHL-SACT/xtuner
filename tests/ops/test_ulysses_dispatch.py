# Copyright © 2026 Huawei Technologies Co., Ltd.
"""KDA SP a2a 异步门控分发（``xtuner/v1/ops/comm/ulysses_dispatch.py``）的契约测试。

全部在纯 CPU / gloo 上运行（CI 无 NPU）。被测对象是 ``kda.py`` 实际 import 的
``ulysses_all_to_all`` 分发函数，契约分四块：

- 路由（``TestDispatchRouting``）：门控 env ``XTUNER_KDA_SP_A2A_ASYNC`` 在**每次调用时**读取。
  未开启（unset / "0" / "" / "junk"）或调用形态不匹配（batch > 1、scatter_dim != 1、
  gather_dim != 2）一律转发同步实现；env == "1" 且 batch-1 + (1, 2) 时改调
  ``ulysses_async.ulysses_scatter_heads_blocking(input, mesh.get_group())``。用例以 monkeypatch
  的记录器（各自返回独立哨兵张量）断言「走了哪条路、传了什么实参」，不触达真实通信。
- 值契约 world==1（``TestDispatchValueWorldOne``）：主进程内 1-rank gloo 组，env 关（同步
  functional collectives 路径）与 env 开（async 的 world==1 早退）都逐位返回输入值。
- 值契约 2-rank（``TestDispatchValueTwoRank``）：沿用 ``test_ulysses_async.py`` 的 spawn
  harness，各 rank 以同一 seed 构造同一全局张量 ``G`` 后按序列切片作本地输入，断言
  dispatch(env on) 与 dispatch(env off) 逐位一致，且都等于文档化的全局切片语义
  ``out[0, h, s] == G[0, rank*(A/sp)+h, s]``（3-D beta 形状同断言），并核实 env 开的 batch-1
  (1, 2) 调用确实经由真实的 ``ulysses_scatter_heads_blocking``。CPU 上
  流/事件不可用，与 sibling 一致地把 ``_a2a_issue`` 换成同步 ``dist.all_to_all_single`` stub、
  ``torch.cuda.Event`` / ``current_stream`` 换成 dummy。
- 接线（``TestKdaImportWiring``）：``kda.py`` import 的 ``ulysses_all_to_all`` 来自分发模块；
  本 box 无 ``fla``，kda **模块本身**可 import（构造 KDA 模块才抛 ImportError，用例不构造）。
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

from xtuner.v1.ops.comm import ulysses_async, ulysses_dispatch


# 被测门控 env 的文档化名称（按文档契约断言，而非引用实现内的常量）。
_ENV = "XTUNER_KDA_SP_A2A_ASYNC"

# 4-D 用例的全局尺寸：heads / 全局 seq / head dim（2-rank 下 a_loc=4, s_loc=8）。
_HEADS = 8
_SEQ = 16
_HDIM = 6

_Call = tuple[str, object]


class TestDispatchRouting:
    """门控 env + 调用形态 → 同步 / async 路径的路由契约（记录器替身，无真实通信）。"""

    def test_env_unset_routes_to_sync(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """env 未设置 → 同步目标以原实参被调，async 目标未被调，返回同步哨兵。"""
        x = torch.randn(1, 4, 6)
        res = _route(monkeypatch, None, x)
        assert res.out is res.sync_sentinel
        assert res.calls == [("sync", x, 1, 2, res.mesh)]

    @pytest.mark.parametrize("env", ["0", "", "junk", "true"])
    def test_env_non_one_values_route_to_sync(self, monkeypatch: pytest.MonkeyPatch, env: str) -> None:
        """env 存在但不等于 "1" → 同步路径。"""
        x = torch.randn(1, 4, 6)
        res = _route(monkeypatch, env, x)
        assert res.out is res.sync_sentinel
        assert res.calls == [("sync", x, 1, 2, res.mesh)]

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
        x = torch.randn(1, 4, 6)
        res = _route(monkeypatch, "1", x, scatter_dim=scatter_dim, gather_dim=gather_dim)
        assert res.out is res.sync_sentinel
        assert res.calls == [("sync", x, scatter_dim, gather_dim, res.mesh)]

    def test_env_on_batch_two_routes_to_sync(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """env == "1" 但 batch > 1 → 同步路径。"""
        x = torch.randn(2, 4, 6)
        res = _route(monkeypatch, "1", x)
        assert res.out is res.sync_sentinel
        assert res.calls == [("sync", x, 1, 2, res.mesh)]

    def test_env_is_read_at_call_time(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """env 在每次调用时读取：两次调用之间翻转 env 即切换路径，无需重新 import。"""
        x = torch.randn(1, 4, 6)
        first = _route(monkeypatch, "0", x)
        assert first.out is first.sync_sentinel
        second = _route(monkeypatch, "1", x)
        assert second.out is second.async_sentinel
        third = _route(monkeypatch, "0", x)
        assert third.out is third.sync_sentinel


class TestDispatchValueWorldOne:
    """world_size == 1（主进程内 1-rank gloo 组）：env 开/关均逐位返回输入值。"""

    def test_env_off_and_on_return_input_values(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        dist.init_process_group("gloo", store=dist.FileStore(str(tmp_path / "gloo_store"), 1), rank=0, world_size=1)
        try:
            mesh = init_device_mesh("cpu", (1,))
            x4 = torch.randn(1, 4, 6, 3)
            x3 = torch.randn(1, 4, 6)

            monkeypatch.delenv(_ENV, raising=False)
            off4 = ulysses_dispatch.ulysses_all_to_all(x4, scatter_dim=1, gather_dim=2, mesh=mesh)
            off3 = ulysses_dispatch.ulysses_all_to_all(x3, scatter_dim=1, gather_dim=2, mesh=mesh)
            monkeypatch.setenv(_ENV, "1")
            on4 = ulysses_dispatch.ulysses_all_to_all(x4, scatter_dim=1, gather_dim=2, mesh=mesh)
            on3 = ulysses_dispatch.ulysses_all_to_all(x3, scatter_dim=1, gather_dim=2, mesh=mesh)
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
    """``kda.py`` 对分发模块的 import 接线（不构造 KDA 模块——那需要 ``fla``）。"""

    def test_kda_imports_dispatch_function(self) -> None:
        # import 本身不得抛异常（本 box 无 fla；kda 的模块级 import 不依赖 fla）。
        import xtuner.v1.module.attention.kda as kda_module

        assert kda_module.ulysses_all_to_all.__module__ == "xtuner.v1.ops.comm.ulysses_dispatch"
        assert kda_module.ulysses_all_to_all is ulysses_dispatch.ulysses_all_to_all


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
    """``torch.cuda.Stream`` 的 CPU dummy：同步 stub 下 wait_event 为 no-op。"""

    def wait_event(self, event: object) -> None:
        return None

    def record_event(self, event: object) -> None:
        return None


class _MeshStub:
    """路由用例的最小 mesh 替身：``get_group()`` 返回固定哨兵，不触达真实进程组。"""

    def __init__(self) -> None:
        self.group: object = object()

    def get_group(self) -> object:
        return self.group


class _RouteResult(NamedTuple):
    """一次路由探测的结果：分发函数返回值、调用记录、两个哨兵与所用 mesh 替身。"""

    out: torch.Tensor
    calls: list[_Call]
    sync_sentinel: torch.Tensor
    async_sentinel: torch.Tensor
    mesh: _MeshStub


def _sync_a2a_issue(send: torch.Tensor, group: dist.ProcessGroup, finished: object) -> torch.Tensor:
    """``ulysses_async._a2a_issue`` 的同步 stub：gloo 上直接 ``all_to_all_single``。

    Args:
        send (torch.Tensor): Uniformly split send buffer (dim 0 chunked per rank).
        group (dist.ProcessGroup): Process group to communicate over.
        finished (object): Dummy event recorded once the collective is complete.

    Returns:
        torch.Tensor: The received buffer (same shape as ``send``).
    """
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    finished.record(None)
    return recv


def _install_worker_cpu_stubs() -> None:
    """spawn 子进程内直接改写（子进程随用例退出，无需还原）。"""
    torch.cuda.Event = _CpuEvent  # type: ignore[assignment]
    torch.cuda.current_stream = lambda *args, **kwargs: _CpuStream()  # type: ignore[assignment]
    ulysses_async._a2a_issue = _sync_a2a_issue  # type: ignore[assignment]


def _patch_route_recorders(monkeypatch: pytest.MonkeyPatch) -> tuple[list[_Call], torch.Tensor, torch.Tensor]:
    """把两个路由目标替换为记录器。

    同步目标 ``ulysses_dispatch._ulysses_all_to_all_sync`` 按 ``(input, scatter_dim,
    gather_dim, mesh)`` 记录；async 目标按 ``(input, group)`` 记录——分发方在调用时才
    ``from xtuner.v1.ops.comm.ulysses_async import ...``，故必须 patch 其宿主模块属性。

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 的 monkeypatch 夹具。

    Returns:
        tuple[list[_Call], torch.Tensor, torch.Tensor]: 调用记录列表、同步哨兵、async 哨兵。
    """
    calls: list[_Call] = []
    sync_sentinel = torch.tensor([1.0])
    async_sentinel = torch.tensor([2.0])

    def fake_sync(inp: torch.Tensor, scatter_dim: int, gather_dim: int, mesh: object) -> torch.Tensor:
        calls.append(("sync", inp, scatter_dim, gather_dim, mesh))
        return sync_sentinel

    def fake_async(inp: torch.Tensor, group: object) -> torch.Tensor:
        calls.append(("async", inp, group))
        return async_sentinel

    monkeypatch.setattr(ulysses_dispatch, "_ulysses_all_to_all_sync", fake_sync)
    monkeypatch.setattr(ulysses_async, "ulysses_scatter_heads_blocking", fake_async)
    return calls, sync_sentinel, async_sentinel


def _route(
    monkeypatch: pytest.MonkeyPatch,
    env: str | None,
    x: torch.Tensor,
    scatter_dim: int = 1,
    gather_dim: int = 2,
) -> _RouteResult:
    """以给定 env 经 ``ulysses_dispatch.ulysses_all_to_all`` 路由一次。

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
    out = ulysses_dispatch.ulysses_all_to_all(x, scatter_dim, gather_dim, mesh)
    return _RouteResult(out, calls, sync_sentinel, async_sentinel, mesh)


def _mesh_and_group(world: int) -> tuple[DeviceMesh, dist.ProcessGroup]:
    """CPU device mesh 及其进程组（与 ``test_ulysses_async.py`` 相同的构建方式）。

    Args:
        world (int): Mesh / 进程组的世界大小。

    Returns:
        tuple[DeviceMesh, dist.ProcessGroup]: mesh 与其底层进程组。
    """
    mesh = init_device_mesh("cpu", (world,))
    return mesh, mesh.get_group()


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
    real_blocking = ulysses_async.ulysses_scatter_heads_blocking
    seen: list[int] = []

    def _spy(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        seen.append(dist.get_world_size(group))
        return real_blocking(x, group)

    ulysses_async.ulysses_scatter_heads_blocking = _spy  # type: ignore[assignment]

    os.environ[_ENV] = "1"
    on4 = ulysses_dispatch.ulysses_all_to_all(local4, scatter_dim=1, gather_dim=2, mesh=mesh)
    on3 = ulysses_dispatch.ulysses_all_to_all(local3, scatter_dim=1, gather_dim=2, mesh=mesh)
    os.environ[_ENV] = "0"
    off4 = ulysses_dispatch.ulysses_all_to_all(local4, scatter_dim=1, gather_dim=2, mesh=mesh)
    off3 = ulysses_dispatch.ulysses_all_to_all(local3, scatter_dim=1, gather_dim=2, mesh=mesh)
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
    "env_on_matches_env_off": _case_env_on_matches_env_off,
}


def _run_spawned(case: str, world: int, tmp_path: Path) -> None:
    """spawn ``world`` 个 worker 跑 ``_CASES[case]``（FileStore gloo 初始化，无端口竞争）。

    Args:
        case (str): ``_CASES`` 中的用例名。
        world (int): 进程组世界大小。
        tmp_path (Path): FileStore 文件的宿主目录（pytest 夹具提供）。
    """
    store_path = str(tmp_path / "gloo_store")
    mp.spawn(_worker, args=(world, store_path, case), nprocs=world, join=True)


def _worker(rank: int, world: int, store_path: str, case: str) -> None:
    dist.init_process_group("gloo", store=dist.FileStore(store_path, world), rank=rank, world_size=world)
    _install_worker_cpu_stubs()
    try:
        _CASES[case](rank, world)
    finally:
        dist.destroy_process_group()
