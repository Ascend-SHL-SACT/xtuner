# Copyright © 2026 Huawei Technologies Co., Ltd.
"""Ulysses a2a staging pool (``xtuner/v1/ops/comm/a2a_pool.py``) 的契约测试。

被测契约：

- 门控 ``XTUNER_ULYSSES_A2A_POOL`` **每次调用时**读取；关闭时 ``_a2a_issue`` 走
  ``torch.empty_like``、``pool_release`` 是 no-op，即与未加池完全一致。
- 复用：同一 (numel, dtype, device) 的缓冲在 release 后被下一次 acquire 原址复用
  （``data_ptr`` 相同）；非池缓冲（调用方自己的张量、``world_size == 1`` 的原样返回）
  release 是 no-op，绝不被池扣留。
- 上限 ``XTUNER_ULYSSES_A2A_POOL_CAP_GIB``：空闲链表超过上限的缓冲被丢弃而不是保留。
- 端到端（2-rank gloo，真 ``_a2a_issue``）：开池与关池的 issue/finish 结果逐位一致；
  一次 join/finish 之后 recv 归还池、send 在 work 退休后被剪枝归还；连续两次同形 a2a
  复用同一块缓冲，且早先 glue 出来的张量不受后续覆盖影响。

纯 CPU / gloo 运行：``torch.cuda`` 的流与事件在本 box 上不可用（NPU 经 transfer_to_npu
映射），用例按 ``tests/ops/test_ulysses_dispatch.py`` 的既有做法把它们替换为 dummy。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

from xtuner.v1.ops.comm import a2a_pool, ulysses_dispatch


_ENV = "XTUNER_ULYSSES_A2A_POOL"
_CAP_ENV = "XTUNER_ULYSSES_A2A_POOL_CAP_GIB"

_HEADS = 8
_SEQ = 16
_HDIM = 6


def _reset_pool() -> None:
    """Drop every pooled buffer and clear the counters (per-test isolation)."""
    a2a_pool._FREE.clear()
    a2a_pool._HELD.clear()
    a2a_pool._FREE_BYTES = 0
    for key in a2a_pool._STATS:
        a2a_pool._STATS[key] = 0


class _CpuEvent:
    """``torch.cuda.Event`` 的 CPU dummy：只记录被 record 过。"""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.recorded = False

    def record(self, stream: object | None = None) -> None:
        self.recorded = True

    def wait(self, stream: object | None = None) -> None:
        return None


class _CpuStream:
    """``torch.cuda.Stream`` 的 CPU dummy：``wait_event`` 记下事件。"""

    def wait_event(self, event: object) -> None:
        return None

    def record_event(self, event: object) -> None:
        return None


class _StreamCtx:
    """``torch.cuda.stream`` 的 CPU dummy：进入时原样返回 dummy 流。"""

    def __init__(self, stream: object) -> None:
        self.stream = stream

    def __enter__(self) -> object:
        return self.stream

    def __exit__(self, *args: object) -> None:
        return None


def _install_worker_cpu_stubs() -> None:
    """spawn 子进程内把 ``torch.cuda`` 流/事件换成 dummy（子进程退出即失效）。"""
    torch.cuda.Event = _CpuEvent  # type: ignore[assignment]
    torch.cuda.Stream = _CpuStream  # type: ignore[assignment]
    torch.cuda.current_stream = lambda *args, **kwargs: _CpuStream()  # type: ignore[assignment]
    torch.cuda.stream = _StreamCtx  # type: ignore[assignment]


class TestPoolGating:
    """门控语义：每次调用读取、关闭时与原路径逐位一致。"""

    def test_gate_read_per_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for value, expected in (("1", True), ("0", False), ("", False), ("junk", False), ("2", False)):
            monkeypatch.setenv(_ENV, value)
            assert a2a_pool.a2a_pool_enabled() is expected, value
        monkeypatch.delenv(_ENV, raising=False)
        assert a2a_pool.a2a_pool_enabled() is False


class TestPoolReuse:
    """复用与释放语义。"""

    def test_release_then_acquire_reuses_block(self) -> None:
        _reset_pool()
        shape, dtype, device = torch.Size((4, 6, 2)), torch.float32, torch.device("cpu")
        first = a2a_pool.pool_acquire(shape, dtype, device)
        a2a_pool.pool_release(first)
        second = a2a_pool.pool_acquire(shape, dtype, device)
        assert second.data_ptr() == first.data_ptr(), "同形缓冲应在 release 后被原址复用"
        assert a2a_pool.a2a_pool_stats()["hit"] == 1.0
        assert a2a_pool.a2a_pool_stats()["miss"] == 1.0
        a2a_pool.pool_release(second)

    def test_release_ignores_foreign_tensor(self) -> None:
        _reset_pool()
        plain = torch.randn(4, 6)
        a2a_pool.pool_release(plain)
        assert a2a_pool.a2a_pool_stats()["released"] == 0.0
        assert not a2a_pool._FREE, "调用方张量不得进入空闲链表"

    def test_cap_drops_overflow(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _reset_pool()
        monkeypatch.setenv(_CAP_ENV, "0.000001")  # ~1 KiB cap
        buf = a2a_pool.pool_acquire(torch.Size((1024,)), torch.float32, torch.device("cpu"))
        a2a_pool.pool_release(buf)
        assert a2a_pool.a2a_pool_stats()["dropped"] == 1.0
        assert not a2a_pool._FREE

    def test_release_drops_aliased_consumer(self) -> None:
        """退化 permute 时 glue 就是缓冲本身：必须丢弃，不能回收。"""
        _reset_pool()
        shape, dtype, device = torch.Size((4, 1, 6)), torch.float32, torch.device("cpu")
        buf = a2a_pool.pool_acquire(shape, dtype, device)
        glue = buf.permute(0, 2, 1)
        assert glue.data_ptr() == buf.data_ptr(), "用例形状未触发退化 permute"
        a2a_pool.pool_release(buf, glue)
        stats = a2a_pool.a2a_pool_stats()
        assert stats["aliased"] == 1.0, "别名 glue 未被识别"
        assert not a2a_pool._FREE, "别名 glue 的缓冲不得进入空闲链表"

    def test_cap_zero_is_unbounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _reset_pool()
        monkeypatch.setenv(_CAP_ENV, "0")
        buf = a2a_pool.pool_acquire(torch.Size((1024,)), torch.float32, torch.device("cpu"))
        a2a_pool.pool_release(buf)
        assert a2a_pool.a2a_pool_stats()["dropped"] == 0.0
        assert len(a2a_pool._FREE) == 1

    def test_cap_default_one_gib(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(_CAP_ENV, raising=False)
        assert a2a_pool.a2a_pool_cap_bytes() == 1 << 30
        monkeypatch.setenv(_CAP_ENV, "junk")
        assert a2a_pool.a2a_pool_cap_bytes() == 1 << 30


def _run_spawned(case: str, world: int, tmp_path: Path) -> None:
    mp.spawn(_worker, args=(world, str(tmp_path / "gloo_store"), case), nprocs=world, join=True)


def _worker(rank: int, world: int, store_path: str, case: str) -> None:
    dist.init_process_group("gloo", store=dist.FileStore(store_path, world), rank=rank, world_size=world)
    _install_worker_cpu_stubs()
    try:
        (_CASES[case])(rank, world)
    finally:
        dist.destroy_process_group()


def _case_pooled_roundtrip(rank: int, world: int) -> None:
    """开池的 issue/finish 与关池逐位一致；缓冲被复用且不泄漏。"""
    _reset_pool()
    mesh = init_device_mesh("cpu", (world,))
    group = mesh.get_group()
    s_loc = _SEQ // world

    torch.manual_seed(31)
    g4 = torch.randn(1, _HEADS, _SEQ, _HDIM)
    local = g4[:, :, rank * s_loc : (rank + 1) * s_loc, :].contiguous()

    os.environ.pop(_ENV, None)
    reference = ulysses_dispatch.ulysses_scatter_heads_blocking(local, group)
    off_stats = a2a_pool.a2a_pool_stats()
    assert off_stats["miss"] == 0.0 and off_stats["hit"] == 0.0, "关池路径不得触达池"

    os.environ[_ENV] = "1"
    recv1, work1, shape1 = ulysses_dispatch.issue_ulysses_dim1(local, group)
    assert a2a_pool.is_pooled_staging(recv1), "开池后 recv 必须来自池"
    out1 = ulysses_dispatch.finish_ulysses_dim1(recv1, work1, shape1)
    assert torch.equal(reference, out1), "开池首轮与关池参考不一致"
    assert not a2a_pool.is_pooled_staging(recv1), "finish 后 recv 应已归还池"
    assert ulysses_dispatch._PENDING_A2A == [], "fence 后 pending 注册表应剪枝为空"

    # 一轮 issue/finish 只向池取一块 recv；第二轮同形 a2a 必须原址复用（miss 不增、hit 增 1）。
    stats1 = a2a_pool.a2a_pool_stats()
    recv2, work2, shape2 = ulysses_dispatch.issue_ulysses_dim1(local, group)
    out2 = ulysses_dispatch.finish_ulysses_dim1(recv2, work2, shape2)
    stats2 = a2a_pool.a2a_pool_stats()
    assert torch.equal(reference, out2), "复用缓冲后的值不一致"
    assert torch.equal(reference, out1), "后一轮 a2a 覆盖缓冲不得影响先前的 glue 结果"
    assert stats2["miss"] == stats1["miss"], "第二轮仍有新分配，缓冲未复用"
    assert stats2["hit"] == stats1["hit"] + 1.0, (stats1["hit"], stats2["hit"])
    assert recv2.data_ptr() == recv1.data_ptr(), "同形 recv 应原址复用"

    # 逆运动同样成立，且两方向互不干扰（不同 shape -> 各自成桶）。
    torch.manual_seed(32)
    back = ulysses_dispatch.issue_ulysses_dim2(out1.transpose(1, 2).contiguous(), group)
    assert a2a_pool.is_pooled_staging(back[0]), "dim2 recv 必须来自池"
    os.environ.pop(_ENV, None)
    ref2 = ulysses_dispatch.ulysses_scatter_seq_blocking(out1.transpose(1, 2).contiguous(), group)
    os.environ[_ENV] = "1"
    got2 = ulysses_dispatch.finish_ulysses_dim2(*back)
    assert torch.equal(ref2, got2), "开池 dim2 与关池参考不一致"

    os.environ.pop(_ENV, None)
    stats = a2a_pool.a2a_pool_stats()
    assert stats["hit"] >= 1.0, "复用未生效"
    assert a2a_pool._HELD == {}, "用例结束仍有缓冲未归还"


def _case_pooled_autograd(rank: int, world: int) -> None:
    """复用缓冲不得破坏 autograd：前向值与反向梯度都与关池逐位一致。"""
    _reset_pool()
    mesh = init_device_mesh("cpu", (world,))
    group = mesh.get_group()
    s_loc = _SEQ // world

    torch.manual_seed(41)
    base = torch.randn(1, _HEADS, _SEQ, _HDIM, dtype=torch.float64)

    def local_input() -> torch.Tensor:
        loc = base[:, :, rank * s_loc : (rank + 1) * s_loc, :].clone()
        loc.requires_grad_(True)
        return loc

    def twice(x: torch.Tensor) -> torch.Tensor:
        # 第二次 issue 会（开池时）拿到第一次已归还的同一块缓冲，而第一次的
        # glue 结果仍在反传图里 —— 值安全与版本计数都由该用例覆盖。
        r1, w1, s1 = ulysses_dispatch.issue_ulysses_dim1(x, group)
        o1 = ulysses_dispatch.finish_ulysses_dim1(r1, w1, s1)
        r2, w2, s2 = ulysses_dispatch.issue_ulysses_dim1(x, group)
        o2 = ulysses_dispatch.finish_ulysses_dim1(r2, w2, s2)
        return (o1 * o2).sum()

    os.environ.pop(_ENV, None)
    x_ref = local_input()
    loss_ref = twice(x_ref)
    loss_ref.backward()

    os.environ[_ENV] = "1"
    x_pool = local_input()
    loss_pool = twice(x_pool)
    loss_pool.backward()

    assert loss_pool.item() == loss_ref.item(), (loss_pool.item(), loss_ref.item())
    assert torch.equal(x_pool.grad, x_ref.grad), "复用缓冲改变了反传梯度"
    assert a2a_pool.a2a_pool_stats()["hit"] >= 1.0, "autograd 用例未触发缓冲复用"

    os.environ.pop(_ENV, None)
    assert a2a_pool._HELD == {}, "用例结束仍有缓冲未归还"


def _case_pooled_degenerate(rank: int, world: int) -> None:
    """退化形状（``num_heads == world`` ⇒ ``A/sp == 1``）的 glue 是缓冲本身，绝不能复用。"""
    _reset_pool()
    mesh = init_device_mesh("cpu", (world,))
    group = mesh.get_group()
    s_loc = _SEQ // world

    torch.manual_seed(51)
    base_a = torch.randn(1, world, _SEQ, _HDIM)
    base_b = torch.randn(1, world, _SEQ, _HDIM)
    local_a = base_a[:, :, rank * s_loc : (rank + 1) * s_loc, :].clone()
    local_b = base_b[:, :, rank * s_loc : (rank + 1) * s_loc, :].clone()

    os.environ.pop(_ENV, None)
    ref_a = ulysses_dispatch.ulysses_scatter_heads_blocking(local_a, group)
    ref_b = ulysses_dispatch.ulysses_scatter_heads_blocking(local_b, group)

    # 两轮携带不同数据，模拟 vision tower 里串行的 q / k / v：若第二轮复用了第一轮
    # 尚未读完的缓冲，out1 会被改写成第二轮的数值。
    os.environ[_ENV] = "1"
    recv1, work1, shape1 = ulysses_dispatch.issue_ulysses_dim1(local_a, group)
    out1 = ulysses_dispatch.finish_ulysses_dim1(recv1, work1, shape1)
    assert torch.equal(ref_a, out1), "开池首轮与关池参考不一致"
    assert out1.data_ptr() == recv1.data_ptr(), "用例形状未触发退化 permute"

    recv2, work2, shape2 = ulysses_dispatch.issue_ulysses_dim1(local_b, group)
    out2 = ulysses_dispatch.finish_ulysses_dim1(recv2, work2, shape2)
    assert torch.equal(ref_b, out2), "开池次轮与关池参考不一致"
    assert torch.equal(ref_a, out1), "第二轮 a2a 复用了仍在使用的 glue 输出"
    assert a2a_pool.a2a_pool_stats()["aliased"] >= 1.0, "退化形状的缓冲未被识别为别名"

    os.environ.pop(_ENV, None)
    assert a2a_pool._HELD == {}, "用例结束仍有缓冲未归还"


_CASES = {
    "pooled_roundtrip": _case_pooled_roundtrip,
    "pooled_autograd": _case_pooled_autograd,
    "pooled_degenerate": _case_pooled_degenerate,
}


class TestPoolEndToEnd:
    """2-rank gloo 端到端：真 ``_a2a_issue`` + 开池。"""

    def test_pooled_issue_finish_matches_unpooled(self, tmp_path: Path) -> None:
        _run_spawned("pooled_roundtrip", 2, tmp_path)

    def test_pooled_buffer_reuse_keeps_gradients_equal(self, tmp_path: Path) -> None:
        _run_spawned("pooled_autograd", 2, tmp_path)

    def test_degenerate_glue_blocks_reuse(self, tmp_path: Path) -> None:
        """回归：``A/sp == 1`` 时 glue 不拷贝，回收后复用会覆盖仍在使用的输出。"""
        _run_spawned("pooled_degenerate", 2, tmp_path)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
