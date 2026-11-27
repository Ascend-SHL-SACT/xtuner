# Copyright © 2026 Huawei Technologies Co., Ltd.
"""seq-major 直发 Ulysses a2a 原语（``xtuner/v1/ops/comm/ulysses_dispatch.py`` 的
``issue_ulysses_dim1_seq_send`` / ``finish_ulysses_dim1_seq_send``，env 门控
``XTUNER_KDA_A2A_SEQ_SEND``）与 ``kda_lever.sp_deferred_projections`` 接线的契约测试。

全部沿用 ``tests/ops/test_ulysses_dispatch.py`` 的 harness（2-rank gloo spawn + FileStore、
``_a2a_issue`` 同步替换、``torch.cuda.*`` CPU dummy、同 seed 全局张量切片语义参考、
``torch.equal`` 逐位断言口径），纯 CPU / gloo 可跑：

- 值等价：同一全局张量（seq-major 构造，投影输出形态）下，新路径（``issue_ulysses_dim1_seq_send``
  直发连续 seq-major 输入 + ``finish_ulysses_dim1_seq_send`` 纯 view finish）与旧路径
  （``issue_ulysses_dim1`` 接 head-major 转置视图 + ``finish_ulysses_dim1_seq_major`` 单拷贝
  finish）的输出逐位一致，且都等于全局切片语义 ``out[0, s, a] == G[0, r*(A/sp)+a, s]``
  （G 为 head-major 全局张量；各 rank 断言自己的头切片）。3-D（q/k/v/g/beta 形状）与
  4-D（按生产 g_raw 的做法把特征轴 fold 成 3-D 走新路径，与 dim1 scatter 旧路径对照）都覆盖；
  特征轴不可整除 world 时 issue 抛 RuntimeError。
- backward 等价：新旧两条 autograd 全链（加权和与消费端转置消费两种形态）的 d_x 逐位一致，
  且等于手写全伴随（movement 输出梯度按目的地分块回传 + 逆排布，由全局切片语义独立导出）；
  非连续 grad 情形以非连续 ``grad_outputs`` 直接打到 issue 的原始 recv 输出上（保证 backward
  真收到非连续 ``grad_recv``、命中 ``grad_recv.contiguous()`` 分支），新旧 d_x 仍逐位一致。
- finish 是 view（零拷贝的直接证据）：``finish_ulysses_dim1_seq_send`` 返回张量与 recv 共享
  storage（``data_ptr`` 相等），``finish_ulysses_dim1_seq_major`` 返回新 storage——这是
  "少一次拷贝"的可观测契约。
- world==1 分支（主进程 1-rank gloo 组）：issue 按 "returned as is" 契约原样返回输入对象、
  work 为 None、out_shape == 输入形状；finish 后数值恒等；autograd 全链梯度恒等。
- kda_lever 接线（主进程 monkeypatch，不构造 KimiDeltaAttention——那需要 fla）：patch
  ``kda_lever.A2A_SEND_SEQ_MAJOR``，stub 全部 issue/finish 记录调用。门开：
  ``sp_deferred_projections`` 对 q/k/v 以投影输出本身（无 transpose、对象恒等）、g 以
  ``view(batch, seq, projection_size)``、beta 以 ``.float()`` 后原样调
  ``issue_ulysses_dim1_seq_send``，全部消费点走 ``finish_ulysses_dim1_seq_send``，
  ``_finish_short_conv`` 收到 ``seq_send=True``；门关：调旧 ``issue_ulysses_dim1``（输入为
  转置视图）+ ``finish_ulysses_dim1_seq_major``，``seq_send=False``；两门返回值逐位一致。
- 真 async（2-rank gloo ``async_op=True``，保留真实 ``_a2a_issue``、只替换 comm 流部分）：
  deferred fence 后数据逐位等于阻塞参考与旧路径；ready 事件从当前流接到 comm 流；
  全链 backward 的反向 a2a 同样真跑且梯度等于手写伴随；fence 后 ``_PENDING_A2A`` 剪枝为空。
- 门控关退化（``A2A_ASYNC=0`` 穿新原语）：``_a2a_issue`` 在调用流上 inline 同步跑完
  ``all_to_all_single(async_op=False)``（不触碰 comm 流/事件、不注册 ``_PENDING_A2A``），
  issue 返回 ``None`` 句柄、finish 的 ``_wait_a2a(None)`` 为 no-op，seq-send 前向与 backward
  数值仍与全局切片语义/手写伴随/旧路径逐位一致——异步门对 seq-send 新路径也是真开关。
- 失败 work 穿新 finish：未 completed 与 completed-but-failed（``is_success()=False``）两形态
  都从 ``finish_ulysses_dim1_seq_send`` 的 fence 抛出存储的后端异常（view 不执行、recv 不被
  静默消费），失败 entry 钉在 ``_PENDING_A2A`` 里不被误摘。
- ``SP_A2A_DIRECT_CONV`` 优先级：``A2A_SEND_SEQ_MAJOR=1`` + ``SP_A2A_DIRECT_CONV=0`` 时
  ``_finish_short_conv`` 在 seq_send=True 下仍走 view finish（不经 SP_A2A_DIRECT_CONV 分支），
  结果与旧路径逐位一致；门关 + ``SP_A2A_DIRECT_CONV=0`` 走 head-major finish 分支（g/beta 恒
  走 seq_major finish），返回值同样与 seq_send 路径逐位一致。
- 门控 env 契约：``XTUNER_KDA_A2A_SEQ_SEND=1`` 于 import 时读取为 True（子进程验证 env 名）；
  未设置时为 False。

测试环境约束与 ``test_ulysses_dispatch.py`` 相同：``ulysses_dispatch`` 的流/事件是
``torch.cuda.*``，默认用例把 ``_a2a_issue`` 替换为同步 ``dist.all_to_all_single`` stub、
流/事件/``torch.cuda.stream`` 替换为记录调用的 dummy；真 async 用例只替换 comm-stream 部分。
world==1 / 接线 / finish 胶水 / env 契约用例在主 pytest 进程内验证。
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

from xtuner.v1.ops.comm import ulysses_dispatch


# 全局尺寸：2-rank 下 s_loc=8、proj_loc=4；4-D 用例 fold 后特征轴 8*6=48（2-rank 下 24）。
_SEQ = 16
_PROJ = 8
_HEADS = 8
_HDIM = 6

# kda_lever 接线用例的形状（sp_size=2：num_heads=4、head_dim=2）。
_LEVER_S_LOC = 4
_LEVER_PROJ = 8
_LEVER_FA_DIM = 5
_LEVER_SP = 2

# 真 async stub 复用的原始 ``_a2a_issue``（worker 内包一层计数 wrapper）。
_REAL_A2A_ISSUE = ulysses_dispatch._a2a_issue


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
    """``dist.Work`` 的 CPU dummy：``is_completed()`` 可配置，``wait()`` 只计数。"""

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


_waited_events: list[object] = []
_recorded_events: list[_CpuEvent] = []
_issued_works: list[_CpuWork] = []


def _sync_a2a_issue(send: torch.Tensor, group: dist.ProcessGroup) -> tuple[torch.Tensor, _CpuWork]:
    """``ulysses_dispatch._a2a_issue`` 的同步 stub：gloo 上直接 all_to_all_single。"""
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    work = _CpuWork(completed=True)
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


def _install_worker_real_async() -> None:
    """真 async 用例的 worker 安装：门控钉开 + 真 ``_a2a_issue``（只替换 comm 流部分）。"""
    ulysses_dispatch.A2A_ASYNC = True
    _install_worker_cpu_stubs(_real_async_a2a_issue)


def _mesh_and_group(world: int) -> tuple[object, dist.ProcessGroup]:
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


def _seq_send_adjoint(go: torch.Tensor, world: int, group: dist.ProcessGroup) -> torch.Tensor:
    """手写全伴随（由全局切片语义独立导出，不依赖被测实现）。

    movement 契约：rank r 的输出 ``out[0, s, a] = x_j[0, r*proj_loc + a, s - j*s_loc]``
    （x_j 为 rank j 的 seq-major 本地输入）。其伴随：rank j 的输入梯度
    ``d_x_j[0, l, r*proj_loc + a] = go_r[0, j*s_loc + l, a]``——即把当前 rank 的 movement
    输出梯度按目的地 j 分块（``[j, l, a]`` 布局）做一次 a2a，再逆排布回 seq-major 输入形状。

    Args:
        go (torch.Tensor): 当前 rank 的 movement 输出梯度 ``[1, S, A/sp]``（布局任意，值即
            逻辑输出梯度；非连续时先按值物化）。
        world (int): 进程组大小。
        group (dist.ProcessGroup): gloo 组。

    Returns:
        torch.Tensor: 手写输入梯度 ``[1, S/sp, A]``（seq-major，与被测路径的 d_x 同形状）。
    """
    s_loc, proj_loc = go.shape[1] // world, go.shape[2]
    gr = go.reshape(world, s_loc, proj_loc).contiguous()
    gx = torch.empty_like(gr)
    dist.all_to_all_single(gx, gr, group=group)
    return gx.permute(1, 0, 2).contiguous().view(1, s_loc, world * proj_loc)


def _case_seqsend_value(rank: int, world: int) -> None:
    """值等价：新旧路径逐位一致 + 全局切片语义 + 4-D fold 对照 + 不可整除分支。"""
    _, group = _mesh_and_group(world)
    s_loc, proj_loc = _SEQ // world, _PROJ // world

    torch.manual_seed(7)
    # seq-major 全局张量（生产投影输出的"全局版"）：G_seq[0, s, a] == G_head[0, a, s]。
    g_seq = torch.randn(1, _SEQ, _PROJ)
    g_head = g_seq.transpose(1, 2)  # [1, _PROJ, _SEQ] head-major 视图（切片语义参考）

    # -- 旧路径：head-major 转置视图（生产 hand-off 形态）+ seq-major 单拷贝 finish --
    x_head = g_seq[:, rank * s_loc : (rank + 1) * s_loc, :].transpose(1, 2)
    recv_o, work_o, out_shape_o = ulysses_dispatch.issue_ulysses_dim1(x_head, group)
    assert out_shape_o == torch.Size((1, proj_loc, _SEQ)), out_shape_o
    out_old = ulysses_dispatch.finish_ulysses_dim1_seq_major(recv_o, work_o)

    # -- 新路径：连续 seq-major 直发 + 纯 view finish --
    x_seq = g_seq[:, rank * s_loc : (rank + 1) * s_loc, :].contiguous()
    recv_n, work_n, out_shape_n = ulysses_dispatch.issue_ulysses_dim1_seq_send(x_seq, group)
    assert out_shape_n == torch.Size((1, s_loc * world, proj_loc)), out_shape_n
    assert tuple(recv_n.shape) == (world, s_loc, proj_loc), recv_n.shape
    assert work_n is not None, "issue 未返回 work 句柄"
    out_new = ulysses_dispatch.finish_ulysses_dim1_seq_send(recv_n, work_n)

    assert out_new.shape == (1, _SEQ, proj_loc), out_new.shape
    assert out_new.is_contiguous() and out_old.is_contiguous(), "两种 finish 输出都应连续"
    assert torch.equal(out_old, out_new), "新旧路径输出不逐位一致"
    assert torch.equal(out_new, g_seq[:, :, rank * proj_loc : (rank + 1) * proj_loc]), "违背全局切片语义"
    for s in range(_SEQ):
        for a in range(proj_loc):
            assert torch.equal(out_new[0, s, a], g_head[0, rank * proj_loc + a, s]), (
                f"out[0, {s}, {a}] != G[0, {rank * proj_loc + a}, {s}]（head-major 切片语义）"
            )
    # view（零拷贝）契约：finish 输出与原始 recv 缓冲共享 storage。
    assert out_new.data_ptr() == recv_n.data_ptr(), "seq_send finish 应零拷贝（与 recv 共享 storage）"

    # -- 4-D：按生产 g_raw 的做法 fold 特征轴走新路径，与 dim1 scatter 旧路径对照 --
    torch.manual_seed(9)
    g4 = torch.randn(1, _SEQ, _HEADS, _HDIM)  # seq-major 4-D 全局（dim1=seq、dim2=heads）
    h_loc = _HEADS // world
    x4_head = g4[:, rank * s_loc : (rank + 1) * s_loc, :, :].transpose(1, 2)
    recv4o, work4o, out_shape4o = ulysses_dispatch.issue_ulysses_dim1(x4_head, group)
    assert out_shape4o == torch.Size((1, h_loc, _SEQ, _HDIM)), out_shape4o
    out4_old = ulysses_dispatch.finish_ulysses_dim1(recv4o, work4o, out_shape4o)

    x4 = g4[:, rank * s_loc : (rank + 1) * s_loc, :, :].contiguous().reshape(1, s_loc, _HEADS * _HDIM)
    recv4n, work4n, out_shape4n = ulysses_dispatch.issue_ulysses_dim1_seq_send(x4, group)
    assert out_shape4n == torch.Size((1, _SEQ, _HEADS * _HDIM // world)), out_shape4n
    out4_new = ulysses_dispatch.finish_ulysses_dim1_seq_send(recv4n, work4n).view(1, _SEQ, h_loc, _HDIM)
    assert torch.equal(out4_new, out4_old.permute(0, 2, 1, 3)), "4-D 新路径 != 旧路径 dim1 scatter"
    assert torch.equal(out4_new, g4[:, :, rank * h_loc : (rank + 1) * h_loc, :]), "4-D 违背全局切片语义"

    # -- 特征轴不可整除 world：send buffer 的 view 构造失败 --
    bad = torch.randn(1, s_loc, _PROJ + 1)
    with pytest.raises(RuntimeError):
        ulysses_dispatch.issue_ulysses_dim1_seq_send(bad, group)


def _case_seqsend_backward(rank: int, world: int) -> None:
    """backward 等价：新旧 autograd 全链 d_x 逐位一致且等于手写全伴随；转置消费形态同断言。"""
    _, group = _mesh_and_group(world)
    s_loc, proj_loc = _SEQ // world, _PROJ // world

    torch.manual_seed(11)
    g_seq = torch.randn(1, _SEQ, _PROJ)
    go = torch.randn(1, _SEQ, proj_loc)  # movement 输出梯度（两 rank 同值）
    local = g_seq[:, rank * s_loc : (rank + 1) * s_loc, :]

    # 旧路径全链：head-major 转置视图进 issue，梯度经 transpose 回传到 seq-major 叶子。
    x_old = local.detach().clone().requires_grad_(True)
    recv_o, work_o, _ = ulysses_dispatch.issue_ulysses_dim1(x_old.transpose(1, 2), group)
    out_old = ulysses_dispatch.finish_ulysses_dim1_seq_major(recv_o, work_o)
    (grad_old,) = torch.autograd.grad((out_old * go).sum(), (x_old,))

    # 新路径全链：seq-major 直发 + view finish。
    x_new = local.detach().clone().requires_grad_(True)
    recv, work, _ = ulysses_dispatch.issue_ulysses_dim1_seq_send(x_new, group)
    out_new = ulysses_dispatch.finish_ulysses_dim1_seq_send(recv, work)
    (grad_new,) = torch.autograd.grad((out_new * go).sum(), (x_new,))

    expected = _seq_send_adjoint(go, world, group)
    assert torch.equal(expected, grad_new), "新路径 backward != 手写全伴随"
    assert torch.equal(expected, grad_old), "旧路径 backward != 手写全伴随"
    assert torch.equal(grad_old, grad_new), "新旧路径 backward d_x 不逐位一致"

    # 消费端产生转置梯度的情形（生产 chunk kernel 以转置视图消费 movement 输出）：
    # 梯度经 transpose 回传，两条路径的 d_x 仍须等于同一手写伴随。
    torch.manual_seed(12)
    w_head = torch.randn(1, proj_loc, _SEQ)  # head-major 消费权重
    x_old2 = local.detach().clone().requires_grad_(True)
    recv_o2, work_o2, _ = ulysses_dispatch.issue_ulysses_dim1(x_old2.transpose(1, 2), group)
    out_old2 = ulysses_dispatch.finish_ulysses_dim1_seq_major(recv_o2, work_o2)
    (grad_old2,) = torch.autograd.grad((out_old2.transpose(1, 2) * w_head).sum(), (x_old2,))
    x_new2 = local.detach().clone().requires_grad_(True)
    recv2, work2, _ = ulysses_dispatch.issue_ulysses_dim1_seq_send(x_new2, group)
    out_new2 = ulysses_dispatch.finish_ulysses_dim1_seq_send(recv2, work2)
    (grad_new2,) = torch.autograd.grad((out_new2.transpose(1, 2) * w_head).sum(), (x_new2,))
    expected2 = _seq_send_adjoint(w_head.transpose(1, 2), world, group)
    assert torch.equal(expected2, grad_new2), "转置消费形态：新路径 backward != 手写全伴随"
    assert torch.equal(expected2, grad_old2), "转置消费形态：旧路径 backward != 手写全伴随"
    assert torch.equal(grad_old2, grad_new2), "转置消费形态：新旧 d_x 不逐位一致"
    # backward 内部的反向 a2a 同样走 issue -> _wait_a2a：每个已发 work 都被 is_completed 轮询。
    assert all(w.is_completed_calls >= 1 for w in _issued_works), "反向 a2a work 未被 fence 轮询"


def _case_seqsend_backward_noncontig(rank: int, world: int) -> None:
    """非连续 grad：以非连续 grad_outputs 直接打到 issue 原始 recv 上（保证 backward 收到
    非连续 grad_recv、命中 ``grad_recv.contiguous()`` 分支），新旧 d_x 仍逐位一致。"""
    _, group = _mesh_and_group(world)
    s_loc, proj_loc = _SEQ // world, _PROJ // world

    torch.manual_seed(13)
    local = torch.randn(1, s_loc, _PROJ)
    base = torch.randn(world, s_loc, 2 * proj_loc)
    go_nc = base[:, :, ::2]  # 非连续 [world, s_loc, proj_loc]，值即逻辑输出梯度
    assert not go_nc.is_contiguous(), "go_nc 必须非连续，否则没有打到 contiguous 分支"

    x_new = local.detach().clone().requires_grad_(True)
    recv_n, work_n, _ = ulysses_dispatch.issue_ulysses_dim1_seq_send(x_new, group)
    (grad_nc_new,) = torch.autograd.grad(outputs=recv_n, inputs=x_new, grad_outputs=go_nc)

    x_old = local.detach().clone().requires_grad_(True)
    recv_o, work_o, _ = ulysses_dispatch.issue_ulysses_dim1(x_old.transpose(1, 2), group)
    go_nc_old = go_nc.permute(0, 2, 1)  # 同一逻辑梯度的 head-major 布局，仍非连续
    assert not go_nc_old.is_contiguous()
    (grad_nc_old,) = torch.autograd.grad(outputs=recv_o, inputs=x_old, grad_outputs=go_nc_old)

    expected = _seq_send_adjoint(go_nc.reshape(1, _SEQ, proj_loc), world, group)
    # go_full[0, c*s_loc + l, a] == go_nc[c, l, a]：非连续梯度按逻辑输出形态摊平后交给伴随。
    assert torch.equal(expected, grad_nc_new), "非连续 grad：新路径 backward != 手写全伴随"
    assert torch.equal(expected, grad_nc_old), "非连续 grad：旧路径 backward != 手写全伴随"
    assert torch.equal(grad_nc_new, grad_nc_old), "非连续 grad：新旧 d_x 不逐位一致"
    # 本用例直接消费原始 recv（未走 finish），两条前向 work 不被 fence 是设计使然；
    # 两条 backward 各自发出的反向 a2a work 仍须被 is_completed 轮询。
    bwd_works = [w for w in _issued_works if w is not work_n and w is not work_o]
    assert len(bwd_works) == 2 and all(w.is_completed_calls >= 1 for w in bwd_works), "反向 a2a work 未被 fence 轮询"


def _case_seqsend_real_async(rank: int, world: int) -> None:
    """真 async（gloo ``async_op=True``）：deferred fence 后数据 == 阻塞参考/旧路径，
    ready 事件流接线成立，全链 backward 反向 a2a 真跑且梯度 == 手写伴随。"""
    _, group = _mesh_and_group(world)
    s_loc, proj_loc = _SEQ // world, _PROJ // world

    torch.manual_seed(17)
    g_seq = torch.randn(1, _SEQ, _PROJ)
    local = g_seq[:, rank * s_loc : (rank + 1) * s_loc, :].contiguous()
    # 手写阻塞参考：同步 a2a + 与 finish 同款的纯 view 胶水。
    send = local.view(s_loc, world, proj_loc).permute(1, 0, 2).contiguous()
    ref_recv = torch.empty_like(send)
    dist.all_to_all_single(ref_recv, send, group=group)
    expected = ref_recv.view(1, _SEQ, proj_loc)

    recv, work, out_shape = ulysses_dispatch.issue_ulysses_dim1_seq_send(local, group)
    assert out_shape == torch.Size((1, _SEQ, proj_loc)), out_shape
    assert work is not None, "真 async a2a 必须返回 work 句柄"
    # issue 的流接线：ready 事件 record 到当前流，comm 流 wait_event 接住同一个事件。
    assert _waited_events and _recorded_events, "issue 未接 ready 事件"
    assert _waited_events[0] in _recorded_events, "comm 流 wait 的不是 issue record 的事件"

    out = ulysses_dispatch.finish_ulysses_dim1_seq_send(recv, work)
    assert torch.equal(expected, out), "真 async fence 后数据 != 阻塞参考"
    assert work.is_completed_calls >= 1, "finish 未轮询 work.is_completed()"
    assert work.wait_calls <= 1, "wait() 至多在 not completed 时发生一次"

    # 旧路径同数据对照（真 async 下同样逐位一致）。
    recv_o, work_o, _ = ulysses_dispatch.issue_ulysses_dim1(local.transpose(1, 2), group)
    out_o = ulysses_dispatch.finish_ulysses_dim1_seq_major(recv_o, work_o)
    assert torch.equal(out_o, expected), "真 async 旧路径 != 阻塞参考"

    # 真 async 全链 backward：反向 a2a 也真跑。
    go = torch.randn(1, _SEQ, proj_loc)
    x = local.detach().clone().requires_grad_(True)
    recv_b, work_b, _ = ulysses_dispatch.issue_ulysses_dim1_seq_send(x, group)
    out_b = ulysses_dispatch.finish_ulysses_dim1_seq_send(recv_b, work_b)
    (grad,) = torch.autograd.grad((out_b * go).sum(), (x,))
    assert torch.equal(_seq_send_adjoint(go, world, group), grad), "真 async backward != 手写全伴随"
    # outstanding send/recv 缓冲注册表在 fence 后应被剪枝，不再滞留引用。
    assert not ulysses_dispatch._PENDING_A2A, "fence 后 _PENDING_A2A 未剪枝"
    assert all(w.is_completed_calls >= 1 for w in _issued_works), "反向 a2a work 未被 fence 轮询"


def _case_seqsend_gate_off_sync(rank: int, world: int) -> None:
    """A2A_ASYNC=0（门关）穿新原语：``_a2a_issue`` 退化为调用流上的 inline 同步 a2a。

    KDA 延迟原语绕过 ``all_to_all.ulysses_all_to_all``，本用例断言门关时新原语同样同步退化：
    issue 返回 ``None`` 句柄、不触碰 comm 流/事件、不注册 ``_PENDING_A2A``，finish 的
    ``_wait_a2a(None)`` 为 no-op，前向与 backward 数值仍与全局切片语义/手写伴随/旧路径逐位
    一致（异步门对 seq-send 新路径也是真开关）。worker 安装保留真实 ``_a2a_issue``——同步
    stub 会掩盖门控分支。
    """
    _, group = _mesh_and_group(world)
    s_loc, proj_loc = _SEQ // world, _PROJ // world

    torch.manual_seed(37)
    g_seq = torch.randn(1, _SEQ, _PROJ)
    local = g_seq[:, rank * s_loc : (rank + 1) * s_loc, :].contiguous()
    expected = g_seq[:, :, rank * proj_loc : (rank + 1) * proj_loc]

    old_gate = ulysses_dispatch.A2A_ASYNC
    try:
        ulysses_dispatch.A2A_ASYNC = False
        recv, work, out_shape = ulysses_dispatch.issue_ulysses_dim1_seq_send(local, group)
        assert out_shape == torch.Size((1, _SEQ, proj_loc)), out_shape
        assert work is None, "门控关时 issue 必须同步跑完并返回 None 句柄"
        assert not ulysses_dispatch._PENDING_A2A, "门控关时不应注册 _PENDING_A2A"
        assert not _waited_events and not _recorded_events, "门控关时不应触碰 comm 流/事件"
        out_new = ulysses_dispatch.finish_ulysses_dim1_seq_send(recv, work)
        assert torch.equal(expected, out_new), "门控关的 seq_send 前向 != 全局切片语义"

        # 旧路径同数据对照（门关下同样同步退化）。
        recv_o, work_o, _ = ulysses_dispatch.issue_ulysses_dim1(local.transpose(1, 2), group)
        assert work_o is None, "门控关时旧 issue 也必须同步"
        out_o = ulysses_dispatch.finish_ulysses_dim1_seq_major(recv_o, work_o)
        assert torch.equal(out_o, expected), "门控关旧路径 != 全局切片语义"

        # backward 的反向 a2a 同走 _a2a_issue：门控关下同步退化且梯度 == 手写全伴随。
        go = torch.randn(1, _SEQ, proj_loc)
        x = local.detach().clone().requires_grad_(True)
        recv_b, work_b, _ = ulysses_dispatch.issue_ulysses_dim1_seq_send(x, group)
        assert work_b is None, "门控关时 backward 前向 issue 也必须同步"
        out_b = ulysses_dispatch.finish_ulysses_dim1_seq_send(recv_b, work_b)
        (grad,) = torch.autograd.grad((out_b * go).sum(), (x,))
        assert torch.equal(_seq_send_adjoint(go, world, group), grad), "门控关 backward != 手写全伴随"
        assert not ulysses_dispatch._PENDING_A2A, "门控关 backward 不应注册 _PENDING_A2A"
    finally:
        ulysses_dispatch.A2A_ASYNC = old_gate


_CASES: dict[str, Callable[[int, int], None]] = {
    "seqsend_value": _case_seqsend_value,
    "seqsend_backward": _case_seqsend_backward,
    "seqsend_backward_noncontig": _case_seqsend_backward_noncontig,
    "seqsend_real_async": _case_seqsend_real_async,
    "seqsend_gate_off_sync": _case_seqsend_gate_off_sync,
}

# 真 async 用例保留真实 ``_a2a_issue``，只替换 comm-stream 部分（事件/流/stream 上下文）；
# 门控关用例同样要真 ``_a2a_issue``（同步 stub 会掩盖门控分支；门关分支不触流，CUDA stub
# 只是兜底）。
_REAL_ASYNC_CASES: dict[str, Callable[[], None]] = {
    "seqsend_real_async": _install_worker_real_async,
    "seqsend_gate_off_sync": lambda: _install_worker_cpu_stubs(_REAL_A2A_ISSUE),
}


# ---------------------------- kda_lever 接线替身 ----------------------------


def _kda_lever_module():
    """惰性 import ``kda_lever``（本机无 fla 也不影响其模块级 import；不构造 KDA 模块）。"""
    import xtuner.v1.module.attention.kda_lever  # noqa: PLC0415

    return xtuner.v1.module.attention.kda_lever


class _StubProj:
    """投影层替身：记录输入、返回固定输出（断言 issue 收到的是否为投影输出本身）。"""

    def __init__(self, out: torch.Tensor) -> None:
        self.out = out
        self.inputs: list[torch.Tensor] = []

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.inputs.append(x)
        return self.out


class _StubConv:
    """短卷积替身：记录 ``__call__`` 收到的 x（应等于对应 finish 的输出）。"""

    def __init__(self, name: str, out: torch.Tensor) -> None:
        self.name = name
        self.out = out
        self.inputs: list[torch.Tensor] = []

    def materialize_weight_bias(self) -> tuple[torch.Tensor, None]:
        return torch.ones(_LEVER_PROJ, 3), None

    def __call__(
        self, x: torch.Tensor, cu_seqlens: object = None, weight: object = None, bias: object = None
    ) -> tuple[torch.Tensor, None]:
        self.inputs.append(x)
        return self.out, None  # 真实 KDAShortConvolution 的 __call__ 返回 (out, aux) 二元组


class _StubAttn:
    """KimiDeltaAttention 替身：只提供 ``sp_deferred_projections`` 触及的属性（不 import fla）。"""

    num_heads = 4
    head_dim = 2

    def __init__(self, s_loc: int) -> None:
        torch.manual_seed(41)
        self.hidden_states = torch.randn(1, s_loc, 6)
        self.q_proj = _StubProj(torch.randn(1, s_loc, _LEVER_PROJ))
        self.k_proj = _StubProj(torch.randn(1, s_loc, _LEVER_PROJ))
        self.v_proj = _StubProj(torch.randn(1, s_loc, _LEVER_PROJ))
        self.f_a_proj = _StubProj(torch.randn(1, s_loc, _LEVER_FA_DIM))
        self.f_b_proj = _StubProj(torch.randn(1, s_loc, _LEVER_PROJ))
        self.b_proj = _StubProj(torch.randn(1, s_loc, _LEVER_PROJ))  # fp32：.float() 恒等别名
        self.q_conv1d = _StubConv("q", torch.full((1, s_loc, 3), 1.0))
        self.k_conv1d = _StubConv("k", torch.full((1, s_loc, 3), 2.0))
        self.v_conv1d = _StubConv("v", torch.full((1, s_loc, 3), 3.0))


class _LeverMeshStub:
    """``sp_deferred_projections`` 的最小 mesh 替身：``get_group()`` 返回固定哨兵。"""

    def __init__(self) -> None:
        self.group: object = object()

    def get_group(self) -> object:
        return self.group


def _logical_move_out(s_loc: int, proj_loc: int) -> torch.Tensor:
    """接线替身用的确定值 movement 输出 ``[1, S, A/sp]``（门开/门关共用同一逻辑值）。"""
    return torch.arange(s_loc * _LEVER_SP * proj_loc, dtype=torch.float32).reshape(1, s_loc * _LEVER_SP, proj_loc)


def _install_lever_stubs(monkeypatch: pytest.MonkeyPatch, gate: bool, direct_conv: bool = True) -> dict[str, list]:
    """stub ``kda_lever`` 命名空间里的全部 comm 函数并 patch 门控，返回调用记录。

    finish 替身复刻真实胶水的布局（view 或 permute 拷贝），recv 缓冲由同一确定值的逻辑
    movement 输出按各自 send 布局物化，使门开/门关两次运行的返回值可以逐位对齐。
    ``direct_conv`` 钉住 ``SP_A2A_DIRECT_CONV``（门关分支的 finish 选择），供优先级用例翻转。
    """
    mod = _kda_lever_module()
    calls: dict[str, list] = {
        "issue_dim1": [],
        "issue_seq": [],
        "finish_major": [],
        "finish_seq": [],
        "finish_dim1": [],
        "conv_seq_send": [],
    }
    sp = _LEVER_SP

    def fake_issue_seq(x: torch.Tensor, group: object) -> tuple[torch.Tensor, object, torch.Size]:
        s_loc, proj_loc = x.shape[1], x.shape[2] // sp
        logical = _logical_move_out(s_loc, proj_loc)
        recv = logical.reshape(sp, s_loc, proj_loc).clone()  # recv[c, l, a] == logical[c*s_loc + l, a]
        calls["issue_seq"].append({"x": x, "work": object(), "recv": recv})
        return recv, calls["issue_seq"][-1]["work"], torch.Size((1, s_loc * sp, proj_loc))

    def fake_issue_dim1(x: torch.Tensor, group: object) -> tuple[torch.Tensor, object, torch.Size]:
        a_loc, s_loc = x.shape[1] // sp, x.shape[2]
        logical = _logical_move_out(s_loc, a_loc)
        # recv[c, a, l] == logical[c*s_loc + l, a]（同一逻辑值的 head-major send 布局）。
        recv = logical.reshape(sp, s_loc, a_loc).permute(0, 2, 1).contiguous()
        calls["issue_dim1"].append({"x": x, "work": object(), "recv": recv})
        return recv, calls["issue_dim1"][-1]["work"], torch.Size((1, a_loc, s_loc * sp))

    def fake_finish_seq(recv: torch.Tensor, work: object) -> torch.Tensor:
        out = recv.view(1, recv.shape[0] * recv.shape[1], recv.shape[2])
        calls["finish_seq"].append((recv, work, out))
        return out

    def fake_finish_major(recv: torch.Tensor, work: object) -> torch.Tensor:
        out = recv.permute(0, 2, 1).contiguous().reshape(1, recv.shape[0] * recv.shape[2], recv.shape[1])
        calls["finish_major"].append((recv, work, out))
        return out

    def fake_finish_dim1(recv: torch.Tensor, work: object, out_shape: torch.Size) -> torch.Tensor:
        calls["finish_dim1"].append((recv, work, out_shape))
        return recv.permute(1, 0, *range(2, recv.dim())).contiguous().reshape(out_shape)

    real_fsc = mod._finish_short_conv

    def spy_finish_conv(
        conv: _StubConv, issued: tuple, sp_rank: int, sp_size: int, cu_seqlens: object, seq_send: bool = False
    ) -> torch.Tensor:
        calls["conv_seq_send"].append((conv.name, seq_send))
        return real_fsc(conv, issued, sp_rank, sp_size, cu_seqlens, seq_send=seq_send)

    monkeypatch.setattr(mod, "issue_ulysses_dim1", fake_issue_dim1)
    monkeypatch.setattr(mod, "issue_ulysses_dim1_seq_send", fake_issue_seq)
    monkeypatch.setattr(mod, "finish_ulysses_dim1_seq_send", fake_finish_seq)
    monkeypatch.setattr(mod, "finish_ulysses_dim1_seq_major", fake_finish_major)
    monkeypatch.setattr(mod, "finish_ulysses_dim1", fake_finish_dim1)
    monkeypatch.setattr(mod, "_finish_short_conv", spy_finish_conv)
    monkeypatch.setattr(mod, "SP_A2A_DIRECT_CONV", direct_conv)  # 钉住门关分支的 finish 选择
    monkeypatch.setattr(mod, "A2A_SEND_SEQ_MAJOR", gate)
    return calls


def _run_lever() -> tuple[_StubAttn, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """以固定形状跑一次 ``sp_deferred_projections``（sp_size=2、sp_rank=1）。"""
    attn = _StubAttn(_LEVER_S_LOC)
    mod = _kda_lever_module()
    out = mod.sp_deferred_projections(
        attn,
        attn.hidden_states,
        _LeverMeshStub(),
        sp_rank=1,
        sp_size=_LEVER_SP,
        cu_seqlens=None,
        batch_size=1,
        seq_len=_LEVER_S_LOC,
    )
    return attn, out


class TestSeqSendValue:
    """值等价（2-rank gloo spawn）：新旧路径逐位一致 + 全局切片语义 + 4-D + 不可整除。"""

    def test_new_path_matches_old_path_and_global_semantics(self, tmp_path: Path) -> None:
        _run_spawned("seqsend_value", 2, tmp_path)


class TestSeqSendBackward:
    """backward 等价（2-rank gloo spawn）：d_x == 手写全伴随 == 新旧互等，含非连续 grad。"""

    def test_backward_matches_old_path_and_hand_adjoint(self, tmp_path: Path) -> None:
        _run_spawned("seqsend_backward", 2, tmp_path)

    def test_backward_with_non_contiguous_grad(self, tmp_path: Path) -> None:
        _run_spawned("seqsend_backward_noncontig", 2, tmp_path)


class TestSeqSendRealAsync:
    """真 async a2a（2-rank gloo ``async_op=True``）：deferred fence + 流接线 + backward。"""

    def test_real_async_deferred_fence_forward_and_backward(self, tmp_path: Path) -> None:
        _run_spawned("seqsend_real_async", 2, tmp_path)


class TestSeqSendA2aGateOff:
    """``A2A_ASYNC=0`` 门关分支穿新原语：inline 同步退化（2-rank gloo spawn）。"""

    def test_gate_off_degrades_seq_send_primitives_to_sync(self, tmp_path: Path) -> None:
        _run_spawned("seqsend_gate_off_sync", 2, tmp_path)


class TestSeqSendFinishView:
    """finish 零拷贝契约（主进程，无进程组）：seq_send finish 是纯 view，seq_major finish 是拷贝。"""

    def test_seq_send_finish_shares_recv_storage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_cpu_stubs(monkeypatch)
        world, s_loc, proj_loc = 2, 3, 4
        # 两个 finish 消费的是各自 issue 的 recv 布局（seq_send 收 [world, s_loc, A/sp] 的
        # seq-major 块，seq_major 收 [world, A/sp, s_loc] 的 head-major 块），这里由同一逻辑
        # movement 输出按各自布局物化两种 recv，使输出值可对齐。
        logical = torch.arange(world * s_loc * proj_loc, dtype=torch.float32).reshape(1, world * s_loc, proj_loc)
        recv_seq = logical.reshape(world, s_loc, proj_loc)
        recv_major = logical.reshape(world, s_loc, proj_loc).permute(0, 2, 1).contiguous()

        out = ulysses_dispatch.finish_ulysses_dim1_seq_send(recv_seq, _CpuWork())
        assert out.shape == (1, world * s_loc, proj_loc), out.shape
        assert out.is_contiguous()
        assert out.data_ptr() == recv_seq.data_ptr(), "seq_send finish 应与 recv 共享 storage（纯 view）"
        assert out.untyped_storage().data_ptr() == recv_seq.untyped_storage().data_ptr()
        for c in range(world):
            for l in range(s_loc):
                for a in range(proj_loc):
                    assert torch.equal(out[0, c * s_loc + l, a], recv_seq[c, l, a])

        # 对照：seq_major finish 是 permute 拷贝，返回新 storage——"少一次拷贝"的直接证据。
        out_major = ulysses_dispatch.finish_ulysses_dim1_seq_major(recv_major, _CpuWork())
        assert torch.equal(out_major, logical), "seq_major finish 值 == 逻辑 movement 输出"
        assert torch.equal(out, out_major), "各自 consume 自己路径的 recv 时输出值一致（同一逻辑 movement）"
        assert out_major.data_ptr() != recv_major.data_ptr(), "seq_major finish 应产生拷贝（新 storage）"
        assert out_major.untyped_storage().data_ptr() != recv_major.untyped_storage().data_ptr()

        # fence 接线：已完成 work 只轮询不 wait。
        work = _CpuWork(completed=True)
        ulysses_dispatch.finish_ulysses_dim1_seq_send(recv_seq, work)
        assert work.is_completed_calls >= 1 and work.wait_calls == 0


class TestSeqSendFailedWork:
    """失败 work 穿新 finish（主进程，无进程组）：``_wait_a2a`` 抛出存储的后端异常而非静默
    消费未写的 recv。stub 处置方式与既有套件的 failed-work 用例一致（wait() raise 与
    completed-but-failed 两形态），失败 entry 必须钉在注册表里。"""

    def test_failed_work_raises_from_seq_send_finish(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_cpu_stubs(monkeypatch)

        class _FailingWork(_CpuWork):
            def is_success(self) -> bool:
                return False

            def wait(self) -> None:
                super().wait()
                raise RuntimeError("backend failure")

        recv = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)

        # 形态一：未 completed → fence 走 wait() raise，异常传播、view 不执行。
        work1 = _FailingWork(completed=False)
        with pytest.raises(RuntimeError):
            ulysses_dispatch.finish_ulysses_dim1_seq_send(recv, work1)
        assert work1.wait_calls == 1, "未 retire 的失败 work 必须被 wait()（异常从 fence 传播）"

        # 形态二：completed-but-failed（is_success()=False → 未 retire → 仍走 wait() raise）。
        work2 = _FailingWork(completed=True)
        with pytest.raises(RuntimeError):
            ulysses_dispatch.finish_ulysses_dim1_seq_send(recv, work2)
        assert work2.wait_calls == 1

        # 失败 entry 钉住：摘除只发生在 wait() 之后，wait() raise 的 entry 不得被误摘。
        work3 = _FailingWork(completed=False)
        entry = (work3, torch.empty(4, 4), recv)
        ulysses_dispatch._PENDING_A2A.append(entry)
        try:
            with pytest.raises(RuntimeError):
                ulysses_dispatch.finish_ulysses_dim1_seq_send(recv, work3)
            assert ulysses_dispatch._PENDING_A2A == [entry], "wait() raise 的失败 entry 被误摘"
        finally:
            ulysses_dispatch._PENDING_A2A.clear()


class TestSeqSendWorldOne:
    """world_size == 1 早退分支（主进程内 1-rank gloo 组）：按 "returned as is" 契约。"""

    def test_world_one_returns_input_as_is(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_cpu_stubs(monkeypatch)
        dist.init_process_group("gloo", store=dist.FileStore(str(tmp_path / "gloo_store"), 1), rank=0, world_size=1)
        try:
            group = dist.group.WORLD
            x = torch.randn(1, 4, 6)
            recv, work, out_shape = ulysses_dispatch.issue_ulysses_dim1_seq_send(x, group)
            assert recv is x, "world==1 issue 应原样返回输入对象"
            assert work is None, "world==1 issue 不产生 work"
            assert out_shape == torch.Size(x.shape), out_shape
            out = ulysses_dispatch.finish_ulysses_dim1_seq_send(recv, work)
            assert out.shape == x.shape
            assert torch.equal(out, x)

            # autograd 全链恒等：world==1 无 a2a 节点，梯度原样回传。
            xg = x.detach().clone().requires_grad_(True)
            recv2, work2, _ = ulysses_dispatch.issue_ulysses_dim1_seq_send(xg, group)
            out2 = ulysses_dispatch.finish_ulysses_dim1_seq_send(recv2, work2)
            (g,) = torch.autograd.grad(out2.sum(), (xg,))
            assert torch.equal(g, torch.ones_like(xg))
        finally:
            dist.destroy_process_group()


class TestKdaLeverSeqSendWiring:
    """``kda_lever.sp_deferred_projections`` 门控接线（主进程 monkeypatch + 替身，不构造 KDA 模块）。"""

    def test_gate_on_issues_seq_major_and_view_finish(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = _install_lever_stubs(monkeypatch, gate=True)
        attn, out = _run_lever()
        _q, _k, _v, g_raw, beta = out
        s_loc, sp = _LEVER_S_LOC, _LEVER_SP
        proj_loc = _LEVER_PROJ // sp

        # 门开：5 个 a2a 全部走 issue_ulysses_dim1_seq_send，旧 issue 未被调用。
        assert len(calls["issue_seq"]) == 5, [c["x"].shape for c in calls["issue_seq"]]
        assert not calls["issue_dim1"], "门开不应调用 issue_ulysses_dim1"
        dq, dk, dv, dg, dbeta = (c["x"] for c in calls["issue_seq"])

        # q/k/v：无 transpose 胶水——issue 收到的就是投影输出对象本身（seq-major 连续）。
        assert dq is attn.q_proj.out and dk is attn.k_proj.out and dv is attn.v_proj.out, "q/k/v 应以投影输出原样直发"
        for x in (dq, dk, dv):
            assert x.shape == (1, s_loc, _LEVER_PROJ) and x.is_contiguous()
        # g：view(batch, seq, projection_size) 的连续 seq-major 视图（与 f_b 输出共享 storage）。
        assert dg.shape == (1, s_loc, _LEVER_PROJ) and dg.is_contiguous()
        assert dg.data_ptr() == attn.f_b_proj.out.data_ptr()
        # beta：.float() 后原样（fp32 恒等别名，无 transpose）。
        assert dbeta is attn.b_proj.out and dbeta.dtype == torch.float32

        # 全部消费点走 view finish；_finish_short_conv 收到 seq_send=True。
        assert len(calls["finish_seq"]) == 5, "门开应有 5 个 view finish（3 conv + g + beta）"
        assert not calls["finish_major"] and not calls["finish_dim1"], "门开不应调用拷贝型 finish"
        assert calls["conv_seq_send"] == [("q", True), ("k", True), ("v", True)], calls["conv_seq_send"]

        # conv 收到的 x 是对应 finish 的输出（recv 的零拷贝 view）；finish 的 work 与 issue 一一对应。
        for i, conv in enumerate((attn.q_conv1d, attn.k_conv1d, attn.v_conv1d)):
            conv_recv, conv_work, conv_out = calls["finish_seq"][i]
            assert conv_work is calls["issue_seq"][i]["work"], "finish 收到的 work 应来自对应 issue"
            assert conv.inputs[0] is conv_out, "conv 输入应为 finish 的返回值本身"
            assert conv.inputs[0].data_ptr() == conv_recv.data_ptr(), "conv 输入应为 recv 的零拷贝 view"
            assert torch.equal(conv_out, _logical_move_out(s_loc, proj_loc)), "finish 输出应为逻辑 movement 输出"
        # g/beta 消费点同样走 view finish（含 g 的 (B, L, H/sp, D) 重排与 beta 的 sigmoid）。
        g_view = calls["finish_seq"][3][2]
        assert torch.equal(g_raw, g_view.view(1, s_loc * sp, attn.num_heads // sp, attn.head_dim))
        assert torch.equal(beta, torch.sigmoid(calls["finish_seq"][4][2]))
        # 投影层只被 hidden_states 调用一次。
        for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.f_a_proj, attn.b_proj):
            assert proj.inputs == [attn.hidden_states]
        assert attn.f_b_proj.inputs == [attn.f_a_proj.out]

    def test_gate_off_keeps_head_major_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = _install_lever_stubs(monkeypatch, gate=False)
        attn, out = _run_lever()
        _q, _k, _v, g_raw, beta = out
        s_loc, sp = _LEVER_S_LOC, _LEVER_SP

        # 门关：5 个 a2a 全部走旧 issue_ulysses_dim1，输入为 head-major 转置视图。
        assert len(calls["issue_dim1"]) == 5, [c["x"].shape for c in calls["issue_dim1"]]
        assert not calls["issue_seq"], "门关不应调用 issue_ulysses_dim1_seq_send"
        dq, dk, dv, dg, dbeta = (c["x"] for c in calls["issue_dim1"])
        for x, proj in ((dq, attn.q_proj), (dk, attn.k_proj), (dv, attn.v_proj)):
            assert x.shape == (1, _LEVER_PROJ, s_loc) and not x.is_contiguous(), "门关 q/k/v 应为转置视图"
            assert x.data_ptr() == proj.out.data_ptr()
            assert torch.equal(x, proj.out.transpose(1, 2))
        assert dg.shape == (1, _LEVER_PROJ, s_loc)
        assert torch.equal(dg, attn.f_b_proj.out.view(1, s_loc, _LEVER_PROJ).transpose(1, 2))
        assert dbeta.dtype == torch.float32
        assert dbeta.data_ptr() == attn.b_proj.out.data_ptr()
        assert torch.equal(dbeta, attn.b_proj.out.transpose(1, 2))

        # 门关消费点全走 seq_major 单拷贝 finish；_finish_short_conv 收到 seq_send=False。
        assert len(calls["finish_major"]) == 5, "门关应有 5 个 seq_major finish"
        assert not calls["finish_seq"] and not calls["finish_dim1"], "门关不应调用 view finish"
        assert calls["conv_seq_send"] == [("q", False), ("k", False), ("v", False)], calls["conv_seq_send"]
        for i, conv in enumerate((attn.q_conv1d, attn.k_conv1d, attn.v_conv1d)):
            conv_recv, conv_work, conv_out = calls["finish_major"][i]
            assert conv_work is calls["issue_dim1"][i]["work"], "finish 收到的 work 应来自对应 issue"
            assert conv.inputs[0] is conv_out, "conv 输入应为 finish 的返回值本身"
            assert conv.inputs[0].data_ptr() != conv_recv.data_ptr(), "门关 finish 应产生拷贝（新 storage）"
            assert torch.equal(conv_out, _logical_move_out(s_loc, _LEVER_PROJ // sp)), "finish 输出应为逻辑 movement 输出"
        assert torch.equal(g_raw, calls["finish_major"][3][2].view(1, s_loc * sp, attn.num_heads // sp, attn.head_dim))
        assert torch.equal(beta, torch.sigmoid(calls["finish_major"][4][2]))

    def test_gate_on_and_off_return_identical_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 两门各自以同 seed 重建替身（recv 缓冲由同一确定值逻辑输出物化），断言返回值逐位一致。
        _install_lever_stubs(monkeypatch, gate=True)
        _attn_on, out_on = _run_lever()
        _install_lever_stubs(monkeypatch, gate=False)
        _attn_off, out_off = _run_lever()
        for name, a, b in zip(("q", "k", "v", "g_raw", "beta"), out_on, out_off):
            assert torch.equal(a, b), f"门开/门关的 {name} 返回值不逐位一致"

    def test_seq_send_takes_priority_over_direct_conv_gate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``A2A_SEND_SEQ_MAJOR=1`` + ``SP_A2A_DIRECT_CONV=0``：``_finish_short_conv`` 在
        seq_send=True 时必须走 view finish（seq_send 优先于 SP_A2A_DIRECT_CONV 分支）。"""
        calls_on = _install_lever_stubs(monkeypatch, gate=True, direct_conv=False)
        _attn_on, out_on = _run_lever()
        assert len(calls_on["finish_seq"]) == 5, "seq_send=True 应仍走 view finish"
        assert not calls_on["finish_major"] and not calls_on["finish_dim1"], "不得落入 SP_A2A_DIRECT_CONV 分支"
        assert calls_on["conv_seq_send"] == [("q", True), ("k", True), ("v", True)], calls_on["conv_seq_send"]

        # 对照旧路径（门关 + SP_A2A_DIRECT_CONV=1）逐位一致。
        _install_lever_stubs(monkeypatch, gate=False, direct_conv=True)
        _attn_off, out_off = _run_lever()
        for name, a, b in zip(("q", "k", "v", "g_raw", "beta"), out_on, out_off):
            assert torch.equal(a, b), f"seq_send 优先路径与旧路径的 {name} 不逐位一致"

        # 补充分支完整性：门关 + SP_A2A_DIRECT_CONV=0 走 head-major finish + transpose
        # （q/k/v 经 _finish_short_conv 的 finish_dim1 分支；g/beta 恒走 seq_major finish），
        # 返回值仍与 seq_send 路径逐位一致。
        calls_off2 = _install_lever_stubs(monkeypatch, gate=False, direct_conv=False)
        _attn_off2, out_off2 = _run_lever()
        assert len(calls_off2["finish_dim1"]) == 3 and len(calls_off2["finish_major"]) == 2, (
            [len(calls_off2["finish_dim1"]), len(calls_off2["finish_major"])]
        )
        assert not calls_off2["finish_seq"], "门关 + DIRECT_CONV=0 不应出现 view finish"
        assert calls_off2["conv_seq_send"] == [("q", False), ("k", False), ("v", False)]
        for name, a, b in zip(("q", "k", "v", "g_raw", "beta"), out_on, out_off2):
            assert torch.equal(a, b), f"head-major finish 分支的 {name} 与 seq_send 路径不逐位一致"


class TestSeqSendGateEnv:
    """门控 env 契约：``XTUNER_KDA_A2A_SEQ_SEND`` 于 import 时读取；未设置为 False。"""

    def test_gate_read_from_env_at_import(self) -> None:
        # 当前进程：门值必须等于 env 的字面读法（默认进程未设置 -> False）。
        assert ulysses_dispatch.A2A_SEND_SEQ_MAJOR == (os.getenv("XTUNER_KDA_A2A_SEQ_SEND", "0") == "1")

        # 子进程：env 名与 "1" 字面量契约（import 时读取，rename/换字面量即翻红）。
        env = dict(os.environ)
        env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        env["XTUNER_KDA_A2A_SEQ_SEND"] = "1"
        code = "import xtuner.v1.ops.comm.ulysses_dispatch as m; print(m.A2A_SEND_SEQ_MAJOR)"
        proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        assert proc.stdout.strip() == "True", proc.stdout + proc.stderr
