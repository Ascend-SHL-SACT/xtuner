# Copyright © 2026 Huawei Technologies Co., Ltd.
"""GLM-5.3 KDA SP 路径 seq-major 单拷贝 a2a finish 回归测试。

覆盖两处实现：

1. ``xtuner.v1.ops.comm.ulysses_dispatch.finish_ulysses_dim1_seq_major``：新公开接口，fence 后
   用一次 ``permute(0, 2, 1).contiguous()`` 直接把 ``[world, A/sp, L/sp]`` 的 a2a 原始接收
   buffer 摆成 seq-major ``[1, world*L/sp, A/sp]``，等价于旧的
   ``finish_ulysses_dim1(...)``（head-major）+ ``transpose(1, 2).contiguous()`` 两拷贝 glue，
   但少一次全量拷贝。元素恒等式：``recv[w, a, l] -> out[0, w*L+l, a]``。
2. ``xtuner.v1.module.attention.kda.KimiDeltaAttention._finish_short_conv``：
   ``XTUNER_KDA_SP_A2A_DIRECT_CONV``（默认开）为 1 时走新 finish，为 0 时走旧两拷贝 glue；
   两条分支的 conv 输出与梯度必须逐位一致（训练走
   ``xtuner/v1/module/attention/kda.py::forward_for_sp`` 活跃路径）。

用例 1 的纯张量部分无需 dist（work=None 时 fence 为 no-op）；``_finish_short_conv`` 分支
等价用 2 进程 gloo spawn（CPU 张量，a2a 用同步 gloo stub）跑通。
"""

import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

import xtuner.v1.module.attention.kda as kda_module
from xtuner.v1.module.attention import kda_lever
import xtuner.v1.ops.comm.ulysses_dispatch as ulysses_dispatch
from xtuner.v1.ops.comm.ulysses_dispatch import (
    finish_ulysses_dim1,
    finish_ulysses_dim1_seq_major,
    issue_ulysses_dim1,
)


def _recv(world: int, a_shard: int, l_loc: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """构造值唯一的 ``[world, A/sp, L/sp]`` recv，元素身份可逐点核对。"""
    vals = torch.arange(world * a_shard * l_loc, dtype=torch.float64) % 97.0
    return vals.reshape(world, a_shard, l_loc).to(dtype) + 1.0


def _old_glue(recv: torch.Tensor, out_shape: torch.Size) -> torch.Tensor:
    """旧两拷贝 glue：head-major finish + transpose-contiguous。"""
    return finish_ulysses_dim1(recv, None, out_shape).transpose(1, 2).contiguous()


class TestFinishUlyssesDim1SeqMajor:
    """``finish_ulysses_dim1_seq_major`` 纯张量契约（CPU，无 dist）。"""

    def test_recv_to_out_element_identity(self) -> None:
        # 元素恒等式 recv[w, a, l] -> out[0, w*L+l, a]；world=1 / A=1 / L=1 边界也在内。
        for world, a_shard, l_loc in ((2, 3, 4), (1, 5, 7), (4, 2, 3), (3, 1, 2)):
            recv = _recv(world, a_shard, l_loc)
            out = finish_ulysses_dim1_seq_major(recv, None)
            assert out.shape == (1, world * l_loc, a_shard), (world, a_shard, l_loc, out.shape)
            for w in range(world):
                for a in range(a_shard):
                    for l in range(l_loc):
                        assert out[0, w * l_loc + l, a] == recv[w, a, l], (world, a_shard, l_loc, w, a, l)

    def test_matches_two_copy_head_major_glue(self) -> None:
        for world, a_shard, l_loc in ((2, 3, 4), (4, 8, 16), (1, 2, 2)):
            recv = _recv(world, a_shard, l_loc)
            out = finish_ulysses_dim1_seq_major(recv, None)
            old = _old_glue(recv, torch.Size((1, a_shard, world * l_loc)))
            assert out.shape == old.shape
            assert torch.equal(out, old), (world, a_shard, l_loc)
            assert out.is_contiguous()
            # 单拷贝产物必须是全新 buffer，不得与 recv 共享存储。
            assert out.data_ptr() != recv.data_ptr()

    def test_gradient_matches_two_copy_glue(self) -> None:
        # 两条路径都是纯数据搬运，反传梯度必须逐位一致。
        torch.manual_seed(0)
        for world, a_shard, l_loc in ((2, 3, 4), (2, 4, 8)):
            shape = (world, a_shard, l_loc)
            out_shape = torch.Size((1, a_shard, world * l_loc))
            w = torch.randn(1, world * l_loc, a_shard)

            recv_new = torch.randn(shape)
            recv_new.requires_grad_(True)
            out = finish_ulysses_dim1_seq_major(recv_new, None)
            out.backward(w)

            recv_old = recv_new.detach().clone()
            recv_old.requires_grad_(True)
            old = _old_glue(recv_old, out_shape)
            old.backward(w)

            assert torch.equal(out.detach(), old.detach())
            assert torch.equal(recv_new.grad, recv_old.grad), (world, a_shard, l_loc)

    def test_fences_work_before_returning(self) -> None:
        # fence 契约：work 未完成时必须 wait()（且只 wait 一次）后才产出可读 buffer。
        class _FakeWork:
            def __init__(self, completed: bool) -> None:
                self.completed = completed
                self.wait_count = 0

            def is_completed(self) -> bool:
                return self.completed

            def wait(self) -> None:
                self.wait_count += 1

        recv = _recv(2, 2, 3)
        pending = _FakeWork(completed=False)
        out = finish_ulysses_dim1_seq_major(recv, pending)  # type: ignore[arg-type]
        assert pending.wait_count == 1
        assert out.shape == (1, 6, 2)

        retired = _FakeWork(completed=True)
        finish_ulysses_dim1_seq_major(recv, retired)  # type: ignore[arg-type]
        assert retired.wait_count == 0


_KDA_HIDDEN = 128
_KDA_HEADS = 4
_KDA_HEAD_DIM = 128
_KDA_CONV_W = 4
_S_LOC = 6


def _build_kda_layer() -> kda_module.KimiDeltaAttention:
    """最小 KDA 层（bf16 权重，CPU），与 test_glm53_kda_sp_async.py 同构。"""
    torch.manual_seed(7)
    module = kda_module.KimiDeltaAttention(
        hidden_size=_KDA_HIDDEN,
        num_heads=_KDA_HEADS,
        head_dim=_KDA_HEAD_DIM,
        conv_kernel_size=_KDA_CONV_W,
    )
    return module.to(torch.bfloat16)


def _sync_a2a_issue(send: torch.Tensor, group: dist.ProcessGroup) -> tuple[torch.Tensor, None]:
    """``ulysses_dispatch._a2a_issue`` 的同步 stub：gloo 上直接 all_to_all_single。"""
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return recv, None


def _hidden_local(rank: int, world: int) -> torch.Tensor:
    s_glob = _S_LOC * world
    hidden_global = torch.randn(1, s_glob, _KDA_HIDDEN, dtype=torch.bfloat16)
    return hidden_global[:, rank * _S_LOC : (rank + 1) * _S_LOC, :].contiguous()


def _case_finish_short_conv_flag(rank: int, world: int) -> None:
    """``_finish_short_conv`` flag 开/关：q 路径 conv 输出与梯度逐位一致。"""
    module = _build_kda_layer()
    s_glob = _S_LOC * world
    cu = [0, s_glob]
    mesh = init_device_mesh("cpu", (world,))
    group = mesh.get_group()
    torch.manual_seed(1234)
    qweight = torch.randn(1, s_glob, module.num_heads * module.head_dim // world)
    hidden_global = torch.randn(1, s_glob, _KDA_HIDDEN, dtype=torch.bfloat16)
    hidden_local = hidden_global[:, rank * _S_LOC : (rank + 1) * _S_LOC, :].contiguous()

    def run_once(flag: bool) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        kda_lever.SP_A2A_DIRECT_CONV = flag
        hidden = hidden_local.clone().requires_grad_(True)
        q_local = module.q_proj(hidden).transpose(1, 2)
        issued = issue_ulysses_dim1(q_local, group)
        q = kda_lever._finish_short_conv(module.q_conv1d, issued, rank, world, cu)
        assert q.shape == (1, s_glob, module.num_heads * module.head_dim // world), q.shape
        (q.float() * qweight).sum().backward()
        proj_grads = [p.grad.clone() for p in module.q_proj.parameters() if p.grad is not None]
        return q.detach().clone(), hidden.grad.clone(), proj_grads

    prev = kda_lever.SP_A2A_DIRECT_CONV
    try:
        on_out, on_hidden_grad, on_proj_grads = run_once(True)
        module.zero_grad(set_to_none=True)
        off_out, off_hidden_grad, off_proj_grads = run_once(False)
    finally:
        kda_lever.SP_A2A_DIRECT_CONV = prev

    assert torch.equal(on_out, off_out), "flag 开/关的 conv 输出不逐位一致"
    assert torch.equal(on_hidden_grad, off_hidden_grad), "flag 开/关的 hidden 梯度不逐位一致"
    for i, (g_on, g_off) in enumerate(zip(on_proj_grads, off_proj_grads)):
        assert torch.equal(g_on, g_off), f"flag 开/关的 q_proj 梯度不逐位一致 (param {i})"


def _case_forward_for_sp_flag(rank: int, world: int) -> None:
    """活跃路径 ``forward_for_sp`` 整链：flag 开/关 projected_output 逐位一致。"""
    module = _build_kda_layer()
    s_glob = _S_LOC * world
    cu = [0, s_glob]
    mesh = init_device_mesh("cpu", (world,))
    seq_ctx = SimpleNamespace(
        sequence_parallel_mesh=mesh,
        sp_rank=rank,
        cu_seq_lens_q=torch.tensor(cu, dtype=torch.int64),
        cu_seq_lens_q_list=cu,
    )
    hidden_local = _hidden_local(rank, world)

    outputs: dict[str, torch.Tensor] = {}
    prev = kda_lever.SP_A2A_DIRECT_CONV
    try:
        for key, flag in (("on", True), ("off", False)):
            kda_lever.SP_A2A_DIRECT_CONV = flag
            out = module.forward_for_sp(hidden_local, seq_ctx)
            assert set(out) == {"raw_output", "projected_output", "softmax_lse"}, set(out)
            assert out["softmax_lse"] is None
            outputs[key] = out["projected_output"].detach().clone()
            assert outputs[key].shape == (1, _S_LOC, _KDA_HIDDEN), outputs[key].shape
    finally:
        kda_lever.SP_A2A_DIRECT_CONV = prev

    assert torch.equal(outputs["on"], outputs["off"]), "forward_for_sp flag 开/关输出不逐位一致"


_CASES = {
    "finish_short_conv_flag": _case_finish_short_conv_flag,
    "forward_for_sp_flag": _case_forward_for_sp_flag,
}


def _worker(rank: int, world: int, store_path: str, cases: list[str]) -> None:
    dist.init_process_group("gloo", store=dist.FileStore(store_path, world), rank=rank, world_size=world)
    torch.set_num_threads(1)
    ulysses_dispatch._a2a_issue = _sync_a2a_issue
    try:
        for case in cases:
            _CASES[case](rank, world)
    finally:
        dist.destroy_process_group()


def _run_spawned(cases: list[str], tmp_path: Path) -> None:
    # NPU 盒上 transfer_to_npu 会把 gloo init 重映射为 hccl；连续 spawn 轮次会与前一轮
    # 异步通信子的拆除竞争，前后各留 2s 让其落地（与 test_glm53_kda_sp_async.py 同款）。
    time.sleep(2.0)
    mp.spawn(_worker, args=(2, str(tmp_path / "gloo_store"), cases), nprocs=2, join=True)
    time.sleep(2.0)


class TestFinishShortConvDirectConvFlag:
    """``XTUNER_KDA_SP_A2A_DIRECT_CONV`` 开/关等价（2 进程 gloo spawn，CPU 张量）。"""

    def test_finish_short_conv_flag_on_matches_off(self, tmp_path: Path) -> None:
        _run_spawned(["finish_short_conv_flag"], tmp_path)

    def test_forward_for_sp_flag_on_matches_off(self, tmp_path: Path) -> None:
        _run_spawned(["forward_for_sp_flag"], tmp_path)
