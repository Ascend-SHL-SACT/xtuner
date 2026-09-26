# Copyright © 2026 Huawei Technologies Co., Ltd.
"""``SequenceContext.split()``（Ulysses 序列并行 pad）新建 cu 张量的设备驻留回归测试。

回归背景：当 (1) ``sequence_parallel_mesh`` 非空、(2) 序列总长不整除 sp 世界大小需要补
padding（``new_padding > 0``）、(3) 之前无 padding（``num_padding == 0``，走新建 cu 张量的
else 分支）三者同时满足时，旧实现用 ``device=self.device``（模型设备字符串，如 "cuda"）新建
``new_cu_seq_lens``，而 collator 构造的 ``cu_seq_lens_q`` 实际驻留 host —— cu 的设备被静默
翻转，下游假设 host 驻留的消费者（mtp roll、``cu_seq_lens_q_list``、
``kda.py::_causal_conv1d_varlen(cu_seq_lens_q: list[int])``）报错或引入热路径 device→host
同步。修复改为 ``device=self.cu_seq_lens_q.device``：新 cu 跟随原 cu 的驻留设备。
注意 split() 把同一个 ``new_cu_seq_lens`` 同时传给 ``cu_seq_lens_q`` 与 ``cu_seq_lens_k``，
故两侧一并断言。

复现构造（纯 CPU 机器即可）：不走 ``from_input_ids``（它会把 cu ``.to(device)`` 提前触
cuda），直接构造 ``SequenceContext`` —— ``input_ids``/``cu_seq_lens_q``/``cu_seq_lens_k``/
``position_ids`` 全 CPU，``device="cuda"`` 仅作为字符串字段（host 计算不应触及），
``num_padding=0``，2-rank gloo CPU DeviceMesh；seq_len=3 不整除 2 触发 else 分支。旧码在
``torch.ones(..., device="cuda")`` 处抛 ``AssertionError: Torch not compiled with CUDA
enabled``（CPU-only 构建），经 ``mp.spawn`` 传回父进程使测试 FAIL；修复后新 cu 留在 CPU，
测试 PASS。分布式初始化用 ``dist.FileStore``（gloo，无端口竞争），无 NPU/GPU/外部服务。

TestSequenceContextSplitPadDevice
    test_new_cu_stays_on_host_when_first_padding: 首次补 padding（新建 cu 的 else 分支）后
        q/k 两侧 cu 仍驻留 CPU、dtype int32、前缀保留原值且末元素 = 原末元素 + new_padding，
        host list 视图（``cu_seq_lens_q_list``）同步；原 ctx 的 cu 未被原地修改；各 rank 的
        input_ids 分片与全局 pad 后张量的切片一致。
    test_existing_padding_clone_keeps_host_device: 既有 padding（clone 分支，本次 bug 未
        触及的相邻路径）同样保持 CPU 驻留与末元素累加语义，固化两分支一致的设备契约。
"""

from collections.abc import Callable
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

from xtuner.v1.data_proto.sequence_context import SequenceContext


_WORLD = 2
_SEQ_LEN = 3  # 不整除 _WORLD，触发 split() 的 new_padding > 0 分支


class TestSequenceContextSplitPadDevice:
    """split() 补 padding 时新建/克隆 cu 张量的设备驻留与数值契约（2-rank gloo CPU）。"""

    def test_new_cu_stays_on_host_when_first_padding(self, tmp_path: Path) -> None:
        """首次补 padding（新建 cu 的 else 分支）后 cu 保持 host 驻留且数值正确。"""
        _run_spawned("first_padding", _WORLD, tmp_path)

    def test_existing_padding_clone_keeps_host_device(self, tmp_path: Path) -> None:
        """既有 padding（clone 分支）同样保持 host 驻留，固化两分支一致的设备契约。"""
        _run_spawned("existing_padding", _WORLD, tmp_path)


def _first_padding_case(rank: int, world: int) -> None:
    """首补 padding 用例：新建 cu 必须留在原 cu 的设备（host）上。

    Args:
        rank (int): 当前进程在 sp mesh 中的 rank。
        world (int): sp mesh 世界大小。
    """
    mesh = init_device_mesh("cpu", (world,))
    cu = torch.tensor([0, 2, 3], dtype=torch.int32)
    seq_ctx = SequenceContext(
        input_ids=torch.arange(_SEQ_LEN, dtype=torch.long).unsqueeze(0),
        cu_seq_lens_q=cu.clone(),
        cu_seq_lens_k=cu.clone(),
        max_length_q=2,
        max_length_k=2,
        num_padding=0,
        sequence_parallel_mesh=mesh,
        device="cuda",  # 纯字符串字段：host 计算不应触及；旧码在此设备上新建 cu 即炸
        position_ids=torch.arange(_SEQ_LEN, dtype=torch.long).unsqueeze(0),
    )
    sp_ctx = seq_ctx.split()

    assert sp_ctx is not seq_ctx, "sp mesh 非空时 split() 应返回新的 SequenceContext"
    # seq_len=3 pad 到 4（new_padding=1）：新 cu 前缀保留原值，末元素 = 3 + 1 = 4。
    expected = [0, 2, 3, 4]
    for name in ("cu_seq_lens_q", "cu_seq_lens_k"):
        cu_tensor = getattr(sp_ctx, name)
        assert cu_tensor.device.type == "cpu", f"{name} 设备被翻转为 {cu_tensor.device}"
        assert cu_tensor.dtype == torch.int32, f"{name} dtype 变为 {cu_tensor.dtype}"
        assert cu_tensor.tolist() == expected, f"{name} 数值错误: {cu_tensor.tolist()}"
    assert sp_ctx.cu_seq_lens_q_list == expected, "host list 视图应与新 cu 同步"
    assert sp_ctx.cu_seq_lens_k_list == expected, "k 侧 host list 视图应与新 cu 同步"
    assert seq_ctx.cu_seq_lens_q.tolist() == [0, 2, 3], "split() 不得原地修改原 cu"

    pad_input_ids = torch.tensor([[0, 1, 2, 0]], dtype=torch.long)
    shard = -(-_SEQ_LEN // world)  # ceil(seq_len/world)，即 split_for_sequence_parallel 的 split_size
    local_ids = pad_input_ids[:, rank * shard : (rank + 1) * shard]
    assert torch.equal(sp_ctx.input_ids, local_ids), f"rank {rank} 的 input_ids 分片与全局 pad 后切片不一致"


def _existing_padding_case(rank: int, world: int) -> None:
    """既有 padding 用例：clone 分支不翻转设备，末元素累加 new_padding。

    Args:
        rank (int): 当前进程在 sp mesh 中的 rank（clone 分支不切片，仅保持签名统一）。
        world (int): sp mesh 世界大小。
    """
    mesh = init_device_mesh("cpu", (world,))
    cu = torch.tensor([0, 1, 3], dtype=torch.int32)  # 1 个真实 token + 1 个既有 padding
    seq_ctx = SequenceContext(
        input_ids=torch.arange(_SEQ_LEN, dtype=torch.long).unsqueeze(0),
        cu_seq_lens_q=cu.clone(),
        cu_seq_lens_k=cu.clone(),
        max_length_q=2,
        max_length_k=2,
        num_padding=1,
        sequence_parallel_mesh=mesh,
        device="cuda",
        position_ids=torch.arange(_SEQ_LEN, dtype=torch.long).unsqueeze(0),
    )
    sp_ctx = seq_ctx.split()

    expected = [0, 1, 4]  # 末元素 3 + new_padding 1，其余元素保持
    for name in ("cu_seq_lens_q", "cu_seq_lens_k"):
        cu_tensor = getattr(sp_ctx, name)
        assert cu_tensor.device.type == "cpu", f"{name} 设备被翻转为 {cu_tensor.device}"
        assert cu_tensor.dtype == torch.int32, f"{name} dtype 变为 {cu_tensor.dtype}"
        assert cu_tensor.tolist() == expected, f"{name} 数值错误: {cu_tensor.tolist()}"


_CASES: dict[str, Callable[[int, int], None]] = {
    "first_padding": _first_padding_case,
    "existing_padding": _existing_padding_case,
}


def _worker(rank: int, world: int, store_path: str, case: str) -> None:
    dist.init_process_group("gloo", store=dist.FileStore(store_path, world), rank=rank, world_size=world)
    try:
        _CASES[case](rank, world)
    finally:
        dist.destroy_process_group()


def _run_spawned(case: str, world: int, tmp_path: Path) -> None:
    mp.spawn(_worker, args=(world, str(tmp_path / "gloo_store"), case), nprocs=world, join=True)
