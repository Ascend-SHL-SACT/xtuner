"""2-rank NPU A/B for the fused split D2H under ``XTUNER_MOE_META_A2A_ASYNC``.

Spawns 2 hccl ranks and, on identical inputs per step, asserts the fused path
(flag=1) produces bit-identical splits and counts to the legacy path (flag=0),
and that the a2a metadata stays cross-rank consistent (rank r's
``input_splits[dst]`` equals rank dst's ``output_splits[r]``). Rank 0 drives
the default stream, rank 1 a ``comm_stream`` context -- mirroring the two
``_dispatch`` call paths (sync dispatch and ``_AsyncDispatch.forward``).
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import xtuner.v1.module.dispatcher.torch_all2all as t_a2a


def _worker(rank: int, world: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world)
    dev = torch.device(f"npu:{rank}")
    torch.npu.set_device(dev)
    group = dist.group.WORLD
    ep_size = world
    n_experts = 8 * ep_size
    n_tokens, topk = 256, 4
    comm_stream = torch.cuda.Stream(device=dev)
    try:
        for step in range(3):
            torch.manual_seed(5000 + step)  # identical ids on all ranks
            ids = torch.randint(0, n_experts, (n_tokens, topk), dtype=torch.int64, device=dev)
            hidden = torch.randn(n_tokens * topk, 128, dtype=torch.bfloat16, device=dev)

            def run(flag_val: bool):
                t_a2a._META_A2A_ASYNC = flag_val
                if rank == 1:
                    with torch.cuda.stream(comm_stream):
                        res = t_a2a._dispatch(hidden, ids, n_experts, group)
                    torch.cuda.current_stream().wait_stream(comm_stream)
                else:
                    res = t_a2a._dispatch(hidden, ids, n_experts, group)
                return res

            out_f, tpeg_f, in_f, outf = run(True)
            out_l, tpeg_l, in_l, outl = run(False)
            t_a2a._META_A2A_ASYNC = True

            assert in_f == in_l, f"r{rank} step{step} input_splits differ: {in_f} vs {in_l}"
            assert outf == outl, f"r{rank} step{step} output_splits differ: {outf} vs {outl}"
            assert sum(in_f) == n_tokens * topk, f"r{rank} step{step} splits sum {sum(in_f)}"
            assert torch.equal(tpeg_f.cpu(), tpeg_l.cpu()), f"r{rank} step{step} counts differ"

            gathered: list = [None] * world
            dist.all_gather_object(gathered, (in_f, outf), group=group)
            for r in range(world):
                for d in range(world):
                    assert gathered[r][0][d] == gathered[d][1][r], f"r{rank} step{step} cross-rank {r}->{d}"

        buf = t_a2a._PINNED_SPLITS_BUFS.bufs[ep_size]
        assert buf.dtype == torch.int64 and buf.shape == (2, ep_size)
        assert buf.is_pinned() and buf[0].is_pinned() and buf[1].is_pinned()
    finally:
        dist.destroy_process_group()


@pytest.mark.gpu
class TestTorchAll2AllD2HFused:
    def test_fused_matches_legacy_2rank_bitwise(self) -> None:
        """2 rank 上 fused 与 legacy 的 splits 逐位一致 + 跨 rank 元数据一致。"""
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        mp.spawn(_worker, args=(2, port), nprocs=2, join=True)
