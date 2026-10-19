# Copyright © 2026 Huawei Technologies Co., Ltd.
"""MTP 整包 gather 缓存被 checkpoint 捕获钉住到步末的回归测试。

回归背景：``SequenceContext.raw_inputs_embeds`` 在 SP 模式下首次访问会跨 SP mesh 做
``gather_dim=1`` 的整包 embedding all-gather（形状 ``(1, pack_len, hidden)``，64K pack +
hidden 4096 时恰 512 MiB/个），并把结果缓存在 ctx 的 ``_raw_inputs_embeds`` 上。MTP 前向
（``moe.py``）在 ``seq_ctx.copy(...)`` 出来的**克隆**上触发该 gather，随后把克隆作为非张量
入参传给 MTP 层；MTP 层被 xtuner 的 reentrant activation checkpointing 包裹
（``apply_activation_checkpointing``），torch 的 ``CheckpointFunction`` 会把非张量入参按引用
存进 ``ctx.inputs`` 且 backward 不清理——于是克隆连同其上的整包缓存一起被 autograd 节点
钉住，直到持有梯度图根的 ``micro_batch_results`` 在 train step 结束时才消亡。实测每
micro-batch 泄漏一个整包张量到步末、步末单点集中释放（SP128+64K +16 GiB 峰值，128K pack
组合 +32 GiB，是那批组合 OOM 的主力贡献项）。

修复：``moe.py`` 在 ``mtp_block`` 返回后（各深度 future 输入均已物化为显式拷贝、checkpoint
replay 重放层调用不再需要整包 base）对克隆调用 ``release_raw_caches()``。

复现构造（纯 CPU，无需 NPU/权重/mesh）：
- 用真实的 ``apply_activation_checkpointing``（内部走 ``torch.utils.checkpoint`` reentrant
  路径）包一层带参数的假 MTP 层，把带缓存写入的克隆作为 ``seq_ctx`` kwarg 传入——与生产
  的入参拓扑一致（克隆是非张量叶子，被 ``ctx.inputs`` 按引用捕获）。
- 用 ``weakref`` 观测缓存张量：forward 后仍存活（节点捕获克隆=复现钉住）、backward 后仍存活
  （saved tensors 释放但节点存活到图根消亡，生产中即步末——bug 复现点）、
  ``release_raw_caches()`` 后立即消亡（修复生效）。

TestSequenceContextReleaseRawCaches
    test_release_drops_raw_caches: 释放后两个缓存属性为 None，且无 mesh 时 property 重新
        访问直接回落到本地 ``inputs_embeds``（可再导出，无残留引用）。
TestCheckpointPinnedMtpCloneReleasesRawCaches
    test_checkpoint_pinned_clone_releases_raw_caches: 见上，三段式断言。
"""

import unittest
import weakref

import torch
import torch.nn as nn

from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.model.utils import apply_activation_checkpointing


def _make_ctx() -> SequenceContext:
    return SequenceContext(
        input_ids=torch.zeros(1, 8, dtype=torch.long),
        cu_seq_lens_q=torch.tensor([0, 8], dtype=torch.int32),
        cu_seq_lens_k=torch.tensor([0, 8], dtype=torch.int32),
        max_length_q=8,
        max_length_k=8,
        device="cpu",
    )


class TestSequenceContextReleaseRawCaches(unittest.TestCase):
    def test_release_drops_raw_caches(self) -> None:
        ctx = _make_ctx()
        ctx.inputs_embeds = torch.zeros(1, 8, 4)
        raw_ids = torch.zeros(1, 8, dtype=torch.long)
        raw_embeds = torch.zeros(1, 8, 4)
        ctx._raw_input_ids = raw_ids
        ctx._raw_inputs_embeds = raw_embeds

        ctx.release_raw_caches()

        self.assertIsNone(ctx._raw_input_ids)
        self.assertIsNone(ctx._raw_inputs_embeds)
        # 无 mesh 时 property 直接回落到本地 inputs_embeds：缓存可再导出，不残留已释放引用。
        self.assertIs(ctx.raw_inputs_embeds, ctx.inputs_embeds)
        self.assertIs(ctx.raw_input_ids, ctx.input_ids)


class _MtpLayer(nn.Module):
    """模拟被 checkpoint 包裹的 MTP 层：hidden_states 张量入参 + seq_ctx 非张量入参。"""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, hidden_states: torch.Tensor, seq_ctx: SequenceContext) -> torch.Tensor:
        return self.lin(hidden_states)


class TestCheckpointPinnedMtpCloneReleasesRawCaches(unittest.TestCase):
    def test_checkpoint_pinned_clone_releases_raw_caches(self) -> None:
        parent = _make_ctx()
        parent.inputs_embeds = torch.randn(1, 8, 4)
        # 生产拓扑：moe.py 对父 ctx 做 copy() 并把整包 gather 缓存写到克隆上。
        clone = parent.copy(inputs_embeds=parent.inputs_embeds.clone())
        base = torch.ones(1, 8, 4) * 3  # 等价于 512 MiB 整包 gather 输出的占位
        clone._raw_inputs_embeds = base
        base_ref = weakref.ref(base)
        del base  # 此后唯一持有方应为克隆缓存（及其后被捕获的 checkpoint 节点）

        layer = apply_activation_checkpointing(_MtpLayer())
        hidden_states = torch.randn(1, 8, 4, requires_grad=True)  # 等价于上层传来的梯度激活
        loss = layer(hidden_states, seq_ctx=clone).sum()
        # reentrant checkpoint 把非张量入参（克隆）按引用存进 ctx.inputs：缓存被钉住。
        self.assertIsNotNone(base_ref())
        loss.backward()
        # backward 释放 saved tensors，但节点仍持有克隆 → 缓存活到图根消亡（生产中即步末）。
        self.assertIsNotNone(base_ref())

        # 修复调用点（moe.py 在 mtp_block 返回后）：解除钉住。
        clone.release_raw_caches()
        self.assertIsNone(base_ref())


if __name__ == "__main__":
    unittest.main()
