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

roll 路径回归（``use_local_future_embeddings=False`` 或 ``inputs_embeds is None`` 时，
``mtp_block._next_future_inputs`` 逐深度调用 ``roll_sequence_context``）：每 depth 的
rolled ctx 上挂着整包 rolled 张量（``_raw_inputs_embeds``）及其 SP 分片视图
（``inputs_embeds``），且 rolled ctx 同样作为非张量入参被 checkpoint 节点捕获——比 gather
路径更重（每 depth 一份整包，生存期同样到步末）。张量入参本身走 ``save_for_backward``，
节点 backward 后即释放，不构成步末钉住；钉住通道只有非张量 rolled ctx。修复：
``mtp_block`` 深度循环后对每个 rolled ctx 调 ``release_rolled_future_inputs()``。

TestSequenceContextReleaseRolledFutureInputs
    test_release_drops_rolled_inputs: 剥离缓存与视图；视图清掉后整包 base 一并释放。
TestCheckpointPinnedRolledContextReleasesFullPackInputs
    test_checkpoint_pinned_rolled_context_releases_full_pack_inputs: roll 路径拓扑
    （整包 base + 分片视图 + future 视图作显式张量入参），按生产顺序 forward → 剥离 →
    backward；断言剥离后整包 base 不再被节点钉住，且 replay 数值与未 checkpoint 参考
    bitwise 一致（证明剥离不影响重放）。
"""

import copy
import unittest
import unittest.mock
import weakref

import torch
import torch.nn as nn

from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.model.utils import apply_activation_checkpointing
from xtuner.v1.module.mtp import MTPBlock, MTPConfig
from xtuner.v1.module.mtp import mtp_block as mtp_block_mod


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


class TestSequenceContextReleaseRolledFutureInputs(unittest.TestCase):
    def test_release_drops_rolled_inputs(self) -> None:
        ctx = _make_ctx()
        raw_embeds = torch.ones(1, 8, 4)  # roll_sequence_context 产出的整包 rolled 张量占位
        raw_ids = torch.zeros(1, 8, dtype=torch.long)
        ctx._raw_inputs_embeds = raw_embeds
        ctx._raw_input_ids = raw_ids
        ctx.inputs_embeds = raw_embeds[:, 2:6]  # SP 分支写入的分片视图
        ctx.input_ids = raw_ids[:, 2:6]
        embeds_ref = weakref.ref(raw_embeds)
        del raw_embeds, raw_ids

        ctx.release_rolled_future_inputs()

        self.assertIsNone(ctx._raw_inputs_embeds)
        self.assertIsNone(ctx._raw_input_ids)
        self.assertIsNone(ctx.inputs_embeds)
        self.assertIsNone(ctx.input_ids)
        # 分片视图不剥离的话整包 base 仍被视图钉住：视图清掉后 base 一并释放。
        self.assertIsNone(embeds_ref())


class _RollMtpLayer(nn.Module):
    """模拟 MTP 层：消费 future_embeddings 张量入参 + seq_ctx 非张量入参（与生产拓扑一致）。"""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(
        self,
        hidden_states: torch.Tensor,
        future_embeddings: torch.Tensor,
        seq_ctx: SequenceContext | None,
    ) -> torch.Tensor:
        return self.lin(hidden_states + future_embeddings)


class TestCheckpointPinnedRolledContextReleasesFullPackInputs(unittest.TestCase):
    def test_checkpoint_pinned_rolled_context_releases_full_pack_inputs(self) -> None:
        parent = _make_ctx()
        # roll 路径拓扑：rolled ctx 挂整包 rolled base + 其 SP 分片视图（模拟
        # roll_sequence_context 的 SP 分支写入）。
        raw_embeds = torch.ones(1, 8, 4) * 3
        raw_ids = torch.zeros(1, 8, dtype=torch.long)
        rolled = parent.copy(
            raw_input_ids=raw_ids,
            raw_inputs_embeds=raw_embeds,
            input_ids=raw_ids[:, 2:6],
            inputs_embeds=raw_embeds[:, 2:6],
        )
        embeds_ref = weakref.ref(raw_embeds)
        del raw_embeds, raw_ids  # 此后持有方应为 rolled ctx（及其后被捕获的 checkpoint 节点）与视图

        ref = _RollMtpLayer()
        layer = apply_activation_checkpointing(copy.deepcopy(ref))
        # hidden 与 future 同为分片形状（生产中 layer 只见本地分片）；整包 base 才是被钉住的。
        hidden_ckpt = torch.randn(1, 4, 4, requires_grad=True)
        future_ref = rolled.inputs_embeds.detach().clone()  # 参考实现的 future 输入（自有拷贝）
        future = rolled.inputs_embeds  # 生产中 mtp_block 传给层的 future_embeddings 视图
        loss = layer(hidden_ckpt, future_embeddings=future, seq_ctx=rolled).sum()
        del future  # 生产拓扑：mtp_block 返回后 future 局部变量消亡
        # 修复调用点（mtp_block 深度循环后）：解除整包 base 的钉住。
        rolled.release_rolled_future_inputs()
        loss.backward()

        self.assertIsNone(embeds_ref())
        self.assertIsNone(rolled.inputs_embeds)
        self.assertIsNone(rolled.input_ids)
        self.assertIsNone(rolled._raw_inputs_embeds)
        self.assertIsNone(rolled._raw_input_ids)

        # 剥离不得破坏 checkpoint replay：与未 checkpoint 的参考实现 bitwise 一致。
        hidden_ref = hidden_ckpt.detach().clone().requires_grad_(True)  # 同为分片形状 (1, 4, 4)
        ref(hidden_ref, future_embeddings=future_ref, seq_ctx=None).sum().backward()
        self.assertTrue(torch.equal(hidden_ckpt.grad, hidden_ref.grad))
        for ckpt_param, ref_param in zip(layer.parameters(), ref.parameters()):
            self.assertTrue(torch.equal(ckpt_param.grad, ref_param.grad))


class _FakeMtpDecoderLayer(nn.Module):
    """duck-typing 替身：签名与 MoEDecoderLayer.forward 的块内调用约定一致。"""

    def forward(self, hidden_states, *, future_embeddings, position_embeddings, seq_ctx):
        if isinstance(hidden_states, list):
            n = len(hidden_states)
            return {
                "hidden_states": [h + f for h, f in zip(hidden_states, future_embeddings)],
                "router_logits": [None] * n,
                "router_weights": [None] * n,
                "router_topk_ids": [None] * n,
            }
        return {
            "hidden_states": hidden_states + future_embeddings,
            "router_logits": None,
            "router_weights": None,
            "router_topk_ids": None,
        }


class TestMtpBlockReleasesRolledContexts(unittest.TestCase):
    def _make_block_and_ctx(self) -> tuple[MTPBlock, SequenceContext]:
        block = MTPBlock(
            mtp_config=MTPConfig(num_layers=2, use_local_future_embeddings=False),
            mtp_layers=[_FakeMtpDecoderLayer(), _FakeMtpDecoderLayer()],
        )
        ctx = _make_ctx()
        ctx.inputs_embeds = torch.randn(1, 8, 4)
        return block, ctx

    def _record_rolled_contexts(self) -> list[SequenceContext]:
        created: list[SequenceContext] = []
        orig_roll = mtp_block_mod.roll_sequence_context

        def recording_roll(seq_ctx: SequenceContext, shifts: int = -1) -> SequenceContext:
            rolled = orig_roll(seq_ctx, shifts)
            created.append(rolled)
            return rolled

        self._patcher = unittest.mock.patch.object(mtp_block_mod, "roll_sequence_context", recording_roll)
        self._patcher.start()
        self.addCleanup(self._patcher.stop)
        return created

    def test_mtp_block_strips_rolled_contexts_after_forward(self) -> None:
        position_embeddings = (torch.zeros(1, 8, 4), torch.zeros(1, 8, 4))
        embed_tokens_fn = lambda token_ids: torch.zeros(token_ids.shape, dtype=torch.float)  # noqa: E731

        # 单微批前向：2 depth ⇒ 2 个 rolled ctx，块返回后全部剥离。
        block, ctx = self._make_block_and_ctx()
        created = self._record_rolled_contexts()
        outputs = block(
            torch.randn(1, 8, 4),
            embed_tokens_fn=embed_tokens_fn,
            position_embeddings=position_embeddings,
            seq_ctx=ctx,
        )
        self.assertEqual(len(outputs), 2)
        self.assertEqual(len(created), 2)
        for rolled_ctx in created:
            self.assertIsNone(rolled_ctx._raw_inputs_embeds)
            self.assertIsNone(rolled_ctx._raw_input_ids)
            self.assertIsNone(rolled_ctx.inputs_embeds)
            self.assertIsNone(rolled_ctx.input_ids)
        self.assertIsNotNone(ctx.inputs_embeds)  # 父 ctx 不受剥离影响

        # 多微批前向：2 depth × 2 微批 ⇒ 4 个 rolled ctx，同样全部剥离。
        created.clear()
        ctx_list = [ctx.copy(inputs_embeds=torch.randn(1, 8, 4)) for _ in range(2)]
        outputs_per_mb = block(
            [torch.randn(1, 8, 4), torch.randn(1, 8, 4)],
            embed_tokens_fn=embed_tokens_fn,
            position_embeddings=[position_embeddings, position_embeddings],
            seq_ctx=ctx_list,
        )
        self.assertEqual(len(outputs_per_mb), 2)
        self.assertTrue(all(len(depths) == 2 for depths in outputs_per_mb))
        self.assertEqual(len(created), 4)
        for rolled_ctx in created:
            self.assertIsNone(rolled_ctx._raw_inputs_embeds)
            self.assertIsNone(rolled_ctx.inputs_embeds)


if __name__ == "__main__":
    unittest.main()
