"""Tests for the hook-based activation (hidden-state) offload on NPU."""

from unittest import mock

import pytest
import torch

from xtuner.v1.utils import activation_offload_npu as offload_mod


def _toy_block() -> torch.nn.Module:
    return torch.nn.Linear(8, 8, bias=False).cuda().to(torch.bfloat16)


class TestActivationOffloadNpuRegistration:
    def test_registering_the_same_chain_twice_raises(self):
        # 重复注册会在每层叠第二对 park/refill 钩子（每步两次 D2H，且 pop 会把先注册的条目
        # 静默顶掉），因此必须在挂第一个钩子之前就拒绝。
        blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4, bias=False) for _ in range(2)])
        offload_mod.register_activation_offload_npu_hooks(
            model_layers=blocks, first_offload_layer=0, chain_id="register_twice"
        )
        try:
            with pytest.raises(RuntimeError, match="already registered"):
                offload_mod.register_activation_offload_npu_hooks(
                    model_layers=blocks, first_offload_layer=0, chain_id="register_twice"
                )
        finally:
            # 钩子留在模块上不影响其他用例（模块随用例回收），只需清掉模块级注册表。
            offload_mod._chain_heads.pop("register_twice", None)
            offload_mod._layer_offloads.clear()


@pytest.mark.gpu
class TestActivationOffloadNpuChainHead:
    """链头不能只取"注册集合的最大层"：checkpoint 只包 recompute 范围内的层，最高被包裹层在无 MTP
    时不是最后一层（moe.py `_should_recompute` 的最后层豁免），若链头取注册最大值，最高被回放层
    每步都走阻塞回退（冷回填 + 拷贝流排空）。这里用 toy 链锁住"未包裹层在原始前向里替链头发起
    prefetch"这一运行时判定。"""

    def test_unwrapped_layer_above_recompute_cut_prefetches_the_head(self):
        import torch._dynamo

        from xtuner.v1.model.utils.checkpointing import apply_activation_checkpointing

        blocks = torch.nn.ModuleList([_toy_block() for _ in range(3)])
        offload_mod.register_activation_offload_npu_hooks(
            model_layers=blocks, first_offload_layer=0, chain_id="chain_head_toy"
        )
        try:
            # 无 MTP 形态：只有 0/1 被包裹，最后一层（链头注册值）不参与 checkpoint。
            blocks[0] = apply_activation_checkpointing(blocks[0], preserve_rng_state=False)
            blocks[1] = apply_activation_checkpointing(blocks[1], preserve_rng_state=False)
            assert offload_mod._chain_heads["chain_head_toy"] == 2

            def run(x):
                for block in blocks:
                    x = block(x)
                return x

            with mock.patch.object(torch._dynamo.config, "disable", True):
                x = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                out = run(x)
                # 最高被包裹层（1）在未包裹层（2）的前向预钩子里已被回填：D2H 之后又发起了 H2D，
                # stat 从 "host" 变为 "device"，反向 replay 入口只剩事件等待，不再冷回填。
                head_entry = offload_mod._layer_offloads[("chain_head_toy", 1)]
                assert [swap.stat for swap in head_entry] == ["device"]
                assert [swap.stat for swap in offload_mod._layer_offloads[("chain_head_toy", 0)]] == ["host"]
                out.pow(2).sum().backward()
            assert offload_mod._layer_offloads == {}
        finally:
            offload_mod._layer_offloads.clear()
            offload_mod._chain_heads.pop("chain_head_toy", None)
