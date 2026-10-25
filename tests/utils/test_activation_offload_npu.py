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
                head_entry = offload_mod._layer_offloads[("chain_head_toy", 1)][0]
                assert [swap.stat for swap in head_entry] == ["device"]
                assert [swap.stat for swap in offload_mod._layer_offloads[("chain_head_toy", 0)][0]] == ["host"]
                out.pow(2).sum().backward()
            assert offload_mod._layer_offloads == {}
        finally:
            offload_mod._layer_offloads.clear()
            offload_mod._chain_heads.pop("chain_head_toy", None)


@pytest.mark.gpu
class TestActivationOffloadNpuMultiPass:
    """Two passes of one chain can be in flight on the same layer at once: the vision tower runs once
    per sample of an ``intra_layer_micro_batch > 1`` iteration (plus a dummy call for a text-only
    sample), so both passes park every block before either reaches its backward. Parking the second
    pass used to evict the first pass's entry (and reuse its pinned buffer), and the first pass's
    replay then read zero-sized storage. Lock the per-pass slots, per-pass buffers and the
    identity-selected refill."""

    def test_two_passes_in_flight_replay_their_own_data(self):
        import torch._dynamo

        from xtuner.v1.model.utils.checkpointing import apply_activation_checkpointing

        def build_chain() -> torch.nn.ModuleList:
            torch.manual_seed(0)
            return torch.nn.ModuleList([_toy_block() for _ in range(3)])

        plain = build_chain()
        offloaded = build_chain()
        offloaded.load_state_dict(plain.state_dict())
        offload_mod.register_activation_offload_npu_hooks(
            model_layers=offloaded, first_offload_layer=0, chain_id="multi_pass_toy"
        )
        try:
            for block_idx in range(2):
                offloaded[block_idx] = apply_activation_checkpointing(offloaded[block_idx], preserve_rng_state=False)

            def run(blocks: torch.nn.ModuleList, xs: list[torch.Tensor]) -> list[torch.Tensor]:
                outs = []
                for x in xs:
                    for block in blocks:
                        x = block(x)
                    outs.append(x)
                return outs

            # Same shape, different values: a refill that picked the wrong pass would show up as a
            # gradient mismatch, not as a crash.
            xs = [torch.randn(4, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(2)]
            ref_xs = [x.detach().clone().requires_grad_(True) for x in xs]
            with mock.patch.object(torch._dynamo.config, "disable", True):
                outs = run(offloaded, xs)  # both passes forward before either backward
                bucket = offload_mod._layer_offloads[("multi_pass_toy", 0)]
                assert len(bucket) == 2
                assert bucket[0][0].tensor_cpu.data_ptr() != bucket[1][0].tensor_cpu.data_ptr()
                sum(out.pow(2).sum() for out in outs).backward()

                ref_outs = run(plain, ref_xs)
                sum(out.pow(2).sum() for out in ref_outs).backward()
            assert offload_mod._layer_offloads == {}
            for got, want in zip(xs, ref_xs, strict=True):
                assert torch.equal(got.grad, want.grad)
            # Same construction order; the checkpoint wrapper renames the offloaded chain's params.
            for param_idx, (got, want) in enumerate(zip(offloaded.parameters(), plain.parameters(), strict=True)):
                assert torch.equal(got.grad, want.grad), param_idx
        finally:
            offload_mod._layer_offloads.clear()
            offload_mod._chain_heads.pop("multi_pass_toy", None)


@pytest.mark.gpu
class TestActivationOffloadNpuAbandonedPasses:
    """A pass that never reaches its backward -- a ``TrainEngine.forward_only`` call or a step aborted
    between forward and backward -- cannot be dropped at the next park without breaking that pass's
    own replay, so the per-layer pass list is bounded and the oldest pass is abandoned past the bound.
    Without the bound, both the registry and the pinned buffers (keyed by pass index) would grow once
    per such call."""

    def test_repeated_forward_only_passes_stay_bounded(self):
        import torch._dynamo

        from xtuner.v1.model.utils.checkpointing import apply_activation_checkpointing

        blocks = torch.nn.ModuleList([_toy_block() for _ in range(3)])
        offload_mod.register_activation_offload_npu_hooks(
            model_layers=blocks, first_offload_layer=0, chain_id="abandoned_toy"
        )
        try:
            for block_idx in range(2):
                blocks[block_idx] = apply_activation_checkpointing(blocks[block_idx], preserve_rng_state=False)

            def run(x: torch.Tensor) -> torch.Tensor:
                for block in blocks:
                    x = block(x)
                return x

            with mock.patch.object(torch._dynamo.config, "disable", True):
                with mock.patch.object(offload_mod.log_rank0, "warning") as warning:
                    passes: list[list] = []
                    for _ in range(offload_mod._MAX_INFLIGHT_PASSES + 3):
                        with torch.no_grad():
                            run(torch.randn(4, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True))
                        passes = offload_mod._layer_offloads[("abandoned_toy", 0)]
                        assert len(passes) <= offload_mod._MAX_INFLIGHT_PASSES
                # The bound was reached, the oldest pass was abandoned, and the pass numbering reused
                # the freed index, so the pinned-buffer cache keys stay in the same small key space.
                assert warning.called
                assert len(passes) == offload_mod._MAX_INFLIGHT_PASSES
                assert passes[-1][0].key == f"abandoned_toy_act_hidden_0_{offload_mod._MAX_INFLIGHT_PASSES - 1}_0"
        finally:
            offload_mod._layer_offloads.clear()
            offload_mod._chain_heads.pop("abandoned_toy", None)
