# Copyright (c) OpenMMLab. All rights reserved.
"""Unit tests for the env-gated dead-chain strip helpers."""

import os

import torch

from xtuner.v1.utils.dead_chain import (
    accumulate_experts_only,
    dead_chain_strip_enabled,
    empty_router_tensor,
)


class _RecordAccumulator:
    """Minimal stand-in for ``AuxLossContext`` recording the accumulate call."""

    def __init__(self) -> None:
        self.kwargs: dict = {}

    def accumulate(self, **kwargs):
        self.kwargs = kwargs
        return kwargs["hidden_states"]


class TestDeadChainStripEnabled:
    def test_env_off_by_default(self) -> None:
        dead_chain_strip_enabled.cache_clear()
        os.environ.pop("XTUNER_DEAD_CHAIN_STRIP", None)
        assert dead_chain_strip_enabled() is False

    def test_env_on(self) -> None:
        dead_chain_strip_enabled.cache_clear()
        os.environ["XTUNER_DEAD_CHAIN_STRIP"] = "1"
        try:
            assert dead_chain_strip_enabled() is True
        finally:
            os.environ.pop("XTUNER_DEAD_CHAIN_STRIP", None)
            dead_chain_strip_enabled.cache_clear()

    def test_gate_result_is_cached(self) -> None:
        dead_chain_strip_enabled.cache_clear()
        assert dead_chain_strip_enabled() is dead_chain_strip_enabled()


class TestEmptyRouterTensor:
    def test_shape_dtype_device(self) -> None:
        tensor = empty_router_tensor("cpu")
        assert tensor.shape == (0,)
        assert tensor.dtype == torch.float32
        assert tensor.device.type == "cpu"

    def test_cached_per_device(self) -> None:
        assert empty_router_tensor("cpu") is empty_router_tensor(torch.device("cpu"))


class TestAccumulateExpertsOnly:
    def test_live_args_forwarded_and_dead_args_emptied(self) -> None:
        torch.manual_seed(0)
        num_tokens, num_experts_per_tok = 6, 8
        router_topk_ids = [torch.randint(0, 4, (num_tokens, num_experts_per_tok)) for _ in range(2)]
        nonpad_indices = torch.tensor([0, 1, 3, 5])
        hidden_states = torch.randn(num_tokens, 4)
        accumulator = _RecordAccumulator()

        returned = accumulate_experts_only(accumulator, router_topk_ids, nonpad_indices, hidden_states)

        assert returned is hidden_states
        expected_experts = torch.cat(router_topk_ids, dim=0).index_select(0, nonpad_indices).contiguous()
        assert torch.equal(accumulator.kwargs["selected_experts"], expected_experts)
        assert accumulator.kwargs["selected_router_weights"] is empty_router_tensor(hidden_states.device)
        assert accumulator.kwargs["selected_router_logits"] is empty_router_tensor(hidden_states.device)
        assert accumulator.kwargs["balancing_ctx"] is None
        assert accumulator.kwargs["z_ctx"] is None
