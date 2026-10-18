"""Regression tests for the sync-free non-padding index derivation.

The MoE model forward derived ``nonpad_indices`` with ``torch.nonzero(seq_ctx.mask)``,
whose mandatory host-device sync parks the host behind the queue every call. The
existing ``nonpad_index_and_count`` helper derives the same indices from host
metadata (``SequenceContext.mask`` is a leading-True prefix whose tail
``num_padding`` positions are cleared), so the forward now calls the helper.
These tests pin the helper's contract against ``torch.nonzero``.
"""

import pytest
import torch

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.moe.moe import nonpad_index_and_count


def _seq_ctx(total_tokens: int, num_padding: int, device: str = "cpu") -> SequenceContext:
    return SequenceContext(
        input_ids=torch.arange(total_tokens, dtype=torch.int64, device=device).unsqueeze(0),
        cu_seq_lens_q=torch.tensor([0, total_tokens], dtype=torch.int32),
        cu_seq_lens_k=torch.tensor([0, total_tokens], dtype=torch.int32),
        max_length_q=total_tokens,
        max_length_k=total_tokens,
        num_padding=num_padding,
        device=device,
    )


class TestNonpadIndexAndCount:
    @pytest.mark.parametrize("total_tokens,num_padding", [(131072, 0), (131072, 5120), (128, 1), (64, 64)])
    def test_matches_nonzero_mask(self, total_tokens: int, num_padding: int) -> None:
        ctx = _seq_ctx(total_tokens, num_padding)
        indices, count = nonpad_index_and_count([ctx])
        torch.testing.assert_close(indices, torch.nonzero(ctx.mask, as_tuple=True)[1])
        assert count == total_tokens - num_padding
        assert count == indices.numel()

    def test_inputs_embeds_fallback(self) -> None:
        ctx = _seq_ctx(256, 32)
        ctx.input_ids = None
        ctx.inputs_embeds = torch.randn(1, 256, 8)
        indices, count = nonpad_index_and_count([ctx])
        torch.testing.assert_close(indices, torch.nonzero(ctx.mask, as_tuple=True)[1])
        assert count == 224

    def test_multi_context_concat(self) -> None:
        first = _seq_ctx(100, 10)
        second = _seq_ctx(50, 0)
        indices, count = nonpad_index_and_count([first, second])
        expected = torch.cat(
            [
                torch.nonzero(first.mask, as_tuple=True)[1],
                torch.nonzero(second.mask, as_tuple=True)[1] + 100,
            ]
        )
        torch.testing.assert_close(indices, expected)
        assert count == 140

    def test_indices_are_contiguous_prefix(self) -> None:
        indices, _ = nonpad_index_and_count([_seq_ctx(4096, 128)])
        assert indices.device.type == "cpu"
        torch.testing.assert_close(indices, torch.arange(4096 - 128, dtype=torch.int64))

    @pytest.mark.parametrize("total_tokens,num_padding", [(128, 129), (64, 65)])
    def test_num_padding_above_total_raises(self, total_tokens: int, num_padding: int) -> None:
        with pytest.raises(AssertionError, match="outside"):
            nonpad_index_and_count([_seq_ctx(total_tokens, num_padding)])

    def test_negative_num_padding_raises(self) -> None:
        with pytest.raises(AssertionError, match="outside"):
            nonpad_index_and_count([_seq_ctx(128, -1)])
