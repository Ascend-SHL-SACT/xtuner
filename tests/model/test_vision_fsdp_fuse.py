"""Unit tests for the vision-tower FSDP unit fusion (``XTUNER_VISION_FSDP_FUSE``).

These pin the parts of :mod:`xtuner.v1.model.compose.glm53.vision_fsdp_fuse` that can be checked without a
process group: the env gate semantics, the block-to-unit grouping (including the trailing short group), and
the dispatch between the grouped ``fully_shard(list)`` call and the per-block ``_fully_shard`` fallback.
Collective-level behaviour (unshard/reshard timing, gradient equality) is out of scope here and is covered
by the 16-die A/B in ``claude_added/p1_vision_fsdp_fuse.md``.
"""

import pytest
import torch.nn as nn

from xtuner.v1.model.compose.glm53 import vision_fsdp_fuse
from xtuner.v1.model.compose.glm53.vision_fsdp_fuse import (
    shard_vision_blocks,
    vision_block_groups,
    vision_fuse_k,
)


class TestVisionFuseK:
    def test_unset_is_off(self, monkeypatch):
        monkeypatch.delenv("XTUNER_VISION_FSDP_FUSE", raising=False)
        assert vision_fuse_k() == 0

    def test_non_positive_and_one_are_off(self, monkeypatch):
        for raw in ("", "0", "1", "-3"):
            monkeypatch.setenv("XTUNER_VISION_FSDP_FUSE", raw)
            assert vision_fuse_k() == 0

    def test_positive_k_is_kept(self, monkeypatch):
        for raw in ("2", "8", "24", " 24 "):
            monkeypatch.setenv("XTUNER_VISION_FSDP_FUSE", raw)
            assert vision_fuse_k() == int(raw)

    def test_invalid_value_raises(self, monkeypatch):
        monkeypatch.setenv("XTUNER_VISION_FSDP_FUSE", "twenty-four")
        with pytest.raises(ValueError, match="XTUNER_VISION_FSDP_FUSE must be an integer"):
            vision_fuse_k()


class TestVisionBlockGroups:
    def test_single_unit_when_k_covers_depth(self):
        assert vision_block_groups(24, 24) == [list(range(24))]

    def test_multiple_full_units(self):
        assert vision_block_groups(24, 8) == [list(range(0, 8)), list(range(8, 16)), list(range(16, 24))]

    def test_trailing_single_block(self):
        assert vision_block_groups(5, 2) == [[0, 1], [2, 3], [4]]

    def test_trailing_partial_group(self):
        assert vision_block_groups(7, 4) == [[0, 1, 2, 3], [4, 5, 6]]

    def test_k_larger_than_depth_yields_one_group(self):
        assert vision_block_groups(24, 25) == [list(range(24))]

    def test_no_blocks(self):
        assert vision_block_groups(0, 4) == []

    def test_k_of_one_is_rejected(self):
        with pytest.raises(AssertionError):
            vision_block_groups(24, 1)


class _RecordingModel:
    """Minimal ``BaseModel`` stand-in that records per-block ``_fully_shard`` calls."""

    def __init__(self):
        self.per_block_calls: list[nn.Module] = []

    def _fully_shard(self, *, mesh, mp_policy, reshard_after_forward, offload_policy, module):
        self.per_block_calls.append(module)


class TestShardVisionBlocks:
    @staticmethod
    def _run(monkeypatch, num_blocks, fuse_k):
        blocks = nn.ModuleList([nn.Module() for _ in range(num_blocks)])
        grouped_calls: list[list[nn.Module]] = []
        grouped_kwargs: list[dict] = []

        def fake_fully_shard(modules, *, mesh, mp_policy, reshard_after_forward, offload_policy):
            grouped_calls.append(list(modules))
            grouped_kwargs.append(
                {
                    "mesh": mesh,
                    "mp_policy": mp_policy,
                    "reshard_after_forward": reshard_after_forward,
                    "offload_policy": offload_policy,
                }
            )

        monkeypatch.setattr(vision_fsdp_fuse, "fully_shard", fake_fully_shard)
        model = _RecordingModel()
        kwargs = {
            "mesh": object(),
            "mp_policy": object(),
            "offload_policy": object(),
            "reshard_after_forward": True,
            "fuse_k": fuse_k,
        }
        shard_vision_blocks(model, blocks, **kwargs)
        return blocks, model, grouped_calls, grouped_kwargs, kwargs

    def test_full_fusion_makes_one_grouped_call(self, monkeypatch):
        blocks, model, grouped_calls, grouped_kwargs, kwargs = self._run(monkeypatch, 24, 24)
        assert model.per_block_calls == []
        assert len(grouped_calls) == 1
        assert grouped_calls[0] == list(blocks)
        assert grouped_kwargs[0] == {
            "mesh": kwargs["mesh"],
            "mp_policy": kwargs["mp_policy"],
            "reshard_after_forward": True,
            "offload_policy": kwargs["offload_policy"],
        }

    def test_trailing_single_block_uses_per_block_path(self, monkeypatch):
        blocks, model, grouped_calls, _, _ = self._run(monkeypatch, 5, 2)
        assert [len(group) for group in grouped_calls] == [2, 2]
        assert grouped_calls[0] == [blocks[0], blocks[1]]
        assert grouped_calls[1] == [blocks[2], blocks[3]]
        assert model.per_block_calls == [blocks[4]]

    def test_no_single_module_is_passed_as_group(self, monkeypatch):
        _, model, grouped_calls, _, _ = self._run(monkeypatch, 7, 4)
        assert all(len(group) > 1 for group in grouped_calls)
        assert model.per_block_calls == []
