# Copyright (c) OpenMMLab. All rights reserved.
"""Unit tests for the per-``SequenceContext`` SP-mode/KV-slice cache.

The cache lives in :mod:`xtuner.v1.ops.sparse_mla.npu_indexer` and is shared
by the indexer and sparse-MLA TND packed paths. These tests pin the push/pop
behavior: one fill per ``(context, shard_start, seq_len)`` key, identity-shared
read-only tensors, and eviction when the context is garbage collected.
"""

import gc
import weakref
from collections.abc import Callable, Iterator

import pytest
import torch

from xtuner.v1.ops.sparse_mla.npu_indexer import _SP_SLICE_CACHE, get_sp_mode_and_slice


class _FakeSeqCtx:
    """Duck-typed stand-in exposing only the attribute the cache reads."""

    def __init__(self, cu_seq_lens_q: torch.Tensor) -> None:
        self.cu_seq_lens_q = cu_seq_lens_q


def _slice_recorder(calls: list[tuple[int, int]]) -> Callable[[torch.Tensor, int, int, torch.device], tuple]:
    def _compute(
        cu_seq_q_global: torch.Tensor, shard_start: int, shard_end: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        calls.append((shard_start, shard_end))
        assert cu_seq_q_global.dtype == torch.int32
        assert torch.device(device) == torch.device("cpu")
        return cu_seq_q_global, cu_seq_q_global, shard_start, shard_end

    return _compute


class TestSpSliceCache:

    @pytest.fixture(autouse=True)
    def _clear_cache(self) -> Iterator[None]:
        _SP_SLICE_CACHE.clear()
        yield
        _SP_SLICE_CACHE.clear()

    def test_sp1_probe_cached_and_slice_not_invoked(self) -> None:
        ctx = _FakeSeqCtx(torch.tensor([0, 8]))
        calls: list[tuple[int, int]] = []

        first = get_sp_mode_and_slice(ctx, 0, 8, torch.device("cpu"), _slice_recorder(calls))
        second = get_sp_mode_and_slice(ctx, 0, 8, torch.device("cpu"), _slice_recorder(calls))

        assert first[0] is True and second[0] is True
        assert first[1] is second[1]
        assert first[2] is None and second[2] is None
        assert calls == []

    def test_sp_gt_1_slice_computed_once_per_key(self) -> None:
        ctx = _FakeSeqCtx(torch.tensor([0, 8, 16]))
        calls: list[tuple[int, int]] = []

        meta_1 = get_sp_mode_and_slice(ctx, 8, 8, torch.device("cpu"), _slice_recorder(calls))[2]
        meta_2 = get_sp_mode_and_slice(ctx, 8, 8, torch.device("cpu"), _slice_recorder(calls))[2]

        assert calls == [(8, 16)]
        assert meta_1 is not None and meta_2 is meta_1
        assert meta_1[2] == 8 and meta_1[3] == 16

    def test_cache_key_distinguishes_shards(self) -> None:
        ctx = _FakeSeqCtx(torch.tensor([0, 8, 16]))
        calls: list[tuple[int, int]] = []

        meta_1 = get_sp_mode_and_slice(ctx, 8, 8, torch.device("cpu"), _slice_recorder(calls))[2]
        meta_2 = get_sp_mode_and_slice(ctx, 0, 8, torch.device("cpu"), _slice_recorder(calls))[2]

        assert calls == [(8, 16), (0, 8)]
        assert meta_1 is not None and meta_2 is not None and meta_1 is not meta_2

    def test_cached_tensors_identity_shared_across_layers(self) -> None:
        ctx = _FakeSeqCtx(torch.tensor([0, 8, 16]))

        _, cu_q_1, meta_1 = get_sp_mode_and_slice(ctx, 8, 8, torch.device("cpu"), _slice_recorder([]))
        _, cu_q_2, meta_2 = get_sp_mode_and_slice(ctx, 8, 8, torch.device("cpu"), _slice_recorder([]))

        assert cu_q_1 is cu_q_2
        assert meta_1 is not None and meta_2 is not None
        assert meta_1[0] is meta_2[0] and meta_1[1] is meta_2[1]

    def test_weakref_eviction_when_context_collected(self) -> None:
        ctx = _FakeSeqCtx(torch.tensor([0, 8]))
        get_sp_mode_and_slice(ctx, 0, 8, torch.device("cpu"), _slice_recorder([]))
        assert len(_SP_SLICE_CACHE) == 1

        ref = weakref.ref(ctx)
        del ctx
        gc.collect()

        assert ref() is None
        assert len(_SP_SLICE_CACHE) == 0
