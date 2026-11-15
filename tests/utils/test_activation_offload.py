"""Tests for saved-tensor offload lifecycle management."""

from unittest import mock

import pytest
import torch

from xtuner.v1.utils.activation_offload import OffloadItem, OffloadManager, SingletonMeta


@pytest.fixture
def manager():
    # OffloadManager is a process-level singleton; rebuild it per test so no
    # runtime state leaks across cases.
    SingletonMeta._instances.pop(OffloadManager, None)
    yield OffloadManager()
    SingletonMeta._instances.pop(OffloadManager, None)


class TestOffloadManager:
    def test_clear_step_releases_runtime_state_and_preserves_pin_cache(self, manager):
        manager.items["text_0_0"] = OffloadItem()
        manager.may_npu_tensors["text_0_1"] = OffloadItem()
        manager.items["other_0_0"] = OffloadItem()
        manager.pin_memory_cache["text_0_0"] = mock.sentinel.pinned_buffer

        manager.clear_step(group="text")

        assert "text_0_0" not in manager.items
        assert "text_0_1" not in manager.may_npu_tensors
        assert "other_0_0" in manager.items
        assert manager.pin_memory_cache["text_0_0"] is mock.sentinel.pinned_buffer

    def test_clear_step_is_noop_without_offload(self, manager):
        manager.clear_step()  # no entries; must not raise
        assert not manager.items

    def test_clear_step_skips_device_sync_when_empty(self, manager):
        # The early exit must precede torch.cuda.synchronize(): clear_step runs unconditionally at the
        # train-step boundary, and steps without offload traffic must not pay a full-device sync.
        with mock.patch("torch.cuda.synchronize") as sync_mock:
            manager.clear_step()
        sync_mock.assert_not_called()

    def test_clear_step_syncs_before_releasing_pending_offloads(self, manager):
        # With pending offload items the fence is mandatory: references may only be dropped after the
        # device drained, since launch_d2h reads tensors on unowned streams.
        manager.items["text_0_0"] = OffloadItem()
        with mock.patch("torch.cuda.synchronize") as sync_mock:
            manager.clear_step()
        sync_mock.assert_called_once_with()
        assert not manager.items


class _NullStorageStub:
    # Stand-in for a tensor whose pinned host storage came back null-backed: CANN 9.2.1's
    # pinned allocator can return such a buffer with the recorded nbytes intact, so numel()
    # looks healthy while the data pointer is NULL (CANN EH0012).
    def __init__(self, shape):
        self._numel = 1
        for dim in torch.Size(shape):
            self._numel *= dim

    def numel(self):
        return self._numel

    def untyped_storage(self):
        return mock.Mock(data_ptr=mock.Mock(return_value=0))


class TestPinnedAllocFallback:
    def test_healthy_pinned_buffer_is_cached_as_is(self, manager):
        buf = manager.get_or_create_pin_memory("k", (4,), torch.float32)
        assert buf.untyped_storage().data_ptr() != 0
        assert manager.pin_memory_cache["k"] is buf

    def test_null_backed_pinned_buffer_is_retried_not_cached(self, manager):
        # CANN 9.2.1 can hand back a null-backed storage on the first binding of a fresh
        # NPU stream (EH0012). Caching it poisons the park path: the buffer's nbytes stay
        # valid, so the failure only resurfaces far away as a fatal "dst cannot be a NULL
        # pointer" D2H (job-dyb-q35-xtuner-132). The helper must detect the null-backed
        # buffer and retry instead of caching it.
        real_empty = torch.empty
        calls = {"n": 0}

        def flaky_empty(*args, **kwargs):
            if kwargs.get("pin_memory"):
                calls["n"] += 1
                if calls["n"] == 1:
                    return _NullStorageStub((2, 3))
            return real_empty(*args, **kwargs)

        with mock.patch("torch.empty", side_effect=flaky_empty):
            buf = manager.get_or_create_pin_memory("k", (2, 3), torch.float32)

        assert calls["n"] == 2
        assert buf.untyped_storage().data_ptr() != 0
        assert manager.pin_memory_cache["k"] is buf

    def test_persistent_null_pinned_falls_back_to_pageable(self, manager):
        # A rank whose pinned allocator stays broken must keep training on a pageable
        # buffer (correct, slower copies) instead of dying at the next park.
        real_empty = torch.empty
        pinned_calls = {"n": 0}

        def always_null_pinned(*args, **kwargs):
            if kwargs.get("pin_memory"):
                pinned_calls["n"] += 1
                return _NullStorageStub((4,))
            return real_empty(*args, **kwargs)

        with mock.patch("torch.empty", side_effect=always_null_pinned):
            buf = manager.get_or_create_pin_memory("k", (4,), torch.float32)

        assert pinned_calls["n"] == 2
        assert buf.untyped_storage().data_ptr() != 0
