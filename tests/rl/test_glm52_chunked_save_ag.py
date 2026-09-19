"""Tests for the chunked save all-gather with page-able staging (GLM-5.2 HF checkpoint save).

The save path (``reusable_staging=False``) cannot use the shared-pool chunked staging; its
fused gather materializes the whole ~12 GiB single-tensor expert bucket on the device plus
a bucket-sized page-locked staging, which killed the node twice (run109/run111). The chunked
save path caps both transients at one shard. These tests verify the gate contract and that
the staged layout and merge result are byte-identical to the original path.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from xtuner.v1.rl.weight_update import chunked_save_unshard as chunked_save_unshard_module
from xtuner.v1.rl.weight_update.chunked_save_unshard import (
    GLM52_CHUNKED_SAVE_AG_ENV,
    GLM52_CHUNKED_SAVE_AG_MIN_GB_ENV,
    foreach_all_gather_stage_chunked_paged,
    try_chunked_save_unshard,
)
from xtuner.v1.utils import load_spec as load_spec_module
from xtuner.v1.utils.cpu_merge import (
    _foreach_all_gather_save_shards_cpu_merge,
    _merge_gathered_save_shard_shared,
)
from xtuner.v1.utils.load_spec import SaveShardStep, ShardDescriptor


@pytest.fixture(scope="module")
def single_rank_group() -> dist.ProcessGroup:
    if not dist.is_initialized():
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29559")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    group = dist.group.WORLD
    assert group is not None
    return group


def _dim0_step(group: dist.ProcessGroup, shape: tuple[int, ...]) -> SaveShardStep:
    return SaveShardStep(shard=ShardDescriptor(dim=0, group=group), shape_before_shard=shape)


class TestGlm52ChunkedSaveAgGate:
    def test_gate_off_without_env(self, monkeypatch: pytest.MonkeyPatch, single_rank_group: dist.ProcessGroup) -> None:
        monkeypatch.delenv(GLM52_CHUNKED_SAVE_AG_ENV, raising=False)
        tensors = [torch.ones(8, 2)]
        steps = [_dim0_step(single_rank_group, (8, 2))]
        assert try_chunked_save_unshard(tensors, steps, [True]) is None

    def test_gate_off_for_single_rank_world(
        self, monkeypatch: pytest.MonkeyPatch, single_rank_group: dist.ProcessGroup
    ) -> None:
        monkeypatch.setenv(GLM52_CHUNKED_SAVE_AG_ENV, "1")
        tensors = [torch.ones(8, 2)]
        steps = [_dim0_step(single_rank_group, (8, 2))]
        # world_size == 1: nothing to chunk, the fused gather is already transient-free.
        assert try_chunked_save_unshard(tensors, steps, [True]) is None

    def test_gate_off_without_merge_on_cpu(
        self, monkeypatch: pytest.MonkeyPatch, single_rank_group: dist.ProcessGroup
    ) -> None:
        monkeypatch.setenv(GLM52_CHUNKED_SAVE_AG_ENV, "1")
        tensors = [torch.ones(8, 2)]
        steps = [_dim0_step(single_rank_group, (8, 2))]
        assert try_chunked_save_unshard(tensors, steps, [False]) is None

    def test_gate_off_below_size_threshold(
        self, monkeypatch: pytest.MonkeyPatch, single_rank_group: dist.ProcessGroup
    ) -> None:
        monkeypatch.setenv(GLM52_CHUNKED_SAVE_AG_ENV, "1")
        monkeypatch.setenv(GLM52_CHUNKED_SAVE_AG_MIN_GB_ENV, "4")
        # Pretend the group has 8 ranks: the (1, 2) fp32 rank-0 local shard gathers to
        # 64 bytes, far below the 4 GiB threshold, so the drop-in must defer to the fused
        # gather. The tensor must be the descriptor-local size — padding now runs before
        # the size check and asserts it.
        monkeypatch.setattr(dist, "get_world_size", lambda group=None: 8)
        tensors = [torch.ones(1, 2)]
        steps = [_dim0_step(single_rank_group, (8, 2))]
        assert try_chunked_save_unshard(tensors, steps, [True]) is None

    def test_gate_tolerates_empty_threshold_env(
        self, monkeypatch: pytest.MonkeyPatch, single_rank_group: dist.ProcessGroup
    ) -> None:
        """The runtime-env whitelist forwards unset knobs as empty strings; the threshold
        must fall back to the default instead of raising."""
        monkeypatch.setenv(GLM52_CHUNKED_SAVE_AG_ENV, "1")
        monkeypatch.setenv(GLM52_CHUNKED_SAVE_AG_MIN_GB_ENV, "")
        monkeypatch.setattr(dist, "get_world_size", lambda group=None: 8)
        tensors = [torch.ones(1, 2)]
        steps = [_dim0_step(single_rank_group, (8, 2))]
        # Empty threshold falls back to the 4 GiB default, so the tiny tensor defers.
        assert try_chunked_save_unshard(tensors, steps, [True]) is None

    def test_gate_off_for_multi_tensor_group(
        self, monkeypatch: pytest.MonkeyPatch, single_rank_group: dist.ProcessGroup
    ) -> None:
        monkeypatch.setenv(GLM52_CHUNKED_SAVE_AG_ENV, "1")
        monkeypatch.setenv(GLM52_CHUNKED_SAVE_AG_MIN_GB_ENV, "0")
        # World size 8 so the single-tensor restriction itself rejects (not the world-1
        # branch): the sliced layout is defined for single-tensor groups, and a multi-
        # tensor group returning one merged tensor would crash the caller's strict zip.
        monkeypatch.setattr(dist, "get_world_size", lambda group=None: 8)
        tensors = [torch.ones(8, 2), torch.ones(4)]
        steps = [_dim0_step(single_rank_group, (8, 2)), _dim0_step(single_rank_group, (4,))]
        assert try_chunked_save_unshard(tensors, steps, [True, True]) is None

    def test_gate_off_for_interleaved_shard(
        self, monkeypatch: pytest.MonkeyPatch, single_rank_group: dist.ProcessGroup
    ) -> None:
        monkeypatch.setenv(GLM52_CHUNKED_SAVE_AG_ENV, "1")
        monkeypatch.setenv(GLM52_CHUNKED_SAVE_AG_MIN_GB_ENV, "0")
        monkeypatch.setattr(dist, "get_world_size", lambda group=None: 8)
        tensors = [torch.ones(8, 2)]
        step = SaveShardStep(
            shard=ShardDescriptor(dim=0, group=single_rank_group, interleave_factor=2),
            shape_before_shard=(8, 2),
        )
        assert try_chunked_save_unshard(tensors, [step], [True]) is None


class TestGlm52ChunkedSaveAgWiring:
    """Regression tests for the cpu_merge wiring (run112): the chunked save path must
    only take ``reusable_staging=False`` groups. Weight-update groups stage into the
    shared pool for zero-copy IPC; a page-able payload there costs a full extra copy."""

    @pytest.fixture(autouse=True)
    def _gate_satisfied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Env conditions that make ``try_chunked_save_unshard`` apply, so the wiring
        condition (``not reusable_staging``) is the only remaining discriminator."""
        monkeypatch.setenv(GLM52_CHUNKED_SAVE_AG_ENV, "1")
        monkeypatch.setenv(GLM52_CHUNKED_SAVE_AG_MIN_GB_ENV, "0")
        monkeypatch.setattr(
            load_spec_module,
            "foreach_all_gather",
            lambda tensor_list, group: [[tensor] for tensor in tensor_list],
        )

    def test_reusable_staging_group_stays_on_fused_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        single_rank_group: dist.ProcessGroup,
    ) -> None:
        tensors = [torch.arange(16, dtype=torch.float32).view(8, 2)]
        steps = [_dim0_step(single_rank_group, (8, 2))]

        def _must_not_run(*args: object, **kwargs: object) -> list[torch.Tensor]:
            raise AssertionError("chunked save path must not take reusable_staging (weight-update) groups")

        monkeypatch.setattr(chunked_save_unshard_module, "try_chunked_save_unshard", _must_not_run)
        merged = _foreach_all_gather_save_shards_cpu_merge(tensors, steps, [True], True)
        assert torch.equal(merged[0].cpu(), tensors[0])

    def test_save_group_takes_chunked_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        single_rank_group: dist.ProcessGroup,
    ) -> None:
        tensors = [torch.arange(16, dtype=torch.float32).view(8, 2)]
        steps = [_dim0_step(single_rank_group, (8, 2))]
        sentinel = torch.zeros(1)

        monkeypatch.setattr(chunked_save_unshard_module, "try_chunked_save_unshard", lambda *a, **k: [sentinel])
        merged = _foreach_all_gather_save_shards_cpu_merge(tensors, steps, [True], False)
        assert merged[0] is sentinel


class TestGlm52ChunkedSaveAgStaging:
    def test_chunked_paged_stage_matches_original_merge(
        self,
        monkeypatch: pytest.MonkeyPatch,
        single_rank_group: dist.ProcessGroup,
    ) -> None:
        """On one rank the chunked paged staged merge must equal the original staged merge."""
        tensors = [torch.arange(16, dtype=torch.float32).view(8, 2)]
        steps = [_dim0_step(single_rank_group, (8, 2))]

        # Original path with the world=1 fused gather (identity, as in the load_spec tests).
        monkeypatch.setattr(
            load_spec_module,
            "foreach_all_gather",
            lambda tensor_list, group: [[tensor] for tensor in tensor_list],
        )
        original = _foreach_all_gather_save_shards_cpu_merge(tensors, steps, [True], False)

        staging_base, host_chunks, staged_shared = foreach_all_gather_stage_chunked_paged(
            tensors[0], single_rank_group
        )
        assert staged_shared is True
        assert len(host_chunks) == 1
        assert host_chunks[0].device.type == "cpu"
        assert staging_base.numel() == tensors[0].numel() * tensors[0].element_size()

        merged = _merge_gathered_save_shard_shared(staging_base, host_chunks, steps[0], staged_shared)
        assert tuple(merged.shape) == (8, 2)
        assert torch.equal(merged.cpu(), original[0].cpu())

    def test_chunked_paged_stage_single_tensor_layout(self, single_rank_group: dist.ProcessGroup) -> None:
        tensors = [torch.arange(8, dtype=torch.float32)]
        staging_base, host_chunks, staged_shared = foreach_all_gather_stage_chunked_paged(
            tensors[0], single_rank_group
        )
        assert staged_shared is True
        assert len(host_chunks) == 1
        assert tuple(host_chunks[0].shape) == (8,)
        assert torch.equal(host_chunks[0], tensors[0])
        assert staging_base.numel() == tensors[0].numel() * tensors[0].element_size()


_SAVE_WORKER_PORT = 29560


def _save_worker(rank: int, world_size: int, global_size: int) -> None:
    """2-rank gloo worker: the chunked paged merge must equal a locally computed dim-0 cat.

    ``global_size`` 16 splits evenly (8 per rank); 9 splits unevenly (5 vs 4), which
    exercises ``tensor_split`` unevenness plus pad bytes end-to-end — both ranks pad to
    5, and the merge must trim rank 1's pad so the result is the 9-element global cat.
    """
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(_SAVE_WORKER_PORT),
    )
    os.environ[GLM52_CHUNKED_SAVE_AG_ENV] = "1"
    os.environ[GLM52_CHUNKED_SAVE_AG_MIN_GB_ENV] = "0"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        group = dist.group.WORLD
        local_size = global_size // world_size + (1 if rank < global_size % world_size else 0)
        local = torch.arange(local_size, dtype=torch.float32) + 10 * rank
        step = _dim0_step(group, (global_size,))
        merged = try_chunked_save_unshard([local], [step], [True])
        assert merged is not None
        expected = torch.cat(
            [
                torch.arange(
                    global_size // world_size + (1 if r < global_size % world_size else 0), dtype=torch.float32
                )
                + 10 * r
                for r in range(world_size)
            ]
        )
        assert tuple(merged[0].shape) == (global_size,)
        assert torch.equal(merged[0], expected)
    finally:
        dist.destroy_process_group()


class TestGlm52ChunkedSaveAgMultiRank:
    @pytest.mark.parametrize("global_size", [16, 9])
    def test_two_rank_chunked_merge(self, global_size: int) -> None:
        mp.spawn(_save_worker, args=(2, global_size), nprocs=2, join=True)
