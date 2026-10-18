"""GLM-5.3-Flash vision Ulysses SP（含 sp_size > num_heads 的头复制扩展）。

纯 Ulysses 拓扑：整 SP 组沿头维 a2a，无子组、无 ring。sp_size <= num_heads 时按头切分；
sp_size > num_heads 时每个头复制 head_replica = sp_size // num_heads 份填满 a2a 槽位，
回程 a2a 后只保留每头的第一个副本（重复槽位的上游梯度为 0，反向恰好计一次）。

TestGlm53VisionUlyssesSp2
    sp=2（4 头切 2）前向/反向 vs 非 SP 一致
TestGlm53VisionUlyssesSp4
    sp=4（4 头切 4，每 rank 1 头，无复制）前向/反向 vs 非 SP 一致
TestGlm53VisionUlyssesSp8Replicated
    sp=8 > 4 头（head_replica=2，复制路径）前向/反向 vs 非 SP 一致
TestGlm53VisionSpMultiDp
    dp=2 x sp=2 副本各自持 sp mesh 跑通（mesh 基础设施回归）
"""

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.testing._internal.common_distributed import DistributedTestBase

from xtuner.v1.model.compose.glm53 import Glm53VisionConfig
from xtuner.v1.utils.test_utils import init_data_mesh


IN_CHANNELS = 3
DEPTH = 2
HIDDEN = 32
NUM_HEADS = 4
INTERMEDIATE = 64
PATCH_SIZE = 4
TEMPORAL_PATCH_SIZE = 2
MERGE = 2


def _build_tower() -> torch.nn.Module:
    cfg = Glm53VisionConfig(
        in_channels=IN_CHANNELS,
        depth=DEPTH,
        hidden_size=HIDDEN,
        num_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE,
        patch_size=PATCH_SIZE,
        temporal_patch_size=TEMPORAL_PATCH_SIZE,
        spatial_merge_size=MERGE,
        rms_norm_eps=1e-6,
        attn_impl="flash_attention",
        fully_shard=False,
    )
    torch.manual_seed(0)
    tower = cfg.build()
    for p in tower.parameters():
        p.data.normal_(mean=0.0, std=0.02)
    return tower


class _UlyssesParityMixin:
    """Tower forward/backward under the SP mesh vs the plain single-rank run.

    20 patches is deliberately not a multiple of ``sp_size * merge_unit`` so the padding
    branch runs, and with ``sp_size > num_heads`` the attention goes through the
    replicated-head a2a slots.
    """

    sp_size: int | None = None  # None = single replica: the whole world is the SP mesh

    def _prepare(self, device: str):
        self.create_pg(device)
        torch.manual_seed(0)
        tower = _build_tower().to(device, torch.bfloat16)
        for p in tower.parameters():
            dist.broadcast(p.data, src=0)

        grid_thw = torch.tensor([[1, 4, 4], [1, 2, 2]], device=device)
        patch_dim = IN_CHANNELS * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE
        pixel_values = torch.randn(20, patch_dim, device=device, dtype=torch.bfloat16)
        dist.broadcast(pixel_values, src=0)

        sp_mesh = init_data_mesh(device, self.world_size if self.sp_size is None else self.sp_size)["sp"]
        sp_size = sp_mesh.size()
        merge_unit = MERGE * MERGE
        padded = -(-pixel_values.shape[0] // (sp_size * merge_unit)) * (sp_size * merge_unit)
        split = padded // sp_size
        return tower, grid_thw, pixel_values, sp_mesh, split, padded

    @pytest.mark.gpu
    def test_forward_matches_non_sp(self, device="cuda") -> None:
        tower, grid_thw, pixel_values, sp_mesh, split, padded = self._prepare(device)
        with torch.no_grad():
            reference = tower(pixel_values, grid_thw)
            local = tower(pixel_values, grid_thw, sequence_parallel_mesh=sp_mesh)
        gathered = [torch.empty_like(local) for _ in range(sp_mesh.size())]
        dist.all_gather(gathered, local.contiguous(), group=sp_mesh.get_group())
        merged = torch.cat(gathered, dim=0)[: pixel_values.shape[0]]

        assert merged.shape == reference.shape
        torch.testing.assert_close(merged, reference, rtol=2e-2, atol=2e-2)

    @pytest.mark.gpu
    def test_backward_matches_non_sp(self, device="cuda") -> None:
        tower, grid_thw, pixel_values, sp_mesh, split, padded = self._prepare(device)
        rank = sp_mesh.get_local_rank()
        out_mask = torch.randn(pixel_values.shape[0], HIDDEN, device=device, dtype=torch.bfloat16)
        dist.broadcast(out_mask, src=0)

        pv_ref = pixel_values.detach().clone().requires_grad_(True)
        (tower(pv_ref, grid_thw) * out_mask).sum().backward()

        pv_sp = pixel_values.detach().clone().requires_grad_(True)
        local = tower(pv_sp, grid_thw, sequence_parallel_mesh=sp_mesh)
        # Pad rows carry no loss: the padded mask is zero beyond the real patches.
        padded_mask = F.pad(out_mask, (0, 0, 0, padded - pixel_values.shape[0]))
        (local * padded_mask[rank * split : (rank + 1) * split]).sum().backward()

        gathered = [torch.empty_like(pv_sp.grad) for _ in range(sp_mesh.size())]
        dist.all_gather(gathered, pv_sp.grad.contiguous(), group=sp_mesh.get_group())
        # Every rank's grad has the full [patches, dim] shape with only its owned rows nonzero,
        # so the ranks merge by SUM -- a concat would keep only rank 0's contribution in
        # [:patches] and silently stop checking every other rank's gradients.
        merged = torch.stack(gathered).sum(dim=0)[: pixel_values.shape[0]]

        assert merged.shape == pv_ref.grad.shape
        torch.testing.assert_close(merged, pv_ref.grad, rtol=5e-2, atol=5e-2)


class TestGlm53VisionUlyssesSp2(_UlyssesParityMixin, DistributedTestBase):
    """sp=2 <= 4 heads: the plain Ulysses head split (2 heads per rank)."""

    @property
    def world_size(self) -> int:
        return 2


class TestGlm53VisionUlyssesSp4(_UlyssesParityMixin, DistributedTestBase):
    """sp=4 == 4 heads: one head per rank, no replication."""

    @property
    def world_size(self) -> int:
        return 4


class TestGlm53VisionUlyssesSp8Replicated(_UlyssesParityMixin, DistributedTestBase):
    """sp=8 > 4 heads: head_replica=2 fills the a2a slots (the sp>num_heads path)."""

    @property
    def world_size(self) -> int:
        return 8


class TestGlm53VisionSpMultiDp(_UlyssesParityMixin, DistributedTestBase):
    """dp=2 x sp=2: sibling DP replicas each drive the SP branch from one call site.

    No sub-groups are created by the vision SP path itself, so this guards the sp-mesh-under-dp
    plumbing end to end (it previously regressed as a cross-replica group-name collision when
    the vision tower created its own process groups).
    """

    sp_size = 2

    @property
    def world_size(self) -> int:
        return 4
