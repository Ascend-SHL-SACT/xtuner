import os
from collections.abc import Callable

import pytest
import torch
import torch.distributed as dist
from pydantic import ValidationError
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Shard as DTensorShard
from torch.distributed.tensor import distribute_tensor

from xtuner.v1.model.base import BaseModel, XTunerBaseModelConfig
from xtuner.v1.utils import cpu_merge as cpu_merge_module
from xtuner.v1.utils import load_spec as load_spec_module
from xtuner.v1.utils.interleaved_shard import InterleavedShard, RuntimeLayout
from xtuner.v1.utils.load_spec import LoadSpec, ShardDescriptor, unshard_tensors_for_hf_save


@pytest.fixture(scope="module")
def single_rank_group() -> dist.ProcessGroup:
    if not dist.is_initialized():
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29555")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    group = dist.group.WORLD
    assert group is not None
    return group


@pytest.fixture(scope="module")
def local_groups(single_rank_group: dist.ProcessGroup):
    groups = tuple(dist.new_group([0]) for _ in range(3))
    yield groups
    for group in groups:
        dist.destroy_process_group(group)


@pytest.fixture
def set_shard_rank(monkeypatch: pytest.MonkeyPatch) -> Callable[[int, int], None]:
    def configure(world_size: int, rank: int) -> None:
        monkeypatch.setattr(dist, "get_world_size", lambda group=None: world_size)
        monkeypatch.setattr(dist, "get_rank", lambda group=None: rank)

    return configure


def set_group_layout(
    monkeypatch: pytest.MonkeyPatch,
    layouts: dict[dist.ProcessGroup, tuple[int, int, list[int]]],
) -> None:
    monkeypatch.setattr(
        dist,
        "get_world_size",
        lambda group=None: layouts[group][0] if group in layouts else 1,
    )
    monkeypatch.setattr(
        dist,
        "get_rank",
        lambda group=None: layouts[group][1] if group in layouts else 0,
    )
    monkeypatch.setattr(
        dist,
        "get_process_group_ranks",
        lambda group: layouts[group][2],
    )


class TestLoadSpecSchema:
    def test_same_unsharded_spec(self) -> None:
        spec = LoadSpec(
            name="layers.0.mlp.gate.weight",
            global_hf_keys=["model.layers.0.mlp.gate.weight"],
            global_shape=(128, 64),
        )

        assert spec.is_fused is False
        assert spec.is_sharded is False
        assert spec.fused_dim is None
        assert spec.shards == []
        assert spec.origin_shape is None
        assert spec.unpadded_global_shape == spec.global_shape

    def test_from_tensor_derives_plain_tensor_layout(self) -> None:
        spec = LoadSpec.from_tensor(
            name="layers.0.experts.fused_w1w3.weight",
            hf_keys=["k0", "k1"],
            tensor=torch.empty(128, 64),
            origin_shape=(120, 64),
        )

        assert spec.global_hf_keys == ["k0", "k1"]
        assert spec.global_shape == (128, 64)
        assert spec.fused_dim == 0
        assert spec.shards == []
        assert spec.origin_shape == (120, 64)

    def test_from_tensor_derives_dtensor_shards(self, single_rank_group: dist.ProcessGroup) -> None:
        mesh = DeviceMesh("cpu", [0])
        tensor = distribute_tensor(torch.empty(128, 64), mesh, [DTensorShard(0)])

        spec = LoadSpec.from_tensor(name="layers.0.mlp.gate.weight", hf_keys=["gate"], tensor=tensor)

        assert spec.global_hf_keys == ["gate"]
        assert spec.global_shape == (128, 64)
        assert spec.fused_dim is None
        assert [(shard.dim, shard.interleave_factor) for shard in spec.shards] == [(0, 1)]

    def test_runtime_layout_normalizes_unordered_interleaved_placement(
        self,
        single_rank_group: dist.ProcessGroup,
    ) -> None:
        class FakeDeviceMesh:
            shape = (2, 2)

            def size(self, mesh_dim: int) -> int:
                return self.shape[mesh_dim]

            def get_group(self, mesh_dim: int) -> dist.ProcessGroup:
                return single_rank_group

            def get_local_rank(self, mesh_dim: int) -> int:
                return 0

        class FakeDTensor:
            shape = (16, 4)
            placements = (DTensorShard(0), InterleavedShard(0, num_local_stripes=2))
            device_mesh = FakeDeviceMesh()

        layout = RuntimeLayout.from_dtensor(FakeDTensor())  # type: ignore[arg-type]

        assert [(shard.dim, shard.interleave_factor) for shard in layout.ordered_shards] == [(0, 1), (0, 2)]
        assert layout.is_interleaved
        assert layout.shard_size(0) == 4
        assert layout.owned_runs() == [
            ((0, 0), (2, 4), 0, 2),
            ((4, 0), (2, 4), 2, 2),
        ]

    def test_fused_spec_requires_fused_dim(self) -> None:
        with pytest.raises(ValidationError, match="fused_dim"):
            LoadSpec(
                name="layers.0.mlp.fused_w1w3.weight",
                global_hf_keys=["gate", "up"],
                global_shape=(256, 64),
            )

    def test_descriptor_order_is_preserved(self, single_rank_group: dist.ProcessGroup) -> None:
        spec = LoadSpec(
            name="layers.0.experts.fused_w1w3.weight",
            global_hf_keys=["gate", "up"],
            global_shape=(256, 64),
            fused_dim=0,
            shards=[
                ShardDescriptor(dim=0, group=single_rank_group),
                ShardDescriptor(dim=0, group=single_rank_group, interleave_factor=2),
            ],
        )

        assert [shard.interleave_factor for shard in spec.shards] == [1, 2]
        assert spec.is_fused is True
        assert spec.is_sharded is True

    def test_uneven_interleave_is_rejected(
        self,
        single_rank_group: dist.ProcessGroup,
        set_shard_rank: Callable[[int, int], None],
    ) -> None:
        set_shard_rank(2, 0)

        with pytest.raises(NotImplementedError, match="Even interleave requires"):
            LoadSpec(
                name="layers.0.experts.fused_w1w3.weight",
                global_hf_keys=["gate"],
                global_shape=(10, 4),
                shards=[ShardDescriptor(dim=0, group=single_rank_group, interleave_factor=3)],
            )

    def test_zero_size_continuous_shard_is_valid(
        self,
        single_rank_group: dist.ProcessGroup,
        set_shard_rank: Callable[[int, int], None],
    ) -> None:
        set_shard_rank(2, 1)
        spec = LoadSpec(
            name="embeddings.cls_embedding",
            global_hf_keys=["embeddings.cls_embedding"],
            global_shape=(1, 1, 1024),
            shards=[ShardDescriptor(dim=0, group=single_rank_group)],
        )

        plan = spec.plan_hf_load()

        assert plan.hf_keys == []
        target = torch.empty(0, 1, 1024)
        plan.load_into([], target, lambda _, tensor: tensor)
        assert target.numel() == 0


class TestHFLoadPlan:
    def test_interleave_selects_smallest_hf_key_envelope(
        self,
        single_rank_group: dist.ProcessGroup,
        set_shard_rank: Callable[[int, int], None],
    ) -> None:
        set_shard_rank(2, 0)
        spec = LoadSpec(
            name="layers.0.experts.fused_w1w3.weight",
            global_hf_keys=["k0", "k1", "k2", "k3"],
            global_shape=(400, 8),
            fused_dim=0,
            shards=[ShardDescriptor(dim=0, group=single_rank_group, interleave_factor=2)],
        )

        plan = spec.plan_hf_load()

        assert plan.hf_keys == ["k0", "k1", "k2"]
        full = torch.arange(400 * 8).reshape(400, 8)
        target = torch.empty(200, 8, dtype=full.dtype)
        plan.load_into([full[:100], full[100:200], full[200:300]], target, lambda _, tensor: tensor)
        torch.testing.assert_close(target, torch.cat((full[:100], full[200:300])))

    def test_non_fused_continuous_shard(
        self,
        single_rank_group: dist.ProcessGroup,
        set_shard_rank: Callable[[int, int], None],
    ) -> None:
        set_shard_rank(2, 1)
        spec = LoadSpec(
            name="layers.0.self_attn.q_proj.weight",
            global_hf_keys=["q_proj"],
            global_shape=(128, 256),
            shards=[ShardDescriptor(dim=1, group=single_rank_group)],
        )

        plan = spec.plan_hf_load()
        full = torch.arange(128 * 256).reshape(128, 256)
        target = torch.empty(128, 128, dtype=full.dtype)
        plan.load_into([full], target, lambda _, tensor: tensor)

        torch.testing.assert_close(target, full[:, 128:])

    def test_origin_shape_clips_runtime_padding(
        self,
        single_rank_group: dist.ProcessGroup,
        set_shard_rank: Callable[[int, int], None],
    ) -> None:
        set_shard_rank(4, 3)
        spec = LoadSpec(
            name="layers.0.experts.fused_w1w3.weight",
            global_hf_keys=["k0", "k1", "k2", "k3"],
            global_shape=(480, 8),
            fused_dim=0,
            shards=[ShardDescriptor(dim=0, group=single_rank_group)],
            origin_shape=(400, 8),
        )

        plan = spec.plan_hf_load()

        assert plan.hf_keys == ["k3"]
        full = torch.arange(400 * 8).reshape(400, 8)
        target = torch.full((120, 8), -1)
        plan.load_into([full[300:400]], target, lambda _, tensor: tensor)
        torch.testing.assert_close(target[:40], full[360:400])
        torch.testing.assert_close(target[40:], torch.zeros_like(target[40:]))

    def test_origin_shape_zeroes_pad_only_rank(
        self,
        single_rank_group: dist.ProcessGroup,
        set_shard_rank: Callable[[int, int], None],
    ) -> None:
        set_shard_rank(8, 7)
        spec = LoadSpec(
            name="layers.0.experts.fused_w1w3.weight",
            global_hf_keys=["k0", "k1", "k2", "k3"],
            global_shape=(480, 8),
            fused_dim=0,
            shards=[ShardDescriptor(dim=0, group=single_rank_group)],
            origin_shape=(400, 8),
        )

        plan = spec.plan_hf_load()

        assert plan.hf_keys == []
        target = torch.ones(60, 8)
        plan.load_into([], target, lambda _, tensor: tensor)
        torch.testing.assert_close(target, torch.zeros_like(target))

    def test_model_canonicalization_runs_before_interleave_copy(
        self,
        single_rank_group: dist.ProcessGroup,
        set_shard_rank: Callable[[int, int], None],
    ) -> None:
        set_shard_rank(2, 0)
        spec = LoadSpec(
            name="layers.0.experts.fused_w1w3.weight",
            global_hf_keys=["packed_gate_up"],
            global_shape=(8, 2),
            shards=[ShardDescriptor(dim=0, group=single_rank_group, interleave_factor=4)],
        )
        packed = torch.arange(16).reshape(2, 2, 4)
        canonical = packed.permute(0, 2, 1).reshape(8, 2)
        target = torch.empty(4, 2, dtype=packed.dtype)

        spec.plan_hf_load().load_into(
            [packed],
            target,
            lambda name, tensor: tensor.permute(0, 2, 1).reshape(8, 2),
        )

        torch.testing.assert_close(target, canonical[(0, 2, 4, 6), :])

    def test_qwen35_packed_adapter_runs_before_expert_tp_copy(
        self,
        single_rank_group: dist.ProcessGroup,
        set_shard_rank: Callable[[int, int], None],
    ) -> None:
        from xtuner.v1.model.moe.qwen3_5_text import Qwen3_5_VLTextMoE

        set_shard_rank(2, 0)
        model = object.__new__(Qwen3_5_VLTextMoE)
        packed = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
        canonical = packed.flatten(0, 1)
        spec = LoadSpec(
            name="layers.0.experts.fused_w1w3.weight",
            global_hf_keys=["model.layers.0.mlp.experts.gate_up_proj"],
            global_shape=tuple(canonical.shape),
            shards=[ShardDescriptor(dim=0, group=single_rank_group, interleave_factor=4)],
        )
        target = torch.empty(4, 3, dtype=packed.dtype)

        spec.plan_hf_load().load_into([packed], target, model.hf_tensor_to_canonical)

        torch.testing.assert_close(target, canonical[(0, 2, 4, 6), :])

    def test_qwen3vl_packed_adapter_runs_before_expert_tp_copy(
        self,
        single_rank_group: dist.ProcessGroup,
        set_shard_rank: Callable[[int, int], None],
    ) -> None:
        from xtuner.v1.model.moe.qwen3vl_text import Qwen3VLTextMoE

        set_shard_rank(2, 0)
        model = object.__new__(Qwen3VLTextMoE)
        packed = torch.arange(2 * 3 * 4).reshape(2, 3, 4)
        canonical = packed.transpose(1, 2).reshape(8, 3)
        spec = LoadSpec(
            name="layers.0.experts.fused_w1w3.weight",
            global_hf_keys=["model.layers.0.mlp.experts.gate_up_proj"],
            global_shape=tuple(canonical.shape),
            shards=[ShardDescriptor(dim=0, group=single_rank_group, interleave_factor=4)],
        )
        target = torch.empty(4, 3, dtype=packed.dtype)

        spec.plan_hf_load().load_into([packed], target, model.hf_tensor_to_canonical)

        torch.testing.assert_close(target, canonical[(0, 2, 4, 6), :])

    def test_gpt_oss_packed_adapter_runs_before_expert_tp_copy(
        self,
        single_rank_group: dist.ProcessGroup,
        set_shard_rank: Callable[[int, int], None],
    ) -> None:
        from xtuner.v1.model.moe.gpt_oss import GptOss

        set_shard_rank(2, 0)
        model = object.__new__(GptOss)
        packed = torch.arange(2 * 3 * 4).reshape(2, 3, 4)
        canonical = model.hf_tensor_to_canonical("layers.0.experts.fused_w1w3.weight", packed)
        spec = LoadSpec(
            name="layers.0.experts.fused_w1w3.weight",
            global_hf_keys=["model.layers.0.mlp.experts.gate_up_proj"],
            global_shape=tuple(canonical.shape),
            shards=[ShardDescriptor(dim=0, group=single_rank_group, interleave_factor=4)],
        )
        target = torch.empty(4, 3, dtype=packed.dtype)

        spec.plan_hf_load().load_into([packed], target, model.hf_tensor_to_canonical)

        torch.testing.assert_close(target, canonical[(0, 2, 4, 6), :])

    def test_glm_per_expert_adapter_runs_before_expert_tp_copy(
        self,
        single_rank_group: dist.ProcessGroup,
        set_shard_rank: Callable[[int, int], None],
    ) -> None:
        from xtuner.v1.model.moe.glm52 import Glm52MoE

        set_shard_rank(2, 0)
        model = object.__new__(Glm52MoE)
        packed = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
        canonical = packed.flatten(0, 1)
        spec = LoadSpec(
            name="layers.0.experts.fused_w1w3.weight",
            global_hf_keys=["gate", "up"],
            global_shape=tuple(canonical.shape),
            fused_dim=0,
            shards=[ShardDescriptor(dim=0, group=single_rank_group, interleave_factor=4)],
        )
        target = torch.empty(4, 3, dtype=packed.dtype)

        spec.plan_hf_load().load_into([packed[:1], packed[1:]], target, model.hf_tensor_to_canonical)

        torch.testing.assert_close(target, canonical[(0, 2, 4, 6), :])

    def test_ep_etp_fsdp_composition_maps_back_to_global_runs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        local_groups: tuple[dist.ProcessGroup, dist.ProcessGroup, dist.ProcessGroup],
    ) -> None:
        ep_group, etp_group, fsdp_group = local_groups
        set_group_layout(
            monkeypatch,
            {
                ep_group: (2, 1, [0, 1]),
                etp_group: (2, 1, [0, 2]),
                fsdp_group: (2, 0, [0, 3]),
            },
        )
        spec = LoadSpec(
            name="weight",
            global_hf_keys=["weight"],
            global_shape=(32, 2),
            shards=[
                ShardDescriptor(dim=0, group=ep_group),
                ShardDescriptor(dim=0, group=etp_group, interleave_factor=2),
                ShardDescriptor(dim=0, group=fsdp_group),
            ],
        )
        full = torch.arange(64).reshape(32, 2)
        target = torch.empty(4, 2, dtype=full.dtype)

        spec.plan_hf_load().load_into([full], target, lambda _, tensor: tensor)

        torch.testing.assert_close(target, full[20:24])


class TestHFSavePolicy:
    def test_fused_keys_are_split_across_save_ranks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = BaseModel(XTunerBaseModelConfig())
        model.config.hf_save_cfg.max_save_rank = 4
        spec = LoadSpec(
            name="layers.0.experts.fused_w1w3.weight",
            global_hf_keys=[f"k{i}" for i in range(8)],
            global_shape=(800, 64),
            fused_dim=0,
        )

        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_world_size", lambda group=None: 8)

        expected_ranges = {0: (0, 2), 1: (2, 4), 2: (4, 6), 3: (6, 8), 4: (0, 0)}
        for rank, expected_range in expected_ranges.items():
            monkeypatch.setattr(dist, "get_rank", lambda group=None, rank=rank: rank)
            assert model._hf_save_key_range(spec.plan_hf_save(distributed_save=True)) == expected_range

    def test_preserved_fused_shard_exposes_local_hf_keys(
        self,
        single_rank_group: dist.ProcessGroup,
        set_shard_rank: Callable[[int, int], None],
    ) -> None:
        set_shard_rank(4, 1)
        spec = LoadSpec(
            name="layers.0.experts.fused_w1w3.weight",
            global_hf_keys=["k0", "k1", "k2", "k3"],
            global_shape=(400, 64),
            fused_dim=0,
            shards=[ShardDescriptor(dim=0, group=single_rank_group)],
        )

        save_plan = spec.plan_hf_save(preserve_process_group=single_rank_group)

        assert save_plan.preserves_shards is True
        assert save_plan.hf_keys == ["k1"]
        assert save_plan.runtime_output_shape == (100, 64)
        assert save_plan.output_shape == (100, 64)

    def test_preserved_fused_shard_must_align_with_hf_key_boundary(
        self,
        single_rank_group: dist.ProcessGroup,
        set_shard_rank: Callable[[int, int], None],
    ) -> None:
        set_shard_rank(8, 1)
        spec = LoadSpec(
            name="layers.0.experts.fused_w1w3.weight",
            global_hf_keys=["k0", "k1", "k2", "k3"],
            global_shape=(400, 64),
            fused_dim=0,
            shards=[ShardDescriptor(dim=0, group=single_rank_group)],
        )

        with pytest.raises(AssertionError, match="must align with HF key size"):
            spec.plan_hf_save(preserve_process_group=single_rank_group)


class TestHFSaveUnshard:
    @staticmethod
    def _patch_foreach_all_gather(
        monkeypatch: pytest.MonkeyPatch,
        responses: list[list[list[torch.Tensor]]] | None = None,
    ) -> list[dict[str, object]]:
        calls: list[dict[str, object]] = []

        def fake_foreach_all_gather(
            tensor_list: list[torch.Tensor],
            group: dist.ProcessGroup,
        ) -> list[list[torch.Tensor]]:
            calls.append(
                {
                    "group": group,
                    "shapes": [tuple(tensor.shape) for tensor in tensor_list],
                    "dtypes": [tensor.dtype for tensor in tensor_list],
                }
            )
            if responses is not None:
                return responses.pop(0)
            return [[tensor] for tensor in tensor_list]

        monkeypatch.setattr(load_spec_module, "foreach_all_gather", fake_foreach_all_gather)
        return calls

    def test_scheduler_batches_same_group_and_respects_dependencies(
        self,
        monkeypatch: pytest.MonkeyPatch,
        single_rank_group: dist.ProcessGroup,
    ) -> None:
        calls = self._patch_foreach_all_gather(monkeypatch)
        specs = [
            LoadSpec(
                name="experts",
                global_hf_keys=["k0", "k1"],
                global_shape=(8, 2),
                fused_dim=0,
                shards=[
                    ShardDescriptor(dim=0, group=single_rank_group),
                    ShardDescriptor(dim=0, group=single_rank_group),
                ],
            ),
            LoadSpec(
                name="gate",
                global_hf_keys=["gate"],
                global_shape=(4, 2),
                shards=[ShardDescriptor(dim=0, group=single_rank_group)],
            ),
        ]

        output = unshard_tensors_for_hf_save(
            [torch.ones(8, 2), torch.ones(4, 2)],
            [spec.plan_hf_save() for spec in specs],
        )

        assert [tuple(tensor.shape) for tensor in output] == [(8, 2), (4, 2)]
        assert [call["shapes"] for call in calls] == [[(8, 2), (4, 2)], [(8, 2)]]

    def test_scheduler_splits_different_dtypes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        single_rank_group: dist.ProcessGroup,
    ) -> None:
        calls = self._patch_foreach_all_gather(monkeypatch)
        specs = [
            LoadSpec(
                name=name,
                global_hf_keys=[name],
                global_shape=(4, 2),
                shards=[ShardDescriptor(dim=0, group=single_rank_group)],
            )
            for name in ("gate", "up")
        ]

        output = unshard_tensors_for_hf_save(
            [torch.ones(4, 2, dtype=torch.float32), torch.ones(4, 2, dtype=torch.float64)],
            [spec.plan_hf_save() for spec in specs],
        )

        assert [tuple(tensor.shape) for tensor in output] == [(4, 2), (4, 2)]
        assert [call["dtypes"] for call in calls] == [[torch.float32], [torch.float64]]

    def test_continuous_collective_padding_restores_runtime_shape(
        self,
        monkeypatch: pytest.MonkeyPatch,
        single_rank_group: dist.ProcessGroup,
        set_shard_rank: Callable[[int, int], None],
    ) -> None:
        set_shard_rank(3, 2)
        calls = self._patch_foreach_all_gather(
            monkeypatch,
            responses=[[[torch.tensor([0, 1]), torch.tensor([2, 3]), torch.tensor([4, 0])]]],
        )
        spec = LoadSpec(
            name="weight",
            global_hf_keys=["weight"],
            global_shape=(5,),
            shards=[ShardDescriptor(dim=0, group=single_rank_group)],
        )

        [output] = unshard_tensors_for_hf_save([torch.tensor([4])], [spec.plan_hf_save()])

        torch.testing.assert_close(output, torch.arange(5))
        assert calls[0]["shapes"] == [(2,)]

    def test_even_interleave_deinterleaves_then_trims_fp8_padding(
        self,
        monkeypatch: pytest.MonkeyPatch,
        single_rank_group: dist.ProcessGroup,
        set_shard_rank: Callable[[int, int], None],
    ) -> None:
        set_shard_rank(2, 0)
        calls = self._patch_foreach_all_gather(
            monkeypatch,
            responses=[[[torch.tensor([0, 1, 4, 5]), torch.tensor([2, 3, 6, 7])]]],
        )
        spec = LoadSpec(
            name="weight",
            global_hf_keys=["weight"],
            global_shape=(8,),
            origin_shape=(6,),
            shards=[ShardDescriptor(dim=0, group=single_rank_group, interleave_factor=2)],
        )

        [output] = unshard_tensors_for_hf_save(
            [torch.tensor([0, 1, 4, 5])],
            [spec.plan_hf_save()],
        )

        torch.testing.assert_close(output, torch.arange(6))
        assert calls[0]["shapes"] == [(4,)]

    def test_preserve_ep_still_deinterleaves_etp(
        self,
        monkeypatch: pytest.MonkeyPatch,
        local_groups: tuple[dist.ProcessGroup, dist.ProcessGroup, dist.ProcessGroup],
    ) -> None:
        ep_group, etp_group, _ = local_groups
        set_group_layout(
            monkeypatch,
            {
                ep_group: (2, 1, [0, 1]),
                etp_group: (2, 0, [0, 2]),
            },
        )
        self._patch_foreach_all_gather(
            monkeypatch,
            responses=[[[torch.tensor([4, 6]), torch.tensor([5, 7])]]],
        )
        spec = LoadSpec(
            name="experts",
            global_hf_keys=["k0", "k1", "k2", "k3"],
            global_shape=(8,),
            fused_dim=0,
            shards=[
                ShardDescriptor(dim=0, group=ep_group),
                ShardDescriptor(dim=0, group=etp_group, interleave_factor=2),
            ],
        )
        plan = spec.plan_hf_save(preserve_process_group=ep_group)

        [output] = unshard_tensors_for_hf_save([torch.tensor([4, 6])], [plan])

        torch.testing.assert_close(output, torch.tensor([4, 5, 6, 7]))
        assert plan.hf_keys == ["k2", "k3"]

    def test_only_gather_fsdp_preserves_etp_and_trims_local_fp8_tail(
        self,
        monkeypatch: pytest.MonkeyPatch,
        local_groups: tuple[dist.ProcessGroup, dist.ProcessGroup, dist.ProcessGroup],
    ) -> None:
        etp_group, fsdp_group, _ = local_groups
        set_group_layout(
            monkeypatch,
            {
                etp_group: (2, 1, [0, 1]),
                fsdp_group: (2, 1, [0, 2]),
            },
        )
        self._patch_foreach_all_gather(
            monkeypatch,
            responses=[[[torch.tensor([4, 5, 6, 7]), torch.tensor([12, 13, 14, 15])]]],
        )
        spec = LoadSpec(
            name="weight",
            global_hf_keys=["weight"],
            global_shape=(16,),
            origin_shape=(14,),
            shards=[
                ShardDescriptor(dim=0, group=etp_group, interleave_factor=2),
                ShardDescriptor(dim=0, group=fsdp_group),
            ],
        )
        plan = spec.plan_hf_save(gather_process_group=fsdp_group)

        [output] = unshard_tensors_for_hf_save([torch.tensor([12, 13, 14, 15])], [plan])

        torch.testing.assert_close(output, torch.tensor([4, 5, 6, 7, 12, 13]))
        assert plan.runtime_output_shape == (8,)
        assert plan.output_shape == (6,)


class TestCpuMergeUnshard:
    """Gated CPU-merge path (``XTUNER_UNSHARD_CPU_MERGE=1``) in ``xtuner.v1.utils.cpu_merge``.

    Every final unshard step merges its gathered chunks on host memory — no size
    threshold, no routing; intermediate steps keep the upstream device merge.
    """

    @staticmethod
    def _patch_foreach_all_gather(
        monkeypatch: pytest.MonkeyPatch,
        responses: list[list[list[torch.Tensor]]] | None = None,
    ) -> list[dict[str, object]]:
        upstream_calls: list[dict[str, object]] = []

        def fake_foreach_all_gather(
            tensor_list: list[torch.Tensor],
            group: dist.ProcessGroup,
        ) -> list[list[torch.Tensor]]:
            upstream_calls.append(
                {
                    "group": group,
                    "shapes": [tuple(tensor.shape) for tensor in tensor_list],
                    "dtypes": [tensor.dtype for tensor in tensor_list],
                }
            )
            if responses is not None:
                return responses.pop(0)
            return [[tensor] for tensor in tensor_list]

        monkeypatch.setattr(load_spec_module, "foreach_all_gather", fake_foreach_all_gather)
        return upstream_calls

    @staticmethod
    def _spy_cpu_merge_entry(monkeypatch: pytest.MonkeyPatch) -> list[int]:
        entered: list[int] = []
        real_entry = cpu_merge_module.unshard_tensors_for_hf_save_with_cpu_merge

        def spy_entry(
            tensors: list[torch.Tensor],
            save_plans: list[cpu_merge_module.HFSavePlan],
        ) -> list[torch.Tensor]:
            entered.append(len(tensors))
            return real_entry(tensors, save_plans)

        monkeypatch.setattr(cpu_merge_module, "unshard_tensors_for_hf_save_with_cpu_merge", spy_entry)
        return entered

    def test_gate_off_keeps_upstream_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        single_rank_group: dist.ProcessGroup,
    ) -> None:
        monkeypatch.delenv(cpu_merge_module.CPU_MERGE_ENV, raising=False)
        upstream_calls = self._patch_foreach_all_gather(monkeypatch)
        spec = LoadSpec(
            name="gate",
            global_hf_keys=["gate"],
            global_shape=(4, 2),
            shards=[ShardDescriptor(dim=0, group=single_rank_group)],
        )

        output = unshard_tensors_for_hf_save([torch.ones(4, 2)], [spec.plan_hf_save()])

        assert [tuple(tensor.shape) for tensor in output] == [(4, 2)]
        assert len(upstream_calls) == 1

    def test_gate_on_merges_final_step_on_host(
        self,
        monkeypatch: pytest.MonkeyPatch,
        single_rank_group: dist.ProcessGroup,
    ) -> None:
        monkeypatch.setenv(cpu_merge_module.CPU_MERGE_ENV, "1")
        upstream_calls = self._patch_foreach_all_gather(monkeypatch)
        entered = self._spy_cpu_merge_entry(monkeypatch)
        spec = LoadSpec(
            name="gate",
            global_hf_keys=["gate"],
            global_shape=(4, 2),
            shards=[ShardDescriptor(dim=0, group=single_rank_group)],
        )

        output = unshard_tensors_for_hf_save([torch.ones(4, 2)], [spec.plan_hf_save()])

        assert [tuple(tensor.shape) for tensor in output] == [(4, 2)]
        torch.testing.assert_close(output[0], torch.ones(4, 2))
        assert entered == [1]  # gate delegated into the cpu-merge orchestrator
        assert len(upstream_calls) == 1  # the upstream batched gather schedule is kept

    def test_gate_on_splits_different_dtypes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        single_rank_group: dist.ProcessGroup,
    ) -> None:
        monkeypatch.setenv(cpu_merge_module.CPU_MERGE_ENV, "1")
        upstream_calls = self._patch_foreach_all_gather(monkeypatch)
        specs = [
            LoadSpec(
                name=name,
                global_hf_keys=[name],
                global_shape=(4, 2),
                shards=[ShardDescriptor(dim=0, group=single_rank_group)],
            )
            for name in ("gate", "up")
        ]

        output = unshard_tensors_for_hf_save(
            [torch.ones(4, 2, dtype=torch.float32), torch.ones(4, 2, dtype=torch.float64)],
            [spec.plan_hf_save() for spec in specs],
        )

        assert [tuple(tensor.shape) for tensor in output] == [(4, 2), (4, 2)]
        assert [call["dtypes"] for call in upstream_calls] == [[torch.float32], [torch.float64]]

    def test_host_merge_restores_padded_runtime_shape(
        self,
        monkeypatch: pytest.MonkeyPatch,
        single_rank_group: dist.ProcessGroup,
        set_shard_rank: Callable[[int, int], None],
    ) -> None:
        set_shard_rank(3, 2)
        monkeypatch.setenv(cpu_merge_module.CPU_MERGE_ENV, "1")
        # Per-rank padded chunks of the (5,) tensor; rank 2's tail element is padding.
        self._patch_foreach_all_gather(
            monkeypatch,
            responses=[[[torch.tensor([0, 1]), torch.tensor([2, 3]), torch.tensor([4, 0])]]],
        )
        spec = LoadSpec(
            name="weight",
            global_hf_keys=["weight"],
            global_shape=(5,),
            shards=[ShardDescriptor(dim=0, group=single_rank_group)],
        )

        [output] = unshard_tensors_for_hf_save([torch.tensor([4])], [spec.plan_hf_save()])

        torch.testing.assert_close(output, torch.arange(5))

    def test_gate_on_merges_final_steps_on_host_via_batched_rounds(
        self,
        monkeypatch: pytest.MonkeyPatch,
        single_rank_group: dist.ProcessGroup,
    ) -> None:
        monkeypatch.setenv(cpu_merge_module.CPU_MERGE_ENV, "1")
        host_merge_flags: list[bool] = []
        real_host_merge = cpu_merge_module._merge_gathered_save_shard_cpu_merge

        def spy_host_merge(
            gathered_chunks: list[torch.Tensor],
            shard_step: cpu_merge_module.SaveShardStep,
            merge_on_cpu: bool,
            reusable_staging: bool,
        ) -> torch.Tensor:
            host_merge_flags.append(merge_on_cpu)
            return real_host_merge(gathered_chunks, shard_step, merge_on_cpu, reusable_staging)

        monkeypatch.setattr(cpu_merge_module, "_merge_gathered_save_shard_cpu_merge", spy_host_merge)
        upstream_calls = self._patch_foreach_all_gather(monkeypatch)
        specs = [
            LoadSpec(
                name="experts",
                global_hf_keys=["k0", "k1"],
                global_shape=(8, 2),
                fused_dim=0,
                shards=[
                    ShardDescriptor(dim=0, group=single_rank_group),
                    ShardDescriptor(dim=0, group=single_rank_group),
                ],
            ),
            LoadSpec(
                name="gate",
                global_hf_keys=["gate"],
                global_shape=(4, 2),
                shards=[ShardDescriptor(dim=0, group=single_rank_group)],
            ),
        ]

        output = unshard_tensors_for_hf_save(
            [torch.ones(8, 2), torch.ones(4, 2)],
            [spec.plan_hf_save() for spec in specs],
        )

        assert [tuple(tensor.shape) for tensor in output] == [(8, 2), (4, 2)]
        # Round 1 gathers the experts' intermediate step (device merge) and the gate's final
        # step (host merge); round 2 gathers the experts' final step (host merge) — one
        # batched foreach per round, the upstream schedule.
        assert host_merge_flags == [False, True, True]
        assert len(upstream_calls) == 2

    def test_gate_on_output_matches_upstream_preserved_ep(
        self,
        monkeypatch: pytest.MonkeyPatch,
        local_groups: tuple[dist.ProcessGroup, dist.ProcessGroup, dist.ProcessGroup],
    ) -> None:
        ep_group, etp_group, _ = local_groups
        set_group_layout(
            monkeypatch,
            {
                ep_group: (2, 1, [0, 1]),
                etp_group: (2, 0, [0, 2]),
            },
        )
        upstream_calls = self._patch_foreach_all_gather(
            monkeypatch,
            responses=[
                [[torch.tensor([4, 6]), torch.tensor([5, 7])]],
                [[torch.tensor([4, 6]), torch.tensor([5, 7])]],
            ],
        )
        spec = LoadSpec(
            name="experts",
            global_hf_keys=["k0", "k1", "k2", "k3"],
            global_shape=(8,),
            fused_dim=0,
            shards=[
                ShardDescriptor(dim=0, group=ep_group),
                ShardDescriptor(dim=0, group=etp_group, interleave_factor=2),
            ],
        )
        local = torch.tensor([4, 6])

        monkeypatch.delenv(cpu_merge_module.CPU_MERGE_ENV, raising=False)
        expected = unshard_tensors_for_hf_save([local], [spec.plan_hf_save(preserve_process_group=ep_group)])[0]

        monkeypatch.setenv(cpu_merge_module.CPU_MERGE_ENV, "1")
        gated = unshard_tensors_for_hf_save([local], [spec.plan_hf_save(preserve_process_group=ep_group)])[0]

        torch.testing.assert_close(gated, expected)
        torch.testing.assert_close(gated, torch.tensor([4, 5, 6, 7]))
        # Both calls ride the batched foreach schedule; the gated one merges the final step
        # on host chunks instead of the device merge.
        assert len(upstream_calls) == 2


class TestDumpStateDictFileSystem:
    def test_reduces_host_tensors_under_file_system_strategy(self) -> None:
        import io
        from multiprocessing.reduction import ForkingPickler

        pinned = torch.arange(8, dtype=torch.float32).pin_memory()
        previous_strategy = torch.multiprocessing.get_sharing_strategy()

        buf = io.BytesIO()
        cpu_merge_module.dump_state_dict_file_system([("w", pinned)], buf)

        buf.seek(0)
        ((name, reduction),) = ForkingPickler.loads(buf.read())
        assert name == "w"
        func, args = reduction
        rebuilt = func(*args)
        assert rebuilt.device.type == "cpu"
        torch.testing.assert_close(rebuilt, pinned.cpu())
        # the process-global sharing strategy is restored after the dump
        assert torch.multiprocessing.get_sharing_strategy() == previous_strategy


class TestNewSharedCpuTensor:
    def test_allocates_shared_segment_and_restores_strategy(self) -> None:
        previous_strategy = torch.multiprocessing.get_sharing_strategy()

        tensor = cpu_merge_module.new_shared_cpu_tensor((3, 4), torch.bfloat16)
        tensor.copy_(torch.arange(12, dtype=torch.float32).view(3, 4).to(torch.bfloat16))

        assert tensor.is_shared()
        assert tensor.dtype == torch.bfloat16
        torch.testing.assert_close(tensor, torch.arange(12, dtype=torch.float32).view(3, 4).to(torch.bfloat16))
        # the process-global sharing strategy is restored after the allocation
        assert torch.multiprocessing.get_sharing_strategy() == previous_strategy

    def test_distinct_tensors_get_distinct_segments(self) -> None:
        first = cpu_merge_module.new_shared_cpu_tensor((4,), torch.float32)
        second = cpu_merge_module.new_shared_cpu_tensor((4,), torch.float32)
        first.fill_(1.0)
        second.fill_(2.0)
        # separate segments: writing one must not leak into the other
        torch.testing.assert_close(first, torch.full((4,), 1.0))
        torch.testing.assert_close(second, torch.full((4,), 2.0))


class _StubShardStep:
    """Minimal SaveShardStep stand-in for direct shared-merge unit tests."""

    def __init__(self, dim: int, interleave_factor: int, runtime_dim_size: int) -> None:
        self.shard = type("Shard", (), {"dim": dim, "interleave_factor": interleave_factor})()
        self.shape_before_shard = {dim: runtime_dim_size}


def _stage_test_chunks(chunks: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Write chunks into a shared pool staging buffer exactly like the sync path."""
    total_bytes = sum(chunk.numel() * chunk.element_size() for chunk in chunks)
    staging = cpu_merge_module.shared_pool_alloc(total_bytes)
    views = []
    offset = 0
    for chunk in chunks:
        nbytes = chunk.numel() * chunk.element_size()
        view = staging[offset : offset + nbytes].view(chunk.dtype).view(tuple(chunk.shape))
        view.copy_(chunk)
        views.append(view)
        offset += nbytes
    return staging, views


class TestSharedPool:
    def test_alloc_slices_are_shared_and_disjoint(self) -> None:
        cpu_merge_module.reset_shared_pool()
        first = cpu_merge_module.shared_pool_alloc(8).view(torch.float32)
        second = cpu_merge_module.shared_pool_alloc(8).view(torch.float32)
        first.fill_(1.0)
        second.fill_(2.0)
        assert first.is_shared() and second.is_shared()
        torch.testing.assert_close(first, torch.full((2,), 1.0))
        torch.testing.assert_close(second, torch.full((2,), 2.0))

    def test_reset_reuses_segments_from_byte_zero(self) -> None:
        cpu_merge_module.reset_shared_pool()
        first = cpu_merge_module.shared_pool_alloc(16)
        cpu_merge_module.reset_shared_pool()
        second = cpu_merge_module.shared_pool_alloc(16)
        assert second.data_ptr() == first.data_ptr()

    def test_alloc_advances_cursor_in_dtype_aligned_steps(self) -> None:
        cpu_merge_module.reset_shared_pool()

        # an odd-size payload must not leave the cursor misaligned for later view(dtype)
        first = cpu_merge_module.shared_pool_alloc(3)
        second = cpu_merge_module.shared_pool_alloc(8).view(torch.float32)

        assert first.numel() == 3
        assert second.shape == (2,)
        assert second.data_ptr() % 4 == 0


class TestSharedMerge:
    def test_dim0_contiguous_chunks_merge_as_zero_copy_view(self) -> None:
        cpu_merge_module.reset_shared_pool()
        chunks = [torch.arange(4, dtype=torch.float32) + 10 * rank for rank in range(2)]
        staging, views = _stage_test_chunks(chunks)
        step = _StubShardStep(dim=0, interleave_factor=1, runtime_dim_size=7)

        merged = cpu_merge_module._merge_gathered_save_shard_shared(staging, views, step, staged_shared=True)

        # runtime trim (7 of 8 rows) is a prefix view of the shared staging buffer
        assert merged.is_shared()
        assert merged.shape == (7,)
        torch.testing.assert_close(merged, torch.cat(chunks)[:7])

    def test_interleave_merge_matches_upstream_math_in_shared_output(self) -> None:
        cpu_merge_module.reset_shared_pool()
        chunks = [torch.arange(4, dtype=torch.float32) + 10 * rank for rank in range(2)]
        staging, views = _stage_test_chunks(chunks)
        step = _StubShardStep(dim=0, interleave_factor=2, runtime_dim_size=8)

        merged = cpu_merge_module._merge_gathered_save_shard_shared(staging, views, step, staged_shared=True)

        expected = torch.cat([chunks[0][:2], chunks[1][:2], chunks[0][2:4], chunks[1][2:4]])
        assert merged.is_shared()
        torch.testing.assert_close(merged, expected)

    def test_dim1_merge_cats_into_shared_output_and_narrows(self) -> None:
        cpu_merge_module.reset_shared_pool()
        chunks = [torch.full((2, 3), float(rank + 1)) for rank in range(2)]
        staging, views = _stage_test_chunks(chunks)
        step = _StubShardStep(dim=1, interleave_factor=1, runtime_dim_size=5)

        merged = cpu_merge_module._merge_gathered_save_shard_shared(staging, views, step, staged_shared=True)

        assert merged.is_shared()
        assert merged.shape == (2, 5)
        torch.testing.assert_close(merged, torch.cat(chunks, dim=1)[:, :5])


class TestStageCpuStateDictToShared:
    def test_stages_cpu_tensors_into_shared_cache(self) -> None:
        payload = torch.arange(6, dtype=torch.bfloat16)

        staged = cpu_merge_module.stage_cpu_state_dict_to_shared({"w": payload})

        assert staged["w"].is_shared()
        torch.testing.assert_close(staged["w"], payload)

    def test_repeated_staging_lands_in_the_reused_pool(self) -> None:
        cpu_merge_module.reset_shared_pool()

        first = cpu_merge_module.stage_cpu_state_dict_to_shared({"w": torch.zeros(4)})["w"]
        cpu_merge_module.reset_shared_pool()
        second = cpu_merge_module.stage_cpu_state_dict_to_shared({"w": torch.ones(4)})["w"]

        # pool segments persist: after a reset the same region serves the next batch
        assert first.data_ptr() == second.data_ptr()
        assert second.is_shared()
        torch.testing.assert_close(second, torch.ones(4))

    def test_same_shape_different_names_get_distinct_segments(self) -> None:
        staged = cpu_merge_module.stage_cpu_state_dict_to_shared({"a": torch.zeros(4), "b": torch.zeros(4)})

        assert staged["a"].data_ptr() != staged["b"].data_ptr()
        torch.testing.assert_close(staged["a"], torch.zeros(4))
        torch.testing.assert_close(staged["b"], torch.zeros(4))

    def test_shared_payloads_pass_through_without_copy(self) -> None:
        shared = cpu_merge_module.new_shared_cpu_tensor((4,), torch.float32)
        shared.fill_(7.0)

        staged = cpu_merge_module.stage_cpu_state_dict_to_shared({"w": shared})

        # already-shared payload (cpu_merge shared staging) must keep its identity
        assert staged["w"] is shared


class TestGetHfParamReusableStaging:
    def _buffer_model(self) -> BaseModel:
        class BufferModel(BaseModel):
            def __init__(self) -> None:
                super().__init__(XTunerBaseModelConfig())
                self.register_buffer("rotary_coef", torch.tensor([1.25], dtype=torch.float32), persistent=True)
                self._init_load_spec()

            def to_hf_key_list(self, key: str) -> list[str]:
                return [key]

        return BufferModel()

    def test_save_path_keeps_fresh_staging_and_sync_flow_reuses_pool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import xtuner.v1.model.base as base_module

        received: list[bool] = []

        def fake_unshard(tensors, save_plans, reusable_staging=False):
            received.append(reusable_staging)
            return list(tensors)

        monkeypatch.setattr(base_module, "unshard_tensors_for_hf_save_with_cpu_merge", fake_unshard)
        monkeypatch.setenv("XTUNER_UNSHARD_CPU_MERGE", "1")
        model = self._buffer_model()

        list(model._get_hf_param(model._load_spec_params(), dtype=torch.bfloat16, distributed_save=True))
        assert received == [False]

        received.clear()
        list(model._get_hf_param(model._load_spec_params(), dtype=torch.bfloat16, reusable_staging=True))
        assert received == [True]


class TestBaseModelHFSave:
    def test_non_dtensor_buffers_keep_runtime_dtype(self) -> None:
        class BufferModel(BaseModel):
            def __init__(self) -> None:
                super().__init__(XTunerBaseModelConfig())
                self.register_buffer("rotary_coef", torch.tensor([1.25], dtype=torch.float32), persistent=True)
                self._init_load_spec()

            def to_hf_key_list(self, key: str) -> list[str]:
                return [key]

        model = BufferModel()

        [(names, tensors)] = list(
            model._get_hf_param(model._load_spec_params(), dtype=torch.bfloat16, distributed_save=True)
        )

        assert names == ["rotary_coef"]
        assert tensors[0].dtype == torch.float32
        assert torch.equal(tensors[0], model.rotary_coef)
