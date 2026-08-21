# Copyright (c) OpenMMLab. All rights reserved.
"""Regression test for the XTUNER_DEVICE_MESH custom rank layout.

Verifies the data/expert mesh rank formulas are valid permutations of
``0..world-1`` and produce the intended intra-node FSDP / inter-node EP/SP
topology. Pure arithmetic (no NPU, no process group); the only heavy
dependency is the ``xtuner.v1.utils`` import chain.
"""

import os

import pytest

from xtuner.v1.utils.device_mesh_custom import (
    NODE_SIZE,
    _rank_for_data,
    _rank_for_expert,
    _validate_data_layout,
    _validate_expert_layout,
    device_mesh_enabled,
    use_custom_mesh,
)


# 744B / 512-NPU regime: EP16 / SP32 / DP1 / tp1 / expert_tp1.
DATA_DIMS = (16, 32, 1)  # (dp_size, sp_size, tp_size)
EXPERT_DIMS = (32, 16)  # (fsdp_size, ep_size)
WORLD = 512


def _data_ranks():
    ranks = []
    for i in range(DATA_DIMS[0]):
        for j in range(DATA_DIMS[1]):
            for k in range(DATA_DIMS[2]):
                ranks.append(_rank_for_data(i, j, k, DATA_DIMS[1], DATA_DIMS[2]))
    return ranks


def _expert_ranks():
    return [_rank_for_expert(i, j, EXPERT_DIMS[1]) for i in range(EXPERT_DIMS[0]) for j in range(EXPERT_DIMS[1])]


def _node_of(rank: int) -> int:
    return rank // NODE_SIZE


class TestDeviceMeshCustom:
    def test_data_mesh_is_permutation_of_world(self):
        ranks = _data_ranks()
        assert len(ranks) == WORLD
        assert sorted(ranks) == list(range(WORLD))

    def test_expert_mesh_is_permutation_of_world(self):
        ranks = _expert_ranks()
        assert len(ranks) == WORLD
        assert sorted(ranks) == list(range(WORLD))

    def test_data_mesh_dp_group_is_intra_node(self):
        # dp (dense FSDP) group: fix sp=j, vary dp=i -> all 16 ranks in 1 node.
        for j in range(DATA_DIMS[1]):
            group = [_rank_for_data(i, j, 0, DATA_DIMS[1], DATA_DIMS[2]) for i in range(DATA_DIMS[0])]
            assert len({_node_of(r) for r in group}) == 1, f"dp group spans >1 node at sp={j}"

    def test_data_mesh_sp_group_is_inter_node(self):
        # sp group: fix dp=i, vary sp=j -> 32 ranks across 32 nodes (1 per node).
        for i in range(DATA_DIMS[0]):
            group = [_rank_for_data(i, j, 0, DATA_DIMS[1], DATA_DIMS[2]) for j in range(DATA_DIMS[1])]
            nodes = {_node_of(r) for r in group}
            assert len(nodes) == DATA_DIMS[1], f"sp group not 1-per-node at dp={i}"

    def test_expert_mesh_fsdp_group_is_two_node(self):
        # fsdp (expert FSDP) group: fix ep=j, vary fsdp=i (0..31) -> 2 nodes.
        for j in range(EXPERT_DIMS[1]):
            group = [_rank_for_expert(i, j, EXPERT_DIMS[1]) for i in range(EXPERT_DIMS[0])]
            nodes = {_node_of(r) for r in group}
            assert len(nodes) == 2, f"fsdp group spans {len(nodes)} nodes at ep={j}"
            assert len(group) == EXPERT_DIMS[0]

    def test_expert_mesh_ep_group_is_inter_node(self):
        # ep group: fix fsdp=i, vary ep=j (0..15) -> 16 nodes (1 per node).
        for i in range(EXPERT_DIMS[0]):
            group = [_rank_for_expert(i, j, EXPERT_DIMS[1]) for j in range(EXPERT_DIMS[1])]
            nodes = {_node_of(r) for r in group}
            assert len(nodes) == EXPERT_DIMS[1], f"ep group not 1-per-node at fsdp={i}"

    def test_expert_mesh_ep32_intra_node_fsdp(self):
        # EP32 on 512: fsdp=16 (=NODE_SIZE), ep=32. fsdp group = 16 ranks in 1
        # node; ep group = 32 ranks across 32 nodes (1 per node).
        fsdp, ep = 16, 32
        ranks = [_rank_for_expert(i, j, ep, fsdp) for i in range(fsdp) for j in range(ep)]
        assert sorted(ranks) == list(range(WORLD)), "EP32 expert mesh not a permutation of 0..511"
        for j in range(ep):
            group = [_rank_for_expert(i, j, ep, fsdp) for i in range(fsdp)]
            assert len({_node_of(r) for r in group}) == 1, f"EP32 fsdp group spans >1 node at ep={j}"
            assert len(group) == fsdp
        for i in range(fsdp):
            group = [_rank_for_expert(i, j, ep, fsdp) for j in range(ep)]
            assert len({_node_of(r) for r in group}) == ep, f"EP32 ep group not 1-per-node at fsdp={i}"

    def test_expert_mesh_ep64_intra_node_fsdp(self):
        # EP64 on 512: fsdp=8 (<NODE_SIZE), ep=64. fsdp group = 8 ranks in 1
        # node (zero-copy intra-node FSDP); ep group = 64 ranks across 32
        # nodes (2 per node). Validates the fsdp<=NODE_SIZE generalization.
        fsdp, ep = 8, 64
        ranks = [_rank_for_expert(i, j, ep, fsdp) for i in range(fsdp) for j in range(ep)]
        assert sorted(ranks) == list(range(WORLD)), "EP64 expert mesh not a permutation of 0..511"
        for j in range(ep):
            group = [_rank_for_expert(i, j, ep, fsdp) for i in range(fsdp)]
            assert len({_node_of(r) for r in group}) == 1, f"EP64 fsdp group spans >1 node at ep={j}"
            assert len(group) == fsdp
        for i in range(fsdp):
            group = [_rank_for_expert(i, j, ep, fsdp) for j in range(ep)]
            nodes = {_node_of(r) for r in group}
            assert len(nodes) == 32, f"EP64 ep group not spanning 32 nodes at fsdp={i}"
            assert len(group) == ep

    def test_ep32_4arg_equals_legacy_3arg(self):
        # Backward-compat: EP32 (fsdp=16=NODE_SIZE) new branch must equal the
        # legacy 3-arg formula byte-for-byte (EP32 is the validated config).
        fsdp, ep = 16, 32
        for i in range(fsdp):
            for j in range(ep):
                assert _rank_for_expert(i, j, ep, fsdp) == _rank_for_expert(i, j, ep), (
                    f"EP32 layout drift at (i={i}, j={j})"
                )

    def test_default_disabled(self):
        # Without XTUNER_DEVICE_MESH the feature is off (default-identical path).
        old = os.environ.pop("XTUNER_DEVICE_MESH", None)
        try:
            assert device_mesh_enabled() is False
        finally:
            if old is not None:
                os.environ["XTUNER_DEVICE_MESH"] = old

    def test_use_custom_mesh_gate(self):
        # Gate = env on AND world spans strictly more than one node (NODE_SIZE=16).
        for ws in (1, 16, 17, 512):
            os.environ["XTUNER_DEVICE_MESH"] = "0"
            assert use_custom_mesh(ws) is False, f"gate must be off when env=0 (ws={ws})"
        os.environ["XTUNER_DEVICE_MESH"] = "1"
        try:
            assert use_custom_mesh(1) is False, "single-node world never uses custom layout"
            assert use_custom_mesh(16) is False, "ws==NODE_SIZE is single-node, not > "
            assert use_custom_mesh(17) is True, "ws>NODE_SIZE with env=1 must use custom"
            assert use_custom_mesh(512) is True
        finally:
            os.environ.pop("XTUNER_DEVICE_MESH", None)

    # --- divisibility guards (regression for the silent non-permutation bug) ---

    def test_data_formula_non_permutation_when_dp_not_divisible(self):
        # Regression: dp_size not a multiple of NODE_SIZE silently produced a
        # non-permutation before the guard. world=256, sp=32, tp=1 -> dp=8.
        dp, sp, tp = 8, 32, 1
        ranks = [_rank_for_data(i, j, k, sp, tp) for i in range(dp) for j in range(sp) for k in range(tp)]
        assert len(ranks) == 256
        assert sorted(ranks) != list(range(256)), "dp=8 must be a non-permutation"

    def test_data_layout_rejects_non_divisible_dp(self):
        # The guard must fail fast on the buggy dp=8 instead of building it.
        with pytest.raises(ValueError, match="dp_size"):
            _validate_data_layout(8, 32, 1)

    def test_data_layout_accepts_divisible_dp(self):
        # dp_size a multiple of NODE_SIZE is the only valid data-layout case.
        _validate_data_layout(16, 32, 1)  # 512-rank regime (job-44/45)
        _validate_data_layout(32, 32, 1)  # 1024-rank
        _validate_data_layout(48, 16, 1)  # 768-rank, non-power-of-2 multiple

    def test_expert_formula_non_permutation_when_fsdp_not_divisor(self):
        # Regression: fsdp not a divisor of NODE_SIZE (new branch) silently
        # produced a non-permutation. world=96, EP16 -> fsdp=6.
        fsdp, ep = 6, 16
        ranks = [_rank_for_expert(i, j, ep, fsdp) for i in range(fsdp) for j in range(ep)]
        assert len(ranks) == 96
        assert sorted(ranks) != list(range(96)), "fsdp=6 must be a non-permutation"

    def test_expert_formula_non_permutation_when_fsdp_not_multiple(self):
        # Regression: fsdp > NODE_SIZE but not a multiple (legacy branch).
        # world=384, EP16 -> fsdp=24.
        fsdp, ep = 24, 16
        ranks = [_rank_for_expert(i, j, ep, fsdp) for i in range(fsdp) for j in range(ep)]
        assert len(ranks) == 384
        assert sorted(ranks) != list(range(384)), "fsdp=24 must be a non-permutation"

    def test_expert_layout_rejects_non_divisor_small_fsdp(self):
        # fsdp <= NODE_SIZE but not a divisor -> new branch breaks; guard fires.
        with pytest.raises(ValueError, match="divisible by fsdp_size"):
            _validate_expert_layout(6, 16)

    def test_expert_layout_rejects_non_multiple_large_fsdp(self):
        # fsdp > NODE_SIZE but not a multiple -> legacy branch breaks; guard fires.
        with pytest.raises(ValueError, match="multiple of NODE_SIZE"):
            _validate_expert_layout(24, 16)

    def test_expert_layout_accepts_valid_configs(self):
        # 512-rank regime: EP16->fsdp32, EP32->fsdp16, EP64->fsdp8 all valid.
        _validate_expert_layout(32, 16)
        _validate_expert_layout(16, 32)
        _validate_expert_layout(8, 64)
