"""Unit tests for ``MoE._ep_load_reduce``, the on-device EP-load imbalance reduction.

The optimized reduction consumes the per-step all_gather matrix with pure view/reduce ops and relies on EP
groups being contiguous world-rank blocks in the gather order. The reference here is the pre-merge CPU
implementation, which grouped rows by the gathered group-id column instead — bit equality between the two
pins both the metric values and the layout assumption.
"""

import torch

from xtuner.v1.model.moe.moe import _ep_load_reduce


def _reference_reduce(gathered: torch.Tensor, ep_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Pre-merge CPU implementation: unique on the group-id column + index_add_/index_reduce_.
    _, group_idx = gathered[:, 0].unique(return_inverse=True)
    recv = gathered[:, 1:].double()
    n_groups = int(group_idx.max()) + 1
    group_sum = recv.new_zeros(n_groups, recv.shape[1]).index_add_(0, group_idx, recv)
    group_max = recv.new_zeros(n_groups, recv.shape[1]).index_reduce_(0, group_idx, recv, "amax")
    fair_share = group_sum / ep_size
    load_ratio = recv.sum(dim=1) / fair_share.sum(dim=1)[group_idx]
    peak_ratio = (recv / fair_share[group_idx]).amax(dim=1)
    straggler_ratio = group_max.sum(dim=1) / fair_share.sum(dim=1)
    return load_ratio, peak_ratio, straggler_ratio


def _gathered_fixture(world: int, ep_size: int, n_dispatch: int, skew: dict[int, float], seed: int) -> torch.Tensor:
    # [world, 1 + n_dispatch] with contiguous-block group ids (column 0 = the group's lowest rank) and
    # per-rank skewed row counts; skew multipliers emulate hot/straggler ranks.
    gen = torch.Generator().manual_seed(seed)
    counts = torch.randint(1, 5000, (world, n_dispatch), generator=gen)
    for rank, factor in skew.items():
        counts[rank] = (counts[rank].double() * factor).to(torch.int64)
    gids = (torch.arange(world) // ep_size) * ep_size
    return torch.cat([gids.unsqueeze(1), counts], dim=1)


class TestEpLoadReduce:
    def test_matches_cpu_reference_on_skewed_routing(self):
        # Box shape (2 EP groups of 8), CI E2E shape (2 groups of 2), and the single-group boundary
        # (world == ep_size). Hot ranks 3/11 in the first two cases emulate stragglers.
        cases = [
            (16, 8, 37, {3: 2.0, 11: 3.0}),
            (4, 2, 13, {1: 2.5}),
            (8, 8, 1, {}),
        ]
        for world, ep_size, n_dispatch, skew in cases:
            gathered = _gathered_fixture(world, ep_size, n_dispatch, skew, seed=world + n_dispatch)
            got = _ep_load_reduce(gathered, ep_size)
            want = _reference_reduce(gathered, ep_size)
            for name, a, b in zip(("load_ratio", "peak_ratio", "straggler_ratio"), got, want):
                assert torch.equal(a, b), f"{name} diverges for world={world} ep_size={ep_size}"
                assert a.dtype == torch.float64
            assert got[0].shape == (world,) and got[1].shape == (world,) and got[2].shape == (world // ep_size,)

    def test_trainer_fields_derive_from_reference(self):
        # The 7 logged scalars are max/argmax/per-rank reads of the three ratios; pin them against the
        # reference for a skewed fixture, computing the derived fields the way both code paths do.
        gathered = _gathered_fixture(16, 8, 37, {3: 2.0, 11: 3.0}, seed=53)
        new = _ep_load_reduce(gathered, 8)
        ref = _reference_reduce(gathered, 8)
        for a, b in zip(new, ref):
            assert a.max() == b.max()
            assert int(a.argmax()) == int(b.argmax())
            assert a[0] == b[0] and a[-1] == b[-1]

    def test_balanced_routing_yields_unity_ratios(self):
        # Every rank of a group receiving identical per-dispatch counts is perfectly balanced: all three
        # ratios must be exactly 1.0 (exact in float64 for power-of-two group sizes).
        ep_size, n_dispatch = 8, 37
        gen = torch.Generator().manual_seed(7)
        per_group = torch.randint(1, 5000, (2, n_dispatch), generator=gen)
        counts = per_group.repeat_interleave(ep_size, dim=0)
        gids = (torch.arange(2 * ep_size) // ep_size) * ep_size
        gathered = torch.cat([gids.unsqueeze(1), counts], dim=1)

        load, peak, straggler = _ep_load_reduce(gathered, ep_size)

        assert torch.all(load == 1.0)
        assert torch.all(peak == 1.0)
        assert torch.all(straggler == 1.0)

    def test_two_rank_group_exact_values(self):
        # Group of 2 with rows (3, 1): the fair share is 2.0, so the loaded rank sits at 1.5x across all
        # step-level views and the empty-side rank at 0.5x.
        gathered = torch.tensor([[0, 3], [0, 1]], dtype=torch.int64)

        load, peak, straggler = _ep_load_reduce(gathered, 2)

        assert torch.equal(load, torch.tensor([1.5, 0.5], dtype=torch.float64))
        assert torch.equal(peak, torch.tensor([1.5, 0.5], dtype=torch.float64))
        assert torch.equal(straggler, torch.tensor([1.5], dtype=torch.float64))
