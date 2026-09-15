"""Regression tests for LengthGroupedSampler cycling (epoch-boundary data stall fix)."""

import itertools

import pytest

from xtuner.v1.datasets.sampler import LengthGroupedSampler, ParallelSampler


class _FakePackDataset:
    """Minimal stand-in: LengthGroupedSampler only needs ``len()`` and ``longest``."""

    def __init__(self, num_packs: int):
        self.longest = [(i * 37) % 101 + 5 for i in range(num_packs)]

    def __len__(self) -> int:
        return len(self.longest)


class TestLengthGroupedSamplerCycle:
    def _sampler(self, cycle: bool | None = None, num_packs: int = 12) -> LengthGroupedSampler:
        sampler = LengthGroupedSampler(_FakePackDataset(num_packs), global_batch_size=3, seed=123)
        if cycle is not None:
            sampler.set_cycle(cycle)
        return sampler

    def test_default_stops_after_single_epoch(self):
        """Without cycling, the iterator raises StopIteration after one epoch and keeps state untouched."""
        sampler = self._sampler()
        it = iter(sampler)
        assert len(list(it)) == sampler.num_samples
        assert next(it, None) is None
        assert sampler.epoch == 0
        assert sampler.step == 0

    def test_cycle_iterates_indefinitely_with_fresh_shuffle_per_epoch(self):
        """Cycling never raises StopIteration and each epoch equals a fresh set_epoch(e) iteration."""
        sampler = self._sampler(cycle=True)
        stream = list(itertools.islice(iter(sampler), 3 * sampler.num_samples + 5))
        assert len(stream) == 3 * sampler.num_samples + 5
        for epoch in range(3):
            fresh = self._sampler()
            fresh.set_epoch(epoch)
            expected = list(fresh)
            assert stream[epoch * sampler.num_samples : (epoch + 1) * sampler.num_samples] == expected

    def test_cycle_is_deterministic(self):
        """Two cycling samplers with the same seed yield the same infinite stream prefix."""
        first = list(itertools.islice(iter(self._sampler(cycle=True)), 3 * self._sampler().num_samples))
        second = list(itertools.islice(iter(self._sampler(cycle=True)), 3 * self._sampler().num_samples))
        assert first == second

    def test_set_cycle_false_restores_single_epoch(self):
        """set_cycle(False) restores the legacy single-epoch behavior."""
        sampler = self._sampler(cycle=True)
        sampler.set_cycle(False)
        it = iter(sampler)
        assert len(list(it)) == sampler.num_samples
        assert next(it, None) is None
        assert sampler.epoch == 0

    def test_parallel_sampler_has_no_set_cycle(self):
        """ParallelSampler stays untouched, so the legacy trainer rebuild path remains live for it."""
        sampler = ParallelSampler(_FakePackDataset(12), global_batch_size=3, seed=123)
        assert not hasattr(sampler, "set_cycle")

    def test_resume_continues_consumed_stream_exactly(self):
        """Under cycling, get_state_dict saves the consumed epoch, so resume is sample-exact even when
        DataLoader prefetch pulls ahead across the epoch boundary."""
        dataset = _FakePackDataset(12)
        reference = LengthGroupedSampler(dataset, global_batch_size=3, seed=123)
        reference.set_cycle(True)
        truth = list(itertools.islice(iter(reference), 40))

        simulated = LengthGroupedSampler(dataset, global_batch_size=3, seed=123)
        simulated.set_cycle(True)
        it = iter(simulated)
        consumed = [next(it) for _ in range(8)]
        for _ in range(6):
            next(it)  # prefetch lookahead crosses the epoch boundary
        assert simulated.epoch == 1

        state = simulated.get_state_dict(total_consumed_steps=8)
        assert state["epoch"] == 0  # consumed epoch, not the prefetch-ahead one
        assert state["step"] == 8

        resumed = LengthGroupedSampler(dataset, global_batch_size=3, seed=123)
        resumed.load_state_dict(state)
        resumed.set_cycle(True)
        rest = list(itertools.islice(iter(resumed), 40 - 8))
        assert consumed + rest == truth

    def test_get_state_dict_legacy_epoch_unchanged(self):
        """Without cycling, get_state_dict keeps saving the set_epoch-driven epoch as before."""
        sampler = self._sampler()
        sampler.set_epoch(2)
        state = sampler.get_state_dict(total_consumed_steps=8)
        assert state["epoch"] == 2
        assert state["step"] == 8 % sampler.total_size


if __name__ == "__main__":
    pytest.main([__file__])
