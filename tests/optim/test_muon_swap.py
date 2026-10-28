# Copyright (c) OpenMMLab. All rights reserved.
"""Muon pinned-host state swap: the deferred-drain gate and swap bit-exactness."""

import os
from unittest import mock

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.placement_types import Replicate, Shard

from xtuner._testing.testcase import DeterministicDDPTestCase
from xtuner.v1.optim import muon as muon_module
from xtuner.v1.optim.muon import Muon


GATE_ENV = "XTUNER_MUON_SWAP_DEFERRED_DRAIN"


class _RunTwoSteps:
    """Build a two-group Muon optimizer, run two steps on fixed grads, return the readout.

    The optimizer is built inside the env gate, which ``Muon`` resolves once at construction.
    """

    @staticmethod
    def run(swap_optimizer: bool, deferred_drain: bool) -> tuple[torch.Tensor, torch.Tensor]:
        env = {GATE_ENV: "1"} if deferred_drain else {}
        with mock.patch.dict(os.environ, env, clear=False):
            if not deferred_drain:
                os.environ.pop(GATE_ENV, None)
            # muon step() assumes DTensor parameters (group_tensors_by_device_mesh_and_placements
            # asserts on plain tensors), so mirror training: 1-rank mesh, muon group sharded
            # on dim 0, adamw group replicated.
            mesh = init_device_mesh("cuda", (1,))
            torch.manual_seed(7)
            # Keep the DTensor references for the final readout: wrapping into
            # ``nn.Parameter`` erases the DTensor type for the static checker.
            w2d_dt = distribute_tensor(torch.randn(128, 96, device="cuda", dtype=torch.float32), mesh, (Shard(0),))
            w1d_dt = distribute_tensor(torch.randn(128, device="cuda", dtype=torch.float32), mesh, (Replicate(),))
            w2d = torch.nn.Parameter(w2d_dt)
            w1d = torch.nn.Parameter(w1d_dt)
            opt = Muon(
                [
                    dict(params=[w2d], algorithm="muon", clip_grad=False),
                    dict(params=[w1d], algorithm="adamw", clip_grad=True),
                ],
                lr=1e-2,
                mu=0.95,
                betas=(0.9, 0.95),
                weight_decay=0.1,
                nesterov=True,
                flatten=True,
                adjust_lr="rms_norm",
                use_triton=False,
                epsilon=1e-8,
                enable_all2all=False,
                remainder_strategy="pad_all2all",
                swap_optimizer=swap_optimizer,
            )
            torch.manual_seed(11)
            grads = [
                (
                    torch.randn(128, 96, device="cuda", dtype=torch.float32),
                    torch.randn(128, device="cuda", dtype=torch.float32),
                )
                for _ in range(2)
            ]
            for g2, g1 in grads:
                opt.zero_grad(set_to_none=True)
                w2d.grad = distribute_tensor(g2.clone(), mesh, (Shard(0),))
                w1d.grad = distribute_tensor(g1.clone(), mesh, (Replicate(),))
                # With deferred drain the tail D2H stays in flight across the step boundary;
                # the next step's H2D is enqueued behind it on the same stream.
                opt.step()
            return w2d_dt.to_local().clone(), w1d_dt.to_local().clone()


class TestMuonSwapBitExact(DeterministicDDPTestCase):
    """Swap on/off must produce bit-identical updates for the same grad
    sequence, in the same process (cross-run bit gates are invalid on this
    stack -- run39/run40 s1 grad_norm already differ with identical code,
    backward matmul atomics). Two steps so the second update consumes a
    D2H-roundtripped momentum, which is the only path the swap can alter."""

    @property
    def world_size(self) -> int:
        return 1

    def test_swap_on_matches_swap_off_bitwise(self) -> None:
        """Two-step updates with swap on must equal swap off bit-for-bit."""
        self.create_pg("cuda")
        w2d_off, w1d_off = _RunTwoSteps.run(swap_optimizer=False, deferred_drain=False)
        w2d_on, w1d_on = _RunTwoSteps.run(swap_optimizer=True, deferred_drain=True)
        assert torch.equal(w2d_on, w2d_off)
        assert torch.equal(w1d_on, w1d_off)

    def test_deferred_drain_matches_step_end_sync_bitwise(self) -> None:
        """Leaving the tail D2H in flight must not change the second step's update."""
        self.create_pg("cuda")
        w2d_sync, w1d_sync = _RunTwoSteps.run(swap_optimizer=True, deferred_drain=False)
        w2d_defer, w1d_defer = _RunTwoSteps.run(swap_optimizer=True, deferred_drain=True)
        assert torch.equal(w2d_defer, w2d_sync)
        assert torch.equal(w1d_defer, w1d_sync)


class TestMuonSwapDeferredDrainGate(DeterministicDDPTestCase):
    """The gate layers onto ``MuonConfig.swap_optimizer`` and must be inert without it."""

    @property
    def world_size(self) -> int:
        return 1

    @staticmethod
    def _build_optimizer(swap_optimizer: bool) -> tuple[Muon, torch.Tensor]:
        mesh = init_device_mesh("cuda", (1,))
        torch.manual_seed(7)
        w2d_dt = distribute_tensor(torch.randn(128, 96, device="cuda", dtype=torch.float32), mesh, (Shard(0),))
        w2d = torch.nn.Parameter(w2d_dt)
        w2d.grad = distribute_tensor(torch.randn(128, 96, device="cuda"), mesh, (Shard(0),))
        opt = Muon(
            [dict(params=[w2d], algorithm="muon", clip_grad=False)],
            lr=1e-2,
            mu=0.95,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            nesterov=True,
            flatten=True,
            adjust_lr="rms_norm",
            use_triton=False,
            epsilon=1e-8,
            enable_all2all=False,
            remainder_strategy="pad_all2all",
            swap_optimizer=swap_optimizer,
        )
        return opt, w2d

    def test_gate_is_inert_without_swap_optimizer(self) -> None:
        """The env gate alone must not enable the swap; the state stays device-resident."""
        self.create_pg("cuda")
        with mock.patch.dict(os.environ, {GATE_ENV: "1"}, clear=False):
            opt, w2d = self._build_optimizer(swap_optimizer=False)
            assert opt._swap_optimizer is False
            assert opt._defer_host_drain is False
            assert not opt._optimizer_state_dict_pre_hooks
            state = opt._get_or_initialize_state(w2d, "muon")
            assert state["momentum"].device.type != "cpu"

    def test_gate_registers_drain_hooks_on_both_reader_paths(self) -> None:
        """With the gate on, the state is pinned and both host-reader paths drain."""
        self.create_pg("cuda")
        with mock.patch.dict(os.environ, {GATE_ENV: "1"}, clear=False):
            opt, w2d = self._build_optimizer(swap_optimizer=True)
            assert opt._defer_host_drain is True
            assert len(opt._optimizer_state_dict_pre_hooks) == 1
            # The clip-grad restore hook is registered unconditionally; ours is the second.
            assert len(opt._optimizer_load_state_dict_pre_hooks) == 2
            assert opt._get_or_initialize_state(w2d, "muon")["momentum"].device.type == "cpu"
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(GATE_ENV, None)
            opt, _ = self._build_optimizer(swap_optimizer=True)
            assert opt._defer_host_drain is False
            assert not opt._optimizer_state_dict_pre_hooks
            assert len(opt._optimizer_load_state_dict_pre_hooks) == 1

    @staticmethod
    def _count_syncs() -> tuple[object, list[int]]:
        """Patch the device module's sync so it still runs and is counted."""
        device_module = muon_module.DEVICE_MODULE
        original = device_module.synchronize
        calls: list[int] = []

        def counting_sync(*args: object, **kwargs: object) -> None:
            calls.append(1)
            original(*args, **kwargs)

        return mock.patch.object(device_module, "synchronize", counting_sync), calls

    def test_step_end_sync_is_kept_when_the_gate_is_off(self) -> None:
        """Default behaviour is unchanged: ``step`` drains the device before returning."""
        self.create_pg("cuda")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(GATE_ENV, None)
            opt, _ = self._build_optimizer(swap_optimizer=True)
            patcher, syncs = self._count_syncs()
            with patcher:
                opt.step()
                assert len(syncs) >= 1, "step() must drain the device when the gate is off"
                before = len(syncs)
                opt.state_dict()
                assert len(syncs) == before, "no extra drain is registered when the gate is off"

    def test_step_leaves_tail_d2h_in_flight_and_readers_drain(self) -> None:
        """Gate on: no drain in ``step``; ``state_dict`` and ``load_state_dict`` fence once."""
        self.create_pg("cuda")
        with mock.patch.dict(os.environ, {GATE_ENV: "1"}, clear=False):
            opt, _ = self._build_optimizer(swap_optimizer=True)
            patcher, syncs = self._count_syncs()
            with patcher:
                opt.step()
                assert not syncs, "step() must not block on the tail D2H when the gate is on"
                opt.state_dict()
                assert len(syncs) == 1, "state_dict() must drain before exposing the host state"
                state_dict = opt.state_dict()
                assert len(syncs) == 2, "every state_dict() call drains"
                opt.load_state_dict(state_dict)
                assert len(syncs) == 3, "load_state_dict() must drain before writing host state"
