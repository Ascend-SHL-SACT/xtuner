# Copyright (c) OpenMMLab. All rights reserved.
"""muon_swap tests: env caching, the host-side D2H completion gate, and swap bit-exactness."""

import os

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.placement_types import Replicate, Shard

from xtuner._testing.testcase import DeterministicDDPTestCase
from xtuner.v1.optim import muon_swap
from xtuner.v1.optim.muon import Muon


class TestMuonSwapEnvAndGate:
    def test_use_copy_stream_env_resolved_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The overlap env var must be resolved once, not on every hot-path call."""
        monkeypatch.setattr(muon_swap, "_USE_COPY_STREAM", None)
        monkeypatch.setenv("XTUNER_MUON_SWAP_OVERLAP", "0")
        assert muon_swap._use_copy_stream() is False
        monkeypatch.setenv("XTUNER_MUON_SWAP_OVERLAP", "1")
        assert muon_swap._use_copy_stream() is False

    def test_host_sync_d2h_noop_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """host_sync_d2h must not construct the swap ctx when no swap ever ran."""
        monkeypatch.setattr(muon_swap, "_SWAP_CTX", None)
        muon_swap.host_sync_d2h()
        assert muon_swap._SWAP_CTX is None


class TestMuonSwapHostSync(DeterministicDDPTestCase):
    @property
    def world_size(self) -> int:
        return 1

    def test_host_sync_d2h_delivers_copy(self) -> None:
        """host_sync_d2h must block until the copy-stream D2H has landed."""
        old_cache = muon_swap._USE_COPY_STREAM
        muon_swap._USE_COPY_STREAM = True  # copy-stream path regardless of the ambient env
        try:
            self.create_pg("cuda")
            device = torch.device("cuda")
            src = torch.arange(0, 4096, device=device, dtype=torch.float32)
            sink = torch.zeros_like(src, device="cpu").pin_memory()
            muon_swap.d2h_momentum([src], [sink])
            assert muon_swap._SWAP_CTX is not None  # the host gate below is live, not a no-op
            muon_swap.host_sync_d2h()
            assert torch.equal(sink.to(device), src)
        finally:
            muon_swap._USE_COPY_STREAM = old_cache

    def test_host_sync_d2h_delivers_fifo_copy(self) -> None:
        """The FIFO fallback must also record d2h_ev so the host gate works."""
        old_cache = muon_swap._USE_COPY_STREAM
        old_ctx = muon_swap._SWAP_CTX
        old_env = os.environ.get("XTUNER_MUON_SWAP_OVERLAP")
        os.environ["XTUNER_MUON_SWAP_OVERLAP"] = "0"
        muon_swap._USE_COPY_STREAM = None
        try:
            self.create_pg("cuda")
            device = torch.device("cuda")
            src = torch.full((4096,), 7.0, device=device, dtype=torch.float32)
            sink = torch.zeros_like(src, device="cpu").pin_memory()
            muon_swap.d2h_momentum([src], [sink])
            assert muon_swap._SWAP_CTX is not None  # the host gate below is live, not a no-op
            muon_swap.host_sync_d2h()
            assert torch.equal(sink.to(device), src)
        finally:
            muon_swap._USE_COPY_STREAM = old_cache
            muon_swap._SWAP_CTX = old_ctx
            if old_env is None:
                del os.environ["XTUNER_MUON_SWAP_OVERLAP"]
            else:
                os.environ["XTUNER_MUON_SWAP_OVERLAP"] = old_env


class TestMuonSwapBitExact(DeterministicDDPTestCase):
    """Swap on/off must produce bit-identical updates for the same grad
    sequence, in the same process (cross-run bit gates are invalid on this
    stack -- run39/run40 s1 grad_norm already differ with identical code,
    backward matmul atomics). Two steps so the second update consumes a
    D2H-roundtripped momentum, which is the only path the swap can alter."""

    @property
    def world_size(self) -> int:
        return 1

    @staticmethod
    def _build_and_run_two_steps(swap: bool) -> tuple[torch.Tensor, torch.Tensor]:
        old_env = os.environ.get("XTUNER_MUON_SWAP")
        os.environ["XTUNER_MUON_SWAP"] = "1" if swap else "0"
        old_cache = muon_swap._USE_COPY_STREAM
        old_ctx = muon_swap._SWAP_CTX
        muon_swap._USE_COPY_STREAM = None
        muon_swap._SWAP_CTX = None
        try:
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
            groups = [
                dict(params=[w2d], algorithm="muon", clip_grad=False),
                dict(params=[w1d], algorithm="adamw", clip_grad=True),
            ]
            opt = Muon(
                groups,
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
            )
            torch.manual_seed(11)
            grads = []
            for _ in range(2):
                g2 = torch.randn(128, 96, device="cuda", dtype=torch.float32)
                g1 = torch.randn(128, device="cuda", dtype=torch.float32)
                grads.append((g2, g1))
            for g2, g1 in grads:
                opt.zero_grad(set_to_none=True)
                w2d.grad = distribute_tensor(g2.clone(), mesh, (Shard(0),))
                w1d.grad = distribute_tensor(g1.clone(), mesh, (Replicate(),))
                opt.step()
                muon_swap.host_sync_d2h()
            return w2d_dt.to_local().clone(), w1d_dt.to_local().clone()
        finally:
            if old_env is None:
                os.environ.pop("XTUNER_MUON_SWAP", None)
            else:
                os.environ["XTUNER_MUON_SWAP"] = old_env
            muon_swap._USE_COPY_STREAM = old_cache
            muon_swap._SWAP_CTX = old_ctx

    def test_swap_on_matches_swap_off_bitwise(self) -> None:
        """Two-step updates with swap on must equal swap off bit-for-bit."""
        self.create_pg("cuda")
        w2d_off, w1d_off = self._build_and_run_two_steps(swap=False)
        w2d_on, w1d_on = self._build_and_run_two_steps(swap=True)
        assert torch.equal(w2d_on, w2d_off)
        assert torch.equal(w1d_on, w1d_off)
