"""Halo MTP future embeddings must match the full-sequence gather bit for bit.

The halo path (``XTUNER_MTP_HALO=1``) replaces the full-sequence
``raw_inputs_embeds`` gather with the local shard plus a next-rank halo
fragment. These tests pin three equivalence levels: the packed-indexing
function against ``local_packed_future_embeddings`` on a shared full sequence,
the MTP block forward against the full-gather path under a fake SP mesh, and a
real two-rank gloo SP run against both the full-gather path and the pack-aware
roll reference, gradients included.
"""

import os
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.module.mtp import MTPBlock, MTPConfig
from xtuner.v1.module.mtp.halo import (
    HaloFutureEmbeds,
    build_halo_future_embeds,
    halo_packed_future_embeddings,
    mtp_halo_enabled,
)
from xtuner.v1.module.mtp.utils import local_packed_future_embeddings, roll_packed_tensor


BOUNDARIES = torch.tensor([0, 2, 2, 9, 10, 17, 20], dtype=torch.int32)
HALO = 7


def test_halo_gate_reads_the_environment(monkeypatch):
    monkeypatch.delenv("XTUNER_MTP_HALO", raising=False)
    assert mtp_halo_enabled() is False
    monkeypatch.setenv("XTUNER_MTP_HALO", "1")
    assert mtp_halo_enabled() is True
    monkeypatch.setenv("XTUNER_MTP_HALO", "0")
    assert mtp_halo_enabled() is False


@pytest.mark.parametrize("depth", range(1, HALO + 1))
@pytest.mark.parametrize("shard_start", [0, 8, 16])
def test_halo_packed_future_matches_full_gather_values_and_grads(depth, shard_start):
    shard_size = 4
    full = (torch.arange(60, dtype=torch.float64).reshape(1, 20, 3) / 100).requires_grad_()
    halo_source = full.detach().clone().requires_grad_()
    reference_source = full.detach().clone().requires_grad_()
    # A perfect neighbor exchange materializes the halo straight from the full
    # sequence; past the last rank the fragment is wrap-around garbage that the
    # pack-end mask never surfaces, so zero fill stands in for it here.
    halo_begin = shard_start + shard_size
    halo_fragment = halo_source[:, halo_begin : min(halo_begin + HALO, 20)]
    pad = halo_source.new_zeros(1, HALO - halo_fragment.shape[1], 3)
    halo = HaloFutureEmbeds(
        buffer=torch.cat([halo_source[:, shard_start:halo_begin], halo_fragment, pad], dim=1),
        shard_start=shard_start,
        shard_size=shard_size,
    )
    future = halo_packed_future_embeddings(halo, BOUNDARIES, depth=depth)
    reference = local_packed_future_embeddings(
        reference_source, BOUNDARIES, depth=depth, shard_start=shard_start, shard_size=shard_size
    )
    torch.testing.assert_close(future, reference, rtol=0, atol=0)
    assert future.is_contiguous() and future.storage_offset() == 0
    assert future.untyped_storage().data_ptr() != halo.buffer.untyped_storage().data_ptr()
    future.float().square().sum().backward()
    reference.float().square().sum().backward()
    read_end = min(shard_start + shard_size + depth, 20)
    torch.testing.assert_close(
        halo_source.grad[:, shard_start:read_end], reference_source.grad[:, shard_start:read_end], rtol=0, atol=0
    )
    rest = torch.cat([halo_source.grad[:, :shard_start], halo_source.grad[:, read_end:]], dim=1)
    assert rest.abs().sum() == 0


def test_halo_build_without_mesh_keeps_a_local_zero_tail():
    embeds = torch.arange(24, dtype=torch.float64).reshape(1, 4, 6).requires_grad_()
    ctx = SequenceContext(
        input_ids=None,
        cu_seq_lens_q=BOUNDARIES,
        cu_seq_lens_k=BOUNDARIES,
        max_length_q=7,
        max_length_k=7,
        inputs_embeds=embeds,
    )
    halo = build_halo_future_embeds(ctx, halo_size=2)
    assert halo.shard_start == 0 and halo.shard_size == 4
    assert halo.buffer.shape == (1, 6, 6)
    torch.testing.assert_close(halo.buffer[:, :4], embeds, rtol=0, atol=0)
    torch.testing.assert_close(halo.buffer[:, 4:], torch.zeros(1, 2, 6, dtype=torch.float64), rtol=0, atol=0)


def test_halo_build_rejects_shards_narrower_than_the_halo():
    ctx = SequenceContext(
        input_ids=None,
        cu_seq_lens_q=BOUNDARIES,
        cu_seq_lens_k=BOUNDARIES,
        max_length_q=7,
        max_length_k=7,
        inputs_embeds=torch.zeros(1, 2, 3),
        position_ids=torch.zeros(1, 2, dtype=torch.long),
        sequence_parallel_mesh=_Mesh(0),
    )
    with pytest.raises(ValueError, match="shard_size"):
        build_halo_future_embeds(ctx, halo_size=HALO)


class _Mesh:
    def __init__(self, rank):
        self.rank = rank

    def size(self):
        return 5

    def get_local_rank(self):
        return self.rank

    def get_group(self):
        return None


class _CheckpointLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5, dtype=torch.float64))
        self.futures = []
        self.contexts = []

    def _compute(self, hidden, future, ctx):
        return (self.weight * hidden + future).tanh() * ctx.mask.unsqueeze(-1)

    def forward(self, hidden_states, *, future_embeddings, position_embeddings, seq_ctx):
        multi = isinstance(hidden_states, list)
        states = hidden_states if multi else [hidden_states]
        futures = future_embeddings if multi else [future_embeddings]
        contexts = seq_ctx if multi else [seq_ctx]
        self.futures.append(futures)
        self.contexts.append(contexts)
        outputs = [self._compute(h, e, c) for h, e, c in zip(states, futures, contexts)]
        result = dict(hidden_states=outputs if multi else outputs[0])
        result.update(
            {
                key: [None] * len(states) if multi else None
                for key in ("router_logits", "router_weights", "router_topk_ids")
            }
        )
        return result


def _context(embeddings, rank):
    start = rank * 4
    ids = torch.arange(20).unsqueeze(0)
    return SequenceContext(
        input_ids=ids[:, start : start + 4],
        inputs_embeds=embeddings,
        raw_input_ids=ids,
        cu_seq_lens_q=BOUNDARIES,
        cu_seq_lens_k=BOUNDARIES,
        max_length_q=7,
        max_length_k=7,
        num_padding=3 if rank == 4 else 0,
        position_ids=ids[:, start : start + 4],
        sequence_parallel_mesh=_Mesh(rank),
        shard_start=start,
        shard_size=4,
    )


@pytest.mark.parametrize("rank", [0, 3])
def test_halo_block_forward_matches_full_gather_under_fake_mesh(rank, monkeypatch):
    monkeypatch.delenv("XTUNER_MTP_HALO", raising=False)
    layer = _CheckpointLayer()
    block = MTPBlock(mtp_config=MTPConfig(num_layers=2, share_weights=True), mtp_layers=[layer])
    full = torch.arange(60, dtype=torch.float64).reshape(1, 20, 3) / 100
    state = torch.full((1, 4, 3), 0.2, dtype=torch.float64)

    def fake_halo_gather(head, gather_dim=1, group=None):
        # Every rank's first-two-token head fragment, concatenated in rank order.
        return torch.cat([full[:, r * 4 : r * 4 + 2] for r in range(5)], dim=1).detach()

    with patch(
        "xtuner.v1.data_proto.sequence_context.all_gather_tensor_autograd",
        return_value=full.detach().clone(),
    ):
        gather_ctx = _context(full[:, rank * 4 : rank * 4 + 4].clone(), rank)
        gather_outputs = block(
            state.clone(),
            embed_tokens_fn=lambda _: pytest.fail("Supplied embeddings must not be re-embedded"),
            position_embeddings=(torch.empty(0), torch.empty(0)),
            seq_ctx=gather_ctx,
        )
    with (
        patch("xtuner.v1.module.mtp.halo.all_gather_tensor_autograd", side_effect=fake_halo_gather),
        patch.dict(os.environ, {"XTUNER_MTP_HALO": "1"}),
    ):
        halo_ctx = _context(full[:, rank * 4 : rank * 4 + 4].clone(), rank)
        halo_outputs = block(
            state.clone(),
            embed_tokens_fn=lambda _: pytest.fail("Supplied embeddings must not be re-embedded"),
            position_embeddings=(torch.empty(0), torch.empty(0)),
            seq_ctx=halo_ctx,
        )
    torch.testing.assert_close(halo_outputs[0]["hidden_states"], gather_outputs[0]["hidden_states"], rtol=0, atol=0)
    # The layer saw four depth calls: depths 1-2 from the full-gather run, then
    # depths 1-2 from the halo run; the per-depth futures must be identical.
    for depth in range(2):
        torch.testing.assert_close(layer.futures[depth][0], layer.futures[depth + 2][0], rtol=0, atol=0)
    assert layer.contexts[0][0] is gather_ctx and layer.contexts[2][0] is halo_ctx


def _distributed_halo_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=120)
    )
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("sp",))
        full = torch.arange(60, dtype=torch.float64).reshape(1, 20, 3) / 100
        ids = torch.arange(20).unsqueeze(0)
        reference = None
        for halo_on in (False, True):
            os.environ["XTUNER_MTP_HALO"] = "1" if halo_on else "0"
            layer = _CheckpointLayer()
            block = MTPBlock(mtp_config=MTPConfig(num_layers=HALO, share_weights=True), mtp_layers=[layer])
            source = full[:, rank * 10 : rank * 10 + 10].clone().requires_grad_()
            state = torch.full_like(source, 0.2, requires_grad=True)
            ctx = SequenceContext(
                input_ids=None,
                cu_seq_lens_q=BOUNDARIES,
                cu_seq_lens_k=BOUNDARIES,
                max_length_q=6,
                max_length_k=6,
                inputs_embeds=source,
                raw_input_ids=ids,
                sequence_parallel_mesh=mesh,
                position_ids=ids[:, rank * 10 : rank * 10 + 10],
                shard_start=rank * 10,
                shard_size=10,
            )
            outputs = block(
                state,
                embed_tokens_fn=lambda _: None,
                position_embeddings=(torch.empty(0), torch.empty(0)),
                seq_ctx=ctx,
            )
            loss = sum(out["hidden_states"].square().mean() for out in outputs)
            loss.backward()
            assert source.grad is not None and source.grad.isfinite().all() and state.grad is not None
            for depth, futures in enumerate(layer.futures, start=1):
                expected = roll_packed_tensor(full, BOUNDARIES, shifts=-depth, dim=1)
                torch.testing.assert_close(futures[0], expected[:, rank * 10 : rank * 10 + 10], rtol=0, atol=0)
            current = (
                layer.weight.grad.clone(),
                state.grad.clone(),
                source.grad.clone(),
                loss.detach().clone(),
                [out["hidden_states"] for out in outputs],
            )
            if reference is None:
                reference = current
            else:
                torch.testing.assert_close(current[3], reference[3], rtol=0, atol=0)
                torch.testing.assert_close(current[0], reference[0], rtol=0, atol=0)
                torch.testing.assert_close(current[1], reference[1], rtol=0, atol=0)
                torch.testing.assert_close(current[2], reference[2], rtol=0, atol=0)
                for out, old_out in zip(current[4], reference[4]):
                    torch.testing.assert_close(out, old_out, rtol=0, atol=0)
    finally:
        os.environ.pop("XTUNER_MTP_HALO", None)
        dist.destroy_process_group()


def test_distributed_halo_matches_full_gather_with_grads(tmp_path):
    mp.spawn(_distributed_halo_worker, args=(str(tmp_path / "mtp-halo-sp"),), nprocs=2, join=True)
