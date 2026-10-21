"""Neighbor-halo future embeddings for the MTP local path.

``SequenceContext.raw_inputs_embeds`` all-gathers every SP rank's embedding
shard into the full packed sequence — ``pack_len x hidden`` fp32 per rank per
micro-batch — although ``local_packed_future_embeddings`` only ever reads the
local shard plus ``depth`` tokens past its end. The halo path materializes
exactly that neighborhood instead: the local shard followed by a
``halo_size``-token fragment gathered from the next SP rank through the same
autograd-aware functional collective, so values and gradients stay bit-identical
to the full-sequence gather while the buffer shrinks from the whole pack to
``shard_size + halo_size`` tokens.

Enabled by the ``XTUNER_MTP_HALO`` environment variable; with the gate off the
MTP block keeps the original full-sequence gather and nothing here runs.
"""

import os

import torch
from torch.distributed._functional_collectives import all_gather_tensor_autograd

from xtuner.v1.data_proto import SequenceContext


HALO_ENV_VAR = "XTUNER_MTP_HALO"


def mtp_halo_enabled() -> bool:
    """Whether the halo future-embedding path is enabled.

    Returns:
        bool: True when the ``XTUNER_MTP_HALO`` environment variable is ``1``.
    """
    return os.environ.get(HALO_ENV_VAR, "0") == "1"


class HaloFutureEmbeds:
    """Local shard plus next-rank halo replacement for the full-sequence embedding gather.

    Args:
        buffer (torch.Tensor): ``[batch, shard_size + halo_size, hidden]`` embeddings: the
            rank's own shard followed by ``halo_size`` tokens read from the next SP rank
            (wrap-around garbage on the last rank, which the pack-end mask never surfaces).
        shard_start (int): Global position of the local shard's first token inside the
            padded packed sequence.
        shard_size (int): Number of tokens the local shard holds.
    """

    __slots__ = ("buffer", "shard_start", "shard_size")

    buffer: torch.Tensor
    shard_start: int
    shard_size: int

    def __init__(self, *, buffer: torch.Tensor, shard_start: int, shard_size: int) -> None:
        self.buffer = buffer
        self.shard_start = shard_start
        self.shard_size = shard_size


def build_halo_future_embeds(seq_ctx: SequenceContext, *, halo_size: int) -> HaloFutureEmbeds:
    """Build the local shard plus next-rank halo buffer for one micro-batch.

    Under SP this issues one autograd-aware all-gather of every rank's first
    ``halo_size`` embedding tokens and keeps only the next rank's fragment, so
    gradients crossing the shard boundary route through the same reduce-scatter
    backward as the full-sequence gather. Without SP (or on a single-rank mesh)
    no communication happens and the zero-filled halo tail is never read: every
    source past the last pack end is masked to zero.

    Args:
        seq_ctx (SequenceContext): Micro-batch context whose ``inputs_embeds`` is this
            rank's embedding shard.
        halo_size (int): Number of future tokens the MTP block can reach past the shard
            end; equals the MTP depth count.

    Returns:
        HaloFutureEmbeds: The halo buffer with its global shard offset.
    """
    inputs_embeds = seq_ctx.inputs_embeds
    assert inputs_embeds is not None
    shard_size = inputs_embeds.shape[1]
    mesh = seq_ctx.sequence_parallel_mesh
    if mesh is None or mesh.size() == 1:
        tail = inputs_embeds.new_zeros(inputs_embeds.shape[0], halo_size, inputs_embeds.shape[2])
        return HaloFutureEmbeds(buffer=torch.cat([inputs_embeds, tail], dim=1), shard_start=0, shard_size=shard_size)
    if shard_size < halo_size:
        raise ValueError(
            f"Halo path needs shard_size ({shard_size}) >= halo_size ({halo_size}); "
            "keep the full-sequence gather for this shard layout"
        )
    head = inputs_embeds[:, :halo_size].contiguous()
    gathered = all_gather_tensor_autograd(head, gather_dim=1, group=mesh)
    next_rank = (mesh.get_local_rank() + 1) % mesh.size()
    halo_fragment = gathered[:, next_rank * halo_size : (next_rank + 1) * halo_size]
    return HaloFutureEmbeds(
        buffer=torch.cat([inputs_embeds, halo_fragment], dim=1),
        shard_start=seq_ctx._shard_start,
        shard_size=shard_size,
    )


def halo_packed_future_embeddings(
    halo: HaloFutureEmbeds,
    cu_seq_lens: torch.IntTensor,
    *,
    depth: int,
) -> torch.Tensor:
    """Materialize one MTP depth's future embeddings from the halo buffer.

    Halo twin of ``local_packed_future_embeddings``: for global query position
    ``i``, read ``i + depth`` only within the same packed sequence, otherwise
    fill zero. Sources inside the shard read the local part of the buffer,
    sources past its end read the halo fragment — the same values the
    full-sequence gather holds at those global positions.

    Args:
        halo (HaloFutureEmbeds): Buffer built by :func:`build_halo_future_embeds`.
        cu_seq_lens (torch.IntTensor): Cumulative sequence lengths of the packed batch,
            shape ``[num_sequences + 1]``.
        depth (int): Future distance; positive and at most the halo size.

    Returns:
        torch.Tensor: ``[batch, shard_size, hidden]`` owned future embeddings for this depth.
    """
    if depth < 1:
        raise ValueError("MTP future depth must be positive")
    if depth > halo.buffer.shape[1] - halo.shard_size:
        raise ValueError("MTP future depth exceeds the halo fragment")
    if halo.shard_size == 0:
        return halo.buffer[:, :0].clone()
    device = halo.buffer.device
    positions = torch.arange(halo.shard_start, halo.shard_start + halo.shard_size, device=device)
    # right=True skips empty packs and assigns a boundary position to its next pack.
    ends = cu_seq_lens[1:].to(device=device)
    pack_indices = torch.searchsorted(ends, positions, right=True)
    pack_ends = ends.index_select(0, pack_indices)
    source_positions = positions + depth
    valid = source_positions < pack_ends
    # Sources past the shard end land in the halo fragment, so the local index
    # stays in bounds up to depth == halo_size without the full-sequence clamp.
    future = halo.buffer.index_select(1, source_positions - halo.shard_start)
    future.masked_fill_(~valid[None, :, None], 0)
    return future
