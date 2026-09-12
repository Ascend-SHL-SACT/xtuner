# Copyright (c) OpenMMLab. All rights reserved.
"""CoC-style collect-compute interleave for sequence-parallel all-gathers.

Gated by ``XTUNER_DSA_KV_GATHER_COC`` (default off). When on,
:func:`coc_all_gather` issues the sequence-parallel all-gather of the DSA-MLA
``key_states`` on a dedicated communication stream backed by a dedicated
process group (a fresh group over the same ranks as the SP mesh) and returns an
``AsyncCollectiveTensor``; the consumer-side wait is deferred to
:func:`coc_wait`, which the caller places immediately before the first op that
reads the gathered tensor. Compute enqueued between the launch and the wait
(the DSA indexer chain) then overlaps the collective — MindSpeed CoC's
"issue-collective then compute until the wait" pattern, applied at whole-tensor
granularity because the consumers (``npu_lightning_indexer`` /
``npu_sparse_mla``) are fused kernels that need the full gathered tensor, so
per-chunk consumption is not available.

The dedicated process group is required for correctness: HCCL matches
collectives per communicator by issue order, so the comm stream must not share
a group with the same-group collectives it overlaps (here the indexer's own
k all-gather on the SP mesh group). The per-launch ``wait_stream`` plus the
consumer-side wait chain every group op into one global order, so two ops of
the CoC group are never in flight unordered. The autograd backward
(reduce-scatter) of ``all_gather_tensor_autograd`` is stream-ordered after the
forward collective through the same consumer-side wait, since the backward is
only enqueued after the recomputed/consumed forward has waited the collective.
"""

from __future__ import annotations

import os
from typing import cast

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import (
    AsyncCollectiveTensor,
    all_gather_tensor_autograd,
)
from torch.distributed.device_mesh import DeviceMesh

from xtuner.v1.utils import get_device


DEVICE = get_device()


def dsa_kv_gather_coc_enabled() -> bool:
    """Return whether the CoC-style SP gather overlap is enabled.

    Returns:
        bool: True when ``XTUNER_DSA_KV_GATHER_COC=1``. Default off, which keeps
        the original same-stream gather path byte-identical.
    """
    return os.environ.get("XTUNER_DSA_KV_GATHER_COC", "0") == "1"


_COC_RESOURCES: dict[tuple[int, ...], tuple[torch.cuda.Stream, dist.ProcessGroup]] = {}


def _coc_resources(group: dist.ProcessGroup) -> tuple[torch.cuda.Stream, dist.ProcessGroup]:
    ranks = tuple(sorted(dist.get_process_group_ranks(group)))
    resources = _COC_RESOURCES.get(ranks)
    if resources is None:
        coc_group = dist.new_group(ranks=list(ranks))
        # Ordering invariant: all_gather_tensor_autograd concatenates shards in
        # the CoC group's group-rank order while the reference path gathers in
        # SP-group order, so the two orderings must agree. They do for every
        # mesh this repo builds (both rank lists are monotonic in the mesh
        # index); check it once at creation so a future non-monotonic mesh
        # fails loudly instead of silently permuting the gathered KV.
        for group_rank in range(len(ranks)):
            if dist.get_global_rank(coc_group, group_rank) != dist.get_global_rank(group, group_rank):
                raise RuntimeError(
                    "CoC group rank order differs from the SP group order; the gathered tensor would be permuted."
                )
        coc_stream = cast(torch.cuda.Stream, torch.cuda.Stream(device=DEVICE))
        resources = (coc_stream, coc_group)
        _COC_RESOURCES[ranks] = resources
    return resources


def warmup_coc_group(mesh: DeviceMesh | None) -> None:
    """Eagerly create the CoC group and stream outside any forward pass.

    Called once from the trainer's mesh-warmup phase: HCCL group creation is a
    collective over all ranks, and building it lazily at the first DSA layer
    can race the other lazily-created communicators at scale (createLink
    deadlock precedent). A no-op when the CoC gather cannot run.

    Args:
        mesh (DeviceMesh | None): The sequence-parallel device mesh; ``None``
            leaves nothing to warm up.
    """
    if mesh is None or mesh.size() == 1:
        return
    sp_group = mesh.get_group()
    assert sp_group is not None, "Sequence-parallel mesh must carry a process group for the CoC gather."
    _coc_resources(sp_group)


def coc_all_gather(input_tensor: torch.Tensor, *, dim: int, mesh: DeviceMesh | None) -> torch.Tensor:
    """Issue the SP all-gather on the CoC comm stream without waiting.

    The collective is enqueued on a dedicated stream/group after a
    ``wait_stream`` on the current (compute) stream, so the caller may run
    independent compute before materializing the result with :func:`coc_wait`.

    Args:
        input_tensor (torch.Tensor): The local tensor shard to gather.
        dim (int): The dimension along which to concatenate the gathered shards.
            The deferred-wait contract requires ``dim=0``: for any other dim
            ``all_gather_tensor_autograd`` materializes the result immediately,
            silently dropping the overlap.
        mesh (DeviceMesh | None): The sequence-parallel device mesh; ``None``
            returns the input unchanged.

    Returns:
        torch.Tensor: An ``AsyncCollectiveTensor`` holding the gathered tensor;
        call :func:`coc_wait` before the first consuming op on the compute
        stream.
    """
    if mesh is None or mesh.size() == 1:
        return input_tensor
    assert dim == 0, "coc_all_gather only supports the dim=0 deferred-wait contract."
    sp_group = mesh.get_group()
    assert sp_group is not None, "Sequence-parallel mesh must carry a process group for the CoC gather."
    coc_stream, coc_group = _coc_resources(sp_group)
    compute_stream = torch.cuda.current_stream()
    with torch.cuda.stream(coc_stream):
        coc_stream.wait_stream(compute_stream)
        gathered = all_gather_tensor_autograd(input_tensor, gather_dim=dim, group=coc_group)
    input_tensor.record_stream(coc_stream)
    return cast(torch.Tensor, gathered)


def coc_wait(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize a :func:`coc_all_gather` result on the current stream.

    A no-op for plain tensors, so callers may invoke it unconditionally on the
    gather output regardless of whether the feature is enabled.

    Args:
        tensor (torch.Tensor): The gather output (plain or async).

    Returns:
        torch.Tensor: The underlying tensor, ordered after the collective on
        the current stream.
    """
    if isinstance(tensor, AsyncCollectiveTensor):
        tensor = tensor.wait()
        tensor.record_stream(torch.cuda.current_stream())
    return tensor
