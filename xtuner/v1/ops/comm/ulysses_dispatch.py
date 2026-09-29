# Copyright © 2026 Huawei Technologies Co., Ltd.
"""Gated async dispatch for the Ulysses all-to-all used by the KDA sequence-parallel path.

``xtuner.v1.module.attention.kda`` imports ``ulysses_all_to_all`` and calls it six times per
layer under sequence parallelism (q/k/v pre-conv, g, beta forward, and the core output back),
always with ``scatter_dim=1`` (the full dimension) and ``gather_dim=2`` (the sharded one) on
batch-1 tensors. Setting ``XTUNER_KDA_SP_A2A_ASYNC=1`` routes those calls through
``ulysses_scatter_heads_blocking`` (``xtuner.v1.ops.comm.ulysses_async``), which issues the
collective on a dedicated comm stream: value- and layout-identical output (verified in
``tests/ops/test_ulysses_async.py``), but wire time and rank-arrival skew stop blocking the
compute stream at the issue point, so the next micro-batch's projections overlap the current
one's KDA scan. The default is off: the synchronous functional-collectives path stays the
validated baseline, and any non-matching call shape (batch > 1, other dims) always takes it.
"""

from __future__ import annotations

import os

import torch
from torch.distributed.device_mesh import DeviceMesh

from xtuner.v1.ops.comm.all_to_all import ulysses_all_to_all as _ulysses_all_to_all_sync


_A2A_ASYNC_ENV = "XTUNER_KDA_SP_A2A_ASYNC"


def _a2a_async_enabled() -> bool:
    return os.getenv(_A2A_ASYNC_ENV, "0") == "1"


def ulysses_all_to_all(
    input: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    mesh: DeviceMesh,
) -> torch.Tensor:
    """Ulysses all-to-all, optionally issued on the async comm stream.

    Drop-in replacement for ``xtuner.v1.ops.comm.all_to_all.ulysses_all_to_all``: identical
    signature and output. When ``XTUNER_KDA_SP_A2A_ASYNC=1`` and the call matches the KDA SP
    movement (batch-1 tensor, ``scatter_dim=1`` / ``gather_dim=2``), the collective is issued
    through the comm-stream path and fenced before return; every other call is forwarded to the
    synchronous implementation unchanged.

    Args:
        input (torch.Tensor): The tensor to redistribute across ``mesh``.
        scatter_dim (int): The dimension along which the input is scattered.
        gather_dim (int): The dimension along which the output is gathered.
        mesh (DeviceMesh): The device mesh defining the process group.

    Returns:
        torch.Tensor: The redistributed tensor, identical to the synchronous helper.
    """
    if _a2a_async_enabled() and input.shape[0] == 1 and scatter_dim == 1 and gather_dim == 2:
        from xtuner.v1.ops.comm.ulysses_async import ulysses_scatter_heads_blocking

        return ulysses_scatter_heads_blocking(input, mesh.get_group())
    return _ulysses_all_to_all_sync(input, scatter_dim=scatter_dim, gather_dim=gather_dim, mesh=mesh)


__all__ = ["ulysses_all_to_all"]
