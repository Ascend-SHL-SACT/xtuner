# Copyright (c) OpenMMLab. All rights reserved.
"""Env-gated dead-chain stripping helpers (``XTUNER_DEAD_CHAIN_STRIP``).

GLM-5.2 runs with ``balancing_loss_cfg=None`` and ``z_loss_cfg=None``, which
makes several per-layer routing chains provably dead:

- ``selected_router_weights`` / ``selected_router_logits`` arguments of
  ``AuxLossContext.accumulate`` are never read when both the balancing and the
  z-loss contexts are ``None`` (both fan-out loops iterate empty lists), yet
  the callers materialize them per layer with cat + index_select + contiguous
  + float over the full token table;
- the ``router_weights`` (renormalized ``scores_for_choice``) returned by the
  no-aux routers only feed those dead arguments;
- ``topkens_per_expert`` (the router-side ``histc``) has zero consumers;
- ``* hidden_factor`` with ``hidden_factor == 1.0`` is a bit-exact identity.

The SP-mode/KV-slice per-context cache used by the sparse-MLA and indexer TND
paths lives in :mod:`xtuner.v1.ops.sparse_mla.npu_indexer`
(``get_sp_mode_and_slice``).

The live ``selected_experts`` histogram chain (feeding
``tokens_per_expert_global`` logging and the router bias update) is
deliberately kept untouched in all modes.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from xtuner.v1.loss.aux_loss import AuxLossContext

_EMPTY_TENSOR_CACHE: dict[torch.device, torch.Tensor] = {}


@lru_cache(maxsize=1)
def dead_chain_strip_enabled() -> bool:
    """Whether ``XTUNER_DEAD_CHAIN_STRIP=1`` enables the dead-chain fast paths.

    Returns:
        bool: True when the env gate is on.
    """
    return os.environ.get("XTUNER_DEAD_CHAIN_STRIP", "0") == "1"


def empty_router_tensor(device: torch.device | str) -> torch.Tensor:
    """Return a cached 0-element float32 tensor used to fill dead arguments.

    Args:
        device (torch.device | str): Device the placeholder must live on.

    Returns:
        torch.Tensor: Shared empty ``[0]`` float32 tensor for that device.
    """
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    cached = _EMPTY_TENSOR_CACHE.get(dev)
    if cached is None:
        cached = torch.empty(0, dtype=torch.float32, device=dev)
        _EMPTY_TENSOR_CACHE[dev] = cached
    return cached


def accumulate_experts_only(
    aux_loss: AuxLossContext,
    router_topk_ids: Sequence[torch.Tensor],
    nonpad_indices: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """``AuxLossContext.accumulate`` for the both-contexts-None configuration.

    Runs only the live expert-histogram chain. The dead ``router_weights`` /
    ``router_logits`` arguments are replaced by a cached empty tensor, which
    ``accumulate`` never reads when ``balancing_ctx`` and ``z_ctx`` are both
    ``None``, so the per-layer cat + index_select + contiguous + float over
    the full token tables is skipped.

    Args:
        aux_loss (AuxLossContext): The model-level aux loss accumulator.
        router_topk_ids (Sequence[torch.Tensor]): Per-micro-batch selected
            expert IDs, each ``(tokens_mb, num_experts_per_tok)``.
        nonpad_indices (torch.Tensor): Flat indices of non-padding tokens in
            the micro-batch-concatenated token table.
        hidden_states (torch.Tensor): Main-path carrier tensor (returned
            unchanged because no z-loss context is attached).

    Returns:
        torch.Tensor: The same ``hidden_states`` handle, matching the
        ``accumulate`` return contract.
    """
    cat_router_topk_ids = torch.cat(list(router_topk_ids), dim=0)
    return aux_loss.accumulate(
        selected_router_weights=empty_router_tensor(hidden_states.device),
        selected_router_logits=empty_router_tensor(hidden_states.device),
        selected_experts=cat_router_topk_ids.index_select(0, nonpad_indices).contiguous(),
        hidden_states=hidden_states,
        balancing_ctx=None,
        z_ctx=None,
    )
