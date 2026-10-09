# Copyright © 2026 Huawei Technologies Co., Ltd.
"""Env-gated levers for the GLM-5.3-Flash KDA sequence-parallel path.

``xtuner.v1.module.attention.kda`` keeps only the gate checks and call wiring;
the lever bodies live here. Each lever is a kill switch: setting its env to
``"0"`` restores the pre-lever behavior in ``kda.py`` unchanged (unset = on),
and every
lever is bit-identical in the values it produces -- they only move *when the
host waits*, *which copy does the layout work*, or *who recomputes a
derivation*.

- ``SP_A2A_DEFER`` (``XTUNER_KDA_SP_A2A_DEFER``, default ``1``):
  ``sp_deferred_projections`` / ``sp_deferred_output`` issue each Ulysses a2a
  as soon as its input is projected and fence it at its consumer
  (``xtuner.v1.ops.comm.ulysses_dispatch``), so the five forward collectives
  share the wire instead of serializing behind per-call fences, and the
  reverse a2a travels while the gate GEMMs run. Same collectives and glue as
  the blocking schedule -- only the wait points move.
- ``CU_HOST_LIST`` (``XTUNER_KDA_CU_HOST_LIST``, default ``1``):
  ``cu_seqlens_arg`` hands the conv/chunk kernels the host list
  ``SequenceContext`` materialized at construction instead of the device
  tensor, removing one blocking D2H sync per kernel call. The kernels call
  ``.tolist()`` on a tensor every invocation; the host list is the same
  values with no sync. Non-NPU backends keep the device tensor.
- ``SP_A2A_DIRECT_CONV`` (``XTUNER_KDA_SP_A2A_DIRECT_CONV``, default ``1``):
  the q/k/v a2a results are consumed with the single-copy seq-major finish
  (``finish_ulysses_dim1_seq_major``) instead of the head-major finish +
  ``transpose(1, 2).contiguous()`` pair -- same values, one permutation copy
  saved forward and one saved backward per conv.
- ``PARAM_CACHE`` (``XTUNER_KDA_PARAM_CACHE``, default ``1``):
  ``cached_gate_param`` skips re-deriving the fp32 form of long-lived
  forget-gate parameters on every call (~2.4k materialize chains per training
  step at 30B/128K).

The ``XTUNER_KDA_SP_A2A_ASYNC`` gate lives in ``xtuner.v1.ops.comm.ulysses_dispatch``
(read per call inside ``all_to_all.ulysses_all_to_all``), not here.
"""

from __future__ import annotations

import os
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.ops.comm.ulysses_dispatch import finish_ulysses_dim1, finish_ulysses_dim1_seq_major, issue_ulysses_dim1
from xtuner.v1.utils.dtensor import materialize_full


if TYPE_CHECKING:
    from xtuner.v1.module.attention.kda import KDAShortConvolution, KimiDeltaAttention


# Deferred-fence SP forward schedule (see ``sp_deferred_projections``).
SP_A2A_DEFER = os.environ.get("XTUNER_KDA_SP_A2A_DEFER", "1") == "1"

# Host-list packed-sequence offsets (see ``cu_seqlens_arg``).
CU_HOST_LIST = os.environ.get("XTUNER_KDA_CU_HOST_LIST", "1") == "1"

# Single-copy seq-major a2a finish for the short convs (see ``_finish_short_conv``).
SP_A2A_DIRECT_CONV = os.environ.get("XTUNER_KDA_SP_A2A_DIRECT_CONV", "1") == "1"

# Param-derived fp32 cache (see ``cached_gate_param``).
PARAM_CACHE = os.environ.get("XTUNER_KDA_PARAM_CACHE", "1") == "1"


def cu_seqlens_arg(
    seq_ctx: SequenceContext, device: torch.device, npu_backend: bool
) -> torch.Tensor | list[int] | None:
    """Pick the packed-sequence offset input for the conv/chunk kernels.

    Returns the ``SequenceContext`` host list when the cu-host-list lever, the
    NPU backend and an NPU device all hold and the list was materialized;
    otherwise the device tensor, exactly as the pre-lever code passed it.

    Args:
        seq_ctx (SequenceContext): The batch's sequence context.
        device (torch.device): The activation device.
        npu_backend (bool): Whether the KDA operator surface is on the fla_npu
            backend (the fla wheel's kernel entries expect the device tensor).

    Returns:
        torch.Tensor | list[int] | None: The value for the ``cu_seqlens``
            argument.
    """
    cu = seq_ctx.cu_seq_lens_q
    if CU_HOST_LIST and npu_backend and device.type == "npu":
        cu_list = seq_ctx.cu_seq_lens_q_list
        if cu_list is not None:
            return cu_list
    return cu


# `_gate_param` re-derives the fp32 form of long-lived parameters on every call (~2.4k
# materialize chains per training step at 30B/128K). Only *alias* derivations are cached:
# when the parameter is already fp32 (the production path -- `fp32_keys_pattern` keeps
# `A_log`/`dt_bias` out of FSDP's mixed-precision cast), `.float()` is the identity and the
# derived tensor shares storage with the live parameter, so it can never go stale and stays
# a re-entrant autograd node across microbatch backwards. The cache entry is still
# revalidated against the live objects (a dead weakref means the id may have been recycled
# -> rebuild) and their local versions. Copy-type derivations (bf16 `.float()`) are
# deliberately never cached -- see `_cached_param_derived`.
#
# The conv weight deliberately stays out of this cache too: it is bf16 and FSDP-sharded, and
# `_version` is blind to FSDP2's unshard path (the optimizer writes the sharded local tensor
# while the next forward refills the reused all-gather buffer without bumping its version),
# so any cached derivation of that buffer could silently go stale. The weight is tiny
# (~48KB) and its rebuild is two view ops (no copy, no compute), so recomputing is free.
_PARAM_DERIVED_CACHE: dict[tuple[int, ...], tuple[tuple[weakref.ref, ...], tuple[int, ...], tuple[Any, ...]]] = {}
_PARAM_DERIVED_CACHE_MAX = 512


def _cached_param_derived(
    params: tuple[torch.Tensor, ...],
    build: Callable[[], tuple[Any, ...]],
) -> tuple[Any, ...]:
    key = tuple(id(p) for p in params)
    entry = _PARAM_DERIVED_CACHE.get(key) if PARAM_CACHE else None
    if entry is not None:
        refs, versions, derived = entry
        if all(ref() is p and materialize_full(p)._version == v for ref, p, v in zip(refs, params, versions)):
            return derived
    derived = build()
    materialized = [materialize_full(p) for p in params]
    # Copy-type derived tensors must not be cached, for two independent reasons:
    # (1) staleness -- under FSDP2 the optimizer writes the sharded local tensor and the
    #     next unshard refills a reused all-gather buffer without bumping its `_version`,
    #     so a cached bf16->fp32 copy would silently freeze the gate values;
    # (2) autograd -- a `.float()` copy of a bf16 leaf is a non-leaf ToCopyBackward node
    #     whose saved tensors are freed by the first backward, so reusing it across
    #     microbatches (mb >= 2) raises "Trying to backward through the graph a second
    #     time".
    # An alias (same dtype, same storage) has neither problem: its values are the live
    # parameter's values, and its grad node (ToTorchTensor for DTensor locals) carries no
    # saved tensors. Anything else rebuilds on every call, matching the pre-cache behavior.
    if (
        not PARAM_CACHE
        or len(derived) != len(materialized)
        or not all(
            isinstance(d, torch.Tensor) and d.dtype == m.dtype and d.data_ptr() == m.data_ptr()
            for d, m in zip(derived, materialized)
        )
    ):
        return derived
    if len(_PARAM_DERIVED_CACHE) >= _PARAM_DERIVED_CACHE_MAX:
        _PARAM_DERIVED_CACHE.clear()
    _PARAM_DERIVED_CACHE[key] = (
        tuple(weakref.ref(p) for p in params),
        tuple(m._version for m in materialized),
        derived,
    )
    return derived


def cached_gate_param(param: torch.Tensor) -> torch.Tensor:
    """``materialize_full(param).float()`` with the param-derived cache applied.

    Only the fp32 case (where ``.float()`` is the identity and the derived tensor aliases the
    live parameter storage) is cached, revalidated against the local tensor's ``_version``
    (``materialize_full(p)._version``), which bumps on the in-place ``to_local`` update the
    optimizer writes. The bf16 case is a real copy and is rebuilt on every call; see
    ``_cached_param_derived``.

    Args:
        param (torch.Tensor): The forget-gate parameter to unshard and pin to fp32.

    Returns:
        torch.Tensor: The fp32 form of the unsharded parameter.
    """
    return _cached_param_derived((param,), lambda: (materialize_full(param).float(),))[0]


def _finish_short_conv(
    conv: KDAShortConvolution,
    issued: tuple[torch.Tensor, Any, torch.Size],
    sp_rank: int,
    sp_size: int,
    cu_seqlens: torch.Tensor | list[int] | None,
) -> torch.Tensor:
    recv, work, out_shape = issued
    if SP_A2A_DIRECT_CONV:
        x = finish_ulysses_dim1_seq_major(recv, work)
    else:
        x = finish_ulysses_dim1(recv, work, out_shape)
        x = x.transpose(1, 2).contiguous()

    weight, bias = conv.materialize_weight_bias()
    weight = weight.chunk(sp_size, dim=0)[sp_rank]
    if bias is not None:
        bias = bias.chunk(sp_size, dim=0)[sp_rank]
    out, _ = conv(x, cu_seqlens=cu_seqlens, weight=weight, bias=bias)
    return out


def sp_deferred_projections(
    attn: KimiDeltaAttention,
    hidden_states: torch.Tensor,
    sp_mesh: Any,
    sp_rank: int,
    sp_size: int,
    cu_seqlens: torch.Tensor | list[int] | None,
    batch_size: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the deferred-fence q/k/v/g/beta schedule of ``forward_for_sp``.

    Issue every a2a as soon as its input is projected so the collectives fly
    under the projections, the convs and each other; fence each at its
    consumer (``_finish_short_conv`` / the glue below). Returns the same five
    tensors the blocking schedule produces, bit-identical.

    Args:
        attn (KimiDeltaAttention): The KDA module whose projections feed the a2as.
        hidden_states (torch.Tensor): The ``[1, S/sp, hidden]`` shard input.
        sp_mesh (DeviceMesh): The sequence-parallel mesh.
        sp_rank (int): This rank's position in the SP group.
        sp_size (int): The SP group size.
        cu_seqlens (torch.Tensor | list[int] | None): Packed-sequence offsets.
        batch_size (int): The (packed, always 1) batch size.
        seq_len (int): The local shard sequence length.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            ``(q, k, v, g_raw, beta)`` exactly as the blocking schedule yields.
    """
    group = sp_mesh.get_group()
    projection_size = attn.num_heads * attn.head_dim
    q_local = attn.q_proj(hidden_states).transpose(1, 2)
    dq = issue_ulysses_dim1(q_local, group)
    k_local = attn.k_proj(hidden_states).transpose(1, 2)
    dk = issue_ulysses_dim1(k_local, group)
    v_local = attn.v_proj(hidden_states).transpose(1, 2)
    dv = issue_ulysses_dim1(v_local, group)
    g_raw = attn.f_b_proj(attn.f_a_proj(hidden_states))
    g_send = g_raw.view(batch_size, seq_len, projection_size).transpose(1, 2)
    dg = issue_ulysses_dim1(g_send, group)
    beta_send = attn.b_proj(hidden_states).float().transpose(1, 2)
    dbeta = issue_ulysses_dim1(beta_send, group)

    q = _finish_short_conv(attn.q_conv1d, dq, sp_rank, sp_size, cu_seqlens)
    k = _finish_short_conv(attn.k_conv1d, dk, sp_rank, sp_size, cu_seqlens)
    v = _finish_short_conv(attn.v_conv1d, dv, sp_rank, sp_size, cu_seqlens)
    g_raw = finish_ulysses_dim1(*dg)
    g_raw = g_raw.transpose(1, 2).view(batch_size, seq_len * sp_size, attn.num_heads // sp_size, attn.head_dim)
    beta = finish_ulysses_dim1(*dbeta).transpose(1, 2).sigmoid()  # (B, L, H/sp)
    return q, k, v, g_raw, beta


def sp_deferred_output(
    attn: KimiDeltaAttention,
    o: torch.Tensor,
    hidden_states: torch.Tensor,
    sp_mesh: Any,
    batch_size: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the deferred-fence output a2a of ``forward_for_sp``.

    Issue the reverse a2a before the gate projections so the wire carries the
    core output while the gate GEMMs run; fence at the o_norm consumption
    point. Returns the same ``(o, gate_out)`` the blocking schedule produces,
    bit-identical.

    Args:
        attn (KimiDeltaAttention): The KDA module producing the gate.
        o (torch.Tensor): The ``[B, L, H/sp, D]`` KDA core output.
        hidden_states (torch.Tensor): The shard input feeding ``_gate_output``.
        sp_mesh (DeviceMesh): The sequence-parallel mesh.
        batch_size (int): The (packed, always 1) batch size.
        seq_len (int): The local shard sequence length.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(o, gate_out)`` -- the fenced
            ``[B, L/sp, H, D]`` movement output and the gate projection.
    """
    group = sp_mesh.get_group()
    do = issue_ulysses_dim1(o, group)
    gate_out = attn._gate_output(hidden_states).view(batch_size, seq_len, attn.num_heads, attn.head_dim)
    o = finish_ulysses_dim1(*do)
    return o, gate_out
