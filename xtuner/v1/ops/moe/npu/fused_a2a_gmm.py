"""Fused MoE dispatch-alltoallv + grouped-matmul (``npu_alltoallv_gmm``) and its combine sibling
(``npu_gmm_alltoallv``), mirroring MindSpeed ``mc2_fuse_a2a.py``.

Env-gated by ``XTUNER_MOE_FUSED_A2A_GMM`` (default OFF → byte-identical to the separated
``TorchAll2AllDispatcher`` + ``GroupedLinear`` path). The fused CANN kernel pipelines the
dispatch/combine all-to-all behind the FC1/FC2 grouped-matmul compute. Both sibling ops own the
a2a+permute+gmm (dispatch) and gmm+a2a (combine), so the fused path bypasses
``dispatch_postprocess``/``combine_preprocess``/``combine_postprocess`` and is self-contained —
it reuses only ``dispatch_preprocess``'s sort-permute and a final ``unpermute``.

The torch_npu ops are forward-only (no autograd registration), so the two ``torch.autograd.Function``
classes hand-write the backward mirroring MindSpeed: dX via the sibling op with counts swapped and
the weight's in/out dims transposed; dW via a per-expert matmul loop (writes directly to the native
``[E, out, in]`` layout and tolerates empty expert groups).

Under activation-checkpoint recompute (``RECOMPUTE_RATIO > 0``) the whole-MoE-layer reentrant
checkpoint re-issues both fused a2a ops in backward. The re-issued grouped-matmul stage nans
unless the forward-only weight clones are detached (``no_grad`` in
``fused_dispatch_mlp_combine``) — see that function for the root cause and the run52/run67 A/B.

Op contract (probed on Ascend910, CANN9.2/torch_npu2.9.1 — see memory
``npu-alltoallv-gmm-probe-verdict``): ``hcom=`` (not ``group=``); ``gmm_weight`` 3D
``[E_local, in, out]`` with ``trans_gmm_weight=False``; ``send_counts``/``recv_counts`` flat
``List[int]`` of length ``E_global = ep * e_local`` (per-global-expert / source-rank-major-then-local
respectively); ``permute_out_flag=True`` returns the 3rd tensor = a2a output already permuted to
local-expert order, reused for dW.

This module also hosts ``fused_forward``, the ``MoEDecoderLayer`` orchestrator (the env-gated entry
point in ``MoEDecoderLayer._pre_moe_forward``) that wires the FSDP2-local expert weights + the MoE
activation + the shared-experts/post-moe merge around ``fused_dispatch_mlp_combine``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  -- NPU-only module; registers npu_alltoallv_gmm / npu_gmm_alltoallv
from torch.distributed.tensor import DTensor

from xtuner.v1.module.dispatcher.torch_all2all import (
    TorchAll2AllDispatcher,
    TorchAll2AllPreDispatchResult,
)
from xtuner.v1.module.grouped_linear.moe_group_linear import GroupedLinear
from xtuner.v1.ops import unpermute


if TYPE_CHECKING:
    from xtuner.v1.module import RouterResults
    from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEDecoderLayer
    from xtuner.v1.module.dispatcher import PreDispatchResult


def is_enabled() -> bool:
    """Return whether fused MoE dispatch-alltoallv+grouped-matmul is on.

    Returns:
        bool: True when ``XTUNER_MOE_FUSED_A2A_GMM=1``. Default OFF (byte-identical to HEAD).
    """
    return os.environ.get("XTUNER_MOE_FUSED_A2A_GMM", "0") == "1"


def _resolve_hcom(ep_group: dist.ProcessGroup) -> tuple[str, int]:
    """Resolve the HCCL communicator name and EP world size for the default NPU backend.

    ``get_hccl_comm_name`` is indexed by the GLOBAL rank of the process, not the
    group-local rank — mirroring ``fused_comm._hcomm_info`` (verified on 13B/16-NPU with
    two DP replicas). Passing the group-local rank silently aliases replica 0's
    communicator on every replica, so the fused op's internal a2a runs on the wrong HCCL
    comm and reads garbage (``grad_norm: nan`` on all non-first replicas only).

    Args:
        ep_group (dist.ProcessGroup): the expert-parallel process group.

    Returns:
        tuple[str, int]: ``(hcom_name, ep_world_size)``.
    """
    rank = dist.get_rank(ep_group)
    global_rank = dist.get_global_rank(ep_group, rank)
    ep_world_size = dist.get_world_size(ep_group)
    hcom = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(global_rank)  # type: ignore[attr-defined]
    return hcom, ep_world_size


def _compute_counts(
    topk_ids: torch.Tensor,
    n_routed_experts: int,
    ep_group: dist.ProcessGroup,
) -> tuple[list[int], list[int], list[int]]:
    """Compute the fused op's send/recv counts and per-local-expert group_list.

    Mirrors ``torch_all2all._dispatch``'s metadata exchange (the fused op replaces only the payload
    a2a + gmm, not this cheap count exchange). No new all-gather: the ``[ep, e_local]`` recv grid is
    the all-to-all of this rank's per-global-expert counts.

    Args:
        topk_ids (torch.Tensor): per-token routed expert ids, ``[n_local, topk]``.
        n_routed_experts (int): global expert count (``= ep * e_local``).
        ep_group (dist.ProcessGroup): the expert-parallel process group.

    Returns:
        tuple[list[int], list[int], list[int]]: ``(send_counts, recv_counts, group_list)`` where
        ``send_counts`` is per-global-expert (length ``ep * e_local``, expert-ordered),
        ``recv_counts`` is the flattened ``[ep, e_local]`` grid (source-rank-major then local-expert),
        and ``group_list`` is the per-local-expert received count (length ``e_local``) for the dW
        matmul loop.
    """
    ep_size = ep_group.size()
    e_local = n_routed_experts // ep_size
    # send_counts: this rank's per-global-expert counts (length n_routed_experts = ep * e_local).
    tokens_per_expert = torch.histc(topk_ids, bins=n_routed_experts, min=0, max=n_routed_experts)
    # recv grid: all-to-all of per-global-expert counts -> [ep, e_local] (source-rank-major).
    recv_grid = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
    dist.all_to_all_single(recv_grid, tokens_per_expert, group=ep_group)
    recv_grid = recv_grid.view(ep_size, e_local)
    send_counts = tokens_per_expert.to("cpu").tolist()
    recv_counts = recv_grid.ravel().to("cpu").tolist()
    # group_list: per-local-expert received count = sum over source ranks (for the dW matmul loop).
    group_list = recv_grid.sum(dim=0).to("cpu").tolist()
    return send_counts, recv_counts, group_list


def _dw_matmul_loop(
    weight: torch.Tensor,
    gmm_input: torch.Tensor,
    grad_out: torch.Tensor,
    group_list: list[int],
) -> torch.Tensor:
    """Compute the expert weight grad as a per-expert matmul loop.

    For each local expert ``e``, ``grad_weight[e] = grad_out_e.T @ gmm_input_e`` which writes directly to the native
    ``[E, out, in]`` layout (no transpose). Tolerates empty expert groups (``tokens == 0``) and
    non-contiguous operands, unlike ``npu_gmm_backward``.

    Args:
        weight (torch.Tensor): native weight ``[E_local, out, in]`` (only its shape/dtype/device
            are used to allocate the result).
        gmm_input (torch.Tensor): the saved forward gmm input ``[n_recv, in]``, expert-grouped.
        grad_out (torch.Tensor): the gmm output grad ``[n_recv, out]``, expert-grouped.
        group_list (list[int]): per-local-expert received token counts (length ``e_local``).

    Returns:
        torch.Tensor: the weight grad, native ``[E_local, out, in]``.
    """
    grad_weight = torch.zeros_like(weight)
    offset = 0
    for expert in range(weight.shape[0]):
        size = group_list[expert]
        if size > 0:
            grad_weight[expert] = grad_out[offset : offset + size].t() @ gmm_input[offset : offset + size]
        offset += size
    return grad_weight


class AlltoallvPermuteGmm(torch.autograd.Function):
    """Fused dispatch alltoallv + src-rank→local-expert permute + FC1 grouped-matmul.

    Forward calls ``npu_alltoallv_gmm`` (``permute_out_flag=True``) and saves the permuted a2a output
    (``permute_out``, reused for dW) and the gmm input; the native weight is read LIVE in backward
    (see ``ctx.weight_ref``) so it does not pin the FSDP full buffer. Backward computes dX via the
    sibling ``npu_gmm_alltoallv`` (counts swapped, native ``[E, out, in]`` weight with
    ``trans_gmm_weight=False`` — the MindSpeed Stack-A convention) and dW via the matmul loop.
    """

    @staticmethod
    def forward(
        ctx: Any,
        gmm_x: torch.Tensor,
        weight: torch.Tensor,
        w_fwd: torch.Tensor,
        ep_group: dist.ProcessGroup,
        send_counts: list[int],
        recv_counts: list[int],
        group_list: list[int],
    ) -> torch.Tensor:
        hcom, ep_world_size = _resolve_hcom(ep_group)
        # ``w_fwd`` is the caller-pre-transposed contiguous weight ``[E, in, out]`` (the op needs
        # this layout with ``trans_gmm_weight=False``), materialized detached (``no_grad``) in
        # ``fused_dispatch_mlp_combine`` before the metadata ``all_to_all_single`` in
        # ``_compute_counts``. The detach is the recompute-nan fix -- see
        # ``fused_dispatch_mlp_combine`` for the root cause and the run52/run67 A/B.
        # The native ``[E, out, in]`` ``weight`` is the FSDP-managed parameter; the backward reads
        # it LIVE (``ctx.weight_ref``) instead of saving it, so the FSDP all-gathered full buffer is
        # not pinned across layers. The MoEDecoderLayer ``pre_backward`` hook re-unshards the
        # parameter before this backward node runs, so reading it live yields the full weight.
        mm1_out, _, permute_out = torch_npu.npu_alltoallv_gmm(
            gmm_x=gmm_x,
            gmm_weight=w_fwd,
            hcom=hcom,
            ep_world_size=ep_world_size,
            send_counts=send_counts,
            recv_counts=recv_counts,
            send_counts_tensor=None,
            recv_counts_tensor=None,
            mm_x=None,
            mm_weight=None,
            trans_gmm_weight=False,
            trans_mm_weight=False,
            permute_out_flag=True,
        )
        ctx.weight_ref = weight
        ctx.save_for_backward(permute_out, gmm_x)
        ctx.ep_group = ep_group
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        ctx.group_list = group_list
        return mm1_out

    @staticmethod
    def backward(ctx: Any, mm1_out_grad: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        permute_out, _gmm_x = ctx.saved_tensors
        weight = ctx.weight_ref
        ep_group = ctx.ep_group
        send_counts = ctx.send_counts
        recv_counts = ctx.recv_counts
        group_list = ctx.group_list
        hcom, ep_world_size = _resolve_hcom(ep_group)
        # dX: reverse a2a + transposed-weight gmm via the sibling op (counts SWAPPED). Native
        # weight [E, out, in] with trans_gmm_weight=False == MindSpeed's rearrange('nhf->nfh').
        gmm_x_grad, _ = torch_npu.npu_gmm_alltoallv(
            gmm_x=mm1_out_grad,
            gmm_weight=weight,
            hcom=hcom,
            ep_world_size=ep_world_size,
            send_counts=recv_counts,
            recv_counts=send_counts,
            send_counts_tensor=None,
            recv_counts_tensor=None,
            mm_x=None,
            mm_weight=None,
            trans_gmm_weight=False,
            trans_mm_weight=False,
        )
        # dW: grad_out=mm1_out_grad, gmm_input=permute_out (the saved a2a output, expert-grouped).
        weight_grad = _dw_matmul_loop(weight, permute_out, mm1_out_grad, group_list)
        # Inputs: (gmm_x, weight, w_fwd, ep_group, send, recv, group_list) -> 7 grad slots.
        # w_fwd is a detached forward-only clone; its grad is None (grad flows to the native param
        # via weight_grad, in the native [E, out, in] layout the optimizer steps).
        return gmm_x_grad, weight_grad, None, None, None, None, None


class GmmUnpermuteAlltoallv(torch.autograd.Function):
    """Fused FC2 grouped-matmul + combine alltoallv (gmm-then-a2a).

    Forward calls ``npu_gmm_alltoallv`` (counts pre-swapped: combine reverses dispatch) and saves the
    gmm input; the native weight is read LIVE in backward (``ctx.weight_ref``) so it does not pin the
    FSDP full buffer. ``npu_gmm_alltoallv`` returns only 2 values, so the backward RE-DERIVES the
    permuted a2a output via a second ``npu_alltoallv_gmm`` call (counts un-swapped,
    ``permute_out_flag=True``) that yields BOTH dX and the ``permute_grad`` reused for dW — the
    verbatim MindSpeed asymmetry (``mc2_fuse_a2a.py:153-185``).
    """

    @staticmethod
    def forward(
        ctx: Any,
        gmm_x: torch.Tensor,
        weight: torch.Tensor,
        w_fwd: torch.Tensor,
        ep_group: dist.ProcessGroup,
        send_counts: list[int],
        recv_counts: list[int],
        group_list: list[int],
    ) -> torch.Tensor:
        hcom, ep_world_size = _resolve_hcom(ep_group)
        # ``w_fwd`` is the caller-pre-transposed contiguous weight ``[E, in, out]``; see
        # ``AlltoallvPermuteGmm.forward`` for the detach + placement rationale.
        alltoall_out, _ = torch_npu.npu_gmm_alltoallv(
            gmm_x=gmm_x,
            gmm_weight=w_fwd,
            hcom=hcom,
            ep_world_size=ep_world_size,
            send_counts=recv_counts,
            recv_counts=send_counts,
            send_counts_tensor=None,
            recv_counts_tensor=None,
            mm_x=None,
            mm_weight=None,
            trans_gmm_weight=False,
            trans_mm_weight=False,
        )
        ctx.weight_ref = weight
        ctx.save_for_backward(gmm_x)
        ctx.ep_group = ep_group
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        ctx.group_list = group_list
        return alltoall_out

    @staticmethod
    def backward(ctx: Any, alltoall_out_grad: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        (gmm_x,) = ctx.saved_tensors
        weight = ctx.weight_ref
        ep_group = ctx.ep_group
        send_counts = ctx.send_counts
        recv_counts = ctx.recv_counts
        group_list = ctx.group_list
        hcom, ep_world_size = _resolve_hcom(ep_group)
        # dX + re-derived permute: one npu_alltoallv_gmm (counts UN-swapped) gives dX (1st) and
        # permute_grad (3rd = the reverse-a2a output = grad of the gmm output). Native weight.
        gmm_x_grad, _, permute_grad = torch_npu.npu_alltoallv_gmm(
            gmm_x=alltoall_out_grad,
            gmm_weight=weight,
            hcom=hcom,
            ep_world_size=ep_world_size,
            send_counts=send_counts,
            recv_counts=recv_counts,
            send_counts_tensor=None,
            recv_counts_tensor=None,
            mm_x=None,
            mm_weight=None,
            trans_gmm_weight=False,
            trans_mm_weight=False,
            permute_out_flag=True,
        )
        # dW: grad_out=permute_grad (re-derived), gmm_input=gmm_x (the saved FC2 input).
        weight_grad = _dw_matmul_loop(weight, gmm_x, permute_grad, group_list)
        # Inputs: (gmm_x, weight, w_fwd, ep_group, send, recv, group_list) -> 7 grad slots.
        return gmm_x_grad, weight_grad, None, None, None, None, None


def fused_dispatch_mlp_combine(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    act_fn: Any,
    topk_ids: torch.Tensor,
    n_routed_experts: int,
    ep_group: dist.ProcessGroup,
) -> torch.Tensor:
    """Run the fused dispatch + FC1 + activation + FC2 + combine alltoallv.

    Mirrors MindSpeed ``dispatcher_mc2.dispatch_mlp_combine``: compute counts once, fuse the dispatch
    a2a+permute+FC1, apply the gated activation, then fuse FC2+combine a2a. The result is in the
    same sort order as the input (``dispatch_preprocess``'s sort), so the caller applies the final
    ``unpermute(..., row_id_map, probs=topk_weights)``.

    Args:
        hidden_states (torch.Tensor): sort-permuted dispatch input ``[n_local, hidden]``.
        w1 (torch.Tensor): native FC1 (gate+up) weight ``[E_local, 2 * inter, hidden]``.
        w2 (torch.Tensor): native FC2 (down) weight ``[E_local, hidden, inter]``.
        act_fn (Any): the MoE activation callable ``act_fn(fused_x, split_dim=-1)``.
        topk_ids (torch.Tensor): per-token routed expert ids ``[n_local, topk]``.
        n_routed_experts (int): global expert count.
        ep_group (dist.ProcessGroup): the expert-parallel process group.

    Returns:
        torch.Tensor: the combined hidden states ``[n_local, hidden]``, in the input's sort order.
    """
    # Materialize the transposed weights ``[E, in, out]`` (the op needs this layout with
    # ``trans_gmm_weight=False``) BEFORE the metadata ``all_to_all_single`` in ``_compute_counts``,
    # and DETACHED under ``no_grad``. This detach is the activation-checkpoint recompute-nan fix:
    # under the whole-MoE-layer reentrant checkpoint the backward re-runs this function with grad
    # enabled, and a grad-tracked ``.contiguous()`` clone (a CopySlices non-leaf) lets the fused
    # op's grouped-matmul stage read an invalid weight buffer and emit nan on most ranks (run52:
    # the a2a stage, reading the stable activation leaf, stays finite; only the gmm stage that
    # reads the weight clone nans). Detaching makes ``w1_fwd``/``w2_fwd`` stable leaves, so the gmm
    # reads a valid weight and the re-issue stays finite (run67/68/69, 10-step stable).
    # The native ``[E, out, in]`` ``w1``/``w2`` are also passed so the hand-written backward reads
    # them unchanged (dX sibling-op + dW loop -> native layout). ``w1_fwd``/``w2_fwd`` are
    # forward-only: their backward grad slot returns ``None`` (dW flows to the native weight via
    # ``weight_grad``, read live through ``ctx.weight_ref``). Memory-lean: no extra tensor is saved
    # -- the clone is forward-only, and the backward reads the native weight live.
    with torch.no_grad():
        w1_fwd = w1.transpose(1, 2).contiguous()
        w2_fwd = w2.transpose(1, 2).contiguous()
    send_counts, recv_counts, group_list = _compute_counts(topk_ids, n_routed_experts, ep_group)
    gate_up_out = AlltoallvPermuteGmm.apply(hidden_states, w1, w1_fwd, ep_group, send_counts, recv_counts, group_list)
    intermediate = act_fn(gate_up_out, split_dim=-1)
    combined = GmmUnpermuteAlltoallv.apply(intermediate, w2, w2_fwd, ep_group, send_counts, recv_counts, group_list)
    return combined


def fused_forward(
    decoder_layer: MoEDecoderLayer,
    *,
    router_results: RouterResults,
    pre_dispatched: PreDispatchResult,
    shared_input_hidden: torch.Tensor,
    residual: torch.Tensor,
    origin_shape: torch.Size,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the fused dispatch+combine a2a+grouped-matmul MoE forward.

    Mirrors ``dispatcher_mc2.dispatch_mlp_combine``: fuse dispatch a2a+permute+FC1, apply the gated
    activation, fuse FC2+combine a2a, then unpermute to the pre-dispatch token order, and merge the
    shared-experts output + residual via ``_post_moe_forward``. Byte-identical to the former inlined
    ``MoEDecoderLayer._fused_a2a_gmm_forward`` (relocated verbatim, ``self.`` -> ``decoder_layer.``).

    The weight-prep (FSDP2 ``to_local`` + ``view``) + the shared-experts/post-moe merge wrap the
    same-module ``fused_dispatch_mlp_combine`` (which owns the two ``AlltoallvPermuteGmm`` /
    ``GmmUnpermuteAlltoallv`` autograd calls and the detached weight clones). The ``cast`` calls
    mirror the gate's ``isinstance`` narrowing (the gate already guaranteed the types) -- needed
    only because this reads the dispatcher/experts outside the gate's narrowed scope.

    Args:
        decoder_layer (MoEDecoderLayer): the calling layer (supplies the FSDP2-local expert weights,
            the MoE activation, the shared-experts forward, the post-moe merge, and the routed-expert
            count -- the existing private helpers, called unchanged).
        router_results (RouterResults): router outputs (``topk_ids``/``topk_weights``/``logits``/
            ``router_weights``).
        pre_dispatched (PreDispatchResult): the ``TorchAll2AllDispatcher.dispatch_preprocess`` output
            (a ``TorchAll2AllPreDispatchResult`` -- the gate in ``MoEDecoderLayer`` guarantees this).
        shared_input_hidden (torch.Tensor): the pre-dispatch hidden states fed to the shared expert
            (independent of the routed tokens' sort-permute).
        residual (torch.Tensor): the residual stream for the post-moe merge.
        origin_shape (torch.Size): the pre-flatten hidden-state shape to restore after unpermute.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ``(hidden_states, router_logits,
        router_weights, topk_ids)`` -- the ``MoEDecoderLayer._pre_moe_forward`` return.
    """
    w1 = cast(GroupedLinear, decoder_layer.experts.fused_w1w3)
    w1_w = w1.weight.to_local() if isinstance(w1.weight, DTensor) else w1.weight
    w1_w = w1_w.view(-1, w1.local_out_features, w1.local_in_features)
    w2 = cast(GroupedLinear, decoder_layer.experts.fused_w2)
    w2_w = w2.weight.to_local() if isinstance(w2.weight, DTensor) else w2.weight
    w2_w = w2_w.view(-1, w2.local_out_features, w2.local_in_features)
    pre = cast(TorchAll2AllPreDispatchResult, pre_dispatched)
    dispatcher = cast(TorchAll2AllDispatcher, decoder_layer.dispatcher)
    ep_group = dispatcher._process_group
    assert ep_group is not None, "fused a2a+gmm requires a non-trivial EP group"
    combined_hidden = fused_dispatch_mlp_combine(
        hidden_states=pre["hidden_states"],
        w1=w1_w,
        w2=w2_w,
        act_fn=decoder_layer.experts.moe_act,
        topk_ids=router_results["topk_ids"],
        n_routed_experts=decoder_layer.n_routed_experts,
        ep_group=ep_group,
    )
    combined_hidden_states = unpermute(
        combined_hidden,
        pre["row_id_map"],
        probs=router_results["topk_weights"],
    )
    combined_hidden_states = combined_hidden_states.view(*origin_shape)
    shared_experts_out: torch.Tensor | None = None
    if decoder_layer.n_shared_experts > 0:
        shared_experts_out = decoder_layer._shared_experts_forward(hidden_states=shared_input_hidden)
    hidden_states = decoder_layer._post_moe_forward(
        combined_hidden_states=combined_hidden_states,
        residual=residual,
        shared_experts_out=shared_experts_out,
    )
    return (
        hidden_states,
        router_results["logits"],
        router_results["router_weights"],
        router_results["topk_ids"],
    )
