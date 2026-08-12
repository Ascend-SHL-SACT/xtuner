"""Expert-weight sub-module FSDP for MoE (single switch ``XTUNER_MOE_SUBMODULE_FSDP``).

Consolidates three formerly-separate envs into one feature, all bound to the
single switch (default OFF):

  * **un-fuse of w1w3** -- build ``fused_w1`` (gate) + ``fused_w3`` (up) instead
    of one ``fused_w1w3`` (out=2*intermediate). Each is ~6 GiB on 744B vs the
    ~12 GiB fused block.
  * **sub-module FSDP2 sharding** -- each expert ``GroupedLinear`` (fused_w1 /
    fused_w3 / fused_w2) becomes its own all-gather unit, dropping the recompute
    all-gather from the whole-layer ~18 GiB to one ~6 GiB expert unit (the
    S=4096 backward-OOM class).
  * **custom grouped-gemm backward** -- manual grad (per-expert matmul loop)
    avoids MindSpeed ``npu_gmm_backward``, which raises EZ1001 on the
    non-contiguous grad slices the un-fused ``torch.cat`` forward emits. The
    fused w1w3 path gets a contiguous grad and is unaffected.

The three are inseparable: the un-fuse is a structural prerequisite for the
sub-module shard (the ``fused_w1``/``fused_w3`` attrs only exist when un-fused),
and the custom backward is a functional prerequisite for the un-fuse. Default
OFF keeps the fused baseline on the fast MindSpeed backward.

This module is the single home for the feature's logic; the call sites in
``group_gemm.py``, ``moe_decoder_layer.py`` and ``moe.py`` are thin unconditional
hooks -- every function here checks the switch internally and no-ops when off,
so the default fused path runs verbatim.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from mindspeed.core.fusions.grouped_matmul import Ops


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
    from torch.nn import Module

    from xtuner.v1.float8 import Float8Config


def is_enabled() -> bool:
    """Return whether expert sub-module FSDP is on (``XTUNER_MOE_SUBMODULE_FSDP=1``).

    Returns:
        bool: True when the feature is enabled. Default OFF.
    """
    return os.environ.get("XTUNER_MOE_SUBMODULE_FSDP", "0") == "1"


def grouped_gemm(
    x: torch.Tensor,
    weights: torch.Tensor,
    split_sizes: torch.Tensor,
) -> torch.Tensor:
    """Grouped GEMM with the feature's custom backward when on, MindSpeed gmm when off.

    Args:
        x (torch.Tensor): Input tensor.
        weights (torch.Tensor): Expert weights ``[E, OUT, IN]``.
        split_sizes (torch.Tensor): Per-expert token counts.

    Returns:
        torch.Tensor: Grouped GEMM output.
    """
    if is_enabled():
        return _NpuGroupedGemm.apply(x, weights, split_sizes)
    weights = weights.transpose(1, 2)
    return Ops.gmm(x, weights, split_sizes, trans_b=False)


def init_expert_linears(
    module: Module,
    *,
    hidden_size: int,
    moe_intermediate_size: int,
    n_routed_experts: int,
    moe_bias: bool,
    ep_mesh: DeviceMesh | None,
    float8_cfg: Float8Config | None,
) -> None:
    """Un-fuse the caller-built ``fused_w1w3`` into ``fused_w1`` + ``fused_w3`` when on.

    The caller (``MoEBlock.__init__``) always builds the fused ``fused_w1w3`` and
    ``fused_w2`` verbatim -- the default path is left untouched. This wiring is
    a no-op when the switch is off (the caller's linears stay as built). When
    on, the fused block is dropped and rebuilt as two un-fused linears so each
    shards as its own FSDP all-gather unit; ``fused_w2`` is identical in both
    paths and is left as the caller built it.

    Args:
        module (Module): The ``MoEBlock`` carrying the caller-built fused_w1w3.
        hidden_size (int): Model hidden size.
        moe_intermediate_size (int): Per-expert intermediate size.
        n_routed_experts (int): Number of routed experts.
        moe_bias (bool): Whether linears have bias.
        ep_mesh (DeviceMesh | None): EP device mesh, or ``None``.
        float8_cfg (Float8Config | None): Float8 config, or ``None``.
    """
    if not is_enabled():
        return
    # Lazy import breaks the load cycle: moe_group_linear's top-level
    # ``from xtuner.v1.ops import group_gemm`` reaches back into ops, which
    # imports this module; deferring build_grouped_linear to call-time (well
    # after init) avoids the cycle.
    from xtuner.v1.module.grouped_linear.moe_group_linear import build_grouped_linear

    del module.fused_w1w3
    module.fused_w1 = build_grouped_linear(
        hidden_size,
        moe_intermediate_size,
        n_routed_experts,
        moe_bias=moe_bias,
        ep_mesh=ep_mesh,
        float8_cfg=float8_cfg,
    )
    module.fused_w3 = build_grouped_linear(
        hidden_size,
        moe_intermediate_size,
        n_routed_experts,
        moe_bias=moe_bias,
        ep_mesh=ep_mesh,
        float8_cfg=float8_cfg,
    )


def expert_mlp_forward(
    module: Module,
    x: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    decoding: bool,
) -> torch.Tensor:
    """MoE expert MLP forward: cat w1/w3 when on, single w1w3 when off, then act + w2.

    Args:
        module (Module): The ``MoEBlock`` carrying fused_w1/fused_w3 (on) or
            fused_w1w3 (off), fused_w2, and moe_act.
        x (torch.Tensor): Input tensor.
        tokens_per_expert (torch.Tensor): Per-expert token counts.
        decoding (bool): Whether in decoding mode.

    Returns:
        torch.Tensor: Expert MLP output.
    """
    if is_enabled():
        gate_out = module.fused_w1(x, tokens_per_expert, decoding)
        up_out = module.fused_w3(x, tokens_per_expert, decoding)
        gate_up_out = torch.cat([gate_out, up_out], dim=-1)
    else:
        gate_up_out = module.fused_w1w3(x, tokens_per_expert, decoding)
    out = module.moe_act(gate_up_out, split_dim=-1)
    return module.fused_w2(out, tokens_per_expert, decoding)


def shard_expert_submodules(
    model: Module,
    layer: Module,
    *,
    layer_idx: int | None = None,
    is_mtp: bool = False,
    mtp_idx: int | None = None,
    mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy,
    offload_policy: CPUOffloadPolicy | None,
) -> None:
    """Shard each expert ``GroupedLinear`` (fused_w1/w3/w2) as its own FSDP unit.

    Thin wiring called unconditionally from the FSDP-shard loops; no-op when the
    switch is off or the layer has no experts (dense layers). Computes its own
    ``reshard_after_forward`` (mirroring the caller's layer-level computation) so
    the call site is a single statement and the caller's reshard line is
    untouched. Must run on the unwrapped layer (pre-checkpoint, ``.experts``
    reachable), before the layer-level ``_fully_shard``.

    Args:
        model (Module): The top-level model (provides ``_fully_shard``).
        layer (Module): A main decoder layer or an ``MTPLayer`` (for MTP, the
            experts live under ``layer.decoder_layer.experts``).
        layer_idx (int | None): Main-layer index; ``None`` for MTP.
        is_mtp (bool): Whether ``layer`` is an MTP layer.
        mtp_idx (int | None): MTP layer index; ``None`` for main layers.
        mesh (DeviceMesh): FSDP device mesh.
        mp_policy (MixedPrecisionPolicy): Mixed-precision policy.
        offload_policy (CPUOffloadPolicy | None): CPU offload policy, or ``None``.
    """
    if not is_enabled():
        return
    if is_mtp:
        decoder_layer = getattr(layer, "decoder_layer", None)
        experts = getattr(decoder_layer, "experts", None) if decoder_layer is not None else None
        reshard_after_forward = mtp_idx != len(model.mtp_block.layers) - 1
    else:
        experts = getattr(layer, "experts", None)
        last_layer = layer_idx >= len(model.layers) - 1 and model.mtp_block is None
        reshard_after_forward = (not last_layer) and model.fsdp_config.reshard_after_forward
    if experts is None or not hasattr(experts, "fused_w1"):
        return
    for expert_linear in (experts.fused_w1, experts.fused_w3, experts.fused_w2):
        model._fully_shard(
            mesh=mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
            offload_policy=offload_policy,
            module=expert_linear,
        )


def get_gate_up_proj_weight(experts: Module) -> torch.Tensor:
    """Return the gate_up_proj weight in the fused ``[E, 2*intermediate, hidden]`` layout.

    Used by the debug HF-expert forward: the fused ``fused_w1w3`` weight is
    returned directly; the un-fused ``fused_w1``/``fused_w3`` weights are
    stacked and reshaped back to the fused layout to align with the HF
    per-expert linear.

    Args:
        experts (Module): The ``MoEBlock`` (has fused_w1w3, or fused_w1+fused_w3).

    Returns:
        torch.Tensor: Gate-up projection weight ``[E*2*intermediate, hidden]``.
    """
    if hasattr(experts, "fused_w1w3"):
        return experts.fused_w1w3.weight
    gw = experts.fused_w1.weight.view(experts.num_routed_experts, experts.intermediate_size, experts.hidden_size)
    uw = experts.fused_w3.weight.view(experts.num_routed_experts, experts.intermediate_size, experts.hidden_size)
    return torch.stack([gw, uw], dim=1).reshape(
        experts.num_routed_experts * 2 * experts.intermediate_size, experts.hidden_size
    )


class _NpuGroupedGemm(torch.autograd.Function):
    """Differentiable wrapper around the (non-autograd) mindspeed ``npu_gmm``.

    On this torch_npu build ``mindspeed::npu_gmm`` does not register a usable
    autograd kernel: its backward (``npu_gmm_backward`` -> aclnnGroupedMatmulV4)
    raises ``EZ1001 ("groupType is 2 and x is not separated, x should be
    transposed")`` when the incoming grad is a non-contiguous slice -- which is
    exactly what the un-fused w1/w3 forward produces (its ``torch.cat`` along the
    last dim is split back into strided slices by autograd). The fused w1w3 path
    gets a contiguous grad and is unaffected, so this Function is gated by
    ``XTUNER_MOE_SUBMODULE_FSDP`` (the feature that un-fuses w1w3) and defaults
    off to keep the fused baseline on the fast MindSpeed backward.

    Forward keeps the validated ``npu_gmm`` call. Backward supplies a correct
    manual grad so the expert weights actually train without hitting
    ``npu_gmm_backward``:

      * ``grad_x  = Ops.gmm(grad_out, weight, ...)`` -- reuses the forward gmm
        (run under ``no_grad`` so it does not build a backward graph).
      * ``grad_weight`` -- per-expert ``torch.matmul`` loop. Plain matmul
        handles non-contiguous operands and empty groups (``tokens==0``)
        gracefully, unlike ``npu_gmm_backward``. Costs ~2-5% of step time
        (grad_weight is a small slice of the backward) which is acceptable for
        the memory lever (S=4096 un-block) it unblocks; the fused TGS path
        stays on the default (off) branch and pays nothing.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        split_sizes: torch.Tensor,
    ) -> torch.Tensor:
        # weight: [E, OUT, IN]; forward uses W^T = weight.transpose(1, 2) -> [E, IN, OUT].
        ctx.save_for_backward(x, weight)
        ctx.split_sizes = split_sizes
        # Cache per-expert sizes both as a CPU list and an on-device tensor, in
        # forward (low scheduler stress). The backward D2H ``.cpu()`` (512-61
        # 507001) and H2D ``.to('npu')`` (512-62 107020) stream-syncs are the
        # crash sites under S=4096 backward scheduler stress, so backward always
        # uses the cached values; the syncful fallback is the known-crashy path
        # and is not retained. split_sizes is forward-deterministic (dispatch
        # token counts); checkpoint recompute re-runs forward, re-caching the
        # (re-dispatched) value before its own backward runs.
        if isinstance(split_sizes, torch.Tensor):
            ctx.split_sizes_cpu = split_sizes.tolist()
            ctx.split_sizes_npu = split_sizes.to("npu")
        else:
            ctx.split_sizes_cpu = list(split_sizes)
            ctx.split_sizes_npu = torch.tensor(split_sizes, device="npu")
        w_t = weight.transpose(1, 2)
        return Ops.gmm(x, w_t, ctx.split_sizes_npu, trans_b=False)

    @staticmethod
    def backward(
        ctx,
        grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        x, weight = ctx.saved_tensors
        split_sizes = ctx.split_sizes
        # nan/inf debug print on its own XTUNER_GGEMM_DEBUG: the .item() calls
        # force a D2H sync per ggemm-bwd call, which stalls the 32-node backward,
        # so the 512 launch keeps it off.
        _dbg = int(os.environ.get("XTUNER_GGEMM_DEBUG", "0"))
        if _dbg:
            try:
                import torch.distributed as _d

                rk = _d.get_rank() if _d.is_initialized() else 0
            except Exception:  # noqa: BLE001
                rk = 0
            print(
                f"[GGEMM-BWD] rk={rk} grad_nan={torch.isnan(grad_out).any().item()} "
                f"grad_inf={torch.isinf(grad_out).any().item()} "
                f"gout={tuple(grad_out.shape)} w={tuple(weight.shape)}",
                flush=True,
            )
        with torch.no_grad():
            # grad_x: grad_out @ weight  ([n, OUT] @ [OUT, IN] -> [n, IN]); forward gmm, no bwd kernel.
            # Use the forward-cached NPU split_sizes so mindspeed's
            # ``cumsum(batch_sizes).to('npu')`` is a no-op H2D; the raw-CPU path
            # is the 107020 crash site (see forward comment). getattr fallback is
            # defensive (forward always caches now) -- kept as a safety net.
            _gmm_sizes = getattr(ctx, "split_sizes_npu", None)
            if _gmm_sizes is None:
                _gmm_sizes = split_sizes
            grad_x = Ops.gmm(grad_out, weight, _gmm_sizes, trans_b=False)
            # grad_weight[e] = grad_out_e^T @ x_e  ([OUT, n_e] @ [n_e, IN] -> [OUT, IN]).
            # Use the forward-cached CPU list to avoid split_sizes.cpu(): that D2H
            # forces AclrtSynchronizeStreamWithTimeout, which under S=4096
            # backward scheduler stress raises 507001 (task-scheduler internal
            # error) even with a healthy free-pool -- the 512-61 crash site.
            # getattr fallback is defensive (forward always caches now).
            _cached = getattr(ctx, "split_sizes_cpu", None)
            if _cached is not None:
                _sizes = _cached
            else:
                _sizes = split_sizes.cpu().tolist() if isinstance(split_sizes, torch.Tensor) else list(split_sizes)
            grad_weight = torch.zeros_like(weight)
            _off = 0
            for _e in range(weight.shape[0]):
                _s = _sizes[_e]
                if _s > 0:
                    grad_weight[_e] = grad_out[_off : _off + _s].t() @ x[_off : _off + _s]
                _off += _s
        return grad_x, grad_weight, None
