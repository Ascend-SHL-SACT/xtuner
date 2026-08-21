# Copyright (c) OpenMMLab. All rights reserved.
"""FSDP cross-layer collective fusion (``XTUNER_FSDP_FUSE_K``).

Group K consecutive decoder layers into a single FSDP2 unit so the per-layer
all-gather / reduce-scatter collectives collapse into one per group, cutting
the serial rendezvous count (the binding comm cost when transit is negligible,
i.e. count-bound). The forward loop still calls each layer individually; FSDP2's
group forward hooks unshard on the first layer's forward and reshard on the
last's, so backward hooks and activation recompute are unaffected
(``PRE_BACKWARD`` short-circuits the forward hooks, so the recompute re-run is a
no-op for unshard/reshard).

Default ``K=1`` = per-layer = byte-identical to the original sharding path
(the per-layer ``_fully_shard`` + prefetch stay inline in ``MoE.fully_shard``);
``MoE.fully_shard`` calls :func:`shard_decoder_layers` only when
``XTUNER_FSDP_FUSE_K > 1`` to take the fused group path.
"""

import re
from typing import TYPE_CHECKING

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed.tensor import DTensor, Replicate, distribute_tensor


if TYPE_CHECKING:
    from xtuner.v1.model.base import BaseModel


def shard_decoder_layers(
    model: "BaseModel",
    layers: list[nn.Module],
    *,
    mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy,
    offload_policy: CPUOffloadPolicy | None,
    fsdp_prefetch: bool,
    fuse_k: int,
) -> nn.Module:
    """Shard decoder layers as K-fused FSDP2 groups and chain forward-prefetch.

    The ``fuse_k <= 1`` (default) per-layer path stays inline in
    :meth:`MoE.fully_shard` (byte-identical to HEAD); this entry is called only
    when ``fuse_k > 1`` and delegates to :func:`shard_layer_groups`.

    Args:
        model (BaseModel): the owning model; provides ``fsdp_config``.
        layers (list[nn.Module]): decoder layers in forward order.
        mesh (DeviceMesh): device mesh for sharding.
        mp_policy (MixedPrecisionPolicy): mixed-precision policy.
        offload_policy (CPUOffloadPolicy | None): CPU offload policy.
        fsdp_prefetch (bool): chain group-N+1 unshard behind group-N compute.
        fuse_k (int): group size (``XTUNER_FSDP_FUSE_K``); must be ``> 1``.

    Returns:
        nn.Module: the last decoder-layer module, reused to chain the decoder ->
        first-MTP-layer forward prefetch. Any member of the last FSDP2 group
        works (a group shares one FSDP state), so ``layers[-1]`` is returned.
    """
    assert model.fsdp_config is not None
    assert fuse_k > 1, "shard_decoder_layers is the K>1 grouped path only; K<=1 stays inline in the caller"
    shard_layer_groups(
        model,
        layers,
        mesh=mesh,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        fsdp_prefetch=fsdp_prefetch,
        fuse_k=fuse_k,
    )
    return layers[-1]


def shard_layer_groups(
    model: "BaseModel",
    layers: list[nn.Module],
    *,
    mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy,
    offload_policy: CPUOffloadPolicy | None,
    fsdp_prefetch: bool,
    fuse_k: int,
) -> None:
    """Shard consecutive decoder-layer groups as single FSDP2 units (K-fusion).

    Splits ``layers`` into contiguous groups of size ``fuse_k`` and shards each
    group as one FSDP2 unit (:func:`_shard_group`), so the per-layer all-gather
    / reduce-scatter collectives collapse into one per group. A trailing group
    of size 1 is sharded with the model's per-layer ``_fully_shard`` to stay
    byte-identical to the default path for the remainder. Forward-prefetch is
    chained across group representatives.

    Args:
        model (BaseModel): the owning model; provides ``_fully_shard``,
            ``_clean_param_name``, ``load_spec_mapping``, ``world_mesh``,
            ``config.hf_save_cfg.fp32_keys_pattern``, ``fsdp_config`` and
            ``mtp_block``.
        layers (list[nn.Module]): decoder layers in forward order.
        mesh (DeviceMesh): device mesh for sharding.
        mp_policy (MixedPrecisionPolicy): mixed-precision policy.
        offload_policy (CPUOffloadPolicy | None): CPU offload policy.
        fsdp_prefetch (bool): chain group-N+1 unshard behind group-N compute.
        fuse_k (int): group size (``XTUNER_FSDP_FUSE_K``).
    """
    assert model.fsdp_config is not None
    groups = [layers[i : i + fuse_k] for i in range(0, len(layers), fuse_k)]
    units: list[nn.Module] = []
    for gi, grp in enumerate(groups):
        if gi == len(groups) - 1 and model.mtp_block is None:
            reshard_after_forward = False
        else:
            reshard_after_forward = model.fsdp_config.reshard_after_forward
        if len(grp) == 1:
            model._fully_shard(
                mesh=mesh,
                mp_policy=mp_policy,
                reshard_after_forward=reshard_after_forward,
                offload_policy=offload_policy,
                module=grp[0],
            )
        else:
            _shard_group(
                model,
                grp,
                mesh=mesh,
                mp_policy=mp_policy,
                reshard_after_forward=reshard_after_forward,
                offload_policy=offload_policy,
            )
        units.append(grp[0])
    if fsdp_prefetch:
        for cur, nxt in zip(units[:-1], units[1:]):
            cur.set_modules_to_forward_prefetch([nxt])  # type: ignore


def _shard_group(
    model: "BaseModel",
    modules: list[nn.Module],
    *,
    mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy,
    reshard_after_forward: bool,
    offload_policy: CPUOffloadPolicy | None,
) -> None:
    """Shard a list of sibling modules as a single FSDP2 unit (collective fusion).

    Mirrors :meth:`BaseModel._fully_shard` but treats ``modules`` as one FSDP2
    unit: it collects the fp32-keys-pattern ignored params across every module
    in the group, then calls :func:`fully_shard` on the whole list so the
    per-layer all-gather / reduce-scatter collapse into one per group.

    Args:
        model (BaseModel): the owning model.
        modules (list[nn.Module]): sibling modules (none a child of another) to
            manage as one FSDP2 unit.
        mesh (DeviceMesh): device mesh for sharding.
        mp_policy (MixedPrecisionPolicy): mixed-precision policy.
        reshard_after_forward (bool): reshard the whole group after forward.
        offload_policy (CPUOffloadPolicy | None): CPU offload policy.
    """
    full_param_name_mapping = {id(param): name for name, param in model.named_parameters()}
    ignored_params: set[nn.Parameter] = set()
    patterns = model.config.hf_save_cfg.fp32_keys_pattern

    if patterns:

        def traverse(module: nn.Module) -> None:
            for name, param in module.named_parameters(recurse=False):
                full_name = full_param_name_mapping[id(param)]
                full_name = model._clean_param_name(full_name)
                load_spec = model.load_spec_mapping.get(full_name)
                if load_spec is None:
                    raise ValueError(f"Internal Error. Parameter {full_name} not found in load_spec_mapping.")
                hf_name_list = load_spec.global_hf_keys
                for hf_name in hf_name_list:
                    if any(re.search(p, hf_name) for p in patterns):  # type: ignore
                        if not isinstance(param, DTensor):
                            assert model.world_mesh is not None
                            dist_param = nn.Parameter(
                                distribute_tensor(
                                    param,
                                    model.world_mesh,
                                    [Replicate() for _ in range(model.world_mesh.ndim)],
                                ),
                                requires_grad=param.requires_grad,
                            )
                            module.register_parameter(name, dist_param)
                            ignored_params.add(dist_param)
                        else:
                            ignored_params.add(param)
                        break
            for child in module.children():
                traverse(child)

        for m in modules:
            traverse(m)

    fully_shard(
        modules,
        mesh=mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
        offload_policy=offload_policy,
        ignored_params=ignored_params if ignored_params else None,
    )
