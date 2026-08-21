# Copyright (c) OpenMMLab. All rights reserved.
"""Custom DeviceMesh rank layout for intra-node FSDP + inter-node EP/SP.

When ``XTUNER_DEVICE_MESH=1`` and ``world_size > NODE_SIZE`` (16), the data
mesh ``(dp, sp, tp)`` and the expert mesh ``(fsdp, ep)`` are laid out so the
FSDP dimension (the ``dp`` / ``fsdp`` allGather/reduceScatter group) is packed
into the first 16 NPUs of each node, exploiting super-node zero-copy, while the
``sp`` / ``ep`` (alltoall) dimensions are pushed inter-node.

Only the *rank arrangement* changes; every group keeps the same size, so the
per-rank parameter memory is identical to the default row-major mesh. The
layout mirrors commit ``be66a4341d9fde069b9226972ba79f0adb6941ff`` of the
Ascend-ShangHai-LLM fork, ported additively: the default path (feature off) is
byte-identical to the original ``init_device_mesh``.

Call sites never duplicate the gate or inline diagnostics: they branch on
:func:`use_custom_mesh` and call :func:`build_custom_data_mesh` /
:func:`build_custom_expert_mesh`, which own the rank-0 confirmation print. The
unconditional ``init_device_mesh`` call stays untouched at every call site
(branching happens *around* it via ``elif`` / early-return, so the eager NCCL
group creation of the default mesh is skipped entirely when the custom layout
is active -- no double group construction).
"""

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh


NODE_SIZE = 16


def device_mesh_enabled() -> bool:
    """Return whether ``XTUNER_DEVICE_MESH=1`` is set."""
    return os.getenv("XTUNER_DEVICE_MESH", "0") == "1"


def use_custom_mesh(world_size: int) -> bool:
    """Return whether the custom rank layout should override the default mesh.

    Centralizes the feature gate (env on **and** the world spans more than one
    node) so call sites branch on a single predicate instead of duplicating the
    ``device_mesh_enabled() and world_size > NODE_SIZE`` check across the
    trainer data-mesh and the two model expert-mesh construction sites.

    Args:
        world_size (int): The distributed world size.

    Returns:
        bool: True iff the custom layout is enabled and the world spans more
        than one node -- the only case where rearranging ranks is meaningful.
    """
    return device_mesh_enabled() and world_size > NODE_SIZE


def build_custom_data_mesh(
    device: str,
    dp_size: int,
    sp_size: int,
    tp_size: int,
) -> DeviceMesh:
    """Build the ``(dp, sp, tp)`` mesh with the ``dp`` dim intra-node.

    Caller must guarantee ``use_custom_mesh(world_size)``. The ``dp`` (dense
    FSDP) dimension packs 16 ranks per node so the FSDP allGather/reduceScatter
    stays intra-node (zero-copy); ``sp`` moves to inter-node. Group *sizes* are
    unchanged versus the default mesh, so per-rank parameter memory is
    identical.

    Args:
        device (str): Device of the mesh.
        dp_size (int): Data-parallel (dense FSDP) dimension size.
        sp_size (int): Sequence-parallel dimension size.
        tp_size (int): Tensor-parallel dimension size.

    Returns:
        DeviceMesh: The custom ``(dp, sp, tp)`` mesh.
    """
    _validate_data_layout(dp_size, sp_size, tp_size)
    # Force CPU: DeviceMesh requires a CPU mesh tensor, but MoE ``__init__`` may
    # run under a meta-device default context, where ``torch.zeros`` without an
    # explicit device would allocate a meta tensor and DeviceMesh rejects it
    # (``ValueError: 'mesh' must be a CPU tensor, got device='meta'``).
    mesh_tensor = torch.zeros(dp_size, sp_size, tp_size, dtype=torch.long, device="cpu")
    for i in range(dp_size):
        for j in range(sp_size):
            for k in range(tp_size):
                mesh_tensor[i, j, k] = _rank_for_data(i, j, k, sp_size, tp_size)
    mesh = DeviceMesh(device, mesh_tensor, mesh_dim_names=("dp", "sp", "tp"))
    if dist.get_rank() == 0:
        print(f"data_mesh (XTUNER_DEVICE_MESH=1): {mesh}")
    return mesh


def build_custom_expert_mesh(
    device: str,
    fsdp_size: int,
    ep_size: int,
    fsdp_dim: str,
    ep_dim: str,
) -> DeviceMesh:
    """Build the ``(fsdp, ep)`` expert mesh with the ``fsdp`` dim intra-node.

    Caller must guarantee ``use_custom_mesh(world_size)``. The ``fsdp`` (expert
    FSDP) dimension packs 16 ranks per node (e.g. 32 ranks span 2 nodes) so the
    expert FSDP allGather/reduceScatter is mostly intra-node; ``ep`` moves
    inter-node. Group *sizes* are unchanged versus the default mesh.

    Args:
        device (str): Device of the mesh.
        fsdp_size (int): Expert FSDP dimension size (``world // ep_size``).
        ep_size (int): Expert-parallel dimension size.
        fsdp_dim (str): Mesh dim name for the fsdp axis.
        ep_dim (str): Mesh dim name for the ep axis.

    Returns:
        DeviceMesh: The custom ``(fsdp, ep)`` mesh.
    """
    _validate_expert_layout(fsdp_size, ep_size)
    # Force CPU (see build_custom_data_mesh for the meta-context rationale).
    mesh_tensor = torch.zeros(fsdp_size, ep_size, dtype=torch.long, device="cpu")
    for i in range(fsdp_size):
        for j in range(ep_size):
            mesh_tensor[i, j] = _rank_for_expert(i, j, ep_size, fsdp_size)
    mesh = DeviceMesh(device, mesh_tensor, mesh_dim_names=(fsdp_dim, ep_dim))
    if dist.get_rank() == 0:
        print(f"model_mesh (XTUNER_DEVICE_MESH=1): {mesh}")
    return mesh


def _rank_for_data(i: int, j: int, k: int, sp_size: int, tp_size: int) -> int:
    """Global rank for data-mesh coord (dp=i, sp=j, tp=k)."""
    return i % NODE_SIZE + i // NODE_SIZE * (NODE_SIZE * sp_size * tp_size) + (j * tp_size + k) * NODE_SIZE


def _rank_for_expert(i: int, j: int, ep_size: int, fsdp_size: int | None = None) -> int:
    """Global rank for expert-mesh coord (fsdp=i, ep=j).

    When ``fsdp_size`` is supplied and ``fsdp_size <= NODE_SIZE`` (i.e.
    ``ep_size >= world / NODE_SIZE``, e.g. EP64 on 512 ranks -> fsdp=8), the
    fsdp dim packs ``fsdp_size`` ranks per intra-node block and the ep dim
    packs ``NODE_SIZE // fsdp_size`` groups per node before going inter-node,
    so every fsdp allGather/reduceScatter group stays intra-node (zero-copy).
    When ``fsdp_size`` is ``None`` (legacy 3-arg callers) or ``> NODE_SIZE``
    (EP < 32 on 512 ranks), the original inter-node fsdp layout is used. The
    two branches coincide for ``fsdp_size == NODE_SIZE`` (EP32 on 512 ranks),
    so EP32 is byte-identical to the original layout.
    """
    if fsdp_size is not None and fsdp_size <= NODE_SIZE:
        groups_per_node = NODE_SIZE // fsdp_size
        return (j // groups_per_node) * NODE_SIZE + (j % groups_per_node) * fsdp_size + i
    return i % NODE_SIZE + i // NODE_SIZE * (NODE_SIZE * ep_size) + j * NODE_SIZE


def _validate_data_layout(dp_size: int, sp_size: int, tp_size: int) -> None:
    """Raise ``ValueError`` unless the data-mesh rank layout is a valid permutation.

    The intra-node FSDP packing in :func:`_rank_for_data` is a valid bijection
    onto ``0..world-1`` only when ``dp_size`` is a multiple of ``NODE_SIZE``;
    otherwise ranks collide or fall out of range (e.g. ``dp_size=8`` on a
    256-rank / SP32 cluster overflows to rank 503). Fail fast at construction
    rather than handing DeviceMesh a non-permutation mesh tensor.

    Args:
        dp_size (int): Data-parallel (dense FSDP) dimension size.
        sp_size (int): Sequence-parallel dimension size (kept for the message).
        tp_size (int): Tensor-parallel dimension size (kept for the message).
    """
    if dp_size % NODE_SIZE != 0:
        raise ValueError(
            f"XTUNER_DEVICE_MESH=1 requires dp_size ({dp_size}) to be a multiple"
            f" of NODE_SIZE ({NODE_SIZE}); the intra-node FSDP rank layout would"
            " not be a valid permutation. Disable XTUNER_DEVICE_MESH or resize the"
            f" cluster / sp_size so dp_size % {NODE_SIZE} == 0"
            f" (sp_size={sp_size}, tp_size={tp_size})."
        )


def _validate_expert_layout(fsdp_size: int, ep_size: int) -> None:
    """Raise ``ValueError`` unless the expert-mesh rank layout is a valid permutation.

    Two validity regimes mirror the two branches of :func:`_rank_for_expert`:
    when ``fsdp_size <= NODE_SIZE`` the intra-node block packing needs
    ``NODE_SIZE % fsdp_size == 0`` (e.g. fsdp=8 on EP64); when
    ``fsdp_size > NODE_SIZE`` the inter-node layout needs
    ``fsdp_size % NODE_SIZE == 0`` (e.g. fsdp=32 on EP16). Any other fsdp size
    yields a non-permutation (e.g. fsdp=6 on a 96-rank world, fsdp=24 on 384).

    Args:
        fsdp_size (int): Expert FSDP dimension size (``world // ep_size``).
        ep_size (int): Expert-parallel dimension size.
    """
    if fsdp_size <= NODE_SIZE:
        if NODE_SIZE % fsdp_size != 0:
            raise ValueError(
                f"XTUNER_DEVICE_MESH=1 requires NODE_SIZE ({NODE_SIZE}) to be"
                f" divisible by fsdp_size ({fsdp_size}) when fsdp_size <= NODE_SIZE;"
                " the intra-node expert FSDP layout would not be a valid"
                " permutation. Disable XTUNER_DEVICE_MESH or pick ep_size so that"
                f" world // ep_size divides {NODE_SIZE} (ep_size={ep_size})."
            )
    elif fsdp_size % NODE_SIZE != 0:
        raise ValueError(
            f"XTUNER_DEVICE_MESH=1 requires fsdp_size ({fsdp_size}) to be a"
            f" multiple of NODE_SIZE ({NODE_SIZE}) when fsdp_size > NODE_SIZE; the"
            " inter-node expert FSDP layout would not be a valid permutation."
            " Disable XTUNER_DEVICE_MESH or pick ep_size so that world // ep_size"
            f" is a multiple of {NODE_SIZE} (ep_size={ep_size})."
        )
