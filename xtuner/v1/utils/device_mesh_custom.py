# Copyright (c) OpenMMLab. All rights reserved.
"""Custom DeviceMesh rank layout for intra-node FSDP + inter-node EP/SP.

When ``XTUNER_DEVICE_MESH=1`` and ``world_size > NODE_SIZE`` (16), the data mesh ``(dp, sp, tp)`` and the expert mesh
``(fsdp, ep)`` are laid out so the FSDP dimension (the ``dp`` / ``fsdp`` allGather/reduceScatter group) is packed into
the first 16 NPUs of each node, exploiting super-node zero-copy, while the ``sp`` / ``ep`` (alltoall) dimensions are
pushed inter-node.

Only the *rank arrangement* changes; every group keeps the same size, so the per-rank parameter memory is identical to
the default row-major mesh. The layout mirrors commit ``be66a4341d9fde069b9226972ba79f0adb6941ff`` of the Ascend-
ShangHai-LLM fork, ported additively: the default path (feature off) is byte-identical to the original
``init_device_mesh``.

Call sites never duplicate the gate or inline diagnostics: they branch on :func:`use_custom_mesh` and call
:func:`build_custom_data_mesh` / :func:`build_custom_expert_mesh`, which own the rank-0 confirmation print. The
unconditional ``init_device_mesh`` call stays untouched at every call site (branching happens *around* it via ``elif``
/ early-return, so the eager NCCL group creation of the default mesh is skipped entirely when the custom layout is
active -- no double group construction).
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
    FSDP) dimension packs ``dp_size`` ranks per node (``NODE_SIZE`` when
    ``dp_size > NODE_SIZE``, else a sub-node block, e.g. 8 ranks for SP64) so the
    FSDP allGather/reduceScatter stays intra-node (zero-copy); ``sp`` moves to
    inter-node. Group *sizes* are unchanged versus the default mesh, so per-rank
    parameter memory is identical.

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
                mesh_tensor[i, j, k] = _rank_for_data(i, j, k, sp_size, tp_size, dp_size)
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


def build_custom_expert_mesh_3d(
    device: str,
    fsdp_size: int,
    ep_size: int,
    etp_size: int,
    fsdp_dim: str,
    ep_dim: str,
    etp_dim: str,
) -> DeviceMesh:
    """Build the ``(fsdp, ep, etp)`` expert mesh with the ``fsdp`` dim intra-node.

    Extension of :func:`build_custom_expert_mesh` for ``expert_tp_size > 1``.
    The ``fsdp`` (expert FSDP) dimension packs 16 ranks per node so the FSDP
    allGather/reduceScatter stays intra-node; the ``(ep, etp)`` plane maps onto
    the data mesh's ``sp`` axis via ``sigma(ep, etp) = ep * etp_size + etp``, so
    the fsdp group (fix ``ep, etp``) is exactly the data-mesh dp group (fix
    ``sp = sigma(ep, etp)``) by construction -- FSDP2 gradient coherence holds.
    The ``etp`` (expert tensor-parallel) group is inter-node; it carries the
    small expert-TP allreduce, an acceptable cost for keeping the dominant FSDP
    comm intra-node.

    Caller must guarantee ``use_custom_mesh(world_size)`` and the alignment
    precondition (see :func:`validate_expert_3d_alignment`):
    ``tp_size == 1 and ep_size * etp_size == sp_size`` (so the ``(ep, etp) ->
    sp`` bijection covers every node and ``fsdp_size == dp_size``).

    Args:
        device (str): Device of the mesh.
        fsdp_size (int): Expert FSDP dim size (``world // (ep_size * etp_size)``).
        ep_size (int): Expert-parallel dimension size.
        etp_size (int): Expert tensor-parallel dimension size.
        fsdp_dim (str): Mesh dim name for the fsdp axis.
        ep_dim (str): Mesh dim name for the ep axis.
        etp_dim (str): Mesh dim name for the etp axis.

    Returns:
        DeviceMesh: The custom ``(fsdp, ep, etp)`` mesh.
    """
    _validate_expert_3d_layout(fsdp_size, ep_size, etp_size)
    # Force CPU (see build_custom_data_mesh for the meta-context rationale).
    mesh_tensor = torch.zeros(fsdp_size, ep_size, etp_size, dtype=torch.long, device="cpu")
    for i in range(fsdp_size):
        for j in range(ep_size):
            for k in range(etp_size):
                mesh_tensor[i, j, k] = _rank_for_expert_3d(i, j, k, ep_size, etp_size)
    mesh = DeviceMesh(device, mesh_tensor, mesh_dim_names=(fsdp_dim, ep_dim, etp_dim))
    if dist.get_rank() == 0:
        print(f"model_mesh (XTUNER_DEVICE_MESH=1, etp={etp_size}): {mesh}")
    return mesh


def override_expert_3d_mesh(
    default_mesh: DeviceMesh,
    device: str,
    fsdp_size: int,
    ep_size: int,
    etp_size: int,
    world_size: int,
    fsdp_dim: str,
    ep_dim: str,
    etp_dim: str,
) -> DeviceMesh:
    """Return the custom 3D expert mesh, or ``default_mesh`` when inactive.

    Single gate-and-build call site for MoE expert-mesh construction: when the
    custom layout is active (``use_custom_mesh(world_size)``), build and return
    the ``(fsdp, ep, etp)`` custom mesh; otherwise return ``default_mesh``
    unchanged. This lets the MoE call site keep its original
    ``init_device_mesh(...)`` line byte-identical and append a single override
    line, instead of an inline ``if/else`` branch that restructures the default
    mesh construction (the original logic stays untouched).

    The ``default_mesh`` built by the caller's ``init_device_mesh`` is, when
    overridden, a discarded CPU mesh tensor only -- DeviceMesh creates its HCCL
    process groups lazily on first ``get_group``/subscript, so an unreferenced
    default mesh never creates a group and there is no double-group
    construction.

    Args:
        default_mesh (DeviceMesh): The default row-major mesh to fall back to.
        device (str): Device of the custom mesh.
        fsdp_size (int): Expert FSDP dim size.
        ep_size (int): Expert-parallel dimension size.
        etp_size (int): Expert tensor-parallel dimension size.
        world_size (int): The distributed world size.
        fsdp_dim (str): Mesh dim name for the fsdp axis.
        ep_dim (str): Mesh dim name for the ep axis.
        etp_dim (str): Mesh dim name for the etp axis.

    Returns:
        DeviceMesh: The custom 3D mesh when active, else ``default_mesh``.
    """
    if not use_custom_mesh(world_size):
        return default_mesh
    return build_custom_expert_mesh_3d(device, fsdp_size, ep_size, etp_size, fsdp_dim, ep_dim, etp_dim)


def validate_expert_3d_alignment(
    ep_size: int,
    expert_tp_size: int,
    sp_size: int,
    tp_size: int,
    world_size: int,
) -> None:
    """Fail fast if the custom 3D expert mesh cannot align with the data mesh.

    The 3D rank formula (:func:`_rank_for_expert_3d`) maps ``(ep, etp)`` onto the
    data mesh's ``sp`` axis via a bijection, which is a valid *and aligned*
    layout only when ``tp_size == 1`` (so ``(ep, etp)`` bijects to ``sp`` alone)
    and ``ep_size * expert_tp_size == sp_size`` (so ``fsdp_size == dp_size`` and
    the fsdp group equals the dp group). Otherwise the custom expert mesh would
    diverge from the custom data mesh -- the FSDP reduce-scatter would cross
    ranks with different data-parallel coordinates, silently corrupting gradients
    (nan/inf). This guard is the single place all four sizes are visible (the
    trainer); the model mesh builder cannot see ``sp_size`` / ``tp_size``.

    No-op when the custom 3D path is inactive (feature off, single node, or
    ``expert_tp_size <= 1``), so the default path stays byte-identical.

    Args:
        ep_size (int): Expert-parallel dimension size.
        expert_tp_size (int): Expert tensor-parallel dimension size.
        sp_size (int): Sequence-parallel dimension size (data mesh).
        tp_size (int): Tensor-parallel dimension size (data mesh).
        world_size (int): The distributed world size.
    """
    if not (device_mesh_enabled() and world_size > NODE_SIZE and expert_tp_size > 1):
        return
    if tp_size != 1 or ep_size * expert_tp_size != sp_size:
        raise ValueError(
            "XTUNER_DEVICE_MESH=1 with expert_tp_size > 1 requires tp_size == 1"
            " and ep_size * expert_tp_size == sp_size so the custom 3D expert"
            " fsdp group equals the data dp group (FSDP2 gradient coherence)."
            f" Got ep_size={ep_size}, expert_tp_size={expert_tp_size},"
            f" sp_size={sp_size}, tp_size={tp_size}"
            f" (ep*etp={ep_size * expert_tp_size}). Either set tp_size=1 and"
            " sp_size=ep_size*expert_tp_size, or disable XTUNER_DEVICE_MESH"
            " (DEVICE_MESH=0 keeps both meshes default-rowmajor, aligned)."
        )


def warmup_mesh_communicators(device: str, *meshes: DeviceMesh | None) -> None:
    """Eagerly init every non-trivial mesh-dim HCCL communicator before
    training.

    A mesh dim untouched by forward/backward (e.g. data ``sp``, or expert ``ep``
    once ``expert_tp_size > 1`` routes the dispatcher through the ``ep_tp`` 2-D
    sub-mesh) stays cold until the first gradient-norm allreduce in
    ``clip_grad_norm``. There its lazy ``createLink`` contends the RoCE/P2P
    resources the expert-TP communicators already hold and deadlocks (``Alloc
    transports failed``). A trivial all-reduce + barrier per dim here, on an
    idle synchronized device, makes every link setup happen once in isolation
    so the optimizer step reuses warm communicators instead of creating them.

    Gated by ``XTUNER_DEVICE_MESH`` (reuses the feature gate, no new env); a
    no-op otherwise so the default path stays byte-identical. ``None`` meshes
    and size-1 dims are skipped; process groups are de-duplicated by identity
    so a sub-mesh and its parent do not warm the same group twice.

    Args:
        device (str): Device of the warm-up scratch tensor (e.g. ``"npu"``).
        *meshes (DeviceMesh | None): Meshes to warm (data mesh + expert meshes).
    """
    if not device_mesh_enabled():
        return
    seen: set[int] = set()
    for mesh in meshes:
        if mesh is None:
            continue
        for dim in range(mesh.ndim):
            if mesh.size(dim) <= 1:
                continue
            pg = mesh.get_group(dim)
            if pg is None or id(pg) in seen:
                continue
            seen.add(id(pg))
            scratch = torch.zeros(1, device=device)
            dist.all_reduce(scratch, group=pg)
            dist.barrier(pg)


def _rank_for_data(i: int, j: int, k: int, sp_size: int, tp_size: int, dp_size: int | None = None) -> int:
    """Global rank for data-mesh coord (dp=i, sp=j, tp=k).

    When ``dp_size`` is supplied and ``dp_size <= NODE_SIZE`` (i.e. ``sp_size >= world / NODE_SIZE``, e.g. SP64 on 512
    ranks -> dp=8), the dp dim packs ``dp_size`` ranks per intra-node block and the ``(sp, tp)`` plane packs
    ``NODE_SIZE // dp_size`` groups per node before going inter-node, so every dense FSDP allGather/reduceScatter group
    stays intra-node (zero-copy). When ``dp_size`` is ``None`` (legacy callers, incl. :func:`_rank_for_expert_3d`) or
    ``> NODE_SIZE`` (e.g. dp=32 with SP16), the original inter-node dp layout is used. The two branches coincide for
    ``dp_size == NODE_SIZE`` (SP32 on 512 ranks), so SP32 is byte-identical to the original layout.
    """
    if dp_size is not None and dp_size <= NODE_SIZE:
        groups_per_node = NODE_SIZE // dp_size
        sp_tp = j * tp_size + k
        return (sp_tp // groups_per_node) * NODE_SIZE + (sp_tp % groups_per_node) * dp_size + i
    return i % NODE_SIZE + i // NODE_SIZE * (NODE_SIZE * sp_size * tp_size) + (j * tp_size + k) * NODE_SIZE


def _rank_for_expert(i: int, j: int, ep_size: int, fsdp_size: int | None = None) -> int:
    """Global rank for expert-mesh coord (fsdp=i, ep=j).

    When ``fsdp_size`` is supplied and ``fsdp_size <= NODE_SIZE`` (i.e. ``ep_size >= world / NODE_SIZE``, e.g. EP64 on
    512 ranks -> fsdp=8), the fsdp dim packs ``fsdp_size`` ranks per intra-node block and the ep dim packs ``NODE_SIZE
    // fsdp_size`` groups per node before going inter-node, so every fsdp allGather/reduceScatter group stays intra-
    node (zero-copy). When ``fsdp_size`` is ``None`` (legacy 3-arg callers) or ``> NODE_SIZE`` (EP < 32 on 512 ranks),
    the original inter-node fsdp layout is used. The two branches coincide for ``fsdp_size == NODE_SIZE`` (EP32 on 512
    ranks), so EP32 is byte-identical to the original layout.
    """
    if fsdp_size is not None and fsdp_size <= NODE_SIZE:
        groups_per_node = NODE_SIZE // fsdp_size
        return (j // groups_per_node) * NODE_SIZE + (j % groups_per_node) * fsdp_size + i
    return i % NODE_SIZE + i // NODE_SIZE * (NODE_SIZE * ep_size) + j * NODE_SIZE


def _rank_for_expert_3d(i: int, j: int, k: int, ep_size: int, etp_size: int) -> int:
    """Global rank for 3D expert-mesh coord (fsdp=i, ep=j, etp=k).

    Maps ``(ep, etp)`` onto the data-mesh ``sp`` axis via
    ``sigma(j, k) = j * etp_size + k`` and reuses :func:`_rank_for_data` so the
    fsdp group (fix ``ep, etp``) is exactly the data-mesh dp group (fix
    ``sp = sigma(j, k)``) by construction. Valid (aligned) only when
    ``tp_size == 1`` and ``ep_size * etp_size == sp_size``; see
    :func:`validate_expert_3d_alignment`.
    """
    return _rank_for_data(i, j * etp_size + k, 0, ep_size * etp_size, 1)


def _validate_data_layout(dp_size: int, sp_size: int, tp_size: int) -> None:
    """Raise ``ValueError`` unless the data-mesh rank layout is a valid
    permutation.

    Two validity regimes mirror the two branches of :func:`_rank_for_data`:
    when ``dp_size <= NODE_SIZE`` the sub-node block packing needs
    ``NODE_SIZE % dp_size == 0`` (e.g. dp=8 with SP64, one dp group per half
    node); when ``dp_size > NODE_SIZE`` the inter-node layout needs
    ``dp_size % NODE_SIZE == 0`` (e.g. dp=32 with SP16, two nodes per dp group).
    Any other dp size yields a non-permutation (e.g. dp=12 on 192 ranks).

    Note the framework itself (trainer ``_build_data_mesh``) only requires
    ``sp_size | world`` and ``tp_size * sp_size | world``; the NODE constraint
    here is specific to this custom intra-node rank layout, not a DeviceMesh /
    FSDP2 requirement -- ``XTUNER_DEVICE_MESH=0`` accepts any dp the framework
    accepts.

    Args:
        dp_size (int): Data-parallel (dense FSDP) dimension size.
        sp_size (int): Sequence-parallel dimension size (kept for the message).
        tp_size (int): Tensor-parallel dimension size (kept for the message).
    """
    if dp_size <= NODE_SIZE:
        if NODE_SIZE % dp_size != 0:
            raise ValueError(
                f"XTUNER_DEVICE_MESH=1 requires NODE_SIZE ({NODE_SIZE}) to be"
                f" divisible by dp_size ({dp_size}) when dp_size <= NODE_SIZE; the"
                " sub-node FSDP rank layout would not be a valid permutation."
                " Disable XTUNER_DEVICE_MESH or pick sp_size so that"
                f" world // (tp_size * sp_size) divides {NODE_SIZE}"
                f" (sp_size={sp_size}, tp_size={tp_size})."
            )
    elif dp_size % NODE_SIZE != 0:
        raise ValueError(
            f"XTUNER_DEVICE_MESH=1 requires dp_size ({dp_size}) to be a multiple"
            f" of NODE_SIZE ({NODE_SIZE}) when dp_size > NODE_SIZE; the inter-node"
            " FSDP rank layout would not be a valid permutation. Disable"
            " XTUNER_DEVICE_MESH or resize the cluster / sp_size so that"
            f" dp_size % {NODE_SIZE} == 0 (sp_size={sp_size}, tp_size={tp_size})."
        )


def _validate_expert_layout(fsdp_size: int, ep_size: int) -> None:
    """Raise ``ValueError`` unless the expert-mesh rank layout is a valid
    permutation.

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


def _validate_expert_3d_layout(fsdp_size: int, ep_size: int, etp_size: int) -> None:
    """Raise ``ValueError`` unless the 3D expert-mesh rank layout is a valid
    permutation.

    :func:`_rank_for_expert_3d` reuses :func:`_rank_for_data`, whose intra-node
    packing is a valid bijection onto ``0..world-1`` only when ``fsdp_size`` is
    a multiple of ``NODE_SIZE`` (same condition as the data mesh's
    ``dp_size % NODE_SIZE == 0``, since ``fsdp_size == dp_size`` under alignment).
    Fail fast rather than hand DeviceMesh a non-permutation tensor.

    Args:
        fsdp_size (int): Expert FSDP dim size (``world // (ep_size * etp_size)``).
        ep_size (int): Expert-parallel dimension size (kept for the message).
        etp_size (int): Expert tensor-parallel dim size (kept for the message).
    """
    if fsdp_size % NODE_SIZE != 0:
        raise ValueError(
            f"XTUNER_DEVICE_MESH=1 requires fsdp_size ({fsdp_size}) to be a multiple"
            f" of NODE_SIZE ({NODE_SIZE}) when expert_tp_size > 1; the 3D expert"
            " FSDP rank layout would not be a valid permutation. Disable"
            " XTUNER_DEVICE_MESH or pick ep_size / expert_tp_size so that"
            f" world // (ep_size * etp_size) is a multiple of {NODE_SIZE}"
            f" (ep_size={ep_size}, etp_size={etp_size})."
        )
