# Selective checkpoint policy for the sequence-parallel all-gather.
#
# Under CheckpointImpl.NO_REENTRANT the layer forward is re-run during backward,
# which re-issues the SP ``all_gather_into_tensor`` collective. At >=256 ranks that
# re-issued async collective flaky-deadlocks the backward recompute. Marking the SP
# gather's native op ``MUST_SAVE`` makes ``torch.utils.checkpoint`` keep the
# gathered tensor and skip re-issuing the collective on recompute, removing the
# deadlock surface (and one collective from the backward).
#
# The SP gather is matched by its resolved process-group name (recorded here at the
# gather call site), NOT by op name -- FSDP2's parameter all-gather shares the same
# op name and must never be MUST_SAVE'd (it would retain ~46 GB of unsharded params
# on 744B). The ``tag`` arg of ``all_gather_tensor_autograd`` is ignored for
# DeviceMesh/ProcessGroup groups by ``_resolve_group_name``, so it cannot tag the
# gather; the group-name match is the only reliable discriminator.
#
# The MoE dispatch all-to-all (a second re-issued collective in the same region) is
# NOT targeted: the policy matches only the SP gather's op name, and the all-to-all
# is a different collective whose re-issue does not exhibit the SP-gather deadlock,
# so saving the SP gather alone is sufficient. (Async dispatch wraps it in
# _AsyncDispatch/_AsyncCombine custom Functions; MUST_SAVE on those is untested
# but unneeded.)
import os
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import _resolve_group_name
from torch.distributed.device_mesh import DeviceMesh
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

# Group names that have flowed through ``gather_for_sequence_parallel``. The policy
# MUST_SAVE's only the gathers whose group_name is in this set.
_SP_GROUP_NAMES: set[str] = set()
_SP_GATHER_OP_NAME = "all_gather_into_tensor"


def _sp_group_name_of(group: DeviceMesh | dist.ProcessGroup) -> str:
    """Return the resolved ``group_name`` str the native all_gather op receives.

    Mirrors ``_resolve_group_name`` so the captured name is byte-identical to the
    one the policy inspects. Falls back to private attrs if the helper is hidden.
    """
    try:
        name = _resolve_group_name(group, "")
        if isinstance(name, str) and name:
            return name
    except Exception:  # noqa: BLE001
        try:
            if isinstance(group, DeviceMesh):
                return group._dim_group_infos[0][2]
            if isinstance(group, dist.ProcessGroup):
                return group.group_name
        except Exception:  # noqa: BLE001
            return ""
    return ""


def record_sp_group(sp_mesh: DeviceMesh) -> None:
    """Record the SP mesh's group name for the selective-ckpt policy to match.

    No-op unless ``XTUNER_MOE_NO_RECOMPUTE_SP_GATHER=1`` (sel-ckpt on); the
    recorded name is consumed only where the ``context_fn`` is attached.
    Called from ``gather_for_sequence_parallel`` strictly before the native op
    dispatches, so the name is in the set before the policy inspects that op.
    Idempotent.
    """
    if os.environ.get("XTUNER_MOE_NO_RECOMPUTE_SP_GATHER", "0") != "1":
        return
    gname = _sp_group_name_of(sp_mesh)
    if gname:
        _SP_GROUP_NAMES.add(gname)


def sp_gather_selective_context_fn() -> tuple[Any, ...]:
    """Build a selective-checkpoint context that MUST_SAVE's the SP all-gather.

    Pass as ``context_fn=`` to ``checkpoint_wrapper(checkpoint_impl=NO_REENTRANT,
    context_fn=<this callable>)``. ``torch.utils.checkpoint`` calls it once per
    checkpoint region to obtain the (save, recompute) context pair.
    """
    def _policy(ctx: Any, op: Any, *args: Any, **kwargs: Any) -> CheckpointPolicy:
        try:
            name = op.name()
        except Exception:  # noqa: BLE001
            name = str(op)
        if _SP_GATHER_OP_NAME in name:
            # all_gather_into_tensor(self, group_size, group_name); group_name is
            # the only str positional arg.
            gname = next(
                (a for a in args if isinstance(a, str)),
                next((v for v in kwargs.values() if isinstance(v, str)), ""),
            )
            is_sp_gather = bool(gname) and gname in _SP_GROUP_NAMES
            return (
                CheckpointPolicy.MUST_SAVE
                if is_sp_gather
                else CheckpointPolicy.MUST_RECOMPUTE
            )
        return CheckpointPolicy.MUST_RECOMPUTE

    return create_selective_checkpoint_contexts(_policy)
