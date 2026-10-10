# Copyright (c) OpenMMLab. All rights reserved.
"""Fused residual add + RMSNorm (``XTUNER_FUSED_ADD_RMS_NORM``).

The decoder-layer pattern ``s = residual + x; residual = s; y = norm(s)``
maps onto ``torch_npu.npu_add_rms_norm`` (one kernel producing both ``y`` and
``s``). The fused op keeps the add in higher precision internally, so ``y``
and the gradients differ from the unfused chain at the bf16 rounding level;
the sum output ``s`` is bitwise identical to ``residual + x``.

The path is opt-in; with the env unset the touched module executes its
original logic unchanged.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from torch.distributed.tensor import DTensor


if TYPE_CHECKING:
    from xtuner.v1.module.rms_norm.rms_norm import RMSNorm


@lru_cache(maxsize=1)
def fused_add_rms_norm_enabled() -> bool:
    """Whether ``XTUNER_FUSED_ADD_RMS_NORM=1`` enables the fused add + RMSNorm.

    Returns:
        bool: True when the env gate is on.
    """
    return os.environ.get("XTUNER_FUSED_ADD_RMS_NORM", "0") == "1"


def can_fuse_add_rms_norm(norm: RMSNorm) -> bool:
    """Whether ``norm`` is eligible for the fused add + RMSNorm path.

    Args:
        norm (RMSNorm): Candidate norm module.

    Returns:
        bool: True when the gate is on, the module is a plain (non
        zero-centered) RMSNorm and the active device backend is NPU, matching
        the ``npu_add_rms_norm`` semantics.
    """
    if not fused_add_rms_norm_enabled() or norm._type != "default":
        return False
    from xtuner.v1.utils.device import get_device

    return get_device() == "npu"


class _FusedAddRMSNormFn(torch.autograd.Function):
    """``npu_add_rms_norm`` forward with an ``npu_rms_norm_backward`` backward.

    torch_npu registers no autograd kernel for ``npu_add_rms_norm``, so this
    Function wires the fused forward to the same backward op the unfused
    ``add -> npu_rms_norm`` chain uses. Returns ``(y, s)`` where ``s = x1 + x2``
    (bitwise identical to the eager add) feeds the residual stream and
    ``y = rms_norm(s)`` replaces the norm output; the backward adds the
    residual-side gradient ``ds`` onto the norm-side gradient, mirroring the
    eager chain's gradient fan-out over the add node.
    """

    @staticmethod
    def forward(
        ctx,
        x1: torch.Tensor,
        x2: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import torch_npu

        y, rstd, s = torch_npu.npu_add_rms_norm(x1, x2, weight, eps)
        ctx.save_for_backward(s, weight, rstd)
        return y, s

    @staticmethod
    def backward(
        ctx,
        dy: torch.Tensor,
        ds: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None]:
        import torch_npu

        s, weight, rstd = ctx.saved_tensors
        dx, dw = torch_npu.npu_rms_norm_backward(dy, s, weight, rstd)
        if ds is not None:
            dx = dx + ds.to(dx.dtype)
        return dx, dx, dw, None


def fused_add_rms_norm(
    x1: torch.Tensor,
    x2: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused ``rms_norm(x1 + x2)`` plus the raw sum in one NPU kernel.

    Args:
        x1 (torch.Tensor): First addend (residual side).
        x2 (torch.Tensor): Second addend (branch output side).
        weight (torch.Tensor): Norm weight, plain (non zero-centered) RMSNorm.
        eps (float): Norm epsilon.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(y, s)`` with ``y = rms_norm(s)``
        and ``s = x1 + x2`` bitwise identical to the eager add.
    """
    return _FusedAddRMSNormFn.apply(x1, x2, weight, eps)  # type: ignore[return-value]


def fused_add_rms_norm_module(
    norm: RMSNorm,
    x1: torch.Tensor,
    x2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``fused_add_rms_norm`` reading weight/epsilon from an ``RMSNorm`` module.

    Args:
        norm (RMSNorm): Plain (``type="default"``) RMSNorm module.
        x1 (torch.Tensor): First addend (residual side).
        x2 (torch.Tensor): Second addend (branch output side).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(s, y)`` in residual/normed order,
        replacing ``s = x1 + x2; y = norm(s)``.
    """
    weight = norm.weight.to_local() if isinstance(norm.weight, DTensor) else norm.weight
    y, s = fused_add_rms_norm(x1, x2, weight, norm.variance_epsilon)
    return s, y
