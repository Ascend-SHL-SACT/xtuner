# Copyright (c) OpenMMLab. All rights reserved.
"""Backward prefetch offset for FSDP2 K-fused groups.

This module isolates the ``XTUNER_FSDP_BACKWARD_PREFETCH_OFFSET`` logic from
``xtuner.v1.model.fsdp_fuse`` so the core sharding path stays unchanged.
"""

import os

import torch.nn as nn


def apply_backward_prefetch_offset(units: list[nn.Module]) -> None:
    """Shift FSDP2 backward all-gather prefetch target earlier in the reverse pass.

    FSDP2's default reverse-post-forward prefetch keeps the next unit's
    all-gather in-flight alongside the current unit's reduce-scatter. At large
    scale this creates symmetric rendezvous "giants" on the shared AICPU
    notify path. Issuing the AG earlier (by ``offset`` units) breaks the
    time-lock between AG and RS while preserving the overlap bandwidth.

    Args:
        units (list[nn.Module]): FSDP2 units in forward order. Each unit must
            expose ``set_modules_to_backward_prefetch``.
    """
    offset_env = os.environ.get("XTUNER_FSDP_BACKWARD_PREFETCH_OFFSET", "0")
    try:
        offset = int(offset_env)
    except ValueError:
        offset = 0

    if offset <= 0:
        return

    for i, cur in enumerate(units):
        target_idx = i - 1 - offset
        target = units[target_idx] if target_idx >= 0 else cur
        cur.set_modules_to_backward_prefetch([target])  # type: ignore
