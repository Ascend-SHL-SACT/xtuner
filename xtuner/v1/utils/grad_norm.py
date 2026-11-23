from typing import List

import torch
from torch.distributed.tensor import DTensor

from .dtensor import cal_total_norm, group_tensors_by_device_mesh_and_placements


def cal_grad_norm(grads: List[DTensor], dtype=torch.float32):
    if grads and grads[0].device.type == "npu":
        # Overflow-safe L2 for NPU (implementation and docs in
        # ops/moe/npu/fused_a2a_gmm.py): the plain sum-of-squares turns
        # finite-but-huge gradients into inf.
        from xtuner.v1.ops.moe.npu.fused_a2a_gmm import overflow_safe_cal_grad_norm

        return overflow_safe_cal_grad_norm(grads, dtype=dtype)
    grouped_grads = group_tensors_by_device_mesh_and_placements(grads)
    total_norms = []
    for grads in grouped_grads.values():
        total_norm = cal_total_norm(grads, norm_type=2.0, foreach=True, dtype=dtype)
        total_norms.append(total_norm)
    grad_norm = torch.linalg.vector_norm(torch.stack(total_norms), ord=2.0, dtype=dtype)
    grad_norm = grad_norm.to(grads[0].dtype)
    return grad_norm, grouped_grads
