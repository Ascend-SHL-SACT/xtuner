import os

import torch
from mindspeed.core.fusions.grouped_matmul import Ops


def npu_group_gemm(x: torch.Tensor, weights: torch.Tensor, split_sizes: torch.Tensor) -> torch.Tensor:
    if os.environ.get("XTUNER_MOE_SUBMODULE_FSDP", "0") == "1":
        from xtuner.v1.model.moe.expert_submodule_fsdp import grouped_gemm

        return grouped_gemm(x, weights, split_sizes)
    weights = weights.transpose(1, 2)

    out = Ops.gmm(x, weights, split_sizes, trans_b=False)

    return out
