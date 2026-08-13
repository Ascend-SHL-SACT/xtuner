from typing import Tuple

import torch


def npu_token_permute(
    input_act: torch.Tensor,
    indices: torch.Tensor,
    num_topK: int | None = None,
    num_out_tokens: int | None = None,
    num_negative_one_in_indices: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if num_out_tokens is not None:
        raise NotImplementedError

    if num_negative_one_in_indices is not None:
        raise NotImplementedError

    if num_topK is not None:
        raise NotImplementedError

    from torch_npu import npu_moe_token_permute

    return npu_moe_token_permute(input_act, indices)


def npu_token_unpermute(
    input_act: torch.Tensor, row_id_map: torch.Tensor, probs: torch.Tensor | None = None
) -> torch.Tensor:
    from torch_npu import npu_moe_token_unpermute

    if probs is not None:
        probs = probs.to(torch.bfloat16)
        # NPU npu_moe_token_unpermute requires probs to be 2D [num_tokens, 1].
        # XTuner passes 1D probs [num_tokens] from topk_weights.
        if probs.dim() == 1:
            probs = probs.unsqueeze(-1)

    # Handle 0-size inputs: NPU aclnnMoeTokenUnpermuteGrad crashes on them
    # ("input shape has 0"). When input has 0 tokens (some experts receive
    # 0 tokens due to router imbalance), return a zero-size output directly.
    # This is numerically correct (0 tokens contribute 0 to the output)
    # and avoids calling the broken grad kernel.
    if input_act.shape[0] == 0:
        # Determine output shape: [num_original_tokens, hidden_dim]
        # When no tokens are dispatched to any expert on this rank,
        # the output should have 0 rows (matching the expected unpermute output).
        # The hidden dim is preserved from input_act.
        out = torch.zeros(
            0, input_act.shape[1], dtype=input_act.dtype, device=input_act.device
        )
        # Make it look like it came from input_act for autograd
        if input_act.requires_grad:
            out = out + input_act.sum() * 0  # connect to graph but add nothing
        return out

    return npu_moe_token_unpermute(input_act, row_id_map, probs=probs)
