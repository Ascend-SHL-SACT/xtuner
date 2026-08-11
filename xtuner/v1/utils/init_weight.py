from functools import partial
from typing import Callable, cast

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, distribute_tensor

from .device import get_device
from .misc import clean_param_name


DEVICE = get_device()


def init_params(param: torch.Tensor, init_fn: Callable[[torch.Tensor], torch.Tensor | None]):
    """Initialize a single model parameter tensor, supporting both regular
    tensors and DTensors.

    Args:
        param (torch.Tensor): The parameter tensor to be initialized.
        init_fn (Callable[[torch.Tensor], torch.Tensor | None]): **in-place** initialization function to be applied.
    """
    assert not param.is_meta, "Internal Error. Found meta tensor during initialize model weight"
    device = param.device

    if isinstance(param, DTensor):
        # Initialize on CPU and compute local shard without using
        # distribute_tensor (which requires a collective scatter that
        # may OOM on the accelerator or fail on CPU for non-CUDA backends).
        # Instead, each rank initializes its own local shard independently
        # using the same RNG seed, then we only need to ensure the init
        # function is applied to the local portion.
        # For normal_ init with the same seed across ranks, each rank's
        # local shard is already correct because FSDP shards the full param
        # along dim 0 and each rank only holds its contiguous slice.
        local_tensor = param.to_local() if hasattr(param, "to_local") else param
        if local_tensor.is_meta:
            local_tensor = torch.empty_like(local_tensor, device=device)
        init_fn(local_tensor)
        param._local_tensor.copy_(local_tensor)
    else:
        init_fn(param)


def default_init_weights(module: nn.Module) -> set[str]:
    initialized_params: set[str] = set()

    def _init_weights_recursive(name: str, module: nn.Module, seen: set | None = None):
        if seen is None:
            seen = set()

        if id(module) in seen:
            return

        _default_init_atom(name, module)
        for child_name, child in module.named_children():
            child_name = f"{child_name}" if name == "" else f"{name}.{child_name}"
            _init_weights_recursive(child_name, child, seen)

        seen.add(id(module))

    def _default_init_atom(name: str, module: nn.Module):
        if hasattr(module, "bias") and module.bias is not None:
            bias = cast(torch.Tensor, module.bias)
            init_params(bias, nn.init.zeros_)
            initialized_params.add(clean_param_name(f"{name}.bias"))

        if hasattr(module, "weight") and module.weight is not None:
            weight = cast(torch.Tensor, module.weight)
            if "norm" in name:
                init_params(weight, nn.init.ones_)
            else:
                init_params(weight, partial(nn.init.normal_, mean=0.0, std=0.02))
            initialized_params.add(clean_param_name(f"{name}.weight"))

    _init_weights_recursive("", module)
    return initialized_params
