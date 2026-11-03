import torch
from torch.optim import Optimizer as TorchOptimizer

from xtuner.v1.utils import get_device, get_torch_device_module


class Optimizer(TorchOptimizer):
    """Base optimizer that can move its existing tensor state between
    devices."""

    def put_state_to_device(self, device: torch.device | str) -> bool:
        """Move existing tensor state onto ``device``.

        Args:
            device (torch.device | str): Target device. An accelerator without an index is the current device.

        Returns:
            bool: Whether any state tensor moved.
        """
        if not self.state:
            return False

        target = torch.device(device)
        # An accelerator device without an index names the current device.
        if target.type != "cpu" and target.index is None:
            target = torch.device(target.type, get_torch_device_module().current_device())
        # Under torch_npu's transfer_to_npu shim a `torch.device("cuda")` argument keeps type
        # "cuda" while live tensors report "npu"; alias the requested accelerator to the runtime
        # accelerator so the per-tensor comparison below sees equivalent devices. No-op when the
        # types already match (plain CUDA or CPU runs).
        accel = get_device()
        if target.type != "cpu" and accel != "cpu" and target.type != accel:
            target = torch.device(accel, target.index)

        moved = False
        for state in self.state.values():
            if not isinstance(state, dict):
                continue
            for key, value in list(state.items()):
                # Parameters, gradients, and non-tensor metadata are not optimizer state entries.
                if isinstance(value, torch.Tensor) and value.device != target:
                    state[key] = value.to(target, non_blocking=True)
                    moved = True
        if moved:
            get_torch_device_module().synchronize()
        return moved
