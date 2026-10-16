import os
from collections.abc import Iterable, Iterator
from typing import Any

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.optim.adam import adam as torch_adam

from xtuner.v1.utils import get_device, get_logger, get_torch_device_module


DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()
logger = get_logger()


class SwapAdamW(torch.optim.AdamW):
    """AdamW optimizer with optimizer-state swap between device and host.

    Optimizer states are kept canonically on pinned CPU memory. During optimizer step, state tensors are copied to
    temporary device buffers, updated by fused kernels, and copied back to CPU.
    """

    _state_keys = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        *,
        maximize: bool = False,
        foreach: bool | None = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: bool | None = None,
        swap_optimizer_times: int = 16,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )
        self._swap_optimizer_times = swap_optimizer_times
        self._param_to_group_map: dict[torch.Tensor, dict] = {}
        self._param_to_cpu_states_map: dict[torch.Tensor, dict[str, torch.Tensor | None]] = {}
        # Created lazily by step_overlap on its first overlapping step; declared
        # here so type checkers see them (None keeps the lazy-init branch intact).
        self._swap_copy_stream: "torch.cuda.Stream | None" = None
        self._swap_d2h_event: "torch.cuda.Event | None" = None
        self._init_swap_states()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load optimizer state and rebind the pinned CPU state map.

        ``Optimizer.load_state_dict`` replaces ``self.state`` entries with fresh
        tensors, while the swap hot paths read m/v from
        ``_param_to_cpu_states_map`` (pinned buffers built at construction);
        without a rebind a checkpoint resume silently restarts m/v from zeros.

        Args:
            state_dict (Mapping[str, Any]): optimizer state to load.
        """
        super().load_state_dict(state_dict)
        self._rebind_cpu_states()

    def wrap_cpu_states_for_dcp(self, optim_state_dict: dict[str, Any]) -> None:
        """Wrap the pinned CPU m/v state values as DTensors for DCP save/load.

        ``get_optimizer_state_dict`` passes the swap optimizer's plain pinned
        CPU buffers through unchanged, so ``dcp.save`` sees an identical
        (offset-0, local-size) chunk on every rank and the global dedup keeps
        only one rank's shard -- multi-rank checkpoints silently lose all other
        ranks' m/v. Wrapping each buffer as a DTensor over its parameter's mesh
        and placements (zero-copy: the DTensor local IS the pinned buffer)
        gives DCP the global coordinates it needs, and ``dcp.load`` reshards
        straight into the buffers. Non-DTensor parameters and values that are
        not swap buffers (e.g. ``step``, ``None``) are left untouched.

        Args:
            optim_state_dict (dict[str, Any]): the optimizer part of a DCP
                state dict as built by ``get_optimizer_state_dict``.
        """
        buffers = self._iter_pinned_buffers()
        wrap_targets: dict[int, DTensor] = {id(buf): param for param, buf in buffers}
        state = optim_state_dict.get("state", {})
        for entry in state.values():
            if not isinstance(entry, dict):
                continue
            for key, value in entry.items():
                if not isinstance(value, torch.Tensor) or isinstance(value, DTensor):
                    continue
                param = wrap_targets.get(id(value))
                if param is None:
                    continue
                # DTensor.__new__ instead of DTensor.from_local: from_local
                # converts the local tensor to the mesh device type, which would
                # copy the pinned CPU buffer onto the device (and break the
                # load-side in-place reshard into the buffer). The raw
                # constructor wraps the buffer zero-copy with the parameter's
                # real mesh/placements, which is all DCP planning needs.
                spec = DTensorSpec(
                    mesh=param.device_mesh,
                    placements=tuple(param.placements),
                    tensor_meta=TensorMeta(shape=param.shape, stride=param.stride(), dtype=value.dtype),
                )
                entry[key] = DTensor(value, spec, requires_grad=False)

    def unwrap_cpu_states_after_dcp_load(self, optim_state_dict: dict[str, Any]) -> None:
        """Restore plain pinned buffers as the values of a DCP-loaded optimizer state dict.

        ``dcp.load`` reshards into the wrapped DTensors' locals in place -- and
        those locals ARE the pinned CPU buffers -- so unwrapping puts the
        already-filled buffers back as plain values before
        ``set_optimizer_state_dict`` hands them to ``load_state_dict`` (torch's
        optimizer load rejects mixed plain/DTensor copies).

        Args:
            optim_state_dict (dict[str, Any]): the optimizer part of a DCP
                state dict after ``dcp.load``.
        """
        buffer_ids = {id(buf) for _, buf in self._iter_pinned_buffers()}
        state = optim_state_dict.get("state", {})
        for entry in state.values():
            if not isinstance(entry, dict):
                continue
            for key, value in entry.items():
                # _local_tensor (not to_local()): to_local routes through an
                # autograd view under grad mode, so it is not the buffer object.
                local = getattr(value, "_local_tensor", None)
                if isinstance(local, torch.Tensor) and id(local) in buffer_ids:
                    entry[key] = local

    def _iter_pinned_buffers(self) -> Iterator[tuple[DTensor, torch.Tensor]]:
        for group in self.param_groups:
            for param in group["params"]:
                cpu_state = self._param_to_cpu_states_map.get(param)
                if cpu_state is None or not isinstance(param, DTensor):
                    continue
                for key in self._state_keys:
                    buf = cpu_state.get(key)
                    if isinstance(buf, torch.Tensor):
                        yield param, buf

    @staticmethod
    def _to_local_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if hasattr(tensor, "to_local"):
            return tensor.to_local()  # type: ignore[no-any-return]
        return tensor

    def _init_swap_states(self) -> None:
        # XTUNER_SWAP_BF16_STATE=1: keep m/v as bf16 on CPU to halve PCIe transfer.
        # Dispatches to step_overlap, which upcasts to fp32 for adam.
        self._swap_bf16_state = int(os.environ.get("XTUNER_SWAP_BF16_STATE", "0")) == 1
        _state_dtype = torch.bfloat16 if self._swap_bf16_state else None
        for group in self.param_groups:
            for param in group["params"]:
                self._param_to_group_map[param] = group

        swap_num = sum(self._to_local_tensor(main_param).numel() for main_param in self._param_to_group_map)
        self.swap_numel = swap_num // self._swap_optimizer_times
        swap_memory = swap_num * 8 / 1024 / 1024
        logger.info(
            f"[Rank {DEVICE_MODULE.current_device()}] swap optimizer param num: {swap_num},  "
            f"param size: {swap_memory}MB\n",
            end="",
        )

        for group in self.param_groups:
            for param in group["params"]:
                cpu_state_dtensor = self.state[param]
                cpu_state: dict[str, torch.Tensor | None] = {}
                amsgrad = bool(self._param_to_group_map[param]["amsgrad"])
                local_param = self._to_local_tensor(param)

                for key in self._state_keys:
                    if key == "max_exp_avg_sq" and not amsgrad:
                        cpu_state_dtensor[key] = None
                        cpu_state[key] = None
                    else:
                        cpu_tensor = torch.zeros_like(
                            local_param, dtype=_state_dtype, memory_format=torch.preserve_format
                        )
                        cpu_tensor = cpu_tensor.to(device="cpu", non_blocking=True)
                        cpu_tensor = cpu_tensor.pin_memory()
                        cpu_state_dtensor[key] = cpu_tensor
                        cpu_state[key] = cpu_tensor

                # Keep per-parameter step semantics consistent with torch.optim.AdamW.
                cpu_state_dtensor["step"] = torch.tensor(0.0, dtype=torch.float32, device="cpu")

                self._param_to_cpu_states_map[param] = cpu_state

        DEVICE_MODULE.synchronize()

    def _rebind_cpu_states(self) -> None:
        # Copy loaded m/v back into the pinned construction-time buffers and
        # point self.state at the same buffers (single source of truth).
        # copy_ handles dtype casts (an fp32 checkpoint loaded in bf16 mode is
        # rounded into the bf16 buffers). Keys absent from the checkpoint are
        # zeroed to match stock AdamW's fresh-state semantics.
        for group in self.param_groups:
            for param in group["params"]:
                cpu_state = self._param_to_cpu_states_map.get(param)
                if cpu_state is None:
                    continue
                param_state = self.state[param]
                for key in self._state_keys:
                    buf = cpu_state[key]
                    if buf is None:
                        continue
                    loaded = param_state.get(key)
                    if loaded is None:
                        buf.zero_()
                    else:
                        if loaded.device.type != "cpu":
                            loaded = loaded.to(device="cpu")
                        buf.copy_(loaded)
                    param_state[key] = buf

    @torch.no_grad()
    def step(self, closure=None):
        # Dispatch to step_overlap when any overlap/bf16 knob is set; each is
        # independent (XTUNER_SWAP_BF16_STATE / D2H_OVERLAP / H2D_OVERLAP) and
        # they may be combined or used alone. All off -> stock serial path.
        if (
            int(os.environ.get("XTUNER_SWAP_D2H_OVERLAP", "0"))
            or int(os.environ.get("XTUNER_SWAP_H2D_OVERLAP", "0"))
            or int(os.environ.get("XTUNER_SWAP_BF16_STATE", "0"))
        ):
            from .swap_adamw_overlap import step_overlap

            return step_overlap(self, closure)
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        params_list = list(self._param_to_group_map.keys())

        for param in params_list:
            if param.grad is None:
                continue
            if param.grad.is_sparse:
                raise RuntimeError("AdamW does not support sparse gradients")

            group = self._param_to_group_map[param]
            param_state = self.state[param]
            step_tensor = param_state.get("step")
            if step_tensor is None:
                step_tensor = torch.tensor(0.0, dtype=torch.float32, device="cpu")
                param_state["step"] = step_tensor
            assert isinstance(step_tensor, torch.Tensor)

            amsgrad = bool(group["amsgrad"])
            beta1, beta2 = group["betas"]
            cpu_state = self._param_to_cpu_states_map[param]
            cpu_exp_avg = cpu_state["exp_avg"]
            cpu_exp_avg_sq = cpu_state["exp_avg_sq"]
            cpu_max_exp_avg_sq = cpu_state.get("max_exp_avg_sq", None)

            assert isinstance(cpu_exp_avg, torch.Tensor)
            assert isinstance(cpu_exp_avg_sq, torch.Tensor)
            exp_avg = cpu_exp_avg.to(device=DEVICE, non_blocking=True)
            exp_avg_sq = cpu_exp_avg_sq.to(device=DEVICE, non_blocking=True)

            max_exp_avg_sq: torch.Tensor | None = None
            if amsgrad:
                assert isinstance(cpu_max_exp_avg_sq, torch.Tensor)
                max_exp_avg_sq = cpu_max_exp_avg_sq.to(device=DEVICE, non_blocking=True)

            local_param = self._to_local_tensor(param)
            local_grad = self._to_local_tensor(param.grad)
            local_exp_avg = self._to_local_tensor(exp_avg)
            local_exp_avg_sq = self._to_local_tensor(exp_avg_sq)
            local_max_exp_avg_sq = self._to_local_tensor(max_exp_avg_sq) if max_exp_avg_sq is not None else None

            torch_adam(
                [local_param],
                [local_grad],
                [local_exp_avg],
                [local_exp_avg_sq],
                [local_max_exp_avg_sq] if local_max_exp_avg_sq is not None else [],
                [step_tensor],
                amsgrad=amsgrad,
                has_complex=torch.is_complex(local_param),
                lr=group["lr"],
                beta1=beta1,
                beta2=beta2,
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                decoupled_weight_decay=group["decoupled_weight_decay"],
            )

            cpu_exp_avg.copy_(exp_avg, non_blocking=True)
            cpu_exp_avg_sq.copy_(exp_avg_sq, non_blocking=True)
            if amsgrad and max_exp_avg_sq is not None:
                assert isinstance(cpu_max_exp_avg_sq, torch.Tensor)
                cpu_max_exp_avg_sq.copy_(max_exp_avg_sq, non_blocking=True)

        DEVICE_MODULE.synchronize()
        return loss
