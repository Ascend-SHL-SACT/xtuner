import os
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.optim.adam import adam as torch_adam

from xtuner.v1.utils import get_device, get_logger, get_torch_device_module

from .adamw import AdamW


DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()
logger = get_logger()


class SwapAdamW(AdamW):
    """AdamW optimizer with optimizer-state swap between device and host.

    Optimizer states are kept canonically on pinned CPU memory. During optimizer step, state tensors are copied to
    temporary device buffers, updated by fused kernels, and copied back to CPU.

    For sharded (``DTensor``) parameters each state entry is a plain pinned host buffer of the parameter's LOCAL
    shape (the value object shared by ``self.state`` and ``_param_to_cpu_states_map``), so every rank checkpoints
    only its own shard: the global-shape/placements ``DTensor`` view DCP needs is created zero-copy over the
    buffers by ``wrap_cpu_states_for_dcp`` on save and unwrapped back to the same buffers after load.
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
        # fused/capturable require device-side step tensors; the swap paths keep
        # states on pinned CPU and force CPU fp32 step scalars. Fail at
        # construction instead of a deep device error on the first step.
        if fused:
            raise ValueError("SwapAdamW does not support fused=True: step uses CPU fp32 step scalars.")
        if capturable:
            raise ValueError("SwapAdamW does not support capturable=True: step uses CPU step tensors.")
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
        # Guards add_param_group: groups added after construction would bypass
        # the CPU state map and be silently skipped by both step paths.
        self._swap_initialized = True

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """Reject post-construction parameter group additions.

        SwapAdamW builds the pinned CPU state map once at construction; groups
        added afterwards would be silently skipped by both step paths.

        Args:
            param_group (dict[str, Any]): the parameter group to add.
        """
        if getattr(self, "_swap_initialized", False):
            raise NotImplementedError(
                "SwapAdamW does not support add_param_group after construction: new parameters "
                "would be silently skipped by the CPU state map."
            )
        super().add_param_group(param_group)

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

    def put_state_to_device(self, device: torch.device | str) -> bool:
        """Keep the canonical optimizer state on the host."""
        return False

    @torch._disable_dynamo
    @torch.no_grad()
    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Load into the preallocated pinned host buffers in place.

        Follows the Optimizer.load_state_dict protocol (pre/post hooks, id map, in-place group update) but never
        delegates to torch.optim.AdamW: the official loader replaces state/group objects and casts moments to the
        parameter device, which would drop pin_memory. copy_ converts floating-point dtypes instead. An empty state
        explicitly resets history (the engine's args-only load). Pre/post hooks are replayed from the base class'
        private hook dicts (torch>=2.6), because calling ``super()`` would re-enter the device cast this method
        exists to avoid.

        NOTE: the engine only reaches this method through ``TrainEngine.load_dcp``, which fixes
        ``StateDictOptions(cpu_offload=True)``: the incoming moments are CPU ``DTensor``s (CPU local tensors before
        this PR), ``step`` is a plain CPU tensor, and they are the very objects held in ``self.state``
        (``.to("cpu")`` is a no-op on CPU tensors), so ``dcp.load`` has already written the checkpoint into these
        buffers. The ``copy_`` below is therefore a self-copy on the DCP path; it is kept so externally built
        ``state_dict``s load correctly as well.

        Args:
            state_dict (Mapping[str, Any]): Optimizer state dict in the ``torch.optim.Optimizer.load_state_dict``
                schema. ``param_groups`` parameter ids may be indices or FQNs; they are paired with this optimizer's
                parameters by position, as the official loader does.

        Returns:
            None
        """
        # 1. Pre-hooks first; a hook may return a replacement state dict.
        state_dict = dict(state_dict)
        for pre_hook in self._optimizer_load_state_dict_pre_hooks.values():
            result = pre_hook(self, state_dict)
            if result is not None:
                state_dict = result

        # 2. Validate the group layout and pair saved ids with live parameters by position.
        id_map = self._validate_state_layout(state_dict)

        # 3. Copy every state entry into the preallocated pinned buffers; an empty state resets history.
        self._copy_state_in_place(state_dict["state"], id_map)

        # 4. Merge hyperparameters into the existing groups in place.
        self._update_param_groups(state_dict["param_groups"])

        # 5. Post-hooks run last, as in torch.optim.Optimizer.load_state_dict.
        for post_hook in self._optimizer_load_state_dict_post_hooks.values():
            post_hook(self)

    def _validate_state_layout(self, state_dict: Mapping[str, Any]) -> dict[Any, torch.Tensor]:
        # 在改动任何 buffer 之前先校验 checkpoint 结构:保存的 group 数量、每组的参数数量、以及 saved state 的
        # id 集合都必须与本优化器完全一致。返回 saved-id -> 当前参数 的映射,供后续拷贝步骤使用。
        saved_groups: list[dict[str, Any]] = state_dict["param_groups"]
        saved_state: Mapping[Any, Mapping[str, Any]] = state_dict["state"]
        if len(saved_groups) != len(self.param_groups):
            raise ValueError("loaded state dict has a different number of parameter groups")
        if any(
            len(saved_group["params"]) != len(group["params"])
            for saved_group, group in zip(saved_groups, self.param_groups)
        ):
            raise ValueError(
                "loaded state dict contains a parameter group that doesn't match the size of optimizer's group"
            )

        id_map = {
            old_id: param
            for saved_group, group in zip(saved_groups, self.param_groups)
            for old_id, param in zip(saved_group["params"], group["params"])
        }
        # Require an exact id match: the official loader silently resets the history of parameters without saved state
        # (and keeps orphans), so a mismatch would resume training from mixed moments without any error.
        if saved_state and set(saved_state) != set(id_map):
            raise ValueError("loaded state dict contains a parameter state that doesn't match any parameter")
        return id_map

    def _copy_state_in_place(
        self, saved_state: Mapping[Any, Mapping[str, Any]], id_map: dict[Any, torch.Tensor]
    ) -> None:
        # 用 copy_ 把 checkpoint 中的 moment 和 step 写入预分配的 pinned host buffer:不替换任何 state 条目或
        # tensor 对象,保留 pin_memory。对 state key 或 local shape 不匹配的情况显式报错,而不是静默加载出错
        # 乱的数据;saved state 为空时表示清零历史。
        for param_id, param in id_map.items():
            target_state = self.state[param]
            # amsgrad=False keeps None placeholders for max_exp_avg_sq; they are not
            # part of the state contract, so ignore None entries on both sides.
            expected_keys = {key for key, value in target_state.items() if value is not None}
            if not saved_state:
                # The engine's args-only load passes an empty state: reset history instead of keeping it.
                for key in expected_keys:
                    self._to_local_tensor(target_state[key]).zero_()
                continue

            source_state = saved_state[param_id]
            source_keys = {key for key, value in source_state.items() if value is not None}
            if source_keys != expected_keys:
                raise ValueError(
                    f"loaded state dict has incompatible optimizer state keys for parameter {param_id}: "
                    f"got {sorted(source_keys)}, expected {sorted(expected_keys)}"
                )
            for key in expected_keys:
                target = self._to_local_tensor(target_state[key])
                source = self._to_local_tensor(source_state[key])
                assert isinstance(source, torch.Tensor)
                # copy_ broadcasts on mismatch, so an old or differently sharded checkpoint would load scrambled
                # moments silently; reject it with the parameter id and both shapes instead.
                if target.shape != source.shape:
                    raise ValueError(
                        f"loaded state dict has incompatible optimizer state shape for parameter {param_id} "
                        f"key {key}: got {tuple(source.shape)}, expected {tuple(target.shape)}"
                    )
                target.copy_(source)

    def _update_param_groups(self, saved_groups: list[dict[str, Any]]) -> None:
        # 把 checkpoint 中的超参(lr、betas、weight decay 等)原地合并进现有的 param group,
        # 使 step() 持续读到更新后的值。
        # Update in place so external references to these group objects (and step() below) see the new
        # hyperparameters. Unlike torch, this merges into the existing groups, so a key absent from the
        # checkpoint keeps its local value.
        for group, saved_group in zip(self.param_groups, saved_groups):
            # Shallow-copy per group so param_groups never aliases mutable values of the incoming state dict.
            saved_group = dict(saved_group)
            group.update({key: value for key, value in saved_group.items() if key != "params"})
            # Fall back to AdamW semantics when the checkpoint predates this key.
            group["decoupled_weight_decay"] = bool(saved_group.get("decoupled_weight_decay", True))

    @staticmethod
    def _to_local_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if hasattr(tensor, "to_local"):
            return tensor.to_local()  # type: ignore[no-any-return]
        return tensor

    def _init_swap_states(self) -> None:
        # XTUNER_SWAP_BF16_STATE=1: keep m/v as bf16 on CPU to halve PCIe transfer.
        # Dispatches to step_overlap, which upcasts to fp32 for adam.
        # Knobs are read once at construction; the per-step hot path never reads
        # environment variables.
        self._swap_bf16_state = int(os.environ.get("XTUNER_SWAP_BF16_STATE", "0")) == 1
        self._swap_d2h_overlap = int(os.environ.get("XTUNER_SWAP_D2H_OVERLAP", "0")) == 1
        self._swap_nchunks = max(1, int(os.environ.get("XTUNER_SWAP_OVERLAP_NCHUNKS", "16")))
        _state_dtype = torch.bfloat16 if self._swap_bf16_state else None
        for group in self.param_groups:
            for param in group["params"]:
                self._param_to_group_map[param] = group

        swap_num = sum(self._to_local_tensor(main_param).numel() for main_param in self._param_to_group_map)
        self.swap_numel = swap_num // self._swap_optimizer_times
        first_param = next(iter(self._param_to_group_map), None)
        elem_size = 2 if _state_dtype is not None else (first_param.element_size() if first_param is not None else 4)
        swap_memory = swap_num * 2 * elem_size / 1024 / 1024
        logger.info(
            f"[Rank {DEVICE_MODULE.current_device()}] swap optimizer param num: {swap_num},  "
            f"m+v size: {swap_memory}MB\n",
            end="",
        )

        for group in self.param_groups:
            amsgrad = bool(group["amsgrad"])
            for param in group["params"]:
                param_state = self.state[param]
                cpu_state: dict[str, torch.Tensor | None] = {}
                amsgrad = bool(self._param_to_group_map[param]["amsgrad"])
                local_param = self._to_local_tensor(param)

                for key in self._state_keys:
                    if key == "max_exp_avg_sq" and not amsgrad:
                        param_state[key] = None
                        cpu_state[key] = None
                    else:
                        # Create zeros directly on CPU: zeros_like on device would
                        # transiently hold an extra device-sized buffer, plus a
                        # D2H copy and a second copy into the pinned tensor.
                        cpu_tensor = torch.zeros_like(
                            local_param,
                            dtype=_state_dtype,
                            device="cpu",
                            memory_format=torch.preserve_format,
                        ).pin_memory()
                        param_state[key] = cpu_tensor
                        cpu_state[key] = cpu_tensor

                # Keep per-parameter step semantics consistent with torch.optim.AdamW.
                param_state["step"] = torch.tensor(0.0, dtype=torch.float32, device="cpu")

                self._param_to_cpu_states_map[param] = cpu_state

        DEVICE_MODULE.synchronize()

    @torch.no_grad()
    def step(self, closure=None):
        # Dispatch to step_overlap when any overlap/bf16 knob is set (read once
        # at construction). All off -> stock serial path. The former
        # XTUNER_SWAP_H2D_OVERLAP prefetch is removed: its H2D targets were
        # allocated on the copy stream -- device blocks allocated on a side
        # stream are never reused by the NPU allocator -- making it a verified
        # fragmentation source.
        if self._swap_d2h_overlap or self._swap_bf16_state:
            from .swap_adamw_overlap import step_overlap

            return step_overlap(self, closure)
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            amsgrad = bool(group["amsgrad"])
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                param_state = self.state[param]
                step_tensor = param_state["step"]
                assert isinstance(step_tensor, torch.Tensor)

                cpu_exp_avg = param_state["exp_avg"]
                cpu_exp_avg_sq = param_state["exp_avg_sq"]
                cpu_max_exp_avg_sq = param_state["max_exp_avg_sq"]
                assert isinstance(cpu_exp_avg, torch.Tensor)
                assert isinstance(cpu_exp_avg_sq, torch.Tensor)

                exp_avg = self._to_local_tensor(cpu_exp_avg).to(device=DEVICE, non_blocking=True)
                exp_avg_sq = self._to_local_tensor(cpu_exp_avg_sq).to(device=DEVICE, non_blocking=True)
                max_exp_avg_sq = None
                if amsgrad:
                    assert isinstance(cpu_max_exp_avg_sq, torch.Tensor)
                    max_exp_avg_sq = self._to_local_tensor(cpu_max_exp_avg_sq).to(device=DEVICE, non_blocking=True)

                torch_adam(
                    [self._to_local_tensor(param)],
                    [self._to_local_tensor(param.grad)],
                    [exp_avg],
                    [exp_avg_sq],
                    [max_exp_avg_sq] if max_exp_avg_sq is not None else [],
                    [step_tensor],
                    amsgrad=amsgrad,
                    has_complex=torch.is_complex(param),
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

                # Write the updated moments back into the same pinned host buffers.
                self._to_local_tensor(cpu_exp_avg).copy_(exp_avg, non_blocking=True)
                self._to_local_tensor(cpu_exp_avg_sq).copy_(exp_avg_sq, non_blocking=True)
                if amsgrad and max_exp_avg_sq is not None:
                    self._to_local_tensor(cpu_max_exp_avg_sq).copy_(max_exp_avg_sq, non_blocking=True)

        DEVICE_MODULE.synchronize()
        return loss
