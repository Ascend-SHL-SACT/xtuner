# Copyright (c) OpenMMLab. All rights reserved.
from typing import Annotated, Any, Literal, Sequence, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from cyclopts import Parameter
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.nn.functional import all_reduce

from xtuner.v1.utils.device import get_device

# from xtuner.v1.profiler.prober import ProberList
from .base_loss_ctx import BaseLossConfig, BaseLossContext, BaseLossKwargs
from .chunk_loss import ChunkLoss
from .utils import sp_gather, sp_split


DEVICE = get_device()


def _ensure_liger_npu() -> None:
    """Force liger_kernel to detect NPU + correct its triton CE forward loss.

    liger's ``infer_device()`` checks ``torch.cuda.is_available()`` first; under
    torch_npu distributed init that returns True (cuda-redirect), so liger
    mis-detects ``'cuda'``, the ``_ascend`` backend never swaps in, and the
    generic triton kernel runs and crashes (``MLIRCompilationError``) on NPU.
    Patch ``infer_device`` -> ``'npu'`` and re-trigger the one-shot backend swap.

    Additionally, liger-ascend's plain CE triton kernel is deterministically
    WRONG in multi-rank training: for the same real inputs that it computes
    correctly (74637) in light-load standalone, it returns ~661 in training --
    a ~100x too-small value that is byte-identical across runs (deterministic,
    not a race) and immune to every sync tried: ``torch.npu.synchronize()``,
    ``acl.rt.synchronize_device`` (CANN-level, reaches triton's raw stream),
    and a pre-kernel CANN sync that drains all in-flight transfers. The defect
    is in the triton-ascend/CANN runtime in the distributed context; the kernel's
    matmul (``logits = _input @ weight.t()``) is correct (eager uses the same
    matmul). ``_patch_liger_ascend_fwd`` therefore OVERWRITES ``loss_1d`` with
    a torch ``F.cross_entropy(reduction='none')`` value computed on the correct
    logits (default stream, training-correct). The no-weight backward derives
    logsumexp from ``loss_1d``, so the correction fixes both loss and grad
    while keeping liger's fused backward (no full grad-logits materialisation).
    The forward loss is read via ``reduction='none'`` + an explicit sum (see
    ``chunk_mode``). Idempotent; no-op off NPU.
    """
    import sys

    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        return
    import liger_kernel.utils as _liger_utils

    if _liger_utils.infer_device() != "npu":
        _liger_utils.infer_device = lambda: "npu"
        ops_mod = sys.modules.get("liger_kernel.ops")
        if ops_mod is not None and hasattr(ops_mod, "_replace_with_impl_ops"):
            ops_mod._replace_with_impl_ops()
        flce_mod = sys.modules.get("liger_kernel.transformers.fused_linear_cross_entropy")
        if flce_mod is not None and ops_mod is not None:
            flce_mod.LigerFusedLinearCrossEntropyFunction = (
                ops_mod.LigerFusedLinearCrossEntropyFunction
            )
    _patch_liger_ascend_fwd()


def _patch_liger_ascend_fwd() -> None:
    """Correct liger-ascend's FLCE forward loss via torch CE.

    The liger-ascend plain CE triton kernel is deterministically WRONG in
    multi-rank training (returns ~661 vs eager ~74637 for the same real
    inputs that are correct in standalone), and the defect is immune to
    every sync -- ``torch.npu.synchronize()``, ``acl.rt.synchronize_device``
    (CANN-level, reaches triton's raw stream), and a pre-kernel CANN sync
    that drains all in-flight transfers. So it is neither a read-before-write
    race nor drainable memory aliasing; it is a triton-ascend/CANN defect in
    the distributed context. The kernel's matmul (``logits = _input @
    weight.t()``) IS correct (eager uses the identical matmul -> 74637); only
    the CE *kernel* miscomputes ``loss_1d``.

    The no-weight backward path derives logsumexp FROM ``loss_1d``
    (``lse = loss_1d + logits[target]``) and reads the saved logits, so a
    wrong ``loss_1d`` propagates to a wrong grad (247934 vs 5.26). Correcting
    ``loss_1d`` with a torch ``F.cross_entropy(reduction='none')`` (default
    stream, correct in training) after the triton forward therefore fixes
    BOTH the loss and the backward grad, while keeping liger's fused backward
    (no full grad-logits materialisation). The triton kernel is still run
    (cheap) so its saved tensors/ce_stats stay available to the backward.

    Idempotent; no-op off NPU or if the ascend FLCE module is absent.
    """
    import importlib

    try:
        flce = importlib.import_module(
            "liger_kernel.ops.backends._ascend.ops.fused_linear_cross_entropy"
        )
    except ImportError:
        return
    if getattr(flce, "_xtuner_fwd_patched", False):
        return
    _orig_fwd = flce.fused_linear_cross_entropy_forward

    def _correct_fwd(*args, **kwargs):
        out = _orig_fwd(*args, **kwargs)
        # out = (loss, None, None, None, loss_1d, ce_stats, None,
        #        plain_fast_path, logits_for_backward)
        if not isinstance(out, tuple) or len(out) < 9:
            return out
        loss_1d = out[4]
        logits = out[8]
        _input = kwargs.get("_input")
        weight = kwargs.get("weight")
        target = kwargs.get("target")
        ignore_index = kwargs.get("ignore_index", -100)
        reduction = kwargs.get("reduction", "mean")
        if _input is None and args:
            _input = args[0]
        if weight is None and len(args) > 1:
            weight = args[1]
        if target is None and len(args) > 2:
            target = args[2]
        if logits is None and _input is not None and weight is not None:
            logits = _input @ weight.t()
        if logits is None or target is None or loss_1d is None:
            return out
        # torch CE on the (correct) logits -- default stream, training-correct.
        correct = F.cross_entropy(
            logits.float(),
            target,
            reduction="none",
            ignore_index=ignore_index,
        )
        out_list = list(out)
        out_list[0] = correct if reduction == "none" else correct.sum()
        out_list[4] = correct
        return tuple(out_list)

    flce.fused_linear_cross_entropy_forward = _correct_fwd
    flce._xtuner_fwd_patched = True


class CELossConfig(BaseLossConfig):
    """Cross-entropy loss configuration.

    Args:
        ignore_idx (int): The index to ignore in the loss computation.
            Defaults to -100.
        mode (str): The mode for loss computation. Options are "eager" and "chunk".
            Defaults to "eager".
        chunk_size (int | None): The chunk size for "chunk" mode. Ignored if mode is "eager".
            Defaults to 1024.
        loss_reduction (str): The reduction mode for the loss. Options are "token", "sample", and "square".
    """

    mode: Annotated[Literal["eager", "chunk", "liger"], Parameter(help="loss calculation mode")] = "eager"  # type: ignore
    loss_reduction: Annotated[Literal["token", "sample", "square"], Parameter(help="loss reduction mode")] = "token"

    @property
    def loss_ctx_cls(self) -> type["CELossContext"]:
        return CELossContext

    @property
    def _loss_kwargs_cls(self) -> type["CELossKwargs"]:
        return CELossKwargs

    def model_post_init(self, _context: Any) -> None:
        if self.mode == "liger":
            assert self.loss_reduction == "token", "Currently, cannot use liger kernel with sample or square reduction"

    def build(
        self,
        data: dict,
        sp_mesh: DeviceMesh | None = None,
    ) -> "CELossContext | None":
        """Build CELossContext from data dict.

        Args:
            data (dict): Data dict containing loss-related fields.
                Required: shifted_labels
            sp_mesh (DeviceMesh | None): Sequence parallel mesh.

        Returns:
            CELossContext | None: Built loss context. Returns None if shifted_labels
                is not present in data dict.
        """
        if "shifted_labels" not in data:
            return None
        # Extract required fields from data
        shifted_labels = data["shifted_labels"]

        loss_kwargs = CELossKwargs(shifted_labels=shifted_labels).to(DEVICE)
        if sp_mesh is not None and sp_mesh.size() > 1:
            loss_kwargs = loss_kwargs.sp_split(sp_mesh)
        loss_ctx = self.loss_ctx_cls(self, loss_kwargs)
        return loss_ctx


class CELossKwargs(BaseLossKwargs):
    """Keyword arguments for cross-entropy loss computation.

    Args:
        shifted_labels (torch.Tensor): The shifted labels for the input sequences.
        loss_weight (torch.Tensor): The weight for each token in the loss computation.
    """

    shifted_labels: torch.Tensor
    loss_weight: torch.Tensor | None = None

    def sp_split(self, sp_mesh: DeviceMesh) -> "CELossKwargs":
        self.shifted_labels = sp_split(self.shifted_labels, sp_mesh=sp_mesh, split_dim=1, padding_value=-100)
        return self

    def to(self, device: torch.device | str) -> "CELossKwargs":
        self.shifted_labels = self.shifted_labels.to(device)
        return self


class LMHeadLossContext(BaseLossContext):
    """Cross-entropy loss context for language models.

    Args:
        loss_cfg (CELossConfig): The configuration for the cross-entropy loss.
        loss_kwargs (CELossKwargs): The keyword arguments for the cross-entropy loss.
    """

    loss_cfg: CELossConfig
    loss_kwargs: CELossKwargs

    def __init__(self, loss_cfg: CELossConfig, loss_kwargs: CELossKwargs):
        super().__init__(loss_cfg, loss_kwargs)

        self._liger_is_npu = False
        if loss_cfg.mode == "liger":
            self._liger_is_npu = hasattr(torch, "npu") and torch.npu.is_available()
            if self._liger_is_npu:
                _ensure_liger_npu()
            from liger_kernel.transformers.fused_linear_cross_entropy import (
                LigerFusedLinearCrossEntropyLoss,
            )

            # NPU: reduction='none' so FLCE returns per-token loss_1d, which
            # _patch_liger_ascend_fwd overwrites with torch CE (ascend triton
            # miscomputes it multi-rank) and the no-weight backward reads for
            # lse. GPU keeps upstream 'sum' -- cuda kernel is correct, no patch.
            self.liger_loss_fct = LigerFusedLinearCrossEntropyLoss(
                reduction="none" if self._liger_is_npu else "sum",
                accum_dtype=torch.float32,
            )
        else:
            self.liger_loss_fct = None

    @staticmethod
    def build_batches(  # type: ignore[override]
        loss_ctx_list: list["CELossContext"],
        cu_seq_lens_list: Sequence[torch.IntTensor] | None = None,
        sp_mesh: DeviceMesh | None = None,
    ) -> list["CELossContext"]:
        assert len(loss_ctx_list) > 0, "loss_ctx_list can not be empty"
        loss_cfg = loss_ctx_list[0].loss_cfg

        loss_weight_list: list[torch.Tensor] = []
        for i, loss_ctx in enumerate(loss_ctx_list):
            shifted_labels = loss_ctx.loss_kwargs.shifted_labels
            if loss_cfg.loss_reduction == "token":
                loss_weight = torch.ones_like(shifted_labels, dtype=torch.float32)
            else:
                assert cu_seq_lens_list is not None, "cu_seq_lens_list must be provided for sample or square reduction"
                cu_seq_lens = cu_seq_lens_list[i].to(shifted_labels.device)
                boundaries = cu_seq_lens[1:]
                num_tokens = cu_seq_lens[1:] - cu_seq_lens[:-1]

                if sp_mesh is not None:
                    # gather shifted_labels from different sp ranks to compute the correct loss weight
                    shifted_labels = sp_gather(shifted_labels, sp_mesh=sp_mesh, dim=1)

                mask = (shifted_labels != loss_cfg.ignore_idx).int()
                num_grad_tokens = torch.zeros_like(boundaries, dtype=torch.int32)
                prev_idx = 0
                for j, boundary in enumerate(boundaries):
                    num_grad_tokens[j] = mask[0, prev_idx:boundary].sum()
                    prev_idx = boundary
                if loss_cfg.loss_reduction == "sample":
                    loss_weight = 1.0 / num_grad_tokens
                elif loss_cfg.loss_reduction == "square":
                    loss_weight = 1.0 / torch.sqrt(num_grad_tokens.float())
                else:
                    raise NotImplementedError(loss_cfg.loss_reduction)
                loss_weight = loss_weight.repeat_interleave(num_tokens).unsqueeze(0)

                if sp_mesh is not None:
                    loss_weight = sp_split(loss_weight, sp_mesh=sp_mesh, split_dim=1, padding_value=0.0)
                    shifted_labels = sp_split(shifted_labels, sp_mesh=sp_mesh, split_dim=1, padding_value=-100)

            loss_weight[shifted_labels == loss_cfg.ignore_idx] = 0.0
            if torch.isnan(loss_weight).any() or torch.isinf(loss_weight).any():
                raise AssertionError(
                    "loss_weight contains NaN or Inf values. Please filter out samples with no valid tokens."
                )
            loss_ctx.loss_kwargs.loss_weight = loss_weight
            loss_weight_list.append(loss_weight)

        # Compute the denominator used in the global calibration of the loss
        rank_denominator = sum(loss_weight.sum() for loss_weight in loss_weight_list)
        rank_denominator = cast(torch.Tensor, rank_denominator)
        global_denominator = rank_denominator
        if dist.is_initialized():
            dist.all_reduce(global_denominator, op=dist.ReduceOp.SUM)

        for loss_ctx in loss_ctx_list:
            loss_ctx._batch_size = len(loss_ctx_list)
            assert loss_ctx.loss_kwargs.loss_weight is not None
            loss_ctx.loss_kwargs.loss_weight /= global_denominator + 1e-12
        return loss_ctx_list

    def loss_fn(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        loss_kwargs: CELossKwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, dict[str, Any]]]:
        # We do linear forward here to simplify the implementation of chunk loss (saving memory).
        logits = F.linear(hidden_states, head_weight, head_bias)
        logits = logits.float()  # (bs, seq_len, vocab_size)

        shifted_labels = loss_kwargs.shifted_labels  # (bs, seq_len)
        loss_weight = loss_kwargs.loss_weight  # (bs, seq_len)
        assert loss_weight is not None, "loss_weight can not be None"

        logits = logits.reshape(-1, logits.size(-1))  # (bs * seq_len, vocab_size)
        shifted_labels = shifted_labels.flatten()
        loss_weight = loss_weight.flatten()

        rank_grad_tokens = (shifted_labels != self.loss_cfg.ignore_idx).sum()
        if rank_grad_tokens == 0:
            loss = logits.sum() * 0
        else:
            loss = F.cross_entropy(logits, shifted_labels, reduction="none", ignore_index=self.loss_cfg.ignore_idx)
            # Step 2.b in the loss calculation: sum the loss over all tokens
            loss = (loss * loss_weight).sum()

        return loss, (logits, {})

    def eager_mode(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        loss_kwargs: CELossKwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, dict[str, Any]]]:
        return self.loss_fn(hidden_states, head_weight, head_bias, loss_kwargs)

    def chunk_mode(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        loss_kwargs: CELossKwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, dict[str, Any]]]:
        if self.loss_cfg.mode == "chunk":
            assert self.loss_cfg.chunk_size is not None, "chunk_size must be set in chunk mode"

            chunks = loss_kwargs.chunk(self.loss_cfg.chunk_size)
            loss, extra_info = ChunkLoss.apply(
                hidden_states, head_weight, head_bias, self.loss_fn, chunks, self.loss_cfg.chunk_size
            )
            return loss, (None, extra_info)
        else:
            assert self.liger_loss_fct is not None, "liger_loss_fct must be initialized in liger mode"
            shifted_labels = loss_kwargs.shifted_labels  # (bs, seq_len)
            loss_weight = loss_kwargs.loss_weight  # (bs, seq_len)
            assert loss_weight is not None, "loss_weight can not be None"

            bs, seq, dim = hidden_states.shape
            hidden_states = hidden_states.reshape(bs * seq, dim)
            shifted_labels = shifted_labels.flatten()
            # liger kernel dont support reduction=="none"
            # step 2.b in the loss calculation: sum the loss over all tokens, then multiply the loss weight (i.e. divide by the global_denominator)
            loss = self.liger_loss_fct(head_weight, hidden_states, shifted_labels)
            # ProberList.record_tensor(loss, "[lm_head.ce_loss][before calibration]loss")
            if self._liger_is_npu:  # NPU: reduction='none' -> per-token; sum to scalar
                loss = loss.sum()
            mask = loss_weight != 0
            w = loss_weight.sum() / mask.sum()  # w equals to 1/global_denominator
            loss = loss * w
            return loss, (None, {})

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None = None,
        skip_all_reduce: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, dict[str, Any]]]:
        from xtuner.v1.model.utils.misc import ModelForwardExtraLogInfo

        assert self.loss_kwargs is not None, "loss_kwargs must be set before calling forward"
        if head_bias is not None:
            raise NotImplementedError("Loss does not support head_bias yet.")

        if self.loss_cfg.mode == "eager":
            loss, (logits, extra_info) = self.eager_mode(hidden_states, head_weight, head_bias, self.loss_kwargs)
        else:
            loss, (logits, extra_info) = self.chunk_mode(hidden_states, head_weight, head_bias, self.loss_kwargs)

        # TODO: yanhuida, should be removed
        if not isinstance(extra_info, ModelForwardExtraLogInfo):
            extra_info = ModelForwardExtraLogInfo(extra_info)

        extra_info["local_base_loss"] = loss.detach().clone()

        if not skip_all_reduce:
            if dist.is_initialized():
                loss = all_reduce(loss, op=dist.ReduceOp.SUM, group=dist.group.WORLD)

        return loss, (logits, extra_info)


# Deprecated: Use LMHeadLossContext instead. Will be removed in version 1.1.0
CELossContext = LMHeadLossContext
