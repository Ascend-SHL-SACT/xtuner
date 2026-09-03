# Copyright (c) OpenMMLab. All rights reserved.
"""NPU liger-ascend FLCE patches and loss memory tooling.

Aggregates the liger-ascend fused-linear-cross-entropy NPU corrections
together with the forward/backward memory optimization (always enabled).
``ce_loss.py`` imports these helpers and wires them in; no liger/ascend
specifics live there.
"""

import importlib
import os
import sys

import torch
import torch.nn.functional as F


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
            setattr(
                flce_mod,
                "LigerFusedLinearCrossEntropyFunction",
                getattr(ops_mod, "LigerFusedLinearCrossEntropyFunction"),
            )
    _patch_liger_ascend_fwd()


# Patch configuration, read once at import. LOSS_CHUNK_SIZE is the same knob
# that feeds CELossConfig.chunk_size in the config layer.
# XTUNER_LIGER_CHUNK_FWD_LIMIT is in MiB.
_CHUNK_FWD_ENABLED = os.environ.get("XTUNER_LIGER_CHUNK_FWD", "1") == "1"
_CHUNK_FWD_LIMIT_BYTES = int(os.environ.get("XTUNER_LIGER_CHUNK_FWD_LIMIT", "2048")) * 1024**2
_CE_CHUNK_ROWS = int(os.environ.get("LOSS_CHUNK_SIZE", "1024"))


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
    try:
        flce = importlib.import_module("liger_kernel.ops.backends._ascend.ops.fused_linear_cross_entropy")
    except ImportError:
        return
    if getattr(flce, "_xtuner_fwd_patched", False):
        return
    _orig_fwd = flce.fused_linear_cross_entropy_forward

    def _correct_fwd(*args, **kwargs):
        _input = kwargs.get("_input")
        weight = kwargs.get("weight")
        target = kwargs.get("target")
        if _input is None and args:
            _input = args[0]
        if weight is None and len(args) > 1:
            weight = args[1]
        if target is None and len(args) > 2:
            target = args[2]
        ignore_index = kwargs.get("ignore_index", -100)
        reduction = kwargs.get("reduction", "mean")
        bias = kwargs.get("bias")
        # Chunked lm-head+CE: when the full ``[BT, V]`` logits would be large,
        # compute the loss in BT-chunks so only a ``[chunk, V]`` slice is live
        # at once. The upstream Ascend FLCE forward always materializes the
        # full ``[BT, V]`` (``_input @ weight.t()``); at large local seq (e.g.
        # 128K/CP16 -> ``[8192, 154880]`` ~= 2.5 GB) that single tensor OOMs
        # with <2.2 GB free. Return empty ``logits_for_backward`` so backward
        # recomputes per chunk (``has_saved_logits=False`` ->
        # ``chunk_size=min(BT, 4096)``). CE is row-independent, so the chunked
        # ``loss_1d`` is identical to the full path; the no-weight backward
        # derives ``lse = loss_1d + x[y]`` from the (torch-correct) loss_1d
        # and the recomputed logits, so grads match. Small BT keeps the fast
        # saved-logits path (no regression at 64K/local4096). Gate: full logits
        # bytes > limit (default 2048 MiB -> triggers at BT=8192, not BT=4096).
        # Trade vs upstream's 4 GiB save limit: in the 2-4 GiB band (BT
        # ~6965..13930, e.g. 128K/CP16 at BT=8192) HEAD's backward reused the
        # saved logits while this path recomputes them per chunk (one extra
        # lm-head GEMM, order tens of ms/step). At 256K (BT=16384, ~4.7 GiB)
        # upstream would not have saved either, so there is no regression.
        # The feature gate restricts the fast path to exactly the plain
        # reduction="none" CE it implements (no ce_weight / smoothing /
        # softcap / token-scaling extras); anything else falls through to
        # upstream so its loss semantics stay upstream-correct.
        _full_bytes = (
            _input.shape[0] * weight.shape[0] * _input.element_size()
            if _input is not None and weight is not None
            else 0
        )
        if (
            _CHUNK_FWD_ENABLED
            and len(args) <= 3  # any positional beyond _input/weight/target -> unknown config
            and _input is not None
            and weight is not None
            and _input.requires_grad
            and bias is None
            and kwargs.get("ce_weight") is None
            and kwargs.get("softcap") is None
            and not kwargs.get("label_smoothing", 0.0)
            and not kwargs.get("lse_square_scale", 0.0)
            and not kwargs.get("use_token_scaling", False)
            and not kwargs.get("return_z_loss", False)
            and not kwargs.get("return_token_accuracy", False)
            and not kwargs.get("return_predicted_tokens", False)
            and reduction == "none"
            and _full_bytes > _CHUNK_FWD_LIMIT_BYTES
        ):
            _bt = _input.shape[0]
            _chunk = _CE_CHUNK_ROWS
            correct = torch.empty(_bt, dtype=torch.float32, device=_input.device)
            for _s in range(0, _bt, _chunk):
                _e = min(_s + _chunk, _bt)
                _lc = _input[_s:_e] @ weight.t()
                correct[_s:_e] = F.cross_entropy(
                    _lc.float(),
                    target[_s:_e],
                    reduction="none",
                    ignore_index=ignore_index,
                )
            loss = correct if reduction == "none" else correct.sum()
            _ce_stats = flce._make_ce_stats_buffer(
                target,
                ignore_index,
                None,
                reduction,
                target_mask=(target != ignore_index),
            )
            # (loss, None, None, None, loss_1d=correct, ce_stats, None,
            #  plain_fast_path, logits_for_backward) -- the empty
            # logits_for_backward makes backward recompute logits per chunk
            # from _input/weight.
            return (loss, None, None, None, correct, _ce_stats, None, True, None)
        out = _orig_fwd(*args, **kwargs)
        # out = (loss, None, None, None, loss_1d, ce_stats, None,
        #        plain_fast_path, logits_for_backward)
        if not isinstance(out, tuple) or len(out) < 9:
            return out
        loss_1d = out[4]
        logits = out[8]
        if logits is None and _input is not None and weight is not None:
            logits = _input @ weight.t()
        if logits is None or target is None or loss_1d is None:
            return out
        # torch CE on the (correct) logits -- default stream, training-correct.
        # Chunk over BT so only a [chunk,V] fp32 logits + the CE log_softmax
        # work buffer are live at once. The full-BT path materializes two full
        # fp32 [BT,V] copies (logits.float() + CE's internal log_softmax);
        # chunking cuts that to [chunk,V]. CE is row-independent so the chunked
        # loss_1d (and the backward grad that derives lse from it) is identical
        # to the full path.
        _bt = logits.shape[0]
        if _bt > _CE_CHUNK_ROWS:
            correct = torch.empty(_bt, dtype=torch.float32, device=logits.device)
            _chunk = _CE_CHUNK_ROWS
            for _s in range(0, _bt, _chunk):
                _e = min(_s + _chunk, _bt)
                correct[_s:_e] = F.cross_entropy(
                    logits[_s:_e].float(),
                    target[_s:_e],
                    reduction="none",
                    ignore_index=ignore_index,
                )
        else:
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

    setattr(flce, "fused_linear_cross_entropy_forward", _correct_fwd)
    setattr(flce, "_xtuner_fwd_patched", True)

    # Stage 2: memory-optimized backward. On Ascend, addmm_ is NOT in-place
    # (it materialises a [V,H] result temp just like matmul+copy_), but
    # torch.matmul(a, b, out=grad_weight) IS a fused GEMM that writes the
    # result directly into the pre-allocated grad_weight with no temp. So for
    # the single-chunk case (has_saved_logits => chunk_size=BT => chunk_id==0)
    # we replace the throwaway grad_weight_ temp + copy_ with matmul(out=...),
    # eliminating the [V,H] grad_weight_ temp (~1.9GB at V=154880). Multi-chunk
    # keeps the original add_ path (its temp is unavoidable without in-place
    # accumulate, which Ascend lacks). Identical grads; purely less transient.
    if getattr(flce, "_xtuner_bwd_patched", False):
        return

    def _memopt_bwd(ctx, grad_output):
        (_input, weight, target, loss_1d, ce_stats, saved_logits) = ctx.saved_tensors
        bias = ctx.bias
        ce_weight = ctx.ce_weight
        scaling_factors_full = ctx.scaling_factors_full

        device = _input.device
        BT = _input.shape[0]
        V = weight.shape[0]

        forward_block_size = flce.get_optimal_block_size(V, has_gradients=False)
        backward_block_size = flce.get_optimal_block_size(V, has_gradients=True)
        if ctx.plain_fast_path and 32768 < V <= 131072:
            backward_block_size = 4096

        has_saved_logits = saved_logits.numel() != 0
        chunk_size = BT if has_saved_logits else min(BT, 4096)
        num_chunks = flce.triton.cdiv(BT, chunk_size)

        num_cores = flce.get_npu_core_count()
        use_no_weight_backward = (
            V > 4096
            and ctx.plain_fast_path
            and ce_weight is None
            and ctx.softcap is None
            and ctx.label_smoothing == 0.0
            and ctx.lse_square_scale == 0.0
        )
        ls_eps = float(ctx.label_smoothing) / float(V) if ctx.label_smoothing else 0.0

        # bf16 grad_weight in the multi-chunk (recompute) path too: at large
        # local seq an fp32 grad_weight is a 3.8 GB persistent allocation
        # ([154880,6144] x fp32) that leaves <0.5 GB for HCCL's all-reduce
        # workspace at grad-norm/step_optimizer -> EL0004/207001. Trade: this
        # ignores the caller's accum_dtype (ctx.accum_dtype is not consulted),
        # and unlike the single-chunk path -- one fp32-internal GEMM rounded
        # to bf16 once, i.e. the fp32 result rounded -- each per-chunk fp32
        # partial here is rounded to bf16 before the bf16 add_ summation,
        # adding up to ~1% relative per-element error on grad_weight vs fp32
        # accumulation. grad_input is unaffected (chunks write disjoint rows,
        # no cross-chunk accumulation). Accepted for the 256K memory floor.
        grad_accum_dtype = None
        grad_input = torch.empty_like(_input)
        grad_weight = torch.empty_like(weight, dtype=grad_accum_dtype or weight.dtype, device=device)
        grad_bias = (
            torch.empty_like(bias, dtype=grad_accum_dtype or bias.dtype, device=device) if bias is not None else None
        )

        has_grad_output_vector = ctx.reduction == "none"
        if has_grad_output_vector and grad_output.stride(-1) != 1:
            grad_output = grad_output.contiguous()
        grad_output_stride = grad_output.stride(-1) if has_grad_output_vector else 0

        for chunk_id in range(num_chunks):
            start_idx = chunk_id * chunk_size
            end_idx = min((chunk_id + 1) * chunk_size, BT)
            input_chunk = _input[start_idx:end_idx]
            target_chunk = target[start_idx:end_idx]
            n_rows = end_idx - start_idx

            if has_saved_logits:
                logits_chunk = saved_logits[start_idx:end_idx]
            else:
                logits_chunk = input_chunk @ weight.t()
                if bias is not None:
                    logits_chunk = logits_chunk + bias

            if not logits_chunk.is_contiguous():
                logits_chunk = logits_chunk.contiguous()
            if not target_chunk.is_contiguous():
                target_chunk = target_chunk.contiguous()

            grad_logits_chunk = torch.empty_like(logits_chunk)

            if use_no_weight_backward:
                loss_1d_slice = loss_1d[start_idx:end_idx]
                flce.liger_cross_entropy_backward_kernel_no_weight[(min(n_rows, num_cores),)](
                    X_ptr=logits_chunk,
                    X_stride=logits_chunk.stride(-2),
                    Y_ptr=target_chunk,
                    lse_ptr=loss_1d_slice,
                    grad_output_ptr=grad_output,
                    grad_output_stride=grad_output_stride,
                    dX_ptr=grad_logits_chunk,
                    dX_stride=grad_logits_chunk.stride(-2),
                    n_cols=V,
                    n_rows=n_rows,
                    ce_stats_ptr=ce_stats,
                    ignore_index=ctx.ignore_index,
                    reduction=ctx.reduction,
                    BLOCK_SIZE=backward_block_size,
                    HAS_LSE=False,
                )
            else:
                lse_chunk = torch.empty(n_rows, dtype=torch.float32, device=device)
                loss_tmp = torch.empty(n_rows, dtype=torch.float32, device=device)
                flce.liger_cross_entropy_forward_kernel[(min(n_rows, num_cores),)](
                    X_ptr=logits_chunk,
                    X_stride=logits_chunk.stride(-2),
                    Y_ptr=target_chunk,
                    weight_ptr=ce_weight,
                    loss_ptr=loss_tmp,
                    z_loss_ptr=loss_tmp,
                    lse_ptr=lse_chunk,
                    token_accuracy_ptr=loss_tmp,
                    token_accuracy_stride=0,
                    predicted_tokens_ptr=target_chunk,
                    predicted_tokens_stride=0,
                    n_cols=V,
                    n_rows=n_rows,
                    ce_stats_ptr=ce_stats,
                    ignore_index=ctx.ignore_index,
                    ls_eps=ls_eps,
                    lse_square_scale=ctx.lse_square_scale,
                    label_smoothing=ctx.label_smoothing,
                    reduction=ctx.reduction,
                    softcap=ctx.softcap,
                    RETURN_Z_LOSS=False,
                    RETURN_LSE=True,
                    RETURN_TOKEN_ACCURACY=False,
                    RETURN_PREDICTED_TOKENS=False,
                    HAS_WEIGHT=True if ce_weight is not None else False,
                    HAS_SOFTCAPPING=True if ctx.softcap is not None else False,
                    BLOCK_SIZE=forward_block_size,
                )
                flce.liger_cross_entropy_backward_kernel[(min(n_rows, num_cores),)](
                    X_ptr=logits_chunk,
                    X_stride=logits_chunk.stride(-2),
                    Y_ptr=target_chunk,
                    weight_ptr=ce_weight,
                    lse_ptr=lse_chunk,
                    grad_output_ptr=grad_output,
                    grad_output_stride=grad_output_stride,
                    dX_ptr=grad_logits_chunk,
                    dX_stride=grad_logits_chunk.stride(-2),
                    n_cols=V,
                    n_rows=n_rows,
                    ce_stats_ptr=ce_stats,
                    ignore_index=ctx.ignore_index,
                    lse_square_scale=ctx.lse_square_scale,
                    label_smoothing=ctx.label_smoothing,
                    reduction=ctx.reduction,
                    softcap=ctx.softcap,
                    BLOCK_SIZE=backward_block_size,
                    HAS_WEIGHT=True if ce_weight is not None else False,
                    HAS_SOFTCAPPING=True if ctx.softcap is not None else False,
                )

            if ctx.use_token_scaling and scaling_factors_full is not None:
                grad_logits_chunk = grad_logits_chunk * scaling_factors_full[start_idx:end_idx].unsqueeze(-1)

            grad_input[start_idx:end_idx] = grad_logits_chunk @ weight
            # matmul(out=grad_weight) writes the result directly into the
            # pre-allocated grad_weight with no [V,H] grad_weight_ temp (~1.9GB
            # at V=154880), but Ascend's matmul(out=) does NOT auto-cast: the
            # out dtype must equal the computed dtype. That only holds for the
            # single-chunk case (num_chunks == 1 -> one matmul, no accumulation
            # needed). Multi-chunk CANNOT use matmul(out=): it overwrites
            # (losing earlier partials) instead of accumulating, so it must use
            # the grad_weight_ temp + copy_/add_ path. grad_accum_dtype is now
            # always None (bf16 grad_weight; see above), so copy_/add_ are
            # bf16->bf16 (no cast, no crash). grad_weight is empty_like
            # (uninitialized), so chunk 0 must overwrite (copy_), never add_
            # (which would add into garbage).
            if num_chunks == 1:
                torch.matmul(grad_logits_chunk.t(), input_chunk, out=grad_weight)
            else:
                grad_weight_ = grad_logits_chunk.t() @ input_chunk
                if chunk_id == 0:
                    grad_weight.copy_(grad_weight_)
                else:
                    grad_weight.add_(grad_weight_)
            if grad_bias is not None:
                grad_bias_ = grad_logits_chunk.sum(dim=0)
                if chunk_id == 0:
                    grad_bias.copy_(grad_bias_)
                else:
                    grad_bias.add_(grad_bias_)

        if grad_accum_dtype is None:
            grad_weight = grad_weight.to(weight.dtype)
            grad_bias = grad_bias.to(bias.dtype) if grad_bias is not None else None

        return (
            grad_input,
            grad_weight,
            None,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    setattr(flce, "fused_linear_cross_entropy_backward", _memopt_bwd)
    setattr(flce, "_xtuner_bwd_patched", True)
