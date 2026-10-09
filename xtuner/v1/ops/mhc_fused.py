# Copyright © 2026 Huawei Technologies Co., Ltd.
"""Fused mHC kernels for the GLM-5.3 hyper-connections (triton, NPU).

The fused mHC changeset consolidated into one module; the decoder layer
(:mod:`xtuner.v1.module.decoder_layer.mhc`) only gates and wires the entry
points. Components:

- **Norm + mix projection** (:func:`mhc_norm_linear`): replaces the eager
  ``flatten(2).float()`` -> ``rms_norm`` (fp32) -> ``F.linear(flat, fn)``
  chain in ``hc_pre``. The eager chain materialises a ``[T, H*D]`` fp32 copy
  (268 MB at the production shape) re-read through three more full-row kernels;
  the fused path keeps the row in bf16 storage and never materialises the
  normalised activations, using two algebraic identities to stay MMA-shaped:
  ``mixes = (x @ fn.T) * rstd`` (the unweighted RMSNorm rescale is a per-row
  scalar, so it factors out of the projection and the GEMM runs directly on the
  raw bf16 inputs -- bf16xbf16 products are exact in the fp32 MMA accumulator)
  and ``d_fn = (dmixes * rstd).T @ x``. Kernels: a per-row sum-of-squares pass,
  a row-tiled MMA forward, a tiled ``tl.dot`` ``d_fn`` reduction and a two-pass
  row-tiled ``d_x`` kernel (the rank-MIX ``dmixes @ fn`` product is recomputed
  chunk-wise, never stored).
- **Split-Sinkhorn** (:func:`hc_split_sinkhorn_fused`): ported from MindSpeed-MM
  ``mindspeed_mm/fsdp/ops/hc/sinkhorn.py`` (upstream provenance: Huawei /
  NVIDIA, Apache-2.0); two forward + two backward triton kernels replacing the
  eager split-sigmoid-softmax + Python Sinkhorn loop, which issues ~5 tiny
  elementwise kernels per Sinkhorn iteration per call site (~692K operator
  instances, ~13 s/step, across the 45 mHC sites of the 30B at the 128K
  training shape). MindSpeed-MM runs this fused op unconditionally on NPU
  whenever the kernel imports, and the pre-merge GLM-5.3 implementation did the
  same (it is the ``run81`` 128K anchor numerics). Numerics vs the eager chain:
  the fused comb path applies one extra eps-stabilised row normalisation
  instead of ``softmax(-1) + eps`` (MindSpeed's own fused-op exception), and
  the sigmoid/exp elementwise results agree to fp32 ULP. The ``hc_scale`` /
  ``hc_base`` gradients accumulate via on-device atomics across programs, so
  they carry run-to-run ±ULP nondeterminism (``XTUNER_DETERMINISTIC`` does not
  remove it). The eager reference stays available for non-NPU / ``hc_mult != 4``
  calls.
- **Combine** (:func:`hc_combine`): the decoder-layer stream update
  ``new[t, n, d] = post[t, n] * out[t, d] + sum_k comb[t, k, n] * res[t, k, d]``
  -- eager chain = 2 Cast + Transpose + broadcast Mul + batched MatMul (K=HC) +
  Add, materialising two ``[T, HC, D]`` intermediates. The kernel replicates
  the three bf16 rounding points (mul result, matmul result, final add) and
  the K=HC sequential accumulation order; token-tiled vectorised loads make it
  1.43x the eager fwd+bwd pair at the training shape (910C bench).
- **Collapse** (:func:`hc_collapse`): the ``hc_pre`` readout tail
  ``collapsed[t, d] = sum_k pre[t, k] * streams[t, k, d]`` -- eager chain =
  broadcast Mul + Sum(dim=2); fp32 accumulation throughout, rounding only the
  final result (1.4x the eager chain, 910C device bench).
- **Matmul-form hc_post** (:func:`_hc_post_eager_matmul`): the eager
  ``hc_post`` fallback in the pre-merge implementation's batched-matmul form.

Residual differences vs the eager chains are limited to reduction orders
(rms / mix-projection / along-D) -- the approved fused-op exceptions the 128K
``run81`` anchor (llm_loss 12.14300537) was trained with -- plus one rounding
placement: the norm-linear path never materialises the bf16-rounded normalized
activations the eager chain feeds into the projection (it scales the fp32 MMA
accumulator by ``rstd`` instead), so its ``mixes``/``d_x``/``d_w`` deviate by up
to ~2% relative on adversarial magnitudes, still under the "rms" exception.
``post`` / ``comb`` /
``pre`` only need fold-consistent ``[B, S, ...]`` strides with a unit innermost
axis -- both the fused sinkhorn outputs (``comb`` is a padded ``[.., HC, 8]``
slice view) and the eager split views qualify; ``out`` / ``res`` / ``streams``
must be contiguous.

Environment switches (eager fallbacks always available; the decoder layer wires
them):

- ``XTUNER_GLM53_MHC_FUSED`` (default on): :func:`mhc_norm_linear`.
- ``XTUNER_GLM53_MHC_SINKHORN_EAGER=1``: escapes the fused sinkhorn (the op
  itself is ungated, mirroring MindSpeed-MM).
- ``XTUNER_GLM53_HC_COMBINE_FUSED`` / ``XTUNER_GLM53_HC_COLLAPSE_FUSED``
  (default off): :func:`hc_combine` / :func:`hc_collapse`.
- ``XTUNER_GLM53_HC_POST_EAGER_BROADCAST=1``: escapes the default matmul-form
  ``hc_post`` back to HEAD's broadcast form.

Persistent-grid contract: the norm-linear and combine/collapse kernels run a
persistent grid of at most ``_MAX_PROGS`` programs -- the aiv physical block
count, so the launcher's auto-blockify clamp (``TRITON_ALL_BLOCKS_PARALLEL=1``)
is a no-op (clamped blockify replay miscompiles at training shapes). The
combine/collapse group resolves the cap on first use with the NaN-sentinel
probe ``_probe_max_progs`` and falls back to ``_MAX_PROGS_FALLBACK`` should a
binary flip to aicore mode; the norm-linear launches and the sinkhorn kernels
carry no runtime probe (offline probe record / anchor-validated MindSpeed-port
grids -- see the constants comment and ``SinkhornFunction.forward``).
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import once_differentiable
from torch.distributed.tensor import DTensor

from xtuner.v1.utils.compile import maybe_compile


try:
    import torch_npu  # noqa: F401

    _NPU_AVAILABLE = torch.npu.is_available()  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - CPU-only environments
    _NPU_AVAILABLE = False

__all__ = [
    "HC_COLLAPSE_FUSED",
    "HC_COMBINE_FUSED",
    "MHC_FUSED_NORM_LINEAR",
    "hc_collapse",
    "hc_combine",
    "hc_split_sinkhorn_fused",
    "mhc_norm_linear",
]

# Env gates read once at import; set XTUNER_GLM53_MHC_FUSED=0 to restore the
# eager rms_norm + F.linear chain, XTUNER_GLM53_HC_COMBINE_FUSED=1 /
# XTUNER_GLM53_HC_COLLAPSE_FUSED=1 to enable the two stream kernels.
MHC_FUSED_NORM_LINEAR = os.environ.get("XTUNER_GLM53_MHC_FUSED", "1") not in ("0", "off", "false")
HC_COMBINE_FUSED = os.environ.get("XTUNER_GLM53_HC_COMBINE_FUSED", "0") == "1"
HC_COLLAPSE_FUSED = os.environ.get("XTUNER_GLM53_HC_COLLAPSE_FUSED", "0") == "1"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_RSTD_D_BLOCK = 1024  # feature chunk of the per-row sum-of-squares pass
_T_TILE = 64  # token rows per MMA tile
_K_TILE = 128  # feature (reduction) chunk per MMA tile
_DX_K2_TILE = 512  # feature chunk of the d_x store pass (see _mhc_dx_kernel)
_DFN_D_TILE = 512  # feature tile of the d_fn reduction
# Persistent-grid cap for the norm-linear and combine/collapse kernels (a
# shared module constant; the combine/collapse group additionally resolves it
# on first use via the NaN-sentinel probe below). With
# TRITON_ALL_BLOCKS_PARALLEL=1 the Ascend triton launcher clamps blockNum to the
# physical core count and replays the overflow blocks inside each core; that replay
# miscompiled these kernels on the 64K shape (step-1 forward NaN, bisected to
# mhc_norm_linear: healthy with the fused path off, NaN with only it fused) at
# grids past 24 in the original single-program-per-tile form. The in-kernel
# strided loop added then keeps per-row/per-tile math unchanged, and the
# miscompile has not reproduced under it: all four norm-linear kernels re-probed
# at grid=48 with NaN-prefilled outputs at T=8192 and T=65536 are NaN-free and
# bitwise-equal to grid=24 (0930 probe; combine/collapse and conv-bwd have run
# 48 since).
# The cap only widens the rstd kernel's parallelism (1.94x); fwd/dx are
# tl.dot-bound and unchanged at 48.
#
# Unlike the combine/collapse kernels (probed below) and causal_conv1d, the
# norm-linear launches carry NO runtime NaN-sentinel probe: the cap rests on
# the offline 0930 probe record above plus TRITON_ALWAYS_COMPILE=1 in the
# launch script. If a future CANN/torch_npu upgrade flips one of these
# binaries to aicore mode, dropped blocks would leave uninitialized rstd rows
# (silent NaN mixes) with no automatic degrade -- re-run the offline grid=48
# probe after any toolchain upgrade and keep this comment current.
_MAX_PROGS = 48
_PROD_HC = 4  # MindSpeed wrapper contract: the fused mHC kernels specialise hc_mult=4
# Upper bound on the mix-projection output width MIX = (2 + hc_mult) * hc_mult: the
# forward/backward kernels allocate a [BLOCK_T=64, MIX_PAD] fp32 accumulator, and past
# MIX ~= 5120 BiShengIR fails to compile with a UB overflow (hard crash at MIX=8192,
# live-verified) instead of the graceful eager fallback the gate promises. 128 covers
# hc_mult <= 10; production is 24 (hc_mult=4).
_MAX_MIX = 128
_BLOCK_D = 1024  # feature tile of the collapse per-token row loops ([HC, BLOCK_D] <= 4096)
_MAX_PROGS_FALLBACK = 24  # long-running production cap (aicore physical block count)
_TRITON_MAX_PROGS: list[int | None] = [None]  # probe-resolved cap; None -> _MAX_PROGS
_LOG_STATE: dict[str, bool] = {"combine": False, "collapse": False}  # one-shot engagement prints
_COMBINE_BLOCK_T = 8  # token rows per combine tile ([BLOCK_T, BLOCK_D] fp32 tiles)
_COMBINE_BLOCK_D = 256  # feature tile of the combine kernels (14 live tiles ~= 112 KB UB)

# Cached scratch buffers of the fused sinkhorn op: the four column-padded
# ``[rows, hc_mult, block_align]`` temps (pad columns pre-filled once with
# their fill value) and the fully-rewritten ``comb_tmp`` intermediates. At
# 128K the op runs ~950 times per training step, and each per-call
# ``F.pad``/``torch.empty`` pair costs ~3 extra host dispatches (~1.5s/step of
# host-bound free). Only pure intermediates are cached — ``comb_flat_padded``
# stays a fresh ``torch.empty`` because the returned ``comb`` is a view of it
# and later calls would clobber tensors the autograd graph still references.
_PADDED_COMB_CACHE: dict[tuple[str, int, str, torch.dtype, float], torch.Tensor] = {}
# Single-compute-stream assumption: unlike _PADDED_COMB_CACHE, the scratch key
# has no slot dimension -- one call's forward and backward share the same
# comb_tmp entry. Safe today because every consumer fully rewrites the buffer
# before any read, both run in issue order on the same compute stream, and the
# autograd graph never references the scratch (saved tensors are only
# mixes/hc_scale/hc_base). If a multi-stream or concurrent-backward orchestration
# (e.g. exotic checkpointing) ever breaks that, add a slot dimension here.
_COMB_SCRATCH_CACHE: dict[tuple[tuple[int, ...], str, torch.dtype], torch.Tensor] = {}


# ------------------------------------------------------------------------------
# -- Norm + mix projection (fused unweighted RMSNorm + linear, rstd factoring)
# ------------------------------------------------------------------------------
@triton.jit
def _mhc_rstd_kernel(
    x_ptr,
    rstd_ptr,
    T,
    N: tl.constexpr,
    EPS,
    BLOCK_D: tl.constexpr,
):
    # Anchor form (bit-exact with the pre-19:33 kernel): one token row per
    # strided iteration, 1D [BLOCK_D] sum-of-squares with the unrolled
    # static_range chunk loop and num_warps=4. The 2D-row-tile variant changed
    # the tl.sum reduction tree and moved the 64K loss off the 12.36008358
    # anchor; the persistent grid is kept (blockify-replay guard).
    pid = tl.program_id(0)
    nprogs = tl.num_programs(0)
    d_off = tl.arange(0, BLOCK_D)
    for row in range(pid, T, nprogs):
        base = row.to(tl.int64) * N  # type: ignore[attr-defined] # triton: row is a device-side loop index
        ssq = tl.zeros([BLOCK_D], dtype=tl.float32)
        for d0 in tl.static_range(0, N, BLOCK_D):
            xv = tl.load(x_ptr + base + d0 + d_off).to(tl.float32)
            ssq += xv * xv
        inv = 1.0 / tl.sqrt(tl.sum(ssq) / N + EPS)
        tl.store(rstd_ptr + row, inv)


@triton.jit
def _mhc_fwd_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    rstd_ptr,
    T,
    N: tl.constexpr,
    MIX: tl.constexpr,
    MIX_PAD: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    nprogs = tl.num_programs(0)
    j_off = tl.arange(0, MIX_PAD)
    j_mask = j_off < MIX
    for blk in range(pid, tl.cdiv(T, BLOCK_T), nprogs):
        t_off = blk * BLOCK_T + tl.arange(0, BLOCK_T)
        t_mask = t_off < T
        t64 = t_off[:, None].to(tl.int64)
        acc = tl.zeros([BLOCK_T, MIX_PAD], dtype=tl.float32)
        for k0 in range(0, N, BLOCK_K):
            k_off = k0 + tl.arange(0, BLOCK_K)
            xv = tl.load(x_ptr + t64 * N + k_off[None, :], mask=t_mask[:, None], other=0.0)
            wv = tl.load(w_ptr + j_off[:, None].to(tl.int64) * N + k_off[None, :], mask=j_mask[:, None], other=0.0)
            acc += tl.dot(xv, tl.trans(wv))
        inv = tl.load(rstd_ptr + t_off, mask=t_mask, other=0.0)
        tl.store(
            y_ptr + t64 * MIX + j_off[None, :],
            acc * inv[:, None],
            mask=t_mask[:, None] & j_mask[None, :],
        )


@triton.jit
def _mhc_dfn_kernel(
    x_ptr,
    dm_ptr,
    rstd_ptr,
    dfn_ptr,
    T,
    N: tl.constexpr,
    MIX: tl.constexpr,
    MIX_PAD: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    nprogs = tl.num_programs(0)
    j_off = tl.arange(0, MIX_PAD)
    j_mask = j_off < MIX
    for blk in range(pid, N // BLOCK_D, nprogs):
        d_off = blk * BLOCK_D + tl.arange(0, BLOCK_D)
        acc = tl.zeros([MIX_PAD, BLOCK_D], dtype=tl.float32)
        for t0 in range(0, T, BLOCK_T):
            t_off = t0 + tl.arange(0, BLOCK_T)
            t_mask = t_off < T
            t64 = t_off[:, None].to(tl.int64)
            dm = tl.load(dm_ptr + t64 * MIX + j_off[None, :], mask=t_mask[:, None] & j_mask[None, :], other=0.0)
            inv = tl.load(rstd_ptr + t_off, mask=t_mask, other=0.0)
            dm_scaled = (dm * inv[:, None]).to(dfn_ptr.dtype.element_ty)
            xv = tl.load(x_ptr + t64 * N + d_off[None, :], mask=t_mask[:, None], other=0.0)
            acc += tl.dot(tl.trans(dm_scaled), xv)
        tl.store(
            dfn_ptr + j_off[:, None].to(tl.int64) * N + d_off[None, :],
            acc.to(dfn_ptr.dtype.element_ty),
            mask=j_mask[:, None],
        )


@triton.jit
def _mhc_dx_kernel(
    x_ptr,
    w_ptr,
    dm_ptr,
    rstd_ptr,
    dx_ptr,
    T,
    N: tl.constexpr,
    MIX: tl.constexpr,
    MIX_PAD: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_K2: tl.constexpr,
):
    # BLOCK_K (pass 1) must match the anchor reduction order: the per-chunk
    # ``tl.sum(g * xv, axis=1)`` tree over BLOCK_K columns sets the bit-exact
    # summation order of ``s`` (a 512-wide pass 1 moves the 64K loss off the
    # 12.36008358 anchor; verified by NPU microbench). Pass 2 only recomputes
    # ``g`` and stores elementwise, so its width is bit-free -- 512 there is
    # ~27% faster than 128 at the production shape.
    pid = tl.program_id(0)
    nprogs = tl.num_programs(0)
    j_off = tl.arange(0, MIX_PAD)
    j_mask = j_off < MIX
    for blk in range(pid, tl.cdiv(T, BLOCK_T), nprogs):
        t_off = blk * BLOCK_T + tl.arange(0, BLOCK_T)
        t_mask = t_off < T
        t64 = t_off[:, None].to(tl.int64)
        dm = tl.load(dm_ptr + t64 * MIX + j_off[None, :], mask=t_mask[:, None] & j_mask[None, :], other=0.0)
        dm_b = dm.to(w_ptr.dtype.element_ty)
        inv = tl.load(rstd_ptr + t_off, mask=t_mask, other=0.0)
        # pass 1: s = <d_normed, x> over the full row (g recomputed chunk-wise)
        s = tl.zeros([BLOCK_T], dtype=tl.float32)
        for k0 in range(0, N, BLOCK_K):
            k_off = k0 + tl.arange(0, BLOCK_K)
            xv = tl.load(x_ptr + t64 * N + k_off[None, :], mask=t_mask[:, None], other=0.0)
            wv = tl.load(w_ptr + j_off[:, None].to(tl.int64) * N + k_off[None, :], mask=j_mask[:, None], other=0.0)
            g = tl.dot(dm_b, wv)
            s += tl.sum(g * xv.to(tl.float32), axis=1)
        scale = s * inv * inv * inv / N
        # pass 2: dx = g * inv - x * scale
        for k0 in range(0, N, BLOCK_K2):
            k_off = k0 + tl.arange(0, BLOCK_K2)
            xv = tl.load(x_ptr + t64 * N + k_off[None, :], mask=t_mask[:, None], other=0.0).to(tl.float32)
            wv = tl.load(w_ptr + j_off[:, None].to(tl.int64) * N + k_off[None, :], mask=j_mask[:, None], other=0.0)
            g = tl.dot(dm_b, wv)
            tl.store(
                dx_ptr + t64 * N + k_off[None, :],
                (g * inv[:, None] - xv * scale[:, None]).to(dx_ptr.dtype.element_ty),
                mask=t_mask[:, None],
            )


def _mhc_servable(x2d: Tensor, w: Tensor) -> bool:
    """Whether the fused kernels can serve this call (shape/dtype contract).

    Args:
        x2d (Tensor): Flattened stream activations ``[T, H*D]``.
        w (Tensor): Mix projection weight ``[MIX, H*D]``.

    Returns:
        bool: ``True`` when the fused path can serve the request.
    """
    if not _NPU_AVAILABLE or x2d.device.type != "npu":
        return False
    if x2d.dim() != 2 or not x2d.is_contiguous() or not w.is_contiguous():
        return False
    n = x2d.shape[1]
    return bool(
        x2d.dtype in (torch.bfloat16, torch.float16)
        and w.dtype == x2d.dtype
        and w.shape[1] == n
        and w.shape[0] <= _MAX_MIX
        and x2d.shape[0] > 0
        and n % _RSTD_D_BLOCK == 0
        and n % _K_TILE == 0
        and n % _DX_K2_TILE == 0
        and n % _DFN_D_TILE == 0
    )


class _MhcNormLinearFn(torch.autograd.Function):
    """Fused unweighted-RMSNorm + linear producing the mHC mix logits."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx, x2d: Tensor, w: Tensor, eps: float
    ) -> Tensor:
        tokens, n = x2d.shape
        mix = w.shape[0]
        mix_pad = max(triton.next_power_of_2(mix), 16)
        rstd = torch.empty((tokens,), dtype=torch.float32, device=x2d.device)
        _mhc_rstd_kernel[(min(tokens, _MAX_PROGS),)](
            x2d, rstd, tokens, N=n, EPS=eps, BLOCK_D=_RSTD_D_BLOCK, num_warps=4
        )
        y = torch.empty((tokens, mix), dtype=torch.float32, device=x2d.device)
        _mhc_fwd_kernel[(min(triton.cdiv(tokens, _T_TILE), _MAX_PROGS),)](
            x2d, w, y, rstd, tokens, N=n, MIX=mix, MIX_PAD=mix_pad, BLOCK_T=_T_TILE, BLOCK_K=_K_TILE, num_warps=4
        )
        ctx.save_for_backward(x2d, w, rstd)  # type: ignore[attr-defined]
        return y

    @staticmethod
    @once_differentiable
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx, dmixes: Tensor | None
    ) -> tuple[Tensor | None, Tensor | None, None]:
        if dmixes is None:
            return None, None, None
        x2d, w, rstd = ctx.saved_tensors  # type: ignore[attr-defined]
        tokens, n = x2d.shape
        mix = w.shape[0]
        mix_pad = max(triton.next_power_of_2(mix), 16)
        dm = dmixes.reshape(tokens, mix).contiguous()
        d_w = None
        if ctx.needs_input_grad[1]:  # type: ignore[attr-defined]
            d_w = torch.empty_like(w)
            _mhc_dfn_kernel[(min(n // _DFN_D_TILE, _MAX_PROGS),)](
                x2d,
                dm,
                rstd,
                d_w,
                tokens,
                N=n,
                MIX=mix,
                MIX_PAD=mix_pad,
                BLOCK_T=_T_TILE,
                BLOCK_D=_DFN_D_TILE,
                num_warps=8,
            )
        d_x = None
        if ctx.needs_input_grad[0]:  # type: ignore[attr-defined]
            d_x = torch.empty_like(x2d)
            _mhc_dx_kernel[(min(triton.cdiv(tokens, _T_TILE), _MAX_PROGS),)](
                x2d,
                w,
                dm,
                rstd,
                d_x,
                tokens,
                N=n,
                MIX=mix,
                MIX_PAD=mix_pad,
                BLOCK_T=_T_TILE,
                BLOCK_K=_K_TILE,
                BLOCK_K2=_DX_K2_TILE,
                num_warps=4,
            )
        return d_x, d_w, None


def mhc_norm_linear(x2d: Tensor, w: Tensor, eps: float) -> Tensor | None:
    """Fused mHC input-norm + mix projection.

    Args:
        x2d (Tensor): Flattened stream activations ``[T, H*D]``.
        w (Tensor): Mix projection weight ``[MIX, H*D]`` (``hc_fn``, in the
            activation dtype).
        eps (float): RMSNorm epsilon.

    Returns:
        Tensor | None: Mix logits ``[T, MIX]`` fp32, or ``None`` when the
        fused path cannot serve the request (caller falls back to the eager
        chain).
    """
    if not (MHC_FUSED_NORM_LINEAR and _mhc_servable(x2d, w)):
        return None
    return _MhcNormLinearFn.apply(x2d, w, eps)


# ------------------------------------------------------------------------------
# -- Split-Sinkhorn (MindSpeed-MM port): kernels, caches, autograd op
# ------------------------------------------------------------------------------
if hasattr(tl, "extract_slice"):
    extract_slice, insert_slice = tl.extract_slice, tl.insert_slice
else:
    from triton.language.extra.cann.extension import (  # type: ignore[no-redef]
        extract_slice,
        insert_slice,
    )


@triton.jit
def _hc_split_sinkhorn_kernel_part1(
    # Input/output tensor pointers
    mixes_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    # Dimension parameters
    batch_seq_size,
    # Constant parameters
    eps: tl.constexpr,
    feat_dim: tl.constexpr,
    # Block size (compile-time constant)
    hc_mult: tl.constexpr,
    group: tl.constexpr,
):
    """Core computation for HC-Split Sinkhorn (Pre/Post components).

    Each program processes ``group`` (batch, seq) samples: the sigmoid pre /
    post streams and the raw comb logits are split out of ``mixes``.

    Args:
        mixes_ptr: Input tensor mixes [batch_seq_size, feat_dim].
        hc_scale_ptr: Scale tensor [3] (pre / post / comb).
        hc_base_ptr: Base tensor [(2+hc_mult)*hc_mult].
        pre_ptr: Output pre tensor [batch_seq_size, hc_mult].
        post_ptr: Output post tensor [batch_seq_size, hc_mult].
        comb_ptr: Output comb logits tensor [batch_seq_size, hc_mult*hc_mult].
        batch_seq_size: Total number of (batch, seq) samples (b*s).
        eps: Small constant to avoid division by zero.
        feat_dim: Total feature dimension (2+hc_mult)*hc_mult.
        hc_mult: HC dimension size (typically 4).
        group: Number of samples processed per program.
    """
    ar4 = tl.arange(0, hc_mult)
    arange_val = tl.arange(0, hc_mult * hc_mult)

    # Calculate program IDs for grouped processing
    pid0 = tl.program_id(0) * group
    pids = pid0 + tl.arange(0, group)
    pid_mask = pids < batch_seq_size

    # Calculate memory offsets for each sample
    pid_comb_off = pids[:, None] * hc_mult * hc_mult
    pid_feat_off = pids[:, None] * feat_dim
    pid_hc_off = pids[:, None] * hc_mult

    # Load scale parameters (pre/post/comb)
    scale_pre = tl.load(hc_scale_ptr + 0)
    scale_post = tl.load(hc_scale_ptr + 1)
    scale_comb = tl.load(hc_scale_ptr + 2)

    # Load base parameters
    base_pre = tl.load(hc_base_ptr + ar4)
    base_post = tl.load(hc_base_ptr + hc_mult + ar4)
    base_comb = tl.load(hc_base_ptr + 2 * hc_mult + arange_val)

    # Load mixes tensor slices for pre/post/comb
    mixes_pre = tl.load(mixes_ptr + pid_feat_off + ar4[None, :], mask=pid_mask[:, None], other=0.0)
    mixes_post = tl.load(mixes_ptr + pid_feat_off + (hc_mult + ar4)[None, :], mask=pid_mask[:, None], other=0.0)
    mixes_comb = tl.load(
        mixes_ptr + pid_feat_off[:, :, None] + (2 * hc_mult + arange_val)[None, :],
        mask=pid_mask[:, None, None],
        other=0.0,
    )

    # Compute pre tensor with sigmoid activation
    pre = tl.sigmoid(mixes_pre * scale_pre + base_pre[None, :]) + eps
    tl.store(pre_ptr + pid_hc_off + ar4[None, :], pre, mask=pid_mask[:, None])

    # Compute post tensor with sigmoid activation
    post = 2.0 * tl.sigmoid(mixes_post * scale_post + base_post[None, :])
    tl.store(post_ptr + pid_hc_off + ar4[None, :], post, mask=pid_mask[:, None])

    # Compute comb logits and store
    comb = mixes_comb * scale_comb + base_comb[None, :]
    comb_flat = tl.reshape(comb, (group, hc_mult * hc_mult))
    tl.store(comb_ptr + pid_comb_off + arange_val[None, :], comb_flat, mask=pid_mask[:, None])


@triton.jit
def _hc_split_sinkhorn_kernel_part2(
    # Input/output tensor pointers
    comb_tmp_ptr,
    comb_ptr,
    # Dimension parameters
    batch_seq_size,
    hc_mult: tl.constexpr,
    sinkhorn_iters: tl.constexpr,
    # Constant parameters
    eps: tl.constexpr,
    group: tl.constexpr,
    BLOCK_ALIGN: tl.constexpr = 8,
):
    """Core computation for HC-Split Sinkhorn (Comb component).

    Implements the comb tensor calculation with Sinkhorn normalization
    iterations: exp-stabilise, then alternate row / column normalisation for
    ``sinkhorn_iters`` rounds (all on chip -- the eager fallback issues ~5
    tiny elementwise kernels per iteration).

    Args:
        comb_tmp_ptr: Temporary comb logits tensor [batch_seq_size, hc_mult*BLOCK_ALIGN].
        comb_ptr: Output comb tensor [batch_seq_size, hc_mult*BLOCK_ALIGN].
        batch_seq_size: Total number of (batch, seq) samples (b*s).
        hc_mult: HC dimension size (typically 4).
        sinkhorn_iters: Number of Sinkhorn normalization iterations.
        eps: Small constant to avoid division by zero.
        group: Number of samples processed per program.
        BLOCK_ALIGN: Compile-time constant for memory alignment (typically 8).
    """
    lin = tl.arange(0, hc_mult * BLOCK_ALIGN)

    # Calculate program IDs for grouped processing
    pid0 = tl.program_id(0) * group
    pids = pid0 + tl.arange(0, group)
    pid_mask = pids < batch_seq_size

    # Column-offset layout for the padded comb row
    pid_comb_off = pids[:, None] * (hc_mult * BLOCK_ALIGN)

    # Load and reshape comb tensor.
    # NOTE: no ``other=`` on this masked load -- masked-off lanes hold undefined
    # values. Deliberately kept as in the verbatim MindSpeed port: every
    # reduction here is token-local (row_max over axis=2, col_sum/row_sum over
    # axis=1, never across tokens) and the stores are masked, so the garbage
    # cannot escape a padded row. Adding an explicit ``other=`` would deviate
    # from the anchor port -- re-verify the run81 bit-exact anchors first.
    comb = tl.load(comb_tmp_ptr + pid_comb_off + lin[None, :], mask=pid_mask[:, None])
    comb = comb.reshape(group, hc_mult, BLOCK_ALIGN)

    # Numerical stability: subtract row max before exp
    row_max = tl.max(comb, axis=2)
    comb = tl.exp(comb - row_max[:, :, None])

    # Sinkhorn normalization iterations
    for _ in range(sinkhorn_iters):
        # Row normalization
        row_sum = tl.sum(comb, axis=2)
        comb = comb / (row_sum[:, :, None] + eps)

        # Column normalization
        col_sum = tl.sum(comb, axis=1)
        comb = comb / (col_sum[:, None, :] + eps)

    # Reshape and store final comb tensor
    comb_flat = tl.reshape(comb, (group, hc_mult * BLOCK_ALIGN))
    tl.store(comb_ptr + pid_comb_off + lin[None, :], comb_flat, mask=pid_mask[:, None])


@triton.jit
def _hc_split_sinkhorn_backward_kernel_part1(
    # Input gradient pointers
    grad_pre_ptr,
    grad_post_ptr,
    # Forward input pointers
    mixes_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    # Output gradient pointers
    comb_tmp_ptr,
    grad_mixes_ptr,
    grad_hc_scale_ptr,
    grad_hc_base_ptr,
    batch_seq_size,
    hc_mult: tl.constexpr = 4,
    group: tl.constexpr = 32,
):
    """Gradients for the Pre/Post components of HC-Split Sinkhorn.

    Calculates gradients for the sigmoid-transformed pre / post tensors and
    atomically accumulates the gradients for ``hc_scale`` / ``hc_base``; also
    re-emits the comb logits for the part2 backward kernel.

    Args:
        grad_pre_ptr: Gradient tensor for the pre output [batch_seq_size, hc_mult].
        grad_post_ptr: Gradient tensor for the post output [batch_seq_size, hc_mult].
        mixes_ptr: Forward input mixes tensor [batch_seq_size, (2+hc_mult)*hc_mult].
        hc_scale_ptr: Forward input scale tensor [3].
        hc_base_ptr: Forward input base tensor [(2+hc_mult)*hc_mult].
        comb_tmp_ptr: Output comb logits for the part2 backward kernel.
        grad_mixes_ptr: Output gradient tensor for the mixes input (pre/post slots).
        grad_hc_scale_ptr: Output gradient tensor for hc_scale (atomic add).
        grad_hc_base_ptr: Output gradient tensor for hc_base (atomic add).
        batch_seq_size: Total number of (batch, seq) samples (b*s).
        hc_mult: HC dimension size (default=4).
        group: Number of samples processed per program (default=32).
    """
    feat_dim = (2 + hc_mult) * hc_mult
    arange_val = tl.arange(0, hc_mult * hc_mult)

    # Calculate program IDs for grouped processing
    pid0 = tl.program_id(0) * group
    pids = pid0 + tl.arange(0, group)
    pid_mask = pids < batch_seq_size

    # Memory offset calculations
    pid_comb_off = pids[:, None] * hc_mult * hc_mult
    pid_feat_off = pids[:, None] * feat_dim
    pid_hc_off = pids[:, None] * hc_mult
    ar4 = tl.arange(0, hc_mult)

    # Load scale parameters
    scale_pre = tl.load(hc_scale_ptr + 0)
    scale_post = tl.load(hc_scale_ptr + 1)
    scale_comb = tl.load(hc_scale_ptr + 2)

    # Load forward input slices
    pre_slice = tl.load(mixes_ptr + pid_feat_off + ar4[None, :], mask=pid_mask[:, None], other=0.0)
    post_slice = tl.load(mixes_ptr + pid_feat_off + (hc_mult + ar4)[None, :], mask=pid_mask[:, None], other=0.0)
    comb_slice = tl.load(
        mixes_ptr + pid_feat_off + (2 * hc_mult + arange_val)[None, :], mask=pid_mask[:, None], other=0.0
    )

    # Load base parameters
    base_pre = tl.load(hc_base_ptr + ar4)
    base_post = tl.load(hc_base_ptr + hc_mult + ar4)
    base_comb = tl.load(hc_base_ptr + 2 * hc_mult + arange_val)

    # Compute gradients for pre component
    pre_input = pre_slice * scale_pre + base_pre[None, :]
    sigmoid_pre = tl.sigmoid(pre_input)
    sigmoid_deriv = sigmoid_pre * (1.0 - sigmoid_pre)
    grad_pre = tl.load(grad_pre_ptr + pid_hc_off + ar4[None, :], mask=pid_mask[:, None], other=0.0)
    grad_pre_input = grad_pre * sigmoid_deriv

    # Update gradients for mixes (pre slice)
    tl.store(grad_mixes_ptr + pid_feat_off + ar4[None, :], grad_pre_input * scale_pre, mask=pid_mask[:, None])

    # Atomic updates for scale and base gradients
    tl.atomic_add(grad_hc_scale_ptr + 0, tl.sum(grad_pre_input * pre_slice))
    grad_pre_input_sum = tl.sum(grad_pre_input, axis=0)
    tl.atomic_add(grad_hc_base_ptr + ar4, grad_pre_input_sum)

    # Compute gradients for post component
    post_input = post_slice * scale_post + base_post[None, :]
    sigmoid_post = tl.sigmoid(post_input)
    sigmoid_deriv_post = sigmoid_post * (1.0 - sigmoid_post)
    grad_post = tl.load(grad_post_ptr + pid_hc_off + ar4[None, :], mask=pid_mask[:, None], other=0.0)
    grad_post_input = grad_post * 2.0 * sigmoid_deriv_post

    # Update gradients for mixes (post slice)
    tl.store(
        grad_mixes_ptr + pid_feat_off + (hc_mult + ar4)[None, :], grad_post_input * scale_post, mask=pid_mask[:, None]
    )

    # Atomic updates for scale and base gradients
    tl.atomic_add(grad_hc_scale_ptr + 1, tl.sum(grad_post_input * post_slice))
    grad_post_input_sum = tl.sum(grad_post_input, axis=0)
    tl.atomic_add(grad_hc_base_ptr + hc_mult + ar4, grad_post_input_sum)

    # Prepare comb logits for Part2 backward kernel
    comb = comb_slice * scale_comb + base_comb[None, :]
    comb_flat = tl.reshape(comb, (group, hc_mult * hc_mult))
    tl.store(comb_tmp_ptr + pid_comb_off + arange_val[None, :], comb_flat, mask=pid_mask[:, None])


@triton.jit
def _hc_split_sinkhorn_backward_kernel_part2(
    # Input gradient pointer
    grad_comb_ptr,
    # Forward input pointers
    mixes_ptr,
    hc_scale_ptr,
    comb_tmp_ptr,
    # Output gradient pointers
    grad_mixes_ptr,
    grad_hc_scale_ptr,
    grad_hc_base_ptr,
    # Constant parameters (compile-time)
    batch_seq_size,
    hc_mult: tl.constexpr = 4,
    sinkhorn_iters: tl.constexpr = 20,
    eps: tl.constexpr = 1e-6,
    BLOCK_ALIGN: tl.constexpr = 8,
    group: tl.constexpr = 32,
):
    """Gradients for the Comb component of HC-Split Sinkhorn.

    Reconstructs the forward Sinkhorn iterations on chip (saving the row /
    column sums) and backpropagates the comb gradient through the
    normalisation chain, the exp-stabilisation and the linear comb mapping.

    Args:
        grad_comb_ptr: Gradient tensor for the comb output [batch_seq_size, hc_mult*BLOCK_ALIGN].
        mixes_ptr: Forward input comb-slice tensor [batch_seq_size, hc_mult*BLOCK_ALIGN].
        hc_scale_ptr: Forward input scale tensor [3].
        comb_tmp_ptr: Comb logits re-emitted by the part1 backward kernel.
        grad_mixes_ptr: Output gradient tensor for the mixes comb slice.
        grad_hc_scale_ptr: Output gradient tensor for hc_scale (atomic add).
        grad_hc_base_ptr: Output gradient tensor for hc_base (atomic add).
        batch_seq_size: Total number of (batch, seq) samples (b*s).
        hc_mult: HC dimension size (default=4).
        sinkhorn_iters: Number of Sinkhorn iterations (default=20).
        eps: Small constant to avoid division by zero (default=1e-6).
        BLOCK_ALIGN: Memory alignment constant (default=8).
        group: Number of samples processed per program (default=32).
    """
    # Initialize indices and masks
    arange_val = tl.arange(0, hc_mult * BLOCK_ALIGN)
    pid0 = tl.program_id(0) * group
    pids = pid0 + tl.arange(0, group)
    pid_mask = pids < batch_seq_size

    # Column mask for alignment handling
    c = tl.arange(0, BLOCK_ALIGN)[None, :]
    col_mask = c < hc_mult
    mask_val = col_mask[None, :, :]
    pid_feat_off = pids[:, None] * hc_mult * BLOCK_ALIGN

    # Load and reshape comb tensors
    comb_slice_flat = tl.load(mixes_ptr + pid_feat_off + arange_val)
    comb_slice = comb_slice_flat.reshape(group, hc_mult, BLOCK_ALIGN)

    # Load scale parameter for comb component
    scale_comb = tl.load(hc_scale_ptr + 2)

    # Load initial comb values re-emitted by part1 backward
    comb_init = tl.load(comb_tmp_ptr + pid_feat_off + arange_val)
    comb_init = comb_init.reshape(group, hc_mult, BLOCK_ALIGN)

    # Reconstruct forward Sinkhorn computation
    row_max = tl.max(comb_init, axis=2).reshape(group, hc_mult, 1)
    exp_comb = tl.exp(comb_init - row_max)

    # Save row/column sums for backward pass
    row_sum_list = tl.full((sinkhorn_iters, group, hc_mult, 1), 0.0, dtype=tl.float32)
    col_sum_list = tl.full((sinkhorn_iters, group, 1, BLOCK_ALIGN), 0.0, dtype=tl.float32)
    K = exp_comb

    # Replay forward iterations to save intermediate values
    for i in range(sinkhorn_iters):
        # Row normalization
        row_sum = tl.sum(K, axis=2).reshape(group, hc_mult, 1)
        K_row = K / (row_sum + eps)

        # Column normalization
        col_sum = tl.sum(K_row, axis=1).reshape(group, 1, BLOCK_ALIGN)
        K_col = K_row / (col_sum + eps)

        # Save intermediate sums
        row_sum_list = insert_slice(
            ful=row_sum_list,
            sub=row_sum[None, :, :, :],
            offsets=[i, 0, 0, 0],
            sizes=[1, group, hc_mult, 1],
            strides=[1, 1, 1, 1],
        )
        col_sum_list = insert_slice(
            ful=col_sum_list,
            sub=col_sum[None, :, :, :],
            offsets=[i, 0, 0, 0],
            sizes=[1, group, 1, BLOCK_ALIGN],
            strides=[1, 1, 1, 1],
        )
        K = K_col

    # Load comb gradient and reshape
    grad_comb_flat = tl.load(grad_comb_ptr + pid_feat_off + arange_val).to(tl.float32)
    dK = grad_comb_flat.reshape(group, hc_mult, BLOCK_ALIGN)

    # Backpropagate through Sinkhorn iterations (reverse order)
    for j in range(sinkhorn_iters):
        i = sinkhorn_iters - j - 1

        # Extract saved intermediate sums
        row_sum = extract_slice(
            row_sum_list,
            [i, 0, 0, 0],
            [1, group, hc_mult, 1],
            [1, 1, 1, 1],
        )
        col_sum = extract_slice(
            col_sum_list,
            [i, 0, 0, 0],
            [1, group, 1, BLOCK_ALIGN],
            [1, 1, 1, 1],
        )

        # Backprop column normalization
        col_sum = col_sum.reshape(group, 1, BLOCK_ALIGN) + eps
        row_sum = row_sum.reshape(group, hc_mult, 1) + eps
        K_col = K * col_sum

        grad_direct = dK / col_sum
        d_col_sum_compressed = -tl.sum(dK * K_col / (col_sum * col_sum), axis=-2)
        dK_row = grad_direct + d_col_sum_compressed[:, None, :]

        # Backprop row normalization
        K_row = K_col * row_sum
        K = K_row

        grad_direct_row = dK_row / row_sum
        d_row_sum_compressed = -tl.sum(dK_row * K_row / (row_sum * row_sum), axis=-1)
        dK = grad_direct_row + d_row_sum_compressed[:, :, None]

        dK = dK * mask_val

    # Backprop through exp and row max subtraction
    d_exp_comb = dK
    d_comb_before_exp = d_exp_comb * exp_comb

    # Handle gradient of row max subtraction
    max_mask = tl.where(comb_init == row_max, 1.0, 0.0)
    max_count = tl.sum(max_mask, axis=-1).reshape(group, hc_mult, 1) + eps
    row_sum_d_before_exp = tl.sum(d_comb_before_exp, axis=-1).reshape(group, hc_mult, 1)
    d_comb_init = d_comb_before_exp - (row_sum_d_before_exp * max_mask / max_count)

    # Backprop through linear transformation
    grad_comb_slice_flat = d_comb_init * scale_comb

    # Update mixes gradient
    tl.store(
        grad_mixes_ptr + pid_feat_off + arange_val[None, :],
        grad_comb_slice_flat.reshape(group, hc_mult * BLOCK_ALIGN),
        mask=pid_mask[:, None],
    )

    # Atomic updates for scale and base gradients (with boundary check)
    tmp_res = d_comb_init * comb_slice
    tmp_res = tl.where(pid_mask[:, None, None], tmp_res, 0.0)
    d_comb_init = tl.where(pid_mask[:, None, None], d_comb_init, 0.0)

    tl.atomic_add(grad_hc_scale_ptr + 2, tl.sum(tmp_res))
    d_comb_init_sum = tl.sum(d_comb_init, axis=0)
    tl.atomic_add(grad_hc_base_ptr + arange_val, d_comb_init_sum.reshape(hc_mult * BLOCK_ALIGN))


def _padded_comb_buffer(rows: int, hc_mult: int, block_align: int, ref: Tensor, fill: float, slot: str) -> Tensor:
    """Return a cached ``[rows, hc_mult, block_align]`` buffer with pad columns
    pre-filled with ``fill``.

    The caller rewrites only the leading ``hc_mult`` columns (slice copy), so
    the pad columns keep their fill value forever and the per-call ``F.pad``
    dispatch disappears. ``slot`` distinguishes the concurrently-live buffers
    of one call (e.g. ``mixes_pad`` and ``grad_comb_flat_pad`` share shape,
    dtype and fill in the backward — one shared buffer would clobber the
    other's leading columns).

    Args:
        rows (int): Leading dimension (``batch_seq_size``).
        hc_mult (int): HC dimension (4); the leading columns kept per row.
        block_align (int): Row stride of the buffer (8).
        ref (Tensor): Reference tensor for device and dtype.
        fill (float): Value pre-filled into the pad columns (``0.0`` or
            ``float("-inf")``).
        slot (str): Role tag distinguishing simultaneously-live buffers.

    Returns:
        Tensor: The cached padded buffer.
    """
    key = (slot, rows, str(ref.device), ref.dtype, fill)
    buf = _PADDED_COMB_CACHE.get(key)
    if buf is None:
        if len(_PADDED_COMB_CACHE) > 64:
            _PADDED_COMB_CACHE.clear()
        buf = torch.empty((rows, hc_mult, block_align), dtype=ref.dtype, device=ref.device)
        buf[:, :, hc_mult:].fill_(fill)
        _PADDED_COMB_CACHE[key] = buf
    return buf


def _comb_scratch_buffer(shape: tuple[int, ...], ref: Tensor) -> Tensor:
    """Return a cached fully-rewritten scratch buffer of ``shape``.

    Args:
        shape (tuple[int, ...]): Shape of the scratch buffer.
        ref (Tensor): Reference tensor for device and dtype.

    Returns:
        Tensor: The cached scratch buffer.
    """
    key = (shape, str(ref.device), ref.dtype)
    buf = _COMB_SCRATCH_CACHE.get(key)
    if buf is None:
        if len(_COMB_SCRATCH_CACHE) > 64:
            _COMB_SCRATCH_CACHE.clear()
        buf = torch.empty(shape, dtype=ref.dtype, device=ref.device)
        _COMB_SCRATCH_CACHE[key] = buf
    return buf


class SinkhornFunction(torch.autograd.Function):
    """MHC split + Sinkhorn projection as two fused triton kernels."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        mixes: Tensor,
        hc_scale: Tensor,
        hc_base: Tensor,
        hc_mult: int = 4,
        sinkhorn_iters: int = 20,
        eps: float = 1e-6,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Split ``mixes`` into pre/post/comb and Sinkhorn-project comb.

        Args:
            ctx (FunctionCtx): Autograd context.
            mixes (Tensor): mHC mix logits ``[B, S, (2+hc_mult)*hc_mult]``.
            hc_scale (Tensor): Per-output learned scales ``[3]``.
            hc_base (Tensor): Learned biases ``[(2+hc_mult)*hc_mult]``.
            hc_mult (int): HC dimension; only 4 is supported by the kernels.
            sinkhorn_iters (int): Sinkhorn-Knopp iteration count.
            eps (float): Stabilising epsilon.

        Returns:
            tuple[Tensor, Tensor, Tensor]: ``pre`` ``[B, S, hc_mult]``, ``post``
            ``[B, S, hc_mult]`` and ``comb`` ``[B, S, hc_mult, hc_mult]``.
        """
        if len(mixes.shape) != 3:
            raise ValueError(f"this op is not supported when mixes.shape != 3, got {tuple(mixes.shape)}")
        if hc_mult != 4:
            raise ValueError(f"hc_mult only support 4, got {hc_mult}")

        # Flatten batch and sequence dimensions for triton processing
        b, s, _ = mixes.shape
        feat_dim = (2 + hc_mult) * hc_mult
        batch_seq_size = b * s
        mixes_flat = mixes.reshape(-1, feat_dim).contiguous()

        # Initialize output tensors (comb_tmp is a cached fully-rewritten
        # scratch; pre/post are fresh so callers may hold them across calls)
        pre_flat = torch.empty((batch_seq_size, hc_mult), dtype=mixes.dtype, device=mixes.device)
        post_flat = torch.empty((batch_seq_size, hc_mult), dtype=mixes.dtype, device=mixes.device)
        comb_tmp = _comb_scratch_buffer((batch_seq_size, hc_mult, hc_mult), mixes)

        # Configure triton kernel parameters
        block_align = 8
        # Grid sizes at the documented production shape (B=1, S=8192 -> BS=8192):
        # part1 grid = cdiv(8192, 256) = 32, part2 grid = cdiv(8192, 32) = 256.
        # Neither is capped/probed here: both binaries are verbatim MindSpeed
        # ports whose grid extents are anchor-validated on this fleet
        # (run81/107/145 bit-exact at grid=256), and 32 <= 48 physical vector
        # cores. Revisit both groups if the per-rank S grows.
        group_part1 = 256
        group_part2 = 32

        # Launch Part1 kernel (Pre/Post computation)
        _hc_split_sinkhorn_kernel_part1[(triton.cdiv(batch_seq_size, group_part1),)](
            mixes_flat,
            hc_scale,
            hc_base,
            pre_flat,
            post_flat,
            comb_tmp,
            batch_seq_size,
            eps,
            feat_dim,
            hc_mult,
            group_part1,
        )

        # Pad comb tensor for memory alignment: cached buffer, pad columns
        # already hold -inf, only the leading hc_mult columns are rewritten.
        comb_tmp_padded = _padded_comb_buffer(
            batch_seq_size, hc_mult, block_align, mixes, float("-inf"), "fwd_comb_tmp"
        )
        comb_tmp_padded[:, :, :hc_mult].copy_(comb_tmp)
        # NOTE: comb_flat_padded stays a fresh empty — the returned comb is a
        # view of it, so a cached buffer would be clobbered by the next call.
        comb_flat_padded = torch.empty((batch_seq_size, hc_mult * block_align), dtype=mixes.dtype, device=mixes.device)

        # Launch Part2 kernel (Comb computation with Sinkhorn normalization)
        _hc_split_sinkhorn_kernel_part2[(triton.cdiv(batch_seq_size, group_part2),)](
            comb_tmp_padded,
            comb_flat_padded,
            batch_seq_size,
            hc_mult,
            sinkhorn_iters,
            eps,
            group_part2,
            BLOCK_ALIGN=block_align,
        )

        # Reshape outputs and restore original layout
        pre = pre_flat.view(b, s, hc_mult)
        post = post_flat.view(b, s, hc_mult)
        comb = comb_flat_padded.view(b, s, hc_mult, block_align)[:, :, :, :hc_mult]

        ctx.save_for_backward(mixes, hc_scale, hc_base)  # type: ignore[attr-defined]

        ctx.hc_mult = hc_mult  # type: ignore[attr-defined]
        ctx.sinkhorn_iters = sinkhorn_iters  # type: ignore[attr-defined]
        ctx.eps = eps  # type: ignore[attr-defined]

        return pre, post, comb

    @staticmethod
    @once_differentiable
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_pre: Tensor,
        grad_post: Tensor,
        grad_comb: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, None, None, None]:
        """Gradients of the fused forward (mixes / hc_scale / hc_base).

        Args:
            ctx (FunctionCtx): Autograd context.
            grad_pre (Tensor): Upstream gradient of pre ``[B, S, hc_mult]``.
            grad_post (Tensor): Upstream gradient of post ``[B, S, hc_mult]``.
            grad_comb (Tensor): Upstream gradient of comb ``[B, S, hc_mult, hc_mult]``.

        Returns:
            tuple[Tensor, Tensor, Tensor, None, None, None]: Gradients w.r.t.
            ``mixes`` / ``hc_scale`` / ``hc_base`` and three ``None`` for the
            non-tensor arguments.
        """
        mixes, hc_scale, hc_base = ctx.saved_tensors  # type: ignore[attr-defined]
        hc_mult = ctx.hc_mult  # type: ignore[attr-defined]
        sinkhorn_iters = ctx.sinkhorn_iters  # type: ignore[attr-defined]
        eps = ctx.eps  # type: ignore[attr-defined]

        # The backward kernels index rows with raw ``pid * feat_dim`` /
        # ``pid * hc_mult`` pointer arithmetic (packed layouts only), and
        # ``reshape`` below may hand back a non-packed *view* without copying.
        # The forward defensively contiguifies mixes; mirror that for every
        # tensor fed to a kernel here. No-op for today's callers (all fresh
        # contiguous tensors), but it prevents a future slice/transpose view
        # from silently misreading gradients (matches mhc_norm_linear's
        # backward, which contiguifies its dmixes the same way).
        mixes = mixes.contiguous()
        grad_pre = grad_pre.contiguous()
        grad_post = grad_post.contiguous()

        # Input dimension retrieval
        b, s, _ = mixes.shape
        batch_seq_size = b * s

        # Initialize gradient tensors with zeros (both backward kernels
        # accumulate with atomic_add, so they must start zeroed). The five
        # buffers are carved as contiguous views of one zeroed flat buffer per
        # dtype: at 128K this backward runs ~950 times per training step, and
        # five separate zeros allocations cost ~4 extra device ops each call
        # (~1.5s/step of host-bound free). Views are contiguous, so the kernels'
        # pointer math and the in-place slice merges below see byte-identical
        # layouts.
        block_align = 8
        feat_dim = (2 + hc_mult) * hc_mult
        n_mixes = batch_seq_size * feat_dim
        n_base = hc_base.numel()
        n_scale = hc_scale.numel()
        if hc_scale.dtype == hc_base.dtype:
            scale_base_flat = torch.zeros(
                n_scale + n_base + hc_mult * block_align, dtype=hc_scale.dtype, device=hc_base.device
            )
            grad_hc_scale = scale_base_flat[:n_scale]
            grad_hc_base = scale_base_flat[n_scale : n_scale + n_base]
            grad_hc_base_pad = scale_base_flat[n_scale + n_base :].view(hc_mult, block_align)
        else:
            grad_hc_scale = torch.zeros_like(hc_scale, device=hc_scale.device)
            grad_hc_base = torch.zeros_like(hc_base, device=hc_base.device)
            grad_hc_base_pad = torch.zeros(
                (hc_mult, block_align), dtype=grad_hc_base.dtype, device=grad_hc_base.device
            )
        bf16_flat = torch.zeros(
            n_mixes + batch_seq_size * hc_mult * block_align, dtype=mixes.dtype, device=mixes.device
        )
        grad_mixes = bf16_flat[:n_mixes].view(b, s, feat_dim)
        grad_mixes_pad = bf16_flat[n_mixes:].view(batch_seq_size, hc_mult, block_align)
        comb_tmp = _comb_scratch_buffer((batch_seq_size, hc_mult, hc_mult), mixes)

        # Flatten gradient tensors for triton processing
        grad_pre_flat = grad_pre.reshape(-1, hc_mult)
        grad_post_flat = grad_post.reshape(-1, hc_mult)

        # Configure triton kernel parameters
        group_part1 = 256  # matches the forward launch (see the grid note there)
        group_part2 = 32

        # Launch Part1 kernel (Pre/Post gradients)
        _hc_split_sinkhorn_backward_kernel_part1[(triton.cdiv(batch_seq_size, group_part1),)](
            grad_pre_flat,
            grad_post_flat,
            mixes,
            hc_scale,
            hc_base,
            comb_tmp,
            grad_mixes,
            grad_hc_scale,
            grad_hc_base,
            batch_seq_size,
            hc_mult=hc_mult,
            group=group_part1,
        )

        # Prepare comb slice for Part2 backward kernel (padding for alignment,
        # via the cached pad-columns-prefilled buffers)
        mixes_flat = mixes.view(-1, (2 + hc_mult) * hc_mult)
        mixes_slice = mixes_flat[:, 2 * hc_mult :].view(-1, hc_mult, hc_mult)
        mixes_pad = _padded_comb_buffer(batch_seq_size, hc_mult, block_align, mixes, 0.0, "bwd_mixes")
        mixes_pad[:, :, :hc_mult].copy_(mixes_slice)

        # Pad comb gradient tensor
        grad_comb_flat = grad_comb.reshape(-1, hc_mult, hc_mult)
        grad_comb_flat_pad = _padded_comb_buffer(batch_seq_size, hc_mult, block_align, mixes, 0.0, "bwd_grad_comb")
        grad_comb_flat_pad[:, :, :hc_mult].copy_(grad_comb_flat)
        comb_tmp_padded = _padded_comb_buffer(
            batch_seq_size, hc_mult, block_align, mixes, float("-inf"), "bwd_comb_tmp"
        )
        comb_tmp_padded[:, :, :hc_mult].copy_(comb_tmp)

        # If batch_seq_size is not an integer multiple of group_part2, the last
        # block would read beyond the end of the tensor. The rows of the three
        # inputs are padded up to a multiple while batch_seq_size passed to the
        # kernel stays unchanged.
        row_pad = -batch_seq_size % group_part2
        if row_pad:
            mixes_pad = F.pad(mixes_pad, (0, 0, 0, 0, 0, row_pad))
            grad_comb_flat_pad = F.pad(grad_comb_flat_pad, (0, 0, 0, 0, 0, row_pad))
            comb_tmp_padded = F.pad(comb_tmp_padded, (0, 0, 0, 0, 0, row_pad))
        # Launch Part2 kernel (Comb gradients)
        _hc_split_sinkhorn_backward_kernel_part2[(triton.cdiv(batch_seq_size, group_part2),)](
            grad_comb_flat_pad,
            mixes_pad,
            hc_scale,
            comb_tmp_padded,
            grad_mixes_pad,
            grad_hc_scale,
            grad_hc_base_pad,
            batch_seq_size,
            hc_mult,
            sinkhorn_iters,
            eps,
            BLOCK_ALIGN=block_align,
            group=group_part2,
        )

        # Merge padded gradients back to original shape
        grad_mixes_slice = grad_mixes_pad[:, :, :hc_mult].reshape(b, s, hc_mult * hc_mult)
        grad_hc_base_slice = grad_hc_base_pad[:, :hc_mult].reshape(hc_mult * hc_mult)

        # Update final gradients
        grad_mixes[:, :, 2 * hc_mult :] = grad_mixes_slice
        grad_hc_base[2 * hc_mult :] = grad_hc_base_slice

        return grad_mixes, grad_hc_scale, grad_hc_base, None, None, None


def hc_split_sinkhorn_fused(
    mixes: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fused split + Sinkhorn projection; mirrors MindSpeed-MM's entry.

    Args:
        mixes (Tensor): mHC mix logits ``[B, S, (2+hc_mult)*hc_mult]``.
        hc_scale (Tensor): Per-output learned scales ``[3]``.
        hc_base (Tensor): Learned biases ``[(2+hc_mult)*hc_mult]``.
        hc_mult (int): HC dimension; only 4 is supported by the kernels.
        sinkhorn_iters (int): Sinkhorn-Knopp iteration count.
        eps (float): Stabilising epsilon.

    Returns:
        tuple[Tensor, Tensor, Tensor]: ``pre``, ``post`` and ``comb`` of shapes
        ``[B, S, hc_mult]`` / ``[B, S, hc_mult]`` / ``[B, S, hc_mult, hc_mult]``.
    """

    def _local(x: Tensor) -> Tensor:
        return x.to_local() if isinstance(x, DTensor) else x

    return SinkhornFunction.apply(_local(mixes), _local(hc_scale), _local(hc_base), hc_mult, sinkhorn_iters, eps)


# ------------------------------------------------------------------------------
# -- Stream combine / collapse (fused decoder-layer stream update + pre-collapse)
# ------------------------------------------------------------------------------
@triton.jit
def _bf16_round(x):
    """Round an fp32 value to bf16 precision, keeping fp32 storage (RTNE).

    Manual bit-twiddling RTNE: the ``.to(tl.bfloat16).to(tl.float32)``
    round-trip gets folded away by the BiShengIR compiler (observed on
    scalars and some vector shapes), so the convert pair is not reliable.
    """
    b = x.to(tl.uint32, bitcast=True)
    r = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return r.to(tl.float32, bitcast=True)


@triton.jit
def _hc_combine_fwd_kernel(
    post_ptr,
    comb_ptr,
    out_ptr,
    res_ptr,
    new_ptr,
    T,
    D,
    POST_ST_T: tl.constexpr,
    COMB_ST_T: tl.constexpr,
    COMB_ST_K: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Token-tiled (BLOCK_T rows) rewrite of the original per-token kernel: the
    # 20 per-token scalar loads (post[4] + comb[4x4]) become ~2.8 vector loads
    # per token and the grid covers all aiv blocks -- 2.2x the eager fwd and
    # 4.2x the per-token kernel at the training shape (910C spike, fwd+bwd pair
    # 4.23 -> 2.96 ms). Static-unrolled over HC with plain 2D tiles only
    # (BiShengIR rejects the 4D broadcast product). The eager bf16 rounding
    # points are replicated bitwise: mul and matmul results keep the manual
    # _bf16_round -- BiShengIR promotes bf16 arithmetic to fp32 without
    # intermediate rounding, so a bf16-typed mul does NOT reproduce the eager
    # mul cast (bitwise-tested, 1-ulp drift) -- while the final add is rounded
    # by the bf16 store cast itself (RTNE of the already-rounded add), dropping
    # one of the three round ops per output element.
    pid = tl.program_id(0)
    nprogs = tl.num_programs(0)
    for blk in range(pid, tl.cdiv(T, BLOCK_T), nprogs):
        t_off = (blk * BLOCK_T + tl.arange(0, BLOCK_T)).to(tl.int64)
        t_mask = t_off < T
        pb = post_ptr + t_off * POST_ST_T
        p0 = _bf16_round(tl.load(pb + 0, mask=t_mask, other=0.0).to(tl.float32))
        p1 = _bf16_round(tl.load(pb + 1, mask=t_mask, other=0.0).to(tl.float32))
        p2 = _bf16_round(tl.load(pb + 2, mask=t_mask, other=0.0).to(tl.float32))
        p3 = _bf16_round(tl.load(pb + 3, mask=t_mask, other=0.0).to(tl.float32))
        cb = comb_ptr + t_off * COMB_ST_T
        c00 = _bf16_round(tl.load(cb + 0 * COMB_ST_K + 0, mask=t_mask, other=0.0).to(tl.float32))
        c01 = _bf16_round(tl.load(cb + 0 * COMB_ST_K + 1, mask=t_mask, other=0.0).to(tl.float32))
        c02 = _bf16_round(tl.load(cb + 0 * COMB_ST_K + 2, mask=t_mask, other=0.0).to(tl.float32))
        c03 = _bf16_round(tl.load(cb + 0 * COMB_ST_K + 3, mask=t_mask, other=0.0).to(tl.float32))
        c10 = _bf16_round(tl.load(cb + 1 * COMB_ST_K + 0, mask=t_mask, other=0.0).to(tl.float32))
        c11 = _bf16_round(tl.load(cb + 1 * COMB_ST_K + 1, mask=t_mask, other=0.0).to(tl.float32))
        c12 = _bf16_round(tl.load(cb + 1 * COMB_ST_K + 2, mask=t_mask, other=0.0).to(tl.float32))
        c13 = _bf16_round(tl.load(cb + 1 * COMB_ST_K + 3, mask=t_mask, other=0.0).to(tl.float32))
        c20 = _bf16_round(tl.load(cb + 2 * COMB_ST_K + 0, mask=t_mask, other=0.0).to(tl.float32))
        c21 = _bf16_round(tl.load(cb + 2 * COMB_ST_K + 1, mask=t_mask, other=0.0).to(tl.float32))
        c22 = _bf16_round(tl.load(cb + 2 * COMB_ST_K + 2, mask=t_mask, other=0.0).to(tl.float32))
        c23 = _bf16_round(tl.load(cb + 2 * COMB_ST_K + 3, mask=t_mask, other=0.0).to(tl.float32))
        c30 = _bf16_round(tl.load(cb + 3 * COMB_ST_K + 0, mask=t_mask, other=0.0).to(tl.float32))
        c31 = _bf16_round(tl.load(cb + 3 * COMB_ST_K + 1, mask=t_mask, other=0.0).to(tl.float32))
        c32 = _bf16_round(tl.load(cb + 3 * COMB_ST_K + 2, mask=t_mask, other=0.0).to(tl.float32))
        c33 = _bf16_round(tl.load(cb + 3 * COMB_ST_K + 3, mask=t_mask, other=0.0).to(tl.float32))
        for d0 in range(0, D, BLOCK_D):
            d_off = d0 + tl.arange(0, BLOCK_D)
            m2 = t_mask[:, None] & (d_off < D)[None, :]
            out_v = tl.load(out_ptr + t_off[:, None] * D + d_off[None, :], mask=m2, other=0.0).to(tl.float32)
            r0 = tl.load(res_ptr + (t_off * HC + 0)[:, None] * D + d_off[None, :], mask=m2, other=0.0).to(tl.float32)
            r1 = tl.load(res_ptr + (t_off * HC + 1)[:, None] * D + d_off[None, :], mask=m2, other=0.0).to(tl.float32)
            r2 = tl.load(res_ptr + (t_off * HC + 2)[:, None] * D + d_off[None, :], mask=m2, other=0.0).to(tl.float32)
            r3 = tl.load(res_ptr + (t_off * HC + 3)[:, None] * D + d_off[None, :], mask=m2, other=0.0).to(tl.float32)
            base = (t_off * HC)[:, None] * D + d_off[None, :]
            mul = _bf16_round(p0[:, None] * out_v)
            mm = _bf16_round(c00[:, None] * r0 + c10[:, None] * r1 + c20[:, None] * r2 + c30[:, None] * r3)
            tl.store(new_ptr + base + 0 * D, (mul + mm).to(tl.bfloat16), mask=m2)
            mul = _bf16_round(p1[:, None] * out_v)
            mm = _bf16_round(c01[:, None] * r0 + c11[:, None] * r1 + c21[:, None] * r2 + c31[:, None] * r3)
            tl.store(new_ptr + base + 1 * D, (mul + mm).to(tl.bfloat16), mask=m2)
            mul = _bf16_round(p2[:, None] * out_v)
            mm = _bf16_round(c02[:, None] * r0 + c12[:, None] * r1 + c22[:, None] * r2 + c32[:, None] * r3)
            tl.store(new_ptr + base + 2 * D, (mul + mm).to(tl.bfloat16), mask=m2)
            mul = _bf16_round(p3[:, None] * out_v)
            mm = _bf16_round(c03[:, None] * r0 + c13[:, None] * r1 + c23[:, None] * r2 + c33[:, None] * r3)
            tl.store(new_ptr + base + 3 * D, (mul + mm).to(tl.bfloat16), mask=m2)


@triton.jit
def _hc_combine_bwd_kernel(
    post_ptr,
    comb_ptr,
    out_ptr,
    res_ptr,
    g_ptr,
    d_post_ptr,
    d_comb_ptr,
    d_out_ptr,
    d_res_ptr,
    T,
    D,
    POST_ST_T: tl.constexpr,
    COMB_ST_T: tl.constexpr,
    COMB_ST_K: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Token-tiled (BLOCK_T rows) rewrite of the original per-token kernel, carrying the
    # same _bf16_round placement as the eager chain (see _hc_combine_fwd_kernel).
    # d_out / d_res accumulate element-wise over D, so they stay bitwise-equal to the
    # per-token kernel; d_post / d_comb reduce over D inside BLOCK_D-wide tiles, so
    # their fp32 sums may differ by one ulp where BLOCK_D differs from the old launch.
    pid = tl.program_id(0)
    nprogs = tl.num_programs(0)
    for blk in range(pid, tl.cdiv(T, BLOCK_T), nprogs):
        t_off = (blk * BLOCK_T + tl.arange(0, BLOCK_T)).to(tl.int64)
        t_mask = t_off < T
        pb = post_ptr + t_off * POST_ST_T
        p0 = _bf16_round(tl.load(pb + 0, mask=t_mask, other=0.0).to(tl.float32))
        p1 = _bf16_round(tl.load(pb + 1, mask=t_mask, other=0.0).to(tl.float32))
        p2 = _bf16_round(tl.load(pb + 2, mask=t_mask, other=0.0).to(tl.float32))
        p3 = _bf16_round(tl.load(pb + 3, mask=t_mask, other=0.0).to(tl.float32))
        cb = comb_ptr + t_off * COMB_ST_T
        c00 = _bf16_round(tl.load(cb + 0 * COMB_ST_K + 0, mask=t_mask, other=0.0).to(tl.float32))
        c01 = _bf16_round(tl.load(cb + 0 * COMB_ST_K + 1, mask=t_mask, other=0.0).to(tl.float32))
        c02 = _bf16_round(tl.load(cb + 0 * COMB_ST_K + 2, mask=t_mask, other=0.0).to(tl.float32))
        c03 = _bf16_round(tl.load(cb + 0 * COMB_ST_K + 3, mask=t_mask, other=0.0).to(tl.float32))
        c10 = _bf16_round(tl.load(cb + 1 * COMB_ST_K + 0, mask=t_mask, other=0.0).to(tl.float32))
        c11 = _bf16_round(tl.load(cb + 1 * COMB_ST_K + 1, mask=t_mask, other=0.0).to(tl.float32))
        c12 = _bf16_round(tl.load(cb + 1 * COMB_ST_K + 2, mask=t_mask, other=0.0).to(tl.float32))
        c13 = _bf16_round(tl.load(cb + 1 * COMB_ST_K + 3, mask=t_mask, other=0.0).to(tl.float32))
        c20 = _bf16_round(tl.load(cb + 2 * COMB_ST_K + 0, mask=t_mask, other=0.0).to(tl.float32))
        c21 = _bf16_round(tl.load(cb + 2 * COMB_ST_K + 1, mask=t_mask, other=0.0).to(tl.float32))
        c22 = _bf16_round(tl.load(cb + 2 * COMB_ST_K + 2, mask=t_mask, other=0.0).to(tl.float32))
        c23 = _bf16_round(tl.load(cb + 2 * COMB_ST_K + 3, mask=t_mask, other=0.0).to(tl.float32))
        c30 = _bf16_round(tl.load(cb + 3 * COMB_ST_K + 0, mask=t_mask, other=0.0).to(tl.float32))
        c31 = _bf16_round(tl.load(cb + 3 * COMB_ST_K + 1, mask=t_mask, other=0.0).to(tl.float32))
        c32 = _bf16_round(tl.load(cb + 3 * COMB_ST_K + 2, mask=t_mask, other=0.0).to(tl.float32))
        c33 = _bf16_round(tl.load(cb + 3 * COMB_ST_K + 3, mask=t_mask, other=0.0).to(tl.float32))
        dp0 = tl.zeros([BLOCK_T], dtype=tl.float32)  # d_post[t, n], accumulated across D chunks
        dp1 = tl.zeros([BLOCK_T], dtype=tl.float32)
        dp2 = tl.zeros([BLOCK_T], dtype=tl.float32)
        dp3 = tl.zeros([BLOCK_T], dtype=tl.float32)
        q00 = tl.zeros([BLOCK_T], dtype=tl.float32)  # d_comb[t, k, n], accumulated across D chunks
        q01 = tl.zeros([BLOCK_T], dtype=tl.float32)
        q02 = tl.zeros([BLOCK_T], dtype=tl.float32)
        q03 = tl.zeros([BLOCK_T], dtype=tl.float32)
        q10 = tl.zeros([BLOCK_T], dtype=tl.float32)
        q11 = tl.zeros([BLOCK_T], dtype=tl.float32)
        q12 = tl.zeros([BLOCK_T], dtype=tl.float32)
        q13 = tl.zeros([BLOCK_T], dtype=tl.float32)
        q20 = tl.zeros([BLOCK_T], dtype=tl.float32)
        q21 = tl.zeros([BLOCK_T], dtype=tl.float32)
        q22 = tl.zeros([BLOCK_T], dtype=tl.float32)
        q23 = tl.zeros([BLOCK_T], dtype=tl.float32)
        q30 = tl.zeros([BLOCK_T], dtype=tl.float32)
        q31 = tl.zeros([BLOCK_T], dtype=tl.float32)
        q32 = tl.zeros([BLOCK_T], dtype=tl.float32)
        q33 = tl.zeros([BLOCK_T], dtype=tl.float32)
        for d0 in range(0, D, BLOCK_D):
            d_off = d0 + tl.arange(0, BLOCK_D)
            m2 = t_mask[:, None] & (d_off < D)[None, :]
            row = (t_off * HC)[:, None] * D + d_off[None, :]  # stream row: (t*HC + h) * D + d
            orow = t_off[:, None] * D + d_off[None, :]  # single-stream row: t * D + d
            out_v = tl.load(out_ptr + orow, mask=m2, other=0.0).to(tl.float32)
            g0 = tl.load(g_ptr + row + 0 * D, mask=m2, other=0.0).to(tl.float32)
            g1 = tl.load(g_ptr + row + 1 * D, mask=m2, other=0.0).to(tl.float32)
            g2 = tl.load(g_ptr + row + 2 * D, mask=m2, other=0.0).to(tl.float32)
            g3 = tl.load(g_ptr + row + 3 * D, mask=m2, other=0.0).to(tl.float32)
            r0 = tl.load(res_ptr + row + 0 * D, mask=m2, other=0.0).to(tl.float32)
            r1 = tl.load(res_ptr + row + 1 * D, mask=m2, other=0.0).to(tl.float32)
            r2 = tl.load(res_ptr + row + 2 * D, mask=m2, other=0.0).to(tl.float32)
            r3 = tl.load(res_ptr + row + 3 * D, mask=m2, other=0.0).to(tl.float32)
            # d_out[t, d] = sum_n bf16(post[n] * g[n, d])  (Mul-bwd + ReduceSum, sequential over HC)
            do_acc = _bf16_round(p0[:, None] * g0)
            do_acc += _bf16_round(p1[:, None] * g1)
            do_acc += _bf16_round(p2[:, None] * g2)
            do_acc += _bf16_round(p3[:, None] * g3)
            tl.store(d_out_ptr + orow, do_acc.to(tl.bfloat16), mask=m2)
            # d_post[t, n] = sum_d bf16(g[n, d] * out[d])  (Mul-bwd + ReduceSum over D)
            dp0 += tl.sum(_bf16_round(g0 * out_v), axis=1)
            dp1 += tl.sum(_bf16_round(g1 * out_v), axis=1)
            dp2 += tl.sum(_bf16_round(g2 * out_v), axis=1)
            dp3 += tl.sum(_bf16_round(g3 * out_v), axis=1)
            # d_res[t, k, d] = sum_n comb[k, n] * g[n, d]  (MatMul-bwd, fp32 accumulation)
            dr0 = c00[:, None] * g0 + c01[:, None] * g1 + c02[:, None] * g2 + c03[:, None] * g3
            dr1 = c10[:, None] * g0 + c11[:, None] * g1 + c12[:, None] * g2 + c13[:, None] * g3
            dr2 = c20[:, None] * g0 + c21[:, None] * g1 + c22[:, None] * g2 + c23[:, None] * g3
            dr3 = c30[:, None] * g0 + c31[:, None] * g1 + c32[:, None] * g2 + c33[:, None] * g3
            tl.store(d_res_ptr + row + 0 * D, dr0.to(tl.bfloat16), mask=m2)
            tl.store(d_res_ptr + row + 1 * D, dr1.to(tl.bfloat16), mask=m2)
            tl.store(d_res_ptr + row + 2 * D, dr2.to(tl.bfloat16), mask=m2)
            tl.store(d_res_ptr + row + 3 * D, dr3.to(tl.bfloat16), mask=m2)
            # d_comb[t, k, n] = sum_d res[k, d] * g[n, d]  (MatMul-bwd, fp32 accumulation)
            q00 += tl.sum(r0 * g0, axis=1)
            q01 += tl.sum(r0 * g1, axis=1)
            q02 += tl.sum(r0 * g2, axis=1)
            q03 += tl.sum(r0 * g3, axis=1)
            q10 += tl.sum(r1 * g0, axis=1)
            q11 += tl.sum(r1 * g1, axis=1)
            q12 += tl.sum(r1 * g2, axis=1)
            q13 += tl.sum(r1 * g3, axis=1)
            q20 += tl.sum(r2 * g0, axis=1)
            q21 += tl.sum(r2 * g1, axis=1)
            q22 += tl.sum(r2 * g2, axis=1)
            q23 += tl.sum(r2 * g3, axis=1)
            q30 += tl.sum(r3 * g0, axis=1)
            q31 += tl.sum(r3 * g1, axis=1)
            q32 += tl.sum(r3 * g2, axis=1)
            q33 += tl.sum(r3 * g3, axis=1)
        tl.store(d_post_ptr + t_off * HC + 0, dp0.to(tl.bfloat16).to(d_post_ptr.dtype.element_ty), mask=t_mask)
        tl.store(d_post_ptr + t_off * HC + 1, dp1.to(tl.bfloat16).to(d_post_ptr.dtype.element_ty), mask=t_mask)
        tl.store(d_post_ptr + t_off * HC + 2, dp2.to(tl.bfloat16).to(d_post_ptr.dtype.element_ty), mask=t_mask)
        tl.store(d_post_ptr + t_off * HC + 3, dp3.to(tl.bfloat16).to(d_post_ptr.dtype.element_ty), mask=t_mask)
        qb = t_off * (HC * HC)
        tl.store(d_comb_ptr + qb + 0, q00.to(tl.bfloat16).to(d_comb_ptr.dtype.element_ty), mask=t_mask)
        tl.store(d_comb_ptr + qb + 1, q01.to(tl.bfloat16).to(d_comb_ptr.dtype.element_ty), mask=t_mask)
        tl.store(d_comb_ptr + qb + 2, q02.to(tl.bfloat16).to(d_comb_ptr.dtype.element_ty), mask=t_mask)
        tl.store(d_comb_ptr + qb + 3, q03.to(tl.bfloat16).to(d_comb_ptr.dtype.element_ty), mask=t_mask)
        tl.store(d_comb_ptr + qb + 4, q10.to(tl.bfloat16).to(d_comb_ptr.dtype.element_ty), mask=t_mask)
        tl.store(d_comb_ptr + qb + 5, q11.to(tl.bfloat16).to(d_comb_ptr.dtype.element_ty), mask=t_mask)
        tl.store(d_comb_ptr + qb + 6, q12.to(tl.bfloat16).to(d_comb_ptr.dtype.element_ty), mask=t_mask)
        tl.store(d_comb_ptr + qb + 7, q13.to(tl.bfloat16).to(d_comb_ptr.dtype.element_ty), mask=t_mask)
        tl.store(d_comb_ptr + qb + 8, q20.to(tl.bfloat16).to(d_comb_ptr.dtype.element_ty), mask=t_mask)
        tl.store(d_comb_ptr + qb + 9, q21.to(tl.bfloat16).to(d_comb_ptr.dtype.element_ty), mask=t_mask)
        tl.store(d_comb_ptr + qb + 10, q22.to(tl.bfloat16).to(d_comb_ptr.dtype.element_ty), mask=t_mask)
        tl.store(d_comb_ptr + qb + 11, q23.to(tl.bfloat16).to(d_comb_ptr.dtype.element_ty), mask=t_mask)
        tl.store(d_comb_ptr + qb + 12, q30.to(tl.bfloat16).to(d_comb_ptr.dtype.element_ty), mask=t_mask)
        tl.store(d_comb_ptr + qb + 13, q31.to(tl.bfloat16).to(d_comb_ptr.dtype.element_ty), mask=t_mask)
        tl.store(d_comb_ptr + qb + 14, q32.to(tl.bfloat16).to(d_comb_ptr.dtype.element_ty), mask=t_mask)
        tl.store(d_comb_ptr + qb + 15, q33.to(tl.bfloat16).to(d_comb_ptr.dtype.element_ty), mask=t_mask)


@triton.jit
def _hc_collapse_fwd_kernel(
    pre_ptr,
    res_ptr,
    out_ptr,
    T,
    D,
    PRE_ST_T: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    nprogs = tl.num_programs(0)
    for t_i in range(pid, T, nprogs):
        t = t_i.to(tl.int64)  # type: ignore[attr-defined] # triton: device-side loop index
        for d0 in range(0, D, BLOCK_D):
            d_off = d0 + tl.arange(0, BLOCK_D)
            mask = d_off < D
            acc = tl.zeros([BLOCK_D], dtype=tl.float32)
            # sequential k accumulation matches the eager Sum(dim=2) reduction order
            for k in tl.static_range(HC):
                pre_k = tl.load(pre_ptr + t * PRE_ST_T + k).to(tl.float32)
                res_v = tl.load(res_ptr + (t * HC + k) * D + d_off, mask=mask, other=0.0).to(tl.float32)
                acc += pre_k * res_v
            tl.store(out_ptr + t * D + d_off, acc.to(tl.bfloat16), mask=mask)


@triton.jit
def _hc_collapse_bwd_kernel(
    pre_ptr,
    res_ptr,
    g_ptr,
    d_pre_ptr,
    d_res_ptr,
    T,
    D,
    PRE_ST_T: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    nprogs = tl.num_programs(0)
    h_off = tl.arange(0, HC)
    for t_i in range(pid, T, nprogs):
        t = t_i.to(tl.int64)  # type: ignore[attr-defined] # triton: device-side loop index
        pre_v = tl.load(pre_ptr + t * PRE_ST_T + h_off).to(tl.float32)  # [HC]
        d_pre_acc = tl.zeros([HC], dtype=tl.float32)
        for d0 in range(0, D, BLOCK_D):
            d_off = d0 + tl.arange(0, BLOCK_D)
            mask = d_off < D
            g_v = tl.load(g_ptr + t * D + d_off, mask=mask, other=0.0).to(tl.float32)
            res_tile = tl.load(
                res_ptr + (t * HC + h_off[:, None]) * D + d_off[None, :], mask=mask[None, :], other=0.0
            ).to(tl.float32)  # [HC, BLOCK_D]
            # eager: fp32 promotion product rounded to bf16 at the stream-dtype cast
            tl.store(
                d_res_ptr + (t * HC + h_off[:, None]) * D + d_off[None, :],
                (pre_v[:, None] * g_v[None, :]).to(tl.bfloat16),
                mask=mask[None, :],
            )
            # eager: fp32 promotion product, fp32 accumulation (no bf16 rounding)
            d_pre_acc += tl.sum(g_v[None, :] * res_tile, axis=1)
        tl.store(d_pre_ptr + t * HC + h_off, d_pre_acc)


def _probe_max_progs(device: torch.device) -> int:
    """Verify the 48-program persistent grid fully executes (NaN sentinel).

    One-shot probe mirroring ``causal_conv1d_ascendc._conv_bwd_probe_max_progs``:
    launches each production-constexpr binary over exactly ``_MAX_PROGS``
    tiles / token rows (combine: ``_COMBINE_BLOCK_T * _MAX_PROGS`` tokens, one
    tile per program; collapse: one token row per program) with outputs
    prefilled with NaN and zero inputs. Under ``TRITON_ALL_BLOCKS_PARALLEL=1``
    the launcher clamps blockNum to the physical block count and silently drops
    the blocks past the clamp (aiv mode 48, aicore mode 24), so any surviving
    NaN means the binary does not fully execute at the cap and the launchers
    must fall back to ``_MAX_PROGS_FALLBACK``. Zero inputs keep every computed
    value finite -- NaN can only come from the sentinel.

    Args:
        device (torch.device): Device to probe on.

    Returns:
        int: ``_MAX_PROGS`` when all four binaries fully executed, else
        ``_MAX_PROGS_FALLBACK``.
    """
    progs = _MAX_PROGS
    t = _COMBINE_BLOCK_T * _MAX_PROGS
    d = _COMBINE_BLOCK_D
    # Mirror the *production* specialization exactly: POST_ST_T / COMB_ST_T /
    # COMB_ST_K are tl.constexpr and triton additionally specializes on pointer
    # element dtypes, so any mismatch compiles a *different* binary than the one
    # production launches. Production sees fp32 post/comb (the sinkhorn mixes
    # are fp32 on both norm paths) with comb as the padded [B, S, HC, 8]
    # buffer's ``[:, :, :, :HC]`` slice view: POST_ST_T=4, COMB_ST_T=32,
    # COMB_ST_K=8; out/res/new/d_out/d_res are bf16 and d_post/d_comb follow
    # post/comb's fp32. (The former bf16/packed probe -- POST_ST_T=1,
    # COMB_ST_T=4, COMB_ST_K=1 -- verified binaries production never ran,
    # leaving the real cap-48 grid unguarded.) Runtime ints T/D only specialize
    # on (==1, %16), which the probe values share with the training shape.
    pad_align = 8  # the sinkhorn forward's block_align: the padded comb row stride
    post = torch.zeros(1, t, _PROD_HC, dtype=torch.float32, device=device)
    comb_padded = torch.zeros(1, t, _PROD_HC, pad_align, dtype=torch.float32, device=device)
    comb = comb_padded[:, :, :, :_PROD_HC]
    feat = torch.zeros(t, d, dtype=torch.bfloat16, device=device)
    res = torch.zeros(t, _PROD_HC, d, dtype=torch.bfloat16, device=device)
    new = torch.full((t, _PROD_HC, d), float("nan"), dtype=torch.bfloat16, device=device)
    grid = (min(triton.cdiv(t, _COMBINE_BLOCK_T), progs),)
    _hc_combine_fwd_kernel[grid](
        post,
        comb,
        feat,
        res,
        new,
        t,
        d,
        POST_ST_T=post.stride(1),
        COMB_ST_T=comb.stride(1),
        COMB_ST_K=comb.stride(2),
        HC=_PROD_HC,
        BLOCK_T=_COMBINE_BLOCK_T,
        BLOCK_D=d,
    )
    if torch.isnan(new).any().item():
        return _MAX_PROGS_FALLBACK
    d_post = torch.full((t, _PROD_HC), float("nan"), dtype=torch.float32, device=device)
    d_comb = torch.full((t, _PROD_HC, _PROD_HC), float("nan"), dtype=torch.float32, device=device)
    d_out = torch.full((t, d), float("nan"), dtype=torch.bfloat16, device=device)
    d_res = torch.full((t, _PROD_HC, d), float("nan"), dtype=torch.bfloat16, device=device)
    _hc_combine_bwd_kernel[grid](
        post,
        comb,
        feat,
        res,
        torch.zeros_like(res),
        d_post,
        d_comb,
        d_out,
        d_res,
        t,
        d,
        POST_ST_T=post.stride(1),
        COMB_ST_T=comb.stride(1),
        COMB_ST_K=comb.stride(2),
        HC=_PROD_HC,
        BLOCK_T=_COMBINE_BLOCK_T,
        BLOCK_D=d,
    )
    if any(torch.isnan(x).any().item() for x in (d_post, d_comb, d_out, d_res)):
        return _MAX_PROGS_FALLBACK
    tc = _MAX_PROGS
    dc = _BLOCK_D
    pre = torch.zeros(tc, _PROD_HC, dtype=torch.float32, device=device)
    streams = torch.zeros(tc, _PROD_HC, dc, dtype=torch.bfloat16, device=device)
    cout = torch.full((tc, dc), float("nan"), dtype=torch.bfloat16, device=device)
    cgrid = (min(tc, progs),)
    _hc_collapse_fwd_kernel[cgrid](pre, streams, cout, tc, dc, PRE_ST_T=pre.stride(1), HC=_PROD_HC, BLOCK_D=dc)
    if torch.isnan(cout).any().item():
        return _MAX_PROGS_FALLBACK
    d_pre = torch.full((tc, _PROD_HC), float("nan"), dtype=torch.float32, device=device)
    d_streams = torch.full((tc, _PROD_HC, dc), float("nan"), dtype=torch.bfloat16, device=device)
    _hc_collapse_bwd_kernel[cgrid](
        pre,
        streams,
        torch.zeros_like(cout),
        d_pre,
        d_streams,
        tc,
        dc,
        PRE_ST_T=pre.stride(1),
        HC=_PROD_HC,
        BLOCK_D=dc,
    )
    if torch.isnan(d_pre).any().item() or torch.isnan(d_streams).any().item():
        return _MAX_PROGS_FALLBACK
    return progs


def _resolve_max_progs(device: torch.device) -> None:
    """Lazily fill ``_TRITON_MAX_PROGS[0]`` from the one-shot NaN-sentinel
    probe."""
    if _TRITON_MAX_PROGS[0] is None:
        try:
            _TRITON_MAX_PROGS[0] = _probe_max_progs(device)
        except Exception as exc:  # noqa: BLE001 - same blanket probe as the fused conv
            # Make the fallback reason visible: any probe failure (including a
            # transient first-call OOM) latches the conservative cap for the
            # whole process, and the engagement prints below do not distinguish
            # a probed cap from a fallback one.
            print(
                f"[glm53] max-progs probe failed ({exc!r}); latching fallback cap {_MAX_PROGS_FALLBACK}",
                flush=True,
            )
            _TRITON_MAX_PROGS[0] = _MAX_PROGS_FALLBACK


def _combine_eager(
    post: torch.Tensor,
    comb: torch.Tensor,
    out: torch.Tensor,
    res: torch.Tensor,
) -> torch.Tensor:
    """Eager reference: the decoder-layer stream-update chain verbatim."""
    dtype = res.dtype
    return post.to(dtype).unsqueeze(-1) * out.unsqueeze(-2) + torch.matmul(comb.to(dtype).transpose(-1, -2), res)


def _collapse_eager(pre: torch.Tensor, streams: torch.Tensor) -> torch.Tensor:
    """Eager reference: the hyper-connection pre-collapse chain verbatim."""
    return (pre.unsqueeze(-1) * streams).sum(dim=2).to(streams.dtype)


class _HcCombineFunction(torch.autograd.Function):
    """Fused combine forward/backward matching the eager autograd chain."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        post: torch.Tensor,
        comb: torch.Tensor,
        out: torch.Tensor,
        res: torch.Tensor,
    ) -> torch.Tensor:
        t_tokens = res.shape[0] * res.shape[1]
        hc = res.shape[2]
        feat = res.shape[3]
        out2 = out.reshape(t_tokens, feat)
        res2 = res.reshape(t_tokens, hc, feat)
        new = torch.empty_like(res2)
        _hc_combine_fwd_kernel[(min(triton.cdiv(t_tokens, _COMBINE_BLOCK_T), _TRITON_MAX_PROGS[0] or _MAX_PROGS),)](
            post,
            comb,
            out2,
            res2,
            new,
            t_tokens,
            feat,
            POST_ST_T=post.stride(1),
            COMB_ST_T=comb.stride(1),
            COMB_ST_K=comb.stride(2),
            HC=hc,
            BLOCK_T=_COMBINE_BLOCK_T,
            BLOCK_D=_COMBINE_BLOCK_D,
        )
        ctx.save_for_backward(post, comb, out2, res2)  # type: ignore[attr-defined]
        ctx.orig_shapes = (post.shape, comb.shape, out.shape, res.shape)  # type: ignore[attr-defined]
        return new.view_as(res)

    @staticmethod
    @once_differentiable
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx, grad_new: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        post2, comb2, out2, res2 = ctx.saved_tensors  # type: ignore[attr-defined]
        t_tokens, hc, feat = res2.shape
        g = grad_new.reshape(t_tokens, hc, feat).contiguous()
        d_post = torch.empty(t_tokens, hc, dtype=post2.dtype, device=res2.device)
        d_comb = torch.empty(t_tokens, hc, hc, dtype=comb2.dtype, device=res2.device)
        d_out = torch.empty(t_tokens, feat, dtype=res2.dtype, device=res2.device)
        d_res = torch.empty_like(res2)
        _hc_combine_bwd_kernel[(min(triton.cdiv(t_tokens, _COMBINE_BLOCK_T), _TRITON_MAX_PROGS[0] or _MAX_PROGS),)](
            post2,
            comb2,
            out2,
            res2,
            g,
            d_post,
            d_comb,
            d_out,
            d_res,
            t_tokens,
            feat,
            POST_ST_T=post2.stride(1),
            COMB_ST_T=comb2.stride(1),
            COMB_ST_K=comb2.stride(2),
            HC=hc,
            BLOCK_T=_COMBINE_BLOCK_T,
            BLOCK_D=_COMBINE_BLOCK_D,
        )
        post_shape, comb_shape, out_shape, res_shape = ctx.orig_shapes  # type: ignore[attr-defined]
        return (
            d_post.view(post_shape),
            d_comb.view(comb_shape),
            d_out.view(out_shape),
            d_res.view(res_shape),
        )


class _HcCollapseFunction(torch.autograd.Function):
    """Fused collapse forward/backward matching the eager autograd chain."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx, pre: torch.Tensor, streams: torch.Tensor
    ) -> torch.Tensor:
        t_tokens = streams.shape[0] * streams.shape[1]
        hc = streams.shape[2]
        feat = streams.shape[3]
        streams2 = streams.reshape(t_tokens, hc, feat)
        out = torch.empty(t_tokens, feat, dtype=streams.dtype, device=streams.device)
        _hc_collapse_fwd_kernel[(min(t_tokens, _TRITON_MAX_PROGS[0] or _MAX_PROGS),)](
            pre, streams2, out, t_tokens, feat, PRE_ST_T=pre.stride(1), HC=hc, BLOCK_D=_BLOCK_D
        )
        ctx.save_for_backward(pre, streams2)  # type: ignore[attr-defined]
        ctx.orig_shapes = (pre.shape, streams.shape)  # type: ignore[attr-defined]
        return out.view(streams.shape[0], streams.shape[1], feat)

    @staticmethod
    @once_differentiable
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pre2, streams2 = ctx.saved_tensors  # type: ignore[attr-defined]
        t_tokens, hc, feat = streams2.shape
        g = grad_out.reshape(t_tokens, feat).contiguous()
        d_pre = torch.empty(t_tokens, hc, dtype=pre2.dtype, device=streams2.device)
        d_res = torch.empty_like(streams2)
        _hc_collapse_bwd_kernel[(min(t_tokens, _TRITON_MAX_PROGS[0] or _MAX_PROGS),)](
            pre2, streams2, g, d_pre, d_res, t_tokens, feat, PRE_ST_T=pre2.stride(1), HC=hc, BLOCK_D=_BLOCK_D
        )
        pre_shape, streams_shape = ctx.orig_shapes  # type: ignore[attr-defined]
        return d_pre.view(pre_shape), d_res.view(streams_shape)


def _fold_strided(t: torch.Tensor) -> bool:
    """``[B, S, ...]`` fold-consistent strides with a unit innermost axis.

    Covers both the contiguous fresh buffers and the padded / split views the
    mHC producers hand out (``comb`` is a ``[.., HC, 8]`` slice view); folding
    ``B`` into ``S`` needs ``stride(0) == S * stride(1)``.
    """
    return t.dim() >= 2 and t.stride(-1) == 1 and t.stride(0) == t.shape[1] * t.stride(1)


def _combine_servable(post: torch.Tensor, comb: torch.Tensor, out: torch.Tensor, res: torch.Tensor) -> bool:
    """Fused-path eligibility: NPU, hc_mult=4, bf16 streams, fold-consistent strides."""
    return (
        _NPU_AVAILABLE
        and res.device.type == "npu"
        and res.shape[2] == _PROD_HC
        and res.dtype == torch.bfloat16
        and post.dtype in (torch.float32, torch.bfloat16)
        and comb.dtype in (torch.float32, torch.bfloat16)
        and out.dtype == torch.bfloat16
        and _fold_strided(post)
        and _fold_strided(comb)
        and out.is_contiguous()
        and res.is_contiguous()
    )


def _collapse_servable(pre: torch.Tensor, streams: torch.Tensor) -> bool:
    """Fused-path eligibility: NPU, hc_mult=4, bf16 streams, fp32/bf16 pre.

    The production ``pre`` is the sinkhorn split output (fp32 on both the eager
    and fused paths -- ``mixes`` is fp32 through ``hc_pre``); the kernels upcast
    it to fp32 in-register, so both dtypes serve (bf16 appears in the unit-test
    config only).
    """
    return (
        _NPU_AVAILABLE
        and streams.device.type == "npu"
        and streams.shape[2] == _PROD_HC
        and streams.dtype == torch.bfloat16
        and pre.dtype in (torch.float32, torch.bfloat16)
        and _fold_strided(pre)
        and streams.is_contiguous()
    )


def hc_combine(
    post: torch.Tensor,
    comb: torch.Tensor,
    out: torch.Tensor,
    res: torch.Tensor,
) -> torch.Tensor:
    """Apply the mHC stream update ``new = post ⊙ out + combᵀ @ res``.

    Args:
        post (torch.Tensor): Stream placement weights ``[B, S, H]``.
        comb (torch.Tensor): Stream mixer matrix ``[B, S, H, H]`` (indexed ``comb[k, n]``).
        out (torch.Tensor): Sublayer output ``[B, S, D]``.
        res (torch.Tensor): Residual streams ``[B, S, H, D]``.

    Returns:
        torch.Tensor: Updated streams ``[B, S, H, D]``.
    """
    if not (HC_COMBINE_FUSED and _combine_servable(post, comb, out, res)):
        return _combine_eager(post, comb, out, res)
    _resolve_max_progs(res.device)
    if not _LOG_STATE["combine"]:
        _LOG_STATE["combine"] = True
        print(f"[glm53] hc_combine fused engaged: token-tiled kernels, max_progs={_TRITON_MAX_PROGS[0]}", flush=True)
    return _HcCombineFunction.apply(post, comb, out, res)


def hc_collapse(pre: torch.Tensor, streams: torch.Tensor) -> torch.Tensor:
    """Collapse streams into the sublayer input ``collapsed = Σₖ preₖ ·
    streamₖ``.

    Args:
        pre (torch.Tensor): Collapse weights ``[B, S, H]`` (fp32 mHC mixes).
        streams (torch.Tensor): Residual streams ``[B, S, H, D]``.

    Returns:
        torch.Tensor: Collapsed sublayer input ``[B, S, D]``.
    """
    if not (HC_COLLAPSE_FUSED and _collapse_servable(pre, streams)):
        return _collapse_eager(pre, streams)
    _resolve_max_progs(streams.device)
    if not _LOG_STATE["collapse"]:
        _LOG_STATE["collapse"] = True
        print(f"[glm53] hc_collapse fused engaged: max_progs={_TRITON_MAX_PROGS[0]}", flush=True)
    return _HcCollapseFunction.apply(pre, streams)


# ------------------------------------------------------------------------------
# -- Eager hc_post in the pre-merge matmul form
# ------------------------------------------------------------------------------
@maybe_compile
def _hc_post_eager_matmul(x: Tensor, residual: Tensor, post: Tensor, comb: Tensor) -> Tensor:
    """NPU eager ``hc_post`` in the pre-merge implementation's matmul form.

    The HF-matching broadcast expression in
    :func:`xtuner.v1.module.decoder_layer.mhc._hc_post_eager` evaluates the ``comb x residual``
    mix through a broadcast-multiply + reduce-sum and never fuses on this tree (the Triton
    kernel is CUDA-only and BiShengIR cannot compile it on 910C), so every call materializes a
    ``[B, S, H, H, D]`` fp32 intermediate -- 1.07 GB per call at the 30B training shape
    (``S=8192, H=4, D=2048``), ~14 s/step of pure Mul/BroadcastTo/ReduceSum traffic in the
    128K profile, with the same again in backward. The pre-merge GLM-5.3 implementation
    (which trained this model end to end on 910C, run81 anchor) evaluated the identical math
    as ``post * x + matmul(comb^T, residual)``: batched-matmul (K=H, tensor-core) with only
    ``[B, S, H, D]`` intermediates, ~16x less traffic.

    Precision: bf16 matmul rounds the mixed result once (fp32 accumulate inside the kernel)
    where the HF chain rounds after the fp32 sum; the fp32-accumulate order along H also
    differs. This is the pre-merge implementation's approved fused-op precision exception --
    the 128K anchor losses agree to 1.9e-4 across the two forms (run81 12.14300537 vs
    run92 12.14281273).
    """
    return post.to(residual.dtype).unsqueeze(-1) * x.unsqueeze(-2) + torch.matmul(
        comb.to(residual.dtype).transpose(-1, -2), residual
    )
