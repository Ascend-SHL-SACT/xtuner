# Copyright © 2026 Huawei Technologies Co., Ltd.
"""Causal depthwise conv1d over varlen packed rows (fla_npu Ascend C op).

Ported from xtuner-qwen ``runkit/q35_npu.py`` (the fused conv path:
``_CausalConv1dFused`` / ``_conv_fwd_aclnn`` / ``_conv_bwd_aclnn`` and the
one-shot probe). One fused varlen Ascend C op per direction over the **full
channel width**: ``x`` stays dim-last ``[T, C]`` end to end (``head_num=0``),
document boundaries come from the host ``cu`` list, silu runs outside the
forward kernel and inside the backward kernel. No per-slice launches, no
head-major layout round trips.

The qwen tree reaches the kernel through ``torch.ops.npu.npu_causal_conv1d{,_bwd}``;
this box's fla_npu wheel does not register that legacy path, so the calls are
mapped to the wheel's public direct APIs ``npu_causal_conv1d_fn`` /
``npu_causal_conv1d_bwd`` -- the same Ascend C kernel with the same parameter
semantics (probe-verified: fwd bit-exact vs eager, bwd at bf16 noise level).

Backward engines (in preference order): a triton tile kernel (``XTUNER_GLM53_CONV_BWD``
default), the vectorised eager shift-MAD path, and the fla_npu Ascend C bwd op.
The Ascend C bwd kernel degrades ~8x with segment count (1.7ms at 1 segment ->
15ms at the production ~1500 segments, 55 GFLOPS); the triton kernel is
segment-count insensitive, streams the row window once and reduces the fp32
partial ``dw`` in a second small sum, with a tile-sized footprint instead of
the eager path's four full-row temporaries.
"""

from __future__ import annotations

import os

import fla_npu  # noqa: F401 - registers the wheel's Ascend C ops
import torch
import torch.nn.functional as F
from fla_npu.ops.ascendc import npu_causal_conv1d_bwd, npu_causal_conv1d_fn


try:
    import triton
    import triton.language as tl

    _TRITON_IMPORT_OK = True
except ImportError:  # pragma: no cover - CPU-only / triton-less environments
    _TRITON_IMPORT_OK = False


__all__ = ["causal_conv1d_ascendc"]

# The Ascend C kernel caps the width at 4 (causal_conv1d_common.h MAX_WIDTH).
_MAX_WIDTH = 4

# Backward engine: "triton" (tile kernel, default), "eager" (vectorised
# shift-MAD ops) or "fla_npu"/"fla"/"tbe"/"0"/"off" (the Ascend C kernel).
_CONV_BWD = os.environ.get("XTUNER_GLM53_CONV_BWD", "triton").lower()
_CONV_BWD_TRITON = _TRITON_IMPORT_OK and _CONV_BWD in ("", "triton")
# Non-fla vectorised path master switch (covers both triton and eager).
_CONV_BWD_EAGER = _CONV_BWD not in ("fla_npu", "fla", "tbe", "0", "off")

_FUSED_STATE: dict[str, bool] = {"probed": False, "ok": False}
_SILU_BWD_STATE: dict[str, bool] = {"probed": False, "ok": False}
_TRITON_STATE: dict[str, bool] = {"probed": False, "ok": False}
# Persistent-grid cap resolved by the NaN-sentinel probe in
# ``_conv_bwd_triton_available``; ``None`` falls back to ``_CONV_BWD_MAX_PROGS``.
_TRITON_MAX_PROGS: list[int | None] = [None]


def causal_conv1d_ascendc(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    cu_seqlens: list[int] | None = None,
) -> torch.Tensor | None:
    """Fused causal depthwise conv1d on a packed row.

    Args:
        x (torch.Tensor): Packed conv input ``[1, T, C]`` (channel-last).
        weight (torch.Tensor): Depthwise conv weight ``[C, W]`` (torch layout).
        bias (torch.Tensor | None): Optional conv bias; the fused path does not
            take one, so a non-``None`` bias disables it.
        activation (str | None): ``None`` or ``"silu"``/``"swish"``.
        cu_seqlens (list[int] | None): Cumulative packed-sequence lengths;
            ``None`` treats the row as one sequence.

    Returns:
        torch.Tensor | None: Convolved ``[1, T, C]`` row, or ``None`` when the
        fused path cannot serve the request (bias set, width > 4, batched /
        non-3D input, or the one-shot probe failed) -- the caller then runs
        its eager fallback.
    """
    if activation not in (None, "silu", "swish"):
        raise ValueError(f"Unsupported causal conv activation: {activation}")
    if x.ndim != 3 or x.shape[0] != 1:
        return None
    width = weight.shape[-1]
    if bias is not None or width > _MAX_WIDTH:
        return None
    if not _fused_conv_available(x.device):
        return None
    bounds = [0, x.shape[1]] if cu_seqlens is None else [int(v) for v in cu_seqlens]
    return _CausalConv1dFused.apply(x, weight, bounds, activation)


class _CausalConv1dFused(torch.autograd.Function):
    """fla_npu Ascend C causal_conv1d: one fused varlen op per direction.

    The Ascend C kernel masks document boundaries natively through the host
    ``cu`` bounds (per-sequence zero history). silu runs inside the Function
    like the fla reference: the backward hands the saved pre-activation to the
    bwd op with ``activation=1`` and the op applies ``silu_backward``
    internally.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bounds: list[int],
        activation: str | None,
    ) -> torch.Tensor:
        op_weight = weight.transpose(-1, -2).contiguous()
        op_x = x.reshape(-1, x.shape[-1]).contiguous()
        pre = _conv_fwd_aclnn(op_x, op_weight, bounds)
        ctx.save_for_backward(op_x, op_weight, pre)
        ctx.bounds = bounds
        ctx.activation = activation
        ctx.x_shape = tuple(x.shape)
        if activation in ("silu", "swish"):
            pre = F.silu(pre)
        return pre.reshape(ctx.x_shape)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        op_x, op_weight, pre = ctx.saved_tensors
        # NTD dy may carry a head-split leading dim ([N, T, Dh]); [1, T, C] is
        # the identity split for a channel-independent depthwise conv.
        op_dy = dy.reshape(1, -1, dy.shape[-1]).contiguous()
        activation = 1 if ctx.activation in ("silu", "swish") else 0
        if _CONV_BWD_EAGER:
            if _CONV_BWD_TRITON and _conv_bwd_triton_available(op_x.device):
                out = _conv_bwd_triton(
                    op_x,
                    pre.unsqueeze(0) if activation else None,
                    op_weight,
                    op_dy,
                    ctx.bounds,
                    activation=activation,
                )
                if out is not None:
                    dx, dw = out
                    return dx.reshape(ctx.x_shape), dw.transpose(0, 1).contiguous(), None, None
            eager = _conv_bwd_eager(
                op_x,
                pre.unsqueeze(0) if activation else None,
                op_weight,
                op_dy,
                ctx.bounds,
                activation=activation,
            )
            if eager is not None:
                dx, dw = eager
                return dx.reshape(ctx.x_shape), dw.transpose(0, 1).contiguous(), None, None
        dx, dw = _conv_bwd_aclnn(
            op_x,
            pre.unsqueeze(0) if activation else None,
            op_weight,
            op_dy,
            ctx.bounds,
            activation=activation,
        )
        return dx.reshape(ctx.x_shape), dw.transpose(0, 1).contiguous(), None, None


def _conv_fwd_aclnn(x2d: torch.Tensor, weight_kc: torch.Tensor, cu: list[int]) -> torch.Tensor:
    """Ascend C causal conv forward.

    Args:
        x2d (torch.Tensor): Flattened input ``[T, C]``.
        weight_kc (torch.Tensor): Conv weight in the kernel-native ``[W, C]``
            layout (transposed from the torch ``[C, W]`` layout).
        cu (list[int]): Host list of per-sequence token starts plus the total
            (``[0, s1, s2, ..., T]``).

    Returns:
        torch.Tensor: Pre-activation ``[T, C]`` (activation is applied by the
        caller).
    """
    return npu_causal_conv1d_fn(x2d, weight_kc, None, None, query_start_loc_cpu=cu, activation="none")


def _conv_bwd_aclnn(
    x2d: torch.Tensor,
    y3d: torch.Tensor | None,
    weight_kc: torch.Tensor,
    dy3d: torch.Tensor,
    cu: list[int],
    activation: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ascend C causal conv backward (NTD layout).

    Args:
        x2d (torch.Tensor): Flattened forward input ``[T, C]``.
        y3d (torch.Tensor | None): Saved pre-activation ``[1, T, C]`` when
            ``activation=1`` (the op applies ``silu_backward`` internally);
            ``None`` with ``activation=0``.
        weight_kc (torch.Tensor): Conv weight in the kernel-native ``[W, C]``
            layout.
        dy3d (torch.Tensor): Upstream gradient ``[1, T, C]``.
        cu (list[int]): Host per-sequence bounds, as in ``_conv_fwd_aclnn``.
        activation (int): ``1`` when silu was applied in the forward, else
            ``0``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``dx`` ``[T, C]`` and ``dw`` in the
        kernel-native ``[W, C]`` layout (transpose for the torch ``[C, W]``).
    """
    dx, dw, _db, _dh0 = npu_causal_conv1d_bwd(
        x2d,
        y3d,
        weight_kc,
        dy3d,
        query_start_loc=cu,
        activation=activation,
        input_layout="NTD",
    )
    return dx, dw


def _conv_bwd_eager(
    x2d: torch.Tensor,
    y3d: torch.Tensor | None,
    weight_kc: torch.Tensor,
    dy3d: torch.Tensor,
    cu: list[int],
    activation: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Vectorised eager causal conv backward (NTD layout, same contract as
    ``_conv_bwd_aclnn``).

    The depthwise causal conv is 4 shifted multiply-adds per direction: the
    upstream gradient is pushed through ``silu_backward`` once, then ``dx``
    gathers the shifted gradient rows (tap ``k`` of position ``t`` reads
    position ``t+k``) and ``dw[k]`` column-sums the shifted input-gradient
    products. Cross-segment taps are masked with a per-token segment id; a
    single-segment row skips the masks entirely (the packed 128K production
    row carries ~1500 segments, so the masked path is the hot one).

    Not bitwise against the Ascend C kernel: the sigmoid implementation and
    the ``dw`` reduction order differ (approved precision exception).

    Args:
        x2d (torch.Tensor): Flattened forward input ``[T, C]``.
        y3d (torch.Tensor | None): Saved pre-activation ``[1, T, C]`` when
            ``activation=1``; ``None`` with ``activation=0``.
        weight_kc (torch.Tensor): Conv weight in the kernel-native ``[W, C]``
            layout.
        dy3d (torch.Tensor): Upstream gradient ``[1, T, C]``.
        cu (list[int]): Host per-sequence bounds, as in ``_conv_fwd_aclnn``.
        activation (int): ``1`` when silu was applied in the forward, else
            ``0``.

    Returns:
        tuple[torch.Tensor, torch.Tensor] | None: ``dx`` ``[T, C]`` and ``dw``
        in the kernel-native ``[W, C]`` layout, or ``None`` when the row
        shape cannot be served (segment bounds not covering ``T``) -- the
        caller then falls back to ``_conv_bwd_aclnn``.
    """
    total, channels = x2d.shape
    dy2d = dy3d.reshape(total, channels)
    # Tap order: the Ascend C kernel matches F.conv1d cross-correlation, where
    # weight row j pairs input lag (W-1-j) (probe-verified against the fwd op).
    # Flip once so rows are lag-indexed, flip dw back on return.
    w_rev = weight_kc.flip(0)
    if activation:
        pre = y3d.reshape(total, channels)  # type: ignore[union-attr]
        if _silu_bwd_available(x2d.device):
            dsilu = torch.ops.aten.silu_backward(dy2d, pre)
        else:
            sig = torch.sigmoid(pre)
            dsilu = dy2d * sig * (1 + pre * (1 - sig))
    else:
        dsilu = dy2d

    if cu is not None and len(cu) > 2:
        lens = [b - a for a, b in zip(cu[:-1], cu[1:])]
        if min(lens) < 0 or sum(lens) != total:
            return None
        seg_id = _cached_segment_ids(lens, x2d.device)
    else:
        seg_id = None

    dx = dsilu * w_rev[0]
    dw_rows = [(x2d * dsilu).sum(0)]
    for k in range(1, w_rev.shape[0]):
        if total <= k:
            dw_rows.append(torch.zeros_like(w_rev[0]))
            continue
        if seg_id is not None:
            same = seg_id[:-k] == seg_id[k:]
            shifted = torch.where(
                same.unsqueeze(-1), dsilu[k:], torch.zeros((), dtype=dsilu.dtype, device=dsilu.device)
            )
        else:
            shifted = dsilu[k:]
        dx[:-k] += shifted * w_rev[k]
        dw_rows.append((x2d[:-k] * shifted).sum(0))
    return dx, torch.stack(dw_rows).flip(0)


if _TRITON_IMPORT_OK:

    @triton.jit
    def _conv_bwd_kernel(
        x_ptr,
        pre_ptr,
        w_ptr,
        dy_ptr,
        dx_ptr,
        seg_ptr,
        pdw_ptr,
        T,
        C,
        W: tl.constexpr,
        ACT: tl.constexpr,
        HAS_SEG: tl.constexpr,
        BT: tl.constexpr,
        BC: tl.constexpr,
    ):
        # Persistent grid: one program strides over the [BT, BC] tiles instead
        # of the 2D (t_tiles, c_tiles) grid -- at 64K tokens the 2D grid hosts
        # thousands of programs, the TRITON_ALL_BLOCKS_PARALLEL blockify-replay
        # transform kicks in (blockNum clamped to the 25/50 physical cores) and
        # deterministically miscompiles this kernel in multi-rank training
        # (grad_norm nan, 0924-box repro; TRITON=0 / conv-bwd=eager healthy).
        # Same failure mode, same guard as hc_norm_linear's persistent grid.
        # The per-tile body is unchanged and pdw rows keep the logical tile
        # index, so the host-side sum(0) reduces the identical per-tile fp32
        # partials in the identical order -- dw stays bitwise the same.
        # Tile budget: BT*BC must stay <= 4096 elements -- 8192 (64x128) fails
        # BiShengIR compilation once ACT + segment masks + per-tap reductions
        # coexist (resource limit, probe-verified); 32x128 is the bench optimum.
        pid = tl.program_id(0)
        nprogs = tl.num_programs(0)
        n_c_tiles = C // BC
        total_tiles = tl.cdiv(T, BT) * n_c_tiles
        for tile in range(pid, total_tiles, nprogs):
            pid_t = tile // n_c_tiles
            pid_c = tile % n_c_tiles
            t_off = pid_t * BT + tl.arange(0, BT)
            c_off = pid_c * BC + tl.arange(0, BC)
            t_mask = t_off < T
            c_mask = c_off < C
            m2 = t_mask[:, None] & c_mask[None, :]
            t64 = t_off[:, None].to(tl.int64) * C

            x = tl.load(x_ptr + t64 + c_off[None, :], mask=m2, other=0.0).to(tl.float32)
            dy = tl.load(dy_ptr + t64 + c_off[None, :], mask=m2, other=0.0).to(tl.float32)
            pre = tl.load(pre_ptr + t64 + c_off[None, :], mask=m2, other=0.0)
            if ACT:
                sig = 1.0 / (1.0 + tl.exp(-pre.to(tl.float32)))
                dsilu = dy * sig * (1.0 + pre.to(tl.float32) * (1.0 - sig))
            else:
                dsilu = dy
            w0 = tl.load(w_ptr + c_off, mask=c_mask, other=0.0).to(tl.float32)
            dx = dsilu * w0[None, :]
            # Per-tap dw partials reduce the [BT, BC] product along t. The direct
            # axis=0 reduction compiles on the current BiShengIR (probe-verified,
            # 2026-10-08) and is ~22-25% faster per call than the prior
            # ``tl.trans`` + ``axis=1`` workaround, which had been needed when the
            # 4096-element tile exhausted the older compiler's reduction resource.
            # dx is bitwise identical to the trans form; dw differs by at most one
            # bf16 ULP (fp32 reduction-tree change) -- within the kernel's existing
            # approved precision exception versus the eager bf16 column sums.
            tl.store(pdw_ptr + (pid_t * W).to(tl.int64) * C + c_off, tl.sum(x * dsilu, axis=0), mask=c_mask)
            if HAS_SEG:
                seg_t = tl.load(seg_ptr + t_off, mask=t_mask, other=-1)
            for k in tl.static_range(1, W):
                tk = t_off + k
                vk = tk < T
                if HAS_SEG:
                    seg_k = tl.load(seg_ptr + tk, mask=vk, other=-2)
                    vk = vk & (seg_k == seg_t)
                mk = vk[:, None] & c_mask[None, :]
                dyk = tl.load(dy_ptr + (tk[:, None].to(tl.int64) * C) + c_off[None, :], mask=mk, other=0.0).to(
                    tl.float32
                )
                prek = tl.load(pre_ptr + (tk[:, None].to(tl.int64) * C) + c_off[None, :], mask=mk, other=0.0)
                if ACT:
                    sigk = 1.0 / (1.0 + tl.exp(-prek.to(tl.float32)))
                    dsk = dyk * sigk * (1.0 + prek.to(tl.float32) * (1.0 - sigk))
                else:
                    dsk = dyk
                wk = tl.load(w_ptr + k * C + c_off, mask=c_mask, other=0.0).to(tl.float32)
                dx += dsk * wk[None, :]
                tl.store(pdw_ptr + (pid_t * W + k).to(tl.int64) * C + c_off, tl.sum(x * dsk, axis=0), mask=c_mask)
            tl.store(dx_ptr + t64 + c_off[None, :], dx.to(dx_ptr.dtype.element_ty), mask=m2)


_CONV_BT = 32  # BT*BC <= 4096: 64x128 exceeds the BiShengIR tile resource limit (see kernel comment)
_CONV_BC = 128
# Persistent-grid program cap. Under TRITON_ALL_BLOCKS_PARALLEL=1 the ascend
# launcher clamps blockNum to the physical block count and silently drops the
# blocks past the clamp (see kda_op._SEG_COPY_MAX_PROGS): 48 in aiv mode, 24 in
# aicore mode, per-binary from the load/store profile. This kernel is pure
# AI-vector work and measures as aiv, so 48 blocks all run -- double the old
# 24-block cap, which serialized the kernel to exactly half the machine (8.9 ms
# -> 4.6 ms per 8192x12288 call). Because the mode is a backend decision, the
# one-shot probe in ``_conv_bwd_triton_available`` verifies full execution at
# the cap with a NaN-sentinel launch on first use and falls back to 24 if a
# future backend ever compiles the kernel aicore-mode.
_CONV_BWD_MAX_PROGS = 48
_CONV_BWD_MAX_PROGS_FALLBACK = 24

# seg_id is a pure function of the per-sequence lengths, and the same lengths recur on every
# backward call of a micro-batch -- rebuilding it per call cost 3 device ops (arange, H2D
# tensor, repeat_interleave) plus ~1.07M python generator frames per training step at 128K
# (the ``any(ln < 0 ...)`` scan). Bounded: cleared wholesale when full, which only happens if
# segment shapes vary without bound.
_CONV_SEG_ID_CACHE: dict[tuple[tuple[int, ...], torch.device], torch.Tensor] = {}
_CONV_SEG_ID_CACHE_MAX = 64


def _cached_segment_ids(lens: list[int], device: torch.device) -> torch.Tensor:
    key = (tuple(lens), device)
    seg_id = _CONV_SEG_ID_CACHE.get(key)
    if seg_id is None:
        if len(_CONV_SEG_ID_CACHE) >= _CONV_SEG_ID_CACHE_MAX:
            _CONV_SEG_ID_CACHE.clear()
        seg_id = torch.repeat_interleave(
            torch.arange(len(lens), device=device, dtype=torch.int32),
            torch.tensor(lens, device=device, dtype=torch.int64),
        )
        _CONV_SEG_ID_CACHE[key] = seg_id
    return seg_id


def _conv_bwd_triton(
    x2d: torch.Tensor,
    y3d: torch.Tensor | None,
    weight_kc: torch.Tensor,
    dy3d: torch.Tensor,
    cu: list[int],
    activation: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Triton tile causal conv backward (same contract as ``_conv_bwd_aclnn``).

    fp32 tap products and fp32 per-tile ``dw`` partials reduced by a host-side
    ``sum(0)`` -- more accurate than the eager bf16 column sums, not bitwise
    against either prior engine (approved precision exception).

    Args:
        x2d (torch.Tensor): Flattened forward input ``[T, C]``.
        y3d (torch.Tensor | None): Saved pre-activation ``[1, T, C]`` when
            ``activation=1``; ``None`` with ``activation=0``.
        weight_kc (torch.Tensor): Conv weight in the kernel-native ``[W, C]``
            layout.
        dy3d (torch.Tensor): Upstream gradient ``[1, T, C]``.
        cu (list[int]): Host per-sequence bounds.
        activation (int): ``1`` when silu was applied in the forward, else ``0``.

    Returns:
        tuple[torch.Tensor, torch.Tensor] | None: ``dx`` ``[T, C]`` and ``dw``
        in the kernel-native ``[W, C]`` layout, or ``None`` when the row shape
        cannot be served (segment bounds not covering ``T``) -- the caller then
        falls back to the eager path.
    """
    total, channels = x2d.shape
    if channels % _CONV_BC != 0:
        return None
    if cu is not None and len(cu) > 2:
        lens = [b - a for a, b in zip(cu[:-1], cu[1:])]
        if min(lens) < 0 or sum(lens) != total:
            return None
        seg_id = _cached_segment_ids(lens, x2d.device)
    else:
        seg_id = None
    width = weight_kc.shape[0]
    dy2d = dy3d.reshape(total, channels)
    pre = (y3d.reshape(total, channels) if activation else dy2d).contiguous()  # type: ignore[union-attr]
    w_rev = weight_kc.flip(0).contiguous()
    n_t_tiles = triton.cdiv(total, _CONV_BT)
    partial = torch.empty((n_t_tiles, width, channels), dtype=torch.float32, device=x2d.device)
    dx = torch.empty((total, channels), dtype=x2d.dtype, device=x2d.device)
    _conv_bwd_kernel[(min(n_t_tiles * (channels // _CONV_BC), _TRITON_MAX_PROGS[0] or _CONV_BWD_MAX_PROGS),)](
        x2d,
        pre,
        w_rev,
        dy2d,
        dx,
        seg_id if seg_id is not None else x2d,
        partial,
        total,
        channels,
        W=width,
        ACT=activation,
        HAS_SEG=seg_id is not None,
        BT=_CONV_BT,
        BC=_CONV_BC,
        num_warps=4,
    )
    dw = partial.sum(0).to(x2d.dtype)
    return dx, dw.flip(0)


def _conv_bwd_probe_max_progs(device: torch.device, has_seg: bool) -> int:
    """Verify the cap-sized persistent grid fully executes (NaN sentinel).

    Launches a production-constexpr binary (``W=4 ACT=1 BT=32 BC=128``) over 64
    ``[BT, BC]`` tiles -- more tiles than the cap, so full coverage requires
    every launched program -- with ``dx`` prefilled with NaN. Under
    ``TRITON_ALL_BLOCKS_PARALLEL=1`` the launcher clamps blockNum to the
    physical block count and silently drops the blocks past the clamp (aiv mode
    48, aicore mode 24; see ``kda_op._SEG_COPY_MAX_PROGS``), so a dropped
    program leaves its tiles unwritten and any surviving NaN means the cap
    exceeds what this binary actually executes. Zero inputs keep every computed
    value finite -- NaN can only come from the sentinel.

    Args:
        device (torch.device): Device to probe on.
        has_seg (bool): ``HAS_SEG`` constexpr of the binary to probe (production
            is single-segment ``False``; multi-segment callers compile a second
            binary whose execution mode the backend decides independently).

    Returns:
        int: ``_CONV_BWD_MAX_PROGS`` when all tiles were written, else
        ``_CONV_BWD_MAX_PROGS_FALLBACK`` (the long-running production cap).
    """
    t = 32 * _CONV_BT
    c = 2 * _CONV_BC
    x = torch.zeros((t, c), dtype=torch.bfloat16, device=device)
    dy = torch.zeros_like(x)
    pre = torch.zeros_like(x)
    w = torch.zeros((4, c), dtype=torch.bfloat16, device=device)
    dx = torch.full((t, c), float("nan"), dtype=torch.bfloat16, device=device)
    seg = torch.zeros((t,), dtype=torch.int32, device=device) if has_seg else x
    partial = torch.empty((32, 4, c), dtype=torch.float32, device=device)
    _conv_bwd_kernel[(min(32 * 2, _CONV_BWD_MAX_PROGS),)](
        x,
        pre,
        w,
        dy,
        dx,
        seg,
        partial,
        t,
        c,
        W=4,
        ACT=1,
        HAS_SEG=has_seg,
        BT=_CONV_BT,
        BC=_CONV_BC,
        num_warps=4,
    )
    return _CONV_BWD_MAX_PROGS if not bool(torch.isnan(dx).any().item()) else _CONV_BWD_MAX_PROGS_FALLBACK


def _conv_bwd_triton_available(device: torch.device) -> bool:
    """Lazy one-shot probe of the triton backward (compile + small launch + cap check)."""
    if not _TRITON_STATE["probed"]:
        _TRITON_STATE["probed"] = True
        if not _TRITON_IMPORT_OK:
            return _TRITON_STATE["ok"]
        try:
            x = torch.randn(70, 256, dtype=torch.bfloat16, device=device)
            w = torch.randn(4, 256, dtype=torch.bfloat16, device=device)
            y = torch.randn(1, 70, 256, dtype=torch.bfloat16, device=device)
            out = _conv_bwd_triton(x, y, w, y, [0, 31, 31, 70], activation=1)
            _TRITON_STATE["ok"] = out is not None and out[0].shape == x.shape and out[1].shape == w.shape
            if _TRITON_STATE["ok"]:
                # Resolve the persistent-grid cap on the exact constexpr binaries
                # production launches (single- and multi-segment); the backend
                # picks the execution mode per binary, so probe both.
                _TRITON_MAX_PROGS[0] = min(
                    _conv_bwd_probe_max_progs(device, has_seg=False),
                    _conv_bwd_probe_max_progs(device, has_seg=True),
                )
        except Exception:  # noqa: BLE001 - same blanket probe as the fused conv
            _TRITON_STATE["ok"] = False
    return _TRITON_STATE["ok"]


def _silu_bwd_available(device: torch.device) -> bool:
    """Lazy one-shot probe of the fused ``aten.silu_backward`` dispatch."""
    if not _SILU_BWD_STATE["probed"]:
        _SILU_BWD_STATE["probed"] = True
        try:
            z = torch.randn(64, 128, dtype=torch.bfloat16, device=device)
            g = torch.ops.aten.silu_backward(z, z)
            _SILU_BWD_STATE["ok"] = g.shape == z.shape
        except Exception:  # noqa: BLE001 - same blanket probe as the fused conv
            _SILU_BWD_STATE["ok"] = False
    return _SILU_BWD_STATE["ok"]


def _fused_conv_available(device: torch.device) -> bool:
    """Lazy one-shot probe of the fused conv path.

    Loads the fla_npu ops on first use and exercises forward and backward on
    small tensors; any failure (extension absent, tiling rejection) marks the
    fused path off for the process lifetime, so the caller falls back to its
    eager conv instead of failing mid-training.

    Args:
        device (torch.device): Device to probe on.

    Returns:
        bool: Whether the fused path is usable.
    """
    if not _FUSED_STATE["probed"]:
        _FUSED_STATE["probed"] = True
        try:
            x = torch.randn(64, 128, dtype=torch.bfloat16, device=device)
            w = torch.randn(4, 128, dtype=torch.bfloat16, device=device)
            y = _conv_fwd_aclnn(x, w, [0, 64])
            # Exercise the backward too (the raw-gradient, no-y route): the op
            # exists but the bwd tiling can in principle reject where the fwd
            # passed; better to learn it here than mid-training. Launch-level
            # failures (op absent, argument/tiling rejection) raise on the
            # host during the call; no device synchronize and no finite check
            # -- either would put a sync point in the middle of training.
            dx, dw = _conv_bwd_aclnn(x, None, w, y.unsqueeze(0), [0, 64], activation=0)
            assert y.shape == (64, 128) and dx.shape == x.shape and dw.shape == w.shape
            _FUSED_STATE["ok"] = True
            # Label reflects the actual dispatch engine, not just the non-fla
            # master switch (which covers both triton and eager): env=triton
            # with the probe ok resolves to triton, not eager.
            if _CONV_BWD_EAGER:
                bwd = "triton" if (_CONV_BWD_TRITON and _conv_bwd_triton_available(device)) else "eager"
            else:
                bwd = "fla_npu"
            print(f"[glm53] causal_conv1d fused engaged: fla_npu AscendC (xtuner-qwen port), bwd={bwd}", flush=True)
        except Exception as exc:  # noqa: BLE001 - q35 probes with the same blanket catch
            print(f"[glm53] causal_conv1d fused unavailable ({exc}); eager fallback", flush=True)
    return _FUSED_STATE["ok"]
