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

# Forensics switch (default off): every backward call appends a one-line
# nan/inf summary per tensor, the first ``_CONV_BWD_DUMP`` calls are dumped in
# full, and the first call whose outputs contain nan/inf is dumped in full
# regardless of the counter -- so a production-only numeric fault can be
# localised to one layer call. Diagnostics only, same pattern as
# XTUNER_OFFLOAD_MEM_DEBUG.
_CONV_BWD_DUMP = int(os.environ.get("XTUNER_GLM53_CONV_BWD_DUMP", "0") or "0")
_CONV_BWD_DUMPED: dict[str, int] = {}
_CONV_BWD_CALL: list[int] = [0]
_CONV_BWD_NAN_DUMPED: list[bool] = [False]


def _conv_forensic_stat(v: torch.Tensor) -> str:
    f = v.float()
    return f"nan={int(torch.isnan(f).sum())} inf={int(torch.isinf(f).sum())} absmax={float(f.abs().max()):.3e}"


def _dump_conv_bwd(tag: str, engine: str, tensors: dict[str, torch.Tensor], meta: dict) -> None:
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if tag == "dyin":
        _CONV_BWD_CALL[0] += 1
    call = _CONV_BWD_CALL[0]
    bad = any(int(torch.isnan(v.float()).sum()) + int(torch.isinf(v.float()).sum()) for v in tensors.values())
    line = (
        f"call{call} {tag} engine={engine} rank={rank} TxC={tuple(tensors.get('dx', tensors.get('dy')).shape)} "  # type: ignore[union-attr]
        + " ".join(f"{k}[{_conv_forensic_stat(v)}]" for k, v in tensors.items())
    )
    os.makedirs("/tmp/convdump", exist_ok=True)
    with open(f"/tmp/convdump/summary_r{rank}.log", "a") as fh:
        fh.write(line + "\n")
    full = _CONV_BWD_DUMPED.get(tag, 0) < _CONV_BWD_DUMP or (bad and tag == "dxdw" and not _CONV_BWD_NAN_DUMPED[0])
    if not full:
        return
    if bad and tag == "dxdw":
        _CONV_BWD_NAN_DUMPED[0] = True
    _CONV_BWD_DUMPED[tag] = _CONV_BWD_DUMPED.get(tag, 0) + 1
    payload = {k: v.detach().cpu() for k, v in tensors.items()}
    payload["meta"] = {"engine": engine, "rank": rank, "call": call, **meta}  # type: ignore[assignment]
    suffix = "nan" if bad else str(_CONV_BWD_DUMPED[tag])
    torch.save(payload, f"/tmp/convdump/r{rank}_{tag}{suffix}.pt")


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
        if _CONV_BWD_DUMP:
            _dump_conv_bwd(
                "dyin",
                "n/a",
                {"x": op_x, "pre": pre, "w": op_weight, "dy": op_dy},
                {"activation": activation, "n_bounds": len(ctx.bounds), "bounds_head": ctx.bounds[:8]},
            )
        engine = "aclnn"
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
                    engine = "triton"
                    if _CONV_BWD_DUMP:
                        _dump_conv_bwd(
                            "dxdw",
                            engine,
                            {"dx": dx, "dw": dw},
                            {"activation": activation, "n_bounds": len(ctx.bounds), "bounds_head": ctx.bounds[:8]},
                        )
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
                engine = "eager"
                if _CONV_BWD_DUMP:
                    _dump_conv_bwd(
                        "dxdw",
                        engine,
                        {"dx": dx, "dw": dw},
                        {"activation": activation, "n_bounds": len(ctx.bounds), "bounds_head": ctx.bounds[:8]},
                    )
                return dx.reshape(ctx.x_shape), dw.transpose(0, 1).contiguous(), None, None
        dx, dw = _conv_bwd_aclnn(
            op_x,
            pre.unsqueeze(0) if activation else None,
            op_weight,
            op_dy,
            ctx.bounds,
            activation=activation,
        )
        if _CONV_BWD_DUMP:
            _dump_conv_bwd(
                "dxdw",
                engine,
                {"dx": dx, "dw": dw},
                {"activation": activation, "n_bounds": len(ctx.bounds), "bounds_head": ctx.bounds[:8]},
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
    row is always single-segment).

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
        if any(ln < 0 for ln in lens) or sum(lens) != total:
            return None
        seg_id = torch.repeat_interleave(
            torch.arange(len(lens), device=x2d.device, dtype=torch.int32),
            torch.tensor(lens, device=x2d.device, dtype=torch.int64),
        )
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
            # Per-tap dw partials reduce the [BT, BC] product along t; trans +
            # axis=1 is the verified-compilable form at the 4096-element tile size.
            tl.store(pdw_ptr + (pid_t * W).to(tl.int64) * C + c_off, tl.sum(tl.trans(x * dsilu), axis=1), mask=c_mask)
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
                tl.store(
                    pdw_ptr + (pid_t * W + k).to(tl.int64) * C + c_off, tl.sum(tl.trans(x * dsk), axis=1), mask=c_mask
                )
            tl.store(dx_ptr + t64 + c_off[None, :], dx.to(dx_ptr.dtype.element_ty), mask=m2)


_CONV_BT = 32  # BT*BC <= 4096: 64x128 exceeds the BiShengIR tile resource limit (see kernel comment)
_CONV_BC = 128
# Persistent-grid program cap, same blockify-replay guard as hc_norm_linear
# (_MAX_PROGS=24): stay under the 25-core blockify threshold of the ascend
# triton backend regardless of the token count.
_CONV_BWD_MAX_PROGS = 24


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
        if any(ln < 0 for ln in lens) or sum(lens) != total:
            return None
        seg_id = torch.repeat_interleave(
            torch.arange(len(lens), device=x2d.device, dtype=torch.int32),
            torch.tensor(lens, device=x2d.device, dtype=torch.int64),
        )
    else:
        seg_id = None
    width = weight_kc.shape[0]
    dy2d = dy3d.reshape(total, channels)
    pre = (y3d.reshape(total, channels) if activation else dy2d).contiguous()  # type: ignore[union-attr]
    w_rev = weight_kc.flip(0).contiguous()
    n_t_tiles = triton.cdiv(total, _CONV_BT)
    partial = torch.empty((n_t_tiles, width, channels), dtype=torch.float32, device=x2d.device)
    dx = torch.empty((total, channels), dtype=x2d.dtype, device=x2d.device)
    _conv_bwd_kernel[(min(n_t_tiles * (channels // _CONV_BC), _CONV_BWD_MAX_PROGS),)](
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


def _conv_bwd_triton_available(device: torch.device) -> bool:
    """Lazy one-shot probe of the triton backward (compile + small launch)."""
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
            bwd = "eager" if _CONV_BWD_EAGER else "fla_npu"
            print(f"[glm53] causal_conv1d fused engaged: fla_npu AscendC (xtuner-qwen port), bwd={bwd}", flush=True)
        except Exception as exc:  # noqa: BLE001 - q35 probes with the same blanket catch
            print(f"[glm53] causal_conv1d fused unavailable ({exc}); eager fallback", flush=True)
    return _FUSED_STATE["ok"]
