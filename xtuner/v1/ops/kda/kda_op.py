# Copyright © 2026 Huawei Technologies Co., Ltd.
"""KDA (Kimi Delta Attention) chunked kernels for Ascend NPU.

Fused path: port of MindSpeed-MM ``mindspeed_mm/fsdp/ops/kda/ascendc/chunk_kda_wrapper.py``
retargeted to the installed ``fla_npu`` (flash-linear-attention-npu) stable AscendC
wrappers (``npu_chunk_kda_fwd`` / ``npu_chunk_kda_bwd``) and fla_npu triton helpers
(autocast guards + l2norm). The fused kernel contract: chunk_size=64, K=V=128,
H == HV, dense inputs, post-sigmoid beta, no initial/final state, gate computed
inside the kernel from the raw gate input plus ``A_log`` / ``dt_bias``.

Packed (varlen) inputs run as one varlen launch per direction -- or, past the
entries' segment / pack-size limits, one per chunk-aligned group -- over a
64-padded rebuild of the pack (``ChunkKDAVarlenFunction``,
``XTUNER_KDA_VARLEN=0`` restores the per-segment dense loop): every segment is
padded to a 64 multiple inside a single packed buffer, so the fla_npu varlen
entries see only aligned segments (their short-tail split path has a wheel
bug) and the per-segment pad/slice/cat and autograd CopySlices launches
disappear.

Eager path: pure-torch reference ported from MindSpeed-MM
``mindspeed_mm/fsdp/ops/kda/chunk_kda_naive.py`` (chunked WY representation with
fp32 state), used as the numerical parity reference and as the ``eager`` fallback
backend. Backward relies on plain autograd.
"""

import os
from collections import OrderedDict

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from fla_npu.ops.ascendc import npu_chunk_kda_bwd, npu_chunk_kda_fwd
from fla_npu.ops.triton.triton_core.l2norm import l2norm_bwd, l2norm_fwd
from fla_npu.ops.triton.triton_core.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard
from typing_extensions import NotRequired, TypedDict


CHUNK_SIZE = 64

# Packed packs run one varlen launch per direction over the 64-padded rebuild;
# ``0`` restores the per-segment dense loop (the pre-optimization reference).
_KDA_VARLEN = os.environ.get("XTUNER_KDA_VARLEN", "1") == "1"

# Head-major (NTD) pass-through: the rebuild, l2norm and forward run directly
# in the varlen kernels' canonical [H, T, D] layout and the backward consumes
# them with zero transposes; ``0`` restores the BSND pipeline (reference).
_KDA_HEAD_MAJOR = os.environ.get("XTUNER_KDA_HEAD_MAJOR", "1") == "1"

# Fuse the q/k l2norm into the head-major rebuild kernel: the segment copy
# normalizes rows in flight and emits the fp32 rstd table, so the padded
# pre-norm pack is never materialized and the separate full-tensor fla l2norm
# pass disappears; ``0`` restores that separate pass (A/B reference).
_KDA_L2NORM_FUSE = os.environ.get("XTUNER_KDA_L2NORM_FUSE", "1") == "1"

# Build the pack-map device tables through pinned host staging with
# non-blocking H2D copies (PyTorch's caching host allocator pools the pinned
# blocks); ``0`` restores the synchronous pageable copies (A/B reference).
_KDA_PINNED_H2D = os.environ.get("XTUNER_KDA_PINNED_H2D", "1") == "1"

# fla_npu l2norm default, mirrored by the fused rebuild kernel.
_L2NORM_EPS = 1e-6


class ChunkKDAKwargs(TypedDict):
    """Keyword bundle shared by the ``chunk_kda_*`` entry points.

    ``scale`` is optional: when omitted both kernels fall back to
    ``head_dim ** -0.5``.
    """

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g_raw: torch.Tensor
    beta: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    scale: NotRequired[float | None]
    use_qk_l2norm: bool
    safe_gate: bool
    lower_bound: float | None
    cu_seqlens: torch.Tensor | list[int] | None


def _as_cu_list(cu_seqlens: torch.Tensor | list[int] | None) -> list[int] | None:
    """Normalize cu_seqlens to a host int list (fla_npu wrappers take host ints)."""
    if cu_seqlens is None:
        return None
    if isinstance(cu_seqlens, torch.Tensor):
        return cu_seqlens.tolist()
    return list(cu_seqlens)


def _bsnd_to_bnsd(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.permute(0, 2, 1, 3).contiguous()


def _bnsd_to_bsnd(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.permute(0, 2, 1, 3).contiguous()


def _bsh_to_bhs(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.permute(0, 2, 1).contiguous()


def _bhs_to_bsh(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.permute(0, 2, 1).contiguous()


def _check_intermediates(
    q: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    w: torch.Tensor,
    qg: torch.Tensor,
    kg: torch.Tensor,
    v_new: torch.Tensor,
    h: torch.Tensor,
    chunks: int,
) -> None:
    batch, tokens, heads, key_dim = q.shape
    value_dim = v.shape[3]
    expected = (
        ("g_cumsum", gk, (batch, heads, tokens, key_dim), torch.float32),
        ("Aqk", Aqk, (batch, heads, tokens, CHUNK_SIZE), q.dtype),
        ("Akk", Akk, (batch, heads, tokens, CHUNK_SIZE), q.dtype),
        ("w", w, (batch, heads, tokens, key_dim), q.dtype),
        ("qg", qg, (batch, heads, tokens, key_dim), q.dtype),
        ("kg", kg, (batch, heads, tokens, key_dim), q.dtype),
        ("v_new", v_new, (batch, heads, tokens, value_dim), q.dtype),
        ("h", h, (batch, chunks, heads, key_dim, value_dim), q.dtype),
    )
    for name, tensor, shape, dtype in expected:
        if tensor is None:
            raise RuntimeError(f"AscendC forward did not produce {name}.")
        if tuple(tensor.shape) != shape:
            raise RuntimeError(f"{name} has shape {tuple(tensor.shape)}, expected {shape}.")
        if tensor.dtype != dtype:
            raise RuntimeError(f"{name} has dtype {tensor.dtype}, expected {dtype}.")
        if not tensor.is_contiguous():
            raise RuntimeError(f"{name} is not contiguous.")


class ChunkKDAAscendCFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        safe_gate: bool = False,
        lower_bound: float | None = None,
    ):
        q_rstd = k_rstd = None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        (o, _final_state, g_cumsum, Aqk, Akk, w, _u, qg, kg, v_new, h, _) = npu_chunk_kda_fwd(
            q,
            k,
            v,
            g,
            beta,
            float(scale),
            CHUNK_SIZE,
            layout="BSND",
            initial_state=None,
            output_final_state=False,
            cu_seqlens=None,
            chunk_indices=None,
            safe_gate=bool(safe_gate),
            lower_bound=lower_bound,
            use_gate_in_kernel=bool(use_gate_in_kernel),
            A_log=A_log if use_gate_in_kernel else None,
            dt_bias=dt_bias if use_gate_in_kernel else None,
            disable_recompute=True,
            return_intermediate_states=False,
            state_v_first=False,
        )

        n_chunks = (q.shape[1] + CHUNK_SIZE - 1) // CHUNK_SIZE
        _check_intermediates(q, v, g_cumsum, Aqk, Akk, w, qg, kg, v_new, h, n_chunks)

        ctx.save_for_backward(
            q, q_rstd, k, k_rstd, v, g, beta, A_log, dt_bias, g_cumsum, Aqk, Akk, w, qg, kg, v_new, h
        )
        ctx.scale = scale
        ctx.safe_gate = safe_gate
        ctx.lower_bound = lower_bound
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_gate_in_kernel = use_gate_in_kernel
        return o.type_as(q)

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor):
        (q, q_rstd, k, k_rstd, v, g_input, beta, A_log, dt_bias, g_cumsum, Aqk, Akk, w, qg, kg, v_new, h) = (
            ctx.saved_tensors
        )

        gate_dt_bias = None
        if ctx.use_gate_in_kernel and dt_bias is not None:
            gate_dt_bias = dt_bias.reshape(q.shape[2], q.shape[3]).contiguous()

        dq_h, dk_h, dv_h, db_h, dg_h, dh0, dA, dbias = npu_chunk_kda_bwd(
            _bsnd_to_bnsd(q),
            _bsnd_to_bnsd(k),
            _bsnd_to_bnsd(v),
            _bsh_to_bhs(beta),
            g_cumsum,
            Aqk,
            Akk,
            w,
            qg,
            kg,
            v_new,
            h,
            _bsnd_to_bnsd(do),
            float(ctx.scale),
            raw_g=_bsnd_to_bnsd(g_input) if ctx.use_gate_in_kernel else None,
            A_log=A_log if ctx.use_gate_in_kernel else None,
            dt_bias=gate_dt_bias,
            initial_state=None,
            dht=None,
            chunk_size=CHUNK_SIZE,
            safe_gate=ctx.safe_gate,
            lower_bound=ctx.lower_bound,
            use_gate_in_kernel=ctx.use_gate_in_kernel,
            disable_recompute=True,
            use_exp2=True,
            state_v_first=False,
        )

        dq, dk, dv = map(_bnsd_to_bsnd, (dq_h, dk_h, dv_h))
        db, dg = _bhs_to_bsh(db_h), _bnsd_to_bsnd(dg_h)
        if dbias is not None:
            dbias = dbias.reshape(dt_bias.shape)

        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)

        # One gradient per positional argument of apply().
        return (dq.to(q), dk.to(k), dv.to(v), dg.to(g_input), db.to(beta), dA, dbias, None, None, None, None, None)


_PACK_MAP_CACHE: OrderedDict[
    tuple[tuple[int, ...], str],
    tuple[list[int], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int],
] = OrderedDict()
_PACK_MAP_CACHE_MAX_ENTRIES = 32


def _h2d_tensor(data: list[int], dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
    """Materialize a host index table on ``device``, through pinned staging when enabled.

    Args:
        data (list[int]): Host index table (cumulative lengths or gather indices).
        dtype (torch.dtype): Element dtype of the returned table.
        device (torch.device | str): Target device.

    Returns:
        torch.Tensor: The table on ``device``.
    """
    device = torch.device(device) if isinstance(device, str) else device
    if _KDA_PINNED_H2D and device.type != "cpu":
        # Pinned staging + non-blocking copy: the H2D leaves the synchronous
        # pageable staging path, and PyTorch's caching host allocator pools
        # the pinned blocks, handing one out only after its in-flight copies
        # retire. Every consumer runs on the current stream, so stream order
        # keeps all reads behind the copy.
        staging = torch.tensor(data, dtype=dtype, pin_memory=True)
        return staging.to(device, non_blocking=True)
    return torch.tensor(data, dtype=dtype, device=device)


def _padded_pack_maps(
    cu: list[int], device: torch.device
) -> tuple[list[int], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Segment-aligned pack maps for the single-launch varlen path.

    Args:
        cu (list[int]): ``[N + 1]`` cumulative sequence lengths of the pack.
        device (torch.device): Device to place the returned position indices on.

    Returns:
        tuple[list[int], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
            The chunk-aligned cumulative lengths (every segment zero-padded up
            to a multiple of ``CHUNK_SIZE``); ``real_pos``, mapping each packed
            token to its row in the padded rebuild; ``pad_to_pack``, the inverse
            map used to gather the rebuild (real rows point at their packed
            source row, pad rows at a sentinel row past the pack end); the
            int32 device tables ``cu_t`` / ``pad_cu_t`` of the kept segments
            (real / padded cumulative starts, zero-length segments dropped)
            that drive the segment-copy repack kernel; and the host-side
            maximum padded segment length for its launch grid.
    """
    key = (tuple(cu), str(device))
    cached = _PACK_MAP_CACHE.get(key)
    if cached is not None:
        _PACK_MAP_CACHE.move_to_end(key)
        return cached

    cu_padded = [0]
    cu_real = [0]
    pos_of: list[int] = []
    pad_to_pack: list[int] = []
    max_padded_len = 0
    n_tokens = cu[-1]
    for start, end in zip(cu[:-1], cu[1:]):
        length = end - start
        if length <= 0:
            continue
        padded_length = length + (-length % CHUNK_SIZE)
        base = cu_padded[-1]
        cu_padded.append(base + padded_length)
        cu_real.append(end)
        max_padded_len = max(max_padded_len, padded_length)
        pos_of.extend(range(base, base + length))
        pad_to_pack.extend(range(start, end))
        pad_to_pack.extend([n_tokens] * (padded_length - length))
    real_pos = _h2d_tensor(pos_of, torch.long, device)
    gather_pos = _h2d_tensor(pad_to_pack, torch.long, device)
    cu_t = _h2d_tensor(cu_real, torch.int32, device)
    pad_cu_t = _h2d_tensor(cu_padded, torch.int32, device)

    if len(_PACK_MAP_CACHE) >= _PACK_MAP_CACHE_MAX_ENTRIES:
        _PACK_MAP_CACHE.popitem(last=False)
    _PACK_MAP_CACHE[key] = (cu_padded, real_pos, gather_pos, cu_t, pad_cu_t, max_padded_len)
    return cu_padded, real_pos, gather_pos, cu_t, pad_cu_t, max_padded_len


_VARLEN_MAX_SEGMENTS = 1024
_VARLEN_MAX_TOKENS = 65536
_VARLEN_MAX_TOKENS_PER_HEAD = 1_572_864


def _varlen_launch_groups(cu_padded: list[int], heads: int) -> list[tuple[int, int, list[int]]]:
    """Split a padded pack into per-launch ``(row_start, row_end, cu)`` groups.

    The varlen aclnn entries cap what one launch accepts (measured on
    ascend910_93): the forward rejects packs with more than
    ``_VARLEN_MAX_SEGMENTS`` segments (GetWorkspaceSize 161002: 1024 segments
    pass, 1025 fail, head-count independent), and the backward additionally
    rejects padded packs longer than ``_VARLEN_MAX_TOKENS`` tokens or with
    ``heads * tokens`` above ``_VARLEN_MAX_TOKENS_PER_HEAD`` (561103: 64K
    tokens pass up to 16 heads; at 32 heads 48K tokens pass and 64K fail). A
    128K pack of short samples easily exceeds both. Group windows are
    chunk-aligned (every ``cu_padded`` boundary is a multiple of
    ``CHUNK_SIZE``), so each group is an independent varlen launch over a
    contiguous slice of the rebuild; the two caps and the greedy walk share
    one grouping because the backward consumes the forward's per-group
    intermediates.
    """
    max_tokens = min(_VARLEN_MAX_TOKENS, _VARLEN_MAX_TOKENS_PER_HEAD // max(heads, 1))
    n_segs = len(cu_padded) - 1
    if n_segs <= _VARLEN_MAX_SEGMENTS and cu_padded[-1] <= max_tokens:
        return [(0, cu_padded[-1], cu_padded)]
    groups = []
    g0 = 0
    while g0 < n_segs:
        g1 = g0 + 1
        while g1 < n_segs and g1 - g0 < _VARLEN_MAX_SEGMENTS and cu_padded[g1 + 1] - cu_padded[g0] <= max_tokens:
            g1 += 1
        start = cu_padded[g0]
        groups.append((start, cu_padded[g1], [c - start for c in cu_padded[g0 : g1 + 1]]))
        g0 = g1
    return groups


# Below this row width the gather's launch floor dominates; the monotone
# index_select stays on those (g / beta carry only one scalar per head).
_SEG_COPY_MIN_ROW_ELEMS = 256
_SEG_COPY_D_BLOCK = 256
_SEG_COPY_ROWS = 64
# Persistent-grid cap for the segment-copy kernels. Under
# TRITON_ALL_BLOCKS_PARALLEL=1 the ascend backend clamps every launch's
# blockNum to the physical block count (launcher C template; measured on
# 0924-box: 48 in aiv mode, 24 in aicore mode -- mode depends on the
# kernel's load/store profile) and rewrites the grid with auto-blockify --
# blocks past the clamp silently never run (0924-box repro: 25-segment
# launches lose segment 24, 64-segment launches lose 48+, bitwise-checked
# against the CPU reference pack; both repros use the HTD (seg, head)
# geometry -- under the old BSND (seg, tile) grid the first lost work shifts
# to (seg48, tile0)). The cap below must therefore stay <= 24:
# it has zero margin against the aicore-mode clamp, and a cache whose
# binaries were compiled without the flag skips the blockify replay loop
# entirely (the flag is not part of triton's disk-cache key). The same
# miscompile class hit hc_norm_linear and conv1d bwd; both were fixed by
# capping the grid and striding the logical blocks inside the kernel, which
# is also the pattern the 65535-program launch cap already required here.
_SEG_COPY_MAX_PROGS = 24
assert _SEG_COPY_MAX_PROGS <= 24, "_SEG_COPY_MAX_PROGS re-enters the aicore blockify-clamp window"


@triton.jit
def _seg_copy_rows_kernel(
    src_ptr,
    dst_ptr,
    cu_ptr,
    pad_cu_ptr,
    stride_src,
    stride_dst,
    nseg,
    D: tl.constexpr,
    D_BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
):
    # One program copies one segment, ROWS rows at a time: reads the contiguous
    # real rows, stores them at the segment's padded base and zero-fills its
    # pad tail (masked lanes load 0.0, exactly the zero row the gather path
    # appends). Programs stride the segment list from a capped 1D grid instead
    # of one program per segment: the per-segment grid trips the
    # TRITON_ALL_BLOCKS_PARALLEL blockify transform (see _SEG_COPY_MAX_PROGS)
    # and the 65535-program launch cap on ascend910_93 (EE1003), and would
    # also size every segment's tile count by the longest one. Addresses come
    # only from the segment table, never from data, so every load/store stays
    # a contiguous block move -- the indexed-gather slow path (~45 GB/s) never
    # engages.
    pid = tl.program_id(0)
    nprogs = tl.num_programs(0)
    for pid_seg in range(pid, nseg, nprogs):
        start = tl.load(cu_ptr + pid_seg)
        end = tl.load(cu_ptr + pid_seg + 1)
        pstart = tl.load(pad_cu_ptr + pid_seg)
        pend = tl.load(pad_cu_ptr + pid_seg + 1)
        for t0 in range(0, tl.cdiv(pend - pstart, ROWS)):
            rows = t0 * ROWS + tl.arange(0, ROWS)
            in_seg = rows < (end - start)
            in_pad = rows < (pend - pstart)
            for d0 in tl.static_range(0, D, D_BLOCK):
                d = d0 + tl.arange(0, D_BLOCK)
                src = src_ptr + (start + rows)[:, None].to(tl.int64) * stride_src + d[None, :]
                vals = tl.load(src, mask=in_seg[:, None], other=0.0)
                dst = dst_ptr + (pstart + rows)[:, None].to(tl.int64) * stride_dst + d[None, :]
                tl.store(dst, vals, mask=in_pad[:, None])


@triton.jit
def _seg_copy_rows_htd_kernel(
    src_ptr,
    dst_ptr,
    cu_ptr,
    pad_cu_ptr,
    stride_src,
    dst_tokens,
    nseg,
    nheads,
    D: tl.constexpr,
    D_BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
):
    # Head-major variant of `_seg_copy_rows_kernel`: the source rows are still
    # the contiguous [S, H * D] BSND row view, but each (row, head) slice
    # lands at dst[h, pstart + row, :] -- a transposing segment copy producing
    # the packed [H, Tp, D] rebuild the varlen NTD entries consume directly.
    # The (seg, head) block list is strided from a capped 1D grid for the same
    # TRITON_ALL_BLOCKS_PARALLEL blockify guard as there (and the 65535-program
    # launch cap on ascend910_93, EE1003). Reads and writes stay D-element
    # contiguous block moves, so the slow indexed path never engages.
    pid = tl.program_id(0)
    nprogs = tl.num_programs(0)
    for blk in range(pid, nseg * nheads, nprogs):
        pid_seg = blk % nseg
        h = (blk // nseg).to(tl.int64)
        start = tl.load(cu_ptr + pid_seg)
        end = tl.load(cu_ptr + pid_seg + 1)
        pstart = tl.load(pad_cu_ptr + pid_seg)
        pend = tl.load(pad_cu_ptr + pid_seg + 1)
        for t0 in range(0, tl.cdiv(pend - pstart, ROWS)):
            rows = t0 * ROWS + tl.arange(0, ROWS)
            in_seg = rows < (end - start)
            in_pad = rows < (pend - pstart)
            src_base = (start + rows)[:, None].to(tl.int64) * stride_src
            dst_base = (pstart + rows)[:, None].to(tl.int64) * D
            for d0 in tl.static_range(0, D, D_BLOCK):
                d = d0 + tl.arange(0, D_BLOCK)
                vals = tl.load(src_ptr + src_base + h * D + d[None, :], mask=in_seg[:, None], other=0.0)
                tl.store(dst_ptr + h * dst_tokens * D + dst_base + d[None, :], vals, mask=in_pad[:, None])


@triton.jit
def _seg_copy_rows_htd_l2norm_kernel(
    src_ptr,
    dst_ptr,
    rstd_ptr,
    cu_ptr,
    pad_cu_ptr,
    stride_src,
    dst_tokens,
    eps,
    nseg,
    nheads,
    D: tl.constexpr,
    D_BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
):
    # Head-major rebuild with the q/k l2norm folded in: the same segment copy
    # as `_seg_copy_rows_htd_kernel`, but each row is normalized in flight and
    # the fp32 rstd table is emitted alongside -- the padded pre-norm pack
    # never exists and the separate full-tensor l2norm pass (a full read +
    # write of q and k each) disappears. The numerics mirror fla_npu's
    # ``l2norm_fwd``: fp32 accumulation, ``rstd = 1 / sqrt(sum(x^2) + eps)``
    # with the fla default eps, and the same fp32 -> bf16 store rounding. Pad
    # rows load as zeros, so their rstd is ``1 / sqrt(eps)`` and their
    # normalized row is exactly zero -- identical to the unfused pass over the
    # zero-padded rebuild. The (seg, head) block list is strided from a capped
    # 1D grid for the same TRITON_ALL_BLOCKS_PARALLEL blockify guard as there.
    pid = tl.program_id(0)
    nprogs = tl.num_programs(0)
    for blk in range(pid, nseg * nheads, nprogs):
        pid_seg = blk % nseg
        h = (blk // nseg).to(tl.int64)
        start = tl.load(cu_ptr + pid_seg)
        end = tl.load(cu_ptr + pid_seg + 1)
        pstart = tl.load(pad_cu_ptr + pid_seg)
        pend = tl.load(pad_cu_ptr + pid_seg + 1)
        for t0 in range(0, tl.cdiv(pend - pstart, ROWS)):
            rows = t0 * ROWS + tl.arange(0, ROWS)
            in_seg = rows < (end - start)
            in_pad = rows < (pend - pstart)
            src_base = (start + rows)[:, None].to(tl.int64) * stride_src
            dst_base = (pstart + rows)[:, None].to(tl.int64) * D
            if D <= D_BLOCK:
                # Whole head dim resident in the tile: one load, one normalize,
                # one store.
                d = tl.arange(0, D)
                vals = tl.load(src_ptr + src_base + h * D + d[None, :], mask=in_seg[:, None], other=0.0).to(tl.float32)
                rstd = 1 / tl.sqrt(tl.sum(vals * vals, 1) + eps)
                y = vals * rstd[:, None]
                tl.store(dst_ptr + h * dst_tokens * D + dst_base + d[None, :], y, mask=in_pad[:, None])
            else:
                # Row wider than the tile: accumulate the fp32 sum of squares over
                # the D blocks, then reload and scale block by block.
                ss = tl.zeros((ROWS,), dtype=tl.float32)
                for d0 in tl.static_range(0, D, D_BLOCK):
                    d = d0 + tl.arange(0, D_BLOCK)
                    vals = tl.load(src_ptr + src_base + h * D + d[None, :], mask=in_seg[:, None], other=0.0)
                    vals = vals.to(tl.float32)
                    ss += tl.sum(vals * vals, 1)
                rstd = 1 / tl.sqrt(ss + eps)
                for d0 in tl.static_range(0, D, D_BLOCK):
                    d = d0 + tl.arange(0, D_BLOCK)
                    vals = tl.load(src_ptr + src_base + h * D + d[None, :], mask=in_seg[:, None], other=0.0)
                    y = vals.to(tl.float32) * rstd[:, None]
                    tl.store(dst_ptr + h * dst_tokens * D + dst_base + d[None, :], y, mask=in_pad[:, None])
            tl.store(rstd_ptr + h * dst_tokens + (pstart + rows), rstd, mask=in_pad)


def _repack_padded(
    tensor: torch.Tensor,
    pad_to_pack: torch.Tensor,
    cu_t: torch.Tensor,
    pad_cu_t: torch.Tensor,
    n_padded: int,
    max_padded_len: int,
    head_major: bool = False,
) -> torch.Tensor:
    """Gather a packed ``[1, T, ...]`` tensor into a zero-padded ``[1, Tp, ...]`` rebuild.

    Wide rows (q / k / v / do, ``H * D`` elements per token) go through the
    segment-copy kernel: one launch whose programs move contiguous segment
    tiles and zero-fill the pad tails, with no data-dependent addressing. The
    CANN gather that path replaces reads the pad rows through a repeated
    sentinel index, which drops the indexed-gather engine to ~45 GB/s on these
    shapes; the segment copy sustains ~400 GB/s effective. Narrow rows
    (g / beta) keep the monotone ``F.pad + index_select`` gather -- a single
    zero row is appended past the pack end and ``pad_to_pack`` gathers the
    whole rebuild in one launch. Both paths reproduce the zero-padded rebuild
    bit-for-bit.

    With ``head_major=True`` the rebuild is returned as the varlen NTD entries'
    canonical ``[H, Tp, ...]`` layout: wide rows go through the transposing
    segment copy directly, and narrow rows take one cheap permute copy on top
    of the gather.
    """
    row_elems = tensor.numel() // tensor.shape[1]
    wide = tensor.dim() in (3, 4) and row_elems >= _SEG_COPY_MIN_ROW_ELEMS and row_elems % _SEG_COPY_D_BLOCK == 0
    if wide and head_major and tensor.dim() == 4:
        heads, head_dim = tensor.shape[2], tensor.shape[3]
        nseg = cu_t.shape[0] - 1
        src = tensor.reshape(tensor.shape[1], row_elems)
        dst = torch.empty((heads, n_padded, head_dim), dtype=tensor.dtype, device=tensor.device)
        grid = (min(nseg * heads, _SEG_COPY_MAX_PROGS),)
        _seg_copy_rows_htd_kernel[grid](
            src,
            dst,
            cu_t,
            pad_cu_t,
            src.stride(0),
            n_padded,
            nseg,
            heads,
            D=head_dim,
            D_BLOCK=min(head_dim, _SEG_COPY_D_BLOCK),
            ROWS=_SEG_COPY_ROWS,
            num_warps=4,
        )
        return dst
    if wide:
        nseg = cu_t.shape[0] - 1
        src = tensor.reshape(tensor.shape[1], row_elems)
        dst = torch.empty((n_padded, row_elems), dtype=tensor.dtype, device=tensor.device)
        grid = (min(nseg, _SEG_COPY_MAX_PROGS),)
        _seg_copy_rows_kernel[grid](
            src,
            dst,
            cu_t,
            pad_cu_t,
            src.stride(0),
            dst.stride(0),
            nseg,
            D=row_elems,
            D_BLOCK=_SEG_COPY_D_BLOCK,
            ROWS=_SEG_COPY_ROWS,
            num_warps=4,
        )
        return dst.reshape((1, n_padded, *tensor.shape[2:]))
    pad_spec = (0, 0, 0, 1) if tensor.dim() == 3 else (0, 0, 0, 0, 0, 1)
    packed = F.pad(tensor, pad_spec).index_select(1, pad_to_pack)
    if not head_major:
        return packed
    # [1, Tp, H, ...] BSND rebuild to the [H, Tp, ...] head-major pack.
    body = packed[0]
    if tensor.dim() == 4:
        return body.permute(1, 0, 2).contiguous()
    return body.permute(1, 0).contiguous()


def _repack_l2norm_padded(
    tensor: torch.Tensor,
    pad_to_pack: torch.Tensor,
    cu_t: torch.Tensor,
    pad_cu_t: torch.Tensor,
    n_padded: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather a packed ``[1, T, H, D]`` q/k into the l2-normalized ``[H, Tp, D]`` head-major pack.

    The zero-padded head-major rebuild and the l2norm run in one kernel: rows
    are normalized in flight (fp32 accumulation, ``rstd = 1 / sqrt(sum(x^2) +
    eps)`` with the fla default ``eps``, fp32 -> bf16 store rounding identical
    to fla_npu's ``l2norm_fwd``) and the fp32 ``[H, Tp]`` rstd table is emitted
    alongside, so the padded pre-norm tensor never exists. Pad rows normalize
    to exactly zero with ``rstd = 1 / sqrt(eps)``, matching the unfused pass
    over the zero-padded rebuild.

    Args:
        tensor (torch.Tensor): ``[1, T, H, D]`` bf16/fp16 pack input.
        pad_to_pack (torch.Tensor): Padded-row -> packed-source-row gather table.
        cu_t (torch.Tensor): Real cumulative segment starts (int32 device table).
        pad_cu_t (torch.Tensor): Padded cumulative segment starts (int32 device table).
        n_padded (int): Total length of the padded rebuild.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The ``[H, Tp, D]`` normalized
            head-major pack and the ``[H, Tp]`` float32 rstd table feeding
            ``l2norm_bwd`` unchanged.
    """
    heads, head_dim = tensor.shape[2], tensor.shape[3]
    nseg = cu_t.shape[0] - 1
    src = tensor.reshape(tensor.shape[1], heads * head_dim)
    dst = torch.empty((heads, n_padded, head_dim), dtype=tensor.dtype, device=tensor.device)
    rstd = torch.empty((heads, n_padded), dtype=torch.float32, device=tensor.device)
    grid = (min(nseg * heads, _SEG_COPY_MAX_PROGS),)
    _seg_copy_rows_htd_l2norm_kernel[grid](
        src,
        dst,
        rstd,
        cu_t,
        pad_cu_t,
        src.stride(0),
        n_padded,
        _L2NORM_EPS,
        nseg,
        heads,
        D=head_dim,
        D_BLOCK=min(head_dim, _SEG_COPY_D_BLOCK),
        ROWS=_SEG_COPY_ROWS,
        num_warps=4,
    )
    return dst, rstd


def _bsnd_to_htd(tensor: torch.Tensor) -> torch.Tensor:
    """[1, T, H, D] BSND pack to the varlen bwd kernel's rank-3 [H, T, D] head-major layout."""
    return tensor.permute(0, 2, 1, 3).squeeze(0).contiguous()


def _htd_to_bsnd(tensor: torch.Tensor) -> torch.Tensor:
    """Rank-3 [H, T, D] head-major gradient back to the [1, T, H, D] BSND pack layout."""
    return tensor.unsqueeze(0).permute(0, 2, 1, 3).contiguous()


def _head_major(tensor: torch.Tensor) -> torch.Tensor:
    """Leading-batch head-major intermediate [1, H, T, ...] to the bwd kernel's rank-3 [H, T, ...]."""
    return tensor.squeeze(0).contiguous()


class ChunkKDAVarlenFunction(torch.autograd.Function):
    """Packed KDA as varlen launches per direction over a 64-padded rebuild.

    Every packed segment is zero-padded to a multiple of ``CHUNK_SIZE`` inside
    a single rebuilt pack, so the fla_npu varlen entries only ever see aligned
    segments (their short-tail split path is broken in the installed wheel)
    and the whole pack -- forward and backward -- costs one kernel launch plus
    a handful of monotone repack gathers instead of a per-segment loop, until the
    pack passes the varlen entries' per-launch limits (forward: 1024 segments;
    backward: 64K padded tokens and a head-scaled bound); beyond them the pack
    splits into chunk-aligned groups, one launch per group per direction.
    Zero pad rows leave real-token outputs and gradients untouched: causality keeps
    real queries off pad keys, the pad-contaminated chunk state sits in each
    segment's last chunk with no successor chunk, and pad output rows are
    dropped by the scatter back to packed positions.
    """

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g_raw: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        safe_gate: bool,
        lower_bound: float | None,
        cu: list[int],
    ):
        cu_padded, real_pos, gather_pos, cu_t, pad_cu_t, max_padded_len = _padded_pack_maps(cu, q.device)
        n_padded = cu_padded[-1]
        # l2norm runs on the padded pack; rstd feeds l2norm_bwd unchanged. With
        # the fuse on (H-major only) the rebuild and the normalization are one
        # kernel per tensor: the padded pre-norm q/k packs never exist and the
        # separate full-tensor l2norm pass (a full read + write of each)
        # disappears. The emitted rstd is [H, Tp] fp32 contiguous -- the flat
        # layout l2norm_bwd indexes, addressing-identical to fla's flat
        # [H * Tp] table.
        if _KDA_L2NORM_FUSE and _KDA_HEAD_MAJOR and q.dim() == 4:
            q_n, q_rstd = _repack_l2norm_padded(q, gather_pos, cu_t, pad_cu_t, n_padded)
            k_n, k_rstd = _repack_l2norm_padded(k, gather_pos, cu_t, pad_cu_t, n_padded)
        else:
            q_p = _repack_padded(q, gather_pos, cu_t, pad_cu_t, n_padded, max_padded_len, head_major=_KDA_HEAD_MAJOR)
            k_p = _repack_padded(k, gather_pos, cu_t, pad_cu_t, n_padded, max_padded_len, head_major=_KDA_HEAD_MAJOR)
            q_n, q_rstd = l2norm_fwd(q_p)
            k_n, k_rstd = l2norm_fwd(k_p)
        v_p = _repack_padded(v, gather_pos, cu_t, pad_cu_t, n_padded, max_padded_len, head_major=_KDA_HEAD_MAJOR)
        g_p = _repack_padded(g_raw, gather_pos, cu_t, pad_cu_t, n_padded, max_padded_len, head_major=_KDA_HEAD_MAJOR)
        beta_p = _repack_padded(beta, gather_pos, cu_t, pad_cu_t, n_padded, max_padded_len, head_major=_KDA_HEAD_MAJOR)

        heads = q_n.shape[0] if _KDA_HEAD_MAJOR else q_n.shape[2]
        groups = _varlen_launch_groups(cu_padded, heads)
        o_parts: list[torch.Tensor] = []
        saved_parts: list[torch.Tensor] = []
        for row_start, row_end, cu_group in groups:
            (o_g, _final_state, gk, Aqk, Akk, w, _u, qg, kg, v_new, h, _) = npu_chunk_kda_fwd(
                q_n[:, row_start:row_end],
                k_n[:, row_start:row_end],
                v_p[:, row_start:row_end],
                g_p[:, row_start:row_end],
                beta_p[:, row_start:row_end],
                float(scale),
                CHUNK_SIZE,
                layout="NTD" if _KDA_HEAD_MAJOR else "BSND",
                initial_state=None,
                output_final_state=False,
                cu_seqlens=cu_group,
                chunk_indices=None,
                safe_gate=bool(safe_gate),
                lower_bound=lower_bound,
                use_gate_in_kernel=True,
                A_log=A_log,
                dt_bias=dt_bias,
                disable_recompute=True,
                return_intermediate_states=False,
                state_v_first=False,
            )
            o_parts.append(o_g)
            saved_parts.extend((gk, Aqk, Akk, w, qg, kg, v_new, h))

        if _KDA_HEAD_MAJOR:
            # The rank-3 forward emits the attention output token-major
            # [T, HV, D] (``attn_shape`` in the wheel's shape table) while every
            # saved intermediate is head-major; index the token axis (dim 0) and
            # unsqueeze straight back to the [1, S, H, D] pack contract.
            o_real = (
                (o_parts[0] if len(o_parts) == 1 else torch.cat(o_parts, dim=0)).index_select(0, real_pos).unsqueeze(0)
            )
        else:
            o_p = o_parts[0] if len(o_parts) == 1 else torch.cat(o_parts, dim=1)
            o_real = o_p.index_select(1, real_pos)
        ctx.save_for_backward(
            q_n,
            q_rstd,
            k_n,
            k_rstd,
            v_p,
            g_p,
            beta_p,
            A_log,
            dt_bias,
            *saved_parts,
            real_pos,
            gather_pos,
            cu_t,
            pad_cu_t,
        )
        ctx.scale = scale
        ctx.safe_gate = safe_gate
        ctx.lower_bound = lower_bound
        ctx.cu_padded = cu_padded
        ctx.groups = groups
        ctx.max_padded_len = max_padded_len
        ctx.kda_heads = heads
        ctx.kda_dim = q_n.shape[-1]
        return o_real

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor):
        (q_n, q_rstd, k_n, k_rstd, v_p, g_p, beta_p, A_log, dt_bias, *parts, real_pos, gather_pos, cu_t, pad_cu_t) = (
            ctx.saved_tensors
        )
        do_p = _repack_padded(
            do, gather_pos, cu_t, pad_cu_t, ctx.cu_padded[-1], ctx.max_padded_len, head_major=_KDA_HEAD_MAJOR
        )

        # The varlen bwd entry takes rank-3 head-major activations and an
        # [H, K] dt_bias; the forward entry above takes the flat [H * K] one.
        # One bwd launch per group: token gradients concatenate along dim 1,
        # the shared A_log / dt_bias gradients sum across groups (the same
        # rule the fla_npu per-segment fallback applies). The per-group
        # intermediates saved by the forward are already group-scoped. In the
        # head-major pipeline every tensor is already [H, T, ...], so the bwd
        # consumes them with zero transposes.
        dq_parts, dk_parts, dv_parts, db_parts, dg_parts = [], [], [], [], []
        dA = None
        dbias = None
        for i, (row_start, row_end, cu_group) in enumerate(ctx.groups):
            gk, Aqk, Akk, w, qg, kg, v_new, h = parts[8 * i : 8 * i + 8]
            if _KDA_HEAD_MAJOR:
                # The rank-3 bwd is a single aclnn launch whose workspace
                # sizing rejects any non-contiguous tensor input (561103).
                # Group slices of a [H, T, ...] pack are strided views
                # whenever the pack spans multiple launch groups, so
                # materialize them (a no-op for single-group packs); the
                # wheel-format intermediates gk..h stay zero-copy.
                dq_h, dk_h, dv_h, db_h, dg_h, _dh0, dA_g, dbias_g = npu_chunk_kda_bwd(
                    q_n[:, row_start:row_end].contiguous(),
                    k_n[:, row_start:row_end].contiguous(),
                    v_p[:, row_start:row_end].contiguous(),
                    beta_p[:, row_start:row_end].contiguous(),
                    gk,
                    Aqk,
                    Akk,
                    w,
                    qg,
                    kg,
                    v_new,
                    h,
                    do_p[:, row_start:row_end].contiguous(),
                    float(ctx.scale),
                    raw_g=g_p[:, row_start:row_end].contiguous(),
                    A_log=A_log,
                    dt_bias=dt_bias.reshape(ctx.kda_heads, ctx.kda_dim).contiguous(),
                    initial_state=None,
                    dht=None,
                    cu_seqlens=cu_group,
                    chunk_indices=None,
                    chunk_size=CHUNK_SIZE,
                    safe_gate=ctx.safe_gate,
                    lower_bound=ctx.lower_bound,
                    use_gate_in_kernel=True,
                    disable_recompute=True,
                    use_exp2=True,
                    state_v_first=False,
                )
            else:
                dq_h, dk_h, dv_h, db_h, dg_h, _dh0, dA_g, dbias_g = npu_chunk_kda_bwd(
                    _bsnd_to_htd(q_n[:, row_start:row_end]),
                    _bsnd_to_htd(k_n[:, row_start:row_end]),
                    _bsnd_to_htd(v_p[:, row_start:row_end]),
                    _bsh_to_bhs(beta_p[:, row_start:row_end]).squeeze(0),
                    _head_major(gk),
                    _head_major(Aqk),
                    _head_major(Akk),
                    _head_major(w),
                    _head_major(qg),
                    _head_major(kg),
                    _head_major(v_new),
                    h.squeeze(0),
                    _bsnd_to_htd(do_p[:, row_start:row_end]),
                    float(ctx.scale),
                    raw_g=_bsnd_to_htd(g_p[:, row_start:row_end]),
                    A_log=A_log,
                    dt_bias=dt_bias.reshape(ctx.kda_heads, ctx.kda_dim).contiguous(),
                    initial_state=None,
                    dht=None,
                    cu_seqlens=cu_group,
                    chunk_indices=None,
                    chunk_size=CHUNK_SIZE,
                    safe_gate=ctx.safe_gate,
                    lower_bound=ctx.lower_bound,
                    use_gate_in_kernel=True,
                    disable_recompute=True,
                    use_exp2=True,
                    state_v_first=False,
                )
            dq_parts.append(dq_h)
            dk_parts.append(dk_h)
            dv_parts.append(dv_h)
            db_parts.append(db_h)
            dg_parts.append(dg_h)
            if dA_g is not None:
                dA = dA_g if dA is None else dA + dA_g
            if dbias_g is not None:
                dbias = dbias_g if dbias is None else dbias + dbias_g

        dq_h = dq_parts[0] if len(dq_parts) == 1 else torch.cat(dq_parts, dim=1)
        dk_h = dk_parts[0] if len(dk_parts) == 1 else torch.cat(dk_parts, dim=1)
        dv_h = dv_parts[0] if len(dv_parts) == 1 else torch.cat(dv_parts, dim=1)
        db_h = db_parts[0] if len(db_parts) == 1 else torch.cat(db_parts, dim=1)
        dg_h = dg_parts[0] if len(dg_parts) == 1 else torch.cat(dg_parts, dim=1)

        # l2norm_bwd must consume the normalized q/k saved above (not the raw
        # inputs) and runs in the padded layout -- dy has to match y's shape;
        # gradients scatter back to packed positions after.
        if _KDA_HEAD_MAJOR:
            dq = _htd_to_bsnd(l2norm_bwd(q_n, q_rstd, dq_h)).index_select(1, real_pos)
            dk = _htd_to_bsnd(l2norm_bwd(k_n, k_rstd, dk_h)).index_select(1, real_pos)
            dv = _htd_to_bsnd(dv_h).index_select(1, real_pos)
            dg = _htd_to_bsnd(dg_h).index_select(1, real_pos)
        else:
            dq = l2norm_bwd(q_n, q_rstd, _htd_to_bsnd(dq_h)).index_select(1, real_pos)
            dk = l2norm_bwd(k_n, k_rstd, _htd_to_bsnd(dk_h)).index_select(1, real_pos)
            dv = _htd_to_bsnd(dv_h).index_select(1, real_pos)
            dg = _htd_to_bsnd(dg_h).index_select(1, real_pos)
        db = db_h.unsqueeze(0).permute(0, 2, 1).contiguous().index_select(1, real_pos)
        if dbias is not None:
            dbias = dbias.reshape(dt_bias.shape)

        # One gradient per positional argument of apply().
        return (dq.to(q_n), dk.to(k_n), dv.to(v_p), dg.to(g_p), db.to(beta_p), dA, dbias, None, None, None, None)


def _validate_kda_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    safe_gate: bool,
    lower_bound: float | None,
) -> float:
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    if q.shape != k.shape:
        raise ValueError(f"q and k must match, got {tuple(q.shape)} vs {tuple(k.shape)}.")
    if H != HV:
        raise NotImplementedError(f"GVA is unsupported: H={H}, HV={HV}.")
    if K != 128 or V != 128:
        raise NotImplementedError(f"Requires K=V=128, got K={K}, V={V}.")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"q must be float16 or bfloat16, got {q.dtype}.")
    if g.shape != (B, T, HV, K):
        raise ValueError(f"g must be {(B, T, HV, K)}, got {tuple(g.shape)}.")
    if beta.shape != (B, T, HV):
        raise ValueError(f"beta must be {(B, T, HV)}, got {tuple(beta.shape)}.")
    if A_log.dtype != torch.float32:
        raise TypeError(f"A_log must be float32, got {A_log.dtype}.")
    if dt_bias is not None and dt_bias.dtype != torch.float32:
        raise TypeError(f"dt_bias must be float32, got {dt_bias.dtype}.")
    if safe_gate:
        if lower_bound is None:
            raise ValueError("lower_bound is required when safe_gate=True.")
        if not -5 <= lower_bound < 0:
            raise ValueError(f"lower_bound must be in [-5, 0), got {lower_bound}.")
    if scale is None:
        return K**-0.5
    return float(scale)


def chunk_kda_fla_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    cu_seqlens: torch.Tensor | list[int] | None = None,
) -> torch.Tensor:
    """Run the fused fla_npu AscendC chunked KDA forward/backward.

    Args:
        q (torch.Tensor): Queries of shape ``[B, T, H, K]``, bf16/fp16.
        k (torch.Tensor): Keys of shape ``[B, T, H, K]``, bf16/fp16.
        v (torch.Tensor): Values of shape ``[B, T, H, V]``, bf16/fp16.
        g_raw (torch.Tensor): Raw gate input of shape ``[B, T, H, K]`` (pre
            gate-transform; the kernel computes the decay internally).
        beta (torch.Tensor): Post-sigmoid betas of shape ``[B, T, H]``.
        A_log (torch.Tensor): ``[H]`` float32 gate amplitude logs.
        dt_bias (torch.Tensor): ``[H * K]`` float32 gate biases (optional).
        scale (float | None): Attention scale; defaults to ``K ** -0.5``.
        use_qk_l2norm (bool): Apply l2norm to q/k inside the autograd Function.
        safe_gate (bool): Use the sigmoid lower-bound safe gate path.
        lower_bound (float | None): Safe-gate lower bound (required when
            ``safe_gate=True``).
        cu_seqlens (torch.Tensor | list[int] | None): ``[N + 1]`` cumulative
            sequence lengths for varlen packed inputs (B must be 1); executed
            as varlen launches per direction over a 64-padded rebuild, split
            into chunk-aligned groups past the entries' per-launch limits
            (``XTUNER_KDA_VARLEN=0`` or ``B > 1`` restores the per-segment
            dense loop). ``None`` for the single-sequence path, which pads T
            to a multiple of 64.

    Returns:
        torch.Tensor: Output of shape ``[B, T, H, V]``, same dtype as ``q``.
    """
    if use_qk_l2norm is False:
        raise NotImplementedError("Only use_qk_l2norm=True is supported.")
    if A_log is None:
        raise ValueError("A_log is required (use_gate_in_kernel=True is always used).")

    scale = _validate_kda_inputs(q, k, v, g_raw, beta, scale, A_log, dt_bias, safe_gate, lower_bound)

    cu = _as_cu_list(cu_seqlens)
    if cu is not None:
        if _KDA_VARLEN and q.shape[0] == 1:
            # One varlen launch per direction over the 64-padded rebuild; the
            # per-segment loop below stays as the reference fallback
            # (XTUNER_KDA_VARLEN=0, or packed rows with B > 1).
            return ChunkKDAVarlenFunction.apply(
                q, k, v, g_raw, beta, A_log, dt_bias, scale, bool(safe_gate), lower_bound, cu
            )
        # fla_npu's packed (rank-3 TND) varlen contract is not exercised here;
        # per-segment dense calls give identical semantics under autograd.
        outs = [
            _chunk_kda_dense(
                q[:, s:e],
                k[:, s:e],
                v[:, s:e],
                g_raw[:, s:e],
                beta[:, s:e],
                A_log,
                dt_bias,
                scale,
                safe_gate,
                lower_bound,
            )
            for s, e in zip(cu[:-1], cu[1:])
        ]
        return torch.cat(outs, dim=1)
    return _chunk_kda_dense(q, k, v, g_raw, beta, A_log, dt_bias, scale, safe_gate, lower_bound)


def _chunk_kda_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    safe_gate: bool,
    lower_bound: float | None,
) -> torch.Tensor:
    orig_T = q.shape[1]
    pad_len = (-orig_T) % CHUNK_SIZE
    if pad_len:
        q = F.pad(q, (0, 0, 0, 0, 0, pad_len), value=1.0)
        k = F.pad(k, (0, 0, 0, 0, 0, pad_len), value=1.0)
        v = F.pad(v, (0, 0, 0, 0, 0, pad_len))
        g_raw = F.pad(g_raw, (0, 0, 0, 0, 0, pad_len))
        beta = F.pad(beta, (0, 0, 0, pad_len))

    o = ChunkKDAAscendCFunction.apply(
        q, k, v, g_raw, beta, A_log, dt_bias, scale, True, True, bool(safe_gate), lower_bound
    )
    if pad_len:
        o = o[:, :orig_T]
    return o


def _l2norm_eager(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    original_dtype = x.dtype
    x = x.float()
    inv_norm = torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)
    return (x * inv_norm).to(original_dtype)


def _kda_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    lower_bound: float | None,
) -> torch.Tensor:
    H, K = g.shape[-2:]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.float().view(H, K)
    A = A_log.float().view(H, 1).exp()
    if lower_bound is not None:
        return lower_bound * torch.sigmoid(A * g)
    return -A * F.softplus(g)


def _chunk_kda_core(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    chunk_size: int,
) -> torch.Tensor:
    B, T, H, K = k.shape
    V = v.shape[-1]

    q, k, v, g, beta = [x.transpose(1, 2).contiguous() for x in (q, k, v, g, beta)]

    pad_size = (chunk_size - T % chunk_size) % chunk_size
    total_length = T + pad_size
    q = F.pad(q, (0, 0, 0, pad_size)) * scale
    k = F.pad(k, (0, 0, 0, pad_size))
    v = F.pad(v, (0, 0, 0, pad_size))
    g = F.pad(g, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))

    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)

    q, k, v, g, k_beta, v_beta = [x.reshape(B, H, -1, chunk_size, x.shape[-1]) for x in (q, k, v, g, k_beta, v_beta)]
    NT = q.shape[2]

    g = g.cumsum(dim=-2)

    mask_lower = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    g_diff = (g.unsqueeze(-2) - g.unsqueeze(-3)).clamp(max=80)
    decay_mask = g_diff.exp()

    attn = -(k_beta.unsqueeze(-2) * k.unsqueeze(-3) * decay_mask).sum(dim=-1)
    attn = attn.masked_fill(mask_lower, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    u = attn @ v_beta
    w = attn @ (k_beta * g.exp())

    S = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    o = torch.zeros_like(v)

    mask_intra = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(NT):
        q_i, k_i, u_i, g_i, w_i = q[:, :, i], k[:, :, i], u[:, :, i], g[:, :, i], w[:, :, i]
        attn_inter = (q_i * g_i.exp()) @ S
        attn_intra = (q_i.unsqueeze(-2) * k_i.unsqueeze(-3) * decay_mask[:, :, i]).sum(dim=-1)
        attn_intra = attn_intra.masked_fill(mask_intra, 0)
        v_new = u_i - w_i @ S
        o[:, :, i] = attn_inter + attn_intra @ v_new
        S = S * g_i[:, :, -1].exp().unsqueeze(-1) + (k_i * (g_i[:, :, -1:] - g_i).exp()).transpose(-1, -2) @ v_new

    o = o.reshape(B, H, total_length, V)[:, :, :T]
    return o.transpose(1, 2).contiguous()


def chunk_kda_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    cu_seqlens: torch.Tensor | list[int] | None = None,
) -> torch.Tensor:
    """Run the pure-torch chunked KDA reference (plain autograd backward).

    Args:
        q (torch.Tensor): Queries of shape ``[B, T, H, K]``, bf16/fp16.
        k (torch.Tensor): Keys of shape ``[B, T, H, K]``, bf16/fp16.
        v (torch.Tensor): Values of shape ``[B, T, H, V]``, bf16/fp16.
        g_raw (torch.Tensor): Raw gate input of shape ``[B, T, H, K]``.
        beta (torch.Tensor): Post-sigmoid betas of shape ``[B, T, H]``.
        A_log (torch.Tensor): ``[H]`` gate amplitude logs.
        dt_bias (torch.Tensor): ``[H * K]`` gate biases (optional).
        scale (float | None): Attention scale; defaults to ``K ** -0.5``.
        use_qk_l2norm (bool): Apply l2norm to q/k (fp32, cast back).
        safe_gate (bool): Use the sigmoid lower-bound safe gate path.
        lower_bound (float | None): Safe-gate lower bound (required when
            ``safe_gate=True``).
        cu_seqlens (torch.Tensor | list[int] | None): ``[N + 1]`` cumulative
            sequence lengths for varlen packed inputs (B must be 1).

    Returns:
        torch.Tensor: Output of shape ``[B, T, H, V]``, same dtype as ``q``.
    """
    if A_log is None:
        raise ValueError("A_log is required.")
    scale = _validate_kda_inputs(q, k, v, g_raw, beta, scale, A_log, dt_bias, safe_gate, lower_bound)

    input_dtype = q.dtype
    if use_qk_l2norm:
        q = _l2norm_eager(q)
        k = _l2norm_eager(k)

    g = _kda_gate(g_raw, A_log, dt_bias, lower_bound if safe_gate else None)
    q_f, k_f, v_f, g_f, b_f = q.float(), k.float(), v.float(), g, beta.float()
    cu = _as_cu_list(cu_seqlens)
    if cu is None:
        o = _chunk_kda_core(q_f, k_f, v_f, g_f, b_f, scale, CHUNK_SIZE)
    else:
        outs = [
            _chunk_kda_core(q_f[:, s:e], k_f[:, s:e], v_f[:, s:e], g_f[:, s:e], b_f[:, s:e], scale, CHUNK_SIZE)
            for s, e in zip(cu[:-1], cu[1:])
        ]
        o = torch.cat(outs, dim=1)
    return o.to(input_dtype)
