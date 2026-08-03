# Copyright (c) OpenMMLab. All rights reserved.
"""Ascend NPU sparse MLA: fused C++ forward + autograd backward on compacted
indices.

Compacts the top-k indices to ``K_eff`` (per-call max valid count) before the
op, so ``npu_sparse_flash_attention`` and its built-in C++ backward process
only ~``K_eff`` real KV slots/query instead of the full ``index_topk`` (2048,
~97% ``-1`` padding). On the full topk the C++ backward is pathologically
slow; on compacted indices it is fast.

Two fixes (see ``_compact_indices`` / ``_prepare_bsnd``):
  * float32 topk -- dispatches AIV TopKV2 instead of AI_CPU Sort.
  * cycled-pad (``-1`` -> ``S_g + (j % N_PAD)``, env
    ``XTUNER_NPU_SPARSE_MLA_NPAD``) -- breaks the C++ bwd gather-conflict.

Compaction (reorder + truncate) is exact. Cycled-pad is NOT exact: pad rows
are zero-value but get softmax weight ``exp(0)=1`` (``sparse_mode=3`` does not
mask explicitly-indexed positions), diluting the output by ``N/(sum+N)``;
N_PAD=16 (default) stays under 1%. Measured speedups/dilution and the N_PAD
sweep are in ``GLM-5p2_xtuner_training.md`` (13B-39/13B-40).
"""

import os

import torch

from .protocol import SparseMLAOutputs

# One-shot K_eff/compaction debug print (gated by XTUNER_NPU_SPARSE_MLA_DBG).
_NPU_SPARSE_MLA_DBG_N = 0

# Above this per-call K_eff, chunk over the query dim so each op call still
# sees a small K_eff. Production K_eff (~300) is well under it.
_K_EFF_SINGLE_MAX = 512

# Cycled zero-pad KV rows (``-1`` slot ``j`` -> ``S_g + (j % N_PAD)``) to break
# the C++ bwd gather-conflict (stock N_PAD=1 routes all pad to one row ->
# gathered ~240x/query). Default 16 (<1% dilution). Env-overridable; <=1
# selects the stock same-pad path (zero dilution, slowest bwd).
_N_PAD_ROWS = int(os.environ.get("XTUNER_NPU_SPARSE_MLA_NPAD", "16"))


def _compact_indices(indices: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Move valid (non -1) indices to the front of each row, truncate to K_eff.

    Args:
        indices (Tensor): ``(S, 1, topk)`` int64 with ``-1`` padding slots.

    Returns:
        tuple: ``compact`` ``(S, 1, K_eff)`` int64 (valid slots front, ``-1``
            tail for rows with fewer valid than ``K_eff``); ``K_eff`` int (max
            valid count across all rows; 0 if every slot is ``-1``).
    """
    idx = indices.squeeze(1)  # [S, topk]  (kv_group == 1)
    valid = idx != -1  # [S, topk]
    k_eff = int(valid.sum(-1).max().item())  # host sync (1/call)
    global _NPU_SPARSE_MLA_DBG_N
    if int(os.environ.get("XTUNER_NPU_SPARSE_MLA_DBG", "0")) and _NPU_SPARSE_MLA_DBG_N < 6:
        _NPU_SPARSE_MLA_DBG_N += 1
        _vsum = valid.sum(-1)
        _topk = idx.shape[-1]
        print(
            f"[NSM-DBG n={_NPU_SPARSE_MLA_DBG_N}] S={idx.shape[0]} topk={_topk} "
            f"K_eff(max_valid)={k_eff} avg_valid/q={float(_vsum.float().mean()):.1f} "
            f"min_valid/q={int(_vsum.min())} max_valid/q={int(_vsum.max())} "
            f"compact_ratio={_topk / max(1, k_eff):.1f}x",
            flush=True,
        )
    if k_eff == 0:
        return idx[:, :0].unsqueeze(1).contiguous(), 0
    # float32 topk: int64/int32 dispatches AI_CPU Sort (~26 ms/call), float32
    # dispatches AIV TopKV2 (~1.2 ms). Result identical (topk over 0/1 values
    # selects valid slots first; 0-slot tie-breaks gather to -1 padding).
    _, sel = torch.topk(valid.float(), k_eff, dim=-1)  # [S, K_eff]
    compact = torch.gather(idx, -1, sel)  # [S, K_eff] valid idx, -1 tail
    return compact.unsqueeze(1).contiguous(), k_eff


def _prepare_bsnd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    value_dim: int,
) -> dict[str, torch.Tensor]:
    """Split absorbed MLA q/kv into (nope, rope), rewrite -1 -> cycled pad rows.

    Operates on the already-compacted indices, so every temp is K_eff-sized.
    The op's backward faults on ``-1`` in packed inputs, so the pad rewrite is
    required (not cosmetic). Cycled pad (``-1`` at slot ``j`` -> ``S_g +
    (j % N_PAD)``) spreads gathers over N_PAD zero rows, breaking the stock
    same-pad gather-conflict; it dilutes the output by ``N_PAD/(sum+N_PAD)``
    (N_PAD=16 < 1%, default). See ``GLM-5p2_xtuner_training.md`` 13B-40 for
    the N_PAD sweep (speed plateau at 32, 64 dominated) and dilution table.

    Args:
        q (Tensor): ``(S, N, value_dim + rope_dim)`` bf16.
        kv (Tensor): ``(S_g, 1, value_dim + rope_dim)`` bf16 (kv_group == 1).
        indices (Tensor): ``(S, 1, K_eff)`` int64 with ``-1`` tail padding.
        value_dim (int): Absorbed output dim (512 for GLM-5.2).

    Returns:
        dict[str, Tensor]: BSND-laid-out tensors for the op (B=1): ``q_nope_b``,
            ``q_rope_b``, ``k_nope_b``, ``k_rope_b``, ``sp``, ``aql``, ``akl``.
    """
    seq_len, _heads, dim = q.shape
    seq_g, kv_group, _ = kv.shape
    if kv_group != 1:
        raise ValueError(
            f"torch_npu sparse MLA expects kv_group == 1 (absorbed MLA), "
            f"got {kv_group}"
        )
    rope_dim = dim - value_dim
    k_eff = indices.shape[-1]
    dev = q.device

    q_nope = q[..., :value_dim]  # [S, N, value_dim]
    q_rope = q[..., value_dim:]  # [S, N, rope_dim]
    kv_nope = kv[..., :value_dim]  # [S_g, 1, value_dim]
    kv_rope = kv[..., value_dim:]  # [S_g, 1, rope_dim]

    # Cycled pad: -1 at column j -> position seq_g + (j % N_PAD). N_PAD zero
    # rows so every cycled pad position indexes a real (zero) row. N_PAD=1
    # reproduces the stock same-pad path.
    n_pad = _N_PAD_ROWS
    if n_pad <= 1:
        pad_pos = torch.full_like(indices, seq_g)
        n_pad_actual = 1
    else:
        slot = torch.arange(k_eff, device=dev, dtype=indices.dtype)
        pad_pos = (seq_g + slot % n_pad).view(1, 1, k_eff).expand_as(indices)
        n_pad_actual = n_pad
    idx_real = torch.where(indices == -1, pad_pos, indices).to(torch.int32)
    pad_rows = torch.zeros(n_pad_actual, 1, dim, device=dev, dtype=kv.dtype)
    kv_nope_p = torch.cat([kv_nope, pad_rows[..., :value_dim]], dim=0)
    kv_rope_p = torch.cat([kv_rope, pad_rows[..., value_dim:]], dim=0)

    return {
        "q_nope_b": q_nope.unsqueeze(0).contiguous(),  # [1, S, N, vd]
        "q_rope_b": q_rope.unsqueeze(0).contiguous(),  # [1, S, N, rope]
        "k_nope_b": kv_nope_p.unsqueeze(0).contiguous(),  # [1, S_g+N_PAD, 1, vd]
        "k_rope_b": kv_rope_p.unsqueeze(0).contiguous(),  # [1, S_g+N_PAD, 1, rope]
        "sp": idx_real.unsqueeze(0).contiguous(),  # [1, S, 1, K_eff]
        "aql": torch.tensor([seq_len], device=dev, dtype=torch.int32),
        "akl": torch.tensor([seq_g + n_pad_actual], device=dev, dtype=torch.int32),
    }


def _run_op(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
    value_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the fused sparse-flash-attn forward (grad-enabled) on compact indices.

    Grad-enabled so autograd registers the op's C++ backward (fast on compact
    ``K_eff``). Returns the forward output and a detached softmax LSE.

    Args:
        q (Tensor): ``(S, N, value_dim + rope_dim)`` bf16.
        kv (Tensor): ``(S_g, 1, value_dim + rope_dim)`` bf16.
        indices (Tensor): ``(S, 1, K_eff)`` int64 (compacted).
        scale (float): Resolved softmax scale.
        value_dim (int): Absorbed output dim.

    Returns:
        tuple[Tensor, Tensor]: ``raw_output`` ``(S, N, value_dim)``;
            ``softmax_lse`` ``(S, N)`` (detached).
    """
    import torch_npu

    p = _prepare_bsnd(q, kv, indices, value_dim)
    attn_out, softmax_max, softmax_sum = torch_npu.npu_sparse_flash_attention(
        p["q_nope_b"],
        p["k_nope_b"],
        p["k_nope_b"],  # value = key_nope (absorbed MLA)
        p["sp"],
        scale_value=scale,
        sparse_block_size=1,
        actual_seq_lengths_query=p["aql"],
        actual_seq_lengths_kv=p["akl"],
        query_rope=p["q_rope_b"],
        key_rope=p["k_rope_b"],
        layout_query="BSND",
        layout_kv="BSND",
        sparse_mode=3,
        attention_mode=2,
        return_softmax_lse=True,
    )
    raw_output = attn_out.squeeze(0).contiguous()  # [S, N, value_dim]
    # softmax_max/sum are [1, S, N]; collapse to [S, N]. Detached by the op.
    lse = softmax_max + torch.log(softmax_sum.clamp(min=1e-30))
    while lse.dim() > 2 and lse.shape[0] == 1:
        lse = lse.squeeze(0)
    return raw_output, lse.contiguous().detach()


def npu_sparse_mla(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    scaling: float | None,
    value_dim: int | None = None,
) -> SparseMLAOutputs:
    """Absorbed sparse MLA on Ascend NPU (fused C++ fwd + bwd).

    Compacts the top-k indices to ``K_eff`` then runs
    ``npu_sparse_flash_attention`` grad-enabled, so the forward and the
    autograd C++ backward process only ~``K_eff`` real KV slots/query instead
    of the full ``index_topk`` (2048). See the module docstring and
    ``GLM-5p2_xtuner_training.md`` (13B-39/13B-40) for measured speedups, the
    exactness/dilution analysis, and the N_PAD sweep.

    Args:
        q (Tensor): ``(seq_len, num_heads, kv_lora + rope)`` bf16.
        kv (Tensor): ``(seq_len_gathered, kv_group, kv_lora + rope)`` bf16.
            ``kv_group`` must be 1 (GLM-5.2 absorbed MLA).
        indices (Tensor): ``(seq_len, kv_group, topk)`` int64 with ``-1``
            padding for invalid slots.
        scaling (float | None): Softmax scale.
        value_dim (int | None): Absorbed output dim (``kv_lora_rank``). Must be
            512 for GLM-5.2.

    Returns:
        SparseMLAOutputs: ``raw_output`` ``(seq_len, num_heads, value_dim)``;
            ``softmax_lse`` ``(seq_len, num_heads)`` (detached).
    """
    _, _, dim = q.shape
    value_dim = value_dim if value_dim is not None else dim
    if value_dim != 512:
        raise ValueError(
            f"torch_npu MLA op requires the absorbed (nope) dim to be 512, "
            f"got {value_dim}"
        )
    scale = float(scaling) if scaling is not None else dim**-0.5

    compact, k_eff = _compact_indices(indices)
    if k_eff == 0:
        # Degenerate (all-padding) chunk: zero output, -inf LSE.
        seq_len, num_heads, _ = q.shape
        raw = q.new_zeros(seq_len, num_heads, value_dim)
        lse = q.new_full((seq_len, num_heads), float("-inf"))
        return SparseMLAOutputs(raw_output=raw, softmax_lse=lse)

    if k_eff <= _K_EFF_SINGLE_MAX:
        raw, lse = _run_op(q, kv, compact, scale, value_dim)
        return SparseMLAOutputs(raw_output=raw, softmax_lse=lse)

    # Fat-tail fallback: chunk over the query dim so each op call sees a small
    # per-chunk K_eff. Shared kv gradient accumulates across chunks via autograd.
    seq_len = q.shape[0]
    chunk = max(1, min(seq_len, 1024))
    outs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    for c in range(0, seq_len, chunk):
        q_c = q[c : c + chunk]
        idx_c = indices[c : c + chunk]
        compact_c, k_eff_c = _compact_indices(idx_c)
        if k_eff_c == 0:
            outs.append(q_c.new_zeros(q_c.shape[0], q_c.shape[1], value_dim))
            lses.append(q_c.new_full((q_c.shape[0], q_c.shape[1]), float("-inf")))
            continue
        raw_c, lse_c = _run_op(q_c, kv, compact_c, scale, value_dim)
        outs.append(raw_c)
        lses.append(lse_c)
    raw_output = torch.cat(outs, dim=0).contiguous()
    softmax_lse = torch.cat(lses, dim=0).contiguous()
    return SparseMLAOutputs(raw_output=raw_output, softmax_lse=softmax_lse)
