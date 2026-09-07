# Copyright (c) OpenMMLab. All rights reserved.
"""Formulation-B validation for the true-ring CP path (``ring_attention``).

Formulation B = feed the per-chunk grad op the MERGED corrected ``(smax,
ssum_true)`` stats and merged corrected output, so each chunk's
``d_s = exp(s - smax_m) / z_merged * (g.v - <g, out_m>)`` is the EXACT gradient
of the merged output w.r.t. that chunk's scores. The backup of this test only
validated a SINGLE chunk; the keystone here is the TWO-chunk case -- one global
top-k split across two ring chunks, per-chunk fwd + online-softmax merge +
per-chunk bwd summed, compared against one-shot exact autograd over the full
KV (the exact multi-chunk claim ``backward_true`` makes).

* ``test_drivers_cp1_match_exact`` runs the REAL ``forward_true`` /
  ``backward_true`` drivers with ``cp_size=1`` (reach 0 -> zero P2P hops,
  zero ``reduce_scatter``, no process group) -- the drivers themselves.
* ``test_two_chunk_merge_and_grad_match_exact`` replicates the rank-0 driver
  loop bodies over local KV slices (a real CP2 ring step needs ``dist``, which
  ``test_ring_attention_cp.py`` covers end-to-end) -- the multi-chunk FORMULA.

NPU-gated (fused kernels required); collectable on a CPU-only runner::

    PYTHONPATH=/workspace/MindSpeed:/workspace/MindSpeed-LLM \
        python -m pytest tests/ops/test_cp_formulation_b.py -x --noconftest
"""

import types

import pytest
import torch


try:
    import torch_npu  # noqa: F401  registers torch.npu / the fused ops

    _HAS_TORCH_NPU = True
except ModuleNotFoundError:
    _HAS_TORCH_NPU = False

from xtuner.v1.ops.cp.ring_attention import (
    Dr,
    Rkv,
    _fused_chunk_attention,
    _grad_op_inputs,
    _merge_stats,
    _parse_lse,
    _remap_all_chunks,
    _seq_tensors,
    backward_true,
    forward_true,
)


DIM = Rkv + Dr  # 576, the kernel's hard-coded qk_head_dim
SCALE = DIM**-0.5
# bf16 tolerances, mirroring tests/ops/test_ring_attention_cp.py (the ring math
# is exact in float32; drift comes from the fused kernels + bf16 stat feeds).
FWD_ATOL, FWD_RTOL = 5e-2, 5e-2
BWD_ATOL, BWD_RTOL = 8e-2, 8e-2

S = 64  # per-chunk / local seq len (math gate, not scale)
N = 32  # query heads
K = 16  # DSA top-k width
DTYPE = torch.bfloat16


def _npu_available() -> bool:
    return _HAS_TORCH_NPU and torch.npu.is_available()


requires_npu = pytest.mark.skipif(not _npu_available(), reason="fused NPU kernels required")


def _make_inputs(cp_size: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build bf16 q/kv and an int32 global top-k with >= 1 real per row.

    The first top-k slot of every row is forced real (an all ``-1`` row makes
    the fp32 reference softmax ``NaN``); later slots are random in the global
    KV range with ~25% ``-1`` padding.

    Args:
        cp_size: Ring degree (global KV length ``= cp_size * S``).
        seed: RNG seed.

    Returns:
        ``(q_states, kv_global, topk, grad_out)`` -- ``[S, N, DIM]`` query,
        ``[cp_size*S, 1, DIM]`` latent KV, ``[S, 1, K]`` int32 global indices,
        fp32 ``[S, N, Rkv]`` output gradient.
    """
    torch.manual_seed(seed)
    device = "npu:0"
    L = cp_size * S
    q = torch.randn(S, N, DIM, device=device, dtype=DTYPE)
    kv = torch.randn(L, 1, DIM, device=device, dtype=DTYPE)
    topk = torch.randint(0, L, (S, K), dtype=torch.int32, device=device)
    pad = torch.rand(S, K, device=device) < 0.25
    pad[:, 0] = False  # row 0 always keeps one real
    topk.masked_fill_(pad, -1)
    grad_out = torch.randn(S, N, Rkv, device=device, dtype=torch.float32)
    return q, kv, topk.unsqueeze(1), grad_out


def _exact_reference(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    topk: torch.Tensor,
) -> torch.Tensor:
    """Exact fp32 attention over the REAL top-k entries only (``-1`` masked).

    This is the oracle semantics of ``tests/ops/test_ring_attention_cp.py``'s
    ``_pytorch_forward``: softmax over the reals, dummies excluded entirely --
    exactly what the ring's count-correction reconstructs.

    Args:
        q_nope: ``[S, N, Rkv]`` float32 query (no-rope part), a grad-tracked leaf.
        q_rope: ``[S, N, Dr]`` float32 query rope, a grad-tracked leaf.
        kv: ``[L, 1, DIM]`` float32 latent KV, a grad-tracked leaf.
        topk: ``[S, 1, K]`` global indices (``-1`` invalid).

    Returns:
        ``[S, N, Rkv]`` float32 exact output.
    """
    valid = (topk != -1).squeeze(1)  # [S, K]
    safe = topk.clamp_min(0).squeeze(1)  # [S, K]
    kv_g = kv[..., :Rkv].squeeze(1)[safe]  # [S, K, Rkv]
    kr_g = kv[..., Rkv:].squeeze(1)[safe]  # [S, K, Dr]
    s = (torch.einsum("snd,skd->snk", q_nope, kv_g) + torch.einsum("snd,skd->snk", q_rope, kr_g)) * SCALE
    s = s.masked_fill(~valid[:, None, :], float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.einsum("snk,skd->snd", p, kv_g)


def _exact_grads(
    q: torch.Tensor,
    kv: torch.Tensor,
    topk: torch.Tensor,
    grad_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Autograd the fp32 reference and return ``(dq, dkv)`` in ring layout.

    Args:
        q: ``[S, N, DIM]`` bf16 query (cast to fp32 leaves inside).
        kv: ``[L, 1, DIM]`` bf16 latent KV.
        topk: ``[S, 1, K]`` int32 global indices.
        grad_out: ``[S, N, Rkv]`` fp32 output gradient.

    Returns:
        ``(dq, dkv)``: ``[S, N, DIM]`` and ``[L, DIM]`` float32 grads (the
        layout ``backward_true`` returns, before dtype cast).
    """
    q_f = q.detach().float().requires_grad_(True)
    kv_f = kv.detach().float().requires_grad_(True)
    out = _exact_reference(q_f[..., :Rkv], q_f[..., Rkv:], kv_f, topk)
    gq, gkv = torch.autograd.grad(out, [q_f, kv_f], grad_outputs=grad_out)
    return gq, gkv.squeeze(1)


def _grad_op_call(
    q_nope_b: torch.Tensor,
    q_rope_b: torch.Tensor,
    grad_out_bf: torch.Tensor,
    out_bf: torch.Tensor,
    smax_native: torch.Tensor,
    ssum_native: torch.Tensor,
    kv_pad: torch.Tensor,
    krope_pad: torch.Tensor,
    local_idx: torch.Tensor,
    device: torch.device,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One ``npu_sparse_flash_attention_grad`` call exactly as ``backward_true`` makes it."""
    _qlen, _kvlen = _seq_tensors(seq_len, device)
    return torch_npu.npu_sparse_flash_attention_grad(
        q_nope_b,
        kv_pad.unsqueeze(0).contiguous(),
        kv_pad.unsqueeze(0).contiguous(),
        local_idx.unsqueeze(0).contiguous(),
        grad_out_bf,
        out_bf,
        smax_native,
        ssum_native,
        SCALE,
        1,  # sparse_block_size
        query_rope=q_rope_b,
        key_rope=krope_pad.unsqueeze(0).contiguous(),
        actual_seq_qlen=_qlen,
        actual_seq_kvlen=_kvlen,
        layout="BSND",
        sparse_mode=0,
        attention_mode=2,
    )


class TestFormulationB:
    """Formulation B (merged-stats per-chunk grad) == exact autograd."""

    @requires_npu
    def test_drivers_cp1_match_exact(self):
        """The REAL drivers end-to-end at cp_size=1 (no P2P, no reduce_scatter)."""
        torch.npu.set_device(0)
        q, kv, topk, grad_out = _make_inputs(cp_size=1)
        seq_ctx = types.SimpleNamespace(_shard_start=0, cu_seq_lens_q_list=[0, S])

        out, smax, ssum, payload = forward_true(q, kv, topk, SCALE, None, 1, 0, seq_ctx)
        assert len(payload) == 1 and payload[0][0] == 0

        dq, dkv_local = backward_true(q, topk, out, smax, ssum, grad_out, payload, SCALE, None, 1, 0)
        dq_ref, dkv_ref = _exact_grads(q, kv, topk, grad_out)

        out_ref = _exact_reference(q[..., :Rkv].float(), q[..., Rkv:].float(), kv.float(), topk)  # noqa: E501-free helper reuse
        torch.testing.assert_close(out.float(), out_ref, atol=FWD_ATOL, rtol=FWD_RTOL)
        torch.testing.assert_close(dq.float(), dq_ref, atol=BWD_ATOL, rtol=BWD_RTOL)
        torch.testing.assert_close(dkv_local.squeeze(1).float(), dkv_ref, atol=BWD_ATOL, rtol=BWD_RTOL)

    @requires_npu
    def test_two_chunk_merge_and_grad_match_exact(self):
        """Keystone: two ring chunks (fwd+merge / per-chunk bwd sum) == one-shot exact.

        Rank-0 view of a cp_size=2 ring: step ``j`` holds chunk ``(0 - j) % 2``
        with global start ``((0 - j) % 2) * S``. The chunks are taken straight
        from the local KV slice (no P2P here -- the ring's rotation of the same
        data is validated end-to-end by ``test_ring_attention_cp.py``). Row 0's
        top-k is forced into chunk 0 and row 1's into chunk 1, so each chunk
        holds an all-dummy ROW (the clamped-``ssum_true`` path) while keeping
        the chunk itself real (``k > 0``).
        """
        torch.npu.set_device(0)
        cp_size = 2
        q, kv, topk, grad_out = _make_inputs(cp_size=cp_size, seed=1)
        # Force row 0 wholly into chunk 0, row 1 wholly into chunk 1.
        topk[0, 0] = torch.randint(0, S, (K,), dtype=torch.int32, device=q.device)
        topk[0, 0, 1:] = -1
        topk[1, 0] = torch.randint(S, cp_size * S, (K,), dtype=torch.int32, device=q.device)
        topk[1, 0, 1:] = -1
        device = q.device

        q_nope, q_rope = q[..., :Rkv], q[..., Rkv:]
        all_idx = _remap_all_chunks(topk, S, cp_size, 0)
        merged_out = merged_smax = merged_ssum = None
        per_chunk: list[tuple[int, int, torch.Tensor, torch.Tensor]] = []
        for j in range(cp_size):
            local_idx, ndummy, k, plan = all_idx[j]
            assert plan is None  # bucket gate off -> full-width plan-less path
            if k == 0:
                continue
            chunk = (0 - j) % cp_size
            kv_j = kv[chunk * S : (chunk + 1) * S]
            o, sm, ss, r = _fused_chunk_attention(q_nope, q_rope, kv_j, local_idx, ndummy, SCALE)
            if merged_out is None:
                # Mirror forward_true: fp32 accumulator, correction folded with
                # weight z_true * rescale (the raw chunk's count correction).
                merged_out = torch.mul(o, r.unsqueeze(-1))
                merged_smax, merged_ssum = sm, ss
            else:
                merged_out, merged_smax, merged_ssum = _merge_stats(merged_out, merged_smax, merged_ssum, o, sm, ss, r)
            per_chunk.append((j, chunk * S, local_idx, kv_j))
        assert len(per_chunk) == 2, "both chunks must carry reals (test construction)"

        # ---- forward: merged == exact over the union of reals ----
        out_ref = _exact_reference(q_nope.float(), q_rope.float(), kv.float(), topk)
        torch.testing.assert_close(merged_out.float(), out_ref, atol=FWD_ATOL, rtol=FWD_RTOL)
        valid = (topk != -1).squeeze(1)
        safe = topk.clamp_min(0).squeeze(1)
        s_ref = (
            torch.einsum("snd,skd->snk", q_nope.float(), kv.float()[..., :Rkv].squeeze(1)[safe])
            + torch.einsum("snd,skd->snk", q_rope.float(), kv.float()[..., Rkv:].squeeze(1)[safe])
        ) * SCALE
        s_ref = s_ref.masked_fill(~valid[:, None, :], float("-inf"))
        lse_ref = torch.logsumexp(s_ref, dim=-1)
        torch.testing.assert_close(_parse_lse(merged_smax, merged_ssum, S, N), lse_ref, atol=FWD_ATOL, rtol=FWD_RTOL)

        # ---- backward: per-chunk grads with MERGED stats, summed == exact ----
        dq_ref, dkv_ref = _exact_grads(q, kv, topk, grad_out)
        smax_native = merged_smax.unsqueeze(0).unsqueeze(0).float().contiguous()
        ssum_native = merged_ssum.unsqueeze(0).unsqueeze(0).float().contiguous()
        out_bf = merged_out.unsqueeze(0).to(DTYPE).contiguous()
        grad_out_bf = grad_out.unsqueeze(0).contiguous().to(DTYPE)
        q_nope_b = q_nope.unsqueeze(0).contiguous()
        q_rope_b = q_rope.unsqueeze(0).contiguous()
        dq_nope = dq_rope = None
        grad_kv_full = torch.zeros(cp_size * S, DIM, device=device, dtype=torch.float32)
        for _j, gstart, local_idx, kv_j in per_chunk:
            kv_pad, krope_pad = _grad_op_inputs(kv_j, device, DTYPE)
            dq, dk, dv, dqr, dkr = _grad_op_call(
                q_nope_b,
                q_rope_b,
                grad_out_bf,
                out_bf,
                smax_native,
                ssum_native,
                kv_pad,
                krope_pad,
                local_idx,
                device,
                S,
            )
            dkv = torch.cat([dk + dv, dkr], dim=-1)  # [1, S+Z, 1, DIM]
            grad_kv_full[gstart : gstart + S].add_(dkv.squeeze(0).squeeze(1)[:S].float())
            dq_nope = dq.squeeze(0).float() if dq_nope is None else dq_nope + dq.squeeze(0).float()
            dq_rope = dqr.squeeze(0).float() if dq_rope is None else dq_rope + dqr.squeeze(0).float()
        dq_ring = torch.cat([dq_nope, dq_rope], dim=-1)
        torch.testing.assert_close(dq_ring, dq_ref, atol=BWD_ATOL, rtol=BWD_RTOL)
        torch.testing.assert_close(grad_kv_full, dkv_ref, atol=BWD_ATOL, rtol=BWD_RTOL)
