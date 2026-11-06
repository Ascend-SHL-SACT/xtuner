"""Parity tests for the env-gated ViT flash-attention recompute skip (NPU).

The stash/restore Function is exercised through the reentrant checkpoint it targets: each
test compares a standard ``npu_flash_varlen_attn`` forward+backward against the skip path
wrapped in ``torch.utils.checkpoint.checkpoint``. Two consecutive passes per module prove
the stash drains per step; the reverse-order backward of two simultaneously live graphs
proves the LIFO pop order, and a same-shape forward-only pass before a live one proves a
stale entry sinks below the live entries instead of being restored.
"""

import pytest
import torch
import torch.utils.checkpoint


torch_npu = pytest.importorskip("torch_npu")

from xtuner.v1.ops.flash_attn.npu import npu_flash_varlen_attn  # noqa: E402
from xtuner.v1.utils.activation_offload_npu import (  # noqa: E402
    _MAX_INFLIGHT_ATTN_STASH,
    _attn_stash,
    vit_flash_attn_skip_recompute,
)


_T, _H, _D = 512, 16, 64
_CU = [0, 200, 512]


def _stash_key(module: object) -> tuple[object, tuple[int, ...]]:
    return (module, (_T, _H, _D))


def _seed_inputs(seed: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    dev = "npu:0"
    q = torch.randn(_T, _H, _D, device=dev).to(torch.bfloat16)
    k = torch.randn(_T, _H, _D, device=dev).to(torch.bfloat16)
    v = torch.randn(_T, _H, _D, device=dev).to(torch.bfloat16)
    w = torch.randn(_T, _H, _D, device=dev).to(torch.bfloat16)
    cu = torch.tensor(_CU, device=dev, dtype=torch.int32)
    return q, k, v, w, cu


def _make_block(cu: torch.Tensor, module: object):
    def block_fn(q_: torch.Tensor, k_: torch.Tensor, v_: torch.Tensor) -> torch.Tensor:
        out = vit_flash_attn_skip_recompute(
            q_.unsqueeze(0).transpose(1, 2),
            k_.unsqueeze(0).transpose(1, 2),
            v_.unsqueeze(0).transpose(1, 2),
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=_T,
            max_seqlen_k=_T,
            softmax_scale=_D**-0.5,
            deterministic=False,
            module=module,
        )
        return out["raw_output"].squeeze(0)

    return block_fn


def _reference_grads(ins: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    q, k, v, w, cu = ins
    q1 = q.clone().requires_grad_()
    k1 = k.clone().requires_grad_()
    v1 = v.clone().requires_grad_()
    out = npu_flash_varlen_attn(q1, k1, v1, cu, cu, _T, _T, softmax_scale=_D**-0.5, causal=False, deterministic=False)
    (out.float() * w.float()).sum().backward()
    return out.detach(), q1.grad, k1.grad, v1.grad


def _skip_grads(ins: tuple[torch.Tensor, ...], module: object) -> tuple[torch.Tensor, ...]:
    q, k, v, w, cu = ins
    q2 = q.clone().requires_grad_()
    k2 = k.clone().requires_grad_()
    v2 = v.clone().requires_grad_()
    out = torch.utils.checkpoint.checkpoint(_make_block(cu, module), q2, k2, v2, use_reentrant=True)
    (out.float() * w.float()).sum().backward()
    return out.detach(), q2.grad, k2.grad, v2.grad


class TestVitSkipRecomputeAttn:
    @pytest.mark.gpu
    def test_skip_recompute_grad_parity_two_steps(self) -> None:
        mod = object()
        for step in range(2):
            ins = _seed_inputs(7 + step)
            ref_out, gq_r, gk_r, gv_r = _reference_grads(ins)
            out, gq_s, gk_s, gv_s = _skip_grads(ins, mod)
            assert torch.equal(out, ref_out)
            for ref, skip in ((gq_r, gq_s), (gk_r, gk_s), (gv_r, gv_s)):
                assert torch.equal(ref, skip)
        assert not _attn_stash[_stash_key(mod)]

    @pytest.mark.gpu
    def test_two_pass_lifo_order(self) -> None:
        mod = object()
        ins1, ins2 = _seed_inputs(11), _seed_inputs(12)
        refs = [_reference_grads(ins) for ins in (ins1, ins2)]
        outs, leaves = [], []
        for ins in (ins1, ins2):
            q, k, v = (t.clone().requires_grad_() for t in ins[:3])
            leaves.append((q, k, v))
            outs.append(torch.utils.checkpoint.checkpoint(_make_block(ins[4], mod), q, k, v, use_reentrant=True))
        # Backward in reverse completion order, as autograd would replay the two graphs.
        (outs[1].float() * ins2[3].float()).sum().backward()
        (outs[0].float() * ins1[3].float()).sum().backward()
        for idx, (ref_out, gq_r, gk_r, gv_r) in enumerate(refs):
            assert torch.equal(outs[idx].detach(), ref_out)
            q, k, v = leaves[idx]
            for ref, skip in ((gq_r, q.grad), (gk_r, k.grad), (gv_r, v.grad)):
                assert torch.equal(ref, skip)
        assert not _attn_stash[_stash_key(mod)]

    @pytest.mark.gpu
    def test_stale_push_sinks_below_live_entries(self) -> None:
        mod = object()
        # A forward-only pass whose replay never runs (e.g. a probe), same shape as the
        # live pass: it stashes one entry without popping it.
        ins0 = _seed_inputs(20)
        with torch.no_grad():
            vit_flash_attn_skip_recompute(
                ins0[0].unsqueeze(0).transpose(1, 2),
                ins0[1].unsqueeze(0).transpose(1, 2),
                ins0[2].unsqueeze(0).transpose(1, 2),
                cu_seqlens_q=ins0[4],
                cu_seqlens_k=ins0[4],
                max_seqlen_q=_T,
                max_seqlen_k=_T,
                softmax_scale=_D**-0.5,
                deterministic=False,
                module=mod,
            )
        # The live pass must pop its own (newest) entry, not the stale one.
        ins = _seed_inputs(21)
        out, gq_s, gk_s, gv_s = _skip_grads(ins, mod)
        ref_out, gq_r, gk_r, gv_r = _reference_grads(ins)
        assert torch.equal(out, ref_out)
        for ref, skip in ((gq_r, gq_s), (gk_r, gk_s), (gv_r, gv_s)):
            assert torch.equal(ref, skip)
        assert len(_attn_stash[_stash_key(mod)]) == 1
        _attn_stash[_stash_key(mod)].clear()

    @pytest.mark.gpu
    def test_inflight_cap_drops_oldest(self) -> None:
        mod = object()
        torch.manual_seed(0)
        tiny = torch.randn(8, _H, _D, device="npu:0").to(torch.bfloat16)
        tiny_cu = torch.tensor([0, 3, 8], device="npu:0", dtype=torch.int32)
        for _ in range(_MAX_INFLIGHT_ATTN_STASH + 2):
            with torch.no_grad():
                vit_flash_attn_skip_recompute(
                    tiny.unsqueeze(0).transpose(1, 2),
                    tiny.unsqueeze(0).transpose(1, 2),
                    tiny.unsqueeze(0).transpose(1, 2),
                    cu_seqlens_q=tiny_cu,
                    cu_seqlens_k=tiny_cu,
                    max_seqlen_q=8,
                    max_seqlen_k=8,
                    softmax_scale=_D**-0.5,
                    deterministic=False,
                    module=mod,
                )
        assert len(_attn_stash[(mod, (8, _H, _D))]) == _MAX_INFLIGHT_ATTN_STASH
        _attn_stash[(mod, (8, _H, _D))].clear()

    @pytest.mark.gpu
    def test_varlen_return_lse_contract(self) -> None:
        q, k, v, _, cu = _seed_inputs(31)
        out = npu_flash_varlen_attn(q, k, v, cu, cu, _T, _T)
        assert isinstance(out, torch.Tensor)
        out3 = npu_flash_varlen_attn(q, k, v, cu, cu, _T, _T, return_lse=True)
        assert isinstance(out3, tuple) and len(out3) == 3
        assert torch.equal(out3[0], out)
        assert out3[1].dtype == torch.float32 and out3[2].dtype == torch.float32
        # The lse pair is only wired on the non-causal branch; causal must fail loudly.
        with pytest.raises(NotImplementedError):
            npu_flash_varlen_attn(q, k, v, cu, cu, _T, _T, causal=True, return_lse=True)

    @pytest.mark.gpu
    def test_replay_guard_paths_fail_loudly(self) -> None:
        mod = object()
        ins = _seed_inputs(51)
        q, k, v = (t.clone().requires_grad_() for t in ins[:3])
        w = ins[3]
        block = _make_block(ins[4], mod)
        out = torch.utils.checkpoint.checkpoint(block, q, k, v, use_reentrant=True)
        key = _stash_key(mod)
        entry = _attn_stash[key].pop()

        # 1) Empty lane at replay: an aborted backward between pops (poisoned lane) or an
        #    unwrapped block (VISION_RECOMPUTE_RATIO < 1.0) hits this guard.
        _attn_stash[key].clear()
        with pytest.raises(RuntimeError, match="stash is empty"):
            (out.float() * w.float()).sum().backward()
        q.grad = k.grad = v.grad = None

        # 2) Shape mismatch at pop: a foreign entry in the lane must not be restored.
        wrong_out = torch.empty(tuple(n * 2 for n in entry[0].shape), dtype=entry[0].dtype, device=entry[0].device)
        _attn_stash[key].append((wrong_out, entry[1], entry[2]))
        out = torch.utils.checkpoint.checkpoint(block, q, k, v, use_reentrant=True)
        _attn_stash[key].pop()
        with pytest.raises(RuntimeError, match="misalignment"):
            (out.float() * w.float()).sum().backward()
        q.grad = k.grad = v.grad = None

        # 3) Batch-size-1 contract at the wrapper entry.
        with pytest.raises(RuntimeError, match="batch size 1"):
            vit_flash_attn_skip_recompute(
                ins[0].unsqueeze(0).transpose(1, 2).repeat(2, 1, 1, 1),
                ins[1].unsqueeze(0).transpose(1, 2),
                ins[2].unsqueeze(0).transpose(1, 2),
                cu_seqlens_q=ins[4],
                cu_seqlens_k=ins[4],
                max_seqlen_q=_T,
                max_seqlen_k=_T,
                softmax_scale=_D**-0.5,
                deterministic=False,
                module=mod,
            )
