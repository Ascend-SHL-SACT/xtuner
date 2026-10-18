# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for fused MoE dispatch-alltoallv + grouped-matmul (``XTUNER_MOE_FUSED_A2A_GMM``).

Per the 744B expert sharding: contiguous expert→rank (``local_experts_start_id = e_local·rank``),
``n_routed_experts=32``, ``topk=4``, GLM-5.2-13B dims (hidden=6144, moe_intermediate=2048). EP
varies via ``XTUNER_TEST_WORLD_SIZE`` (8 or 16 on the 13B node; 32 on the cluster only):
``e_local = 32 // ep`` → 4 @ EP8, 2 @ EP16, 1 @ EP32.

Input layout: a cycling ``topk_ids`` (token t → experts [t, t+1, ..., t+topk-1] mod E) with a
RANK-DEPENDENT token count ``n_local = 8192 + 128·rank``. Each expert's count is then
``1024 + 16·rank`` — a multiple of 16 (the probe showed the fused CANN kernel tiles cleanly on
16-aligned counts; unaligned random counts risk ``561002`` tiling errors), but the count DIFFERS
per rank. That makes ``send_counts`` (rank-dependent, uniform per expert) ≠ ``recv_counts``
(source-rank-dependent) and ≠ a send-derived ``group_list``, so the backward **counts-swap** (dX
via the sibling op with send/recv swapped) and a **send-vs-recv group_list** bug are actually
caught — a symmetric/uniform input would mask both.

1. ``TestAlltoallvGmmCounts`` — ``_compute_counts`` vs an independent all-gather+index baseline.
2. ``TestAlltoallvGmmForward`` — ``AlltoallvPermuteGmm`` ``mm1_out`` vs eager
   ``dispatch``+``dispatch_postprocess``+``Ops.gmm``, per local expert as a row-multiset (robust
   to intra-expert row order, which the CANN permute kernel need not match between the fused op
   and ``npu_moe_token_permute``).
3. ``TestAlltoallvGmmBackward`` — dX + dW vs the eager reference. A per-local-expert CONSTANT
   ``grad_out`` makes the comparison order-robust (dW is a per-expert sum = commutative; dX is
   restored to the leaf's own sort-permuted order by the reverse a2a).
4. ``TestFusedDispatchMlpCombine`` — the full orchestrator vs the ``TorchAll2AllDispatcher``
   6-method lifecycle with a real SwiGLU MLP; forward + backward ``assert_close`` in the original
   token order (the robust end-to-end gate — the final ``unpermute`` cancels intra-expert order).
5. ``TestFusedRecomputeRegression`` — finite-grads gate for the activation-checkpoint recompute path
   (the a2a-nan fix). Wraps ``fused_dispatch_mlp_combine`` in an outer reentrant checkpoint,
   recreating the whole-decoder re-issue that nanned in the 13B run52 reproducer.
6. ``TestFusedRecvTokensHook`` — CPU-only contract gate (no distributed init): the fused
   orchestrator must fire ``recv_tokens_hook`` with ``sum(recv_counts)`` — the received routed-row
   count the dispatcher path reports and the step-end EP-load reduction consumes; its absence
   crashed every rank at the first MoE forward (``_ep_load_reduce`` got a 0-length dispatch column).
7. ``TestPlanLocalExpertChunks`` — CPU-only: the >256-entry chunk planner (balanced split, cap
   boundary, EP-over-cap error).
8. ``TestFusedChunkedCountsContract`` — CPU-only: with ``E_global > 256`` the orchestrator must
   issue per-chunk calls whose counts re-index the ``(source, chunk-expert)`` grid, fire the hook
   once with the full total, and reassemble every row (no loss/duplication).
9. ``TestFusedChunkedMatchesSingle`` — the >256 regression gate with the REAL kernels: forced
   2-chunk vs unchunked ``fused_dispatch_mlp_combine`` on the same leaves, forward + grads, plus
   an all-ranks-empty-chunk-0 case gating the per-chunk structural pad (all-zero outbound counts
   hard-fail the op without it).

Run, e.g. (``DistributedTestBase`` is torch's ``MultiProcessTestCase`` — a single pytest process
spawns the NPU ranks itself, so invoke directly, NOT under ``torchrun``; torchrun double-spawns and
collides on the HCCL port)::

    XTUNER_TEST_WORLD_SIZE=8 python -m pytest tests/ops/test_alltoallv_gmm.py -x --noconftest
    XTUNER_TEST_WORLD_SIZE=16 python -m pytest tests/ops/test_alltoallv_gmm.py -x --noconftest

``--noconftest`` skips ``tests/conftest.py``, whose eager ``xtuner._testing`` import pulls the
float8/triton stack (a pre-existing collection-time issue unrelated to this op).
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  registers torch.npu / transfers cuda->npu for the NPU dist tests
from torch.testing._internal.common_distributed import DistributedTestBase

from xtuner.v1.module.dispatcher.torch_all2all import TorchAll2AllDispatcher
from xtuner.v1.ops import unpermute
from xtuner.v1.ops.act_fn import get_act_fn
from xtuner.v1.ops.moe.npu import fused_a2a_gmm
from xtuner.v1.ops.moe.npu.fused_a2a_gmm import (
    AlltoallvPermuteGmm,
    _compute_counts,
    fused_dispatch_mlp_combine,
)
from xtuner.v1.ops.moe.npu.group_gemm import npu_group_gemm


# GLM-5.2-13B dims (the 744B MoE shapes at 13B scale).
HIDDEN = 6144
INTER = 2048
N_EXPERTS = 32
TOPK = 4
BASE_N_LOCAL = 8192  # + 128·rank per rank → 16-aligned per-expert counts, cross-rank asymmetric

# bf16 tolerances. The fused CANN gmm and ``Ops.gmm`` are different bf16 kernels, so forward
# diverges by ~bf16 precision; backward grads accumulate over rows (larger). A real bug (transposed
# weight, swapped counts, wrong permute) diverges by ~the value magnitudes, far outside these.
FWD_ATOL, FWD_RTOL = 5e-2, 5e-2
DX_ATOL, DX_RTOL = 5e-2, 5e-2
DW_ATOL, DW_RTOL = 1e-1, 1e-1
FULL_ATOL, FULL_RTOL = 2e-1, 1e-1  # orchestrator grads: bf16 + cross-expert a2a + permute


def _n_local(rank: int) -> int:
    """Rank-dependent token count: 16-aligned per-expert, cross-rank asymmetric (see module doc)."""
    return BASE_N_LOCAL + 128 * rank


def _cycling_topk_ids(n_local: int, device: torch.device) -> torch.Tensor:
    """Token t → experts [t, t+1, ..., t+TOPK-1] mod N_EXPERTS (distinct, uniform per expert).

    Every expert receives exactly ``n_local * TOPK / N_EXPERTS = n_local / 8`` tokens (a multiple
    of 16 for ``n_local`` a multiple of 128), so the fused op never hits an empty/tiny-expert
    tiling edge, and the count bookkeeping is clean to audit.
    """
    base = torch.arange(n_local, device=device).unsqueeze(1)
    offs = torch.arange(TOPK, device=device).unsqueeze(0)
    return ((base + offs) % N_EXPERTS).to(torch.int64)


def _make_weights(
    e_local: int, in_feat: int, out_feat: int, device: torch.device, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two independent leaf weights with the same init (one for fused, one for the eager ref)."""
    torch.manual_seed(seed)
    w_f = (torch.randn(e_local, out_feat, in_feat, device=device, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    torch.manual_seed(seed)
    w_e = (torch.randn(e_local, out_feat, in_feat, device=device, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    return w_f, w_e


def _topk_weights(n_local: int, device: torch.device) -> torch.Tensor:
    torch.manual_seed(777)
    return torch.softmax(torch.randn(n_local, TOPK, device=device, dtype=torch.float32), dim=-1).to(torch.bfloat16)


def _chunk0_empty_topk_ids(n_local: int, device: torch.device, size_c: int, e_local: int) -> torch.Tensor:
    """Route every token into chunk-1 local experts (``>= size_c``) → chunk 0 sends 0 rows.

    Regression input for the chunked path's structural pad: with every rank's tokens off chunk 0,
    that call's outbound count list is all zero on EVERY rank, which ``npu_alltoallv_gmm``
    rejects (561002) unless the pad row is present. Falls back to the plain cycling ids when the
    geometry has no chunk 1 (``e_local <= size_c``, e.g. ws=32 where ``e_local=1``).
    """
    base = torch.arange(n_local, device=device).unsqueeze(1)
    offs = torch.arange(TOPK, device=device).unsqueeze(0)
    mixed = (base + offs) % N_EXPERTS
    if e_local <= size_c:
        return mixed.to(torch.int64)
    owner = mixed // e_local
    local = size_c + (mixed + offs) % (e_local - size_c)
    return (owner * e_local + local).to(torch.int64)


def _assert_per_expert_close(
    fused: torch.Tensor, eager: torch.Tensor, group_list: list[int], *, atol: float, rtol: float
) -> None:
    """Compare two local-expert-ordered tensors as per-expert row-multisets.

    ``npu_moe_token_permute`` (eager) and the fused op's internal permute are both CANN kernels
    that need not produce the same intra-expert row order, so a strict elementwise compare is
    fragile. Per local expert, the SET of gmm-output rows must match (the gmm is per-expert and
    row-independent); ``sort(dim=0)`` compares the per-column multiset, which a real bug (wrong
    weight, swapped counts, wrong-expert tokens) breaks by ~the value magnitudes.
    """
    assert fused.shape == eager.shape, f"shape mismatch: {fused.shape} vs {eager.shape}"
    offset = 0
    for expert, size in enumerate(group_list):
        if size > 0:
            torch.testing.assert_close(
                fused[offset : offset + size].sort(dim=0).values.float(),
                eager[offset : offset + size].sort(dim=0).values.float(),
                atol=atol,
                rtol=rtol,
                msg=f"expert {expert} (size={size})",
            )
        offset += size


def _eager_dispatch_to_local_expert(
    dispatcher: TorchAll2AllDispatcher,
    sort_permuted_input: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
):
    """Run ``dispatch`` + ``dispatch_postprocess`` on the (already sort-permuted) input.

    Returns the ``dispatch_postprocess`` result, whose ``hidden_states`` is the a2a'd +
    permuted-to-local-expert input (= the fused op's ``permute_out``) and whose
    ``tokens_per_expert`` is the per-local-expert received count (= the fused ``group_list``).
    """
    pre = {"hidden_states": sort_permuted_input, "topk_ids": topk_ids}
    dispatched = dispatcher.dispatch(pre_dispatched=pre, topk_weights=topk_weights, decoding=False)
    return dispatcher.dispatch_postprocess(pre_dispatched=pre, dispatched=dispatched, decoding=False)


def _eager_moe_lifecycle(
    dispatcher: TorchAll2AllDispatcher,
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    act_fn,
) -> torch.Tensor:
    """Full 6-method ``TorchAll2AllDispatcher`` lifecycle with a real SwiGLU expert MLP."""
    pre = dispatcher.dispatch_preprocess(hidden_states=hidden, topk_ids=topk_ids, topk_weights=topk_weights)
    dispatched = dispatcher.dispatch(pre_dispatched=pre, topk_weights=topk_weights, decoding=False)
    post = dispatcher.dispatch_postprocess(pre_dispatched=pre, dispatched=dispatched, decoding=False)
    gate_up = npu_group_gemm(post["hidden_states"], w1, post["tokens_per_expert"])
    intermediate = act_fn(gate_up, split_dim=-1)
    experts_out = npu_group_gemm(intermediate, w2, post["tokens_per_expert"])
    pre_combined = dispatcher.combine_preprocess(
        hidden_states=experts_out,
        pre_dispatched=pre,
        dispatched=dispatched,
        post_dispatched=post,
        decoding=False,
    )
    combined = dispatcher.combine(
        pre_dispatched=pre,
        dispatched=dispatched,
        post_dispatched=post,
        pre_combined=pre_combined,
        decoding=False,
    )
    result = dispatcher.combine_postprocess(
        pre_dispatched=pre,
        dispatched=dispatched,
        post_dispatched=post,
        pre_combined=pre_combined,
        combined=combined,
    )
    return result["hidden_states"]


@pytest.mark.gpu
class TestAlltoallvGmmCounts(DistributedTestBase):
    """``_compute_counts`` vs an independent all-gather+index baseline (no all-to-all)."""

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))

    def _set_device(self) -> None:
        if self.rank >= 0:
            torch.npu.set_device(self.rank % torch.npu.device_count())

    def test_counts_match_baseline(self):
        self._set_device()
        self.create_pg("cuda")
        ws = dist.get_world_size()
        rank = dist.get_rank()
        assert N_EXPERTS % ws == 0, f"N_EXPERTS={N_EXPERTS} must divide ep={ws}"
        e_local = N_EXPERTS // ws
        device = torch.device("cuda", rank % torch.npu.device_count())
        n_local = _n_local(rank)

        topk_ids = _cycling_topk_ids(n_local, device)
        ep_group = dist.group.WORLD

        send, recv, gl = _compute_counts(topk_ids, N_EXPERTS, ep_group)

        # Baseline: all-gather each rank's per-global-expert counts, then index this rank's local
        # experts out of every source rank's counts (no all-to-all → independent of _compute_counts).
        local_counts = torch.histc(topk_ids, bins=N_EXPERTS, min=0, max=N_EXPERTS)
        all_counts = [torch.empty_like(local_counts) for _ in range(ws)]
        dist.all_gather(all_counts, local_counts, group=ep_group)
        my_start = e_local * rank
        recv_grid = torch.stack([all_counts[src][my_start : my_start + e_local] for src in range(ws)], dim=0)
        send_ref = local_counts.to("cpu").tolist()
        recv_ref = recv_grid.flatten().to("cpu").tolist()
        gl_ref = recv_grid.sum(dim=0).to("cpu").tolist()

        assert send == send_ref, f"send_counts: {send} != {send_ref}"
        assert recv == recv_ref, f"recv_counts: {recv} != {recv_ref}"
        assert gl == gl_ref, f"group_list: {gl} != {gl_ref}"
        # Invariants the fused op's tiler relies on. With rank-asymmetric n_local, per-rank
        # sum(send) (= n_local·TOPK, what this rank dispatches) ≠ sum(recv) (what this rank
        # receives = Σ over source ranks); only the GLOBAL total conserves. The exact list
        # equality recv == recv_ref above (built from an all-gather, no all-to-all) is the strong
        # check; these are secondary per-rank sanity invariants.
        assert sum(send) == n_local * TOPK, f"sum(send)={sum(send)} != {n_local * TOPK}"
        assert sum(gl) == sum(recv), f"sum(gl)={sum(gl)} != sum(recv)={sum(recv)}"
        assert len(gl) == e_local, f"len(group_list)={len(gl)} != e_local={e_local}"


@pytest.mark.gpu
class TestAlltoallvGmmForward(DistributedTestBase):
    """``AlltoallvPermuteGmm`` forward vs eager dispatch+postprocess+``Ops.gmm``."""

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))

    def _set_device(self) -> None:
        if self.rank >= 0:
            torch.npu.set_device(self.rank % torch.npu.device_count())

    def test_forward_matches_eager(self):
        self._set_device()
        self.create_pg("cuda")
        ws = dist.get_world_size()
        rank = dist.get_rank()
        e_local = N_EXPERTS // ws
        device = torch.device("cuda", rank % torch.npu.device_count())
        dtype = torch.bfloat16
        n_local = _n_local(rank)

        torch.manual_seed(1234 + rank)
        hidden = torch.randn(n_local, HIDDEN, device=device, dtype=dtype)
        topk_ids = _cycling_topk_ids(n_local, device)
        topk_weights = _topk_weights(n_local, device)

        dispatcher = TorchAll2AllDispatcher(
            n_routed_experts=N_EXPERTS, process_group=dist.group.WORLD, training_dtype="bf16"
        )
        pre0 = dispatcher.dispatch_preprocess(hidden_states=hidden, topk_ids=topk_ids, topk_weights=topk_weights)
        # The fused op takes the SORT-PERMUTED dispatch input (pre0["hidden_states"]).
        gmm_x = pre0["hidden_states"].detach().clone().requires_grad_(True)
        w1_f, w1_e = _make_weights(e_local, HIDDEN, 2 * INTER, device, seed=42 + rank)

        send, recv, gl = _compute_counts(topk_ids, N_EXPERTS, dist.group.WORLD)
        # The op takes the caller-pre-transposed weight [E, in, out] (native is [E, out, in]).
        w1_f_fwd = w1_f.transpose(1, 2).contiguous()
        fused_mm1 = AlltoallvPermuteGmm.apply(gmm_x, w1_f, w1_f_fwd, dist.group.WORLD, send, recv, gl)

        post = _eager_dispatch_to_local_expert(dispatcher, gmm_x, topk_ids, topk_weights)
        # Same weight prep as the fused op: native [E,out,in] → transpose(1,2) → [E,in,out].
        eager_mm1 = npu_group_gemm(post["hidden_states"], w1_e, post["tokens_per_expert"])

        assert fused_mm1.shape == eager_mm1.shape, f"mm1_out shape: fused {fused_mm1.shape} vs eager {eager_mm1.shape}"
        _assert_per_expert_close(fused_mm1, eager_mm1, gl, atol=FWD_ATOL, rtol=FWD_RTOL)


@pytest.mark.gpu
class TestAlltoallvGmmBackward(DistributedTestBase):
    """``AlltoallvPermuteGmm`` backward (dX + dW) vs the eager reference.

    A per-local-expert CONSTANT ``grad_out`` (all rows of expert e share one vector ``v_e``) makes
    the comparison order-robust: dW is a per-expert sum (commutative), and dX is restored to the
    leaf's own sort-permuted order by the reverse a2a, so both grads are directly comparable
    elementwise despite any intra-expert forward-order difference.
    """

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))

    def _set_device(self) -> None:
        if self.rank >= 0:
            torch.npu.set_device(self.rank % torch.npu.device_count())

    def _constant_per_expert_grad(self, group_list: list[int], out_feat: int, device: torch.device):
        torch.manual_seed(2024)
        pieces = [torch.randn(out_feat, device=device, dtype=torch.bfloat16) * 0.02 for _ in group_list]
        return torch.cat([v.unsqueeze(0).expand(size, out_feat) for size, v in zip(group_list, pieces)], dim=0)

    def test_backward_matches_eager(self):
        self._set_device()
        self.create_pg("cuda")
        ws = dist.get_world_size()
        rank = dist.get_rank()
        e_local = N_EXPERTS // ws
        device = torch.device("cuda", rank % torch.npu.device_count())
        dtype = torch.bfloat16
        n_local = _n_local(rank)

        torch.manual_seed(1234 + rank)
        hidden = torch.randn(n_local, HIDDEN, device=device, dtype=dtype)
        topk_ids = _cycling_topk_ids(n_local, device)
        topk_weights = _topk_weights(n_local, device)

        dispatcher = TorchAll2AllDispatcher(
            n_routed_experts=N_EXPERTS, process_group=dist.group.WORLD, training_dtype="bf16"
        )
        pre0 = dispatcher.dispatch_preprocess(hidden_states=hidden, topk_ids=topk_ids, topk_weights=topk_weights)
        send, recv, gl = _compute_counts(topk_ids, N_EXPERTS, dist.group.WORLD)

        # Two independent leaves (same init) for fused vs eager so backward grads don't accumulate.
        gmm_x_f = pre0["hidden_states"].detach().clone().requires_grad_(True)
        gmm_x_e = pre0["hidden_states"].detach().clone().requires_grad_(True)
        w1_f, w1_e = _make_weights(e_local, HIDDEN, 2 * INTER, device, seed=42 + rank)
        # The op takes the caller-pre-transposed weight [E, in, out] (native is [E, out, in]).
        w1_f_fwd = w1_f.transpose(1, 2).contiguous()

        fused_mm1 = AlltoallvPermuteGmm.apply(gmm_x_f, w1_f, w1_f_fwd, dist.group.WORLD, send, recv, gl)
        post = _eager_dispatch_to_local_expert(dispatcher, gmm_x_e, topk_ids, topk_weights)
        eager_mm1 = npu_group_gemm(post["hidden_states"], w1_e, post["tokens_per_expert"])

        grad_out = self._constant_per_expert_grad(gl, 2 * INTER, device)
        fused_mm1.backward(grad_out)
        eager_mm1.backward(grad_out)

        # dX: both grads are in gmm_x's own (sort-permuted) order — directly comparable.
        assert gmm_x_f.grad is not None and gmm_x_e.grad is not None
        torch.testing.assert_close(
            gmm_x_f.grad.float(), gmm_x_e.grad.float(), atol=DX_ATOL, rtol=DX_RTOL, msg="dX (gmm_x)"
        )
        # dW: native [E, out, in], a per-expert sum → order-independent.
        assert w1_f.grad is not None and w1_e.grad is not None
        torch.testing.assert_close(w1_f.grad.float(), w1_e.grad.float(), atol=DW_ATOL, rtol=DW_RTOL, msg="dW (w1)")


@pytest.mark.gpu
class TestFusedDispatchMlpCombine(DistributedTestBase):
    """Full orchestrator vs the 6-method ``TorchAll2AllDispatcher`` lifecycle (the robust gate).

    Both paths end in the original token order (the final ``unpermute`` cancels intra-expert
    ordering differences), so forward + backward are directly comparable elementwise.
    """

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))

    def _set_device(self) -> None:
        if self.rank >= 0:
            torch.npu.set_device(self.rank % torch.npu.device_count())

    def test_forward_and_backward_match_eager(self):
        self._set_device()
        self.create_pg("cuda")
        ws = dist.get_world_size()
        rank = dist.get_rank()
        e_local = N_EXPERTS // ws
        device = torch.device("cuda", rank % torch.npu.device_count())
        dtype = torch.bfloat16
        n_local = _n_local(rank)

        torch.manual_seed(1234 + rank)
        hidden_f = torch.randn(n_local, HIDDEN, device=device, dtype=dtype, requires_grad=True)
        hidden_e = hidden_f.detach().clone().requires_grad_(True)
        topk_ids = _cycling_topk_ids(n_local, device)
        topk_weights = _topk_weights(n_local, device)

        w1_f, w1_e = _make_weights(e_local, HIDDEN, 2 * INTER, device, seed=42 + rank)
        w2_f, w2_e = _make_weights(e_local, INTER, HIDDEN, device, seed=99 + rank)
        act_fn = get_act_fn("swiglu")

        dispatcher = TorchAll2AllDispatcher(
            n_routed_experts=N_EXPERTS, process_group=dist.group.WORLD, training_dtype="bf16"
        )

        # Fused: dispatch_preprocess (sort) + fused_dispatch_mlp_combine + final unpermute.
        pre_f = dispatcher.dispatch_preprocess(hidden_states=hidden_f, topk_ids=topk_ids, topk_weights=topk_weights)
        combined_f = fused_dispatch_mlp_combine(
            pre_f["hidden_states"], w1_f, w2_f, act_fn, topk_ids, N_EXPERTS, dist.group.WORLD
        )
        out_f = unpermute(combined_f, pre_f["row_id_map"], probs=topk_weights)

        # Eager: full 6-method lifecycle with the real SwiGLU MLP.
        out_e = _eager_moe_lifecycle(dispatcher, hidden_e, topk_ids, topk_weights, w1_e, w2_e, act_fn)

        torch.testing.assert_close(out_f.float(), out_e.float(), atol=FWD_ATOL, rtol=FWD_RTOL, msg="forward combined")

        # Backward with the same per-token grad (original order → directly comparable).
        torch.manual_seed(31337)
        grad_out = torch.randn(n_local, HIDDEN, device=device, dtype=dtype)
        out_f.backward(grad_out)
        out_e.backward(grad_out)

        torch.testing.assert_close(
            hidden_f.grad.float(), hidden_e.grad.float(), atol=FULL_ATOL, rtol=FULL_RTOL, msg="dHidden"
        )
        torch.testing.assert_close(w1_f.grad.float(), w1_e.grad.float(), atol=FULL_ATOL, rtol=FULL_RTOL, msg="dW1")
        torch.testing.assert_close(w2_f.grad.float(), w2_e.grad.float(), atol=FULL_ATOL, rtol=FULL_RTOL, msg="dW2")


@pytest.mark.gpu
class TestFusedRecomputeRegression(DistributedTestBase):
    """Finite-grads gate for the activation-checkpoint recompute path (the a2a-nan fix).

    The fused MoE dispatch op (``npu_alltoallv_gmm``) nans when its forward is RE-ISSUED during
    activation-checkpoint recompute. In the 13B run52 reproducer the whole-decoder
    ``checkpoint_wrapper`` re-ran ``fused_dispatch_mlp_combine`` in backward; the re-issued op's
    grouped-matmul stage nans because its forward-only weight clone -- built under ``enable_grad``
    as a grad-tracked ``.contiguous()`` CopySlices non-leaf -- has its buffer recycled before the
    gmm reads it (the a2a stage, reading the stable activation leaf, stays finite). The fix
    materializes ``w1_fwd``/``w2_fwd`` under ``torch.no_grad()`` in ``fused_dispatch_mlp_combine``
    so the gmm reads a stable detached leaf and the re-issue stays finite (13B run67/68/69, 10-step
    stable, finite ``grad_norm``, ~38.5 GB flat -- lean, no extra saved tensors).

    The nan is 13B-context-specific (FSDP2 + sequence-parallel + recompute + the allocator pressure
    that recycles the clone's buffer; the isolated 8-rank probe is 8/8 finite even under recompute),
    so this op-level case is a FINITE-GRADS GATE, not a reproduction. It wraps
    ``fused_dispatch_mlp_combine`` in an OUTER reentrant checkpoint -- recreating the whole-decoder
    re-issue path that nanned in 13B -- and asserts every grad is finite. The 13B run is the
    end-to-end gate. A regression (the op nans, or the weight-clone ``no_grad`` detach is dropped so
    the gmm reads a recycled grad-tracked clone again) surfaces in the 13B run; this gate keeps the
    recompute path exercised at the op level.
    """

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))

    def _set_device(self) -> None:
        if self.rank >= 0:
            torch.npu.set_device(self.rank % torch.npu.device_count())

    def test_finite_grads_under_recompute(self):
        self._set_device()
        self.create_pg("cuda")
        ws = dist.get_world_size()
        rank = dist.get_rank()
        e_local = N_EXPERTS // ws
        device = torch.device("cuda", rank % torch.npu.device_count())
        dtype = torch.bfloat16
        n_local = _n_local(rank)

        torch.manual_seed(1234 + rank)
        hidden = torch.randn(n_local, HIDDEN, device=device, dtype=dtype, requires_grad=True)
        topk_ids = _cycling_topk_ids(n_local, device)
        topk_weights = _topk_weights(n_local, device)
        w1, _ = _make_weights(e_local, HIDDEN, 2 * INTER, device, seed=42 + rank)
        w2, _ = _make_weights(e_local, INTER, HIDDEN, device, seed=99 + rank)
        act_fn = get_act_fn("swiglu")

        dispatcher = TorchAll2AllDispatcher(
            n_routed_experts=N_EXPERTS, process_group=dist.group.WORLD, training_dtype="bf16"
        )
        pre = dispatcher.dispatch_preprocess(hidden_states=hidden, topk_ids=topk_ids, topk_weights=topk_weights)

        def _run(x: torch.Tensor) -> torch.Tensor:
            combined = fused_dispatch_mlp_combine(x, w1, w2, act_fn, topk_ids, N_EXPERTS, dist.group.WORLD)
            return unpermute(combined, pre["row_id_map"], probs=topk_weights)

        # Outer reentrant checkpoint: recreate the whole-decoder re-issue path (the original nan
        # source). The re-issued fused_dispatch_mlp_combine re-runs npu_alltoallv_gmm; the no_grad
        # weight clones (the fix) keep its grouped-matmul stage finite under the recompute.
        out = torch.utils.checkpoint.checkpoint(_run, pre["hidden_states"], use_reentrant=True)

        torch.manual_seed(31337)
        grad_out = torch.randn(n_local, HIDDEN, device=device, dtype=dtype)
        out.backward(grad_out)

        for name, grad in (("dHidden", hidden.grad), ("dW1", w1.grad), ("dW2", w2.grad)):
            assert grad is not None, f"{name} is None"
            assert torch.isfinite(grad).all(), f"{name} has nan/inf (max={grad.abs().max()})"


class TestFusedRecvTokensHook:
    """Contract gate: the fused path must drive ``recv_tokens_hook`` like the dispatcher path.

    The fused branch in ``MoEDecoderLayer._pre_moe_forward`` returns before ``dispatcher.dispatch``,
    so nothing fired ``recv_tokens_hook``; ``_ep_recv_tokens`` stayed empty and the step-end EP-load
    reduction crashed on every rank (``aclnnAmax EZ1001: Expected reducution dim 2 to have non-zero
    size`` — ``_ep_load_reduce``'s ``view`` got a 0-length dispatch column). The fix fires the hook
    inside ``fused_dispatch_mlp_combine`` with ``sum(recv_counts)`` — the same received routed-row
    count the dispatcher path reports as ``post_dispatched["hidden_states"].shape[0]``. These tests
    pin the contract with the counts and the two autograd ops stubbed (CPU-only, no distributed
    init; the CANN kernels are never invoked).
    """

    @staticmethod
    def _stub_ops(monkeypatch):
        # ``apply`` returns its first argument, so the orchestrator runs end-to-end and the
        # activation is the identity of the stub's pass-through.
        stub = type("_StubGmm", (), {"apply": staticmethod(lambda x, *args, **kwargs: x)})
        monkeypatch.setattr(fused_a2a_gmm, "AlltoallvPermuteGmm", stub)
        monkeypatch.setattr(fused_a2a_gmm, "GmmUnpermuteAlltoallv", stub)

    def test_hook_receives_total_recv_counts(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            fused_a2a_gmm,
            "_compute_counts",
            lambda topk_ids, n_routed_experts, ep_group: ([3, 4], [2, 5], [7]),
        )
        self._stub_ops(monkeypatch)
        out = fused_a2a_gmm.fused_dispatch_mlp_combine(
            hidden_states=torch.zeros(4, 8),
            w1=torch.zeros(2, 8, 8),
            w2=torch.zeros(2, 8, 8),
            act_fn=lambda x, split_dim=-1: x,
            topk_ids=torch.zeros(4, 2, dtype=torch.long),
            n_routed_experts=2,
            ep_group=None,
            recv_tokens_hook=calls.append,
        )
        assert calls == [7]
        assert out.shape == (4, 8)

    def test_hook_optional(self, monkeypatch):
        monkeypatch.setattr(
            fused_a2a_gmm,
            "_compute_counts",
            lambda topk_ids, n_routed_experts, ep_group: ([0, 0], [0, 0], [0]),
        )
        self._stub_ops(monkeypatch)
        out = fused_a2a_gmm.fused_dispatch_mlp_combine(
            hidden_states=torch.zeros(4, 8),
            w1=torch.zeros(2, 8, 8),
            w2=torch.zeros(2, 8, 8),
            act_fn=lambda x, split_dim=-1: x,
            topk_ids=torch.zeros(4, 2, dtype=torch.long),
            n_routed_experts=2,
            ep_group=None,
            recv_tokens_hook=None,
        )
        assert out.shape == (4, 8)


class TestPlanLocalExpertChunks:
    """CPU-only: the >256-entry chunk planner (``_plan_local_expert_chunks``).

    GLM-5.3-Flash routes 288 experts at EP32 → 288 send/recv count entries, and the cluster CANN
    tiling rejects lists over 256 (``grouped_mat_mul_allto_allv_tiling.cpp:350``, "The length of
    recvCnts[288] should be in (0, 256]" — job-dyb-q35-xtuner-52). The planner must return one full
    range within the cap, a balanced split over it, and refuse EP sizes that cannot fit at all.
    """

    def test_within_cap_single_chunk(self):
        assert fused_a2a_gmm._plan_local_expert_chunks(8, 4) == [(0, 4)]

    def test_cap_boundary_single_chunk(self):
        # 32 * 8 = 256 entries exactly — still the unchunked path.
        assert fused_a2a_gmm._plan_local_expert_chunks(32, 8) == [(0, 8)]

    def test_over_cap_balanced(self):
        # GLM-5.3-Flash shape: EP32 x 9 local experts → [5, 4], entries 160/128.
        assert fused_a2a_gmm._plan_local_expert_chunks(32, 9) == [(0, 5), (5, 4)]

    def test_over_cap_uneven(self):
        # 64 * 7 = 448 entries; per-chunk max 4 → [4, 3], entries 256/192.
        assert fused_a2a_gmm._plan_local_expert_chunks(64, 7) == [(0, 4), (4, 3)]

    def test_ep_over_cap_raises(self):
        # Each call needs at least ep_size entries, so EP > 256 cannot fit regardless of chunking.
        with pytest.raises(ValueError, match="cap"):
            fused_a2a_gmm._plan_local_expert_chunks(300, 1)


class TestFusedChunkedCountsContract:
    """CPU-only contract gate for the chunked orchestrator (``E_global > 256``, stubbed ops).

    Per chunk the two autograd ops must receive the counts re-indexed onto the chunk's
    ``(source, chunk-expert)`` grid — ``send_c[s * size_c + j] == send_counts[s * e_local + c0 + j]``
    (and likewise recv) — PLUS the structural pad: +1 on this rank's own ``(source=self, j=0)``
    entry (``ep_rank=0`` here), matching the pad row the orchestrator inserts into each chunk's
    gathered rows and slices off the combine output. The weight slices cover exactly the chunk's
    local experts, and the hook must still fire ONCE with the unchanged full total. The identity
    stubs return the gathered rows unchanged, so the reassembled output must equal the input
    row-for-row (no row lost or duplicated across chunks, pad included).
    """

    EP, E_LOCAL = 2, 200  # E_global = 400 > 256 → chunked; per-chunk max 128 → 2 chunks of 100

    @staticmethod
    def _make_stub(calls: list[tuple[str, tuple]], tag: str):
        def _apply(x, *args):
            calls.append((tag, args))
            return x

        return type("_StubGmm", (), {"apply": staticmethod(_apply)})

    def test_chunked_counts_hook_and_reassembly(self, monkeypatch):
        ep, e_local = self.EP, self.E_LOCAL
        n_experts = ep * e_local
        send_counts = [i % 4 + 1 for i in range(n_experts)]
        recv_counts = [(s * e_local + j) % 5 for s in range(ep) for j in range(e_local)]
        group_list = [sum(recv_counts[s * e_local + j] for s in range(ep)) for j in range(e_local)]
        monkeypatch.setattr(
            fused_a2a_gmm,
            "_compute_counts",
            lambda topk_ids, n_routed_experts, ep_group: (send_counts, recv_counts, group_list),
        )
        # The chunked branch reads the EP-group rank for the pad attribution; there is no
        # distributed init here, so pin it to 0 (pad lands on each chunk's first local expert).
        monkeypatch.setattr(dist, "get_rank", lambda group=None: 0)

        calls: list[tuple[str, tuple]] = []
        monkeypatch.setattr(fused_a2a_gmm, "AlltoallvPermuteGmm", self._make_stub(calls, "dispatch"))
        monkeypatch.setattr(fused_a2a_gmm, "GmmUnpermuteAlltoallv", self._make_stub(calls, "combine"))

        n_send = sum(send_counts)
        hidden = torch.arange(n_send * 8, dtype=torch.float32).view(n_send, 8)
        w1 = torch.zeros(e_local, 8, 8)
        w2 = torch.zeros(e_local, 8, 8)
        hook_calls: list[int] = []
        out = fused_a2a_gmm.fused_dispatch_mlp_combine(
            hidden_states=hidden,
            w1=w1,
            w2=w2,
            act_fn=lambda x, split_dim=-1: x,
            topk_ids=torch.zeros(1, 1, dtype=torch.long),
            n_routed_experts=n_experts,
            ep_group=None,
            recv_tokens_hook=hook_calls.append,
        )

        # The hook fires once, with the unchanged full received-row total.
        assert hook_calls == [sum(recv_counts)]
        # Identity reassembly: every row returns exactly once (bijective gather + put).
        assert out.shape == hidden.shape and torch.equal(out, hidden)

        # Per-chunk op contract: 2 chunks of 100 local experts, counts re-indexed onto the
        # (source, chunk-expert) grid, weights sliced to the chunk's local experts.
        for tag in ("dispatch", "combine"):
            chunk_calls = [args for t, args in calls if t == tag]
            assert len(chunk_calls) == 2, f"{tag}: expected 2 chunk calls, got {len(chunk_calls)}"
            for chunk_i, args in enumerate(chunk_calls):
                weight, _w_fwd, _ep_group, send_c, recv_c, gl_c = args
                c0 = chunk_i * 100
                size_c = 100
                assert weight.shape[0] == size_c and torch.equal(weight, w1[c0 : c0 + size_c])
                ref_send = [send_counts[s * e_local + c0 + j] for s in range(ep) for j in range(size_c)]
                ref_recv = [recv_counts[s * e_local + c0 + j] for s in range(ep) for j in range(size_c)]
                ref_gl = group_list[c0 : c0 + size_c]
                # Structural pad: ep_rank=0 → +1 on this rank's own (source=self, j=0) entry.
                ref_send[0] += 1
                ref_recv[0] += 1
                ref_gl[0] += 1
                assert send_c == ref_send, f"{tag} chunk {chunk_i}: send counts mis-indexed"
                assert recv_c == ref_recv, f"{tag} chunk {chunk_i}: recv counts mis-indexed"
                assert gl_c == ref_gl, f"{tag} chunk {chunk_i}: group_list mis-sliced"


@pytest.mark.gpu
class TestFusedChunkedMatchesSingle(DistributedTestBase):
    """Forced chunking vs the unchunked fused call, REAL kernels, forward + grads (>256 gate).

    Runs ``fused_dispatch_mlp_combine`` twice on the same leaves — unchunked, then with
    ``_MAX_FUSED_COUNT_ENTRIES`` lowered to force the 2-chunk split — and compares the
    unpermuted output and dHidden/dW1/dW2. Both sides run the same CANN ops over the same
    per-expert row sets, so a wrong per-chunk (source, local-expert) count mapping, a
    mis-gathered send block, or a reassembly order bug diverges by ~the value magnitudes, far
    outside the tolerances.
    """

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))

    def _set_device(self) -> None:
        if self.rank >= 0:
            torch.npu.set_device(self.rank % torch.npu.device_count())

    def test_chunked_forward_and_backward_match_single(self):
        self._set_device()
        self.create_pg("cuda")
        ws = dist.get_world_size()
        rank = dist.get_rank()
        e_local = N_EXPERTS // ws
        device = torch.device("cuda", rank % torch.npu.device_count())
        dtype = torch.bfloat16
        n_local = _n_local(rank)

        torch.manual_seed(1234 + rank)
        hidden0 = torch.randn(n_local, HIDDEN, device=device, dtype=dtype)
        topk_ids = _cycling_topk_ids(n_local, device)
        topk_weights = _topk_weights(n_local, device)
        act_fn = get_act_fn("swiglu")
        dispatcher = TorchAll2AllDispatcher(
            n_routed_experts=N_EXPERTS, process_group=dist.group.WORLD, training_dtype="bf16"
        )
        pre = dispatcher.dispatch_preprocess(hidden_states=hidden0, topk_ids=topk_ids, topk_weights=topk_weights)

        x_s = pre["hidden_states"].detach().clone().requires_grad_(True)
        x_c = pre["hidden_states"].detach().clone().requires_grad_(True)
        w1_s, w1_c = _make_weights(e_local, HIDDEN, 2 * INTER, device, seed=42 + rank)
        w2_s, w2_c = _make_weights(e_local, INTER, HIDDEN, device, seed=99 + rank)

        combined_s = fused_dispatch_mlp_combine(x_s, w1_s, w2_s, act_fn, topk_ids, N_EXPERTS, dist.group.WORLD)
        out_s = unpermute(combined_s, pre["row_id_map"], probs=topk_weights)

        # Force the split at any world size that allows it: cap EP·(e_local-1) so the planner must
        # emit ≥ 2 chunks whenever e_local > 1 (ws=8 → [(0,2),(2,2)], ws=16 → [(0,1),(1,1)];
        # ws=32 → e_local=1 cannot chunk since the smallest call already needs ep entries, so the
        # A/B degenerates to single-vs-single). unittest-style ranks can't use the monkeypatch
        # fixture.
        old_cap = fused_a2a_gmm._MAX_FUSED_COUNT_ENTRIES
        fused_a2a_gmm._MAX_FUSED_COUNT_ENTRIES = ws * max(1, e_local - 1)
        try:
            plan = fused_a2a_gmm._plan_local_expert_chunks(ws, e_local)
            if e_local > 1:
                assert len(plan) > 1, f"forced cap did not trigger chunking: {plan}"
            if ws == 8:
                assert plan == [(0, 2), (2, 2)]
            combined_c = fused_dispatch_mlp_combine(x_c, w1_c, w2_c, act_fn, topk_ids, N_EXPERTS, dist.group.WORLD)
        finally:
            fused_a2a_gmm._MAX_FUSED_COUNT_ENTRIES = old_cap
        out_c = unpermute(combined_c, pre["row_id_map"], probs=topk_weights)

        torch.testing.assert_close(out_s.float(), out_c.float(), atol=FWD_ATOL, rtol=FWD_RTOL, msg="forward combined")

        torch.manual_seed(31337)
        grad_out = torch.randn(n_local, HIDDEN, device=device, dtype=dtype)
        out_s.backward(grad_out)
        out_c.backward(grad_out)

        torch.testing.assert_close(x_s.grad.float(), x_c.grad.float(), atol=FULL_ATOL, rtol=FULL_RTOL, msg="dHidden")
        torch.testing.assert_close(w1_s.grad.float(), w1_c.grad.float(), atol=FULL_ATOL, rtol=FULL_RTOL, msg="dW1")
        torch.testing.assert_close(w2_s.grad.float(), w2_c.grad.float(), atol=FULL_ATOL, rtol=FULL_RTOL, msg="dW2")

    def test_chunked_empty_send_chunk_padded(self):
        """Regression: a chunk whose send rows are empty on every rank must not 561002-crash.

        All tokens route into chunk-1 local experts, so chunk 0's outbound count list is all zero
        on EVERY rank — the shape the op rejects without the structural pad. The unchunked call
        tolerates the same routing, so both sides must still agree forward + grads.
        """
        self._set_device()
        self.create_pg("cuda")
        ws = dist.get_world_size()
        rank = dist.get_rank()
        e_local = N_EXPERTS // ws
        device = torch.device("cuda", rank % torch.npu.device_count())
        dtype = torch.bfloat16
        n_local = _n_local(rank)

        old_cap = fused_a2a_gmm._MAX_FUSED_COUNT_ENTRIES
        forced_cap = ws * max(1, e_local - 1)
        fused_a2a_gmm._MAX_FUSED_COUNT_ENTRIES = forced_cap
        try:
            size_c = fused_a2a_gmm._plan_local_expert_chunks(ws, e_local)[0][1]
        finally:
            fused_a2a_gmm._MAX_FUSED_COUNT_ENTRIES = old_cap
        if e_local > 1:
            assert size_c < e_local, "expected a chunk 1 to route into"

        torch.manual_seed(1234 + rank)
        hidden0 = torch.randn(n_local, HIDDEN, device=device, dtype=dtype)
        topk_ids = _chunk0_empty_topk_ids(n_local, device, size_c, e_local)
        topk_weights = _topk_weights(n_local, device)
        act_fn = get_act_fn("swiglu")
        dispatcher = TorchAll2AllDispatcher(
            n_routed_experts=N_EXPERTS, process_group=dist.group.WORLD, training_dtype="bf16"
        )
        pre = dispatcher.dispatch_preprocess(hidden_states=hidden0, topk_ids=topk_ids, topk_weights=topk_weights)

        x_s = pre["hidden_states"].detach().clone().requires_grad_(True)
        x_c = pre["hidden_states"].detach().clone().requires_grad_(True)
        w1_s, w1_c = _make_weights(e_local, HIDDEN, 2 * INTER, device, seed=42 + rank)
        w2_s, w2_c = _make_weights(e_local, INTER, HIDDEN, device, seed=99 + rank)

        combined_s = fused_dispatch_mlp_combine(x_s, w1_s, w2_s, act_fn, topk_ids, N_EXPERTS, dist.group.WORLD)
        out_s = unpermute(combined_s, pre["row_id_map"], probs=topk_weights)

        fused_a2a_gmm._MAX_FUSED_COUNT_ENTRIES = forced_cap
        try:
            combined_c = fused_dispatch_mlp_combine(x_c, w1_c, w2_c, act_fn, topk_ids, N_EXPERTS, dist.group.WORLD)
        finally:
            fused_a2a_gmm._MAX_FUSED_COUNT_ENTRIES = old_cap
        out_c = unpermute(combined_c, pre["row_id_map"], probs=topk_weights)

        torch.testing.assert_close(out_s.float(), out_c.float(), atol=FWD_ATOL, rtol=FWD_RTOL, msg="forward combined")

        torch.manual_seed(31337)
        grad_out = torch.randn(n_local, HIDDEN, device=device, dtype=dtype)
        out_s.backward(grad_out)
        out_c.backward(grad_out)

        torch.testing.assert_close(x_s.grad.float(), x_c.grad.float(), atol=FULL_ATOL, rtol=FULL_RTOL, msg="dHidden")
        torch.testing.assert_close(w1_s.grad.float(), w1_c.grad.float(), atol=FULL_ATOL, rtol=FULL_RTOL, msg="dW1")
        torch.testing.assert_close(w2_s.grad.float(), w2_c.grad.float(), atol=FULL_ATOL, rtol=FULL_RTOL, msg="dW2")
