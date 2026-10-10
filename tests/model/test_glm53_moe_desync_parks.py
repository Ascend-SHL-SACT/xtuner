"""Regression tests for the host-sync park eliminations on the MoE loss path.

The changeset replaces device->host fences on the hot micro-batch path with
stream-ordered device work or host-side metadata (xtuner/v1/model/moe/moe.py,
xtuner/v1/loss/chunk_loss.py, xtuner/v1/loss/ce_loss.py,
xtuner/v1/engine/train_engine.py). These tests pin the numerical identities the
eliminations rely on:

- ``nonpad_index_and_count`` (imported from the production module, so a drift in the
  production loop fails here instead of hiding behind a mirror copy) is bit-identical
  to ``torch.nonzero`` on the concatenated mask. If ``SequenceContext.mask`` ever
  gains interior False positions -- the tail slice in the mask property is its only
  writer today -- ``TestNonpadIndicesIdentity`` fails and the helper must go back to
  nonzero.
- ``ChunkLoss.backward`` scales with the upstream grad tensor itself via an
  unconditional stream-ordered ``mul_``: ``x*1.0`` is IEEE-exact, so the
  ``scale==1.0`` path stays bit-identical to the historical skip, and
  ``scale!=1.0`` matches the historical ``mul_(python float)``.
- ``LMHeadLossContext.loss_fn`` runs ``cross_entropy`` unconditionally (no
  ``rank_grad_tokens == 0`` host-sync branch): an all-ignored chunk must produce an
  exact 0.0 loss with an all-zero gradient, and a mixed chunk must match the
  reference cross-entropy bit-for-bit.
- ``TrainEngine._get_total_loss`` accumulates from the first loss tensor with an
  explicit fp32 accumulator (``.float()`` is a no-op on fp32 inputs): the result must
  stay bit-identical to the historical fp32-seeded sum even when the first loss field
  is a narrower dtype.
"""

from typing import cast

import torch

from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.loss.ce_loss import CELossConfig, CELossKwargs, LMHeadLossContext
from xtuner.v1.loss.chunk_loss import ChunkLoss
from xtuner.v1.model.moe.moe import nonpad_index_and_count


def _make_ctx(seq_len: int, num_padding: int) -> SequenceContext:
    """Build a minimal CPU SequenceContext with a trailing-padding mask."""
    return SequenceContext(
        input_ids=cast(torch.LongTensor, torch.arange(seq_len, dtype=torch.long).reshape(1, seq_len)),
        cu_seq_lens_q=cast(torch.IntTensor, torch.tensor([0, seq_len - num_padding], dtype=torch.int32)),
        cu_seq_lens_k=cast(torch.IntTensor, torch.tensor([0, seq_len - num_padding], dtype=torch.int32)),
        max_length_q=max(seq_len - num_padding, 1),
        max_length_k=max(seq_len - num_padding, 1),
        num_padding=num_padding,
        device="cpu",
    )


class TestNonpadIndicesIdentity:
    def test_matches_nonzero_with_mixed_padding(self):
        """Padding on the first, a middle, and the last context; one with none."""
        seq_ctx_list = [_make_ctx(16, 3), _make_ctx(8, 0), _make_ctx(12, 5), _make_ctx(10, 2)]
        got, _ = nonpad_index_and_count(seq_ctx_list)
        want = torch.nonzero(torch.cat([c.mask for c in seq_ctx_list], dim=1), as_tuple=True)[1]
        assert got.dtype == torch.int64
        assert want.dtype == torch.int64
        assert torch.equal(got, want)

    def test_non_pad_token_matches_numel(self):
        seq_ctx_list = [_make_ctx(16, 3), _make_ctx(8, 0), _make_ctx(12, 5)]
        _, non_pad_token = nonpad_index_and_count(seq_ctx_list)
        total = sum(int(c.input_ids.shape[1]) for c in seq_ctx_list)
        pads = sum(int(c.num_padding) for c in seq_ctx_list)
        assert non_pad_token == total - pads

    def test_first_context_fully_padded_edge(self):
        seq_ctx_list = [_make_ctx(4, 4), _make_ctx(6, 0)]
        got, non_pad_token = nonpad_index_and_count(seq_ctx_list)
        want = torch.nonzero(torch.cat([c.mask for c in seq_ctx_list], dim=1), as_tuple=True)[1]
        assert torch.equal(got, want)
        assert non_pad_token == 6

    def test_inputs_embeds_variant(self):
        ctx = _make_ctx(16, 3)
        ctx.input_ids = None
        ctx.inputs_embeds = torch.randn(1, 16, 4)
        got, _ = nonpad_index_and_count([ctx])
        want = torch.nonzero(ctx.mask, as_tuple=True)[1]
        assert torch.equal(got, want)


def _chunk_loss_grad(hidden: torch.Tensor, chunk_size: int, scale: float, per_chunk: float) -> torch.Tensor:
    """Run the real ChunkLoss.apply/backward and return the input grad."""

    def loss_forward(h, head_weight, head_bias, kwargs):
        return (h * per_chunk).sum(), (None, {})

    n_chunks = hidden.shape[1] // chunk_size
    h = hidden.clone().requires_grad_(True)
    out, _ = ChunkLoss.apply(
        h,
        torch.zeros(1),  # requires_grad=False -> weight path skipped
        None,
        loss_forward,
        [{} for _ in range(n_chunks)],
        chunk_size,
    )
    (out * scale).backward()
    assert h.grad is not None
    return h.grad


def _reference_grad(hidden: torch.Tensor, chunk_size: int, scale: float, per_chunk: float) -> torch.Tensor:
    """Direct autograd with the same chunk decomposition and scaling point."""
    ref = hidden.clone().requires_grad_(True)
    loss = sum((c * per_chunk).sum() for c in ref.split(chunk_size, dim=1))
    (loss * scale).backward()
    assert ref.grad is not None
    return ref.grad


class TestChunkLossBackwardScale:
    def test_grad_matches_direct_autograd_scale_one(self):
        """The scale==1.0 path must stay bit-identical to the historical skip."""
        torch.manual_seed(0)
        hidden = torch.randn(1, 12, 4, dtype=torch.float32)
        got = _chunk_loss_grad(hidden, chunk_size=4, scale=1.0, per_chunk=1.0)
        want = _reference_grad(hidden, chunk_size=4, scale=1.0, per_chunk=1.0)
        assert torch.equal(got, want)

    def test_grad_matches_direct_autograd_scale_not_one(self):
        """scale!=1.0 must match scaling the loss on the reference path."""
        torch.manual_seed(1)
        hidden = torch.randn(1, 12, 4, dtype=torch.float32)
        got = _chunk_loss_grad(hidden, chunk_size=4, scale=2.5, per_chunk=1.0)
        want = _reference_grad(hidden, chunk_size=4, scale=2.5, per_chunk=1.0)
        assert torch.equal(got, want)

    def test_weight_grad_scaled_when_requires_grad(self):
        torch.manual_seed(2)
        hidden = torch.randn(1, 8, 4, dtype=torch.float32)

        def loss_forward(h, head_weight, head_bias, kwargs):
            return (h * head_weight).sum(), (None, {})

        weight = torch.randn(4, requires_grad=True)
        h = hidden.clone().requires_grad_(True)
        out, _ = ChunkLoss.apply(h, weight, None, loss_forward, [{}, {}], 4)
        (out * 3.0).backward()
        ref_h = hidden.clone().requires_grad_(True)
        ref_w = weight.detach().clone().requires_grad_(True)  # leaf, so .grad populates
        ref_loss = (ref_h * ref_w).sum() * 3.0
        ref_loss.backward()
        # h.grad is elementwise-deterministic -> bit-identical; weight.grad sums over
        # rows, and the chunked add_ reorders that reduction, so allow reduction-order
        # ULPs there (the backward scale itself is still exact).
        assert torch.equal(h.grad, ref_h.grad)
        assert torch.allclose(weight.grad, ref_w.grad, rtol=1e-6, atol=1e-7)


def _loss_fn_call(
    hidden: torch.Tensor, labels: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the real LMHeadLossContext.loss_fn and return (loss, input grad, head)."""
    vocab, dim = 6, hidden.shape[-1]
    head = torch.randn(vocab, dim)
    ctx = LMHeadLossContext(CELossConfig(), CELossKwargs(shifted_labels=labels, loss_weight=weight))
    h = hidden.clone().requires_grad_(True)
    loss, _ = ctx.loss_fn(h, head, None, ctx.loss_kwargs)
    loss.backward()
    assert h.grad is not None
    return loss.detach(), h.grad, head


class TestCeLossAllIgnored:
    def test_all_ignored_chunk_zero_loss_zero_grad(self):
        """The deleted rank_grad_tokens==0 branch: exact 0.0 loss, exact zero grads."""
        torch.manual_seed(3)
        hidden = torch.randn(1, 8, 4, dtype=torch.float32)
        labels = torch.full((1, 8), -100)
        weight = torch.zeros(1, 8)
        loss, grad, _ = _loss_fn_call(hidden, labels, weight)
        assert loss.item() == 0.0
        assert bool((grad == 0).all())

    def test_mixed_ignored_matches_reference(self):
        """Non-degenerate chunks stay bit-identical to the reference cross-entropy."""
        torch.manual_seed(4)
        hidden = torch.randn(1, 8, 4, dtype=torch.float32)
        labels = torch.tensor([[-100, 1, 2, -100, 3, 4, -100, 5]])
        weight = torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]])
        got, got_grad, head = _loss_fn_call(hidden, labels, weight)
        ref_h = hidden.clone().requires_grad_(True)
        logits = torch.nn.functional.linear(ref_h, head).reshape(-1, 6).float()
        ref = (
            torch.nn.functional.cross_entropy(logits, labels.flatten(), reduction="none", ignore_index=-100)
            * weight.flatten()
        ).sum()
        ref.backward()
        assert torch.equal(got, ref.detach())
        assert torch.equal(got_grad, ref_h.grad)


class TestTotalLossAccumulator:
    def test_fp32_accumulator_with_narrow_first_field(self):
        """A bf16 first loss field must still sum in fp32, like the old 0.0 seed."""
        from xtuner.v1.engine.train_engine import TrainEngine

        class _FakeOutputs:
            model_fields = {"mtp_loss": None, "loss": None}  # bf16 field first on purpose

            def __init__(self, mtp, loss):
                self.mtp_loss, self.loss = mtp, loss

        bf = torch.tensor([1.5], dtype=torch.bfloat16)
        fp = torch.tensor([2.25], dtype=torch.float32)
        out = TrainEngine._get_total_loss(None, _FakeOutputs(bf, fp))  # self unused by the method
        ref = torch.tensor(0.0) + bf.float() + fp  # the historical fp32-seeded sum
        assert out.dtype == torch.float32
        assert torch.equal(out, ref)
