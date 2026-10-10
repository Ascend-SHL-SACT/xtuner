"""Tests for the sparse-MLA ``torch_npu`` backend selection and the pool-index cache lever
(R6).

- ``get_sparse_mla("torch_npu")`` returns the rope-padding adapter, which dispatches on input
  geometry: rope-tailed absorbed inputs (GLM-5.2 DSA full-form and the pre-split SPLITQ tuple)
  pass through to ``npu_sparse_mla`` untouched; NoPE full-form queries (GLM-5.3-Flash,
  ``q.shape[-1] == value_dim``) are zero-padded to the kernel's 512+64 geometry.
- ``get_kpool_topk_indices("torch_npu")`` returns the NPU KPool indexer.
- ``XTUNER_SPARSE_MLA_POOL_INDEX_CACHE`` (default on, ``0`` off) gates the ``build_pool_index``
  per-context cache.
"""

import typing

import pytest
import torch
from torch import Tensor

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.moe.glm53.nope_dsa_mla import NoPEDSAMultiLatentAttention
from xtuner.v1.ops.sparse_mla import get_kpool_topk_indices, get_sparse_mla, kpool
from xtuner.v1.ops.sparse_mla.protocol import KPoolIndexerBackend, SparseMLABackend, SparseMLAOutputs


DEVICE = "cpu"


def _make_ctx(doc_lens: list[int], device: str | torch.device = DEVICE) -> SequenceContext:
    """Build a single-rank (no SP mesh) packed ``SequenceContext``.

    Args:
        doc_lens (list[int]): token count per document, packed in order.
        device (str | torch.device): context device tag.

    Returns:
        SequenceContext: context with ``cu_seq_lens_q == cu_seq_lens_k`` and ``shard_start == 0``.
    """
    cu = [0]
    for ln in doc_lens:
        cu.append(cu[-1] + ln)
    cu_q = typing.cast(torch.IntTensor, torch.tensor(cu, dtype=torch.int32))
    return SequenceContext(
        input_ids=None,
        cu_seq_lens_q=cu_q,
        cu_seq_lens_k=cu_q,
        max_length_q=max(doc_lens),
        max_length_k=max(doc_lens),
        device=device,
    )


def _make_attn() -> NoPEDSAMultiLatentAttention:
    """Build a minimal CPU-resident NoPE DSA attention configured for the torch_npu backends.

    Returns:
        NoPEDSAMultiLatentAttention: module whose ``__init__`` resolves the backend callables.
    """
    return NoPEDSAMultiLatentAttention(
        index_topk=16,
        index_head_dim=16,
        index_n_heads=2,
        index_kpool=4,
        index_kpool_always_select_tail=True,
        indexer_types=None,
        sparse_mla_backend="torch_npu",
        indexer_backend="torch_npu",
        indexer_topk_query_chunk_size=None,
        freeze_dsa_indexer=True,
        hidden_size=32,
        num_attention_heads=2,
        q_lora_rank=16,
        qk_nope_head_dim=64,
        qk_rope_head_dim=0,
        kv_lora_rank=32,
        v_head_dim=32,
        head_dim=32,
    )


class TestSparseMlaNpuBackendSelection:
    def test_sparse_mla_torch_npu_returns_rope_adapter(self) -> None:
        """``get_sparse_mla("torch_npu")`` hands out the geometry-dispatching rope adapter."""
        from xtuner.v1.ops.sparse_mla.npu_sparse_mla import npu_sfa_rope_padded

        assert get_sparse_mla("torch_npu") is npu_sfa_rope_padded

    def test_kpool_indexer_torch_npu_returns_npu_indexer(self) -> None:
        """``get_kpool_topk_indices("torch_npu")`` hands out the NPU KPool indexer."""
        from xtuner.v1.ops.sparse_mla.kpool_npu import npu_kpool_topk_indices

        assert get_kpool_topk_indices("torch_npu") is npu_kpool_topk_indices

    def test_model_layer_resolves_npu_callables(self) -> None:
        """NoPE DSA attention built with the torch_npu backends wires both NPU callables."""
        from xtuner.v1.ops.sparse_mla.kpool_npu import npu_kpool_topk_indices
        from xtuner.v1.ops.sparse_mla.npu_sparse_mla import npu_sfa_rope_padded

        attn = _make_attn()
        assert attn.sparse_mla_backend == "torch_npu"
        assert attn.sparse_mla_func is npu_sfa_rope_padded
        assert attn.indexer.indexer_backend == "torch_npu"
        assert attn.indexer._topk_indices_fn is npu_kpool_topk_indices
        assert attn.alignment == 1

    def test_literals_contain_torch_npu(self) -> None:
        """Both backend Literals gained the ``torch_npu`` member (pydantic runtime validation)."""
        assert "torch_npu" in typing.get_args(KPoolIndexerBackend)
        assert "torch_npu" in typing.get_args(SparseMLABackend)


class TestSfaRopePaddedDispatch:
    @staticmethod
    def _capture(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
        """Replace ``npu_sparse_mla`` inside the adapter's module with a capturing stub."""
        from xtuner.v1.ops.sparse_mla import npu_sparse_mla

        calls: list[dict] = []

        def _stub(q, kv, indices, scaling, value_dim=None, *, seq_ctx=None):
            calls.append(
                {
                    "q": q,
                    "kv": kv,
                    "indices": indices,
                    "scaling": scaling,
                    "value_dim": value_dim,
                    "seq_ctx": seq_ctx,
                }
            )
            return SparseMLAOutputs(raw_output=torch.zeros(1), softmax_lse=torch.zeros(1))

        monkeypatch.setattr(npu_sparse_mla, "npu_sparse_mla", _stub)
        return calls

    def test_nope_full_form_is_padded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """NoPE absorbed query (width == value_dim) is zero-padded to ``Rkv + 64``."""
        from xtuner.v1.ops.sparse_mla.npu_sparse_mla import npu_sfa_rope_padded

        calls = self._capture(monkeypatch)
        q = torch.arange(4 * 2 * 512, dtype=torch.float32).reshape(4, 2, 512)
        kv = torch.arange(4 * 1 * 512, dtype=torch.float32).reshape(4, 1, 512)
        indices = torch.zeros(4, 1, 8, dtype=torch.int32)
        out = npu_sfa_rope_padded(q, kv, indices, 0.1, value_dim=512)
        assert len(calls) == 1
        call = calls[0]
        assert call["q"].shape == (4, 2, 576)
        assert torch.equal(call["q"][..., :512], q)
        assert torch.equal(call["q"][..., 512:], torch.zeros(4, 2, 64))
        assert call["kv"].shape == (4, 1, 576)
        assert torch.equal(call["kv"][..., :512], kv)
        assert torch.equal(call["kv"][..., 512:], torch.zeros(4, 1, 64))
        assert call["indices"] is indices
        assert call["value_dim"] == 512
        assert out.raw_output.numel() == 1

    def test_rope_tailed_full_form_passes_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GLM-5.2-style 576-wide query (width != value_dim) reaches the kernel unpadded."""
        from xtuner.v1.ops.sparse_mla.npu_sparse_mla import npu_sfa_rope_padded

        calls = self._capture(monkeypatch)
        q = torch.arange(4 * 2 * 576, dtype=torch.float32).reshape(4, 2, 576)
        kv = torch.arange(4 * 1 * 576, dtype=torch.float32).reshape(4, 1, 576)
        indices = torch.zeros(4, 1, 8, dtype=torch.int32)
        out = npu_sfa_rope_padded(q, kv, indices, 0.1, value_dim=512)
        assert len(calls) == 1
        assert calls[0]["q"] is q
        assert calls[0]["kv"] is kv
        assert out.raw_output.numel() == 1

    def test_split_query_tuple_passes_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The pre-split ``(q_nope, q_rope)`` tuple reaches the kernel untouched."""
        from xtuner.v1.ops.sparse_mla.npu_sparse_mla import npu_sfa_rope_padded

        calls = self._capture(monkeypatch)
        q_nope = torch.zeros(4, 2, 512)
        q_rope = torch.zeros(4, 2, 64)
        kv = torch.zeros(4, 1, 576)
        indices = torch.zeros(4, 1, 8, dtype=torch.int32)
        out = npu_sfa_rope_padded((q_nope, q_rope), kv, indices, 0.1, value_dim=512)
        assert len(calls) == 1
        assert calls[0]["q"] == (q_nope, q_rope)
        assert out.raw_output.numel() == 1

    def test_value_dim_none_passes_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without ``value_dim`` the NoPE discriminator is unavailable -- pass through."""
        from xtuner.v1.ops.sparse_mla.npu_sparse_mla import npu_sfa_rope_padded

        calls = self._capture(monkeypatch)
        q = torch.zeros(4, 2, 512)
        kv = torch.zeros(4, 1, 512)
        out = npu_sfa_rope_padded(q, kv, torch.zeros(4, 1, 8, dtype=torch.int32), None, value_dim=None)
        assert len(calls) == 1
        assert calls[0]["q"] is q
        assert out.raw_output.numel() == 1


class TestPoolIndexCacheLever:
    def test_lever_off_recomputes_each_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cache off: repeated calls rebuild (fresh object each time) with identical values."""
        monkeypatch.setenv("XTUNER_SPARSE_MLA_POOL_INDEX_CACHE", "0")
        doc_lens = [7, 3, 16]
        seq_len = sum(doc_lens)
        ctx = _make_ctx(doc_lens)
        first = kpool.build_pool_index(ctx, seq_len, 4, DEVICE)
        second = kpool.build_pool_index(ctx, seq_len, 4, DEVICE)
        assert first is not second
        assert torch.equal(first, second)

    def test_lever_off_populates_no_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cache off: nothing is recorded under the context's weak key."""
        monkeypatch.setenv("XTUNER_SPARSE_MLA_POOL_INDEX_CACHE", "0")
        ctx = _make_ctx([5, 9])
        kpool.build_pool_index(ctx, 14, 4, DEVICE)
        assert ctx not in kpool._POOL_INDEX_CACHE

    def test_lever_on_default_reuses_object(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cache on (default): repeated calls return the same shared read-only tensor."""
        monkeypatch.delenv("XTUNER_SPARSE_MLA_POOL_INDEX_CACHE", raising=False)
        doc_lens = [7, 3, 16]
        seq_len = sum(doc_lens)
        ctx = _make_ctx(doc_lens)
        first: Tensor = kpool.build_pool_index(ctx, seq_len, 4, DEVICE)
        second: Tensor = kpool.build_pool_index(ctx, seq_len, 4, DEVICE)
        assert second is first

    def test_lever_off_result_matches_cached_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The lever toggles caching only: off-leg values equal the cached leg's."""
        doc_lens = [7, 3, 16]
        seq_len = sum(doc_lens)
        ctx = _make_ctx(doc_lens)
        monkeypatch.setenv("XTUNER_SPARSE_MLA_POOL_INDEX_CACHE", "0")
        off = kpool.build_pool_index(ctx, seq_len, 4, DEVICE)
        monkeypatch.delenv("XTUNER_SPARSE_MLA_POOL_INDEX_CACHE", raising=False)
        on = kpool.build_pool_index(ctx, seq_len, 4, DEVICE)
        assert torch.equal(off, on)
