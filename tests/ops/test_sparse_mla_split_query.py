"""Regression tests for the sparse-MLA query consumption seam.

TestSparseMlaSplitQuery
    test_tnd_tuple_dispatch_matches_concatenated_query: pre-split tuple reaches the TND
        kernel with byte-identical arguments as the concatenated query.
    test_single_sequence_tuple_falls_back_to_concatenated_query: non-TND inputs re-route
        through the public entry on the re-concatenated tensor.
    test_split_query_public_entry_delegates_to_npu_sparse_mla: the public split entry is a
        thin wrapper over the tuple dispatch.
    test_tuple_query_requires_value_dim / test_tuple_query_rejects_mismatched_value_dim:
        the pre-split form validates ``value_dim``.
    test_dsa_mla_consumes_backend_via_sparse_mla_func_seam: dsa_mla.py must funnel BOTH
        query forms through ``self.sparse_mla_func``. Regression for run181 (2026-09-11):
        calling the module-level ``sparse_mla_split_query`` directly bypassed
        consumer-side wrappers installed on that attribute (the DSA top-k offload refill
        shim), so the backward replay consumed top-k ids whose device storage had been
        parked on CPU -> "tensor has non-zero number of elements, but its data is not
        allocated yet".
"""

from pathlib import Path

import pytest


pytest.importorskip("torch_npu")

import torch  # noqa: E402

from xtuner.v1.ops.sparse_mla import npu_sparse_mla as npu_sparse_mla_mod  # noqa: E402
from xtuner.v1.ops.sparse_mla.npu_sparse_mla import (  # noqa: E402
    SparseMLAOutputs,
    npu_sparse_mla,
    sparse_mla_split_query,
)


# dsa_mla.py is read from disk instead of imported: a source-contract check needs no
# module execution, and importing the full glm52 model chain pulls the float8 triton
# kernels, which cannot load under the deterministic autotune pin on triton 3.2.0
# (separate pre-existing issue).
_DSA_MLA_PATH = Path(npu_sparse_mla_mod.__file__).resolve().parents[2] / "model" / "moe" / "glm52" / "dsa_mla.py"

S, N, VALUE_DIM, ROPE_DIM = 8, 2, 4, 2


def _make_inputs():
    q = torch.randn(S, N, VALUE_DIM + ROPE_DIM)
    kv = torch.randn(S, 1, VALUE_DIM + ROPE_DIM)
    indices = torch.randint(0, S, (S, 1, 3), dtype=torch.int32)
    return q, kv, indices


def _record_tnd_calls(monkeypatch):
    calls = []

    def fake_tnd(q_nope, q_rope, kv_compressed, k_rope, indices, seq_ctx, seq_len, kv_len, num_heads, scale_value):
        calls.append((q_nope, q_rope, kv_compressed, k_rope, indices, seq_len, kv_len, num_heads, scale_value))
        return SparseMLAOutputs(raw_output=q_nope, softmax_lse=q_nope.sum(-1))

    monkeypatch.setattr(npu_sparse_mla_mod, "_sparse_mla_tnd_packed", fake_tnd)
    monkeypatch.setattr(npu_sparse_mla_mod, "_is_tnd_packed", lambda seq_ctx: True)
    return calls


def _record_bsnd_calls(monkeypatch):
    calls = []

    def fake_bsnd(q_nope, q_rope, kv_compressed, k_rope, indices, seq_len, kv_len, num_heads, scale_value):
        calls.append((q_nope, q_rope, kv_compressed, k_rope, indices, seq_len, kv_len, num_heads, scale_value))
        return SparseMLAOutputs(raw_output=q_nope, softmax_lse=q_nope.sum(-1))

    monkeypatch.setattr(npu_sparse_mla_mod, "_sparse_mla_bsnd_single", fake_bsnd)
    monkeypatch.setattr(npu_sparse_mla_mod, "_is_tnd_packed", lambda seq_ctx: False)
    return calls


def _assert_same_call(a, b):
    for x, y in zip(a, b):
        if isinstance(x, torch.Tensor):
            assert torch.equal(x, y)
        else:
            assert x == y


class TestSparseMlaSplitQuery:
    def test_tnd_tuple_dispatch_matches_concatenated_query(self, monkeypatch):
        calls = _record_tnd_calls(monkeypatch)
        q, kv, indices = _make_inputs()

        npu_sparse_mla(q, kv, indices, None, value_dim=VALUE_DIM, seq_ctx=None)
        assert len(calls) == 1
        tensor_call = calls[0]

        npu_sparse_mla((q[..., :VALUE_DIM], q[..., VALUE_DIM:]), kv, indices, None, value_dim=VALUE_DIM, seq_ctx=None)
        assert len(calls) == 2
        _assert_same_call(tensor_call, calls[1])

    def test_single_sequence_tuple_falls_back_to_concatenated_query(self, monkeypatch):
        calls = _record_bsnd_calls(monkeypatch)
        q, kv, indices = _make_inputs()

        npu_sparse_mla(q, kv, indices, None, value_dim=VALUE_DIM, seq_ctx=None)
        assert len(calls) == 1
        tensor_call = calls[0]

        npu_sparse_mla((q[..., :VALUE_DIM], q[..., VALUE_DIM:]), kv, indices, None, value_dim=VALUE_DIM, seq_ctx=None)
        assert len(calls) == 2
        _assert_same_call(tensor_call, calls[1])

    def test_split_query_public_entry_delegates_to_npu_sparse_mla(self, monkeypatch):
        calls = _record_tnd_calls(monkeypatch)
        q, kv, indices = _make_inputs()
        parts = (q[..., :VALUE_DIM], q[..., VALUE_DIM:])

        out = sparse_mla_split_query(parts, kv, indices, None, VALUE_DIM, None)

        assert len(calls) == 1
        assert out.raw_output is calls[0][0]

    def test_tuple_query_requires_value_dim(self):
        q, kv, indices = _make_inputs()
        parts = (q[..., :VALUE_DIM], q[..., VALUE_DIM:])

        with pytest.raises(ValueError, match="value_dim is required"):
            npu_sparse_mla(parts, kv, indices, None, seq_ctx=None)

    def test_tuple_query_rejects_mismatched_value_dim(self):
        q, kv, indices = _make_inputs()
        parts = (q[..., :VALUE_DIM], q[..., VALUE_DIM:])

        with pytest.raises(ValueError, match="must equal value_dim"):
            npu_sparse_mla(parts, kv, indices, None, value_dim=VALUE_DIM + 1, seq_ctx=None)

    def test_dsa_mla_consumes_backend_via_sparse_mla_func_seam(self):
        source = _DSA_MLA_PATH.read_text()
        assert "sparse_mla_split_query" not in source, (
            "dsa_mla must not call the module-level split entry directly: consumer-side "
            "wrappers (the DSA top-k offload refill shim) are installed on the "
            "DSAMultiLatentAttention.sparse_mla_func attribute and would be bypassed."
        )
        assert "self.sparse_mla_func(" in source
