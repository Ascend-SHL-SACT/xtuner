import pytest
import torch


torch_npu = pytest.importorskip("torch_npu")

from xtuner.v1.ops.flash_attn.npu import npu_flash_varlen_attn  # noqa: E402


def _fake_nfa_capture(calls: list[dict]) -> object:
    """Replace torch_npu.npu_fusion_attention with a kwargs recorder returning a 7-tuple."""

    def fake_nfa(q, k, v, head_num, **kwargs):
        calls.append({"q": q, "head_num": head_num, **kwargs})
        # The 7-tuple arity is not contractual: the wrapper only consumes element [0],
        # while the real kernel returns a different arity.
        return ("out_marker", "lse_marker", "seed", "offset", "numels", "extra1", "extra2")

    return fake_nfa


class TestNpuFlashVarlenAttn:
    """Contract tests for the NPU varlen wrapper (no real NPU work: NFA is mocked)."""

    @staticmethod
    def _tnd_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.randn(4, 2, 8)
        k = torch.randn(4, 2, 8)
        v = torch.randn(4, 2, 8)
        cu = torch.tensor([0, 4], dtype=torch.int32)
        return q, k, v, cu

    def test_softmax_scale_passthrough_non_causal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[dict] = []
        monkeypatch.setattr(torch_npu, "npu_fusion_attention", _fake_nfa_capture(calls))
        q, k, v, cu = self._tnd_tensors()
        out = npu_flash_varlen_attn(q, k, v, cu, cu, torch.tensor(4), torch.tensor(4), softmax_scale=0.07)
        # Single-tensor contract: only NFA output[0] escapes, lse stays unproduced.
        assert out == "out_marker"
        assert len(calls) == 1
        assert calls[0]["scale"] == 0.07
        assert calls[0]["head_num"] == 2
        assert calls[0]["input_layout"] == "TND"
        assert calls[0]["actual_seq_qlen"] == (4,)
        assert calls[0]["atten_mask"] is None

    def test_default_scale_matches_cuda_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[dict] = []
        monkeypatch.setattr(torch_npu, "npu_fusion_attention", _fake_nfa_capture(calls))
        q, k, v, cu = self._tnd_tensors()
        npu_flash_varlen_attn(q, k, v, cu, cu, torch.tensor(4), torch.tensor(4))
        assert calls[0]["scale"] == pytest.approx(8**-0.5)

    def test_softmax_scale_passthrough_causal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[dict] = []
        monkeypatch.setattr(torch_npu, "npu_fusion_attention", _fake_nfa_capture(calls))
        q, k, v, cu = self._tnd_tensors()
        npu_flash_varlen_attn(q, k, v, cu, cu, torch.tensor(4), torch.tensor(4), softmax_scale=0.07, causal=True)
        assert calls[0]["scale"] == 0.07
        assert calls[0]["sparse_mode"] == 3
        assert calls[0]["atten_mask"].dtype == torch.bool

    def test_list_cu_seqlens_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[dict] = []
        monkeypatch.setattr(torch_npu, "npu_fusion_attention", _fake_nfa_capture(calls))
        q, k, v, _ = self._tnd_tensors()
        npu_flash_varlen_attn(q, k, v, [0, 4], [0, 4], torch.tensor(4), torch.tensor(4))
        assert calls[0]["actual_seq_qlen"] == [4]
        assert calls[0]["actual_seq_kvlen"] == [4]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"softcap": 2.0},
            {"alibi_slopes": torch.ones(2)},
            {"window_size": (128, 128)},
            {"block_table": torch.zeros(1, 1, dtype=torch.int32)},
        ],
    )
    def test_unsupported_semantics_fail_loudly(self, monkeypatch: pytest.MonkeyPatch, kwargs: dict) -> None:
        calls: list[dict] = []
        monkeypatch.setattr(torch_npu, "npu_fusion_attention", _fake_nfa_capture(calls))
        q, k, v, cu = self._tnd_tensors()
        with pytest.raises(NotImplementedError):
            npu_flash_varlen_attn(q, k, v, cu, cu, torch.tensor(4), torch.tensor(4), **kwargs)
        assert calls == []  # NFA must not be invoked at all

    def test_deterministic_true_warns_and_proceeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # No deterministic knob exists on NFA: it must not change the math, only warn.
        calls: list[dict] = []
        monkeypatch.setattr(torch_npu, "npu_fusion_attention", _fake_nfa_capture(calls))
        q, k, v, cu = self._tnd_tensors()
        with pytest.warns(RuntimeWarning, match="deterministic"):
            npu_flash_varlen_attn(q, k, v, cu, cu, torch.tensor(4), torch.tensor(4), deterministic=True)
        assert len(calls) == 1
