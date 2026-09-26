# Copyright © 2026 Huawei Technologies Co., Ltd.
"""Tests for ``xtuner.v1.ops.kda.npu_backend``: the fla 0.4.2 KDA surface re-exposed on fla_npu.

TestNpuImplSelected
    XTUNER_KDA_BACKEND matrix (npu/fla/junk/case+space) and the unset fallback to the device
    helper.

TestGateStash
    The one-slot ``fused_kda_gate`` -> ``chunk_kda`` / ``fused_recurrent_kda`` handoff: identity
    return, missing/mismatched record, mandatory ``dt_bias``, single-shot consumption.

TestChunkKdaParity / TestChunkKdaVarlen / TestFusedRecurrentKda
    NPU forward+backward parity against the CPU eager route (identical inputs, bf16 tolerance),
    plus CPU-side varlen boundary isolation.

TestCausalConv1d
    NPU AscendC dispatch and dense fallback correctness, varlen boundary isolation, bias
    fallback, unsupported kwargs.

TestRmsNormGated
    CPU fp32 math, NPU fp32/bf16 regimes, the ``XTUNER_NPU_FUSED_GATED_NORM`` env switch, and
    the unsupported-option rejections.

TestBackendModules / TestGetterDispatch
    The base modules' init state and the getters' NPU dispatch.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from xtuner.v1.ops.kda import get_causal_conv1d_fn, get_chunk_kda_fn, get_fused_kda_gate_fn, npu_backend
from xtuner.v1.ops.kda.npu_backend import (
    FusedRMSNormGated,
    ShortConvolution,
    causal_conv1d,
    chunk_kda,
    fused_kda_gate,
    fused_recurrent_kda,
    npu_impl_selected,
    rms_norm_gated,
)


try:
    import torch_npu  # noqa: F401

    _NPU_AVAILABLE = torch.npu.is_available()
except ImportError:
    _NPU_AVAILABLE = False

requires_npu = pytest.mark.skipif(not _NPU_AVAILABLE, reason="requires an Ascend NPU")

# bf16 kernel-vs-eager parity budget (observed max abs error is <= 1e-3 on every tensor).
_PARITY_RTOL = 2e-2
_PARITY_ATOL = 4e-3


def _assert_parity(actual: torch.Tensor, expected: torch.Tensor, label: str, atol: float = _PARITY_ATOL) -> None:
    """Assert bf16-tolerance agreement between two tensors, on host, with a labelled failure."""

    def _msg(message: str) -> str:
        return f"{label}: {message}"

    torch.testing.assert_close(
        actual.detach().float().cpu(), expected.detach().float().cpu(), rtol=_PARITY_RTOL, atol=atol, msg=_msg
    )


def _make_kda_inputs(device: str, seq_len: int, heads: int, seed: int) -> dict[str, torch.Tensor]:
    """Build identical KDA inputs on any device (drawn on CPU: CPU and NPU RNGs differ).

    Args:
        device (str): Target device (``"cpu"`` or ``"npu"``).
        seq_len (int): Sequence length ``T``.
        heads (int): Number of heads ``H`` (must equal the value-head count).
        seed (int): Seed for the CPU generator.

    Returns:
        dict[str, torch.Tensor]: Leaf tensors requiring grad (``grad`` is the plain upstream
        gradient), plus ``beta`` already post-sigmoid in fp32, mirroring the module's
        ``b_proj(...).float().sigmoid()``.
    """
    torch.manual_seed(seed)
    batch, dim = 1, 128
    cpu = {
        "q": torch.randn(batch, seq_len, heads, dim, dtype=torch.bfloat16),
        "k": torch.randn(batch, seq_len, heads, dim, dtype=torch.bfloat16),
        "v": torch.randn(batch, seq_len, heads, dim, dtype=torch.bfloat16),
        "g_raw": torch.randn(batch, seq_len, heads, dim, dtype=torch.bfloat16),
        "beta": torch.sigmoid(torch.randn(batch, seq_len, heads, dtype=torch.float32)),
        "A_log": torch.randn(heads, dtype=torch.float32),
        "dt_bias": torch.randn(heads * dim, dtype=torch.float32),
        "grad": torch.randn(batch, seq_len, heads, dim, dtype=torch.bfloat16),
    }
    moved: dict[str, torch.Tensor] = {}
    for name, tensor in cpu.items():
        tensor = tensor.to(device)
        moved[name] = tensor if name == "grad" else tensor.requires_grad_()
    return moved


def _run_kda(
    fn: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
    device: str,
    seq_len: int = 96,
    heads: int = 2,
    seed: int = 0,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run one KDA entry point with the module's call shape and return the output plus grads.

    Args:
        fn (Callable[..., tuple[torch.Tensor, torch.Tensor | None]]): ``chunk_kda`` or
            ``fused_recurrent_kda``.
        device (str): ``"cpu"`` or ``"npu"``.
        seq_len (int): Sequence length.
        heads (int): Number of heads.
        seed (int): Input seed.
        cu_seqlens (torch.Tensor | None): Packed-sequence offsets, or ``None`` for dense.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: The detached output and the per-input
        gradients ``q/k/v/g_raw/beta/A_log/dt_bias``.
    """
    ins = _make_kda_inputs(device, seq_len, heads, seed)
    gate = fused_kda_gate(ins["g_raw"], ins["A_log"], dt_bias=ins["dt_bias"], lower_bound=-5.0)
    o, final_state = fn(
        q=ins["q"],
        k=ins["k"],
        v=ins["v"],
        g=gate,
        beta=ins["beta"],
        scale=128**-0.5,
        cu_seqlens=cu_seqlens,
        safe_gate=True,
        use_qk_l2norm_in_kernel=True,
        transpose_state_layout=True,
    )
    assert final_state is None
    o.backward(ins["grad"])
    grads = {name: ins[name].grad for name in ("q", "k", "v", "g_raw", "beta", "A_log", "dt_bias")}
    missing = [name for name, tensor in grads.items() if tensor is None]
    assert not missing, f"no gradient reached {missing}"
    return o.detach(), dict(grads)


def _tiny_kda_tensors() -> dict[str, torch.Tensor]:
    """Minimal KDA tensors that fail the kernel's K=V=128 contract only after the gate dispatch.

    Returns:
        dict[str, torch.Tensor]: Small CPU tensors, including two distinct gate tensors ``g1``
        and ``g2`` for the mismatched-identity case.
    """
    batch, seq_len, heads, dim = 1, 4, 1, 4
    return {
        "q": torch.randn(batch, seq_len, heads, dim, dtype=torch.bfloat16),
        "k": torch.randn(batch, seq_len, heads, dim, dtype=torch.bfloat16),
        "v": torch.randn(batch, seq_len, heads, dim, dtype=torch.bfloat16),
        "g1": torch.randn(batch, seq_len, heads, dim, dtype=torch.bfloat16),
        "g2": torch.randn(batch, seq_len, heads, dim, dtype=torch.bfloat16),
        "beta": torch.rand(batch, seq_len, heads, dtype=torch.float32),
        "A_log": torch.randn(heads, dtype=torch.float32),
        "dt_bias": torch.randn(heads * dim, dtype=torch.float32),
    }


def _constant_segment_kda(device: str, seg0: float, seg1: float) -> dict[str, torch.Tensor]:
    """KDA inputs whose values are constant per segment (boundary at token 50 of 96).

    Args:
        device (str): ``"cpu"`` or ``"npu"``.
        seg0 (float): Constant value on segment ``[0, 50)``.
        seg1 (float): Constant value on segment ``[50, 96)``.

    Returns:
        dict[str, torch.Tensor]: Leaf tensors requiring grad, keyed as in ``_make_kda_inputs``.
    """
    seq_len, heads, dim = 96, 2, 128

    def _fill(v0: float, v1: float, rows: int, cols: int) -> torch.Tensor:
        tensor = torch.empty(1, seq_len, rows, cols, dtype=torch.bfloat16)
        tensor[:, :50] = v0
        tensor[:, 50:] = v1
        return tensor.to(device)

    def _fill_beta(v0: float, v1: float) -> torch.Tensor:
        tensor = torch.empty(1, seq_len, heads, dtype=torch.bfloat16)
        tensor[:, :50] = v0
        tensor[:, 50:] = v1
        return tensor.float().to(device)

    return {
        "q": _fill(seg0, seg1, heads, dim).requires_grad_(),
        "k": _fill(seg0, seg1, heads, dim).requires_grad_(),
        "v": _fill(seg0, seg1, heads, dim).requires_grad_(),
        "g_raw": _fill(seg0, seg1, heads, dim).requires_grad_(),
        "beta": _fill_beta(0.3, 0.7).requires_grad_(),
        "A_log": torch.full((heads,), -1.0, dtype=torch.float32, device=device).requires_grad_(),
        "dt_bias": torch.full((heads * dim,), 0.5, dtype=torch.float32, device=device).requires_grad_(),
    }


def _causal_conv_reference(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, activation: str | None
) -> torch.Tensor:
    """Independent fp32 causal depthwise conv reference: an explicit sliding-window sum.

    Args:
        x (torch.Tensor): Input ``[B, T, D]`` (any dtype; promoted to fp32).
        weight (torch.Tensor): Depthwise kernel ``[D, W]``.
        bias (torch.Tensor | None): Optional per-channel bias ``[D]``.
        activation (str | None): ``"silu"``/``"swish"`` or ``None``.

    Returns:
        torch.Tensor: fp32 reference output ``[B, T, D]`` (computed on host).
    """
    xf = x.detach().float().cpu()
    wf = weight.detach().float().cpu()
    biasf = None if bias is None else bias.detach().float().cpu()
    width = wf.shape[-1]
    batch, seq_len, channels = xf.shape
    padded = F.pad(xf.transpose(1, 2), (width - 1, 0))
    y = torch.zeros(batch, channels, seq_len, dtype=torch.float32)
    for j in range(width):
        y = y + wf[:, j].view(1, channels, 1) * padded[:, :, j : j + seq_len]
    if biasf is not None:
        y = y + biasf.view(1, channels, 1)
    y = y.transpose(1, 2)
    if activation in ("silu", "swish"):
        y = F.silu(y)
    return y


def _dense_conv_formula(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, activation: str | None
) -> torch.Tensor:
    """The dense depthwise path the backend falls back to, spelled out (native dtypes).

    Args:
        x (torch.Tensor): Input ``[B, T, D]``.
        weight (torch.Tensor): Depthwise kernel ``[D, W]``.
        bias (torch.Tensor | None): Optional per-channel bias ``[D]``.
        activation (str | None): ``"silu"``/``"swish"`` or ``None``.

    Returns:
        torch.Tensor: Output ``[B, T, D]`` in the input dtype.
    """
    y = F.conv1d(
        x.transpose(1, 2),
        weight.unsqueeze(1),
        bias=bias,
        stride=1,
        padding=weight.shape[-1] - 1,
        groups=x.shape[-1],
    )[:, :, : x.shape[1]].transpose(1, 2)
    if activation in ("silu", "swish"):
        y = F.silu(y)
    return y


def _rms_reference(x: torch.Tensor, g: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Strict-fp32 gated RMSNorm reference, op-for-op the eager branch of ``rms_norm_gated``.

    Args:
        x (torch.Tensor): Input ``[..., D]``.
        g (torch.Tensor): Gate, broadcastable against ``x``.
        weight (torch.Tensor): Norm scale ``[D]``.
        eps (float): Variance epsilon.

    Returns:
        torch.Tensor: Gated normed output, same dtype as ``x``.
    """
    xf = x.float()
    normed = weight.float() * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps))
    return (normed * torch.sigmoid(g.float())).to(x.dtype)


def _bf16_ulp_distance(actual: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Per-element ULP distance between two bf16 tensors, via their int16 bit patterns.

    Args:
        actual (torch.Tensor): Candidate values (bf16).
        ref (torch.Tensor): Reference values (bf16).

    Returns:
        torch.Tensor: Non-negative per-element ULP distance (sign-magnitude bit patterns mapped
        onto a monotonic scale, so binade boundaries are handled exactly).
    """

    def _to_monotonic(bits: torch.Tensor) -> torch.Tensor:
        bits = bits.to(torch.int32)
        return torch.where(bits < 0, -32768 - bits, bits)

    a = _to_monotonic(actual.contiguous().cpu().view(torch.int16))
    e = _to_monotonic(ref.contiguous().cpu().view(torch.int16))
    return (a - e).abs().float()


def _bf16_regime_reference(x: torch.Tensor, g: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The fused bf16 regime spelled out with the public ``torch_npu.npu_rms_norm`` op."""
    from torch_npu import npu_rms_norm

    y = npu_rms_norm(x, weight.to(torch.float32), 1e-5)[0]
    return y * torch.sigmoid(g)


class TestNpuImplSelected:
    """XTUNER_KDA_BACKEND matrix and the unset fallback to the device helper."""

    def test_env_npu_selects_npu_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XTUNER_KDA_BACKEND", "npu")
        assert npu_impl_selected() is True

    def test_env_fla_selects_fla_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XTUNER_KDA_BACKEND", "fla")
        assert npu_impl_selected() is False

    def test_env_junk_selects_fla_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XTUNER_KDA_BACKEND", "cuda")
        assert npu_impl_selected() is False

    def test_env_value_is_case_and_space_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XTUNER_KDA_BACKEND", "  NPU ")
        assert npu_impl_selected() is True
        monkeypatch.setenv("XTUNER_KDA_BACKEND", "Fla")
        assert npu_impl_selected() is False

    def test_unset_falls_back_to_device_helper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("XTUNER_KDA_BACKEND", raising=False)
        monkeypatch.setattr("xtuner.v1.utils.get_device", lambda: "npu")
        assert npu_impl_selected() is True
        monkeypatch.setattr("xtuner.v1.utils.get_device", lambda: "cpu")
        assert npu_impl_selected() is False

    def test_unset_returns_bool_without_device_mocking(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # No dependence on what the accelerator actually is: the contract is only "a bool".
        monkeypatch.delenv("XTUNER_KDA_BACKEND", raising=False)
        assert isinstance(npu_impl_selected(), bool)


class TestGateStash:
    """The one-slot fused_kda_gate -> chunk_kda / fused_recurrent_kda handoff."""

    @pytest.fixture(autouse=True)
    def _reset_gate_stash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(npu_backend, "_GATE_BUNDLE", None)

    def test_gate_returns_g_unchanged(self) -> None:
        g_raw = torch.randn(2, 8, 2, 4, dtype=torch.bfloat16, requires_grad=True)
        a_log = torch.randn(2, dtype=torch.float32)
        dt_bias = torch.randn(8, dtype=torch.float32)
        gate = fused_kda_gate(g_raw, a_log, dt_bias=dt_bias, lower_bound=-5.0)
        assert gate is g_raw
        assert gate.requires_grad

    def test_gate_rejects_non_fp32_output_dtype(self) -> None:
        g_raw = torch.randn(2, 8, 2, 4, dtype=torch.bfloat16)
        with pytest.raises(NotImplementedError):
            fused_kda_gate(g_raw, torch.randn(2), dt_bias=torch.randn(8), output_dtype=torch.float16)

    def test_chunk_kda_without_prior_gate_raises(self) -> None:
        t = _tiny_kda_tensors()
        with pytest.raises(RuntimeError, match="fused_kda_gate"):
            chunk_kda(q=t["q"], k=t["k"], v=t["v"], g=t["g1"], beta=t["beta"])

    def test_chunk_kda_rejects_mismatched_g_tensor(self) -> None:
        t = _tiny_kda_tensors()
        fused_kda_gate(t["g1"], t["A_log"], dt_bias=t["dt_bias"], lower_bound=-5.0)
        with pytest.raises(RuntimeError, match="fused_kda_gate"):
            chunk_kda(q=t["q"], k=t["k"], v=t["v"], g=t["g2"], beta=t["beta"])

    def test_chunk_kda_requires_dt_bias(self) -> None:
        t = _tiny_kda_tensors()
        gate = fused_kda_gate(t["g1"], t["A_log"], dt_bias=None, lower_bound=-5.0)
        with pytest.raises(RuntimeError, match="dt_bias"):
            chunk_kda(q=t["q"], k=t["k"], v=t["v"], g=gate, beta=t["beta"])

    def test_stash_is_consumed_by_one_call(self) -> None:
        t = _tiny_kda_tensors()
        gate = fused_kda_gate(t["g1"], t["A_log"], dt_bias=t["dt_bias"], lower_bound=-5.0)
        # The tiny geometry runs the eager fallback (no K=V=128 contract there) and consumes the
        # record -- so a second call on the same gate must find the stash empty.
        o, final_state = chunk_kda(q=t["q"], k=t["k"], v=t["v"], g=gate, beta=t["beta"])
        assert final_state is None
        assert o.shape == t["q"].shape and o.dtype == torch.bfloat16
        with pytest.raises(RuntimeError, match="fused_kda_gate"):
            chunk_kda(q=t["q"], k=t["k"], v=t["v"], g=gate, beta=t["beta"])

    def test_fused_recurrent_kda_shares_stash_contract(self) -> None:
        t = _tiny_kda_tensors()
        with pytest.raises(RuntimeError, match="fused_kda_gate"):
            fused_recurrent_kda(q=t["q"], k=t["k"], v=t["v"], g=t["g1"], beta=t["beta"])


class TestChunkKdaParity:
    """Dense chunk_kda: NPU AscendC forward+backward vs the CPU eager route."""

    @pytest.mark.gpu
    @requires_npu
    def test_dense_forward_backward_matches_cpu_reference(self) -> None:
        o_npu, grads_npu = _run_kda(chunk_kda, "npu", seq_len=96, heads=2, seed=0)
        o_cpu, grads_cpu = _run_kda(chunk_kda, "cpu", seq_len=96, heads=2, seed=0)
        assert o_npu.shape == (1, 96, 2, 128)
        assert o_npu.dtype == torch.bfloat16
        _assert_parity(o_npu, o_cpu, "o")
        for name in grads_npu:
            _assert_parity(grads_npu[name], grads_cpu[name], f"d{name}")


class TestChunkKdaVarlen:
    """Packed (cu_seqlens) chunk_kda: NPU parity, and boundary isolation on CPU."""

    @pytest.mark.gpu
    @requires_npu
    def test_packed_forward_backward_matches_cpu_reference(self) -> None:
        cu = torch.tensor([0, 50, 96])
        o_npu, grads_npu = _run_kda(chunk_kda, "npu", seq_len=96, heads=2, seed=1, cu_seqlens=cu)
        o_cpu, grads_cpu = _run_kda(chunk_kda, "cpu", seq_len=96, heads=2, seed=1, cu_seqlens=cu)
        _assert_parity(o_npu, o_cpu, "o")
        for name in grads_npu:
            _assert_parity(grads_npu[name], grads_cpu[name], f"d{name}")

    def test_segments_never_cross_boundaries(self) -> None:
        cu = torch.tensor([0, 50, 96])
        ins = _constant_segment_kda("cpu", seg0=0.5, seg1=1.5)
        gate = fused_kda_gate(ins["g_raw"], ins["A_log"], dt_bias=ins["dt_bias"], lower_bound=-5.0)
        o, _ = chunk_kda(
            q=ins["q"],
            k=ins["k"],
            v=ins["v"],
            g=gate,
            beta=ins["beta"],
            scale=128**-0.5,
            cu_seqlens=cu,
            safe_gate=True,
            use_qk_l2norm_in_kernel=True,
        )
        # Changing ONLY the first segment's payload must leave the second segment bitwise intact.
        ins2 = _constant_segment_kda("cpu", seg0=0.25, seg1=1.5)
        gate2 = fused_kda_gate(ins2["g_raw"], ins2["A_log"], dt_bias=ins2["dt_bias"], lower_bound=-5.0)
        o2, _ = chunk_kda(
            q=ins2["q"],
            k=ins2["k"],
            v=ins2["v"],
            g=gate2,
            beta=ins2["beta"],
            scale=128**-0.5,
            cu_seqlens=cu,
            safe_gate=True,
            use_qk_l2norm_in_kernel=True,
        )
        assert torch.equal(o2[:, 50:], o[:, 50:])
        assert not torch.equal(o2[:, :50], o[:, :50])
        # And the packed output must equal the per-segment dense forwards, concatenated.
        parts = []
        for start, end in ((0, 50), (50, 96)):
            seg = _constant_segment_kda("cpu", seg0=0.5, seg1=1.5)
            # Slice BEFORE recording: the stash validates gate identity, so the record must be
            # made on the exact (sliced) tensor that chunk_kda receives.
            gate_seg = fused_kda_gate(
                seg["g_raw"][:, start:end], seg["A_log"], dt_bias=seg["dt_bias"], lower_bound=-5.0
            )
            o_seg, _ = chunk_kda(
                q=seg["q"][:, start:end],
                k=seg["k"][:, start:end],
                v=seg["v"][:, start:end],
                g=gate_seg,
                beta=seg["beta"][:, start:end],
                scale=128**-0.5,
                safe_gate=True,
                use_qk_l2norm_in_kernel=True,
            )
            parts.append(o_seg)
        assert torch.equal(o, torch.cat(parts, dim=1))

    def test_small_geometry_runs_eager_fallback(self) -> None:
        # Head dim != 128 runs the eager torch path (the Ascend C kernels' K=V=128 contract does
        # not apply there), so small parity geometries work end to end.
        t = _tiny_kda_tensors()
        gate = fused_kda_gate(t["g1"], t["A_log"], dt_bias=t["dt_bias"], lower_bound=-5.0)
        o, final_state = chunk_kda(q=t["q"], k=t["k"], v=t["v"], g=gate, beta=t["beta"], scale=0.5)
        assert final_state is None
        assert o.shape == t["v"].shape


class TestFusedRecurrentKda:
    """Short-sequence entry: same core as chunk_kda, parity against the CPU eager route."""

    @pytest.mark.gpu
    @requires_npu
    def test_short_seq_forward_backward_matches_cpu_reference(self) -> None:
        o_npu, grads_npu = _run_kda(fused_recurrent_kda, "npu", seq_len=48, heads=2, seed=2)
        o_cpu, grads_cpu = _run_kda(fused_recurrent_kda, "cpu", seq_len=48, heads=2, seed=2)
        _assert_parity(o_npu, o_cpu, "o")
        for name in grads_npu:
            _assert_parity(grads_npu[name], grads_cpu[name], f"d{name}")

    @pytest.mark.gpu
    @requires_npu
    def test_short_seq_routes_to_same_core_as_chunk_kda(self) -> None:
        ins = _make_kda_inputs("npu", seq_len=48, heads=2, seed=3)
        gate_rec = fused_kda_gate(ins["g_raw"], ins["A_log"], dt_bias=ins["dt_bias"], lower_bound=-5.0)
        o_rec, _ = fused_recurrent_kda(
            q=ins["q"],
            k=ins["k"],
            v=ins["v"],
            g=gate_rec,
            beta=ins["beta"],
            scale=128**-0.5,
            safe_gate=True,
            use_qk_l2norm_in_kernel=True,
        )
        gate_chunk = fused_kda_gate(ins["g_raw"], ins["A_log"], dt_bias=ins["dt_bias"], lower_bound=-5.0)
        o_chunk, _ = chunk_kda(
            q=ins["q"],
            k=ins["k"],
            v=ins["v"],
            g=gate_chunk,
            beta=ins["beta"],
            scale=128**-0.5,
            safe_gate=True,
            use_qk_l2norm_in_kernel=True,
        )
        assert torch.equal(o_rec, o_chunk)

    def test_rejects_unsupported_call_shapes(self) -> None:
        t = _tiny_kda_tensors()
        gate = fused_kda_gate(t["g1"], t["A_log"], dt_bias=t["dt_bias"], lower_bound=-5.0)
        with pytest.raises(NotImplementedError, match="L2-norm"):
            fused_recurrent_kda(q=t["q"], k=t["k"], v=t["v"], g=gate, beta=t["beta"], use_qk_l2norm_in_kernel=False)
        with pytest.raises(NotImplementedError, match="initial_state"):
            # The gate record is untouched: the kwarg check runs before the stash is consumed.
            fused_recurrent_kda(
                q=t["q"],
                k=t["k"],
                v=t["v"],
                g=gate,
                beta=t["beta"],
                use_qk_l2norm_in_kernel=True,
                initial_state=torch.ones(1),
            )


class TestCausalConv1d:
    """Causal depthwise short convolution: AscendC dispatch, dense fallback, varlen isolation."""

    @pytest.mark.gpu
    @requires_npu
    def test_npu_b1_routes_to_ascendc_and_matches_dense_reference(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import xtuner.v1.ops.kda.causal_conv1d_ascendc as ascendc_mod

        calls: list[tuple[int, ...]] = []
        real = ascendc_mod.causal_conv1d_ascendc

        def _spy(
            x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, activation: str | None, cu: Any
        ) -> Any:
            calls.append(tuple(x.shape))
            return real(x, weight, bias, activation, cu)

        monkeypatch.setattr(ascendc_mod, "causal_conv1d_ascendc", _spy)
        torch.manual_seed(1)
        channels, seq_len, width = 64, 100, 4
        x = torch.randn(1, seq_len, channels, dtype=torch.bfloat16, device="npu")
        weight = (torch.randn(channels, width, dtype=torch.bfloat16, device="npu") * 0.1).requires_grad_()
        y, final_state = causal_conv1d(x, weight, None, "silu")
        assert final_state is None
        assert calls == [(1, seq_len, channels)], "B=1 width-4 bias-free input must hit the AscendC op"
        # Bitwise agreement with the dense math (same dtypes), plus an independent fp32 check.
        assert torch.equal(y, _dense_conv_formula(x, weight, None, "silu"))
        _assert_parity(y, _causal_conv_reference(x, weight, None, "silu"), "conv", atol=8e-3)

    def test_cpu_dense_path_matches_manual_reference(self) -> None:
        torch.manual_seed(2)
        channels, seq_len, width = 32, 40, 4
        x = torch.randn(2, seq_len, channels, dtype=torch.bfloat16)
        weight = torch.randn(channels, width, dtype=torch.bfloat16) * 0.1
        y, final_state = causal_conv1d(x, weight, None, "silu")
        assert final_state is None
        assert y.shape == x.shape and y.dtype == torch.bfloat16
        # B>1 goes down the dense path; against both the F.conv1d spelling and the fp32 window sum.
        assert torch.equal(y, _dense_conv_formula(x, weight, None, "silu"))
        _assert_parity(y, _causal_conv_reference(x, weight, None, "silu"), "conv", atol=8e-3)
        # activation=None must skip the silu.
        y_raw, _ = causal_conv1d(x, weight, None, None)
        assert torch.equal(y_raw, _dense_conv_formula(x, weight, None, None))

    def test_cpu_varlen_segments_never_cross_boundaries(self) -> None:
        torch.manual_seed(3)
        channels, seq_len, width = 32, 100, 4
        x = torch.randn(1, seq_len, channels, dtype=torch.bfloat16)
        weight = torch.randn(channels, width, dtype=torch.bfloat16) * 0.1
        cu = torch.tensor([0, 40, 100])
        y, _ = causal_conv1d(x, weight, None, "silu", cu_seqlens=cu)
        per_seg = torch.cat(
            [_dense_conv_formula(x[:, start:end], weight, None, "silu") for start, end in ((0, 40), (40, 100))], dim=1
        )
        assert torch.equal(y, per_seg)

    @pytest.mark.gpu
    @requires_npu
    def test_npu_varlen_segments_never_cross_boundaries(self) -> None:
        torch.manual_seed(4)
        channels, seq_len, width = 64, 100, 4
        x = torch.randn(1, seq_len, channels, dtype=torch.bfloat16, device="npu")
        weight = (torch.randn(channels, width, dtype=torch.bfloat16, device="npu") * 0.1).requires_grad_()
        cu = torch.tensor([0, 40, 100])
        y, _ = causal_conv1d(x, weight, None, "silu", cu_seqlens=cu)
        per_seg = torch.cat(
            [_dense_conv_formula(x[:, start:end], weight, None, "silu") for start, end in ((0, 40), (40, 100))], dim=1
        )
        _assert_parity(y, per_seg, "varlen conv", atol=8e-3)
        # Changing only the first segment's payload cannot leak into the second segment.
        x2 = x.clone()
        x2[:, :40] = x2[:, :40] + 1.0
        y2, _ = causal_conv1d(x2, weight, None, "silu", cu_seqlens=cu)
        assert torch.equal(y2[:, 40:], y[:, 40:])
        assert not torch.equal(y2[:, :40], y[:, :40])

    @pytest.mark.gpu
    @requires_npu
    def test_npu_bias_falls_back_to_dense_and_is_correct(self) -> None:
        torch.manual_seed(5)
        channels, seq_len, width = 64, 100, 4
        x = torch.randn(1, seq_len, channels, dtype=torch.bfloat16, device="npu")
        weight = torch.randn(channels, width, dtype=torch.bfloat16, device="npu") * 0.1
        bias = torch.randn(channels, dtype=torch.bfloat16, device="npu")
        y, _ = causal_conv1d(x, weight, bias, "silu")
        assert torch.equal(y, _dense_conv_formula(x, weight, bias, "silu"))
        _assert_parity(y, _causal_conv_reference(x, weight, bias, "silu"), "conv+bias", atol=2e-2)
        y_no_bias, _ = causal_conv1d(x, weight, None, "silu")
        assert not torch.equal(y, y_no_bias), "the bias must actually be applied"

    def test_unsupported_kwargs_raise(self) -> None:
        x = torch.randn(1, 8, 4, dtype=torch.bfloat16)
        weight = torch.randn(4, 4, dtype=torch.bfloat16)
        for kwarg in ("residual", "initial_state", "output_final_state", "cp_context"):
            with pytest.raises(NotImplementedError, match=kwarg):
                causal_conv1d(x, weight, None, None, **{kwarg: torch.ones(1)})

    def test_three_dim_weight_is_squeezed(self) -> None:
        torch.manual_seed(6)
        channels, seq_len, width = 8, 12, 4
        x = torch.randn(1, seq_len, channels, dtype=torch.bfloat16)
        weight = torch.randn(channels, width, dtype=torch.bfloat16) * 0.1
        y_flat, _ = causal_conv1d(x, weight, None, "silu")
        y_3d, _ = causal_conv1d(x, weight.unsqueeze(1), None, "silu")
        assert torch.equal(y_flat, y_3d)


class TestRmsNormGated:
    """rms_norm_gated: CPU fp32 math, the two NPU regimes, and the unsupported-option rejections."""

    def test_cpu_fp32_matches_manual_reference_bitwise(self) -> None:
        torch.manual_seed(7)
        x = torch.randn(2, 16, 64, dtype=torch.float32)
        g = torch.randn(2, 16, 64, dtype=torch.float32)
        weight = torch.rand(64, dtype=torch.float32) + 0.5
        y = rms_norm_gated(x, g, weight)
        assert y.dtype == torch.float32
        assert torch.equal(y, _rms_reference(x, g, weight))

    def test_cpu_low_precision_input_matches_fp32_math(self) -> None:
        torch.manual_seed(8)
        x = torch.randn(2, 16, 64, dtype=torch.bfloat16)
        g = torch.randn(2, 16, 64, dtype=torch.bfloat16)
        weight = torch.rand(64, dtype=torch.float32) + 0.5
        y = rms_norm_gated(x, g, weight)
        assert y.dtype == torch.bfloat16
        assert torch.equal(y, _rms_reference(x, g, weight))

    def test_rejects_unsupported_options(self) -> None:
        x = torch.randn(2, 8, 16, dtype=torch.float32)
        g = torch.randn(2, 8, 16, dtype=torch.float32)
        weight = torch.rand(16, dtype=torch.float32) + 0.5
        with pytest.raises(NotImplementedError, match="bias"):
            rms_norm_gated(x, g, weight, bias=torch.zeros(16))
        with pytest.raises(NotImplementedError, match="sigmoid"):
            rms_norm_gated(x, g, weight, activation="relu")
        with pytest.raises(NotImplementedError, match="sigmoid"):
            rms_norm_gated(x, g, weight, activation=None)
        with pytest.raises(NotImplementedError, match="residual"):
            rms_norm_gated(x, g, weight, residual=torch.zeros_like(x))
        with pytest.raises(NotImplementedError, match="residual"):
            rms_norm_gated(x, g, weight, prenorm=True)
        with pytest.raises(NotImplementedError, match="residual"):
            rms_norm_gated(x, g, weight, residual_in_fp32=True)

    @pytest.mark.gpu
    @requires_npu
    def test_npu_fp32_matches_cpu_reference(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Not bit-exact: npu_rms_norm reduces in a different order (observed 1 fp32 ulp), so the
        # budget here is 1e-6 rather than bitwise equality.
        monkeypatch.setenv("XTUNER_NPU_FUSED_GATED_NORM", "0")
        torch.manual_seed(9)
        x = torch.randn(2, 64, 256, dtype=torch.float32)
        g = torch.randn(2, 64, 256, dtype=torch.float32)
        weight = torch.rand(256, dtype=torch.float32) + 0.5
        y_npu = rms_norm_gated(x.to("npu"), g.to("npu"), weight.to("npu"))
        y_cpu = rms_norm_gated(x, g, weight)
        assert y_npu.dtype == torch.float32
        _assert_parity(y_npu, y_cpu, "rms fp32", atol=1e-6)

    @pytest.mark.gpu
    @requires_npu
    def test_npu_bf16_regime_within_two_ulp_of_fp32_regime(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XTUNER_NPU_FUSED_GATED_NORM", "1")
        torch.manual_seed(10)
        x = torch.randn(2, 64, 256, dtype=torch.bfloat16)
        g = torch.randn(2, 64, 256, dtype=torch.bfloat16)
        weight = torch.rand(256, dtype=torch.float32) + 0.5
        y_bf16 = rms_norm_gated(x.to("npu"), g.to("npu"), weight.to("npu"))
        y_fp32_regime = rms_norm_gated(x.float().to("npu"), g.float().to("npu"), weight.to("npu")).to(torch.bfloat16)
        assert y_bf16.dtype == torch.bfloat16
        distance = _bf16_ulp_distance(y_bf16, y_fp32_regime)
        assert bool((distance <= 2).all()), f"max {distance.max().item()} bf16 ulp, expected <= 2"

    @pytest.mark.gpu
    @requires_npu
    def test_env_switches_gated_norm_regime(self, monkeypatch: pytest.MonkeyPatch) -> None:
        torch.manual_seed(11)
        x = torch.randn(2, 32, 128, dtype=torch.bfloat16)
        g = torch.randn(2, 32, 128, dtype=torch.bfloat16)
        weight = torch.rand(128, dtype=torch.float32) + 0.5
        monkeypatch.setenv("XTUNER_NPU_FUSED_GATED_NORM", "0")
        y_env0 = rms_norm_gated(x.to("npu"), g.to("npu"), weight.to("npu"))
        monkeypatch.setenv("XTUNER_NPU_FUSED_GATED_NORM", "1")
        y_env1 = rms_norm_gated(x.to("npu"), g.to("npu"), weight.to("npu"))
        y_fp32_regime = rms_norm_gated(x.float().to("npu"), g.float().to("npu"), weight.to("npu")).to(torch.bfloat16)
        # env=0 routes the bf16 input through the fp32 regime, bit-for-bit.
        assert torch.equal(y_env0, y_fp32_regime)
        assert not torch.equal(y_env1, y_env0), "env=1 must select the fused bf16 regime"
        assert torch.equal(y_env1, _bf16_regime_reference(x.to("npu"), g.to("npu"), weight.to("npu")))


class TestBackendModules:
    """The fla-surface base modules' init state and their deliberately inert forwards."""

    def test_fused_rms_norm_gated_init_attrs(self) -> None:
        module = FusedRMSNormGated(64, eps=1e-6)
        assert module.weight.shape == (64,)
        assert bool(torch.all(module.weight == 1.0))
        assert module.bias is None
        assert module.eps == 1e-6
        assert module.activation == "sigmoid"

    def test_fused_rms_norm_gated_rejects_bad_activation(self) -> None:
        with pytest.raises(NotImplementedError, match="relu"):
            FusedRMSNormGated(64, activation="relu")

    def test_fused_rms_norm_gated_forward_raises(self) -> None:
        module = FusedRMSNormGated(16)
        x = torch.randn(2, 4, 16)
        with pytest.raises(NotImplementedError):
            module(x, torch.randn(2, 4, 16))

    def test_short_convolution_init_attrs(self) -> None:
        module = ShortConvolution(32, kernel_size=4)
        assert module.conv_dim == 32
        assert module.kernel_size == 4
        assert module.weight.shape == (32, 1, 4)
        assert module.bias is None
        assert module.activation == "silu"
        assert module.backend == "ascendc"
        assert bool(torch.isfinite(module.weight).all())

    def test_short_convolution_bias_param(self) -> None:
        module = ShortConvolution(32, kernel_size=4, bias=True)
        assert module.bias is not None
        assert module.bias.shape == (32,)
        # reset_parameters re-inits the bias uniformly within 1/sqrt(fan_in), nn.Conv1d-style.
        assert float(module.bias.abs().max()) <= 1 / math.sqrt(32)

    def test_short_convolution_forward_raises(self) -> None:
        module = ShortConvolution(16, kernel_size=4)
        with pytest.raises(NotImplementedError):
            module(torch.randn(1, 8, 16))

    def test_short_convolution_reset_parameters_reproducible(self) -> None:
        torch.manual_seed(12)
        first = ShortConvolution(16, kernel_size=4)
        torch.manual_seed(12)
        second = ShortConvolution(16, kernel_size=4)
        assert torch.equal(first.weight, second.weight)
        torch.manual_seed(12)
        first.reset_parameters()
        torch.manual_seed(12)
        second.reset_parameters()
        assert torch.equal(first.weight, second.weight)
        assert first.weight.shape == (16, 1, 4)


class TestGetterDispatch:
    """With XTUNER_KDA_BACKEND=npu the three getters hand out the npu_backend implementations."""

    def test_getters_return_npu_backend_impls_when_env_npu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XTUNER_KDA_BACKEND", "npu")
        assert get_chunk_kda_fn() is npu_backend.chunk_kda
        assert get_fused_kda_gate_fn() is npu_backend.fused_kda_gate
        assert get_causal_conv1d_fn() is npu_backend.causal_conv1d
