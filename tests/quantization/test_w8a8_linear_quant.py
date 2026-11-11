"""Tests for the W8A8 dynamic linear QAT feature: env config, kernels, modules, decoder-layer wiring, HF export and the downstream consumer contract.

Tests run on CPU unless a path needs the deployment kernel itself: the NPU fused-kernel
tests and the export's training-grid consistency tests are gated on torch_npu
availability; MindSpeed's ``npu_quant_gmm`` and ``_dual_path_gmm``'s real-op plumbing are
exercised through recorded stubs. The consumer-contract half checks the export
against the verbatim downstream logic copied in ``_w8a8_consumer_ref.py`` (vllm /
vllm-ascend), so a change on either side of the contract breaks a test here.
"""

import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F

from xtuner.v1.float8.config import Float8Config, ScalingGranularity
from xtuner.v1.module.grouped_linear.moe_group_linear import GroupedLinear, build_grouped_linear
from xtuner.v1.module.linear.linear import _Linear, build_linear
from xtuner.v1.quantization.kv.kernels import int8_fake_quant_static
from xtuner.v1.quantization.linear import (
    Int8W8A8DynamicGroupedLinear,
    Int8W8A8DynamicLinear,
    LinearQuantConfig,
    int8_fake_quant_dynamic,
    linear_quant_config_from_env,
    maybe_write_hf_w8a8_linear,
    per_channel_weight_fake_quant,
    per_channel_weight_quant,
    per_token_act_fake_quant,
    per_token_act_quant,
)
from xtuner.v1.quantization.linear import export as w8a8_export
from xtuner.v1.quantization.linear.config import _dequant_grid_from_env
from xtuner.v1.quantization.linear.kernels import _per_channel_weight_scale

# Relative import: a regular "tests" package in site-packages shadows the repo's
# tests/ namespace, so absolute "tests.quantization..." can never resolve.
from ._w8a8_consumer_ref import (
    AscendModelSlimLookupConfig,
    AscendMoEMethodHarness,
    AscendW8A8DynamicFusedMoEMethod,
    AscendW8A8DynamicLinearMethod,
    RoutedExpertsScaleLoaderHarness,
)


_H = 16
_ISP = 8
_E = 4
_HF_INDEX_FILENAME = "model.safetensors.index.json"
_DESCRIPTION_FILENAME = "quant_model_description.json"
_BYSTANDER_KEY = "model.language_model.layers.1.mlp.gate.weight"
_TORCH_NPU_AVAILABLE = importlib.util.find_spec("torch_npu") is not None


class _FakeMesh:
    """Duck-typed TP mesh: GroupedLinear only ever calls size()/get_local_rank()."""

    def size(self) -> int:
        return 2

    def get_local_rank(self) -> int:
        return 0


class _FakeGlm53Text(nn.Module):
    """Text-model stand-in carrying a verbatim mirror of GLM53's ``to_hf_key_list``."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=False, num_hidden_layers=2, n_routed_experts=_E)

    # Verbatim mirror of xtuner/v1/model/moe/glm53/glm53.py GLM53.to_hf_key_list: the export
    # must stay aligned with the mapping the real model applies to the rest of the checkpoint.
    def to_hf_key_list(self, key: str) -> list[str]:
        if self.config.tie_word_embeddings and "lm_head" in key:
            key = key.replace("lm_head", "embed_tokens")

        if key.startswith("mtp_block."):
            match = re.match(r"mtp_block\.layers\.0\.(.+)", key)
            assert match is not None, f"Unexpected GLM-5.3-Flash MTP key: {key}"
            key = f"layers.{self.config.num_hidden_layers}.{match.group(1)}"
            key = key.replace(".decoder_layer.", ".")
            key = re.sub(r"layers\.(\d+)\.final_layernorm\.", r"layers.\1.shared_head.norm.", key)

        if "layers" in key or "embed_tokens" in key:
            key = "model.language_model." + key
        elif key.startswith("norm."):
            return [key.replace("norm.", "model.language_model.norm.")]

        if "layers" in key:
            key = re.sub(r"layers\.(\d+)\.(experts|gate|shared_experts)", r"layers.\1.mlp.\2", key)

        if "fused_w1w3.weight" in key:
            return [
                key.replace("fused_w1w3.weight", f"{i}.{proj}_proj.weight")
                for i in range(self.config.n_routed_experts)
                for proj in ("gate", "up")
            ]
        if "fused_w2.weight" in key:
            return [
                key.replace("fused_w2.weight", f"{i}.down_proj.weight") for i in range(self.config.n_routed_experts)
            ]
        if "router.e_score_correction_bias" in key:
            return [key.replace("router.e_score_correction_bias", "e_score_correction_bias")]
        return [key]


class _FakeGlm53Compose(nn.Module):
    """Compose wrapper holding the text model under ``language_model`` like the real model."""

    def __init__(self, text_model: _FakeGlm53Text) -> None:
        super().__init__()
        self.language_model = text_model


def _place_module(root: nn.Module, path: str, leaf: nn.Module) -> None:
    parts = path.split(".")
    node = root
    for part in parts[:-1]:
        child = getattr(node, part, None)
        if child is None:
            child = nn.Module()
            node.add_module(part, child)
        node = child
    node.add_module(parts[-1], leaf)


def _dense_key(layer: int, proj: str) -> str:
    return f"model.language_model.layers.{layer}.mlp.{proj}_proj.weight"


def _expert_key(i: int, proj: str) -> str:
    return f"model.language_model.layers.1.mlp.experts.{i}.{proj}_proj.weight"


def _bystander() -> torch.Tensor:
    # Deterministic values so the test can recompute them after the export rewrote the shard.
    return torch.arange(4 * _H, dtype=torch.float32).reshape(4, _H).to(torch.bfloat16)


def _full_w8a8_tree() -> tuple[_FakeGlm53Compose, dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    masters: dict[str, torch.Tensor] = {}
    for proj in ("gate", "up", "down"):
        out, inp = (_ISP, _H) if proj != "down" else (_H, _ISP)
        masters[_dense_key(0, proj)] = torch.randn(out, inp, dtype=torch.bfloat16)
    for i in range(_E):
        for proj in ("gate", "up", "down"):
            out, inp = (_ISP, _H) if proj != "down" else (_H, _ISP)
            masters[_expert_key(i, proj)] = torch.randn(out, inp, dtype=torch.bfloat16)

    text = _FakeGlm53Text()
    root = _FakeGlm53Compose(text)
    dense_gate = Int8W8A8DynamicLinear(_H, _ISP, bias=False)
    dense_up = Int8W8A8DynamicLinear(_H, _ISP, bias=False)
    dense_down = Int8W8A8DynamicLinear(_ISP, _H, bias=False)
    with torch.no_grad():
        dense_gate.weight.copy_(masters[_dense_key(0, "gate")])
        dense_up.weight.copy_(masters[_dense_key(0, "up")])
        dense_down.weight.copy_(masters[_dense_key(0, "down")])
    _place_module(text, "layers.0.mlp.gate_proj", dense_gate)
    _place_module(text, "layers.0.mlp.up_proj", dense_up)
    # The activation-checkpointing wrapper infix must be unwrapped by the export.
    _place_module(text, "layers.0.mlp._checkpoint_wrapped_module.down_proj", dense_down)

    # Fused grouped weights hold expert i's rows in block i (gate stacked over up for w1w3).
    fused_w1w3 = Int8W8A8DynamicGroupedLinear(_H, 2 * _ISP, _E, num_fused_projections=2)
    fused_w2 = Int8W8A8DynamicGroupedLinear(_ISP, _H, _E)
    w13 = torch.stack(
        [torch.cat([masters[_expert_key(i, "gate")], masters[_expert_key(i, "up")]], dim=0) for i in range(_E)]
    )
    w2 = torch.stack([masters[_expert_key(i, "down")] for i in range(_E)])
    with torch.no_grad():
        fused_w1w3.weight.copy_(w13.flatten(0, 1))
        fused_w2.weight.copy_(w2.flatten(0, 1))
    _place_module(text, "layers.1.experts.fused_w1w3", fused_w1w3)
    _place_module(text, "layers.1.experts.fused_w2", fused_w2)

    # A bystander linear the feature must leave alone.
    _place_module(text, "layers.2.mlp.gate_proj", nn.Linear(_H, _ISP, bias=False))
    return root, masters


def _small_tree() -> tuple[_FakeGlm53Compose, torch.Tensor]:
    torch.manual_seed(0)
    master = torch.randn(_ISP, _H, dtype=torch.bfloat16)
    text = _FakeGlm53Text()
    root = _FakeGlm53Compose(text)
    layer = Int8W8A8DynamicLinear(_H, _ISP, bias=False)
    with torch.no_grad():
        layer.weight.copy_(master)
    _place_module(text, "layers.0.mlp.gate_proj", layer)
    return root, master


def _write_full_fixture(hf_dir: Path, masters: dict[str, torch.Tensor]) -> None:
    hf_dir.mkdir(parents=True)
    dense_keys = [_dense_key(0, proj) for proj in ("gate", "up", "down")]
    expert_keys = [_expert_key(i, proj) for i in range(_E) for proj in ("gate", "up", "down")]
    save_file({key: masters[key] for key in dense_keys}, hf_dir / "model-00001-of-00002.safetensors")
    expert_tensors = {key: masters[key] for key in expert_keys}
    expert_tensors[_BYSTANDER_KEY] = _bystander()
    save_file(expert_tensors, hf_dir / "model-00002-of-00002.safetensors")
    weight_map = {key: "model-00001-of-00002.safetensors" for key in dense_keys}
    weight_map.update({key: "model-00002-of-00002.safetensors" for key in expert_tensors})
    index = {"metadata": {"total_size": 12345}, "weight_map": weight_map}
    (hf_dir / _HF_INDEX_FILENAME).write_text(json.dumps(index, indent=4) + "\n")


def _write_shard_and_index(
    hf_dir: Path,
    tensors: dict[str, torch.Tensor],
    *,
    shard_omit: str | None = None,
    index_omit: str | None = None,
) -> None:
    hf_dir.mkdir(parents=True)
    shard_name = "model-00001-of-00001.safetensors"
    save_file({key: value for key, value in tensors.items() if key != shard_omit}, hf_dir / shard_name)
    weight_map = {key: shard_name for key in tensors if key != index_omit}
    index = {"metadata": {"total_size": 12345}, "weight_map": weight_map}
    (hf_dir / _HF_INDEX_FILENAME).write_text(json.dumps(index, indent=4) + "\n")


def _load_all(hf_dir: Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for shard in sorted(hf_dir.glob("*.safetensors")):
        tensors.update(load_file(shard))
    return tensors


def _training_grid_reference(master: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # The grid the NPU training forward fake-quantized against: the deployment kernel on
    # an NPU box, the eager chain on a CPU-only box (the export's documented fallback).
    # Written out here instead of calling the export's own helper so the reference stays
    # independent of the code under test.
    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available() and master.dtype in (torch.bfloat16, torch.float16):
        q, scale = per_channel_weight_quant(master.to("npu"))
        return q.cpu(), scale.cpu()
    return per_channel_weight_quant(master)


def _training_fake_quant_reference(w: torch.Tensor) -> torch.Tensor:
    # Fake-quant twin of ``_training_grid_reference``: what the NPU training forward
    # actually consumed (deployment kernel on an NPU box, eager chain on CPU-only).
    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available() and w.dtype in (torch.bfloat16, torch.float16):
        return per_channel_weight_fake_quant(w.to("npu")).cpu()
    return per_channel_weight_fake_quant(w)


def _set_linear_env(
    monkeypatch: pytest.MonkeyPatch,
    main: str | None,
    gmm: str | None = None,
    dequant: str | None = None,
) -> None:
    if main is None:
        monkeypatch.delenv("XTUNER_QUANT_LINEAR", raising=False)
    else:
        monkeypatch.setenv("XTUNER_QUANT_LINEAR", main)
    if gmm is None:
        monkeypatch.delenv("XTUNER_QUANT_LINEAR_GMM", raising=False)
    else:
        monkeypatch.setenv("XTUNER_QUANT_LINEAR_GMM", gmm)
    if dequant is None:
        monkeypatch.delenv("XTUNER_QUANT_LINEAR_DEQUANT", raising=False)
    else:
        monkeypatch.setenv("XTUNER_QUANT_LINEAR_DEQUANT", dequant)


class TestLinearQuantConfig:
    def test_off_values(self, monkeypatch):
        for raw in (None, "", "off", "0", "false", "OFF", "  Off  ", "FALSE"):
            _set_linear_env(monkeypatch, raw)
            assert linear_quant_config_from_env() is None

    def test_w8a8_dynamic_enables_config(self, monkeypatch):
        _set_linear_env(monkeypatch, "w8a8_dynamic")
        cfg = linear_quant_config_from_env()
        assert isinstance(cfg, LinearQuantConfig)
        assert cfg.linear_quant == "w8a8_dynamic"
        assert cfg.gmm_engine == "fake"

    def test_w8a8_dynamic_case_insensitive(self, monkeypatch):
        _set_linear_env(monkeypatch, " W8A8_DYNAMIC ")
        cfg = linear_quant_config_from_env()
        assert cfg is not None
        assert cfg.linear_quant == "w8a8_dynamic"

    def test_gmm_engine_fake_values(self, monkeypatch):
        for raw in (None, "", "fake", "FAKE", " Fake "):
            _set_linear_env(monkeypatch, "w8a8_dynamic", raw)
            cfg = linear_quant_config_from_env()
            assert cfg is not None
            assert cfg.gmm_engine == "fake"

    def test_gmm_engine_npu_quant_gmm(self, monkeypatch):
        _set_linear_env(monkeypatch, "w8a8_dynamic", "npu_quant_gmm")
        cfg = linear_quant_config_from_env()
        assert cfg is not None
        assert cfg.gmm_engine == "npu_quant_gmm"

    def test_gmm_engine_invalid(self, monkeypatch):
        _set_linear_env(monkeypatch, "w8a8_dynamic", "bogus")
        with pytest.raises(ValueError, match="expected 'fake' or 'npu_quant_gmm'"):
            linear_quant_config_from_env()

    def test_gmm_engine_requires_main_switch(self, monkeypatch):
        _set_linear_env(monkeypatch, "off", "npu_quant_gmm")
        with pytest.raises(ValueError, match="requires XTUNER_QUANT_LINEAR='w8a8_dynamic'"):
            linear_quant_config_from_env()

    def test_gmm_aux_defaults_pass_when_off(self, monkeypatch):
        # Unset / default aux values keep the feature-off no-op working (launch scripts
        # export both variables unconditionally).
        for main in ("off", None, ""):
            for gmm in (None, "", "fake", "FAKE"):
                _set_linear_env(monkeypatch, main, gmm)
                assert linear_quant_config_from_env() is None

    def test_gmm_engine_invalid_raises_even_when_off(self, monkeypatch):
        # Symmetric with the dequant variable: both aux values parse before the mode
        # branch, so a typo raises with the feature off too.
        for main in ("off", None, ""):
            _set_linear_env(monkeypatch, main, "bogus")
            with pytest.raises(ValueError, match="expected 'fake' or 'npu_quant_gmm'"):
                linear_quant_config_from_env()

    def test_dequant_grid_parse_values(self, monkeypatch):
        for raw, expect in (
            (None, "fp32"),
            ("", "fp32"),
            ("fp32", "fp32"),
            ("FP32", "fp32"),
            ("  fp32  ", "fp32"),
            ("native", "native"),
            ("  Native  ", "native"),
        ):
            _set_linear_env(monkeypatch, "w8a8_dynamic", dequant=raw)
            assert _dequant_grid_from_env() == expect

    def test_dequant_grid_invalid_raises_even_when_off(self, monkeypatch):
        # A typo must never pass silently: the value is validated before the mode branch,
        # so it raises with the feature off too (both aux variables share this rule).
        for main in ("off", "w8a8_dynamic", None):
            _set_linear_env(monkeypatch, main, dequant="bogus")
            with pytest.raises(ValueError, match="expected 'fp32' or 'native'"):
                linear_quant_config_from_env()

    def test_dequant_native_requires_main_switch(self, monkeypatch):
        for main in ("off", None):
            _set_linear_env(monkeypatch, main, dequant="native")
            with pytest.raises(ValueError, match="requires XTUNER_QUANT_LINEAR='w8a8_dynamic'"):
                linear_quant_config_from_env()

    def test_dequant_grid_accepted_with_w8a8_dynamic(self, monkeypatch):
        for dequant in ("native", "fp32", None):
            _set_linear_env(monkeypatch, "w8a8_dynamic", dequant=dequant)
            cfg = linear_quant_config_from_env()
            assert cfg is not None
            assert cfg.linear_quant == "w8a8_dynamic"
            assert cfg.gmm_engine == "fake"

    def test_invalid_main_value(self, monkeypatch):
        for raw in ("int4", "w8a8", "on"):
            _set_linear_env(monkeypatch, raw)
            with pytest.raises(ValueError, match="expected 'off' or 'w8a8_dynamic'"):
                linear_quant_config_from_env()

    def test_pydantic_rejects_unknown_values(self):
        with pytest.raises(ValidationError):
            LinearQuantConfig(linear_quant="bogus")
        with pytest.raises(ValidationError):
            LinearQuantConfig(linear_quant="w8a8_dynamic", gmm_engine="bogus")
        with pytest.raises(ValidationError):
            LinearQuantConfig(surprise=1)


class TestLinearQuantKernels:
    def test_per_channel_grid_identity(self):
        # levels on the int8 grid with per-row max |level| == 127 and a power-of-two 0.5 step:
        # the per-row scale is exactly 0.5 so round-tripping must be bitwise exact.
        levels = torch.randint(-127, 128, (6, _H), dtype=torch.float32)
        levels[:, 0] = 127.0
        w = levels * 0.5
        fake = per_channel_weight_fake_quant(w)
        assert torch.equal(fake, w)

        q, scale = per_channel_weight_quant(w)
        assert q.dtype == torch.int8
        assert torch.equal(q.float(), levels)
        assert torch.equal(scale, torch.full((6, 1), 0.5))

    def test_dequantized_export_matches_training_fake_quant(self):
        # The deployment engine consumes (q, scale); the tensor it reconstructs must be
        # bitwise-equal to what training fake-quantized — both compute q*scale in fp32 and
        # cast back, so parity holds for bf16 (training dtype) and fp32 alike.
        for dtype in (torch.bfloat16, torch.float32):
            w = torch.randn(8, _H, dtype=torch.float32).to(dtype)
            q, scale = per_channel_weight_quant(w)
            recon = (q.float() * scale).to(dtype)
            assert torch.equal(recon, per_channel_weight_fake_quant(w))

    def test_per_channel_zero_rows(self):
        w = torch.randn(5, _H)
        w[2] = 0.0
        fake = per_channel_weight_fake_quant(w)
        assert torch.isfinite(fake).all()
        assert torch.equal(fake[2], torch.zeros_like(fake[2]))
        q, scale = per_channel_weight_quant(w)
        assert torch.equal(scale[2], torch.zeros(1))
        assert torch.equal(q[2].float(), torch.zeros(_H))
        recon = (q.float() * scale).to(w.dtype)
        assert torch.equal(recon[2], torch.zeros(_H))

    def test_per_token_act_quant(self):
        x = torch.randn(3, _H)
        x[1] = 0.0
        q, s = per_token_act_quant(x)
        assert q.shape == x.shape
        assert q.dtype == torch.int8
        assert s.shape == (3,)
        assert s.dtype == torch.float32
        # All-zero token: the scale floor kicks in and the levels stay zero.
        assert torch.equal(s[1], torch.full((), 1e-12))
        assert torch.equal(q[1].float(), torch.zeros(_H))
        # Non-empty tokens saturate their row amax to level 127.
        assert q[0].abs().max().item() == 127
        assert q[2].abs().max().item() == 127
        # Same grid as the dynamic fake-quant the forward path uses.
        fake = int8_fake_quant_dynamic(x)
        assert torch.equal(q.float() * s.unsqueeze(-1), fake.float())

    def test_ste_backward_passthrough(self):
        w = torch.randn(4, _H, requires_grad=True)
        out = per_channel_weight_fake_quant(w)
        grad = torch.randn_like(out)
        out.backward(grad)
        assert torch.equal(w.grad, grad)

        x = torch.randn(4, _H, requires_grad=True)
        out = int8_fake_quant_dynamic(x)
        grad = torch.randn_like(out)
        out.backward(grad)
        assert torch.equal(x.grad, grad)


class TestLinearQuantModules:
    def test_build_linear_dispatch(self):
        cfg = LinearQuantConfig(linear_quant="w8a8_dynamic")
        layer = build_linear(_H, _ISP, bias=False, linear_quant_cfg=cfg)
        assert isinstance(layer, Int8W8A8DynamicLinear)
        plain = build_linear(_H, _ISP, bias=False)
        assert isinstance(plain, _Linear)
        assert not isinstance(plain, Int8W8A8DynamicLinear)

    def test_dense_forward_parity(self):
        torch.manual_seed(1)
        layer = Int8W8A8DynamicLinear(_H, _ISP, bias=False)
        with torch.no_grad():
            layer.weight.copy_(torch.randn(_ISP, _H))
        x = torch.randn(6, _H)
        expected = F.linear(int8_fake_quant_dynamic(x), per_channel_weight_fake_quant(layer.weight), layer.bias)
        assert torch.equal(layer(x), expected)

        biased = Int8W8A8DynamicLinear(_H, _ISP)
        expected = F.linear(int8_fake_quant_dynamic(x), per_channel_weight_fake_quant(biased.weight), biased.bias)
        assert torch.equal(biased(x), expected)

    def test_float8_mutual_exclusion_dense(self):
        f8 = Float8Config(scaling_granularity_gemm=ScalingGranularity.TILEWISE)
        with pytest.raises(ValueError, match="mutually exclusive"):
            build_linear(_H, _ISP, float8_cfg=f8, linear_quant_cfg=LinearQuantConfig(linear_quant="w8a8_dynamic"))

    def test_float8_mutual_exclusion_grouped(self):
        f8 = Float8Config(scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE)
        with pytest.raises(ValueError, match="mutually exclusive"):
            build_grouped_linear(
                _H, _ISP, _E, float8_cfg=f8, linear_quant_cfg=LinearQuantConfig(linear_quant="w8a8_dynamic")
            )

    def test_float8_non_conflicting_fields_coexist(self):
        # Each builder only conflicts with the float8 field it would actually replace.
        cfg = LinearQuantConfig(linear_quant="w8a8_dynamic")
        grouped_only = Float8Config(scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE)
        layer = build_linear(_H, _ISP, float8_cfg=grouped_only, linear_quant_cfg=cfg)
        assert isinstance(layer, Int8W8A8DynamicLinear)
        gemm_only = Float8Config(scaling_granularity_gemm=ScalingGranularity.TILEWISE)
        grouped = build_grouped_linear(_H, _ISP, _E, float8_cfg=gemm_only, linear_quant_cfg=cfg)
        assert isinstance(grouped, Int8W8A8DynamicGroupedLinear)

    def test_build_grouped_linear_dispatch(self):
        cfg = LinearQuantConfig(linear_quant="w8a8_dynamic")
        grouped = build_grouped_linear(_H, 2 * _ISP, _E, num_fused_projections=2, linear_quant_cfg=cfg)
        assert isinstance(grouped, Int8W8A8DynamicGroupedLinear)
        assert isinstance(grouped, GroupedLinear)
        assert tuple(grouped.weight.shape) == (_E * 2 * _ISP, _H)
        assert grouped._quant_gmm is None
        plain = build_grouped_linear(_H, _ISP, _E)
        assert type(plain) is GroupedLinear

    def test_gmm_engine_wiring(self):
        try:
            from mindspeed.ops.quant_gmm import npu_quant_gmm
        except ImportError:
            with pytest.raises(ImportError, match="npu_quant_gmm"):
                Int8W8A8DynamicGroupedLinear(_H, _ISP, _E, gmm_engine="npu_quant_gmm")
            return
        grouped = Int8W8A8DynamicGroupedLinear(_H, _ISP, _E, gmm_engine="npu_quant_gmm")
        assert grouped.gmm_engine == "npu_quant_gmm"
        assert grouped._quant_gmm is npu_quant_gmm

    def test_row_parallel_guard(self):
        with pytest.raises(NotImplementedError, match="EP only"):
            Int8W8A8DynamicGroupedLinear(_H, _ISP, _E, expert_tp_mesh=_FakeMesh(), parallel_style="row")

    def test_column_parallel_allowed(self):
        grouped = Int8W8A8DynamicGroupedLinear(_H, _ISP, _E, expert_tp_mesh=_FakeMesh(), parallel_style="column")
        assert grouped.local_out_features == _ISP // 2
        assert tuple(grouped.weight.shape) == (_E * (_ISP // 2), _H)

    def test_dual_path_gmm_empty_batch(self):
        grouped = Int8W8A8DynamicGroupedLinear(_H, _ISP, _E)

        def _boom(*args, **kwargs):
            raise AssertionError("the real op must not run on an empty batch")

        grouped._quant_gmm = _boom
        fake_out = torch.randn(0, _ISP)
        out = grouped._dual_path_gmm(
            torch.zeros(0, _H), grouped.weight.view(_E, _ISP, _H), torch.zeros(_E, dtype=torch.int64), fake_out
        )
        assert out is fake_out

    def test_dual_path_gmm_plumbing(self):
        torch.manual_seed(2)
        grouped = Int8W8A8DynamicGroupedLinear(_H, _ISP, _E)
        # GroupedLinear leaves weights uninitialized; a NaN bit pattern in the recycled
        # memory would make the value-equality asserts below fail spuriously.
        with torch.no_grad():
            grouped.weight.normal_()
        recorded: dict[str, object] = {}
        const = torch.zeros(6, _ISP)

        def _fake_quant_gmm(qx, qw, scale, *, offset, per_token_scale, bias, group_list, output_dtype):
            recorded["qx"] = qx
            recorded["qw"] = qw
            recorded["scale"] = scale
            recorded["offset"] = offset
            recorded["per_token_scale"] = per_token_scale
            recorded["bias"] = bias
            recorded["group_list"] = group_list
            recorded["output_dtype"] = output_dtype
            return const

        grouped._quant_gmm = _fake_quant_gmm
        x = torch.randn(6, _H)
        weight = grouped.weight.view(_E, _ISP, _H)
        fake_out = torch.randn(6, _ISP)

        out = grouped._dual_path_gmm(x, weight, torch.tensor([2, 1, 3, 0]), fake_out)

        # npu_quant_gmm contract: int8 x [M, K], int8 weight [E, K, N] (K-major — transposed
        # from the training layout), bf16 per-channel scale [E, N], fp32 per-token scale [M],
        # int64 cumulative group_list, bf16 output.
        qx = recorded["qx"]
        qw = recorded["qw"]
        scale = recorded["scale"]
        assert qx.dtype == torch.int8
        assert tuple(qx.shape) == (6, _H)
        assert torch.equal(qx, per_token_act_quant(x)[0])
        assert qw.dtype == torch.int8
        assert tuple(qw.shape) == (_E, _H, _ISP)
        assert torch.equal(qw, per_channel_weight_quant(weight)[0].transpose(1, 2))
        assert scale.dtype == torch.bfloat16
        assert tuple(scale.shape) == (_E, _ISP)
        assert torch.equal(scale, per_channel_weight_quant(weight)[1].squeeze(-1).to(torch.bfloat16))
        assert recorded["offset"] is None
        assert recorded["bias"] is None
        per_token_scale = recorded["per_token_scale"]
        assert per_token_scale.dtype == torch.float32
        assert tuple(per_token_scale.shape) == (6,)
        assert torch.equal(per_token_scale, per_token_act_quant(x)[1])
        assert torch.equal(recorded["group_list"], torch.tensor([2, 3, 6, 6], dtype=torch.int64))
        assert recorded["output_dtype"] is torch.bfloat16
        # real.detach() + (fake - fake.detach()) with real == 0 is exactly 0: the real
        # value flows bitwise (the correction term is exactly zero).
        assert torch.equal(out, torch.zeros(6, _ISP))

    def test_dual_path_gmm_detach_semantics(self):
        torch.manual_seed(3)
        grouped = Int8W8A8DynamicGroupedLinear(_H, _ISP, _E)
        grouped._quant_gmm = lambda *args, **kwargs: torch.zeros(2, _ISP)
        x = torch.randn(2, _H, requires_grad=True)
        weight = grouped.weight.view(_E, _ISP, _H).detach().clone().requires_grad_(True)
        # A fake arm with the (tokens, out) shape whose gradient is 2.0 w.r.t. every input.
        fake_out = x.sum(dim=-1, keepdim=True) * torch.full((1, _ISP), 2.0 / _ISP)

        out = grouped._dual_path_gmm(x, weight, torch.tensor([1, 1, 0, 0]), fake_out)

        # Value comes from the real arm; gradient comes only from the fake arm.
        assert torch.equal(out, torch.zeros(2, _ISP))
        out.sum().backward()
        assert torch.equal(x.grad, torch.full((2, _H), 2.0))
        assert weight.grad is None


class TestFusedA2AGmmQuantHooks:
    """The fused a2a+gmm path consumes quantization through duck-typed module hooks."""

    def test_plain_grouped_linear_has_no_hooks(self):
        plain = GroupedLinear(_H, _ISP, _E)
        assert getattr(plain, "fused_weight_fake_quant", None) is None
        assert getattr(plain, "fused_act_fake_quant", None) is None

    def test_fused_weight_hook_matches_kernel_and_is_ste(self):
        torch.manual_seed(4)
        grouped = Int8W8A8DynamicGroupedLinear(_H, _ISP, _E)
        weight = torch.randn(_E, _ISP, _H, requires_grad=True)
        out = grouped.fused_weight_fake_quant(weight)
        assert torch.equal(out, per_channel_weight_fake_quant(weight))
        grad = torch.randn_like(out)
        out.backward(grad)
        # Straight-through estimator: the backward is the identity.
        assert torch.equal(weight.grad, grad)

    def test_fused_act_hook_matches_kernel_zero_row_and_permutation(self):
        torch.manual_seed(5)
        grouped = Int8W8A8DynamicGroupedLinear(_H, _ISP, _E)
        x = torch.randn(6, _H)
        x[2] = 0  # the chunked path's structural pad row
        out = grouped.fused_act_fake_quant(x)
        assert torch.equal(out, int8_fake_quant_dynamic(x))
        assert torch.equal(out[2], torch.zeros(_H))  # exact zeros through the scale floor, no NaN
        perm = torch.tensor([3, 0, 5, 1, 4, 2])
        # The pre-dispatch license: per-token scales are row-local, so quantizing before the
        # a2a row permutation equals quantizing after it.
        assert torch.equal(grouped.fused_act_fake_quant(x[perm]), out[perm])

    def test_fused_weight_hook_refuses_real_op_engine(self):
        try:
            from mindspeed.ops.quant_gmm import npu_quant_gmm  # noqa: F401
        except ImportError:
            pytest.skip("MindSpeed npu_quant_gmm not importable")
        grouped = Int8W8A8DynamicGroupedLinear(_H, _ISP, _E, gmm_engine="npu_quant_gmm")
        with pytest.raises(ValueError, match="XTUNER_MOE_FUSED_A2A_GMM"):
            grouped.fused_weight_fake_quant(torch.randn(_E, _ISP, _H))


@pytest.mark.skipif(not _TORCH_NPU_AVAILABLE, reason="moe_decoder_layer imports torch_npu at module level")
class TestMoeDecoderLayerWiring:
    def test_moe_block_builds_quantized_grouped_linears(self, monkeypatch):
        from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEActFnConfig, MoEBlock

        monkeypatch.setenv("XTUNER_QUANT_LINEAR", "w8a8_dynamic")
        block = MoEBlock(
            hidden_size=32,
            moe_intermediate_size=8,
            n_routed_experts=4,
            moe_act_fn_cfg=MoEActFnConfig(),
        )
        assert isinstance(block.fused_w1w3, Int8W8A8DynamicGroupedLinear)
        assert isinstance(block.fused_w2, Int8W8A8DynamicGroupedLinear)
        assert block.fused_w1w3._quant_gmm is None
        assert block.fused_w2._quant_gmm is None

    def test_moe_block_allows_fused_a2a(self, monkeypatch):
        from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEActFnConfig, MoEBlock

        monkeypatch.setenv("XTUNER_QUANT_LINEAR", "w8a8_dynamic")
        monkeypatch.setenv("XTUNER_MOE_FUSED_A2A_GMM", "1")
        block = MoEBlock(
            hidden_size=32,
            moe_intermediate_size=8,
            n_routed_experts=4,
            moe_act_fn_cfg=MoEActFnConfig(),
        )
        # The fused a2a+gmm path no longer excludes quantization: both grouped linears stay
        # quantized and fused_forward picks their grids up through the hooks.
        assert isinstance(block.fused_w1w3, Int8W8A8DynamicGroupedLinear)
        assert isinstance(block.fused_w2, Int8W8A8DynamicGroupedLinear)

    def test_dense_mlp_builds_quantized_linears(self, monkeypatch):
        from xtuner.v1.module.decoder_layer.dense_decoder_layer import DenseMLP

        monkeypatch.setenv("XTUNER_QUANT_LINEAR", "w8a8_dynamic")
        mlp = DenseMLP(hidden_size=32, intermediate_size=8, hidden_act="silu")
        assert isinstance(mlp.gate_proj, Int8W8A8DynamicLinear)
        assert isinstance(mlp.up_proj, Int8W8A8DynamicLinear)
        assert isinstance(mlp.down_proj, Int8W8A8DynamicLinear)

    def test_env_off_builds_plain_modules(self, monkeypatch):
        from xtuner.v1.module.decoder_layer.dense_decoder_layer import DenseMLP
        from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEActFnConfig, MoEBlock

        monkeypatch.setenv("XTUNER_QUANT_LINEAR", "off")
        mlp = DenseMLP(hidden_size=32, intermediate_size=8, hidden_act="silu")
        assert not isinstance(mlp.gate_proj, Int8W8A8DynamicLinear)
        block = MoEBlock(
            hidden_size=32,
            moe_intermediate_size=8,
            n_routed_experts=4,
            moe_act_fn_cfg=MoEActFnConfig(),
        )
        assert type(block.fused_w1w3) is GroupedLinear
        assert type(block.fused_w2) is GroupedLinear


@pytest.mark.skipif(not _TORCH_NPU_AVAILABLE, reason="fused_a2a_gmm imports torch_npu at module level")
class TestFusedForwardQuantWiring:
    """fused_forward picks the quant grids up through the grouped-linears' hooks."""

    def _run_fused_forward(self, monkeypatch, quant_env):
        import xtuner.v1.ops.moe.npu.fused_a2a_gmm as fused_a2a_gmm
        from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEActFnConfig, MoEBlock

        monkeypatch.setenv("XTUNER_QUANT_LINEAR", quant_env)
        block = MoEBlock(
            hidden_size=32,
            moe_intermediate_size=8,
            n_routed_experts=4,
            moe_act_fn_cfg=MoEActFnConfig(),
        )
        torch.manual_seed(6)
        with torch.no_grad():  # GroupedLinear leaves weights uninitialized; avoid empty-memory NaNs
            for param in block.parameters():
                param.normal_()
        x = torch.randn(5, 32)
        ep_group = object()
        recorded = {}

        def fake_combine(**kwargs):
            recorded.update(kwargs)
            # A graph-connected stand-in: the combined output depends on both weight args.
            return kwargs["w1"].sum() + kwargs["w2"].sum() + torch.zeros(5, 32)

        monkeypatch.setattr(fused_a2a_gmm, "fused_dispatch_mlp_combine", fake_combine)
        monkeypatch.setattr(fused_a2a_gmm, "unpermute", lambda combined, row_id_map, probs: combined)
        # Exact for the all-zeros topk_ids below: a single expert group keeps token order,
        # so the sort-permute is a plain topk replication.
        monkeypatch.setattr(
            fused_a2a_gmm, "permute", lambda act, ids: (act.repeat_interleave(ids.shape[1], dim=0), None)
        )
        decoder_layer = SimpleNamespace(
            experts=block,
            dispatcher=SimpleNamespace(_process_group=ep_group),
            n_routed_experts=4,
            recv_tokens_hook=None,
            n_shared_experts=0,
            _post_moe_forward=lambda **kw: kw["combined_hidden_states"],
        )
        outputs = fused_a2a_gmm.fused_forward(
            decoder_layer,
            router_results={
                "topk_ids": torch.zeros(5, 2, dtype=torch.long),
                "topk_weights": torch.full((5, 2), 0.5),
                "logits": torch.randn(5, 4),
                "router_weights": torch.full((5, 2), 0.5),
            },
            pre_dispatched={
                "hidden_states": x,
                "row_id_map": torch.arange(10).view(5, 2),
            },
            shared_input_hidden=x,
            residual=x,
            origin_shape=torch.Size((5, 32)),
        )
        return block, x, ep_group, recorded, outputs

    def test_fused_forward_applies_quant_hooks(self, monkeypatch):
        block, x, ep_group, recorded, outputs = self._run_fused_forward(monkeypatch, "w8a8_dynamic")
        w1 = block.fused_w1w3
        w2 = block.fused_w2
        w1_view = w1.weight.view(-1, w1.local_out_features, w1.local_in_features)
        w2_view = w2.weight.view(-1, w2.local_out_features, w2.local_in_features)
        assert torch.equal(recorded["w1"], per_channel_weight_fake_quant(w1_view))
        assert torch.equal(recorded["w2"], per_channel_weight_fake_quant(w2_view))
        assert recorded["act_quant_fn"] == w1.fused_act_fake_quant
        # Lever-2 dedup: fused_forward quantizes the pre-permute hidden once and re-permutes
        # the quantized rows, so the combine receives a pre-quantized entry (its own FC1-input
        # quant is skipped; the FC2-input quant still runs via the same hook).
        assert recorded["entry_prequantized"] is True
        assert torch.equal(recorded["hidden_states"], int8_fake_quant_dynamic(x).repeat_interleave(2, dim=0))
        assert recorded["recv_tokens_hook"] is None
        assert recorded["ep_group"] is ep_group
        hidden = outputs[0]
        assert hidden.shape == (5, 32)
        # The hand-written backward's weight grad flows through the STE to the master weights.
        # The fake combine broadcasts each scalar weight sum across the [5, 32] output, so the
        # master grad is the output numel times the STE identity.
        hidden.sum().backward()
        assert torch.equal(w1.weight.grad, torch.full_like(w1.weight, 5 * 32))
        assert torch.equal(w2.weight.grad, torch.full_like(w2.weight, 5 * 32))

    def test_fused_forward_feature_off_is_byte_identical(self, monkeypatch):
        block, x, _, recorded, outputs = self._run_fused_forward(monkeypatch, "off")
        w1 = block.fused_w1w3
        w2 = block.fused_w2
        assert torch.equal(recorded["w1"], w1.weight.view(-1, w1.local_out_features, w1.local_in_features))
        assert torch.equal(recorded["w2"], w2.weight.view(-1, w2.local_out_features, w2.local_in_features))
        assert recorded["act_quant_fn"] is None
        # No dedup on the legacy path: the combine still receives the raw pre-permute rows.
        assert recorded["hidden_states"] is x
        assert recorded["entry_prequantized"] is False
        assert outputs[0].shape == (5, 32)


@pytest.mark.skipif(not _TORCH_NPU_AVAILABLE, reason="fused_a2a_gmm imports torch_npu at module level")
class TestFusedForwardSharedActQuantDedup:
    """fused_forward feeds the shared experts the pre-permute quantized hidden.

    Without a shared-expert sigmoid gate, the shared gate/up consume that already-quantized
    tensor and skip their internal per-token quant (scoped via ``_input_prequantized``);
    with a gate, the shared side falls back to the raw hidden and full internal quant while
    the routed-side dedup stays active.
    """

    def _run(self, monkeypatch, *, with_shared_expert_gate: bool):
        import xtuner.v1.ops.moe.npu.fused_a2a_gmm as fused_a2a_gmm
        from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEActFnConfig, MoEBlock, MoEMLP

        monkeypatch.setenv("XTUNER_QUANT_LINEAR", "w8a8_dynamic")
        block = MoEBlock(
            hidden_size=32,
            moe_intermediate_size=8,
            n_routed_experts=4,
            moe_act_fn_cfg=MoEActFnConfig(),
        )
        shared = MoEMLP(hidden_size=32, n_shared_experts=1, moe_intermediate_size=8, hidden_act="silu")
        torch.manual_seed(6)
        with torch.no_grad():  # linears leave weights uninitialized; avoid empty-memory NaNs
            for param in block.parameters():
                param.normal_()
            for param in shared.parameters():
                param.normal_()
        x = torch.randn(5, 2, 32)  # 3D, as the decoder layer hands it to fused_forward
        recorded = {}

        def fake_combine(**kwargs):
            recorded.update(kwargs)
            return torch.zeros(10, 32)

        monkeypatch.setattr(fused_a2a_gmm, "fused_dispatch_mlp_combine", fake_combine)
        monkeypatch.setattr(fused_a2a_gmm, "unpermute", lambda combined, row_id_map, probs: combined)
        # Exact for the all-zeros topk_ids below: a single expert group keeps token order,
        # so the sort-permute is a plain topk replication.
        monkeypatch.setattr(
            fused_a2a_gmm, "permute", lambda act, ids: (act.repeat_interleave(ids.shape[1], dim=0), None)
        )
        decoder_layer = SimpleNamespace(
            experts=block,
            dispatcher=SimpleNamespace(_process_group=object()),
            n_routed_experts=4,
            recv_tokens_hook=None,
            n_shared_experts=1,
            with_shared_expert_gate=with_shared_expert_gate,
            shared_experts=shared,
            _shared_experts_forward=lambda hidden_states: shared(hidden_states),
            # Surface the shared output so the tests can inspect what the shared side consumed.
            _post_moe_forward=lambda **kw: kw["shared_experts_out"],
        )
        outputs = fused_a2a_gmm.fused_forward(
            decoder_layer,
            router_results={
                "topk_ids": torch.zeros(5, 2, dtype=torch.long),
                "topk_weights": torch.full((5, 2), 0.5),
                "logits": torch.randn(5, 4),
                "router_weights": torch.full((5, 2), 0.5),
            },
            pre_dispatched={
                "hidden_states": x.view(10, 32),
                "row_id_map": torch.arange(10).view(5, 2),
            },
            shared_input_hidden=x,
            residual=x,
            origin_shape=torch.Size((5, 2, 32)),
        )
        return shared, x, recorded, outputs

    def test_shared_experts_consume_prequantized_hidden(self, monkeypatch):
        shared, x, recorded, outputs = self._run(monkeypatch, with_shared_expert_gate=False)
        assert isinstance(shared.gate_proj, Int8W8A8DynamicLinear)
        qx = int8_fake_quant_dynamic(x)
        # The combine receives the re-permuted pre-quantized rows (entry quant skipped).
        assert recorded["entry_prequantized"] is True
        assert torch.equal(recorded["hidden_states"], qx.view(10, 32).repeat_interleave(2, dim=0))
        # The shared MLP ran on the pre-quantized hidden: gate/up skipped their internal act
        # quant (identical values -- the STE node sits at qx), down_proj still quantized its
        # own input (the gated activation, never quantized upstream).
        w_g = per_channel_weight_fake_quant(shared.gate_proj.weight)
        w_u = per_channel_weight_fake_quant(shared.up_proj.weight)
        w_d = per_channel_weight_fake_quant(shared.down_proj.weight)
        inter = shared.act_fn(F.linear(qx, w_g), F.linear(qx, w_u))
        expected = F.linear(int8_fake_quant_dynamic(inter), w_d)
        assert torch.equal(outputs[0], expected)
        # The skip flag is scoped: reset after the fused forward on the two consumers, and
        # down_proj (which never reads it) is untouched.
        assert shared.gate_proj._input_prequantized is False
        assert shared.up_proj._input_prequantized is False
        assert shared.down_proj._input_prequantized is False
        # A direct call afterwards re-runs the full internal chain (no sticky skip): gate and
        # up re-quantize their input internally.
        g = F.linear(int8_fake_quant_dynamic(x), w_g)
        u = F.linear(int8_fake_quant_dynamic(x), w_u)
        direct = F.linear(int8_fake_quant_dynamic(shared.act_fn(g, u)), w_d)
        assert torch.equal(shared(x), direct)

    def test_shared_expert_gate_falls_back_to_raw_hidden(self, monkeypatch):
        shared, x, recorded, outputs = self._run(monkeypatch, with_shared_expert_gate=True)
        qx = int8_fake_quant_dynamic(x)
        # The routed-side dedup stays active...
        assert recorded["entry_prequantized"] is True
        assert torch.equal(recorded["hidden_states"], qx.view(10, 32).repeat_interleave(2, dim=0))
        # ...but the shared side consumed the RAW hidden (the unquantized sigmoid-gate linear
        # needs it), so the shared modules ran their full internal quant chain.
        w_g = per_channel_weight_fake_quant(shared.gate_proj.weight)
        w_u = per_channel_weight_fake_quant(shared.up_proj.weight)
        w_d = per_channel_weight_fake_quant(shared.down_proj.weight)
        g = F.linear(int8_fake_quant_dynamic(x), w_g)
        u = F.linear(int8_fake_quant_dynamic(x), w_u)
        expected = F.linear(int8_fake_quant_dynamic(shared.act_fn(g, u)), w_d)
        assert torch.equal(outputs[0], expected)
        # The prequant flag was never set on this path.
        assert shared.gate_proj._input_prequantized is False
        assert shared.up_proj._input_prequantized is False


class TestHfW8A8LinearExport:
    def test_full_tree_export(self, tmp_path):
        root, masters = _full_w8a8_tree()
        hf_dir = tmp_path / "hf"
        _write_full_fixture(hf_dir, masters)
        maybe_write_hf_w8a8_linear(root, hf_dir)

        tensors = _load_all(hf_dir)
        for weight_key, master in masters.items():
            q_expected, scale_expected = _training_grid_reference(master)
            module_key = weight_key.removesuffix(".weight")
            assert tensors[weight_key].dtype == torch.int8
            assert torch.equal(tensors[weight_key], q_expected)
            scale = tensors[f"{module_key}.weight_scale"]
            assert scale.dtype == torch.float32
            assert tuple(scale.shape) == (master.shape[0], 1)
            assert torch.equal(scale, scale_expected)
            assert torch.equal(tensors[f"{module_key}.weight_offset"], torch.zeros_like(scale_expected))

        # The bystander weight is untouched.
        assert tensors[_BYSTANDER_KEY].dtype == torch.bfloat16
        assert torch.equal(tensors[_BYSTANDER_KEY], _bystander())

        # The index keeps its metadata and now lists every artifact next to its weight.
        index = json.loads((hf_dir / _HF_INDEX_FILENAME).read_text())
        assert index["metadata"] == {"total_size": 12345}
        weight_map = index["weight_map"]
        assert len(weight_map) == 3 * len(masters) + 1
        for weight_key in masters:
            module_key = weight_key.removesuffix(".weight")
            assert weight_map[weight_key] in weight_map.values()
            for artifact in ("weight_scale", "weight_offset"):
                assert weight_map[f"{module_key}.{artifact}"] == weight_map[weight_key]

        # Every quantized module is marked W8A8_DYNAMIC by its bare (or experts.0) prefix;
        # every other exported weight is claimed FLOAT by its full key; the top-level
        # format metadata is filled in.
        description = json.loads((hf_dir / _DESCRIPTION_FILENAME).read_text())
        expected_desc_keys = {
            "model.language_model.layers.0.mlp.gate_proj",
            "model.language_model.layers.0.mlp.up_proj",
            "model.language_model.layers.0.mlp.down_proj",
            "model.language_model.layers.1.mlp.experts.0.gate_proj",
            "model.language_model.layers.1.mlp.experts.0.up_proj",
            "model.language_model.layers.1.mlp.experts.0.down_proj",
        }
        expected_float_keys = {_BYSTANDER_KEY} | {
            _expert_key(i, proj) for i in range(1, _E) for proj in ("gate", "up", "down")
        }
        assert set(description) == expected_desc_keys | expected_float_keys | {"version", "model_quant_type"}
        assert description["version"] == "1.0.0"
        assert description["model_quant_type"] == "W8A8_DYNAMIC"
        assert all(description[key] == "W8A8_DYNAMIC" for key in expected_desc_keys)
        assert all(description[key] == "FLOAT" for key in expected_float_keys)

    def test_description_merge_preserves_existing_entries(self, tmp_path):
        root, masters = _full_w8a8_tree()
        hf_dir = tmp_path / "hf"
        _write_full_fixture(hf_dir, masters)
        pre_existing = {
            "version": "9.9.9",
            "fa_quant_type": "FAKQuant",
            "model.language_model.layers.5.self_attn.fa_k.scale": "FAQuant",
        }
        (hf_dir / _DESCRIPTION_FILENAME).write_text(json.dumps(pre_existing))
        maybe_write_hf_w8a8_linear(root, hf_dir)
        description = json.loads((hf_dir / _DESCRIPTION_FILENAME).read_text())
        for key, value in pre_existing.items():
            assert description[key] == value
        assert description["model.language_model.layers.0.mlp.gate_proj"] == "W8A8_DYNAMIC"
        assert description["model.language_model.layers.1.mlp.experts.0.down_proj"] == "W8A8_DYNAMIC"
        # The merge added the plan entries, the FLOAT completion, and the format metadata;
        # nothing pre-existing was touched or clobbered.
        added = set(description) - set(pre_existing)
        assert added == {
            "model.language_model.layers.0.mlp.gate_proj",
            "model.language_model.layers.0.mlp.up_proj",
            "model.language_model.layers.0.mlp.down_proj",
            "model.language_model.layers.1.mlp.experts.0.gate_proj",
            "model.language_model.layers.1.mlp.experts.0.up_proj",
            "model.language_model.layers.1.mlp.experts.0.down_proj",
            "model_quant_type",
            _BYSTANDER_KEY,
        } | {_expert_key(i, proj) for i in range(1, _E) for proj in ("gate", "up", "down")}

    def test_float_completion_marks_unquantized_weights(self, tmp_path):
        # Regression: the deployment engine attributes every exported weight through the
        # description; weights outside the W8A8 plan (bystander linears, non-template
        # routed-expert slices) used to ship unclaimed and failed the consumer's load.
        root, masters = _full_w8a8_tree()
        hf_dir = tmp_path / "hf"
        _write_full_fixture(hf_dir, masters)
        maybe_write_hf_w8a8_linear(root, hf_dir)
        description = json.loads((hf_dir / _DESCRIPTION_FILENAME).read_text())
        assert description[_BYSTANDER_KEY] == "FLOAT"
        for i in range(1, _E):
            for proj in ("gate", "up", "down"):
                assert description[_expert_key(i, proj)] == "FLOAT"
        # Quantized modules stay claimed by their bare prefix only: the full ``.weight``
        # keys of W8A8 weights must not carry a contradicting FLOAT entry.
        assert _dense_key(0, "gate") not in description
        assert _expert_key(0, "gate") not in description

    def test_float_completion_preserves_faquant_entries(self, tmp_path):
        # C8+W8A8 combined export: the KV writer's FAQuant-marked scale entries are listed
        # in the index weight_map; the FLOAT completion must honor them, not re-claim them.
        root, masters = _full_w8a8_tree()
        hf_dir = tmp_path / "hf"
        _write_full_fixture(hf_dir, masters)
        fa_key = "model.language_model.layers.5.self_attn.fa_k.scale"
        shard = hf_dir / "model-00002-of-00002.safetensors"
        tensors = load_file(shard)
        tensors[fa_key] = torch.zeros(1, 1, dtype=torch.float32)
        save_file(tensors, shard)
        index = json.loads((hf_dir / _HF_INDEX_FILENAME).read_text())
        index["weight_map"][fa_key] = shard.name
        (hf_dir / _HF_INDEX_FILENAME).write_text(json.dumps(index, indent=4) + "\n")
        (hf_dir / _DESCRIPTION_FILENAME).write_text(json.dumps({"fa_quant_type": "FAKQuant", fa_key: "FAQuant"}))
        maybe_write_hf_w8a8_linear(root, hf_dir)
        description = json.loads((hf_dir / _DESCRIPTION_FILENAME).read_text())
        assert description[fa_key] == "FAQuant"

    def test_description_conflict_fails_fast(self, tmp_path):
        root, masters = _full_w8a8_tree()
        hf_dir = tmp_path / "hf"
        _write_full_fixture(hf_dir, masters)
        (hf_dir / _DESCRIPTION_FILENAME).write_text(
            json.dumps({"model.language_model.layers.0.mlp.gate_proj": "W4A16"})
        )
        with pytest.raises(ValueError, match="already maps"):
            maybe_write_hf_w8a8_linear(root, hf_dir)

    def test_missing_index(self, tmp_path):
        root, _ = _small_tree()
        hf_dir = tmp_path / "hf"
        hf_dir.mkdir()
        save_file(
            {_dense_key(0, "gate"): torch.randn(_ISP, _H, dtype=torch.bfloat16)},
            hf_dir / "model-00001-of-00001.safetensors",
        )
        with pytest.raises(FileNotFoundError, match="requires a sharded HF checkpoint index"):
            maybe_write_hf_w8a8_linear(root, hf_dir)

    def test_index_missing_weight(self, tmp_path):
        root, master = _small_tree()
        hf_dir = tmp_path / "hf"
        _write_shard_and_index(hf_dir, {_dense_key(0, "gate"): master}, index_omit=_dense_key(0, "gate"))
        with pytest.raises(KeyError, match="missing from the index"):
            maybe_write_hf_w8a8_linear(root, hf_dir)

    def test_shard_missing_weight(self, tmp_path):
        root, master = _small_tree()
        hf_dir = tmp_path / "hf"
        _write_shard_and_index(hf_dir, {_dense_key(0, "gate"): master}, shard_omit=_dense_key(0, "gate"))
        with pytest.raises(KeyError, match="in the index but not in shard"):
            maybe_write_hf_w8a8_linear(root, hf_dir)

    def test_non_2d_weight(self, tmp_path):
        root, _ = _small_tree()
        hf_dir = tmp_path / "hf"
        _write_shard_and_index(hf_dir, {_dense_key(0, "gate"): torch.randn(2, 3, 4)})
        with pytest.raises(ValueError, match="Expected a 2D linear weight"):
            maybe_write_hf_w8a8_linear(root, hf_dir)

    def test_refuses_to_overwrite_existing_artifact(self, tmp_path):
        root, master = _small_tree()
        hf_dir = tmp_path / "hf"
        module_key = _dense_key(0, "gate").removesuffix(".weight")
        tensors = {
            _dense_key(0, "gate"): master,
            f"{module_key}.weight_scale": torch.rand(_ISP, 1),
        }
        _write_shard_and_index(hf_dir, tensors)
        with pytest.raises(ValueError, match="refusing to overwrite"):
            maybe_write_hf_w8a8_linear(root, hf_dir)

    def test_dense_module_mapped_to_multiple_keys(self, tmp_path):
        root, _ = _small_tree()

        def _two_keys(key: str) -> list[str]:
            return [key, key + ".alias"]

        root.language_model.to_hf_key_list = _two_keys
        # The plan fails before anything touches disk.
        with pytest.raises(ValueError, match="expected exactly one"):
            maybe_write_hf_w8a8_linear(root, tmp_path / "hf")

    def test_noop_without_quant_modules(self, tmp_path):
        # Feature-off models never reach to_hf_key_list: the plan is empty and nothing is written.
        model = nn.Sequential(nn.Linear(_H, _ISP))
        maybe_write_hf_w8a8_linear(model, tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_export_grid_matches_training_kernel_on_npu(self, tmp_path):
        # W1 regression: with torch_npu present the export must quantize on the training
        # kernel grid, not the eager chain. The seed-0 fixture weights actually separate
        # the two grids (9/15 scales differ by 1 ulp fp32, expert 1's gate flips one
        # level), so kernel equality plus at least one eager mismatch pins the kernel
        # path; on a CPU-only box only the fallback contract (eager equality) is
        # assertable and the mismatch count stays unchecked.
        root, masters = _full_w8a8_tree()
        hf_dir = tmp_path / "hf"
        _write_full_fixture(hf_dir, masters)
        maybe_write_hf_w8a8_linear(root, hf_dir)
        tensors = _load_all(hf_dir)
        eager_mismatches = 0
        for weight_key, master in masters.items():
            module_key = weight_key.removesuffix(".weight")
            q_kernel, scale_kernel = _training_grid_reference(master)
            assert torch.equal(tensors[weight_key], q_kernel)
            assert torch.equal(tensors[f"{module_key}.weight_scale"], scale_kernel)
            q_eager, scale_eager = per_channel_weight_quant(master)
            if not torch.equal(tensors[weight_key], q_eager) or not torch.equal(
                tensors[f"{module_key}.weight_scale"], scale_eager
            ):
                eager_mismatches += 1
        npu = getattr(torch, "npu", None)
        if npu is not None and npu.is_available():
            assert eager_mismatches > 0

    def test_export_cpu_fallback_uses_eager_chain(self, tmp_path, monkeypatch):
        # The CPU-only export keeps the eager chain (bounded kernel-grid delta documented
        # in kernels.py); pin the fallback branch even on NPU boxes.
        monkeypatch.setattr(w8a8_export, "_npu_export_available", lambda: False)
        root, master = _small_tree()
        hf_dir = tmp_path / "hf"
        _write_shard_and_index(hf_dir, {_dense_key(0, "gate"): master})
        maybe_write_hf_w8a8_linear(root, hf_dir)
        tensors = _load_all(hf_dir)
        module_key = _dense_key(0, "gate").removesuffix(".weight")
        q_eager, scale_eager = per_channel_weight_quant(master)
        assert torch.equal(tensors[_dense_key(0, "gate")], q_eager)
        assert torch.equal(tensors[f"{module_key}.weight_scale"], scale_eager)


class TestW8A8ConsumerContract:
    def _real_export(self, tmp_path) -> dict[str, str]:
        root, masters = _full_w8a8_tree()
        hf_dir = tmp_path / "hf"
        _write_full_fixture(hf_dir, masters)
        maybe_write_hf_w8a8_linear(root, hf_dir)
        return json.loads((hf_dir / _DESCRIPTION_FILENAME).read_text())

    def test_lookup_from_real_export(self, tmp_path):
        description = self._real_export(tmp_path)
        cfg = AscendModelSlimLookupConfig(description)
        assert cfg.get_quant_type("glm5_next", "model.language_model.layers.0.mlp.gate_proj") == "W8A8_DYNAMIC"
        assert cfg.get_quant_type("glm5_next", "model.language_model.layers.0.mlp.down_proj") == "W8A8_DYNAMIC"
        # The routed-experts prefix resolves through the packed-module shard expansion.
        assert cfg.get_quant_type("glm5_next", "model.language_model.layers.1.mlp.experts") == "W8A8_DYNAMIC"
        # vllm's runtime namespace (language_model.model.*) remaps to the exported one.
        assert cfg.get_quant_type("glm5_next", "language_model.model.layers.1.mlp.experts") == "W8A8_DYNAMIC"
        assert cfg.packed_modules_mapping["experts"] == [
            "experts.0.gate_proj",
            "experts.0.up_proj",
            "experts.0.down_proj",
        ]

    def test_lookup_hand_dict(self):
        cfg = AscendModelSlimLookupConfig({"model.layers.5.mlp.fc1.weight": "FLOAT"})
        assert cfg.get_quant_type("llama", "model.layers.5.mlp.fc1") is None

        cfg = AscendModelSlimLookupConfig({"model.layers.5.mlp.fc1.weight": "W8A8_DYNAMIC"})
        assert cfg.get_quant_type("llama", "model.layers.5.mlp.fc2") is None

    def test_lookup_mixed_shard_types_raise(self):
        cfg = AscendModelSlimLookupConfig(
            {
                "model.layers.1.experts.0.gate_proj.weight": "W8A8_DYNAMIC",
                "model.layers.1.experts.0.up_proj.weight": "W8A8_DYNAMIC",
                "model.layers.1.experts.0.down_proj.weight": "W4A16",
            },
            packed_modules_mapping={"experts": ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"]},
        )
        with pytest.raises(ValueError, match="same quant type"):
            cfg.get_quant_type("llama", "model.layers.1.experts")

    def test_lookup_partial_shards_raise(self):
        cfg = AscendModelSlimLookupConfig(
            {
                "model.layers.1.experts.0.gate_proj.weight": "W8A8_DYNAMIC",
                "model.layers.1.experts.0.up_proj.weight": "W8A8_DYNAMIC",
            },
            packed_modules_mapping={"experts": ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"]},
        )
        with pytest.raises(KeyError, match="down_proj.weight"):
            cfg.get_quant_type("llama", "model.layers.1.experts")

    def test_lookup_fused_resolution(self):
        cfg = AscendModelSlimLookupConfig(
            {
                "model.layers.1.experts.gate_up_proj.weight": "W8A8_DYNAMIC",
                "model.layers.1.experts.down_proj.weight": "W8A8_DYNAMIC",
            },
            packed_modules_mapping={"experts": ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"]},
        )
        assert cfg.get_quant_type("llama", "model.layers.1.experts") == "W8A8_DYNAMIC"

    def test_lookup_auto_discovers_packed_mapping(self):
        cfg = AscendModelSlimLookupConfig(
            {
                "model.layers.1.experts.0.w1.weight": "W8A8_DYNAMIC",
                "model.layers.1.experts.0.w2.weight": "W8A8_DYNAMIC",
                "model.layers.1.experts.0.w3.weight": "W8A8_DYNAMIC",
            }
        )
        assert cfg.get_quant_type("mystery_model", "model.layers.1.experts") == "W8A8_DYNAMIC"
        assert cfg.packed_modules_mapping["experts"] == ["experts.0.w1", "experts.0.w2", "experts.0.w3"]

    def test_moe_scheme_contract(self):
        scheme = AscendW8A8DynamicFusedMoEMethod()
        assert not hasattr(scheme, "group_size")
        assert scheme.act_quant_type is torch.int8
        assert scheme.supports_eplb is True
        w = scheme.get_weight(
            num_experts=_E, intermediate_size_per_partition=_ISP, hidden_sizes=_H, params_dtype=torch.bfloat16
        )
        assert w["w13_weight"].dtype == torch.int8
        assert tuple(w["w13_weight"].shape) == (_E, 2 * _ISP, _H)
        assert w["w2_weight"].dtype == torch.int8
        assert tuple(w["w2_weight"].shape) == (_E, _H, _ISP)
        p = scheme.get_dynamic_quant_param(
            num_experts=_E, intermediate_size_per_partition=_ISP, hidden_sizes=_H, params_dtype=torch.float32
        )
        assert tuple(p["w13_weight_scale"].shape) == (_E, 2 * _ISP, 1)
        assert tuple(p["w2_weight_scale"].shape) == (_E, _H, 1)
        assert p["w13_weight_offset"].dtype == torch.float32

    def test_w13_reconstruction_matches_training_fake_quant(self, tmp_path):
        # End-to-end: the export artifacts, assembled the way vllm-ascend packs its w13
        # (gate over up per expert), reconstruct bitwise what training fake-quantized.
        root, masters = _full_w8a8_tree()
        hf_dir = tmp_path / "hf"
        _write_full_fixture(hf_dir, masters)
        maybe_write_hf_w8a8_linear(root, hf_dir)
        tensors = _load_all(hf_dir)

        w13_master = torch.stack(
            [torch.cat([masters[_expert_key(i, "gate")], masters[_expert_key(i, "up")]], dim=0) for i in range(_E)]
        )
        expected_w13 = _training_fake_quant_reference(w13_master)
        recon_w13 = torch.stack(
            [
                torch.cat([tensors[_expert_key(i, "gate")], tensors[_expert_key(i, "up")]], dim=0).float()
                * torch.cat(
                    [
                        tensors[f"{_expert_key(i, 'gate').removesuffix('.weight')}.weight_scale"],
                        tensors[f"{_expert_key(i, 'up').removesuffix('.weight')}.weight_scale"],
                    ],
                    dim=0,
                )
                for i in range(_E)
            ]
        ).to(torch.bfloat16)
        assert recon_w13.shape == (_E, 2 * _ISP, _H)
        assert torch.equal(recon_w13, expected_w13)

        w2_master = torch.stack([masters[_expert_key(i, "down")] for i in range(_E)])
        expected_w2 = _training_fake_quant_reference(w2_master)
        recon_w2 = torch.stack(
            [
                tensors[_expert_key(i, "down")].float()
                * tensors[f"{_expert_key(i, 'down').removesuffix('.weight')}.weight_scale"]
                for i in range(_E)
            ]
        ).to(torch.bfloat16)
        assert recon_w2.shape == (_E, _H, _ISP)
        assert torch.equal(recon_w2, expected_w2)

    def test_create_weights_routes_scales_to_channel_loader(self):
        harness = AscendMoEMethodHarness(AscendW8A8DynamicFusedMoEMethod(), moe_config=SimpleNamespace(tp_rank=0))
        layer = nn.Module()
        harness.create_weights(
            layer,
            num_experts=_E,
            hidden_size=_H,
            intermediate_size_per_partition=_ISP,
            params_dtype=torch.bfloat16,
        )
        # W8A8_DYNAMIC has no group_size: the scales keep the CHANNEL tag vllm's
        # RoutedExperts.weight_loader dispatches to _load_per_channel_weight_scale.
        assert not hasattr(layer.w13_weight, "quant_method")
        assert not hasattr(layer.w2_weight, "quant_method")
        for name in ("w13_weight_scale", "w13_weight_offset", "w2_weight_scale", "w2_weight_offset"):
            assert getattr(layer, name).quant_method == "channel"

    def _channel_param(self, shape: tuple[int, ...]) -> nn.Parameter:
        param = nn.Parameter(torch.zeros(shape))
        param.quant_method = "channel"
        return param

    def test_load_scale_per_channel_w1(self):
        loader = RoutedExpertsScaleLoaderHarness(moe_config=SimpleNamespace(tp_rank=0))
        param = self._channel_param((_E, 2 * _ISP, 1))
        loaded = torch.rand(2 * _ISP, 1)
        out = loader.load_scale(param, loaded, "model.experts.w1.weight_scale", "w1", expert_id=0)
        assert out is None
        assert len(loader.calls) == 1
        kind, kwargs = loader.calls[0]
        assert kind == "per_channel"
        assert kwargs["shard_id"] == "w1"
        assert kwargs["shard_dim"] == 0
        assert kwargs["loaded_weight"] is loaded
        # The full w13 scale row range is handed to the terminal loader, which narrows the
        # first half to gate and the second half to up.
        assert kwargs["expert_data"].shape == (2 * _ISP, 1)
        assert kwargs["expert_data"].data_ptr() == param.data[0].data_ptr()
        assert kwargs["tp_rank"] == 0

    def test_load_scale_per_channel_w2(self):
        loader = RoutedExpertsScaleLoaderHarness(moe_config=SimpleNamespace(tp_rank=0))
        param = self._channel_param((_E, _H, 1))
        loaded = torch.rand(_H, 1)
        loader.load_scale(param, loaded, "model.experts.w2.weight_scale", "w2", expert_id=0)
        kind, kwargs = loader.calls[0]
        assert kind == "per_channel"
        assert kwargs["shard_dim"] == 1
        assert kwargs["expert_data"].data_ptr() == param.data[0].data_ptr()

    def test_load_scale_transposed_flips_shard_dim(self):
        loader = RoutedExpertsScaleLoaderHarness(moe_config=SimpleNamespace(tp_rank=0))
        param = self._channel_param((_E, 2 * _ISP, 1))
        param.is_transposed = True
        loader.load_scale(param, torch.rand(2 * _ISP, 1), "model.experts.w1.weight_scale", "w1", expert_id=0)
        assert loader.calls[0][1]["shard_dim"] == 1

    def test_load_scale_full_load(self):
        loader = RoutedExpertsScaleLoaderHarness(moe_config=SimpleNamespace(tp_rank=0))
        param = self._channel_param((_E, 2 * _ISP, 1))
        loaded = torch.rand(_E, 2 * _ISP, 1)
        loader.load_scale(param, loaded, "model.experts.w13.weight_scale", "w1", expert_id=0)
        kind, kwargs = loader.calls[0]
        assert kind == "per_channel"
        assert kwargs["shard_dim"] == 1
        assert kwargs["expert_data"].data_ptr() == param.data.data_ptr()

    def test_load_scale_invalid_shard_id(self):
        loader = RoutedExpertsScaleLoaderHarness(moe_config=SimpleNamespace(tp_rank=0))
        param = self._channel_param((_E, 2 * _ISP, 1))
        with pytest.raises(ValueError, match="shard_id"):
            loader.load_scale(param, torch.rand(2 * _ISP, 1), "model.experts.w9.weight_scale", "w9", expert_id=0)

    def test_load_scale_missing_quant_method(self):
        loader = RoutedExpertsScaleLoaderHarness(moe_config=SimpleNamespace(tp_rank=0))
        param = nn.Parameter(torch.zeros(_E, 2 * _ISP, 1))
        with pytest.raises(ValueError, match="quant method must be one of"):
            loader.load_scale(param, torch.rand(2 * _ISP, 1), "model.experts.w1.weight_scale", "w1", expert_id=0)

    def test_load_scale_group_and_tensor_dispatch(self):
        loader = RoutedExpertsScaleLoaderHarness(moe_config=SimpleNamespace(tp_rank=0))
        param = nn.Parameter(torch.zeros(_E, 2 * _ISP, 1))
        param.quant_method = "group"
        loaded = torch.rand(2 * _ISP, 1)
        assert loader.load_scale(param, loaded, "model.experts.w1.weight_scale", "w1", expert_id=0) is None
        kind, kwargs = loader.calls[0]
        assert kind == "group"
        assert kwargs["shard_id"] == "w1"
        assert kwargs["shard_dim"] == 0
        assert kwargs["is_scale"] is True

        param = nn.Parameter(torch.zeros(_E, 2 * _ISP, 1))
        param.quant_method = "tensor"
        out = loader.load_scale(param, loaded, "model.experts.w1.weight_scale", "w1", expert_id=0, return_success=True)
        assert out is True
        kind, kwargs = loader.calls[1]
        assert kind == "per_tensor"
        assert set(kwargs) == {"shard_id", "param", "loaded_weight", "expert_id"}

    def test_load_scale_block_refine(self):
        refine_loader = RoutedExpertsScaleLoaderHarness(
            moe_config=SimpleNamespace(tp_rank=0), quant_method=SimpleNamespace(weight_scale_refine=(2, 2))
        )
        param = nn.Parameter(torch.zeros(_E, 2 * _ISP, 1))
        param.quant_method = "block"
        refine_loader.load_scale(param, torch.ones(1, 2, 2), "model.experts.w1.weight_scale", "w1", expert_id=0)
        kind, kwargs = refine_loader.calls[0]
        assert kind == "group"
        assert tuple(kwargs["loaded_weight"].shape) == (1, 4, 4)

        plain_loader = RoutedExpertsScaleLoaderHarness(
            moe_config=SimpleNamespace(tp_rank=0), quant_method=SimpleNamespace()
        )
        loaded = torch.ones(1, 2, 2)
        plain_loader.load_scale(param, loaded, "model.experts.w1.weight_scale", "w1", expert_id=0)
        kind, kwargs = plain_loader.calls[0]
        assert kind == "group"
        assert kwargs["loaded_weight"] is loaded

    def test_load_scale_non_scale_name(self):
        loader = RoutedExpertsScaleLoaderHarness(moe_config=SimpleNamespace(tp_rank=0))
        param = self._channel_param((_E, 2 * _ISP, 1))
        with pytest.raises(AssertionError):
            loader.load_scale(param, torch.rand(2 * _ISP, 1), "model.experts.w1.weight", "w1", expert_id=0)

    def test_linear_scheme_contract(self):
        scheme = AscendW8A8DynamicLinearMethod()
        assert scheme.act_quant_type is torch.int8
        w = scheme.get_weight(input_size=_H, output_size=_ISP, params_dtype=torch.bfloat16)
        assert w["weight"].dtype == torch.int8
        assert tuple(w["weight"].shape) == (_ISP, _H)
        p = scheme.get_perchannel_param(output_size=_ISP, params_dtype=torch.float32)
        assert tuple(p["weight_scale"].shape) == (_ISP, 1)
        assert p["weight_scale"].dtype == torch.float32
        assert tuple(p["weight_offset"].shape) == (_ISP, 1)

    def test_linear_scheme_process_weights(self):
        scheme = AscendW8A8DynamicLinearMethod()
        layer = SimpleNamespace(
            weight=nn.Parameter(torch.randint(-127, 128, (_ISP, _H), dtype=torch.int8), requires_grad=False),
            weight_scale=nn.Parameter(torch.rand(_ISP, 1)),
            weight_offset=nn.Parameter(torch.rand(_ISP, 1)),
        )
        scheme.process_weights_after_loading(layer)
        # The engine consumes the weight K-major ([in, out]) and flattened scales.
        assert tuple(layer.weight.data.shape) == (_H, _ISP)
        assert tuple(layer.weight_scale.data.shape) == (_ISP,)
        assert layer.weight_scale_fp32.dtype == torch.float32
        assert tuple(layer.weight_offset.data.shape) == (_ISP,)


def _ref_dynamic_quant(x: torch.Tensor) -> torch.Tensor:
    """Verbatim copy of the pre-fusion ``_Int8FakeQuant.forward`` dynamic chain."""
    x32 = x.float()
    scale = x32.abs().amax(dim=-1, keepdim=True).div_(127.0).clamp_min_(1e-12)
    q = torch.clamp(torch.round(x32 / scale), -127.0, 127.0)
    return q.mul(scale).to(x.dtype)


def _ref_static_quant(x: torch.Tensor, scale: torch.Tensor, clip: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Verbatim copy of the pre-fusion ``_Int8FakeQuant.forward`` static chain."""
    x32 = x.float()
    mask = x32.abs().le(127.0 * scale) if clip else None
    q = torch.clamp(torch.round(x32 / scale), -127.0, 127.0)
    return q.mul(scale).to(x.dtype), mask


def _ref_per_channel_weight_scale(w: torch.Tensor) -> torch.Tensor:
    """Verbatim copy of the pre-fusion per-channel weight scale chain."""
    return w.detach().abs().amax(dim=-1, keepdim=True).float().div(127.0)


class TestFakeQuantBitExactRegression:
    """The fused eager chain must reproduce the original chains bit-for-bit.

    The int8 fake-quant kernels were restructured from six full-size fp32 temporaries to
    one fused chain (``aminmax`` scale + in-place rounding). Any bit divergence would
    silently change trained weights, exported scales and the A/B loss parity, so this
    class pins the fused implementation to verbatim copies of the original chains.
    """

    @staticmethod
    def _assert_bitwise(actual: torch.Tensor, expected: torch.Tensor) -> None:
        nan_actual = torch.isnan(actual)
        assert torch.equal(nan_actual, torch.isnan(expected))
        if nan_actual.any():
            assert torch.equal(actual[~nan_actual], expected[~nan_actual])
        else:
            assert torch.equal(actual, expected)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("shape", [(7, 64), (64, 2048), (3, 5, 128)])
    def test_dynamic_bit_exact_random(self, shape: tuple, dtype: torch.dtype) -> None:
        torch.manual_seed(0)
        x = (torch.randn(shape) * 3).to(dtype)
        self._assert_bitwise(int8_fake_quant_dynamic(x), _ref_dynamic_quant(x))

    def test_dynamic_bit_exact_edge_values(self) -> None:
        huge = torch.finfo(torch.bfloat16).max
        fixtures = [
            torch.zeros(4, 32, dtype=torch.bfloat16),
            torch.full((4, 32), -0.0, dtype=torch.bfloat16),
            torch.tensor([[0.0, -0.0] * 16] * 4, dtype=torch.bfloat16),
            torch.full((4, 32), -2.5, dtype=torch.bfloat16),
            torch.full((4, 32), 3.5, dtype=torch.bfloat16),
            torch.full((1, 32), huge, dtype=torch.bfloat16),
            torch.full((1, 32), -huge, dtype=torch.bfloat16),
            (torch.randn(4, 32) * 1e-38).to(torch.bfloat16),
            torch.randn(1, 1, dtype=torch.bfloat16),
        ]
        for x in fixtures:
            self._assert_bitwise(int8_fake_quant_dynamic(x), _ref_dynamic_quant(x))

    def test_dynamic_bit_exact_inf_nan_rows(self) -> None:
        x = torch.randn(4, 32, dtype=torch.bfloat16)
        x[2, 7] = float("inf")
        x[3, 9] = float("nan")
        self._assert_bitwise(int8_fake_quant_dynamic(x), _ref_dynamic_quant(x))

    def test_static_bit_exact_live_clamp(self) -> None:
        # A static scale half the data's amax scale pushes levels to ~254, so the clamp is
        # live and the in-place ``clamp_`` must match the original out-of-place clamp.
        torch.manual_seed(1)
        x = (torch.randn(16, 128) * 5).to(torch.bfloat16)
        scale = x.abs().amax(dim=-1, keepdim=True).float().div(127.0).mul(0.5)
        self._assert_bitwise(int8_fake_quant_static(x, scale, clip=False), _ref_static_quant(x, scale, False)[0])

    def test_static_bit_exact_loose_scale(self) -> None:
        # A static scale larger than the data's amax scale leaves the clamp dead; the
        # per-channel broadcast shape mirrors the weight path's [E, out, 1] scale.
        torch.manual_seed(2)
        x = (torch.randn(4, 32, 128) * 0.02).to(torch.bfloat16)
        scale = _ref_per_channel_weight_scale(x).mul(4.0)
        self._assert_bitwise(int8_fake_quant_static(x, scale, clip=False), _ref_static_quant(x, scale, False)[0])

    def test_static_clip_backward_bit_exact(self) -> None:
        # The clip branch (the K1 static KV path) is untouched by the fusion; pin both its
        # forward and its masked straight-through backward.
        torch.manual_seed(3)
        x = (torch.randn(8, 64) * 8).to(torch.bfloat16).requires_grad_(True)
        scale = torch.rand(8, 1) * 0.02 + 1e-4
        out = int8_fake_quant_static(x, scale, clip=True)
        ref_out, mask = _ref_static_quant(x.detach(), scale, True)
        self._assert_bitwise(out, ref_out)
        grad = torch.randn_like(out)
        out.backward(grad)
        ref_grad = torch.where(mask, grad, torch.zeros((), dtype=grad.dtype))
        assert torch.equal(x.grad, ref_grad)

    def test_dynamic_ste_backward_passthrough(self) -> None:
        torch.manual_seed(4)
        x = torch.randn(8, 64, dtype=torch.bfloat16, requires_grad=True)
        out = int8_fake_quant_dynamic(x)
        grad = torch.randn_like(out)
        out.backward(grad)
        assert torch.equal(x.grad, grad)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("shape", [(16, 64), (4, 32, 128)])
    def test_per_channel_weight_scale_bit_exact(self, shape: tuple, dtype: torch.dtype) -> None:
        torch.manual_seed(5)
        w = (torch.randn(shape) * 0.02).to(dtype)
        assert torch.equal(_per_channel_weight_scale(w), _ref_per_channel_weight_scale(w))

    def test_per_channel_weight_scale_edge_values(self) -> None:
        for w in [
            torch.zeros(4, 32, dtype=torch.bfloat16),
            torch.full((4, 32), -0.0, dtype=torch.bfloat16),
            torch.full((4, 32), -0.5, dtype=torch.bfloat16),
        ]:
            assert torch.equal(_per_channel_weight_scale(w), _ref_per_channel_weight_scale(w))

    def test_weight_fake_quant_bit_exact(self) -> None:
        # End-to-end weight chain: the fused static path against the original scale +
        # static-quant chains composed by hand.
        torch.manual_seed(6)
        w = (torch.randn(4, 32, 128) * 0.02).to(torch.bfloat16)
        scale = _ref_per_channel_weight_scale(w)
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        ref, _ = _ref_static_quant(w, safe, False)
        self._assert_bitwise(per_channel_weight_fake_quant(w), ref)


def _npu_ready() -> bool:
    if not _TORCH_NPU_AVAILABLE:
        return False
    try:
        import torch_npu  # noqa: F401

        return torch.npu.device_count() > 0
    except (ImportError, RuntimeError):
        return False


@pytest.mark.gpu
@pytest.mark.skipif(not _npu_ready(), reason="requires an NPU device")
class TestNpuFusedKernels:
    """The NPU fused path vs the deployment kernel and the eager replica grid.

    On NPU the linear-quant kernels dispatch to ``torch.ops.npu.npu_dynamic_quant`` --
    the exact op vllm-ascend's W8A8_DYNAMIC path calls at serve time. These tests pin:
    the (levels, scale) pair to the op bitwise, the scale formula to the fp32 reciprocal
    multiply (``amax * (1 / 127)``, at most 1 ulp from the eager chain's IEEE division),
    the fake value to the export reconstruction (levels x scale in fp32), the zero-row
    and STE semantics, and the one-level knife-edge tolerance against the eager grid
    (the same bounded delta the #152 dynamic-block parity test documents: CANN's
    internal rounding flips ~0.03% of levels by exactly one).
    """

    @staticmethod
    def _x(shape: tuple[int, ...], seed: int = 0, scale: float = 2.0) -> torch.Tensor:
        torch.manual_seed(seed)
        return (torch.randn(shape) * scale).to(torch.bfloat16).to("npu")

    def test_act_quant_is_deployment_op(self) -> None:
        x = self._x((64, 256))
        q, s = per_token_act_quant(x)
        q_ref, s_ref = torch.ops.npu.npu_dynamic_quant(x, dst_type=torch.int8)
        assert q.dtype == torch.int8
        assert s.dtype == torch.float32 and s.shape == (64,)
        assert torch.equal(q, q_ref)
        assert torch.equal(s, s_ref)

    def test_act_scale_is_reciprocal_multiply(self) -> None:
        x = self._x((128, 512), seed=1)
        _, s = per_token_act_quant(x)
        mn, mx = torch.aminmax(x, dim=-1, keepdim=True)
        ref = torch.maximum(mx, mn.neg()).float().mul_(1.0 / 127.0).squeeze(-1)
        assert torch.equal(s, ref)

    def test_act_fake_quant_reconstructs_from_levels(self) -> None:
        # The STE forward must equal the export/replay reconstruction -- levels x scale
        # in fp32, cast back -- which is what a consumer multiplying the artifacts back
        # together computes.
        x = self._x((64, 512), seed=2)
        q, s = per_token_act_quant(x)
        recon = (q.float() * s.float().unsqueeze(-1)).to(x.dtype)
        assert torch.equal(per_token_act_fake_quant(x), recon)

    def test_act_fake_quant_within_one_level_of_eager_grid(self) -> None:
        x = self._x((256, 1024), seed=3)
        fused = per_token_act_fake_quant(x)
        eager = int8_fake_quant_dynamic(x)
        _, s = per_token_act_quant(x)
        # A flipped knife-edge level is one scale step; the bf16 cast of values near the
        # top of the range (127 * s, where one bf16 ulp is ~0.5 s) can add up to ~half a
        # step more, so 2 s bounds "at most one level" while still failing a two-level
        # divergence.
        delta = (fused.float() - eager.float()).abs()
        assert bool((delta <= s.unsqueeze(-1) * 2.0).all())

    def test_zero_rows_quantize_to_zero(self) -> None:
        x = self._x((8, 128), seed=4)
        x[2] = 0
        x[5] = 0
        q, s = per_token_act_quant(x)
        assert float(s[2]) == 0.0 and float(s[5]) == 0.0
        assert bool((q[2] == 0).all()) and bool((q[5] == 0).all())
        fake = per_token_act_fake_quant(x)
        assert bool((fake[2] == 0).all()) and bool((fake[5] == 0).all())
        assert not bool(torch.isnan(fake.float()).any())

    def test_weight_quant_per_out_channel_and_expert_isolation(self) -> None:
        wg = (torch.randn(4, 32, 128) * 0.02).to(torch.bfloat16).to("npu")
        q, s = per_channel_weight_quant(wg)
        assert q.shape == wg.shape and q.dtype == torch.int8
        assert s.shape == (4, 32, 1) and s.dtype == torch.float32
        # Per-row scales are row-local: quantizing one expert alone must reproduce its
        # slice of the grouped result bitwise.
        q3, s3 = per_channel_weight_quant(wg[3])
        assert torch.equal(q3, q[3])
        assert torch.equal(s3, s[3])

    def test_weight_fake_quant_matches_quant_reconstruction(self) -> None:
        w = (torch.randn(64, 128) * 0.02).to(torch.bfloat16).to("npu")
        w[3] = 0  # an all-zero output channel: scale 0, levels 0, fake value 0
        q, s = per_channel_weight_quant(w)
        assert float(s[3]) == 0.0 and bool((q[3] == 0).all())
        recon = (q.float() * s).to(w.dtype)
        fake = per_channel_weight_fake_quant(w)
        assert torch.equal(fake, recon)
        assert bool((fake[3] == 0).all())

    def test_ste_backward_passthrough(self) -> None:
        for fn in (per_token_act_fake_quant, per_channel_weight_fake_quant):
            x = self._x((8, 128), seed=7, scale=0.05).detach().requires_grad_(True)
            out = fn(x)
            grad = torch.randn_like(out)
            out.backward(grad)
            assert torch.equal(x.grad, grad)

    def test_determinism_repeat_call(self) -> None:
        x = self._x((128, 256), seed=8)
        assert torch.equal(per_token_act_fake_quant(x), per_token_act_fake_quant(x))
        q1, s1 = per_token_act_quant(x)
        q2, s2 = per_token_act_quant(x)
        assert torch.equal(q1, q2) and torch.equal(s1, s2)

    def test_3d_input_equals_2d_flatten(self) -> None:
        # fused_forward feeds the FC2 quant a leading-extended activation; the
        # row-flattened kernel must make the leading axes irrelevant.
        x2 = self._x((64, 256), seed=9)
        x3 = x2.view(4, 16, 256)
        fake2 = per_token_act_fake_quant(x2)
        fake3 = per_token_act_fake_quant(x3)
        assert torch.equal(fake3.reshape(64, 256), fake2)
        _, s3 = per_token_act_quant(x3)
        _, s2 = per_token_act_quant(x2)
        assert torch.equal(s3.reshape(-1), s2)

    def test_dequant_env_fp32_grid_is_bitwise_default(self, monkeypatch) -> None:
        # Unset and "fp32" must stay bitwise the deployment reconstruction: the switch
        # may never perturb the default grid (the arm-E baseline keeps its numbers).
        for raw in (None, "fp32"):
            if raw is None:
                monkeypatch.delenv("XTUNER_QUANT_LINEAR_DEQUANT", raising=False)
            else:
                monkeypatch.setenv("XTUNER_QUANT_LINEAR_DEQUANT", raw)
            x = self._x((64, 512), seed=10)
            q, s = per_token_act_quant(x)
            recon = (q * s.unsqueeze(-1)).to(x.dtype)
            assert torch.equal(per_token_act_fake_quant(x), recon)

    def test_dequant_env_native_grid_bitwise_own_dtype(self, monkeypatch) -> None:
        # "native" multiplies in the tensor's own dtype: int8 levels cast exactly, the
        # fp32 scale re-rounds once. Both entry points funnel through the same fused
        # Function, so one activation and one weight check pin the switch end to end.
        monkeypatch.setenv("XTUNER_QUANT_LINEAR_DEQUANT", "native")
        x = self._x((64, 512), seed=11)
        q, s = per_token_act_quant(x)
        native = q.to(x.dtype) * s.to(x.dtype).unsqueeze(-1)
        assert torch.equal(per_token_act_fake_quant(x), native)

        w = (torch.randn(32, 128) * 0.02).to(torch.bfloat16).to("npu")
        qw, sw = per_channel_weight_quant(w)
        assert torch.equal(per_channel_weight_fake_quant(w), qw.to(w.dtype) * sw.to(w.dtype))

    def test_dequant_env_native_bounded_delta_vs_fp32(self, monkeypatch) -> None:
        # The only perturbation is the scale's fp32->bf16 re-rounding -- RTNE, at most
        # half an ulp of the 8-significant-bit format (2^-8 relative) per row -- plus at
        # most one final-product ulp, so the two grids stay within ~2 scale steps
        # elementwise (a real grid bug such as a wrong broadcast axis overshoots by
        # orders of magnitude). Zero rows (scale 0) are exact zero on both grids.
        x = self._x((128, 512), seed=12)
        x[3] = 0
        q, s = per_token_act_quant(x)
        nz = s != 0
        s_rel = (s.to(x.dtype).float() - s).abs()[nz] / s.abs()[nz]
        assert bool((s_rel <= 2.0**-8).all())
        monkeypatch.setenv("XTUNER_QUANT_LINEAR_DEQUANT", "native")
        nat = per_token_act_fake_quant(x)
        monkeypatch.setenv("XTUNER_QUANT_LINEAR_DEQUANT", "fp32")
        ref = per_token_act_fake_quant(x)
        assert bool(torch.isfinite(nat.float()).all())
        assert bool((nat[3] == 0).all()) and bool((ref[3] == 0).all())
        delta = (nat.float() - ref.float()).abs()
        assert bool((delta <= s.unsqueeze(-1) * 2.0).all())

    def test_dequant_env_native_ste_backward_passthrough(self, monkeypatch) -> None:
        monkeypatch.setenv("XTUNER_QUANT_LINEAR_DEQUANT", "native")
        for fn in (per_token_act_fake_quant, per_channel_weight_fake_quant):
            x = self._x((8, 128), seed=13, scale=0.05).detach().requires_grad_(True)
            out = fn(x)
            grad = torch.randn_like(out)
            out.backward(grad)
            assert torch.equal(x.grad, grad)

    def test_dequant_env_ignored_by_eager_path(self, monkeypatch) -> None:
        # The switch selects the NPU fused path's grid only; fp32 input keeps the eager
        # replica (the deployment reconstruction off NPU) regardless of the env value.
        x32 = self._x((32, 128), seed=14).float()
        eager_ref = per_token_act_fake_quant(x32)
        for raw in ("native", "fp32"):
            monkeypatch.setenv("XTUNER_QUANT_LINEAR_DEQUANT", raw)
            assert torch.equal(per_token_act_fake_quant(x32), eager_ref)
