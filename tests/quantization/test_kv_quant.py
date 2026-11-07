"""Tests for the KV-cache quantization config, int8 fake-quant kernels and static-mode calibration (CPU-only)."""

import json

import pytest
import torch
from pydantic import ValidationError
from safetensors.torch import load_file
from torch import nn

from xtuner.v1.quantization.kv import (
    STATIC_KV_QUANT_MODES,
    KVQuantConfig,
    KVScaleLazyCalibrator,
    int8_fake_quant_dynamic,
    int8_fake_quant_dynamic_block,
    int8_fake_quant_static,
    kv_fakequant_enabled,
    kv_quant_config_from_env,
    maybe_load_kv_scale_sidecar,
    maybe_save_kv_scale_sidecar,
    maybe_write_hf_kv_scales,
    needs_kv_calibration,
)
from xtuner.v1.quantization.kv.export import _hf_scale_key
from xtuner.v1.quantization.kv.kernels import _DYNAMIC_BLOCK_TILE

# Relative import: a regular "tests" package in site-packages shadows the repo's
# tests/ namespace, so absolute "tests.quantization..." can never resolve.
from ._modelslim_consumer_ref import (
    AscendFAQuantAttentionMethod,
    AscendModelSlimConfig,
    _fa_quant_weight_loader,
    detect_quantization_method,
)
from ._sfa_c8_dynamic_block_ref import SFA_QSFA_TILE_SIZE, custom_kv_rmsnorm_rope_quant_int8


class TestInt8FakeQuantDynamic:
    def test_per_token_roundtrip_on_grid(self):
        # Grid-aligned inputs (integer multiples of the per-token scale) must come back exact.
        # max |level| = 127 so the derived scale is exactly 0.25 (power of two: fp32-exact).
        levels = torch.tensor([-127.0, -64.0, 0.0, 32.0, 100.0, 127.0])
        x = (levels * 0.25).repeat(2, 1)  # [2, 6]
        out = int8_fake_quant_dynamic(x)
        scale = x.abs().amax(dim=-1, keepdim=True) / 127.0
        assert out.shape == x.shape
        assert out.dtype == x.dtype
        # Every output sits on the int8 grid: out / scale is an integer.
        assert torch.equal(out.float() / scale, (out.float() / scale).round())
        # Grid-aligned inputs land on their own level: bitwise identity.
        assert torch.equal(out, x)

    def test_max_error_bounded_by_half_scale(self):
        torch.manual_seed(0)
        x = torch.randn(3, 16, 32)
        out = int8_fake_quant_dynamic(x)
        scale = x.abs().amax(dim=-1, keepdim=True) / 127.0
        assert torch.all((out.float() - x).abs() <= 0.5 * scale + 1e-6)

    def test_zero_row_no_nan(self):
        x = torch.zeros(2, 5, 8)
        x[1, :, :] = 1.0
        out = int8_fake_quant_dynamic(x)
        assert torch.isfinite(out).all()
        assert torch.equal(out[0], torch.zeros_like(out[0]))

    def test_ste_backward_passthrough_bitwise(self):
        x = torch.randn(2, 7, 16, requires_grad=True)
        out = int8_fake_quant_dynamic(x)
        grad = torch.randn_like(out)
        out.backward(grad)
        assert x.grad is not None
        # Straight-through estimator: the gradient bypasses the rounding untouched.
        assert torch.equal(x.grad, grad)


class TestInt8FakeQuantStatic:
    def test_clamp_out_of_range(self):
        scale = torch.full((4,), 0.125)
        x = torch.tensor([[100.0, -100.0, 12.0, -12.0], [15.9, -15.9, 0.0, 1.0]])
        out = int8_fake_quant_static(x, scale)
        # 100 / 0.125 = 800 -> clamps to 127 * scale = 15.875; 15.9 / 0.125 = 127.2 rounds to
        # 127 and lands on the same clamped level; in-range grid points come back exact.
        expected = torch.tensor([[15.875, -15.875, 12.0, -12.0], [15.875, -15.875, 0.0, 1.0]])
        assert torch.equal(out, expected)

    def test_clip_zeroes_out_of_range_gradients(self):
        scale = torch.full((3,), 1.0)
        x = torch.tensor([[2.0, 200.0, -200.0]], requires_grad=True)
        out = int8_fake_quant_static(x, scale, clip=True)
        assert torch.equal(out, torch.tensor([[2.0, 127.0, -127.0]]))
        out.backward(torch.ones_like(out))
        assert x.grad is not None
        # In-range entries pass through; out-of-range entries are zeroed by the clip mask.
        assert torch.equal(x.grad, torch.tensor([[1.0, 0.0, 0.0]]))

    def test_no_clip_passes_all_gradients(self):
        scale = torch.full((3,), 1.0)
        x = torch.tensor([[0.5, 100.0, -200.0]], requires_grad=True)
        out = int8_fake_quant_static(x, scale, clip=False)
        out.backward(torch.ones_like(out))
        assert x.grad is not None
        assert torch.equal(x.grad, torch.ones_like(x))

    def test_per_tensor_scalar_scale(self):
        # per_tensor_static freezes a 0-dim scale: it must broadcast against any input shape
        # and satisfy the same grid / error / STE contracts as the per-channel path.
        torch.manual_seed(0)
        x = torch.randn(2, 7, 16, requires_grad=True)
        scale = torch.tensor(0.03125)  # 2^-5: power of two keeps on-grid values fp32-exact
        out = int8_fake_quant_static(x, scale)
        levels = out.float() / scale
        assert torch.all((levels - levels.round()).abs() <= 1e-6)  # every element on the grid
        assert torch.all((out.float() - x.detach()).abs() <= 0.5 * scale + 1e-6)  # error bound
        out.backward(torch.ones_like(out))
        assert x.grad is not None
        assert torch.equal(x.grad, torch.ones_like(x))  # STE passthrough unchanged

    def test_scale_must_be_float32(self):
        # fp32 keeps the quantization math aligned with the deployment kernel; anything else
        # must fail at the entry instead of drifting silently.
        x = torch.randn(2, 5)
        with pytest.raises(ValueError, match="float32"):
            int8_fake_quant_static(x, torch.full((5,), 0.5, dtype=torch.bfloat16))

    def test_scale_must_live_on_x_device(self):
        x = torch.randn(2, 5)
        meta_scale = torch.full((5,), 0.5, device="meta")
        with pytest.raises(ValueError, match="device"):
            int8_fake_quant_static(x, meta_scale)

    def test_scale_shape_must_broadcast(self):
        x = torch.randn(4, 5)
        with pytest.raises(ValueError, match="broadcast"):
            int8_fake_quant_static(x, torch.full((7,), 0.5))


class TestInt8FakeQuantDynamicBlock:
    # Grid contract of the deployment kernel (torch_npu.npu_dynamic_block_quant, probed
    # on-device): one fp32 scale = amax * (1/127) per (1, 128) token tile, int8 levels in
    # [-127, 127], dequant y * scale, all-zero tile -> scale 0 / levels 0.

    def test_tile_scales_differ_from_per_token(self):
        # Tile 0 carries the row amax (1.0), tile 1 half of it: the block kernel must place
        # tile 1's 0.5 anchor on its own 127-level (scale 0.5/127), while a whole-token scale
        # (1/127) pushes the same entry to the 63-level -- different dequantized values.
        x = torch.zeros(1, 1, 2 * _DYNAMIC_BLOCK_TILE)
        x[0, 0, 0] = 1.0
        x[0, 0, _DYNAMIC_BLOCK_TILE] = 0.5
        out = int8_fake_quant_dynamic_block(x)
        scale_tile1 = 0.5 * (1.0 / 127.0)
        anchor = out[0, 0, _DYNAMIC_BLOCK_TILE].item()
        assert anchor == pytest.approx(127 * scale_tile1, abs=1e-6)
        assert (out.float() - x).abs().max().item() <= 0.5 * scale_tile1 + 1e-6
        per_token = int8_fake_quant_dynamic(x)
        assert per_token[0, 0, _DYNAMIC_BLOCK_TILE].item() != anchor

    def test_matches_torch_mirror_bitwise(self):
        # The kernel must be exactly its documented formula: fp32 amax * (1/127) per tile,
        # round-half-even of the safe division, clamp to [-127, 127], dequant y * scale.
        torch.manual_seed(1)
        x = (torch.randn(4, 6, 4 * _DYNAMIC_BLOCK_TILE) * 0.05).bfloat16()
        out = int8_fake_quant_dynamic_block(x)
        x32 = x.float()
        tiles = x32.reshape(*x32.shape[:-1], -1, _DYNAMIC_BLOCK_TILE)
        scale = tiles.abs().amax(dim=-1, keepdim=True).mul_(1.0 / 127.0)
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        q = torch.clamp(torch.round(tiles / safe), -127.0, 127.0)
        assert torch.equal(out, q.mul(scale).reshape(x32.shape).to(x.dtype))

    def test_grid_and_tight_error(self):
        # fp32 input: the dequant contract q * scale is an fp32 product; a bf16 output cast
        # adds one bf16 ulp on top of scale/2 (the NPU parity test budgets for that).
        torch.manual_seed(2)
        x = torch.randn(3, 5, 512) * 0.05
        out = int8_fake_quant_dynamic_block(x)
        tiles = x.float().reshape(3, 5, 4, 128)
        scale = tiles.abs().amax(dim=-1, keepdim=True).mul_(1.0 / 127.0)
        err = (out.float() - x.float()).reshape(3, 5, 4, 128)
        assert torch.all(err.abs() <= 0.5 * scale + 1e-6)
        levels = out.float().reshape(3, 5, 4, 128) / scale
        assert torch.all((levels - levels.round()).abs() <= 1e-4)

    def test_zero_tile_no_nan(self):
        x = torch.zeros(2, 1, 512)
        x[1, 0, 0] = 1.0
        out = int8_fake_quant_dynamic_block(x)
        assert torch.isfinite(out).all()
        assert torch.equal(out[0], torch.zeros_like(out[0]))

    def test_ste_backward_passthrough_bitwise(self):
        x = torch.randn(2, 3, 512, requires_grad=True)
        out = int8_fake_quant_dynamic_block(x)
        grad = torch.randn_like(out)
        out.backward(grad)
        assert x.grad is not None
        assert torch.equal(x.grad, grad)

    def test_last_dim_must_be_tile_multiple(self):
        with pytest.raises(ValueError, match="multiple"):
            int8_fake_quant_dynamic_block(torch.randn(2, 100))


class TestDynamicBlock128Config:
    def test_env_parses_dynamic_block_mode(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("XTUNER_QUANT_KV", "dynamic_block_1x128")
        cfg = kv_quant_config_from_env()
        assert cfg is not None and cfg.kv_quant == "dynamic_block_1x128"
        assert kv_fakequant_enabled(cfg, "nope_dsa_ckv")

    def test_env_calib_batches_rejected_for_dynamic_block(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("XTUNER_QUANT_KV", "dynamic_block_1x128")
        monkeypatch.setenv("XTUNER_QUANT_KV_CALIB_BATCHES", "8")
        with pytest.raises(ValueError, match="XTUNER_QUANT_KV_CALIB_BATCHES"):
            kv_quant_config_from_env()

    def test_dynamic_block_is_outside_static_machinery(self):
        # The dynamic-block mode needs no calibration, sidecar or export (the deployment
        # kernel derives its scales per forward): it must stay out of STATIC_KV_QUANT_MODES
        # so the static-only machinery never collects or persists anything for it.
        assert "dynamic_block_1x128" not in STATIC_KV_QUANT_MODES
        model = _FakeModel(["dynamic_block_1x128"], calib_batches=2)
        assert not needs_kv_calibration(model)
        calibrator = KVScaleLazyCalibrator(model)
        assert not calibrator.active
        calibrator.on_train_step_end()  # no-op, no hooks installed
        assert model.attns[0]._kv_scale is None

    def test_fake_attn_dispatches_dynamic_block(self):
        torch.manual_seed(3)
        x = (torch.randn(1, 8, 512) * 0.05).bfloat16()
        model = _FakeModel(["dynamic_block_1x128"], channels=512)
        out = model(seq_ctx=x)
        assert torch.equal(out, int8_fake_quant_dynamic_block(model.attns[0].kv_a_layernorm(x)))


class TestKVQuantConfig:
    def test_default_is_off(self):
        cfg = KVQuantConfig()
        assert cfg.kv_quant == "off"
        assert not kv_fakequant_enabled(cfg, "nope_dsa_ckv")
        assert not kv_fakequant_enabled(None, "nope_dsa_ckv")

    def test_gate_respects_scope(self):
        cfg = KVQuantConfig(kv_quant="per_token_dynamic")
        assert kv_fakequant_enabled(cfg, "nope_dsa_ckv")
        assert not kv_fakequant_enabled(cfg, "some_other_face")

    def test_scope_rejects_unknown_face(self):
        # kv_quant_scope is a closed Literal set: a typo'd face name must fail loudly at
        # config time instead of silently gating the fake-quant off.
        with pytest.raises(ValidationError):
            KVQuantConfig(kv_quant="per_token_dynamic", kv_quant_scope=["nope_dsa_ckv "])
        with pytest.raises(ValidationError):
            KVQuantConfig(kv_quant_scope=["some_other_face"])

    def test_env_parsing(self, monkeypatch: pytest.MonkeyPatch):
        for off_value in (None, "", "off", "0", "false"):
            if off_value is None:
                monkeypatch.delenv("XTUNER_QUANT_KV", raising=False)
            else:
                monkeypatch.setenv("XTUNER_QUANT_KV", off_value)
            assert kv_quant_config_from_env() is None
        monkeypatch.setenv("XTUNER_QUANT_KV", "per_token_dynamic")
        cfg = kv_quant_config_from_env()
        assert cfg is not None and cfg.kv_quant == "per_token_dynamic"

    def test_env_parses_static_modes(self, monkeypatch: pytest.MonkeyPatch):
        for mode in ("per_tensor_static",):
            monkeypatch.setenv("XTUNER_QUANT_KV", mode)
            cfg = kv_quant_config_from_env()
            assert cfg is not None and cfg.kv_quant == mode
            # Static modes default to 64 calibration steps unless overridden.
            assert cfg.kv_calib_num_batches == 64

    def test_env_calib_batches_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("XTUNER_QUANT_KV", "per_tensor_static")
        monkeypatch.setenv("XTUNER_QUANT_KV_CALIB_BATCHES", "8")
        cfg = kv_quant_config_from_env()
        assert cfg is not None and cfg.kv_calib_num_batches == 8
        # The override only touches the static modes; setting it under dynamic is an error
        # instead of a silent ignore.
        monkeypatch.setenv("XTUNER_QUANT_KV", "per_token_dynamic")
        with pytest.raises(ValueError, match="XTUNER_QUANT_KV_CALIB_BATCHES"):
            kv_quant_config_from_env()

    def test_env_calib_batches_override_invalid(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("XTUNER_QUANT_KV", "per_tensor_static")
        for bad in ("0", "-3", "abc"):
            monkeypatch.setenv("XTUNER_QUANT_KV_CALIB_BATCHES", bad)
            with pytest.raises(ValueError, match="XTUNER_QUANT_KV_CALIB_BATCHES"):
                kv_quant_config_from_env()

    def test_calib_batches_must_be_positive(self):
        with pytest.raises(ValidationError):
            KVQuantConfig(kv_quant="per_tensor_static", kv_calib_num_batches=0)

    def test_env_invalid_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("XTUNER_QUANT_KV", "int4")
        with pytest.raises(ValueError, match="XTUNER_QUANT_KV"):
            kv_quant_config_from_env()


class _FakeAttn(nn.Module):
    # Mirrors the NoPEDSAMultiLatentAttention static-KV contract: the kv_a_layernorm output is
    # the quantization face, and the scale lives in plain attributes filled by the calibrator /
    # checkpoint sidecar (never a registered buffer).
    def __init__(self, mode: str, channels: int = 8, calib_batches: int = 2):
        super().__init__()
        self.kv_a_layernorm = nn.LayerNorm(channels)
        self._kv_quant_mode = mode
        self._kv_scale: torch.Tensor | None = None
        self._kv_scale_ready = False
        self._kv_calibrating = False
        self.kv_calib_num_batches = calib_batches

    def forward(self, seq_ctx: torch.Tensor) -> torch.Tensor:
        kv = self.kv_a_layernorm(seq_ctx)
        if self._kv_quant_mode == "per_token_dynamic":
            kv = int8_fake_quant_dynamic(kv)
        elif self._kv_quant_mode == "dynamic_block_1x128":
            kv = int8_fake_quant_dynamic_block(kv)
        elif self._kv_calibrating:
            pass
        elif not self._kv_scale_ready:
            raise ValueError("frozen KV scale required")
        else:
            assert self._kv_scale is not None
            kv = int8_fake_quant_static(kv, self._kv_scale)
        return kv


class _FakeModel(nn.Module):
    def __init__(self, modes: list[str], channels: int = 8, calib_batches: int = 2):
        super().__init__()
        self.attns = nn.ModuleList([_FakeAttn(m, channels, calib_batches) for m in modes])

    def forward(self, seq_ctx: torch.Tensor, loss_ctx: dict | None = None) -> torch.Tensor:
        for attn in self.attns:
            seq_ctx = attn(seq_ctx)
        return seq_ctx


def _expected_scale(attn: _FakeAttn, batches: list[torch.Tensor]) -> torch.Tensor:
    amaxes = [attn.kv_a_layernorm(batch).detach().float().abs().amax() for batch in batches]
    return torch.stack(amaxes).amax(dim=0).div(127.0).clamp_min(1e-12)


class TestKvScaleCalibration:
    def test_per_tensor_calibration_freezes_scalar(self):
        torch.manual_seed(0)
        model = _FakeModel(["per_tensor_static"], calib_batches=2)
        batches = [torch.randn(1, 12, 8) for _ in range(2)]

        calibrator = KVScaleLazyCalibrator(model)
        for batch in batches:
            model(seq_ctx=batch)
            calibrator.on_train_step_end()

        attn = model.attns[0]
        scale = attn._kv_scale
        assert scale is not None and scale.ndim == 0
        assert torch.allclose(scale, _expected_scale(attn, batches))

    def test_calibration_freezes_only_after_threshold(self):
        # Regression for run38/run40's init-time pass: calibration must ride the training
        # loop's own forwards, stay in passthrough until kv_calib_num_batches steps are
        # collected, and freeze exactly at the threshold -- without touching any dataloader.
        torch.manual_seed(0)
        model = _FakeModel(["per_tensor_static"], calib_batches=2)
        calibrator = KVScaleLazyCalibrator(model)
        assert calibrator.active

        model(seq_ctx=torch.randn(1, 8, 8))
        calibrator.on_train_step_end()
        assert calibrator.active
        assert not model.attns[0]._kv_scale_ready
        assert model.attns[0]._kv_calibrating  # still collecting: passthrough

        model(seq_ctx=torch.randn(1, 8, 8))
        calibrator.on_train_step_end()
        assert model.attns[0]._kv_scale_ready
        assert not model.attns[0]._kv_calibrating
        assert not calibrator.active
        calibrator.on_train_step_end()  # past the threshold: cheap no-op
        assert model.attns[0]._kv_scale_ready

    def test_calibration_passes_raw_values_through(self):
        # While calibrating, the hook must see the unquantized LayerNorm output -- the frozen
        # scale has to cover the exact values the deployment kernel will quantize.
        torch.manual_seed(0)
        model = _FakeModel(["per_tensor_static"], calib_batches=1)
        batches = [torch.randn(1, 16, 8)]
        seen: list[torch.Tensor] = []
        handle = model.attns[0].kv_a_layernorm.register_forward_hook(
            lambda _m, _a, output: seen.append(output.detach().clone())
        )
        try:
            calibrator = KVScaleLazyCalibrator(model)
            model(seq_ctx=batches[0])
            calibrator.on_train_step_end()
        finally:
            handle.remove()
        assert torch.equal(seen[0], model.attns[0].kv_a_layernorm(batches[0]).detach())

    def test_calibration_noop_when_all_ready(self):
        model = _FakeModel(["per_tensor_static"] * 2)
        for attn in model.attns:
            attn._kv_scale = torch.tensor(0.5)
            attn._kv_scale_ready = True

        calibrator = KVScaleLazyCalibrator(model)
        assert not calibrator.active
        model(seq_ctx=torch.randn(1, 8, 8))
        calibrator.on_train_step_end()
        for attn in model.attns:
            assert torch.equal(attn._kv_scale, torch.tensor(0.5))

    def test_calibration_noop_without_static_modules(self):
        model = _FakeModel(["per_token_dynamic"])
        calibrator = KVScaleLazyCalibrator(model)
        assert not calibrator.active
        model(seq_ctx=torch.randn(1, 8, 8))
        calibrator.on_train_step_end()

    def test_calibration_inconsistent_num_batches_raises(self):
        model = _FakeModel(["per_tensor_static"] * 2)
        model.attns[1].kv_calib_num_batches = 5
        with pytest.raises(ValueError, match="Inconsistent kv_calib_num_batches"):
            KVScaleLazyCalibrator(model)
        # Construction failed before any hook was installed: no module may be left collecting.
        assert not model.attns[0]._kv_calibrating
        assert not model.attns[1]._kv_calibrating

    def test_per_tensor_full_loop_freeze_then_quantized_forward(self):
        # per_tensor_static full loop at unit level: collect on training steps, freeze the
        # 0-dim scale, then the post-freeze forward must quantize through the frozen scalar
        # grid -- downstream attns now see the upstream attn's QUANTIZED output.
        torch.manual_seed(0)
        model = _FakeModel(["per_tensor_static"] * 2, calib_batches=1)
        batch = torch.randn(1, 16, 8)
        calibrator = KVScaleLazyCalibrator(model)
        model(seq_ctx=batch)
        calibrator.on_train_step_end()

        scale0 = model.attns[0]._kv_scale
        scale1 = model.attns[1]._kv_scale
        assert scale0 is not None and scale0.ndim == 0
        assert scale1 is not None and scale1.ndim == 0
        out = model(seq_ctx=batch)
        expected = int8_fake_quant_static(model.attns[0].kv_a_layernorm(batch), scale0)
        expected = int8_fake_quant_static(model.attns[1].kv_a_layernorm(expected), scale1)
        assert torch.equal(out, expected)

    def test_sidecar_roundtrip(self, tmp_path):
        torch.manual_seed(0)
        model = _FakeModel(["per_tensor_static"] * 2, calib_batches=1)
        calibrator = KVScaleLazyCalibrator(model)
        model(seq_ctx=torch.randn(1, 16, 8))
        calibrator.on_train_step_end()
        saved = {name: attn._kv_scale.clone() for name, attn in zip(("attns.0", "attns.1"), model.attns)}

        maybe_save_kv_scale_sidecar(model, tmp_path)
        assert (tmp_path / "kv_scale_sidecar.pt").exists()

        # A fresh, differently-seeded model with the same tree picks the scales up by name.
        torch.manual_seed(7)
        fresh = _FakeModel(["per_tensor_static"] * 2)
        assert needs_kv_calibration(fresh)
        maybe_load_kv_scale_sidecar(fresh, tmp_path)
        assert not needs_kv_calibration(fresh)
        for name, attn in zip(("attns.0", "attns.1"), fresh.attns):
            assert attn._kv_scale_ready and torch.equal(attn._kv_scale, saved[name])

    def test_sidecar_shape_mismatch_raises(self, tmp_path):
        # A [C] vector would silently broadcast where the frozen grid needs a scalar: the load
        # must fail fast instead of quantizing on the wrong grid.
        torch.save({"attns.0": torch.full((8,), 0.5)}, tmp_path / "kv_scale_sidecar.pt")
        model = _FakeModel(["per_tensor_static"])
        with pytest.raises(ValueError, match="different kv_quant mode"):
            maybe_load_kv_scale_sidecar(model, tmp_path)

    def test_sidecar_save_skips_partial_ready(self, tmp_path):
        # A checkpoint fired while lazy calibration is still collecting must not write a
        # partial sidecar: the resume load fails on any missing scale, so the save is
        # all-or-nothing.
        model = _FakeModel(["per_tensor_static"] * 2)
        model.attns[0]._kv_scale = torch.tensor(0.5)
        model.attns[0]._kv_scale_ready = True

        maybe_save_kv_scale_sidecar(model, tmp_path)
        assert not (tmp_path / "kv_scale_sidecar.pt").exists()

    def test_sidecar_save_noop_without_static_modules(self, tmp_path):
        model = _FakeModel(["per_token_dynamic"])
        maybe_save_kv_scale_sidecar(model, tmp_path)
        assert not (tmp_path / "kv_scale_sidecar.pt").exists()

    def test_sidecar_missing_is_noop(self, tmp_path):
        model = _FakeModel(["per_tensor_static"])
        maybe_load_kv_scale_sidecar(model, tmp_path)
        assert not model.attns[0]._kv_scale_ready

    def test_sidecar_none_checkpoint_path_is_noop(self):
        model = _FakeModel(["per_tensor_static"])
        maybe_load_kv_scale_sidecar(model, None)
        assert not model.attns[0]._kv_scale_ready

    def test_sidecar_unknown_module_raises(self, tmp_path):
        torch.save({"ghost_module": torch.tensor(0.5)}, tmp_path / "kv_scale_sidecar.pt")
        model = _FakeModel(["per_tensor_static"])
        with pytest.raises(ValueError, match="absent from this model"):
            maybe_load_kv_scale_sidecar(model, tmp_path)


class TestKVQuantSPShardLocality:
    # SP splits the sequence before the fake-quant runs, so quantizing the whole sequence must
    # equal quantizing its shards independently and concatenating -- the CPU property test that
    # stands in for the 2-rank check (W3).
    def test_per_tensor_static_shard_concat_bitwise(self):
        # per_tensor broadcasts one scalar: whole-sequence and shard-wise quantization agree.
        torch.manual_seed(0)
        x = torch.randn(1, 32, 8)
        scale = torch.tensor(0.03125)
        whole = int8_fake_quant_static(x, scale)
        sharded = torch.cat(
            [int8_fake_quant_static(x[:, :13], scale), int8_fake_quant_static(x[:, 13:], scale)], dim=1
        )
        assert torch.equal(whole, sharded)

    def test_per_token_dynamic_shard_concat_bitwise(self):
        torch.manual_seed(0)
        x = torch.randn(1, 32, 8)
        whole = int8_fake_quant_dynamic(x)
        sharded = torch.cat([int8_fake_quant_dynamic(x[:, :13]), int8_fake_quant_dynamic(x[:, 13:])], dim=1)
        assert torch.equal(whole, sharded)


def _glm_like_to_hf_key(key: str) -> list[str]:
    # Minimal mirror of Glm53TextMoE.to_hf_key_list (MTP rewrite + language_model prefix): pins
    # the exported scale names without importing the model stack, same pattern as _FakeAttn.
    if key.startswith("mtp_block.layers.0."):
        key = f"layers.45.{key[len('mtp_block.layers.0.') :]}"
        key = key.replace(".decoder_layer.", ".")
    if "layers" in key:
        key = f"model.language_model.{key}"
    return [key]


def _place_module(root: nn.Module, path: str, leaf: nn.Module) -> None:
    parts = path.split(".")
    node = root
    for part in parts[:-1]:
        existing = getattr(node, part, None)
        if not isinstance(existing, nn.Module):
            existing = nn.Module()
            node.add_module(part, existing)
        node = existing
    node.add_module(parts[-1], leaf)


def _export_tree(specs: dict[str, tuple[str, torch.Tensor | None]]) -> nn.Module:
    # A compose-shaped tree: static KV quant modules under ``language_model``, with the text
    # model owning ``to_hf_key_list``. Scales mirror the training contract: plain attributes
    # filled once the calibration freezes (never registered buffers).
    root = nn.Module()
    root.add_module("language_model", nn.Module())
    root.language_model.to_hf_key_list = _glm_like_to_hf_key
    for path, (mode, scale) in specs.items():
        attn = _FakeAttn(mode)
        attn._kv_scale = scale
        attn._kv_scale_ready = scale is not None
        _place_module(root.language_model, path, attn)
    return root


class TestHfKvScaleKeyMapping:
    def test_layer_path_maps_to_hf_k_scale(self):
        key = _hf_scale_key(
            "language_model.layers.3._checkpoint_wrapped_module.self_attn", _glm_like_to_hf_key, "fa_k.scale"
        )
        assert key == "model.language_model.layers.3.self_attn.fa_k.scale"

    def test_mtp_path_maps_to_hf_k_scale(self):
        key = _hf_scale_key(
            "language_model.mtp_block.layers.0._checkpoint_wrapped_module.decoder_layer.self_attn",
            _glm_like_to_hf_key,
            "fa_k.scale",
        )
        assert key == "model.language_model.layers.45.self_attn.fa_k.scale"


class TestHfKvScaleExport:
    # The six ModelSlim FAKQuant weight-entry suffixes, in the order the export writes them.
    _FA_SUFFIXES = ("fa_q.scale", "fa_k.scale", "fa_v.scale", "fa_q.offset", "fa_k.offset", "fa_v.offset")

    def test_export_per_tensor_writes_modelslim_artifacts(self, tmp_path):
        scale3 = torch.tensor(0.044537)
        scale_mtp = torch.tensor(0.048474)
        root = _export_tree(
            {
                "layers.3._checkpoint_wrapped_module.self_attn": ("per_tensor_static", scale3),
                "mtp_block.layers.0._checkpoint_wrapped_module.decoder_layer.self_attn": (
                    "per_tensor_static",
                    scale_mtp,
                ),
            }
        )
        config_before = json.dumps({"model_type": "glm5_next_text"}, indent=2)
        (tmp_path / "config.json").write_text(config_before)
        index = {
            "weight_map": {
                "model.language_model.layers.3.self_attn.kv_a_proj_with_mqa.weight": (
                    "model-language-00001-of-00002.safetensors"
                )
            },
            "metadata": {},
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index, indent=4))

        maybe_write_hf_kv_scales(root, tmp_path)

        layer_paths = (
            "model.language_model.layers.3.self_attn",
            "model.language_model.layers.45.self_attn",
        )
        scales = load_file(tmp_path / "kv_scale.safetensors")
        assert set(scales) == {f"{path}.{suffix}" for path in layer_paths for suffix in self._FA_SUFFIXES}
        # The frozen per-tensor scale rides the fa_k.scale entry, reshaped to the consumer's
        # [num_kv_heads, 1] parameter shape (MLA latent caches have one KV head).
        assert scales["model.language_model.layers.3.self_attn.fa_k.scale"].shape == (1, 1)
        assert scales["model.language_model.layers.3.self_attn.fa_k.scale"].dtype == torch.float32
        assert torch.equal(scales["model.language_model.layers.3.self_attn.fa_k.scale"], scale3.reshape(1, 1))
        assert torch.equal(scales["model.language_model.layers.45.self_attn.fa_k.scale"], scale_mtp.reshape(1, 1))
        # Symmetric int8 has no zero point; the Q/V scales the MLA runtime never reads export
        # at their identity so the loader initializes them instead of leaving empty garbage.
        for path in layer_paths:
            for suffix in ("fa_q.offset", "fa_k.offset", "fa_v.offset"):
                offset = scales[f"{path}.{suffix}"]
                assert offset.dtype == torch.int8
                assert torch.equal(offset, torch.zeros(1, 1, dtype=torch.int8))
            assert torch.equal(scales[f"{path}.fa_q.scale"], torch.ones(1, 1))
            assert torch.equal(scales[f"{path}.fa_v.scale"], torch.ones(1, 1))
        merged = json.loads((tmp_path / "model.safetensors.index.json").read_text())
        for path in layer_paths:
            for suffix in self._FA_SUFFIXES:
                assert merged["weight_map"][f"{path}.{suffix}"] == "kv_scale.safetensors"
        # The pre-existing weight entry must survive the merge untouched.
        assert (
            merged["weight_map"]["model.language_model.layers.3.self_attn.kv_a_proj_with_mqa.weight"]
            == "model-language-00001-of-00002.safetensors"
        )
        description = json.loads((tmp_path / "quant_model_description.json").read_text())
        assert description["version"] == "1.0.0"
        assert description["fa_quant_type"] == "FAKQuant"
        assert {key for key in description if ".fa_" in key} == {
            f"{path}.{suffix}" for path in layer_paths for suffix in self._FA_SUFFIXES
        }
        meta = json.loads((tmp_path / "kv_scale_meta.json").read_text())
        assert meta == {
            "format_version": 1,
            "kv_quant_mode": "per_tensor_static",
            "kv_scales_file": "kv_scale.safetensors",
            "producer": "xtuner-kv-qat",
            "consumer_recipe": "quant_model_description.json",
        }
        # config.json must stay byte-identical: the ModelSlim consumer activates on the
        # description-file presence only when config.json carries no quantization_config,
        # while an unknown quant_method there makes upstream vLLM reject the checkpoint.
        assert (tmp_path / "config.json").read_text() == config_before

    def test_export_skips_until_all_scales_frozen(self, tmp_path):
        # All-or-nothing like the training sidecar: one unfrozen module means no artifacts at
        # all, so a mid-calibration export never describes a grid the weights do not carry.
        root = _export_tree(
            {
                "layers.3.self_attn": ("per_tensor_static", torch.tensor(0.04)),
                "layers.7.self_attn": ("per_tensor_static", None),
            }
        )
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "glm5_next_text"}))

        maybe_write_hf_kv_scales(root, tmp_path)

        assert not (tmp_path / "kv_scale.safetensors").exists()
        assert not (tmp_path / "kv_scale_meta.json").exists()
        assert json.loads((tmp_path / "config.json").read_text()) == {"model_type": "glm5_next_text"}

    def test_export_noop_without_static_modules(self, tmp_path):
        root = nn.Module()
        root.net = nn.Linear(4, 4)

        # A plain run must not grow files, and must not require an export layout to exist.
        maybe_write_hf_kv_scales(root, tmp_path)
        assert list(tmp_path.iterdir()) == []


class TestPerTensorModelSlimConsumer:
    """Feed the per-tensor export through the real vllm-ascend ModelSlim consumer code.

    The consumer code lives verbatim in ``_modelslim_consumer_ref.py`` (sources and
    deviations documented there), so these tests run offline with no vllm/vllm-ascend
    installation.

    The consumer contract these tests pin: the export is detected by the
    ``quant_model_description.json`` file presence (config.json stays free of
    ``quantization_config``); the recipe parses into ``enable_fa_quant`` plus the quantized
    layer list; the exported ``.fa_*.{scale,offset}`` tensor names are what the loader-side
    cache-scale mapper consumes; and the FAKQuant scheme's weight loader plus
    ``process_weights_after_loading`` turn the exported tensors into the runtime
    ``fak_descale``/``quant_kscale`` the prolog kernel reads.
    """

    def test_detect_and_parse_exported_description(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        root = _export_tree(
            {
                "layers.3._checkpoint_wrapped_module.self_attn": ("per_tensor_static", torch.tensor(0.044537)),
                "layers.7.self_attn": ("per_tensor_static", torch.tensor(0.051234)),
            }
        )
        maybe_write_hf_kv_scales(root, model_dir)
        # The exported config.json has no quantization_config: its absence is exactly what
        # lets the ascend method auto-activate on NPU (override_quantization_method).
        (model_dir / "config.json").write_text(json.dumps({"model_type": "glm5_next_text"}))

        assert detect_quantization_method(str(model_dir)) == "ascend"

        config = AscendModelSlimConfig()
        config.maybe_update_config(str(model_dir))
        assert config.enable_fa_quant is True
        assert config.kvcache_quant_layers == [3, 7]
        assert config.quant_description["fa_quant_type"] == "FAKQuant"

    def test_exported_fa_names_remap_to_mla_params(self):
        config = AscendModelSlimConfig(
            {
                "fa_quant_type": "FAKQuant",
                "model.layers.3.self_attn.fa_k.scale": "FAQuant",
            }
        )

        mapper = config.get_cache_scale_mapper()

        # The same regex the weight loader applies to our kv_scale.safetensors keys: the
        # exported names carry the HF self_attn segment, and the mapper inserts the MLA
        # nesting the consumer's attention modules own their FA parameters under.
        assert (
            mapper._map_name("model.layers.3.self_attn.fa_k.scale")
            == "model.layers.3.self_attn.mla_attn.mla_attn.fa_k.scale"
        )
        assert (
            mapper._map_name("model.layers.3.self_attn.fa_k.offset")
            == "model.layers.3.self_attn.mla_attn.mla_attn.fa_k.offset"
        )

    def test_fakquant_scheme_loads_exported_tensors(self, tmp_path):
        scale = torch.tensor(0.044537)
        root = _export_tree({"layers.3._checkpoint_wrapped_module.self_attn": ("per_tensor_static", scale)})
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        maybe_write_hf_kv_scales(root, model_dir)
        tensors = load_file(model_dir / "kv_scale.safetensors")

        # Bypass __init__ (it reads the serving engine's vllm_config); everything downstream
        # of construction only needs the two ranks the scheme caches from the HF config.
        scheme = AscendFAQuantAttentionMethod.__new__(AscendFAQuantAttentionMethod)
        scheme.kv_lora_rank = 512
        scheme.qk_rope_head_dim = 64

        layer = nn.Module()
        layer.num_heads = 1
        layer.num_kv_heads = 1  # MLA latent caches have a single KV head
        scheme.create_weights(layer)

        # The parameter shapes pin the export-side [1, 1] tensor shapes.
        assert layer.fa_k.scale.shape == (1, 1)
        assert layer.fa_k.offset.dtype == torch.int8

        layer_prefix = "model.language_model.layers.3.self_attn"
        for name in ("fa_q.scale", "fa_k.scale", "fa_v.scale", "fa_q.offset", "fa_k.offset", "fa_v.offset"):
            module_name, param_name = name.rsplit(".", 1)
            param = getattr(getattr(layer, module_name), param_name)
            _fa_quant_weight_loader(param, tensors[f"{layer_prefix}.{name}"])

        assert layer.fa_k.scale.item() == pytest.approx(scale.item())

        scheme.process_weights_after_loading(layer)

        # The runtime dequant reads fak_descale = scale and quant_kscale = 1/scale broadcast
        # over the ckv latent channels: x = int8_cache * scale, our amax/127 convention.
        assert layer.fak_descale.shape == (1,)
        assert layer.fak_descale.item() == pytest.approx(scale.item())
        reciprocal = (1.0 / scale.to(torch.float32)).reshape(1, 1).expand(1, 512)
        assert layer.quant_kscale.shape == (1, 512)
        assert torch.allclose(layer.quant_kscale, reciprocal)
        assert layer.fak_offset.item() == 0.0


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401

        return torch.npu.device_count() > 0
    except (ImportError, RuntimeError):
        return False


@pytest.mark.gpu
class TestDynamicBlockNpuOpParity:
    # The training kernel vs the exact op vllm-ascend calls on the serving path (verbatim
    # excerpt in _sfa_c8_dynamic_block_ref.py). Scales must match the op BITWISE (the probe
    # showed the op computes amax * (1/127) in fp32); int8 levels may differ on ~0.03%
    # knife-edge values by exactly one level (CANN-internal rounding path), which is
    # immaterial against the scale/2 grid error the dequant contract guarantees.

    @pytest.mark.skipif(not _npu_available(), reason="requires an NPU device")
    def test_matches_deployment_op(self):
        for seed in range(4):
            torch.manual_seed(seed)
            x = (torch.randn(8, 512) * 0.05).bfloat16().to("npu")
            y_ref, scale_bytes = custom_kv_rmsnorm_rope_quant_int8(x, 512)
            scale_ref = scale_bytes.view(torch.float32)  # unpack the consumer byte transport

            out = int8_fake_quant_dynamic_block(x)

            # Scale parity, bitwise: our kernel and the op must derive identical fp32 scales.
            tiles = x.float().view(8, 4, _DYNAMIC_BLOCK_TILE)
            scale_kernel = tiles.abs().amax(dim=-1, keepdim=True).mul_(1.0 / 127.0)
            assert scale_kernel.view(8, 4).dtype == torch.float32
            assert torch.equal(scale_kernel.view(8, 4), scale_ref)
            assert _DYNAMIC_BLOCK_TILE == SFA_QSFA_TILE_SIZE

            # Level parity within one int8 level on the shared grid.
            s4 = scale_ref.view(8, 4, 1)
            q_mirror = torch.clamp(torch.round(tiles / torch.where(s4 > 0, s4, torch.ones_like(s4))), -127.0, 127.0)
            diff = (q_mirror - y_ref.float().view(8, 4, _DYNAMIC_BLOCK_TILE)).abs()
            assert diff.max().item() <= 1.0
            assert (diff > 0).float().mean().item() < 0.005

            # Dequant contract: our fake-quant output stays within the scale/2 grid error.
            # The kernel casts the fp32 product back to bf16, which adds one bf16 rounding
            # step (relative ulp <= 2^-7) on top of the fp32 contract bound.
            err = (out.float() - x.float()).view(8, 4, _DYNAMIC_BLOCK_TILE)
            bf16_ulp = x.float().abs().view(8, 4, _DYNAMIC_BLOCK_TILE) * (2.0**-7)
            assert torch.all(err.abs() <= 0.5 * s4 + bf16_ulp + 1e-6)
