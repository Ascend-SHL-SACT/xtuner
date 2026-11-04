"""Tests for the KV-cache quantization config and int8 fake-quant kernels (CPU-only, self-contained)."""

import pytest
import torch
from pydantic import ValidationError

from xtuner.v1.quantization.kv import (
    KVQuantConfig,
    int8_fake_quant_dynamic,
    int8_fake_quant_static,
    kv_fakequant_enabled,
    kv_quant_config_from_env,
)


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
    def test_per_channel_scale_broadcast(self):
        # Power-of-two scale keeps the division/multiplication exact in fp32.
        scale = torch.tensor([2.0**-3] * 3 + [2.0**-5] * 5)
        levels = torch.arange(-127, 128, dtype=torch.float32)
        x = (levels[:, None] * scale[None, :])[:64]  # [64, 8], every entry on the grid
        out = int8_fake_quant_static(x, scale)
        assert torch.equal(out, x)

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

    def test_env_invalid_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("XTUNER_QUANT_KV", "int4")
        with pytest.raises(ValueError, match="XTUNER_QUANT_KV"):
            kv_quant_config_from_env()
