"""Parity tests for the env-gated NPU fused ViT RMSNorm/rotary paths.

The gates are read once at import time of ``modeling_vision``, so each test runs a
subprocess with the gate env pinned before the module import; setting the env inside the
pytest process would leak the fused paths into sibling test modules that compare bitwise.
"""

import os
import subprocess
import sys

import pytest


torch_npu = pytest.importorskip("torch_npu")

_PARITY_SCRIPT = r"""
import torch

dev = "npu:0"
EPS = 1e-5

from xtuner.v1.model.compose.glm53 import modeling_vision as mv


def eager_rms_norm(x, weight, eps):
    input_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(input_dtype)


def eager_rotary(q, k, cos, sin):
    def rotate_half(t):
        x1 = t[..., : t.shape[-1] // 2]
        x2 = t[..., t.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    oq, ok = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    return ((q * cos) + (rotate_half(q) * sin)).to(oq), ((k * cos) + (rotate_half(k) * sin)).to(ok)


torch.manual_seed(0)
q = torch.randn(8192, 16, 64, device=dev).to(torch.bfloat16)
k = torch.randn(8192, 16, 64, device=dev).to(torch.bfloat16)
freqs = torch.randn(8192, 32, device=dev)
cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
qkv = torch.randn(8192, 3, 16, 64, device=dev).to(torch.bfloat16)
pq, pk, _ = qkv.permute(1, 0, 2, 3).unbind(0)
weight = (torch.randn(1024, device=dev) * 0.2 + 1.0).to(torch.bfloat16)
norm = mv.Glm53VisionRMSNorm(1024, eps=EPS).to(device=dev, dtype=torch.bfloat16)
with torch.no_grad():
    norm.weight.copy_(weight)

# Rotary must be bitwise-identical on dense inputs and on the production-style
# permute/unbind views; RMSNorm must be bitwise at 4096 rows and at most 1 bf16 ulp off
# at 8192 rows (the NPU kernel's fp32 rstd differs from torch's reduction by 1 fp32 ulp).
if mv._VIT_FUSED_NORM_ENABLED and mv._VIT_FUSED_ROTARY_ENABLED:
    for qq, kk in ((q, k), (pq, pk)):
        fq, fk = mv.apply_rotary_pos_emb_vision(qq, kk, cos, sin)
        rq, rk = eager_rotary(qq, kk, cos, sin)
        assert torch.equal(fq, rq) and torch.equal(fk, rk)
    for rows, expect_bitwise in ((4096, True), (8192, False)):
        x = (torch.randn(rows, 1024, device=dev) * 2.0).to(torch.bfloat16)
        ref = eager_rms_norm(x, weight, EPS)
        out = norm(x)
        if expect_bitwise:
            assert torch.equal(out, ref)
        else:
            tol = ref.float().abs().max().item() * (2.0**-7)
            assert (out.float() - ref.float()).abs().max().item() <= tol
else:
    assert not mv._VIT_FUSED_NORM_ENABLED and not mv._VIT_FUSED_ROTARY_ENABLED
    for qq, kk in ((q, k), (pq, pk)):
        fq, fk = mv.apply_rotary_pos_emb_vision(qq, kk, cos, sin)
        rq, rk = eager_rotary(qq, kk, cos, sin)
        assert torch.equal(fq, rq) and torch.equal(fk, rk)
    for rows in (4096, 8192):
        x = (torch.randn(rows, 1024, device=dev) * 2.0).to(torch.bfloat16)
        assert torch.equal(norm(x), eager_rms_norm(x, weight, EPS))
print("PARITY_OK")
"""


def _run_parity_script(gates: dict[str, str]) -> None:
    env = {**os.environ, **gates}
    result = subprocess.run(
        [sys.executable, "-c", _PARITY_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0 and "PARITY_OK" in result.stdout, result.stderr[-2000:]


class TestGlm53VisionFusedOps:
    @pytest.mark.gpu
    def test_npu_fused_rotary_bitwise_and_norm_within_1_ulp(self) -> None:
        _run_parity_script({"XTUNER_VIT_FUSED_NORM": "1", "XTUNER_VIT_FUSED_ROTARY": "1"})

    @pytest.mark.gpu
    def test_gates_default_off_production_path_matches_eager(self) -> None:
        _run_parity_script({"XTUNER_VIT_FUSED_NORM": "0", "XTUNER_VIT_FUSED_ROTARY": "0"})
