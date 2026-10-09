"""GLM-5.3 KDA ``cu_seqlens`` host-list 回归测试（``XTUNER_KDA_CU_HOST_LIST``）。

背景 bug：训练态把 ``SequenceContext.cu_seq_lens_q``（device tensor）直接喂给 conv/KDA 内核包装，
包装每次调用 ``.tolist()`` 做 D2H 同步停等（~480 次/步，且常停在最近的 a2a fence 后面）。修复分两层：

1. ``xtuner.v1.module.attention.kda``：新增 ``_cu_seqlens_arg(seq_ctx, device)`` —— npu 设备、
   flag 开（``XTUNER_KDA_CU_HOST_LIST``，默认开）且 ``cu_seq_lens_q_list`` 非 None 时返回
   ``SequenceContext.__init__`` 一次性物化的 host list（与 tensor 恒等），否则维持 device tensor；
   非 SP ``forward`` 与 ``forward_for_sp`` 两处调用点均改用它。
2. ``xtuner.v1.ops.kda.npu_backend.causal_conv1d``：入参已是 list 时直接使用（不再 ``tolist``），
   Tensor 走 ``tolist()``，None 保持 None；host list 与 tensor 两条入参必须产出逐位一致的结果。

全部用例 CPU 运行、不依赖真 NPU 设备：npu 分支用 ``torch.device("npu")`` 类型对象驱动（
``device.type`` 判断不需要真实设备），数值路径走 ``causal_conv1d`` 的 CPU dense 后备实现。
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

import xtuner.v1.module.attention.kda as kda_module
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.module.attention import kda_lever
from xtuner.v1.module.attention.kda import KDAShortConvolution
from xtuner.v1.ops.kda.kda_op import _as_cu_list
from xtuner.v1.ops.kda.npu_backend import causal_conv1d


_REPO_ROOT = str(Path(__file__).resolve().parents[2])


class _FakeSeqContext:
    def __init__(self, cu_seq_lens_q: torch.Tensor, cu_seq_lens_q_list: list[int] | None) -> None:
        self.cu_seq_lens_q = cu_seq_lens_q
        self.cu_seq_lens_q_list = cu_seq_lens_q_list


def _install_tolist_spy(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    counter = [0]
    original = torch.Tensor.tolist

    def counting_tolist(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        counter[0] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "tolist", counting_tolist)
    return counter


def _capture_conv(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def stub(**kwargs: Any) -> tuple[torch.Tensor, None]:
        calls.append(kwargs)
        return kwargs["x"], None

    monkeypatch.setattr(kda_module, "causal_conv1d", stub)
    return calls


class TestCuSeqlensArgBranch:
    """``cu_seqlens_arg`` 分支选择：flag、``device.type``、host list 是否存在三条件的组合."""

    def test_flag_on_npu_device_returns_host_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(kda_lever, "CU_HOST_LIST", True)
        cu = torch.tensor([0, 32, 64])
        ctx = _FakeSeqContext(cu, [0, 32, 64])
        out = kda_lever.cu_seqlens_arg(ctx, torch.device("npu"), True)
        assert isinstance(out, list)
        assert out == cu.tolist()
        assert out is ctx.cu_seq_lens_q_list

    def test_flag_off_returns_device_tensor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(kda_lever, "CU_HOST_LIST", False)
        cu = torch.tensor([0, 32, 64])
        ctx = _FakeSeqContext(cu, [0, 32, 64])
        assert kda_lever.cu_seqlens_arg(ctx, torch.device("npu"), True) is cu

    def test_cpu_device_returns_device_tensor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(kda_lever, "CU_HOST_LIST", True)
        cu = torch.tensor([0, 32, 64])
        ctx = _FakeSeqContext(cu, [0, 32, 64])
        assert kda_lever.cu_seqlens_arg(ctx, torch.device("cpu"), True) is cu

    def test_missing_host_list_returns_device_tensor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(kda_lever, "CU_HOST_LIST", True)
        cu = torch.tensor([0, 32, 64])
        ctx = _FakeSeqContext(cu, None)
        assert kda_lever.cu_seqlens_arg(ctx, torch.device("npu"), True) is cu

    def test_real_sequence_context_feeds_host_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(kda_lever, "CU_HOST_LIST", True)
        segments = (torch.arange(16).unsqueeze(0), torch.arange(32).unsqueeze(0))
        ctx = SequenceContext.from_input_ids(segments, device="cpu")
        out = kda_lever.cu_seqlens_arg(ctx, torch.device("npu"), True)
        assert out is ctx.cu_seq_lens_q_list
        assert out == [0, 16, 48]
        assert out == ctx.cu_seq_lens_q.tolist()


class TestShortConvHostListPassthrough:
    """``KDAShortConvolution.forward`` 透传层：``cu_seqlens`` 原样交给内核包装，自身不做任何 ``tolist``."""

    def test_forward_passes_host_list_untouched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conv = KDAShortConvolution(32, kernel_size=4, activation="silu")
        calls = _capture_conv(monkeypatch)
        spy = _install_tolist_spy(monkeypatch)
        x = torch.randn(1, 64, 32)
        weight = torch.randn(32, 4)
        cu = [0, 32, 64]
        y, state = conv(x, cu_seqlens=cu, weight=weight)
        assert len(calls) == 1
        assert calls[0]["cu_seqlens"] is cu
        assert calls[0]["weight"] is weight
        assert calls[0]["bias"] is None
        assert calls[0]["activation"] == "silu"
        assert spy[0] == 0
        assert torch.equal(y, x)
        assert state is None

    def test_forward_passes_device_tensor_untouched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conv = KDAShortConvolution(32, kernel_size=4, activation=None)
        calls = _capture_conv(monkeypatch)
        spy = _install_tolist_spy(monkeypatch)
        x = torch.randn(1, 64, 32)
        cu = torch.tensor([0, 32, 64])
        conv(x, cu_seqlens=cu, weight=torch.randn(32, 4))
        assert len(calls) == 1
        assert calls[0]["cu_seqlens"] is cu
        assert spy[0] == 0

    def test_forward_materializes_weight_when_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conv = KDAShortConvolution(32, kernel_size=4, activation=None)
        calls = _capture_conv(monkeypatch)
        conv(torch.randn(1, 64, 32), cu_seqlens=[0, 32, 64])
        assert len(calls) == 1
        weight = calls[0]["weight"]
        assert weight.shape == (32, 4)
        assert weight.dtype == torch.float32
        assert torch.equal(weight, conv.weight.detach().squeeze(1))
        assert calls[0]["bias"] is None


class TestNpuBackendCausalConv1dDense:
    """``causal_conv1d`` CPU dense 路径：host list 与 tensor 入参逐位等价，且 list 不触发 D2H ``tolist``."""

    def _inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(7)
        return torch.randn(1, 64, 32), torch.randn(32, 4)

    def test_list_and_tensor_inputs_bitwise_equal(self) -> None:
        x, weight = self._inputs()
        y_list, state_list = causal_conv1d(x=x, weight=weight, cu_seqlens=[0, 32, 64])
        y_tensor, state_tensor = causal_conv1d(x=x, weight=weight, cu_seqlens=torch.tensor([0, 32, 64]))
        assert torch.equal(y_list, y_tensor)
        assert y_list.shape == (1, 64, 32)
        assert state_list is None and state_tensor is None

    def test_list_input_avoids_tolist(self, monkeypatch: pytest.MonkeyPatch) -> None:
        x, weight = self._inputs()
        spy = _install_tolist_spy(monkeypatch)
        causal_conv1d(x=x, weight=weight, cu_seqlens=[0, 32, 64])
        assert spy[0] == 0
        causal_conv1d(x=x, weight=weight, cu_seqlens=torch.tensor([0, 32, 64]))
        assert spy[0] >= 1

    def test_varlen_matches_per_segment_dense(self) -> None:
        _, weight = self._inputs()
        x = torch.zeros(1, 64, 32)
        x[:, :32] = 1.0
        x[:, 32:] = 2.0
        y_varlen, _ = causal_conv1d(x=x, weight=weight, cu_seqlens=[0, 32, 64])
        y_full, _ = causal_conv1d(x=x, weight=weight)
        assert not torch.equal(y_varlen, y_full)
        y0, _ = causal_conv1d(x=x[:, :32], weight=weight)
        y1, _ = causal_conv1d(x=x[:, 32:], weight=weight)
        assert torch.equal(y_varlen, torch.cat([y0, y1], dim=1))

    def test_none_equals_single_full_segment(self) -> None:
        x, weight = self._inputs()
        y_none, _ = causal_conv1d(x=x, weight=weight)
        y_full_range, _ = causal_conv1d(x=x, weight=weight, cu_seqlens=[0, 64])
        assert torch.equal(y_none, y_full_range)

    def test_empty_segment_is_skipped(self) -> None:
        x, weight = self._inputs()
        y_empty, _ = causal_conv1d(x=x, weight=weight, cu_seqlens=[0, 16, 16, 64])
        y_plain, _ = causal_conv1d(x=x, weight=weight, cu_seqlens=[0, 16, 64])
        assert y_empty.shape == (1, 64, 32)
        assert torch.equal(y_empty, y_plain)


class TestAsCuList:
    """``kda_op._as_cu_list``：Tensor 归一为 host list、list 等值、None 直通的契约."""

    def test_tensor_normalized_to_int_list(self) -> None:
        out = _as_cu_list(torch.tensor([0, 16, 48, 96], dtype=torch.int32))
        assert out == [0, 16, 48, 96]
        assert isinstance(out, list)
        assert all(isinstance(v, int) for v in out)

    def test_list_normalized_without_value_change(self) -> None:
        cu = [0, 16, 48, 96]
        assert _as_cu_list(cu) == cu

    def test_none_passes_through(self) -> None:
        assert _as_cu_list(None) is None


class TestSequenceContextHostList:
    """``SequenceContext`` host list 物化：``from_input_ids`` 多段 cumsum 与 device tensor 恒等."""

    def test_from_input_ids_materializes_cumsum_host_list(self) -> None:
        segments = (
            torch.arange(16).unsqueeze(0),
            torch.arange(32).unsqueeze(0),
            torch.arange(48).unsqueeze(0),
        )
        ctx = SequenceContext.from_input_ids(segments, device="cpu")
        assert ctx.cu_seq_lens_q_list == [0, 16, 48, 96]
        assert ctx.cu_seq_lens_q_list == ctx.cu_seq_lens_q.tolist()
        assert all(isinstance(v, int) for v in ctx.cu_seq_lens_q_list)


class TestCuHostListFlagDefault:
    """``XTUNER_KDA_CU_HOST_LIST`` 默认语义：未设置=开（host list），显式 ``0``=关（device tensor）."""

    def _flag_in_subprocess(self, env_value: str | None) -> bool:
        env = os.environ.copy()
        env.pop("XTUNER_KDA_CU_HOST_LIST", None)
        if env_value is not None:
            env["XTUNER_KDA_CU_HOST_LIST"] = env_value
        env["PYTHONPATH"] = os.pathsep.join([_REPO_ROOT, env.get("PYTHONPATH", "")])
        code = "import xtuner.v1.module.attention.kda_lever as k; print(k.CU_HOST_LIST)"
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        return proc.stdout.strip() == "True"

    def test_flag_defaults_on_and_env_zero_disables(self) -> None:
        assert self._flag_in_subprocess(None) is True
        assert self._flag_in_subprocess("0") is False
