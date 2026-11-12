"""Tests for the HF-export copy branch: a verbatim-copied ``config.json`` must not keep a
source-inherited ``quantization_config``.

``_save_hf`` and ``_write_hf_index_and_config`` copy every non-safetensors source file,
so an export built from a source published under a quantization contract (for example
the official fp8 GLM-5.3-Flash release) inherits that annotation even though the
exported weights never carry the described format. The consumer contract is pinned in
``tests/quantization/_w8a8_consumer_ref.py``: upstream vLLM routes on
``config.json``'s ``quantization_config`` (an fp8 annotation on bf16/int8 weights fails
weight loading), and vllm-ascend's ModelSlim detection requires a ``config.json``
without ``quant_method``.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from xtuner.v1.model.base import BaseModel, _strip_inherited_quantization_config


_FP8_QUANTIZATION_CONFIG = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128],
}


def _write_source_checkpoint(source_dir: Path, with_quantization_config: bool) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    config: dict[str, object] = {
        "architectures": ["Glm5NextForConditionalGeneration"],
        "model_type": "glm5_next",
        "text_config": {"dtype": "bfloat16", "num_hidden_layers": 2},
    }
    if with_quantization_config:
        config["quantization_config"] = dict(_FP8_QUANTIZATION_CONFIG)
    (source_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (source_dir / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")


def _make_export_model(source_dir: Path) -> BaseModel:
    model = object.__new__(BaseModel)
    model.config = SimpleNamespace(hf_config=None, hf_save_cfg=SimpleNamespace(worker_per_rank=1))
    model._hf_path = source_dir  # type: ignore[attr-defined]
    return model


class TestStripInheritedQuantizationConfig:
    def test_strips_quantization_config_and_preserves_key_order(self, tmp_path: Path) -> None:
        """A copied config.json loses exactly the inherited quantization_config key."""
        config_path = tmp_path / "config.json"
        payload = {
            "model_type": "glm5_next",
            "quantization_config": dict(_FP8_QUANTIZATION_CONFIG),
            "text_config": {"dtype": "bfloat16"},
            "torch_dtype": "bfloat16",
        }
        config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        _strip_inherited_quantization_config(tmp_path)

        stripped = json.loads(config_path.read_text(encoding="utf-8"))
        assert "quantization_config" not in stripped
        assert list(stripped.keys()) == ["model_type", "text_config", "torch_dtype"]
        assert stripped["text_config"] == {"dtype": "bfloat16"}

    def test_leaves_config_without_quantization_config_byte_identical(self, tmp_path: Path) -> None:
        """A clean source must export byte-identically: the strip is a no-op."""
        config_path = tmp_path / "config.json"
        content = '{\n  "model_type": "glm5_next"\n}\n'
        config_path.write_text(content, encoding="utf-8")

        _strip_inherited_quantization_config(tmp_path)

        assert config_path.read_text(encoding="utf-8") == content

    def test_missing_config_json_is_noop(self, tmp_path: Path) -> None:
        """An export directory without config.json is left untouched."""
        _strip_inherited_quantization_config(tmp_path)

        assert not (tmp_path / "config.json").exists()

    def test_non_object_config_json_is_noop(self, tmp_path: Path) -> None:
        """A malformed (non-object) config.json must not be rewritten or raise."""
        config_path = tmp_path / "config.json"
        content = "[1, 2]\n"
        config_path.write_text(content, encoding="utf-8")

        _strip_inherited_quantization_config(tmp_path)

        assert config_path.read_text(encoding="utf-8") == content


class TestHfExportCopyBranchStrip:
    def test_write_hf_index_and_config_strips_inherited_quantization_config(self, tmp_path: Path) -> None:
        """The async export write path drops the inherited key and copies the rest verbatim."""
        source_dir = tmp_path / "source"
        _write_source_checkpoint(source_dir, with_quantization_config=True)
        export_dir = tmp_path / "export"
        export_dir.mkdir(parents=True, exist_ok=True)
        model = _make_export_model(source_dir)

        model._write_hf_index_and_config(hf_dir=export_dir, weight_map={})

        exported = json.loads((export_dir / "config.json").read_text(encoding="utf-8"))
        assert "quantization_config" not in exported
        assert exported["text_config"] == {"dtype": "bfloat16", "num_hidden_layers": 2}
        assert (export_dir / "tokenizer_config.json").read_text(encoding="utf-8") == "{}\n"
        index = json.loads((export_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
        assert index == {"weight_map": {}, "metadata": {}}

    def test_save_hf_copy_branch_strips_inherited_quantization_config(self, tmp_path: Path) -> None:
        """The sync export path drops the inherited key and copies the rest verbatim."""
        source_dir = tmp_path / "source"
        _write_source_checkpoint(source_dir, with_quantization_config=True)
        export_dir = tmp_path / "export"
        model = _make_export_model(source_dir)

        with (
            patch.object(BaseModel, "_get_hf_param", return_value=iter([])),
            patch.object(BaseModel, "_load_spec_params", return_value=[]),
            patch("xtuner.v1.model.base.DEVICE_MODULE"),
        ):
            model._save_hf(hf_dir=export_dir)

        exported = json.loads((export_dir / "config.json").read_text(encoding="utf-8"))
        assert "quantization_config" not in exported
        assert exported["architectures"] == ["Glm5NextForConditionalGeneration"]
        assert (export_dir / "tokenizer_config.json").read_text(encoding="utf-8") == "{}\n"
