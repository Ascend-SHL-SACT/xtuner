# Copyright (c) OpenMMLab. All rights reserved.
"""Regression test for the XTUNER_DATASET_BLOB_GETITEM_MAX_MB RAM-resident read path.

The blob branch of ``JsonlDataset.__getitem__`` must return a byte-identical
line to the original ``open()+seek()+readline()`` path for every sample,
including edge cases (empty lines, unicode, last line with/without a trailing
newline). The test mirrors the exact slicing logic of ``__getitem__``: offsets
are line-start byte positions with the trailing file-size sentinel stripped,
so the last line's end falls back to ``len(blob)``.
"""

import json
from pathlib import Path

import numpy as np
import pytest


def _build_offsets(path: Path) -> np.ndarray:
    """Mirror ``JsonlDataset.count_offsets`` + the ``__init__`` sentinel strip."""
    offsets = [0]
    with open(path, "rb") as f:
        for line in f:
            offsets.append(offsets[-1] + len(line))
    return np.array(offsets[:-1])  # strip trailing file_size sentinel


def _blob_line(blob: bytes, offsets: np.ndarray, item: int) -> str:
    """Mirror the XTUNER_DATASET_BLOB_GETITEM_MAX_MB branch of ``__getitem__``."""
    start = int(offsets[item])
    end = int(offsets[item + 1]) if item + 1 < len(offsets) else len(blob)
    return blob[start:end].decode()


def _readline_line(path: Path, offsets: np.ndarray, item: int) -> str:
    """Mirror the default (open-per-sample) branch of ``__getitem__``."""
    with open(path) as f:
        f.seek(int(offsets[item]))
        return f.readline()


class TestBlobGetitemByteEquivalence:
    @pytest.mark.parametrize(
        "content",
        [
            "hello\nworld\nfoo\n",
            "\n\nx\n",  # empty lines
            "naïve café\n日本語\nemoji 🚀\n",  # unicode
            "a\nb\nc",  # last line NO trailing newline
            "single line no newline",  # one line, no newline
            "trailing\n\n",  # trailing empty line
        ],
    )
    def test_blob_equals_readline(self, tmp_path, content):
        path = tmp_path / "data.jsonl"
        path.write_text(content, encoding="utf-8")
        blob = path.read_bytes()
        offsets = _build_offsets(path)
        assert len(offsets) >= 1
        for i in range(len(offsets)):
            blob_line = _blob_line(blob, offsets, i)
            rl_line = _readline_line(path, offsets, i)
            assert blob_line == rl_line, f"line {i}: blob={blob_line!r} readline={rl_line!r}"

    def test_blob_lines_valid_json_on_real_file(self):
        path = Path("/deeplink/dyb/datasets/alpaca/alpaca_messages.jsonl")
        if not path.exists():
            pytest.skip("real alpaca dataset not available on this host")
        blob = path.read_bytes()
        offsets = _build_offsets(path)
        assert len(offsets) > 0
        for i in range(len(offsets)):
            json.loads(_blob_line(blob, offsets, i))  # must not raise
