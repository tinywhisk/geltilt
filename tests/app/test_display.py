from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from geltilt.app.display import DisplayPreprocessor
from geltilt.imageio import to_gray
from geltilt.preprocess import preview_u8


def _write(path: Path, arr: np.ndarray) -> None:
    if arr.dtype == np.uint16:
        Image.fromarray(arr).save(path)
    else:
        Image.fromarray(arr.astype(np.uint8), mode="L").save(path)


def test_display_matches_preview_transform(tmp_path: Path):
    arr = np.random.default_rng(0).integers(0, 255, size=(40, 30), dtype=np.uint8)
    path = tmp_path / "gel.png"
    _write(path, arr)

    pre = DisplayPreprocessor()
    u8 = pre.display_u8(path)
    expected = preview_u8(to_gray(arr))
    assert np.array_equal(u8, expected)


def test_display_caches(tmp_path: Path):
    arr = np.random.default_rng(1).integers(0, 255, size=(20, 20), dtype=np.uint8)
    path = tmp_path / "gel.png"
    _write(path, arr)
    pre = DisplayPreprocessor()
    first = pre.display_u8(path)
    second = pre.display_u8(path)
    assert first is second
    assert pre.get_cached_u8("gel.png") is first


def test_display_does_not_mutate_source(tmp_path: Path):
    arr = np.random.default_rng(2).integers(0, 65535, size=(30, 30), dtype=np.uint16)
    path = tmp_path / "gel16.tif"
    Image.fromarray(arr).save(path)
    pre = DisplayPreprocessor()
    source_before = pre.source_array(path).copy()
    pre.display_u8(path)
    assert np.array_equal(pre.source_array(path), source_before)
    assert pre.source_array(path).dtype == np.uint16