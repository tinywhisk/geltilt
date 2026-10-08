from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from geltilt.imageio import load_array, load_gray, save_array, to_gray


def test_to_gray_from_rgb():
    rgb = np.zeros((4, 5, 3), np.uint8)
    rgb[:, :, 0] = 100
    gray = to_gray(rgb)
    assert gray.shape == (4, 5)
    assert gray.dtype == np.float32
    assert np.allclose(gray, 0.299 * 100, atol=0.5)


def test_roundtrip_uint8(tmp_path):
    arr = np.arange(256, dtype=np.uint8).reshape(16, 16)
    path = tmp_path / "a.png"
    save_array(path, arr)
    loaded = load_array(path)
    assert loaded.dtype == np.uint8
    assert np.array_equal(loaded, arr)


def test_roundtrip_uint16(tmp_path):
    arr = (np.arange(256, dtype=np.uint16) * 257).reshape(16, 16)
    path = tmp_path / "a.tif"
    save_array(path, arr)
    loaded = load_array(path)
    assert loaded.dtype == np.uint16
    assert np.array_equal(loaded, arr)


def test_load_gray_float(tmp_path):
    arr = np.full((8, 8), 120, np.uint8)
    path = tmp_path / "g.png"
    save_array(path, arr)
    gray = load_gray(path)
    assert gray.dtype == np.float32
    assert gray.shape == (8, 8)


def test_save_rejects_bad_shape(tmp_path):
    with pytest.raises(ValueError):
        save_array(tmp_path / "bad.png", np.zeros((2, 2, 5), np.uint8))