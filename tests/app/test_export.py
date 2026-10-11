from __future__ import annotations

import io
import struct

import numpy as np
import pytest
from PIL import Image

from geltilt.app import export
from geltilt.imageio import load_array


def test_to_uint8_passthrough_and_scales():
    assert export.to_uint8(np.array([[0, 255]], np.uint8)).dtype == np.uint8
    out = export.to_uint8(np.array([[0, 65535]], np.uint16))
    assert out.dtype == np.uint8
    assert out.tolist() == [[0, 255]]
    # a float image in [0, 1] maps to full scale; larger values are clipped
    f = export.to_uint8(np.array([[0.25, 1.0]], np.float32))
    assert f.tolist() == [[64, 255]]
    assert export.to_uint8(np.array([[300.0]], np.float32)).tolist() == [[255]]


def test_rotated_export_preserves_dtype_and_size():
    arr = np.zeros((50, 40), np.uint16)
    arr[10:20, 5:15] = 50000
    out = export.rotated_export(arr, 8.0)
    assert out.dtype == np.uint16
    assert out.shape == arr.shape


def test_rotated_export_changes_pixels():
    arr = np.zeros((60, 60), np.uint8)
    arr[10:20, 25:35] = 255
    assert np.array_equal(export.rotated_export(arr, 0.0), arr)
    assert not np.array_equal(export.rotated_export(arr, 15.0), arr)


def test_save_image_preserves_16bit_tiff(tmp_path):
    arr = (np.arange(20 * 16, dtype=np.uint16).reshape(20, 16) * 111).astype(np.uint16)
    dest = tmp_path / "out.tif"
    export.save_image(dest, arr)
    assert dest.exists()
    back = load_array(dest)
    assert back.dtype == np.uint16
    assert np.array_equal(back, arr)


def test_save_image_downcasts_for_jpeg(tmp_path):
    arr = np.full((20, 16), 40000, np.uint16)
    dest = tmp_path / "out.jpg"
    export.save_image(dest, arr)
    assert dest.exists()
    assert load_array(dest).dtype == np.uint8


def test_tiff_bytes_roundtrip_preserves_depth():
    arr = np.full((8, 8), 12345, np.uint16)
    with Image.open(io.BytesIO(export.tiff_bytes(arr))) as im:
        assert im.mode in ("I;16", "I;16B", "I;16L")
        assert im.size == (8, 8)


def test_dib_bytes_header_and_size():
    arr = np.zeros((4, 5), np.uint8)
    data = export.dib_bytes(arr)
    size, w, h, planes, bits = struct.unpack("<IiiHH", data[:16])
    assert (size, w, h, planes, bits) == (40, 5, 4, 1, 24)
    row = ((5 * 3 + 3) // 4) * 4  # 3-byte BGR rows padded to 4 bytes
    assert len(data) == 40 + row * 4


def test_clipboard_copy_is_windows_only(monkeypatch):
    monkeypatch.setattr(export.sys, "platform", "linux")
    with pytest.raises(RuntimeError):
        export.copy_tiff_to_clipboard(np.zeros((4, 4), np.uint8))