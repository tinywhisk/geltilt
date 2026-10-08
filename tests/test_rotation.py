from __future__ import annotations

import numpy as np
import pytest

from geltilt.rotation import (
    expanded_size,
    rotate_image,
    safe_crop_fraction,
)


def test_rotate_preserves_shape_and_dtype():
    arr = (np.random.default_rng(0).random((40, 30)) * 255).astype(np.uint8)
    out = rotate_image(arr, 10.0)
    assert out.shape == arr.shape
    assert out.dtype == arr.dtype


def test_rotate_preserves_channels():
    arr = (np.random.default_rng(0).random((20, 25, 3)) * 255).astype(np.uint8)
    out = rotate_image(arr, -5.0)
    assert out.shape == arr.shape


def test_rotate_uint16_preserved():
    arr = (np.random.default_rng(0).random((20, 20)) * 60000).astype(np.uint16)
    out = rotate_image(arr, 3.0)
    assert out.dtype == np.uint16


def test_expanded_size_grows_for_rotation():
    w, h = expanded_size((100, 200), 30.0)
    assert w > 200 and h > 100


def test_safe_crop_fraction_bounds():
    frac = safe_crop_fraction(15.0)
    assert 0.5 < frac < 1.0
    assert safe_crop_fraction(0.0) == pytest.approx(1.0)