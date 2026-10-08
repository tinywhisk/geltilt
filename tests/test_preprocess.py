from __future__ import annotations

import numpy as np
import pytest

from geltilt.preprocess import (
    content_bbox,
    content_mask,
    detect_polarity,
    normalize_signal,
    robust_limits,
    to_signal,
    to_u8,
)


def _bright_gel(h=64, w=64):
    img = np.full((h, w), 5.0, np.float32)
    img[30:34, 10:50] = 200.0
    return img


def _dark_gel(h=64, w=64):
    img = np.full((h, w), 240.0, np.float32)
    img[30:34, 10:50] = 20.0
    return img


def test_polarity_bright_and_dark():
    assert detect_polarity(_bright_gel()) == "bright"
    assert detect_polarity(_dark_gel()) == "dark"


def test_to_signal_is_non_negative_and_polarity_aware():
    sig_b, pol_b, _ = to_signal(_bright_gel(), polarity="auto")
    sig_d, pol_d, _ = to_signal(_dark_gel(), polarity="auto")
    assert pol_b == "bright" and pol_d == "dark"
    assert sig_b.min() >= 0 and sig_d.min() >= 0
    assert sig_b.max() > 0 and sig_d.max() > 0


def test_content_bbox_around_bands():
    mask = content_mask(normalize_signal(to_signal(_bright_gel())[0]))
    bbox = content_bbox(mask, margin_frac=0.0, min_margin=0)
    assert bbox is not None
    y0, y1, x0, x1 = bbox
    assert y0 <= 30 and y1 >= 34
    assert x0 <= 10 and x1 >= 50


def test_content_mask_empty_returns_none_bbox():
    mask = content_mask(np.zeros((16, 16), np.float32))
    assert content_bbox(mask) is None

def test_robust_limits_percentiles():
    arr = np.arange(1000, dtype=np.float32).reshape(10, 100)
    lo, hi = robust_limits(arr, 0.5, 99.5)
    assert lo < hi
    assert lo == pytest.approx(5.0, abs=1.0)
    assert hi == pytest.approx(994.0, abs=1.0)


def test_robust_limits_flat_image_has_span():
    lo, hi = robust_limits(np.full((8, 8), 7, np.uint8))
    assert hi == lo + 1.0


def test_to_u8_maps_range_and_rounds():
    arr = np.array([[0.0, 0.5, 1.0]], np.float32)
    out = to_u8(arr, 0.0, 1.0)
    assert out.dtype == np.uint8
    assert out.tolist() == [[0, 128, 255]]


def test_to_u8_clips_out_of_range():
    arr = np.array([[-10.0, 200.0]], np.float32)
    out = to_u8(arr, 0.0, 100.0)
    assert out.tolist() == [[0, 255]]


def test_to_u8_gamma_brightens():
    arr = np.array([[0.25]], np.float32)
    assert to_u8(arr, 0.0, 1.0, gamma=0.5)[0, 0] > to_u8(arr, 0.0, 1.0)[0, 0]