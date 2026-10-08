from __future__ import annotations

import numpy as np

from geltilt.preprocess import (
    content_bbox,
    content_mask,
    detect_polarity,
    normalize_signal,
    to_signal,
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