from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from geltilt.banddetect import band_bbox, band_centroids
from geltilt.bandmask import default_model_path
from geltilt.core import estimate_rotation_ladders
from geltilt.imageio import load_gray

SAMPLES = sorted(list((Path(__file__).resolve().parent.parent / "samples").glob("*.Tif")))


def _model_ready() -> bool:
    env = os.environ.get("GELTILT_GELGENIE_ONNX")
    return bool(env and Path(env).exists()) or default_model_path().exists()


requires = pytest.mark.skipif(
    not _model_ready() or not SAMPLES, reason="GelGenie model or samples not available"
)


@requires
def test_band_centroids_shape():
    xy = band_centroids(load_gray(SAMPLES[0]))
    assert xy.ndim == 2 and xy.shape[1] == 2
    assert len(xy) > 0
    assert np.isfinite(xy).all()


@requires
def test_estimate_rotation_ladders_returns_contract():
    res = estimate_rotation_ladders(load_gray(SAMPLES[0]))
    assert isinstance(res["ok"], bool)
    for key in ("angle", "se", "reason", "ladders", "n_bands"):
        assert key in res

class _FakeSegmenter:
    """Returns a canned probability map (no ONNX)."""

    def __init__(self, prob: np.ndarray) -> None:
        self._prob = prob

    def probability(self, _image) -> np.ndarray:
        return self._prob


def test_band_bbox_is_union_of_kept_components():
    prob = np.zeros((40, 40), np.float32)
    prob[5:9, 3:7] = 0.9  # area 16
    prob[20:25, 30:35] = 0.9  # area 25
    prob[38, 38] = 0.9  # area 1 -> dropped by min_area
    seg = _FakeSegmenter(prob)
    bbox = band_bbox(np.zeros((40, 40), np.float32), segmenter=seg, min_area=5, open_ksize=1)
    assert bbox == (3, 5, 35, 25)


def test_band_bbox_none_when_no_bands():
    seg = _FakeSegmenter(np.zeros((20, 20), np.float32))
    assert band_bbox(np.zeros((20, 20), np.float32), segmenter=seg) is None


def test_band_bbox_agrees_with_centroids_on_a_band():
    prob = np.zeros((30, 30), np.float32)
    prob[10:14, 12:20] = 0.9
    seg = _FakeSegmenter(prob)
    img = np.zeros((30, 30), np.float32)
    xy = band_centroids(img, segmenter=seg)
    bbox = band_bbox(img, segmenter=seg)
    assert len(xy) == 1
    assert bbox == (12, 10, 20, 14)

def test_band_bbox_drops_far_right_debris():
    prob = np.zeros((40, 100), np.float32)
    prob[10:20, 20:40] = 0.9  # real band cluster
    prob[10:20, 96:100] = 0.9  # debris blob in the rightmost 3% (cx=97.5)
    seg = _FakeSegmenter(prob)
    bbox = band_bbox(np.zeros((40, 100), np.float32), segmenter=seg, min_area=5, open_ksize=1)
    assert bbox == (20, 10, 40, 20)


def test_band_bbox_keeps_bands_that_are_inside_the_right_margin():
    prob = np.zeros((40, 100), np.float32)
    prob[10:20, 80:95] = 0.9  # near-right but inside the margin (cx=87 < 0.97*100)
    seg = _FakeSegmenter(prob)
    bbox = band_bbox(np.zeros((40, 100), np.float32), segmenter=seg, min_area=5, open_ksize=1)
    assert bbox == (80, 10, 95, 20)


def test_band_bbox_keeps_everything_when_filter_would_empty():
    prob = np.zeros((40, 100), np.float32)
    prob[10:20, 98:100] = 0.9  # only component sits in the margin
    seg = _FakeSegmenter(prob)
    bbox = band_bbox(np.zeros((40, 100), np.float32), segmenter=seg, min_area=5, open_ksize=1)
    assert bbox == (98, 10, 100, 20)