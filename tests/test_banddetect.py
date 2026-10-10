from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from geltilt.banddetect import band_centroids
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