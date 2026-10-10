from __future__ import annotations

from pathlib import Path

import pytest

from geltilt.core import correct_rotation, estimate_rotation
from geltilt.imageio import load_gray
from geltilt.projection import orientation_score

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "samples"
SAMPLES = sorted(list(SAMPLE_DIR.glob("*.Tif")) + list(SAMPLE_DIR.glob("*.tif")))
SUBSET = SAMPLES[:6]
ORIGINAL = SAMPLE_DIR / "2026 0508 PPA6-1 PPA6-2 13772.Tif"


@pytest.mark.skipif(not SAMPLES, reason="no sample gel images available")
@pytest.mark.parametrize("path", SUBSET, ids=lambda p: p.name)
def test_sample_estimate_is_bounded_and_preserves_shape(path):
    gray = load_gray(path)
    est = estimate_rotation(gray)
    assert -15.0 <= est.angle <= 15.0
    assert est.polarity in ("bright", "dark")
    corrected = correct_rotation(gray, est.angle)
    assert corrected.shape == gray.shape


@pytest.mark.skipif(not ORIGINAL.exists(), reason="original sample not present")
def test_original_sample_angle_and_improvement():
    gray = load_gray(ORIGINAL)
    est = estimate_rotation(gray)
    assert -5.0 < est.angle < -1.0
    corrected = correct_rotation(gray, est.angle)
    assert orientation_score(corrected, "tv2") >= orientation_score(gray, "tv2")