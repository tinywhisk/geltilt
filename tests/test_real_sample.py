from __future__ import annotations

from pathlib import Path

import pytest

from geltilt.core import correct_rotation, estimate_rotation
from geltilt.imageio import load_gray
from geltilt.projection import orientation_score

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "samples"
SAMPLES = sorted(list(SAMPLE_DIR.glob("*.Tif")) + list(SAMPLE_DIR.glob("*.tif")))


@pytest.mark.skipif(not SAMPLES, reason="no sample gel images available")
def test_sample_rotation_is_stable_and_small():
    gray = load_gray(SAMPLES[0])
    est = estimate_rotation(gray)
    assert est.polarity == "bright"
    assert -5.0 < est.angle < -1.0


@pytest.mark.skipif(not SAMPLES, reason="no sample gel images available")
def test_sample_correction_improves_projection():
    gray = load_gray(SAMPLES[0])
    est = estimate_rotation(gray)
    corrected = correct_rotation(gray, est.angle)
    assert orientation_score(corrected, "tv2") >= orientation_score(gray, "tv2")