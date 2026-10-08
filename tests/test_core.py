from __future__ import annotations

import numpy as np
import pytest

from geltilt.core import auto_correct, correct_rotation, crop_to_content, estimate_rotation
from geltilt.projection import orientation_score
from geltilt.rotation import rotate_image

from conftest import make_gel


@pytest.mark.parametrize("tilt", [-8.0, -4.0, -1.5, 1.5, 4.0, 8.0])
def test_estimate_recovers_known_tilt(gel_factory, tilt):
    rotated = rotate_image(gel_factory(seed=1), tilt)
    est = estimate_rotation(rotated)
    assert est.polarity == "bright"
    assert est.angle == pytest.approx(-tilt, abs=0.3)


def test_upright_gel_needs_no_correction(gel_factory):
    est = estimate_rotation(gel_factory(seed=2))
    assert abs(est.angle) < 0.3


def test_correction_increases_projection_sharpness(gel_factory):
    rotated = rotate_image(gel_factory(seed=3), 6.0)
    est = estimate_rotation(rotated)
    corrected = correct_rotation(rotated, est.angle)
    assert orientation_score(corrected, "tv2") > orientation_score(rotated, "tv2")


def test_correct_rotation_preserves_size_by_default(gel_factory):
    img = gel_factory()
    out = correct_rotation(img, 5.0)
    assert out.shape == img.shape
    assert out.dtype == img.dtype


def test_correct_rotation_expand_grows_canvas(gel_factory):
    img = gel_factory()
    out = correct_rotation(img, 5.0, expand=True)
    assert out.shape[0] >= img.shape[0]
    assert out.shape[1] >= img.shape[1]


def test_auto_correct_returns_corrected_and_estimate(gel_factory):
    rotated = rotate_image(gel_factory(seed=4), -7.0)
    corrected, est = auto_correct(rotated)
    assert corrected.shape == rotated.shape
    assert est.angle == pytest.approx(7.0, abs=0.3)


def test_crop_to_content_shrinks_around_bands():
    img = np.zeros((100, 100), np.float32)
    img[40:60, 30:70] = 200.0
    cropped = crop_to_content(img, margin_frac=0.0, min_margin=0)
    assert cropped.shape[0] < img.shape[0]
    assert cropped.shape[1] < img.shape[1]


def test_estimate_rejects_bad_method():
    with pytest.raises(ValueError):
        estimate_rotation(np.zeros((32, 32), np.float32), method="bogus")


def test_estimate_roi_optional(gel_factory):
    rotated = rotate_image(gel_factory(seed=5), 3.0)
    with_roi = estimate_rotation(rotated, use_roi=True)
    without_roi = estimate_rotation(rotated, use_roi=False)
    assert with_roi.angle == pytest.approx(without_roi.angle, abs=0.5)