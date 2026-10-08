from __future__ import annotations

import numpy as np
import pytest

from geltilt.projection import (
    MAXIMIZE,
    orientation_score,
    sharpness,
    vertical_projection,
)
from geltilt.rotation import rotate_image

from conftest import make_gel


def test_vertical_projection_sums_to_one():
    signal = np.abs(np.random.default_rng(0).normal(size=(20, 30))).astype(np.float32)
    profile = vertical_projection(signal)
    assert pytest.approx(profile.sum(), rel=1e-9) == 1.0


def test_projection_sharpest_when_lanes_vertical():
    base = make_gel()
    tilted = rotate_image(base, 7.0)
    assert orientation_score(base, "tv2") > orientation_score(tilted, "tv2")


def test_sharpness_methods_ordering():
    base = make_gel()
    tilted = rotate_image(base, 6.0)
    for method, maximize in MAXIMIZE.items():
        s_base = orientation_score(base, method)
        s_tilt = orientation_score(tilted, method)
        if maximize:
            assert s_base > s_tilt, method
        else:
            assert s_base < s_tilt, method


def test_sharpness_unknown_method():
    with pytest.raises(ValueError):
        sharpness(np.ones(10), "nope")