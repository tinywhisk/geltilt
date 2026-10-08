from __future__ import annotations

import cv2
import numpy as np
import pytest


def make_gel(h: int = 640, w: int = 480, n_lanes: int = 6, seed: int = 0) -> np.ndarray:
    """Build an upright synthetic gel: bright bands/smears on a dark background."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w), np.float32)
    for li, x0 in enumerate(np.linspace(60, w - 60, n_lanes)):
        img += 0.10 * np.exp(-((xx - x0) ** 2) / (2 * 3.0 ** 2))
        for yb in range(70, h - 50, 55 + li * 3):
            img += (1.2 + 0.4 * li) * np.exp(
                -((xx - x0) ** 2) / (2 * 6.0 ** 2) - ((yy - yb) ** 2) / (2 * 4.0 ** 2)
            )
    img += rng.normal(0, 0.02, img.shape).astype(np.float32)
    img = np.clip(img, 0, None)
    return cv2.GaussianBlur(img, (3, 3), 0.8)


@pytest.fixture
def gel_factory():
    return make_gel