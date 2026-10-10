"""Front end for the lane-based rotation estimator: GelGenie band mask -> band centroids.

The estimator in :mod:`geltilt.ladders` reads no pixels - only band centroids ``(x, y)``. This
module produces those centroids from an image: run the GelGenie ONNX band segmenter, threshold the
band probability, take connected components, and return each component's centroid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .bandmask import BandSegmenter


def band_centroids(
    image: np.ndarray,
    segmenter: Optional[BandSegmenter] = None,
    model_path: Optional[Path] = None,
    threshold: float = 0.5,
    min_area: int = 30,
    open_ksize: int = 3,
) -> np.ndarray:
    """Return an ``(N, 2)`` float array of band centroids ``(x, y)`` in image pixels.

    Follows BAND_DETECTION.md §5: threshold the band probability at 0.5, morphological opening
    (``open_ksize``) to drop single-pixel responses, 8-connected components, keep those with area
    ``>= min_area``, and take each component's centroid.
    """
    if segmenter is None:
        segmenter = BandSegmenter(model_path=model_path)
    prob = segmenter.probability(image)
    mask = (prob > float(threshold)).astype(np.uint8)
    if open_ksize and open_ksize > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_ksize, open_ksize), np.uint8))
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8),
                                                                    connectivity=8)
    points = [centroids[i] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= int(min_area)]
    if not points:
        return np.zeros((0, 2), float)
    return np.asarray(points, float)


_DEFAULT_SEGMENTERS: dict = {}


def default_segmenter(model_path: Optional[Path] = None) -> BandSegmenter:
    """A cached :class:`BandSegmenter` so the ONNX model is loaded once per process."""
    key = str(Path(model_path)) if model_path else "__default__"
    seg = _DEFAULT_SEGMENTERS.get(key)
    if seg is None:
        seg = BandSegmenter(model_path=model_path)
        _DEFAULT_SEGMENTERS[key] = seg
    return seg