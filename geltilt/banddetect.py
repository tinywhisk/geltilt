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


def _kept_components(
    image: np.ndarray,
    segmenter: BandSegmenter,
    threshold: float,
    min_area: int,
    open_ksize: int,
):
    """Return ``(stats, centroids, kept)`` for the band mask, per BAND_DETECTION.md §5.

    Thresholds the band probability, applies morphological opening (``open_ksize``)
    to drop single-pixel responses, labels 8-connected components and keeps the
    indices whose area is ``>= min_area``.
    """
    prob = segmenter.probability(image)
    mask = (prob > float(threshold)).astype(np.uint8)
    if open_ksize and open_ksize > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_ksize, open_ksize), np.uint8))
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8),
                                                                    connectivity=8)
    kept = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= int(min_area)]
    return stats, centroids, kept


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
    _stats, centroids, kept = _kept_components(image, segmenter, threshold, min_area, open_ksize)
    if not kept:
        return np.zeros((0, 2), float)
    return np.asarray([centroids[i] for i in kept], float)


def band_mask(
    image: np.ndarray,
    segmenter: Optional[BandSegmenter] = None,
    model_path: Optional[Path] = None,
    threshold: float = 0.5,
    open_ksize: int = 3,
) -> np.ndarray:
    """Binary uint8 band mask (probability thresheld + opened), native size.

    No area filtering — callers that need a component floor apply it themselves
    (the composite lane model drops blobs below its own ``MIN_BLOB_AREA``).
    """
    if segmenter is None:
        segmenter = BandSegmenter(model_path=model_path)
    prob = segmenter.probability(image)
    mask = (prob > float(threshold)).astype(np.uint8)
    if open_ksize and open_ksize > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_ksize, open_ksize), np.uint8))
    return mask


def bbox_from_mask(
    mask: np.ndarray,
    min_area: int = 30,
    right_margin_frac: float = 0.03,
) -> Optional[Tuple[int, int, int, int]]:
    """Union bounding box ``(x0, y0, x1, y1)`` over kept components of ``mask``.

    ``right_margin_frac`` drops components whose centroid falls in the rightmost
    fraction of the frame (GelGenie's fixed far-right debris blob); the filter is
    skipped if it would remove every component.
    """
    m = (np.asarray(mask) > 0).astype(np.uint8)
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
    kept = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= int(min_area)]
    if not kept:
        return None
    width = int(m.shape[1])
    if right_margin_frac > 0 and width > 0:
        cut = (1.0 - float(right_margin_frac)) * width
        filtered = [i for i in kept if float(centroids[i][0]) < cut]
        if filtered:
            kept = filtered
    x0 = min(int(stats[i, cv2.CC_STAT_LEFT]) for i in kept)
    y0 = min(int(stats[i, cv2.CC_STAT_TOP]) for i in kept)
    x1 = max(int(stats[i, cv2.CC_STAT_LEFT]) + int(stats[i, cv2.CC_STAT_WIDTH]) for i in kept)
    y1 = max(int(stats[i, cv2.CC_STAT_TOP]) + int(stats[i, cv2.CC_STAT_HEIGHT]) for i in kept)
    return (x0, y0, x1, y1)


def band_bbox(
    image: np.ndarray,
    segmenter: Optional[BandSegmenter] = None,
    model_path: Optional[Path] = None,
    threshold: float = 0.5,
    min_area: int = 30,
    open_ksize: int = 3,
    right_margin_frac: float = 0.03,
) -> Optional[Tuple[int, int, int, int]]:
    """Union bounding box over the kept components of :func:`band_mask`."""
    mask = band_mask(image, segmenter=segmenter, model_path=model_path,
                     threshold=threshold, open_ksize=open_ksize)
    return bbox_from_mask(mask, min_area=min_area, right_margin_frac=right_margin_frac)


_DEFAULT_SEGMENTERS: dict = {}


def default_segmenter(model_path: Optional[Path] = None) -> BandSegmenter:
    """A cached :class:`BandSegmenter` so the ONNX model is loaded once per process."""
    key = str(Path(model_path)) if model_path else "__default__"
    seg = _DEFAULT_SEGMENTERS.get(key)
    if seg is None:
        seg = BandSegmenter(model_path=model_path)
        _DEFAULT_SEGMENTERS[key] = seg
    return seg