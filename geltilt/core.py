"""High-level rotation estimation and correction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

from .imageio import to_gray
from .preprocess import content_bbox, content_mask, normalize_signal, to_signal
from .projection import MAXIMIZE, orientation_score
from .rotation import expanded_size, rotate_image, safe_crop_fraction


@dataclass
class RotationEstimate:
    """Result of a rotation estimation run."""

    angle: float
    score: float
    method: str
    polarity: str
    max_angle: float
    coarse_step: float
    fine_step: float
    roi: Optional[Tuple[int, int, int, int]] = None
    curve: np.ndarray = field(default_factory=lambda: np.empty((2, 0)))

    def to_dict(self) -> dict:
        return {
            "angle": round(float(self.angle), 4),
            "score": float(self.score),
            "method": self.method,
            "polarity": self.polarity,
            "max_angle": self.max_angle,
            "coarse_step": self.coarse_step,
            "fine_step": self.fine_step,
            "roi": list(self.roi) if self.roi is not None else None,
        }


def _prepare(
    gray: np.ndarray,
    polarity: str,
    use_roi: bool,
    downsample_to: Optional[int],
    mask_mode: Optional[str] = None,
    segmenter=None,
) -> Tuple[np.ndarray, str, Optional[Tuple[int, int, int, int]], Optional[np.ndarray]]:
    signal, pol, _ = to_signal(gray, polarity=polarity)
    signal = normalize_signal(signal)
    band = None
    if mask_mode:
        from .bandmask import band_probability

        band = band_probability(gray, segmenter=segmenter)
    h, w = signal.shape
    if downsample_to and max(h, w) > downsample_to:
        scale = downsample_to / float(max(h, w))
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        signal = cv2.resize(signal, new_size, interpolation=cv2.INTER_AREA)
        if band is not None:
            band = cv2.resize(band, new_size, interpolation=cv2.INTER_AREA)
    roi = None
    if use_roi:
        bbox = content_bbox(content_mask(signal))
        if bbox is not None:
            y0, y1, x0, x1 = bbox
            spans_frame = (y1 - y0) >= 0.85 * h and (x1 - x0) >= 0.85 * w
            if not spans_frame and (y1 - y0) > 8 and (x1 - x0) > 8:
                signal = signal[y0:y1, x0:x1]
                if band is not None:
                    band = band[y0:y1, x0:x1]
                roi = bbox
    if band is not None and mask_mode == "bands":
        signal = signal * band
    return signal, pol, roi, band


def _scan(
    signal: np.ndarray,
    angles: np.ndarray,
    method: str,
    log: bool,
    max_angle: float,
    band: Optional[np.ndarray] = None,
    slab_suppress: bool = False,
    band_thr: float = 0.5,
    min_band_px: int = 2,
) -> np.ndarray:
    h, w = signal.shape
    frac = safe_crop_fraction(max_angle)
    ch, cw = max(1, int(h * frac)), max(1, int(w * frac))
    oy, ox = (h - ch) // 2, (w - cw) // 2
    center = ((w - 1) / 2.0, (h - 1) / 2.0)
    scores = np.empty(angles.size, dtype=np.float64)
    for i, angle in enumerate(angles):
        rotated = rotate_image(signal, float(angle), center=center, fill=0.0)
        rotated = rotated[oy:oy + ch, ox:ox + cw]
        if slab_suppress and band is not None:
            rb = rotate_image(band, float(angle), center=center, fill=0.0)[oy:oy + ch, ox:ox + cw]
            mass = (rb > band_thr).sum(axis=0)
            bg = np.median(rotated, axis=0, keepdims=True)
            keep = (mass >= min_band_px)[None, :]
            rotated = np.where(keep, rotated, np.clip(rotated - bg, 0.0, None))
        scores[i] = orientation_score(rotated, method=method, log=log)
    return scores


def estimate_rotation(
    image: np.ndarray,
    max_angle: float = 15.0,
    coarse_step: float = 0.5,
    fine_step: float = 0.05,
    method: str = "tv2",
    polarity: str = "auto",
    log: bool = True,
    use_roi: bool = True,
    downsample_to: Optional[int] = 1200,
    band_mask: bool = False,
    slab_suppress: bool = False,
    segmenter=None,
) -> RotationEstimate:
    """Estimate the rotation (degrees) that makes lanes vertical.

    The returned ``angle`` is the correction to apply, i.e. pass it to
    :func:`correct_rotation` / :func:`rotate_image`. It is found by maximising the
    sharpness of the vertical projection of the band smears over a coarse angle
    sweep, then refining with a fine sweep.

    With ``band_mask=True`` the projection is restricted to GelGenie-segmented
    band pixels (aggressive). With ``slab_suppress=True`` the smear signal is kept
    but the gel slab edge is removed by column-detrending only those columns that
    contain no band pixels -- fixing slab-edge lock-on without discarding smears.
    """
    if method not in MAXIMIZE:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(MAXIMIZE)}")
    if max_angle <= 0:
        raise ValueError("max_angle must be positive")

    mask_mode = "bands" if band_mask else "slab" if slab_suppress else None
    gray = to_gray(image)
    work, pol, roi, band = _prepare(
        gray, polarity, use_roi, downsample_to, mask_mode=mask_mode, segmenter=segmenter
    )

    maximize = MAXIMIZE[method]
    scan_kwargs = dict(band=band, slab_suppress=slab_suppress)

    coarse = np.arange(-max_angle, max_angle + coarse_step / 2, coarse_step)
    coarse_scores = _scan(work, coarse, method, log, max_angle, **scan_kwargs)
    best_coarse = float(coarse[int(np.argmax(coarse_scores) if maximize else np.argmin(coarse_scores))])

    lo = max(-max_angle, best_coarse - coarse_step)
    hi = min(max_angle, best_coarse + coarse_step)
    fine = np.arange(lo, hi + fine_step / 2, fine_step)
    fine_scores = _scan(work, fine, method, log, max_angle, **scan_kwargs)
    best_idx = int(np.argmax(fine_scores) if maximize else np.argmin(fine_scores))
    angle = float(np.clip(fine[best_idx], -max_angle, max_angle))

    curve = np.vstack([np.concatenate([coarse, fine]),
                       np.concatenate([coarse_scores, fine_scores])])
    return RotationEstimate(
        angle=round(angle, 3),
        score=float(fine_scores[best_idx]),
        method=method,
        polarity=pol,
        max_angle=float(max_angle),
        coarse_step=float(coarse_step),
        fine_step=float(fine_step),
        roi=roi,
        curve=curve,
    )


def _default_fill(arr: np.ndarray):
    if arr.ndim == 2:
        return float(np.median(arr))
    return tuple(float(np.median(arr[:, :, c])) for c in range(arr.shape[2]))


def crop_to_content(
    arr: np.ndarray, margin_frac: float = 0.05, min_margin: int = 10
) -> np.ndarray:
    """Crop an image to the bounding box of its band content."""
    signal, _, _ = to_signal(to_gray(arr))
    bbox = content_bbox(content_mask(signal), margin_frac=margin_frac, min_margin=min_margin)
    if bbox is None:
        return arr
    y0, y1, x0, x1 = bbox
    return arr[y0:y1, x0:x1]


def correct_rotation(
    image: np.ndarray,
    angle: float,
    expand: bool = False,
    fill: Optional[float] = None,
    center: Optional[Tuple[float, float]] = None,
    crop_to_content_flag: bool = False,
) -> np.ndarray:
    """Apply a rotation correction, preserving dtype, channels and (by default) size."""
    arr = np.asarray(image)
    h, w = arr.shape[:2]
    if center is None:
        center = ((w - 1) / 2.0, (h - 1) / 2.0)
    if fill is None:
        fill = _default_fill(arr)
    out_size = expanded_size(arr.shape, angle) if expand else (w, h)
    corrected = rotate_image(arr, angle, center=center, output_size=out_size, fill=fill)
    if crop_to_content_flag:
        corrected = crop_to_content(corrected)
    return corrected


def auto_correct(
    image: np.ndarray,
    expand: bool = False,
    fill: Optional[float] = None,
    crop: bool = False,
    **estimate_kwargs,
) -> Tuple[np.ndarray, RotationEstimate]:
    """Estimate the tilt and return the corrected image plus the estimate."""
    estimate = estimate_rotation(image, **estimate_kwargs)
    corrected = correct_rotation(
        image, estimate.angle, expand=expand, fill=fill, crop_to_content_flag=crop
    )
    return corrected, estimate

def estimate_rotation_ladders(
    image: np.ndarray,
    segmenter=None,
    model_path=None,
    threshold: float = 0.5,
    min_area: int = 30,
) -> dict:
    """The lane-based rotation estimator: GelGenie band mask -> centroids -> `ladders.estimate`.

    Returns the ``ladders.estimate`` result dict (``ok``, ``angle``, ``se``, ``reason``,
    ``warnings``, ``ladders``, ...). ``angle`` is passed as-is to :func:`correct_rotation`.
    """
    from . import ladders
    from .banddetect import band_centroids, default_segmenter

    seg = segmenter or default_segmenter(model_path)
    xy = band_centroids(image, segmenter=seg, threshold=threshold, min_area=min_area)
    return ladders.estimate(xy)