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
) -> Tuple[np.ndarray, str, Optional[Tuple[int, int, int, int]]]:
    signal, pol, _ = to_signal(gray, polarity=polarity)
    signal = normalize_signal(signal)
    h, w = signal.shape
    if downsample_to and max(h, w) > downsample_to:
        scale = downsample_to / float(max(h, w))
        signal = cv2.resize(
            signal,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    roi = None
    if use_roi:
        bbox = content_bbox(content_mask(signal))
        if bbox is not None:
            y0, y1, x0, x1 = bbox
            spans_frame = (y1 - y0) >= 0.85 * h and (x1 - x0) >= 0.85 * w
            if not spans_frame and (y1 - y0) > 8 and (x1 - x0) > 8:
                signal = signal[y0:y1, x0:x1]
                roi = bbox
    return signal, pol, roi


def _scan(
    signal: np.ndarray,
    angles: np.ndarray,
    method: str,
    log: bool,
    max_angle: float,
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
) -> RotationEstimate:
    """Estimate the rotation (degrees) that makes lanes vertical.

    The returned ``angle`` is the correction to apply, i.e. pass it to
    :func:`correct_rotation` / :func:`rotate_image`. It is found by maximising the
    sharpness of the vertical projection of the band smears over a coarse angle
    sweep, then refining with a fine sweep.
    """
    if method not in MAXIMIZE:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(MAXIMIZE)}")
    if max_angle <= 0:
        raise ValueError("max_angle must be positive")

    gray = to_gray(image)
    work, pol, roi = _prepare(gray, polarity, use_roi, downsample_to)

    maximize = MAXIMIZE[method]

    coarse = np.arange(-max_angle, max_angle + coarse_step / 2, coarse_step)
    coarse_scores = _scan(work, coarse, method, log, max_angle)
    best_coarse = float(coarse[int(np.argmax(coarse_scores) if maximize else np.argmin(coarse_scores))])

    lo = max(-max_angle, best_coarse - coarse_step)
    hi = min(max_angle, best_coarse + coarse_step)
    fine = np.arange(lo, hi + fine_step / 2, fine_step)
    fine_scores = _scan(work, fine, method, log, max_angle)
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