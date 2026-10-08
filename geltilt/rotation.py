"""Low-level image rotation helpers."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


def safe_crop_fraction(max_angle_deg: float) -> float:
    """Largest centred fraction of a rotated image guaranteed to be border-free.

    For a rotation of at most ``max_angle_deg`` the central fraction
    ``1 / (cos + sin)`` always lies inside the original frame, so projection
    scoring never sees the constant-filled corners.
    """
    a = np.radians(abs(max_angle_deg))
    return float(1.0 / (np.cos(a) + np.sin(a)))


def expanded_size(shape: Sequence[int], angle_deg: float) -> Tuple[int, int]:
    """Output (width, height) needed to contain the whole image after rotation."""
    h, w = int(shape[0]), int(shape[1])
    a = np.radians(angle_deg)
    c, s = abs(np.cos(a)), abs(np.sin(a))
    return int(round(w * c + h * s)), int(round(h * c + w * s))


def _fill_value(fill, channels: int):
    if np.isscalar(fill):
        return (float(fill),) * channels if channels > 1 else float(fill)
    return tuple(fill)


def rotate_image(
    arr: np.ndarray,
    angle_deg: float,
    center: Optional[Tuple[float, float]] = None,
    output_size: Optional[Tuple[int, int]] = None,
    fill: float = 0.0,
) -> np.ndarray:
    """Rotate an array by ``angle_deg`` (positive = counter-clockwise).

    Preserves dtype and channel count. ``center`` defaults to the image centre.
    """
    arr = np.asarray(arr)
    h, w = arr.shape[:2]
    if center is None:
        center = ((w - 1) / 2.0, (h - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
    if output_size is None:
        out_w, out_h = w, h
    else:
        out_w, out_h = int(output_size[0]), int(output_size[1])
        matrix[0, 2] += (out_w - 1) / 2.0 - center[0]
        matrix[1, 2] += (out_h - 1) / 2.0 - center[1]
    channels = 1 if arr.ndim == 2 else arr.shape[2]
    return cv2.warpAffine(
        arr,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=_fill_value(fill, channels),
    )