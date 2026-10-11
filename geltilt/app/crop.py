"""Crop-rectangle geometry in the corrected (display) frame.

The viewer maps a source pixel ``p`` to the screen with
``screen = center + scale * M(angle) (p - anchor)`` (see
:func:`geltilt.labeler.source_to_screen`). The corrected/display frame drops
``center`` and ``scale`` and keeps ``q = M(angle) (p - anchor)``, so a rectangle
axis-aligned in ``q`` is always axis-aligned (upright) on screen regardless of
the view's zoom or pan. Crop rectangles are stored in this ``q`` frame.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np

Rect = Tuple[float, float, float, float]

MIN_CROP_SIZE = 8.0


def rotate_mask(mask: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a binary mask about the image centre (same geometry as correct_rotation).

    Nearest-neighbour so the mask stays binary; used to bring the cached source-frame
    band mask into the corrected frame that the export slices.
    """
    m = (np.asarray(mask) > 0).astype(np.uint8)
    h, w = m.shape[:2]
    center = ((w - 1) / 2.0, (h - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
    out = cv2.warpAffine(m, matrix, (w, h), flags=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out > 0


def pad_rect(rect: Rect, pad: float) -> Rect:
    """Grow ``rect`` by ``pad`` pixels on every side."""
    x0, y0, x1, y1 = rect
    return (x0 - pad, y0 - pad, x1 + pad, y1 + pad)


def source_bbox_to_display_rect(bbox: Rect, angle_deg: float, anchor: Tuple[float, float]) -> Rect:
    """Map a source-axis-aligned bbox into the corrected frame (anchor-relative).

    The four source corners are rotated into ``q = M(angle) (p - anchor)`` and
    their axis-aligned bounds are returned.
    """
    th = math.radians(angle_deg)
    c, s = math.cos(th), math.sin(th)
    x0, y0, x1, y1 = bbox
    qs = []
    for px, py in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
        dx, dy = px - anchor[0], py - anchor[1]
        qs.append((c * dx + s * dy, -s * dx + c * dy))
    xs = [q[0] for q in qs]
    ys = [q[1] for q in qs]
    return (min(xs), min(ys), max(xs), max(ys))


def screen_rect(rect: Rect, center: Tuple[float, float], scale: float) -> Rect:
    """Map a corrected-frame rect to screen coordinates."""
    x0, y0, x1, y1 = rect
    return (
        center[0] + scale * x0,
        center[1] + scale * y0,
        center[0] + scale * x1,
        center[1] + scale * y1,
    )


def screen_to_display(sx: float, sy: float, center: Tuple[float, float], scale: float) -> Tuple[float, float]:
    """Inverse of :func:`screen_rect` for a single point."""
    return ((sx - center[0]) / scale, (sy - center[1]) / scale)


def display_rect_to_corrected_box(
    rect: Rect,
    angle_lock_deg: float,
    anchor: Tuple[float, float],
    angle_now_deg: float,
    size: Tuple[int, int],
) -> Optional[Tuple[int, int, int, int]]:
    """Map a display-frame crop rect to a pixel box in the corrected image.

    The display frame is ``q = M(angle_lock)(p - anchor)`` with
    ``M = [[cos, sin], [-sin, cos]]`` — the same rotation used to render the
    viewer and by :func:`geltilt.core.correct_rotation`. The corrected image
    rotates the source about its centre by ``angle_now``. Returns an integer
    ``(x0, y0, x1, y1)`` box clamped to ``size`` (h, w), or ``None`` if empty.
    """
    h, w = int(size[0]), int(size[1])
    pcx, pcy = (w - 1) / 2.0, (h - 1) / 2.0
    tl = math.radians(angle_lock_deg)
    cl, sl = math.cos(tl), math.sin(tl)
    tn = math.radians(angle_now_deg)
    cn, sn = math.cos(tn), math.sin(tn)
    x0, y0, x1, y1 = rect
    pts = []
    for qx, qy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
        px = anchor[0] + cl * qx - sl * qy
        py = anchor[1] + sl * qx + cl * qy
        dx, dy = px - pcx, py - pcy
        pts.append((pcx + cn * dx + sn * dy, pcy - sn * dx + cn * dy))
    bx0 = max(0, int(math.floor(min(p[0] for p in pts))))
    by0 = max(0, int(math.floor(min(p[1] for p in pts))))
    bx1 = min(w, int(math.ceil(max(p[0] for p in pts))))
    by1 = min(h, int(math.ceil(max(p[1] for p in pts))))
    if bx1 <= bx0 or by1 <= by0:
        return None
    return (bx0, by0, bx1, by1)


def normalize_rect(rect: Rect, min_size: float = MIN_CROP_SIZE) -> Rect:
    """Order the corners and enforce a minimum extent on each axis."""
    x0, y0, x1, y1 = rect
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    if x1 - x0 < min_size:
        x1 = x0 + min_size
    if y1 - y0 < min_size:
        y1 = y0 + min_size
    return (x0, y0, x1, y1)