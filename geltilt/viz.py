"""Presentation helpers: auto-contrast, vertical guides and side-by-side figures."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from .imageio import save_array, to_gray
from .preprocess import content_bbox, content_mask, robust_limits, to_signal, to_u8

PathLike = Union[str, Path]


def auto_contrast(
    image: np.ndarray,
    plo: float = 0.5,
    phi: float = 99.5,
    gamma: float = 1.0,
) -> np.ndarray:
    """Stretch brightness/contrast using robust limits, returning uint8 grayscale.

    Thin wrapper over :func:`geltilt.preprocess.robust_limits` and
    :func:`geltilt.preprocess.to_u8`.
    """
    gray = to_gray(image)
    lo, hi = robust_limits(gray, plo, phi)
    return to_u8(gray, lo, hi, gamma)


def content_window(
    image: np.ndarray,
    percentile: float = 99.5,
    margin_frac: float = 0.25,
    min_margin: int = 40,
) -> Optional[tuple]:
    """Bounding box (y0, y1, x0, x1) framing the visible band content."""
    gray = to_gray(image)
    signal, _, _ = to_signal(gray)
    thr = float(np.percentile(signal, percentile))
    mask = signal >= thr if thr > 0 else content_mask(signal)
    return content_bbox(mask, margin_frac=margin_frac, min_margin=min_margin)


def _apply_window(image: np.ndarray, window: Optional[tuple]) -> np.ndarray:
    gray = to_gray(image)
    if window is None:
        return gray
    y0, y1, x0, x1 = window
    return gray[y0:y1, x0:x1]


def display_crop(
    image: np.ndarray,
    percentile: float = 99.5,
    margin_frac: float = 0.25,
    min_margin: int = 40,
) -> np.ndarray:
    """Crop to the visible band content so the gel fills the frame."""
    return _apply_window(image, content_window(image, percentile, margin_frac, min_margin))


def _resize_to_height(gray_u8: np.ndarray, height: int) -> np.ndarray:
    h, w = gray_u8.shape
    scale = height / float(h)
    return cv2.resize(gray_u8, (max(1, int(round(w * scale))), height), interpolation=cv2.INTER_AREA)


def _draw_text_box(
    canvas: np.ndarray,
    text: str,
    origin: tuple,
    font_scale: float,
    color: tuple,
    thickness: int = 1,
    pad: int = 6,
    bg: Optional[tuple] = None,
) -> None:
    x, y = origin
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    if bg is not None:
        cv2.rectangle(canvas, (x - pad, y - th - pad), (x + tw + pad, y + base + pad), bg, -1)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def draw_vertical_guides(
    panel_bgr: np.ndarray, step: int, color: tuple = (0, 255, 255), thickness: int = 1
) -> np.ndarray:
    """Overlay evenly spaced vertical guide lines on a BGR panel."""
    h, w = panel_bgr.shape[:2]
    for x in range(step, w, step):
        cv2.line(panel_bgr, (x, 0), (x, h), color, thickness, cv2.LINE_AA)
    return panel_bgr


def _to_display_raw(image: np.ndarray) -> np.ndarray:
    """Cast an image to uint8 for display without any contrast stretch."""
    gray = to_gray(image)
    peak = float(gray.max()) if gray.size else 0.0
    if peak <= 255.0:
        return np.clip(gray, 0, 255).astype(np.uint8)
    return np.clip(gray / peak * 255.0, 0, 255).astype(np.uint8)


def _compose_two_panel(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    left_title: str,
    right_title: str,
    info_lines: list,
    panel_height: int,
) -> np.ndarray:
    gap = 14
    title_h = 40
    info_h = 30 * max(1, len(info_lines)) + 24
    total_w = left_bgr.shape[1] + gap + right_bgr.shape[1]
    total_h = info_h + title_h + panel_height

    canvas = np.full((total_h, total_w, 3), 24, np.uint8)
    cv2.rectangle(canvas, (0, 0), (total_w, info_h), (12, 12, 12), -1)
    cv2.line(canvas, (0, info_h - 1), (total_w, info_h - 1), (0, 160, 0), 2)

    y = 30
    for text, color, font_scale in info_lines:
        _draw_text_box(canvas, text, (16, y), font_scale, color, 1)
        y += 30

    body_y = info_h
    _draw_text_box(canvas, left_title, (10, body_y + 27), 0.7, (0, 200, 255), 2)
    _draw_text_box(canvas, right_title, (left_bgr.shape[1] + gap + 10, body_y + 27), 0.7, (0, 255, 0), 2)

    canvas[body_y + title_h:body_y + title_h + panel_height, 0:left_bgr.shape[1]] = left_bgr
    canvas[body_y + title_h:body_y + title_h + panel_height,
           left_bgr.shape[1] + gap:left_bgr.shape[1] + gap + right_bgr.shape[1]] = right_bgr
    return canvas


def render_comparison(
    original: np.ndarray,
    corrected: np.ndarray,
    angle: float,
    info: Optional[dict] = None,
    source_name: Optional[str] = None,
    panel_height: int = 760,
    guide_step: int = 90,
    plo: float = 0.5,
    phi: float = 99.5,
    gamma: float = 1.0,
) -> np.ndarray:
    """Build a BGR side-by-side figure: original vs corrected, with guides and info.

    Both panels are auto-contrast-stretched, cropped to the same window (derived
    from the original) and share the same display height so lane tilt before and
    after the correction is directly comparable.
    """
    window = content_window(original)
    orig_crop = _apply_window(original, window)
    corr_crop = _apply_window(corrected, window)

    lo, hi = robust_limits(to_gray(original), plo, phi)
    left = _resize_to_height(to_u8(orig_crop, lo, hi, gamma), panel_height)
    right = _resize_to_height(to_u8(corr_crop, lo, hi, gamma), panel_height)

    left = draw_vertical_guides(cv2.cvtColor(left, cv2.COLOR_GRAY2BGR), guide_step)
    right = draw_vertical_guides(cv2.cvtColor(right, cv2.COLOR_GRAY2BGR), guide_step)

    info_line = info or {}
    info_lines = []
    if source_name:
        info_lines.append((f"file: {source_name}", (200, 200, 200), 0.6))
    info_lines.append(
        (f"geltilt  |  detected correction: {angle:+.2f} deg  |  method: {info_line.get('method', '?')}"
         f"  |  polarity: {info_line.get('polarity', '?')}", (0, 255, 0), 0.62)
    )
    info_lines.append(
        (f"max_angle: {info_line.get('max_angle', '?')}  |  step: {info_line.get('coarse_step', '?')}/"
         f"{info_line.get('fine_step', '?')}  |  guide spacing: {guide_step}px", (150, 200, 255), 0.52)
    )

    return _compose_two_panel(
        left, right,
        "BEFORE  (lanes tilted)",
        f"AFTER  (corrected {angle:+.2f} deg -> lanes vertical)",
        info_lines, panel_height,
    )


def render_preprocessing_comparison(
    image: np.ndarray,
    plo: float = 0.5,
    phi: float = 99.5,
    gamma: float = 1.0,
    source_name: Optional[str] = None,
    panel_height: int = 760,
    guide_step: int = 90,
    guides: bool = True,
) -> np.ndarray:
    """Build a BGR side-by-side figure: raw image vs preprocessed image.

    Left = raw values cast to uint8 (no stretch); right = ``robust_limits`` +
    ``to_u8``. Both use the same content crop and display height.
    """
    window = content_window(image)
    crop = _apply_window(image, window)
    lo, hi = robust_limits(to_gray(image), plo, phi)

    raw_u8 = _resize_to_height(_to_display_raw(crop), panel_height)
    prep_u8 = _resize_to_height(to_u8(crop, lo, hi, gamma), panel_height)

    left = cv2.cvtColor(raw_u8, cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(prep_u8, cv2.COLOR_GRAY2BGR)
    if guides:
        left = draw_vertical_guides(left, guide_step)
        right = draw_vertical_guides(right, guide_step)

    info_lines = []
    if source_name:
        info_lines.append((f"file: {source_name}", (200, 200, 200), 0.6))
    info_lines.append(
        (f"preprocessing: robust_limits(plo={plo}, phi={phi}) -> to_u8(gamma={gamma})", (0, 255, 0), 0.6)
    )
    factor = 255.0 / max(hi - lo, 1.0)
    info_lines.append(
        (f"window (full frame) lo={lo:.1f} hi={hi:.1f}  x{factor:.1f}  |  raw mean={raw_u8.mean():.1f} "
         f"-> preprocessed mean={prep_u8.mean():.1f}", (150, 200, 255), 0.55)
    )

    return _compose_two_panel(
        left, right,
        "RAW  (no preprocessing)",
        f"PREPROCESSED  (robust_limits {plo}/{phi}, gamma={gamma})",
        info_lines, panel_height,
    )


def save_comparison(path: PathLike, figure_bgr: np.ndarray) -> None:
    """Save a BGR comparison figure to disk (encoded as PNG/TIFF via RGB)."""
    save_array(path, figure_bgr[:, :, ::-1])