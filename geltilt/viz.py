"""Presentation helpers: auto-contrast, vertical guides and side-by-side figures."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from .imageio import save_array, to_gray
from .preprocess import content_bbox, content_mask, normalize_signal, robust_limits, to_signal, to_u8

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
    if info_line.get("status"):
        info_lines.append((str(info_line["status"]), (0, 215, 255), 0.55))

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


def _letterbox(img: np.ndarray, cw: int, ch: int, bg: tuple = (18, 18, 18)) -> np.ndarray:
    """Fit an image into a cw x ch BGR cell, preserving aspect, centred on bg."""
    h, w = img.shape[:2]
    scale = min(cw / float(w), ch / float(h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    if resized.ndim == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    out = np.full((ch, cw, 3), bg, np.uint8)
    oy, ox = (ch - nh) // 2, (cw - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = resized
    return out


def _draw_curve_panel(curve: np.ndarray, angle: float, cw: int, ch: int, maximize: bool = True) -> np.ndarray:
    """Draw the angle-scan score curve with the chosen angle marked."""
    panel = np.full((ch, cw, 3), 18, np.uint8)
    if curve.size == 0:
        _draw_text_box(panel, "no curve", (12, ch // 2), 0.5, (150, 200, 255))
        return panel
    order = np.argsort(curve[0])
    ang, score = curve[0][order], curve[1][order]
    pad_x, pad_y = 34, 46
    a_min, a_max = float(ang.min()), float(ang.max())
    s_min, s_max = float(score.min()), float(score.max())
    if a_max <= a_min:
        a_max = a_min + 1.0
    if s_max <= s_min:
        s_max = s_min + 1.0
    xs = np.interp(ang, (a_min, a_max), (pad_x, cw - pad_x)).astype(np.int32)
    ys = np.interp(score, (s_min, s_max), (ch - pad_y, pad_y)).astype(np.int32)
    pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
    cv2.polylines(panel, [pts], False, (0, 200, 255), 1, cv2.LINE_AA)

    best_idx = int(np.argmax(score) if maximize else np.argmin(score))
    cv2.circle(panel, (int(xs[best_idx]), int(ys[best_idx])), 4, (0, 255, 0), -1, cv2.LINE_AA)
    xa = int(np.interp(angle, (a_min, a_max), (pad_x, cw - pad_x)))
    cv2.line(panel, (xa, pad_y - 14), (xa, ch - pad_y), (0, 255, 0), 1, cv2.LINE_AA)
    _draw_text_box(panel, f"{angle:+.2f}", (xa + 5, 24), 0.45, (0, 255, 0))
    _draw_text_box(panel, "tv2", (6, 20), 0.45, (150, 200, 255))
    _draw_text_box(panel, f"angle (deg)  [{a_min:.0f} .. {a_max:.0f}]", (cw - 210, ch - 8), 0.42, (150, 200, 255))
    return panel


def render_pipeline(
    image: np.ndarray,
    estimate,
    source_name: Optional[str] = None,
    cell: tuple = (440, 440),
    plo: float = 0.5,
    phi: float = 99.5,
    gamma: float = 1.0,
    guide_step: int = 90,
) -> np.ndarray:
    """Visualise the full tilt-correction pipeline as a 2x3 stage montage.

    Panels: (1) input with ROI box, (2) background-subtracted signal,
    (3) ROI + preprocessing, (4) angle-scan score curve, (5) before, (6) after.
    """
    from .projection import MAXIMIZE

    cw, ch = cell
    gray = to_gray(image)
    signal, polarity, bg = to_signal(gray)
    signal_norm = normalize_signal(signal)
    window = content_window(gray)
    lo, hi = robust_limits(gray, plo, phi)
    factor = 255.0 / max(hi - lo, 1.0)

    input_bgr = cv2.cvtColor(to_u8(gray, lo, hi, gamma), cv2.COLOR_GRAY2BGR)
    if window is not None:
        y0, y1, x0, x1 = window
        cv2.rectangle(input_bgr, (x0, y0), (x1, y1), (0, 0, 255), 2)
    p_input = _letterbox(input_bgr, cw, ch)

    sig_lo, sig_hi = robust_limits(signal_norm, plo, phi)
    p_signal = _letterbox(to_u8(signal_norm, sig_lo, sig_hi, gamma), cw, ch)

    crop = _apply_window(gray, window)
    p_roi = _letterbox(to_u8(crop, lo, hi, gamma), cw, ch)

    maximize = MAXIMIZE.get(getattr(estimate, "method", "tv2"), True)
    p_curve = _draw_curve_panel(np.asarray(estimate.curve), estimate.angle, cw, ch, maximize)

    before = draw_vertical_guides(cv2.cvtColor(to_u8(crop, lo, hi, gamma), cv2.COLOR_GRAY2BGR), guide_step)
    from .core import correct_rotation
    corrected_full = correct_rotation(image, estimate.angle)
    after = draw_vertical_guides(
        cv2.cvtColor(to_u8(_apply_window(corrected_full, window), lo, hi, gamma), cv2.COLOR_GRAY2BGR), guide_step
    )
    p_before = _letterbox(before, cw, ch)
    p_after = _letterbox(after, cw, ch)

    titles = [
        "1  INPUT  (raw, ROI box)",
        f"2  SIGNAL  (bg-subtracted, polarity={polarity})",
        f"3  ROI + PREPROCESS  (lo={lo:.0f} hi={hi:.0f}, x{factor:.1f})",
        f"4  ANGLE SCAN  ({estimate.method}, argmax)",
        "5  BEFORE  (lanes tilted)",
        f"6  AFTER  (corrected {estimate.angle:+.2f} deg)",
    ]
    panels = [p_input, p_signal, p_roi, p_curve, p_before, p_after]

    gap, title_h, margin = 16, 30, 18
    cols, rows = 3, 2
    grid_w = cols * cw + (cols - 1) * gap
    header_h = 96
    footer_h = 28
    total_w = grid_w + 2 * margin
    total_h = header_h + margin + rows * (title_h + ch) + (rows - 1) * gap + footer_h

    canvas = np.full((total_h, total_w, 3), 24, np.uint8)
    cv2.rectangle(canvas, (0, 0), (total_w, header_h), (12, 12, 12), -1)
    cv2.line(canvas, (0, header_h - 1), (total_w, header_h - 1), (0, 160, 0), 2)
    _draw_text_box(canvas, "geltilt  -  full tilt-correction pipeline", (margin, 30), 0.72, (0, 255, 0), 2)
    _draw_text_box(
        canvas,
        "input -> gray -> background + polarity -> signal -> ROI crop -> angle sweep "
        "(rotate-safe crop -> vertical projection -> tv2) -> argmax -> correct rotation -> output",
        (margin, 56), 0.5, (150, 200, 255),
    )
    _draw_text_box(
        canvas,
        f"detected correction {estimate.angle:+.2f} deg  |  method {estimate.method}  |  polarity {polarity}  |  "
        f"window lo={lo:.0f} hi={hi:.0f} (x{factor:.1f})  |  file {source_name or '-'}",
        (margin, 80), 0.5, (200, 200, 200),
    )

    for i, (title, panel) in enumerate(zip(titles, panels)):
        r, c = divmod(i, cols)
        x = margin + c * (cw + gap)
        y = header_h + margin + r * (title_h + ch + gap)
        _draw_text_box(canvas, title, (x + 4, y + 20), 0.5, (0, 200, 255))
        canvas[y + title_h:y + title_h + ch, x:x + cw] = panel
        if c < cols - 1:
            ax, ay = x + cw + 3, y + title_h + ch // 2
            cv2.arrowedLine(canvas, (ax, ay), (ax + gap - 6, ay), (0, 255, 0), 1, cv2.LINE_AA, tipLength=0.4)

    footer = ("panels 1-3 are the preprocessing stages shown on the same full-frame window; "
              "panel 4 is the objective being maximised; panels 5-6 the rotation in and out")
    _draw_text_box(canvas, footer, (margin, total_h - 8), 0.45, (150, 160, 172))
    return canvas


def _draw_index_badge(canvas: np.ndarray, index: int, total: Optional[int] = None) -> None:
    """Draw a prominent index badge in the top-right corner of a figure."""
    label = f"#{index:02d}" if total is None else f"#{index:02d}/{total:02d}"
    scale, thick, pad = 0.95, 2, 10
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x1, y0 = canvas.shape[1] - 12, 10
    x0 = x1 - tw - 2 * pad
    y1 = y0 + th + base + 2 * pad
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 215, 255), 2)
    cv2.putText(canvas, label, (x0 + pad, y0 + pad + th), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 215, 255), thick, cv2.LINE_AA)


def render_gt_vs_algorithm(
    image: np.ndarray,
    gt_angle: float,
    algorithm_angle: float,
    source_name: Optional[str] = None,
    panel_height: int = 760,
    guide_step: int = 90,
    plo: float = 0.5,
    phi: float = 99.5,
    gamma: float = 1.0,
    index: Optional[int] = None,
    index_total: Optional[int] = None,
    status: Optional[str] = None,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build a BGR figure: LEFT = ground-truth rotation, RIGHT = algorithm rotation.

    Both panels are corrected by their respective angle, cropped to the same
    window, auto-contrast-stretched on the full-frame window, and overlaid with
    vertical guides so the residual lane lean can be compared directly.
    ``index``/``index_total`` draw a top-right badge so figures can be referenced
    by number instead of filename.
    """
    from .core import correct_rotation

    window = content_window(image)
    if mask is not None:
        ys, xs = np.where(mask)
        if xs.size:
            h, w = image.shape[:2]
            m_y0, m_y1 = int(ys.min()), int(ys.max()) + 1
            m_x0, m_x1 = int(xs.min()), int(xs.max()) + 1
            if window is None:
                y0, y1, x0, x1 = m_y0, m_y1, m_x0, m_x1
            else:
                wy0, wy1, wx0, wx1 = window
                y0, y1 = min(wy0, m_y0), max(wy1, m_y1)
                x0, x1 = min(wx0, m_x0), max(wx1, m_x1)
            my = max(20, int(0.06 * (y1 - y0)))
            mx = max(20, int(0.06 * (x1 - x0)))
            window = (max(0, y0 - my), min(h, y1 + my), max(0, x0 - mx), min(w, x1 + mx))

    lo, hi = robust_limits(to_gray(image), plo, phi)

    gt_crop = _apply_window(correct_rotation(image, gt_angle), window)
    algo_crop = _apply_window(correct_rotation(image, algorithm_angle), window)

    left = cv2.cvtColor(_resize_to_height(to_u8(gt_crop, lo, hi, gamma), panel_height), cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(_resize_to_height(to_u8(algo_crop, lo, hi, gamma), panel_height), cv2.COLOR_GRAY2BGR)

    if mask is not None and window is not None:
        y0, y1, x0, x1 = window
        m = mask.astype(np.uint8) * 255
        for panel, ang in ((left, gt_angle), (right, algorithm_angle)):
            rm = correct_rotation(m, ang)
            rm = (rm > 127).astype(np.uint8)[y0:y1, x0:x1]
            rm = cv2.resize(rm, (panel.shape[1], panel.shape[0]), interpolation=cv2.INTER_NEAREST)
            cnts = cv2.findContours(rm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            cv2.drawContours(panel, cnts, -1, (0, 255, 255), 2)

    left = draw_vertical_guides(left, guide_step)
    right = draw_vertical_guides(right, guide_step)

    diff = algorithm_angle - gt_angle
    diff_color = (0, 255, 0) if abs(diff) <= 1.0 else (0, 0, 255)
    info_lines = []
    if source_name:
        prefix = f"#{index:02d}  " if index is not None else ""
        info_lines.append((f"{prefix}file: {source_name}", (200, 200, 200), 0.6))
    info_lines.append(
        (f"GROUND TRUTH {gt_angle:+.2f} deg   vs   ALGORITHM {algorithm_angle:+.2f} deg   "
         f"diff {diff:+.2f} deg", diff_color, 0.62)
    )
    info_lines.append(
        (f"left = ground-truth rotation, right = algorithm rotation  |  cyan lines are vertical  |  "
         f"window lo={lo:.0f} hi={hi:.0f}", (150, 200, 255), 0.5)
    )
    if status:
        bad = any(k in status.lower() for k in ("refus", "no_peak", "halves_differ", "not_equivariant"))
        info_lines.append((status, (0, 0, 255) if bad else (0, 215, 255), 0.55))
    if mask is not None:
        info_lines.append(("yellow outline = adopted slab mask", (0, 255, 255), 0.5))

    figure = _compose_two_panel(
        left, right,
        f"GROUND TRUTH  ({gt_angle:+.2f} deg)",
        f"ALGORITHM  ({algorithm_angle:+.2f} deg, diff {diff:+.2f})",
        info_lines, panel_height,
    )
    if index is not None:
        _draw_index_badge(figure, index, index_total)
    return figure


def save_comparison(path: PathLike, figure_bgr: np.ndarray) -> None:
    """Save a BGR comparison figure to disk (encoded as PNG/TIFF via RGB)."""
    save_array(path, figure_bgr[:, :, ::-1])