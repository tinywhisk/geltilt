"""Background estimation, polarity detection, signal extraction and ROI masks."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def estimate_background(gray: np.ndarray) -> float:
    """Robust background level of a grayscale image (median intensity)."""
    return float(np.median(gray))


def robust_limits(arr: np.ndarray, plo: float = 0.5, phi: float = 99.5) -> Tuple[float, float]:
    """Robust (low, high) intensity limits from percentiles of an array."""
    f = arr.astype(np.float32)
    lo, hi = np.percentile(f, (plo, phi))
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def to_u8(arr: np.ndarray, lo: float, hi: float, gamma: float = 1.0) -> np.ndarray:
    """Map an array to uint8 using robust limits with clipping and optional gamma."""
    f = arr.astype(np.float32)
    span = float(hi) - float(lo)
    if span <= 0:
        span = 1.0
    f = (f - float(lo)) / span
    np.clip(f, 0.0, 1.0, out=f)
    if gamma != 1.0:
        np.power(f, float(gamma), out=f)
    return (f * 255.0 + 0.5).astype(np.uint8)


def preview_u8(arr: np.ndarray, gamma: float = 0.45, percentile: float = 99.7) -> np.ndarray:
    """Display-only transform: ``clip(arr / p99.7, 0, 1) ** gamma`` scaled to uint8.

    Unlike :func:`to_u8` there is no low-end subtraction; the image is simply
    divided by a high percentile (so the brightest signal maps to ~1.0), clipped
    and gamma-corrected. Returns a 2D uint8 grayscale array (feeding :func:`to_u8`
    with these settings is equivalent to the reference ``preview`` snippet, minus
    the BGR wrapping used only for OpenCV display).
    """
    gray = np.asarray(arr, dtype=np.float32)
    scale = max(float(np.percentile(gray, percentile)), 1e-6)
    v = np.clip(gray / scale, 0.0, 1.0) ** float(gamma)
    return (v * 255.0).astype(np.uint8)


def detect_polarity(gray: np.ndarray, background: Optional[float] = None) -> str:
    """Return 'bright' for bright bands on a dark background, else 'dark'.

    The sign of the intensity skewness about the background separates the two
    common gel imaging modes (fluorescence vs. Coomassie/dye staining).
    """
    if background is None:
        background = estimate_background(gray)
    d = gray.astype(np.float64) - background
    m2 = float(np.mean(d * d))
    m3 = float(np.mean(d * d * d))
    if m2 <= 0:
        return "bright"
    skew = m3 / (m2 ** 1.5)
    return "bright" if skew >= 0 else "dark"


def to_signal(
    gray: np.ndarray,
    polarity: str = "auto",
    background: Optional[float] = None,
) -> Tuple[np.ndarray, str, float]:
    """Convert a grayscale image into a non-negative band-signal image.

    Returns (signal, polarity, background).
    """
    gray = gray.astype(np.float32)
    if background is None:
        background = estimate_background(gray)
    if polarity not in ("auto", "bright", "dark"):
        raise ValueError(f"polarity must be auto/bright/dark, got {polarity!r}")
    if polarity == "auto":
        polarity = detect_polarity(gray, background)
    if polarity == "bright":
        signal = gray - background
    else:
        signal = background - gray
    return np.clip(signal, 0.0, None).astype(np.float32), polarity, float(background)


def normalize_signal(signal: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    """Scale the signal so that a high percentile maps to ~1.0."""
    scale = float(np.percentile(signal, percentile))
    if scale <= 0:
        return signal
    return (signal / scale).astype(np.float32)


def content_mask(signal: np.ndarray, k: float = 3.0, frac: float = 0.1) -> np.ndarray:
    """Boolean mask of pixels that carry band signal (robust against darkness).

    Thresholds at a robust MAD-based level, but never below ``frac`` of the
    99.5th percentile so that near-uniform (mostly-black) gels still yield a
    meaningful region instead of selecting sensor noise.
    """
    med = float(np.median(signal))
    mad = float(np.median(np.abs(signal - med)))
    mad_thr = med + k * 1.4826 * mad
    peak_thr = frac * float(np.percentile(signal, 99.5))
    thr = max(mad_thr, peak_thr)
    mask = signal > thr
    if mask.mean() < 1e-4:
        peak = float(signal.max())
        mask = signal > 0.05 * peak if peak > 0 else mask
    return mask


def content_bbox(
    mask: np.ndarray, margin_frac: float = 0.1, min_margin: int = 20
) -> Optional[Tuple[int, int, int, int]]:
    """Bounding box (y0, y1, x0, x1) of masked content, expanded by a margin."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    h, w = mask.shape
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    my = max(min_margin, int(round((y1 - y0) * margin_frac)))
    mx = max(min_margin, int(round((x1 - x0) * margin_frac)))
    return max(0, y0 - my), min(h, y1 + my), max(0, x0 - mx), min(w, x1 + mx)