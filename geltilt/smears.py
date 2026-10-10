"""Default auto-rotation estimator, implemented to SPEC-AUTOROTATE.md.

Measures the faint vertical *smear* running down each lane rather than the slab
geometry: the gel body is masked, eroded by 25 px (which removes the slab's own
left/right edges), high-passed (``flatten``), passed through an anisotropic ridge
matched-filter for vertical streaks, and scored by two cues -- ``contrast`` and
``persistence`` -- over one pass of 81 angles.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from math import isfinite
from os import cpu_count
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

# --- constants (SPEC §2) -------------------------------------------------------
BODY_FRAC = 0.04
BODY_MIN_AREA = 1500
BODY_BLUR = 9
EDGE_ERODE = 25
BG_SIGMA = 25.0
RIDGE_SIGMA_Y = 6.0
RIDGE_SIGMA_X = 2.0
RANGE_DEG = 10.0
STEP = 0.25
COARSE_STEP = 1.0
PREBLUR = 0.8
MIN_SUPPORT = 800
SLICE_HEIGHT = 36
SLICE_MIN_ROWS = 24
EDGE_GUARD = 1.0
AGREE_TOL = 1.5
SPLIT_TOL = 1.3
EQUIV_TOL = 0.5
EQUIV_DELTAS = (-2.0, 2.0)

# --- helpers (SPEC §3) ---------------------------------------------------------

def robust_limits(arr: np.ndarray, plo: float = 0.1, phi: float = 99.9) -> Tuple[float, float]:
    f = np.asarray(arr, dtype=np.float32)
    lo, hi = np.percentile(f, (plo, phi))
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def rotate_about(img: np.ndarray, centre: Tuple[float, float], angle: float,
                 border_value: float = 0) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])), float(angle), 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=float(border_value))


def rotate_roi(roi: np.ndarray, angle: float, preblur: float = 0.0) -> np.ndarray:
    h, w = roi.shape[:2]
    src = roi
    if preblur and preblur > 0:
        src = cv2.GaussianBlur(roi.astype(np.float32), (0, 0), float(preblur))
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle), 1.0)
    out = cv2.warpAffine(src, m, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return out


def map_chunks(fn, items: np.ndarray):
    """Apply fn to contiguous chunks; concatenate. Bit-identical to a serial loop."""
    items = np.asarray(items)
    n = items.size
    if n == 0:
        return fn(items)
    threads = min(8, max(1, (cpu_count() or 1) - 1), n // 6)
    if threads <= 1:
        return fn(items)
    bounds = np.linspace(0, n, threads + 1).astype(int)
    chunks = [items[bounds[i]:bounds[i + 1]] for i in range(threads)]
    with ThreadPoolExecutor(max_workers=threads) as ex:
        results = list(ex.map(fn, chunks))
    if isinstance(results[0], tuple):
        return tuple(np.concatenate([r[j] for r in results]) for j in range(len(results[0])))
    return np.concatenate(results)


# --- gel mask / interior (SPEC §4, §5) ----------------------------------------

def gel_mask(img: np.ndarray, frac: float = BODY_FRAC, min_area: int = BODY_MIN_AREA,
             blur: float = BODY_BLUR) -> np.ndarray:
    f = img.astype(np.float32)
    b = cv2.GaussianBlur(f, (0, 0), float(blur))
    lo, hi = robust_limits(f, 0.1, 99.9)
    if hi <= lo:
        return np.zeros(img.shape[:2], np.uint8)
    m = (b > lo + frac * (hi - lo)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 1
    return out


def interior(img: np.ndarray, mask: np.ndarray, erode: int = EDGE_ERODE):
    m = np.asarray(mask, dtype=np.uint8)
    if m.sum() == 0:
        return None
    if erode > 0:
        me = cv2.erode(m, np.ones((erode, erode), np.uint8))
        if me.sum() >= MIN_SUPPORT:
            m = me
    ys, xs = np.where(m > 0)
    if xs.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sub = img[y0:y1, x0:x1].astype(np.float32)
    subm = m[y0:y1, x0:x1].astype(np.float32)
    if subm.sum() < MIN_SUPPORT:
        return None
    return sub, subm


# --- per-angle responses (SPEC §6) --------------------------------------------

def flatten(roi: np.ndarray, sigma: float = BG_SIGMA) -> np.ndarray:
    f = roi.astype(np.float32)
    return np.clip(f - cv2.GaussianBlur(f, (0, 0), float(sigma)), 0.0, None)


def ridge(roi: np.ndarray) -> np.ndarray:
    f = roi.astype(np.float32)
    d2 = gaussian_filter(f, sigma=(RIDGE_SIGMA_Y, RIDGE_SIGMA_X), order=(0, 2))
    return np.clip(-d2, 0.0, None)


def persistence(resp: np.ndarray) -> float:
    rows = resp.shape[0]
    if rows < SLICE_HEIGHT + SLICE_MIN_ROWS:
        return 0.0
    profs = []
    y = 0
    while y <= rows - SLICE_MIN_ROWS:
        p = resp[y:y + SLICE_HEIGHT].sum(axis=0)
        sd = float(p.std())
        if sd > 0:
            profs.append((p - p.mean()) / sd)
        y += SLICE_HEIGHT
    if len(profs) < 2:
        return 0.0
    m = np.asarray(profs)
    corr = m @ m.T / m.shape[1]
    iu = np.triu_indices(m.shape[0], k=1)
    return float(corr[iu].mean())


def scores_serial(strip, angles: np.ndarray, preblur: float = PREBLUR):
    sub, subm = strip
    angles = np.asarray(angles, dtype=np.float64)
    contrast = np.empty(angles.size, dtype=np.float64)
    persist = np.empty(angles.size, dtype=np.float64)
    for k, a in enumerate(angles):
        rot = rotate_roi(sub, float(a), preblur=preblur)
        rm = rotate_roi(subm, float(a), preblur=0.0)
        supp = rm > 0.5
        resp = ridge(flatten(rot, BG_SIGMA)) * supp
        p = resp.sum(axis=0)
        mean = float(p.mean())
        contrast[k] = float(p.std()) / mean if mean > 0 else 0.0
        persist[k] = persistence(resp)
    return contrast, persist


# --- the estimator (SPEC §7, §9) ----------------------------------------------

def _angles(step: float) -> np.ndarray:
    return np.arange(-RANGE_DEG, RANGE_DEG + 1e-9, step)


def _peak(curve: np.ndarray, step: float):
    i = int(np.argmax(curve))
    best = float(_angles(step)[i])
    if 0 < i < len(curve) - 1:
        d = curve[i - 1] - 2 * curve[i] + curve[i + 1]
        if abs(d) > 1e-15:
            best += 0.5 * (curve[i - 1] - curve[i + 1]) / d * step
    boundary = abs(best) >= RANGE_DEG - EDGE_GUARD
    return best, boundary


def _core(img, mask, step):
    strip = interior(img, mask)
    if strip is None:
        return None
    ang = _angles(step)
    contrast, persist = map_chunks(lambda a: scores_serial(strip, a), ang)
    a_cv, edge_cv = _peak(contrast, step)
    a_ps, edge_ps = _peak(persist, step)
    # Prefer the sharper contrast cue; fall back to persistence only when contrast
    # has run off the range end (its peak is then an artefact of the sweep limit).
    # (Contrast was previously gated on agreement with persistence, which let a
    # broad spurious persistence peak override a correct contrast estimate.)
    if not edge_cv:
        return a_cv, False
    return a_ps, edge_ps


def _estimate_angle(img, mask, step: float):
    r = _core(img, mask, step)
    if r is None:
        return None
    return r


def _gates(img, mask, angle, split_tol, equiv_tol):
    xs = np.where(mask.any(axis=0))[0]
    if xs.size < 2:
        return None, None, "no_gel"
    mid = (int(xs.min()) + int(xs.max())) // 2
    left = mask.copy()
    left[:, mid:] = 0
    right = mask.copy()
    right[:, :mid] = 0
    a_left = _estimate_angle(img, left, STEP)
    a_right = _estimate_angle(img, right, STEP)
    if a_left is None or a_right is None:
        return None, None, "no_gel"
    split = abs(a_left[0] - a_right[0])
    if split > split_tol:
        return split, None, "halves_differ"

    h, w = img.shape[:2]
    centre = (w / 2.0, h / 2.0)
    dev = 0.0
    for d in EQUIV_DELTAS:
        rot = rotate_about(img, centre, d)
        a_d = _estimate_angle(rot, gel_mask(rot), COARSE_STEP)
        if a_d is None:
            continue
        dev = max(dev, abs(a_d[0] - (angle - d)))
    if dev > equiv_tol:
        return split, dev, "not_equivariant"
    return split, dev, ""


def estimate(img: np.ndarray, mask: Optional[np.ndarray] = None, gates: bool = True,
             reference: Optional[float] = None, min_delta: float = 0.5,
             split_tol: float = SPLIT_TOL, equiv_tol: float = EQUIV_TOL):
    """Return (ok, angle, info). Apply ``angle`` as-is to level the smears."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mask is None:
        mask = gel_mask(img)

    if interior(img, mask) is None:
        info = {"angle": 0.0, "boundary": True, "split": None, "equiv": None,
                "reason": "no_peak", "gated": bool(gates)}
        return True, 0.0, info

    core = _core(img, mask, STEP)
    angle, boundary = core
    info = {"angle": angle, "boundary": boundary, "split": None, "equiv": None,
            "reason": "", "gated": bool(gates)}

    if boundary:
        info["reason"] = "no_peak"
        return False, angle, info
    if reference is not None and abs(angle - reference) < min_delta:
        info["reason"] = "agrees"
        return False, angle, info
    if not gates:
        return True, angle, info

    split, dev, reason = _gates(img, mask, angle, split_tol, equiv_tol)
    info["split"], info["equiv"] = split, dev
    if reason:
        info["reason"] = reason
        return False, angle, info
    return True, angle, info


def smear_angle(img: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Bare estimator: the smear angle with the gates omitted (SPEC §1)."""
    return float(estimate(img, mask=mask, gates=False)[1])