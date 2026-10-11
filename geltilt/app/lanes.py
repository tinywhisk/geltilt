"""Composite-crop lane model (see SPEC-COMPOSITE.md).

Pure, Tk-free functions: band blobs -> lanes -> boundaries -> edited boundaries,
the lane-spec language, and the column-stitch composition. All coordinates are
image pixels. A lane is the *gap between two boundaries* (``spans`` is derived
from ``edges``), which is what makes gaps/overlaps/reordering inexpressible.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

MIN_BLOB_AREA = 50          # GelGenie's own minimum_pixel_count (§4.1)
PITCH_TOLERANCE = 0.3       # §4.3
LANE_MIN_PX = 8             # §5.5
NUDGE_EXPAND_PX = 4         # widen the band mask left/right this much for lane nudging only

Rect = Tuple[int, int, int, int]  # (x0, y0, x1, y1)


def expand_mask_horizontally(mask: np.ndarray, px: int = NUDGE_EXPAND_PX) -> np.ndarray:
    """Dilate a mask ``px`` pixels to the left and right (no vertical change)."""
    m = (np.asarray(mask) > 0).astype(np.uint8)
    px = int(px)
    if px <= 0:
        return m
    return cv2.dilate(m, np.ones((1, 2 * px + 1), np.uint8))


class LaneSpecError(Exception):
    """A lane-spec parse failure: a locale key plus its interpolation args."""

    def __init__(self, key: str, **kwargs) -> None:
        self.key = key
        self.kwargs = kwargs
        super().__init__(key)


def spans_from_edges(edges: Sequence[int]) -> List[Tuple[int, int]]:
    return [(int(a), int(b)) for a, b in zip(edges, edges[1:])]


# --------------------------------------------------------------------------- #
# §4 Lane identification                                                        #
# --------------------------------------------------------------------------- #
def band_blobs(mask: np.ndarray, min_area: int = MIN_BLOB_AREA):
    """Connected components >= ``min_area`` -> (blobs, dropped)."""
    m = (np.asarray(mask) > 0).astype(np.uint8)
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
    blobs: List[Dict[str, object]] = []
    dropped = 0
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < int(min_area):
            dropped += 1
            continue
        x0 = int(stats[i, cv2.CC_STAT_LEFT])
        y0 = int(stats[i, cv2.CC_STAT_TOP])
        blobs.append({
            "x0": x0, "x1": x0 + int(stats[i, cv2.CC_STAT_WIDTH]),
            "y0": y0, "y1": y0 + int(stats[i, cv2.CC_STAT_HEIGHT]),
            "cx": float(centroids[i][0]), "cy": float(centroids[i][1]),
            "area": area,
        })
    return blobs, dropped


def label_lanes(blobs: Sequence[Dict[str, object]]) -> List[List[Dict[str, object]]]:
    """Group blobs into lanes left-to-right (QuPath/GelGenie convention)."""
    remaining = list(blobs)
    lanes: List[List[Dict[str, object]]] = []
    while remaining:
        left = min(remaining, key=lambda b: (b["cx"], b["cy"]))
        lo, hi = float(left["x0"]), float(left["x1"])
        taken = [
            b for b in remaining
            if (lo < b["cx"] < hi)
            or (b["x0"] < hi and b["x1"] > hi)
            or (b["x0"] < lo and b["x1"] > lo)
        ]
        if not taken:
            taken = [left]  # progress guard (T-1): never consume nothing
        for b in taken:
            remaining.remove(b)
        taken.sort(key=lambda b: b["cy"])
        lanes.append(taken)
        if len(lanes) > len(blobs):
            raise RuntimeError("label_lanes made no progress (centroid/edge mismatch)")
    lanes.sort(key=lambda lane: float(np.mean([b["cx"] for b in lane])))
    return lanes


def fill_empty_lanes(centres: Sequence[float]):
    """Insert evenly spaced centres for empty lanes implied by the comb pitch."""
    centres = [float(c) for c in centres]
    if len(centres) < 2:
        return list(centres), 0.0, 0
    diffs = np.diff(np.asarray(centres))
    pitch = float(np.median(diffs))
    if pitch <= 0:
        return list(centres), 0.0, 0
    full = [centres[0]]
    inserted = 0
    for a, b in zip(centres, centres[1:]):
        gap = (b - a) / pitch
        k = max(1, int(round(gap)))
        if k > 1 and abs(gap - k) > PITCH_TOLERANCE:
            k = 1
        for j in range(1, k):
            full.append(a + (b - a) * j / k)
            inserted += 1
        full.append(b)
    return full, pitch, inserted


def lane_layout(mask: np.ndarray, region: Rect, min_area: int = MIN_BLOB_AREA,
                count: Optional[int] = None,
                nudge_expand_px: int = NUDGE_EXPAND_PX) -> Dict[str, object]:
    """Identify lanes of ``mask`` confined to ``region`` = [x0, y0, x1, y1].

    ``nudge_blobs`` is a second blob set from the mask widened left/right by
    ``nudge_expand_px`` — used **only** by :func:`nudge_off_bands`, so a boundary
    clears the visible band (which the raw mask can under-cover), while lane
    identification and naming still use the untouched ``blobs``.
    """
    m = (np.asarray(mask) > 0).astype(np.uint8)
    h, w = m.shape[:2]
    x0 = max(0, min(int(region[0]), w))
    y0 = max(0, min(int(region[1]), h))
    x1 = max(0, min(int(region[2]), w))
    y1 = max(0, min(int(region[3]), h))
    confined = np.zeros_like(m)
    if x1 > x0 and y1 > y0:
        confined[y0:y1, x0:x1] = m[y0:y1, x0:x1]

    blobs, dropped = band_blobs(confined, min_area)
    nudge_blobs, _ = band_blobs(expand_mask_horizontally(confined, nudge_expand_px), min_area)
    lanes = label_lanes(blobs)
    centres = [float(np.mean([b["cx"] for b in lane])) for lane in lanes]

    if count and count > 1 and len(centres) >= 2:
        lo, hi = min(centres), max(centres)
        full = list(np.linspace(lo, hi, count))
        pitch = (hi - lo) / (count - 1)
        inserted = max(0, count - len(centres))
    else:
        full, pitch, inserted = fill_empty_lanes(centres)

    if not full:
        return {"lanes": [], "edges": [], "spans": [], "blobs": [], "nudge_blobs": [],
                "centres": [], "pitch": 0.0, "inserted": 0, "dropped": dropped, "count": 0}

    named: List[List[Dict[str, object]]] = [[] for _ in full]
    for b in blobs:
        i = int(np.argmin([abs(float(b["cx"]) - c) for c in full]))
        named[i].append(b)
    for i, lane in enumerate(named, 1):
        lane.sort(key=lambda b: b["cy"])
        for k, band in enumerate(lane, 1):
            band["name"] = f"L{i}-{k}"

    raw = [full[0] - pitch / 2.0] + [(a + b) / 2.0 for a, b in zip(full, full[1:])] \
        + [full[-1] + pitch / 2.0]
    edges = [max(x0, min(int(round(e)), x1)) for e in raw]
    for i in range(1, len(edges)):
        edges[i] = max(edges[i], edges[i - 1] + 1)

    return {
        "lanes": named, "edges": edges, "spans": spans_from_edges(edges), "blobs": blobs,
        "nudge_blobs": nudge_blobs,
        "centres": full, "pitch": pitch, "inserted": inserted, "dropped": dropped,
        "count": len(edges) - 1,
    }


# --------------------------------------------------------------------------- #
# §5 Editing the boundaries                                                     #
# --------------------------------------------------------------------------- #
def move_edge(edges: Sequence[int], index: int, x: float,
              width: Optional[int] = None, min_side: int = 1) -> List[int]:
    """Move one boundary (inner or outer), clamped so it cannot cross a neighbour.

    Outer boundaries are bounded by ``0`` / ``width`` (the frame), matching §5.1,
    so the lane region's two ends can be adjusted too.
    """
    edges = [int(e) for e in edges]
    if index < 0 or index >= len(edges):
        return edges
    low = edges[index - 1] + int(min_side) if index > 0 else 0
    if index < len(edges) - 1:
        high = edges[index + 1] - int(min_side)
    else:
        high = int(width) if width is not None else edges[index]
    if low > high:
        return edges
    new = list(edges)
    new[index] = max(low, min(int(round(x)), high))
    return new


def insert_edge(edges: Sequence[int], x: float, min_side: int = LANE_MIN_PX):
    """Split: add a boundary at ``x``. Returns (edges, index) or (edges, None)."""
    edges = [int(e) for e in edges]
    xi = int(round(x))
    if xi <= edges[0] or xi >= edges[-1]:
        return edges, None
    for i in range(1, len(edges)):
        if edges[i - 1] < xi < edges[i]:
            if xi - edges[i - 1] < int(min_side) or edges[i] - xi < int(min_side):
                return edges, None
            new = list(edges)
            new.insert(i, xi)
            return new, i
    return edges, None


def remove_edge(edges: Sequence[int], index: int, min_side: int = LANE_MIN_PX):
    """Join: drop an inner boundary. Returns (edges, index) or (edges, None)."""
    edges = [int(e) for e in edges]
    if len(edges) < 4:
        return edges, None
    if index <= 0 or index >= len(edges) - 1:
        return edges, None
    new = list(edges)
    new.pop(index)
    return new, index


def renumber_after_split(order: Sequence[int], lane: int) -> List[int]:
    return [n if n <= lane else n + 1 for n in order]


def renumber_after_merge(order: Sequence[int], lane: int) -> List[int]:
    mapped = [lane if n == lane + 1 else (n - 1 if n > lane + 1 else n) for n in order]
    seen: List[int] = []
    for n in mapped:
        if n not in seen:
            seen.append(n)
    return seen


# --------------------------------------------------------------------------- #
# §6 Even widths                                                                #
# --------------------------------------------------------------------------- #
def even_edges(edges: Sequence[int]) -> List[int]:
    edges = [int(e) for e in edges]
    n = len(edges) - 1
    if n < 1:
        return edges
    left, right = edges[0], edges[-1]
    if right - left < n:
        return edges
    return [left + int(round(i * (right - left) / n)) for i in range(n + 1)]


def _cuts(x: int, boxes: Sequence[Tuple[int, int]]) -> bool:
    return any(x0 < x < x1 for x0, x1 in boxes)


def nudge_off_bands(edges: Sequence[int], blobs: Sequence[Dict[str, object]],
                    min_side: int = LANE_MIN_PX):
    """Push internal boundaries that cut a band clear of it. Returns (edges, nudged).

    Candidate positions are **every band edge** in range, not just the two edges of the
    bands currently cut (§6.2's rule oscillates when a boundary sits between two
    overlapping bands: each move lands back inside the other). The move is verified to
    cut **no** band, so it converges in one step; a boundary that cannot clear without
    crossing a neighbour is left alone.
    """
    edges = [int(e) for e in edges]
    boxes = [(int(b["x0"]), int(b["x1"])) for b in blobs]
    nudged = 0
    for index in range(1, len(edges) - 1):
        for _ in range(4):
            low = edges[index - 1] + int(min_side)
            high = edges[index + 1] - int(min_side)
            x = edges[index]
            if low > high or not _cuts(x, boxes):
                break
            cand = {low, high}
            for x0, x1 in boxes:
                if low <= x0 <= high:
                    cand.add(x0)
                if low <= x1 <= high:
                    cand.add(x1)
            safe = [c for c in cand if not _cuts(c, boxes)]
            if not safe:
                break  # cannot clear without crossing a neighbour
            new_x = min(safe, key=lambda c: (abs(c - x), c))
            if new_x == x:
                break
            edges[index] = new_x
            nudged += 1
    return edges, nudged


# --------------------------------------------------------------------------- #
# §7 The spec language                                                          #
# --------------------------------------------------------------------------- #
_TERM_RE = re.compile(r"^[Ll]?(\d+)(?:\s*-\s*[Ll]?(\d+))?$")


def parse_lane_spec(text: Optional[str], n_lanes: int) -> List[List[int]]:
    """Parse ``3,1-2,4-6`` -> ``[[3], [1, 2], [4, 5, 6]]`` (raises LaneSpecError)."""
    if text is None or text.strip() == "":
        raise LaneSpecError("lane_spec.empty")
    if "{" in text or "}" in text:
        raise LaneSpecError("lane_spec.braces")
    groups: List[List[int]] = []
    for term in text.split(","):
        t = term.strip()
        if t == "":
            raise LaneSpecError("lane_spec.empty_term")
        m = _TERM_RE.match(t)
        if not m:
            raise LaneSpecError("lane_spec.unparsed", term=term)
        first = int(m.group(1))
        last = int(m.group(2)) if m.group(2) else first
        if first < 1 or last < 1:
            raise LaneSpecError("lane_spec.not_positive", term=term)
        if first > n_lanes or last > n_lanes:
            raise LaneSpecError("lane_spec.out_of_range", term=term, count=n_lanes)
        if last < first:
            raise LaneSpecError("lane_spec.reversed", term=term)
        groups.append(list(range(first, last + 1)))
    return groups


def spec_groups(groups: Sequence[Sequence[int]]) -> List[int]:
    """The flat order, the single source of truth for output order."""
    return [int(n) for group in groups for n in group]


# --------------------------------------------------------------------------- #
# §8 Window, §9 Composition, §10 Export                                          #
# --------------------------------------------------------------------------- #
def composite_rows(crop: Optional[Sequence[float]], height: int) -> Tuple[int, int]:
    """The kept row band (image pixels) — the crop's y0/y1, no size argument."""
    if crop is None:
        return 0, int(height)
    y0 = max(0, min(int(crop[1]), int(height)))
    y1 = max(0, min(max(y0, int(crop[3])), int(height)))
    return y0, y1


def output_runs(groups: Sequence[Sequence[int]]) -> List[List[int]]:
    """Flat order -> maximal blocks of consecutive lane numbers."""
    runs: List[List[int]] = []
    for n in spec_groups(groups):
        # consecutive continues the run; an equal value is a duplicate, not a second copy
        if runs and n in (runs[-1][-1] + 1, runs[-1][-1]):
            runs[-1].append(n)
        else:
            runs.append([n])
    return runs


def stitch_lanes(arr: np.ndarray, edges: Sequence[int], groups: Sequence[Sequence[int]]) -> np.ndarray:
    """Concatenate the chosen lane blocks (each run taken as one column slice)."""
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D grey image, got shape {arr.shape}")
    parts = [arr[:, int(edges[run[0] - 1]):int(edges[run[-1]])] for run in output_runs(groups)]
    if not parts:
        return arr[:, :0]
    return np.concatenate(parts, axis=1)