from __future__ import annotations

import numpy as np
import pytest

from geltilt.projview import (
    clamp_roi,
    default_roi_from_u8,
    normalize_roi,
    profile_sharpness,
    profile_to_points,
    roi_corners,
    vertical_projection_roi,
)

from conftest import make_gel


def test_normalize_roi_orders_corners():
    assert normalize_roi((10, 20, 5, 8)) == (5.0, 8.0, 10.0, 20.0)


def test_roi_corners_clockwise_from_topleft():
    assert roi_corners((0, 0, 2, 3)) == [(0.0, 0.0), (2.0, 0.0), (2.0, 3.0), (0.0, 3.0)]


def test_clamp_roi_to_shape():
    assert clamp_roi((-5, -5, 999, 999), (10, 20)) == (0.0, 0.0, 20.0, 10.0)


def test_default_roi_stays_inside_frame():
    img = np.zeros((60, 80), np.uint8)
    img[10:50, 20:60] = 200
    x0, y0, x1, y1 = default_roi_from_u8(img)
    assert 0 <= x0 < x1 <= 80
    assert 0 <= y0 < y1 <= 60


def test_projection_sharper_when_lanes_vertical():
    gel = make_gel(h=400, w=400, n_lanes=5)
    roi = (0, 0, 400, 400)
    upright = vertical_projection_roi(gel, roi, 0.0, (200.0, 200.0))
    tilted = vertical_projection_roi(gel, roi, 12.0, (200.0, 200.0))
    assert upright.size > 100
    assert profile_sharpness(upright) > profile_sharpness(tilted)


def test_projection_empty_roi_is_empty():
    gel = make_gel(h=50, w=50)
    prof = vertical_projection_roi(gel, (10, 10, 10, 10), 0.0, (0.0, 0.0))
    assert prof.size == 0
    assert profile_sharpness(prof) == 0.0


def test_projection_respects_roi_extent():
    gel = np.zeros((20, 100), np.float64)
    gel[:, 40:60] = 5.0
    full = vertical_projection_roi(gel, (0, 0, 100, 20), 0.0, (0.0, 0.0))
    left = vertical_projection_roi(gel, (0, 0, 30, 20), 0.0, (0.0, 0.0))
    assert full.sum() > left.sum()
    assert left.sum() == pytest.approx(0.0)


def test_profile_to_points_fits_canvas_and_inverts_y():
    prof = np.array([0.0, 1.0, 0.0, 2.0])
    pts = profile_to_points(prof, 200, 100)
    assert len(pts) == 8
    xs, ys = pts[0::2], pts[1::2]
    assert min(xs) >= 40 and max(xs) <= 200
    assert min(ys) >= 0 and max(ys) <= 100
    assert ys[3] < ys[1] < ys[0]  # larger value sits higher (smaller y)


def test_viewer_constructs(tmp_path):
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    root.withdraw()
    from geltilt.projview import ProjectionViewer

    app = ProjectionViewer(root, run_dir=tmp_path)
    assert app.profile is None
    root.destroy()