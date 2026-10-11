from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from geltilt.app.domain import LaneLine
from geltilt.app.widgets.viewer import SingleViewer


def _evt(x, y):
    return SimpleNamespace(x=x, y=y)


@pytest.fixture
def viewer(tk_root):
    v = SingleViewer(tk_root, ui_scale=1.0)
    v.pack(fill="both", expand=True)
    v._canvas_size = lambda: (400, 300)  # deterministic geometry without mapping
    return v


def _display():
    arr = np.zeros((80, 60), np.uint8)
    arr[20:60, 20:40] = 200
    return arr


def test_renders_exactly_one_image_surface(viewer):
    viewer.set_image(_display())
    items = viewer.canvas.find_all()
    image_items = [i for i in items if viewer.canvas.type(i) == "image"]
    assert len(image_items) == 1


def test_rotation_is_applied_without_negating(viewer):
    viewer.set_image(_display())
    viewer.set_rotation(2.4)
    assert viewer.angle == 2.4


def test_fit_preserves_aspect_and_leaves_source_unchanged(viewer):
    u8 = _display()
    snapshot = u8.copy()
    viewer.set_image(u8)
    assert viewer.scale > 0
    # display-only: source array is untouched
    assert np.array_equal(u8, snapshot)


def test_default_view_fits_the_whole_image_without_zooming_in(viewer):
    from geltilt.labeler import source_to_screen

    viewer.set_image(_display())
    assert viewer.scale <= 1.0  # never magnifies by default
    sw, sh = viewer._src_shape
    for p in ((0.0, 0.0), (sw, 0.0), (0.0, sh), (sw, sh)):
        sx, sy = source_to_screen(p, viewer.angle, viewer.scale, viewer.anchor, viewer.center)
        assert -1.0 <= sx <= 401.0
        assert -1.0 <= sy <= 301.0


def test_overlay_toggles_are_independent(viewer):
    assert viewer.show_guides is False
    assert viewer.show_centroids is False
    viewer.toggle_overlay("guides")
    viewer.toggle_overlay("centroids")
    assert viewer.show_guides is True
    assert viewer.show_centroids is True
    viewer.toggle_overlay("lane_lines")
    assert viewer.show_lane_lines is True


def test_centroids_and_lane_overlays_draw(viewer):
    viewer.set_image(_display())
    base = len(viewer.canvas.find_all())
    centroids = np.array([[10.0, 20.0], [30.0, 40.0]], float)
    lanes = (LaneLine(0.05, -2.8, 40.0, 25.0, np.zeros((0, 2))),)
    viewer.set_overlays(centroids, lanes)
    viewer.show_centroids = True
    viewer.show_lane_lines = True
    viewer._render()
    assert len(viewer.canvas.find_all()) > base


def test_unknown_overlay_kind_raises(viewer):
    with pytest.raises(ValueError):
        viewer.toggle_overlay("nonsense")

# --------------------------------------------------------------------------- #
# Crop rectangle                                                               #
# --------------------------------------------------------------------------- #
def test_set_image_clears_a_previous_crop(viewer):
    viewer.set_image(_display())
    viewer.set_crop((10.0, 10.0, 40.0, 60.0))
    assert viewer.has_crop
    viewer.set_image(_display())
    assert not viewer.has_crop


def test_crop_draws_outline_and_eight_handles(viewer):
    viewer.set_image(_display())
    base = len(viewer.canvas.find_all())
    viewer.set_crop((10.0, 10.0, 40.0, 60.0))
    assert viewer.has_crop
    assert viewer.crop_rect() == (10.0, 10.0, 40.0, 60.0)
    assert len(viewer.canvas.find_all()) == base + 9  # outline + 8 handles


def test_crop_handles_are_hit_tested_in_screen_space(viewer):
    viewer.set_image(_display())
    viewer.set_crop((10.0, 10.0, 40.0, 60.0))
    x0, y0, x1, y1 = viewer._crop_screen_rect()
    assert viewer._hit_crop(x0, y0) == "nw"
    assert viewer._hit_crop((x0 + x1) / 2.0, y1) == "s"
    assert viewer._hit_crop(x0 + 500.0, y0 + 500.0) is None


def test_crop_edge_is_hittable_between_handles(viewer):
    viewer.set_image(_display())
    viewer.set_crop((5.0, 5.0, 55.0, 75.0))  # a long enough edge in screen pixels
    x0, y0, x1, y1 = viewer._crop_screen_rect()
    # along the top edge, between the nw corner and the n handle
    qx = x0 + (x1 - x0) * 0.3
    assert viewer._hit_crop(qx, y0) == "n"
    assert viewer._cursor_for(qx, y0) == viewer._CURSOR_V
    # along the left edge, between the nw corner and the w handle
    qy = y0 + (y1 - y0) * 0.3
    assert viewer._hit_crop(x0, qy) == "w"
    assert viewer._cursor_for(x0, qy) == viewer._CURSOR_H
    # corners still read as diagonal
    assert viewer._hit_crop(x0, y0) == "nw"


def test_crop_drag_moves_the_corner_and_notifies(viewer):
    viewer.set_image(_display())
    viewer.set_crop((10.0, 10.0, 40.0, 60.0))
    seen = []
    viewer.bind_crop_change(seen.append)
    x0, y0, _x1, _y1 = viewer._crop_screen_rect()
    viewer._on_press(_evt(x0, y0))
    viewer._on_drag(_evt(x0 + 6.0, y0 + 6.0))
    viewer._on_release(_evt(0, 0))
    nx0, ny0, _nx1, _ny1 = viewer.crop_rect()
    assert nx0 > 10.0 and ny0 > 10.0
    assert seen and seen[-1] == viewer.crop_rect()


def test_crop_stays_fixed_when_rotation_changes(viewer):
    viewer.set_image(_display())
    viewer.set_crop((10.0, 10.0, 40.0, 60.0))
    before = viewer._crop_screen_rect()
    viewer.set_rotation(5.0)
    assert viewer._crop_screen_rect() == before

# --------------------------------------------------------------------------- #
# Mouse bindings: left = crop box, right = pan                                 #
# --------------------------------------------------------------------------- #
def test_left_drag_creates_a_crop_and_notifies(viewer):
    viewer.set_image(_display())
    seen = []
    viewer.bind_crop_change(seen.append)
    viewer._on_press(_evt(50, 60))
    viewer._on_drag(_evt(150, 120))
    assert viewer.has_crop
    viewer._on_release(_evt(150, 120))
    assert seen and seen[-1] == viewer.crop_rect()


def test_left_click_without_drag_does_not_create_a_crop(viewer):
    viewer.set_image(_display())
    viewer._on_press(_evt(50, 60))
    viewer._on_release(_evt(50, 60))
    assert not viewer.has_crop


def test_left_drag_does_not_pan_the_view(viewer):
    viewer.set_image(_display())
    viewer.set_crop((10.0, 10.0, 40.0, 60.0))
    center = viewer.center
    x0, y0, x1, y1 = viewer._crop_screen_rect()
    viewer._on_press(_evt((x0 + x1) / 2.0, (y0 + y1) / 2.0))  # inside, away from handles
    viewer._on_drag(_evt((x0 + x1) / 2.0 + 20.0, (y0 + y1) / 2.0 + 20.0))
    viewer._on_release(_evt((x0 + x1) / 2.0 + 20.0, (y0 + y1) / 2.0 + 20.0))
    assert viewer.center == center


def test_right_drag_pans_the_view(viewer):
    viewer.set_image(_display())
    before = viewer.center
    viewer._on_pan_press(_evt(100, 100))
    viewer._on_pan_drag(_evt(130, 110))
    viewer._on_pan_release(_evt(130, 110))
    assert viewer.center == (before[0] + 30.0, before[1] + 10.0)

# --------------------------------------------------------------------------- #
# Composite overlay                                                             #
# --------------------------------------------------------------------------- #
def _composite_data():
    return {"q_edges": [0.0, 100.0, 200.0], "spans": [(0, 100), (100, 200)],
            "kx": 0.0, "count": 2, "order": [2], "window": (0.0, 60.0)}


def test_composite_overlay_hit_tests_lanes_and_boundaries(viewer):
    viewer.set_image(_display())
    viewer.set_composite(_composite_data())
    assert viewer.composite_on
    bx = viewer.center[0] + viewer.scale * 100.0
    by = viewer.center[1] + viewer.scale * 30.0
    assert viewer._hit_edge(bx, by) == 1                     # the middle boundary
    assert viewer._lane_at_qx(50.0) == 1
    assert viewer._lane_at_qx(150.0) == 2
    assert viewer._lane_at_qx(-50.0) is None                 # outside the lanes


def test_composite_click_dispatch(viewer):
    viewer.set_image(_display())
    events = []
    viewer.bind_composite_edit(lane=lambda n: events.append(("lane", n)),
                               split=lambda x: events.append(("split", x)),
                               join=lambda i: events.append(("join", i)))
    viewer.set_composite(_composite_data())
    by = viewer.center[1] + viewer.scale * 30.0
    lane_x = viewer.center[0] + viewer.scale * 50.0
    edge_x = viewer.center[0] + viewer.scale * 100.0
    viewer._on_press(_evt(lane_x, by))
    assert events[-1] == ("lane", 1)
    viewer._on_middle_click(_evt(edge_x, by))
    assert events[-1] == ("join", 1)
    viewer._on_middle_click(_evt(lane_x, by))
    assert events[-1][0] == "split"


def test_composite_mode_draws_without_the_rect_crop_handles(viewer):
    viewer.set_image(_display())
    viewer.set_crop((10.0, 10.0, 40.0, 60.0))
    rect_items = len(viewer.canvas.find_all())
    viewer.set_composite(_composite_data())
    # composite replaces the yellow rect + 8 handles with its own overlay
    assert viewer.composite_on
    assert len(viewer.canvas.find_all()) != rect_items

def test_composite_click_above_the_window_does_not_select(viewer):
    viewer.set_image(_display())
    events = []
    viewer.bind_composite_edit(lane=lambda n: events.append(n))
    viewer.set_composite(_composite_data())  # window rows at q 0..60
    lane_x = viewer.center[0] + viewer.scale * 50.0
    above = viewer.center[1] + viewer.scale * (-30.0)   # above the top yellow row
    inside = viewer.center[1] + viewer.scale * 30.0
    viewer._on_press(_evt(lane_x, above))
    assert events == []
    viewer._on_press(_evt(lane_x, inside))
    assert events == [1]


def test_composite_window_row_hit_and_drag_dispatch(viewer):
    viewer.set_image(_display())
    events = []
    viewer.bind_composite_edit(window=lambda side, y: events.append((side, y)))
    viewer.set_composite(_composite_data())  # window rows at q y 0 and 60
    top = viewer.center[1] + viewer.scale * 0.0
    bottom = viewer.center[1] + viewer.scale * 60.0
    assert viewer._hit_window_row(top) == "top"
    assert viewer._hit_window_row(bottom) == "bottom"
    assert viewer._hit_window_row(viewer.center[1] + viewer.scale * 30.0) is None
    free_x = viewer.center[0] + viewer.scale * 250.0  # away from any lane boundary
    viewer._on_press(_evt(free_x, top))
    assert viewer._window_drag == "top"
    drop = viewer.center[1] + viewer.scale * 10.0
    viewer._on_drag(_evt(free_x, drop))
    assert events == []                       # no calculation/commit during the drag
    viewer._on_release(_evt(free_x, drop))
    assert events and events[-1][0] == "top"
    assert events[-1][1] == pytest.approx(10.0)  # committed q-y on release
    assert viewer._window_drag is None


def test_cursor_changes_over_bounds(viewer):
    viewer.set_image(_display())
    viewer.set_composite(_composite_data())
    edge_x = viewer.center[0] + viewer.scale * 100.0
    by = viewer.center[1] + viewer.scale * 30.0
    assert viewer._cursor_for(edge_x, by) == viewer._CURSOR_H
    free_x = viewer.center[0] + viewer.scale * 250.0
    top = viewer.center[1] + viewer.scale * 0.0
    assert viewer._cursor_for(free_x, top) == viewer._CURSOR_V
    assert viewer._cursor_for(free_x, by) == ""


def test_composite_edge_drag_passes_corrected_x(viewer):
    viewer.set_image(_display())
    data = _composite_data()
    data["kx"] = 25.0
    got = []
    viewer.bind_composite_edit(move=lambda i, x: got.append((i, x)))
    viewer.set_composite(data)
    edge_x = viewer.center[0] + viewer.scale * 100.0
    by = viewer.center[1] + viewer.scale * 30.0
    viewer._on_press(_evt(edge_x, by))
    assert viewer._edge_drag == 1
    drop = edge_x + viewer.scale * 10.0
    viewer._on_drag(_evt(drop, by))
    assert got == []                          # boundary moves visually only during the drag
    viewer._on_release(_evt(drop, by))
    assert got and got[-1][0] == 1
    assert got[-1][1] == pytest.approx(135.0)  # q=110 + kx=25, committed on release


def test_overlay_only_render_keeps_the_image_item(viewer):
    viewer.set_image(_display())
    viewer.set_crop((10.0, 10.0, 40.0, 60.0))
    item = viewer._img_item
    assert item is not None
    x0, y0, x1, y1 = viewer._crop_screen_rect()
    viewer._on_press(_evt(x0, (y0 + y1) / 2.0))
    viewer._on_drag(_evt(x0 + 10.0, (y0 + y1) / 2.0))
    assert viewer._img_item == item           # image not recomposited mid-drag
    assert item in viewer.canvas.find_all()


# --------------------------------------------------------------------------- #
# Render coalescing (image-switch latency)                                     #
# --------------------------------------------------------------------------- #
def _count_renders(viewer):
    calls = []
    real = viewer._render

    def counted(*args, **kwargs):
        if viewer._batch == 0:
            calls.append(1)
        real(*args, **kwargs)
    viewer._render = counted
    return calls


def test_pan_slides_items_without_re_rendering(viewer):
    viewer.set_image(_display())
    calls = _count_renders(viewer)
    viewer._on_pan_press(_evt(100, 100))
    viewer._on_pan_drag(_evt(110, 108))       # first motion renders the margin once
    after_first = len(calls)
    for i in range(2, 8):                      # further small motions: move items only
        viewer._on_pan_drag(_evt(100 + i, 100 + i))
    assert len(calls) == after_first           # no re-render per motion
    viewer._on_pan_release(_evt(0, 0))
    assert len(calls) == after_first + 1        # one re-center on release


def test_batch_coalesces_several_setters_into_one_render(viewer):
    viewer.set_image(_display())
    calls = _count_renders(viewer)
    with viewer.batch():
        viewer.set_rotation(1.0)
        viewer.set_rotation(2.0)
        viewer.set_overlays(None, ())
        viewer.clear_crop()
    assert len(calls) == 1  # four setter renders collapse to one


def test_set_composite_none_when_already_none_does_not_render(viewer):
    viewer.set_image(_display())
    calls = _count_renders(viewer)
    viewer.set_composite(None)  # already None -> no render (fires every switch)
    assert calls == []