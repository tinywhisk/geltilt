from __future__ import annotations

import numpy as np
import pytest

from geltilt.app import lanes
from geltilt.app.lanes import LaneSpecError


def _three_lane_mask():
    mask = np.zeros((40, 120), np.uint8)
    mask[10:20, 10:20] = 1   # lane 1
    mask[10:20, 50:60] = 1   # lane 2
    mask[10:20, 90:100] = 1  # lane 3
    return mask


# --- C1: grouping always makes progress ------------------------------------ #
def test_c1_one_pixel_blobs_group_without_hanging():
    mask = np.zeros((20, 20), np.uint8)
    for x, y in [(5, 5), (5, 10), (5, 15), (10, 5), (10, 10)]:
        mask[y, x] = 1
    blobs, _ = lanes.band_blobs(mask, min_area=1)
    grouped = lanes.label_lanes(blobs)
    assert 0 < len(grouped) <= len(blobs)


# --- C2: the area floor ---------------------------------------------------- #
def test_c2_blobs_below_min_area_are_dropped_and_counted():
    mask = np.zeros((20, 40), np.uint8)
    mask[0:10, 0:10] = 1   # area 100
    mask[0:2, 20:25] = 1   # area 10
    blobs, dropped = lanes.band_blobs(mask, min_area=50)
    assert len(blobs) == 1
    assert dropped == 1


# --- C3: pitch and empty-lane insertion ------------------------------------ #
def test_c3_gap_of_two_pitches_inserts_one_lane():
    full, pitch, inserted = lanes.fill_empty_lanes([0.0, 20.0, 40.0, 80.0])
    assert pitch == pytest.approx(20.0)
    assert inserted == 1
    assert full == [0.0, 20.0, 40.0, 60.0, 80.0]


def test_c3_gap_of_2_62_pitches_inserts_none():
    _full, _pitch, inserted = lanes.fill_empty_lanes([0.0, 20.0, 40.0, 92.4])
    assert inserted == 0


def test_c3_three_detected_lane_limitation_is_pinned():
    # known limitation: evenly spaced detected lanes hide an empty lane between them
    full, _pitch, inserted = lanes.fill_empty_lanes([0.0, 100.0, 200.0])
    assert full == [0.0, 100.0, 200.0]
    assert inserted == 0


# --- C4/C5/C6: names, spans, shared boundaries ----------------------------- #
def test_c4_bands_named_and_spans_tile_edges():
    layout = lanes.lane_layout(_three_lane_mask(), (0, 0, 120, 40))
    assert layout["count"] == 3
    names = [b["name"] for lane in layout["lanes"] for b in lane]
    assert names == ["L1-1", "L2-1", "L3-1"]
    assert layout["spans"] == lanes.spans_from_edges(layout["edges"])
    assert len(layout["edges"]) == layout["count"] + 1


def test_c6_neighbours_share_their_boundary():
    layout = lanes.lane_layout(_three_lane_mask(), (0, 0, 120, 40))
    spans = layout["spans"]
    for left, right in zip(spans, spans[1:]):
        assert left[1] == right[0]


def test_c5_spans_are_centred_on_their_lane():
    layout = lanes.lane_layout(_three_lane_mask(), (0, 0, 120, 40))
    for (x0, x1), centre in zip(layout["spans"], layout["centres"]):
        midpoint = (x0 + x1) / 2.0
        nearest = min(layout["centres"], key=lambda c: abs(c - midpoint))
        assert nearest == centre


# --- C7: adjacent lanes are one block -------------------------------------- #
def test_c7_stitching_neighbours_equals_the_contiguous_slice():
    arr = np.arange(40 * 120, dtype=np.uint16).reshape(40, 120)
    layout = lanes.lane_layout(_three_lane_mask(), (0, 0, 120, 40))
    edges = layout["edges"]
    stitched = lanes.stitch_lanes(arr, edges, [[1, 2]])
    assert np.array_equal(stitched, arr[:, edges[0]:edges[2]])


# --- C8: splitting, joining and renumbering -------------------------------- #
def test_c8_insert_edge_refusals_and_success():
    edges = [0, 50, 100]
    assert lanes.insert_edge(edges, 3, min_side=8)[1] is None    # too close
    assert lanes.insert_edge(edges, 105, min_side=8)[1] is None  # outside region
    new, index = lanes.insert_edge(edges, 25, min_side=8)
    assert index == 1 and new == [0, 25, 50, 100]


def test_c8_remove_edge_refusals_and_success():
    assert lanes.remove_edge([0, 50, 100], 1)[1] is None      # only two lanes
    assert lanes.remove_edge([0, 30, 60, 90], 0)[1] is None   # outer boundary
    assert lanes.remove_edge([0, 30, 60, 90], 3)[1] is None   # outer boundary
    new, index = lanes.remove_edge([0, 30, 60, 90], 2)
    assert index == 2 and new == [0, 30, 90]


def test_move_edge_moves_inner_and_outer_boundaries():
    assert lanes.move_edge([0, 50, 100], 1, 60, min_side=8) == [0, 60, 100]
    assert lanes.move_edge([0, 50, 100], 1, 95, min_side=8) == [0, 92, 100]   # clamp to neighbour
    assert lanes.move_edge([0, 50, 100], 1, 5, min_side=8) == [0, 8, 100]
    assert lanes.move_edge([0, 50, 100], 1, 200, min_side=8) == [0, 92, 100]
    # outer boundaries move, bounded by 0 / width
    assert lanes.move_edge([10, 50, 90], 0, -20, width=200, min_side=8) == [0, 50, 90]
    assert lanes.move_edge([10, 50, 90], 2, 500, width=200, min_side=8) == [10, 50, 200]


def test_c8_renumbering_after_edits():
    assert lanes.renumber_after_split([1, 2, 3], 1) == [1, 3, 4]
    assert lanes.renumber_after_merge([1, 2, 3], 1) == [1, 2]
    assert lanes.renumber_after_merge([1, 3], 1) == [1, 2]


# --- C9: averaging the widths ---------------------------------------------- #
def test_c9_even_edges_holds_outer_bounds_and_is_idempotent():
    once = lanes.even_edges([0, 10, 17, 30])
    assert once == [0, 10, 20, 30]
    assert once[0] == 0 and once[-1] == 30
    assert lanes.even_edges(once) == once  # second press: no change


def test_c9_even_edges_noop_when_too_narrow():
    assert lanes.even_edges([0, 1, 1]) == [0, 1, 1]


# --- C10: boundaries nudged off the bands ---------------------------------- #
def test_c10_nudge_moves_a_boundary_out_of_a_band():
    blobs = [{"x0": 38, "x1": 44, "y0": 0, "y1": 5}]
    edges, nudged = lanes.nudge_off_bands([0, 40, 80, 120], blobs, min_side=8)
    assert nudged == 1
    for x in edges[1:-1]:
        assert not any(b["x0"] < x < b["x1"] for b in blobs)


def test_c10_leaves_a_boundary_that_cannot_clear_without_crossing():
    blobs = [{"x0": 0, "x1": 200, "y0": 0, "y1": 5}]  # wider than the whole region
    edges, nudged = lanes.nudge_off_bands([0, 30, 70, 120], blobs, min_side=8)
    assert nudged == 0
    assert edges == [0, 30, 70, 120]


def test_c10_nudge_clears_a_boundary_wedged_between_overlapping_bands():
    # boundary 50 cuts band A [47,53]; A's edges 47 is inside band B [40,48] — the old
    # two-option rule oscillated 47<->48 forever; the all-edges rule jumps to 53 instead
    blobs = [{"x0": 47, "x1": 53, "y0": 0, "y1": 5},
             {"x0": 40, "x1": 48, "y0": 0, "y1": 5}]
    edges, nudged = lanes.nudge_off_bands([0, 50, 90, 140], blobs, min_side=8)
    assert nudged >= 1
    assert not any(b["x0"] < edges[1] < b["x1"] for b in blobs)


def test_c10_no_internal_boundary_lands_inside_a_blob_after_nudging():
    blobs = [{"x0": x, "x1": x + 6, "y0": 0, "y1": 5}
             for x in range(10, 130, 20)]
    edges, _nudged = lanes.nudge_off_bands(list(range(0, 150, 12)), blobs, min_side=8)
    for x in edges[1:-1]:
        assert not any(b["x0"] < x < b["x1"] for b in blobs)


def test_expand_mask_horizontally_widens_left_and_right_only():
    m = np.zeros((10, 20), np.uint8)
    m[4:6, 8:12] = 1
    e = lanes.expand_mask_horizontally(m, 4)
    xs = np.where(e.any(axis=0))[0]
    ys = np.where(e.any(axis=1))[0]
    assert (int(xs.min()), int(xs.max())) == (4, 15)  # [8,12) -> [4,16)
    assert list(map(int, ys)) == [4, 5]               # no vertical change


def test_lane_layout_exposes_widened_nudge_blobs():
    layout = lanes.lane_layout(_three_lane_mask(), (0, 0, 120, 40))
    blobs = sorted(layout["blobs"], key=lambda b: b["x0"])
    nudge = sorted(layout["nudge_blobs"], key=lambda b: b["x0"])
    assert len(nudge) == len(blobs) == 3
    for b, nb in zip(blobs, nudge):
        assert (b["x0"] - nb["x0"], nb["x1"] - b["x1"]) == (4, 4)


def test_nudging_against_the_widened_mask_clears_a_boundary_the_raw_mask_allows():
    raw = [{"x0": 100, "x1": 110, "y0": 0, "y1": 5}]
    widened = [{"x0": 96, "x1": 116, "y0": 0, "y1": 5}]
    edges = [0, 114, 200, 300]
    # 114 is clear of the raw mask, so no nudge would happen with it
    assert not any(b["x0"] < 114 < b["x1"] for b in raw)
    assert lanes.nudge_off_bands(edges, raw, min_side=8)[1] == 0
    # but it cuts the widened mask, so nudging moves it clear
    moved, nudged = lanes.nudge_off_bands(edges, widened, min_side=8)
    assert nudged >= 1
    assert not any(b["x0"] < moved[1] < b["x1"] for b in widened)


# --- C11: the count override ----------------------------------------------- #
def test_c11_count_override_forces_even_spacing():
    layout2 = lanes.lane_layout(_three_lane_mask(), (0, 0, 120, 40), count=2)
    layout6 = lanes.lane_layout(_three_lane_mask(), (0, 0, 120, 40), count=6)
    assert layout2["count"] == 2
    assert layout6["count"] == 6
    assert len(layout6["centres"]) == 6


# --- C12: empty and degenerate inputs -------------------------------------- #
def test_c12_degenerate_inputs_return_without_raising():
    empty = lanes.lane_layout(np.zeros((40, 120), np.uint8), (0, 0, 120, 40))
    assert empty["count"] == 0
    assert empty["edges"] == []  # §16: constant shape (edges key present)

    outside = lanes.lane_layout(_three_lane_mask(), (500, 500, 600, 600))
    assert outside["count"] == 0

    impossible = lanes.lane_layout(_three_lane_mask(), (0, 0, 120, 40), min_area=10 ** 9)
    assert impossible["count"] == 0

    narrow = lanes.lane_layout(_three_lane_mask(), (0, 0, 20, 40))
    assert narrow["count"] == 1  # a one-lane-wide region is sane, not a crash


# --- C13: parsing the lane spec -------------------------------------------- #
def test_c13_valid_specs_produce_the_written_order():
    assert lanes.parse_lane_spec("3,1-2,4-6", 6) == [[3], [1, 2], [4, 5, 6]]
    assert lanes.parse_lane_spec("L3,L1-L2", 3) == [[3], [1, 2]]
    assert lanes.parse_lane_spec("3", 3) == [[3]]
    assert lanes.spec_groups([[3], [1, 2]]) == [3, 1, 2]


@pytest.mark.parametrize("text,n_lanes,key", [
    ("", 3, "lane_spec.empty"),
    ("   ", 3, "lane_spec.empty"),
    ("{1,2}", 3, "lane_spec.braces"),
    ("1,,2", 3, "lane_spec.empty_term"),
    ("x", 3, "lane_spec.unparsed"),
    ("-2", 3, "lane_spec.unparsed"),
    ("0", 3, "lane_spec.not_positive"),
    ("5", 3, "lane_spec.out_of_range"),
    ("3-1", 3, "lane_spec.reversed"),
])
def test_c13_error_cases_raise_their_exact_key(text, n_lanes, key):
    with pytest.raises(LaneSpecError) as exc:
        lanes.parse_lane_spec(text, n_lanes)
    assert exc.value.key == key


# --- C14: the composite is the written order, right pixels, original depth -- #
def test_c14_stitch_is_order_pixels_and_depth():
    arr = np.arange(10 * 41, dtype=np.uint16).reshape(10, 41)
    edges = [0, 10, 20, 30, 40]
    out = lanes.stitch_lanes(arr, edges, [[3], [1, 2], [4]])
    assert out.shape == (10, 40)
    assert out.dtype == np.uint16


def test_c14_duplicate_lane_yields_one_copy():
    arr = np.arange(10 * 41, dtype=np.uint16).reshape(10, 41)
    out = lanes.stitch_lanes(arr, [0, 10, 20, 30, 40], [[2, 2]])
    assert out.shape == (10, 10)


def test_c14_skip_gives_two_slices():
    arr = np.arange(10 * 41, dtype=np.uint16).reshape(10, 41)
    out = lanes.stitch_lanes(arr, [0, 10, 20, 30, 40], [[1], [3]])
    assert out.shape == (10, 20)


def test_c14_three_d_input_raises():
    with pytest.raises(ValueError):
        lanes.stitch_lanes(np.zeros((4, 4, 3), np.uint8), [0, 2, 4], [[1]])


# --- §8 window + §9 runs ---------------------------------------------------- #
def test_composite_rows_follows_the_crop_in_image_pixels():
    assert lanes.composite_rows(None, 500) == (0, 500)
    assert lanes.composite_rows((0, 40, 100, 120), 500) == (40, 120)
    assert lanes.composite_rows((0, -5, 100, 999), 500) == (0, 500)


def test_output_runs_groups_consecutive_lanes():
    assert lanes.output_runs([[3], [1, 2], [4]]) == [[3], [1, 2], [4]]
    assert lanes.output_runs([[1], [2], [3]]) == [[1, 2, 3]]
    assert lanes.output_runs([[1], [3]]) == [[1], [3]]