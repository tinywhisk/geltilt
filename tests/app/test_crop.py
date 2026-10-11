from __future__ import annotations

import pytest

from geltilt.app.crop import (
    display_rect_to_corrected_box,
    normalize_rect,
    pad_rect,
    screen_rect,
    screen_to_display,
    source_bbox_to_display_rect,
)


def test_pad_rect_grows_all_sides():
    assert pad_rect((10.0, 20.0, 30.0, 40.0), 5.0) == (5.0, 15.0, 35.0, 45.0)


def test_source_bbox_to_display_rect_identity_at_zero_angle():
    rect = source_bbox_to_display_rect((10.0, 20.0, 30.0, 40.0), 0.0, (0.0, 0.0))
    assert rect == (10.0, 20.0, 30.0, 40.0)


def test_source_bbox_to_display_rect_swaps_axes_at_90_degrees():
    rect = source_bbox_to_display_rect((0.0, 0.0, 10.0, 20.0), 90.0, (0.0, 0.0))
    assert rect == pytest.approx((0.0, -10.0, 20.0, 0.0))


def test_screen_rect_and_inverse_roundtrip():
    assert screen_rect((1.0, 2.0, 3.0, 4.0), (100.0, 50.0), 2.0) == (102.0, 54.0, 106.0, 58.0)
    assert screen_to_display(102.0, 54.0, (100.0, 50.0), 2.0) == (1.0, 2.0)


def test_normalize_rect_orders_corners_and_enforces_minimum():
    assert normalize_rect((10.0, 10.0, 5.0, 5.0), 2.0) == (5.0, 5.0, 10.0, 10.0)
    assert normalize_rect((10.0, 10.0, 11.0, 11.0), 8.0) == (10.0, 10.0, 18.0, 18.0)

def test_display_rect_to_corrected_box_identity_at_zero_angle():
    # q = p - anchor at 0 degrees, and the corrected image is the source at 0 deg
    box = display_rect_to_corrected_box((5.0, 5.0, 25.0, 30.0), 0.0, (100.0, 50.0), 0.0, (200, 300))
    assert box == (105, 55, 125, 80)


def test_display_rect_to_corrected_box_clamps_and_can_be_empty():
    assert display_rect_to_corrected_box(
        (-50.0, -50.0, 10.0, 10.0), 0.0, (0.0, 0.0), 0.0, (40, 40)
    ) == (0, 0, 10, 10)
    assert display_rect_to_corrected_box(
        (500.0, 500.0, 600.0, 600.0), 0.0, (0.0, 0.0), 0.0, (40, 40)
    ) is None