from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import geltilt.app.export as export_module
from geltilt.app import lanes as lane_mod
from geltilt.app.controller import AppController
from geltilt.app.domain import AppliedRotation
from geltilt.app.log import JsonlDiagnosticLog

from .conftest import FakeEstimator, FakeRoot, make_outcome, pump


class _RecordingViewer:
    """Captures the crop calls the controller makes into the viewer."""

    def __init__(self) -> None:
        self.crops = []
        self.cleared = 0
        self.change_cb = None

    def set_image(self, _u8, reset: bool = True) -> None:
        pass

    def set_rotation(self, _angle: float) -> None:
        pass

    def set_overlays(self, _centroids=None, _lanes=None) -> None:
        pass

    def bind_crop_change(self, callback) -> None:
        self.change_cb = callback

    def set_crop(self, rect) -> None:
        self.crops.append(tuple(rect))

    def clear_crop(self) -> None:
        self.cleared += 1


def _bands(bbox, mask=None):
    """Fake ``extract_bands``: returns (band mask, bbox)."""
    m = np.zeros((60, 80), np.uint8) if mask is None else mask
    return lambda _arr: (m, bbox)


def _make_controller(app_config, estimator, errors=None, warnings=None,
                     viewer=None, extract_bands=None, crop_states=None) -> AppController:
    controller = AppController(FakeRoot(), app_config, estimator=estimator,
                               extract_bands=extract_bands)
    controller.attach(
        None, viewer, None, None,
        error_cb=lambda t, m: (errors if errors is not None else []).append((t, m)),
        warn_cb=lambda t, m: (warnings if warnings is not None else []).append((t, m)),
        crop_cb=(lambda a, h: crop_states.append((a, h))) if crop_states is not None else None,
    )
    return controller


# --------------------------------------------------------------------------- #
# Browsing and navigation                                                      #
# --------------------------------------------------------------------------- #
def test_open_folder_lists_in_ascending_order(app_config, sample_folder):
    controller = _make_controller(app_config, FakeEstimator())
    assert controller.open_folder(sample_folder) is True
    assert [p.name for p in controller.samples] == ["a_gel.png", "b_gel.png", "c_gel.png"]
    assert controller.index == 0
    assert controller.folder == sample_folder
    assert controller.session_store.load().folder == str(sample_folder)


def test_open_empty_folder_keeps_prior_state(app_config, sample_folder, tmp_path):
    warnings = []
    controller = _make_controller(app_config, FakeEstimator(), warnings=warnings)
    controller.open_folder(sample_folder)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert controller.open_folder(empty) is False
    assert controller.folder == sample_folder
    assert warnings, "expected a warning listing supported extensions"


def test_open_missing_folder_reports_error(app_config, tmp_path):
    errors = []
    controller = _make_controller(app_config, FakeEstimator(), errors=errors)
    assert controller.open_folder(tmp_path / "does-not-exist") is False
    assert errors and "Not a folder" in errors[0][1]


def test_navigation_clamps_without_wrap(app_config, sample_folder):
    controller = _make_controller(app_config, FakeEstimator())
    controller.open_folder(sample_folder)
    controller.prev_sample()
    assert controller.index == 0
    controller.goto(99)
    assert controller.index == 2
    controller.next_sample()
    assert controller.index == 2
    controller.goto(-5)
    assert controller.index == 0


def test_per_sample_state_restored_after_navigation(app_config, sample_folder):
    controller = _make_controller(app_config, FakeEstimator())
    controller.open_folder(sample_folder)
    controller.set_manual_angle(2.5)
    controller.goto(1)
    assert controller.state.get("b_gel.png").applied.value == 0.0
    controller.goto(0)
    assert controller.state.get("a_gel.png").applied.value == 2.5
    assert controller.state.get("a_gel.png").applied.source == "manual"


def test_session_survives_restart(app_config, sample_folder):
    controller = _make_controller(app_config, FakeEstimator())
    controller.open_folder(sample_folder)
    controller.goto(2)
    controller.set_manual_angle(-1.7)

    reopened = _make_controller(app_config, FakeEstimator())
    reopened.start()  # restores folder + index + rotations from the session file
    assert reopened.folder == sample_folder
    assert reopened.index == 2
    assert reopened.state.get("c_gel.png").applied.value == -1.7
    assert reopened.state.get("c_gel.png").applied.source == "manual"


# --------------------------------------------------------------------------- #
# Auto-rotation                                                                #
# --------------------------------------------------------------------------- #
def test_auto_rotate_applies_canonical_angle_as_is(app_config, sample_folder):
    estimator = FakeEstimator(make_outcome(ok=True, angle=2.4))
    controller = _make_controller(app_config, estimator)
    controller.open_folder(sample_folder)
    controller.trigger_auto_rotate()
    pump(controller)
    applied = controller.state.get("a_gel.png").applied
    assert applied.source == "auto"
    assert applied.value == pytest.approx(2.4)  # never negated
    assert estimator.calls == 1
    # one run-log line written
    lines = JsonlDiagnosticLog(app_config.run_dir).path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


def test_auto_rotate_refusal_leaves_image_unrotated(app_config, sample_folder):
    estimator = FakeEstimator(make_outcome(ok=False, angle=float("nan"),
                                           reason="fewer than 4 bands"))
    controller = _make_controller(app_config, estimator)
    controller.open_folder(sample_folder)
    controller.set_manual_angle(3.0)
    controller.trigger_auto_rotate()
    pump(controller)
    applied = controller.state.get("a_gel.png").applied
    assert applied.source == "manual"  # prior angle preserved, no silent 0 deg
    assert applied.value == pytest.approx(3.0)
    assert controller.state.get("a_gel.png").last_estimate.ok is False


def test_auto_rotate_replaces_manual_and_is_deterministic(app_config, sample_folder):
    estimator = FakeEstimator(make_outcome(ok=True, angle=-1.3))
    controller = _make_controller(app_config, estimator)
    controller.open_folder(sample_folder)
    controller.set_manual_angle(5.0)
    controller.trigger_auto_rotate()
    pump(controller)
    assert controller.state.get("a_gel.png").applied.source == "auto"
    assert controller.state.get("a_gel.png").applied.value == pytest.approx(-1.3)

    controller.trigger_auto_rotate()
    pump(controller)
    assert controller.state.get("a_gel.png").applied.value == pytest.approx(-1.3)


def test_manual_edit_supersedes_in_flight_auto(app_config, sample_folder):
    estimator = FakeEstimator(make_outcome(ok=True, angle=4.0), delay=0.2)
    controller = _make_controller(app_config, estimator)
    controller.open_folder(sample_folder)
    controller.trigger_auto_rotate()
    controller.set_manual_angle(1.0)  # bumps rotation_epoch while the worker runs
    pump(controller)
    applied = controller.state.get("a_gel.png").applied
    assert applied.source == "manual"
    assert applied.value == pytest.approx(1.0)  # manual edit not overwritten
    # diagnostics still record the estimate
    assert controller.state.get("a_gel.png").last_estimate is not None


def test_estimator_error_is_isolated(app_config, sample_folder):
    errors = []
    estimator = FakeEstimator(exc=RuntimeError("boom"))
    controller = _make_controller(app_config, estimator, errors=errors)
    controller.open_folder(sample_folder)
    controller.trigger_auto_rotate()
    pump(controller)
    assert errors and "Estimate failed" in errors[0][0]
    # navigation still works after the failure
    controller.next_sample()
    assert controller.index == 1


# --------------------------------------------------------------------------- #
# Manual override                                                              #
# --------------------------------------------------------------------------- #
def test_manual_angle_clamps_and_rounds(app_config, sample_folder):
    controller = _make_controller(app_config, FakeEstimator())
    controller.open_folder(sample_folder)
    controller.set_manual_angle(500.0)
    assert controller.state.get("a_gel.png").applied.value == 180.0
    controller.set_manual_angle(2.36)
    assert controller.state.get("a_gel.png").applied.value == 2.4


def test_nudge_and_zero(app_config, sample_folder):
    controller = _make_controller(app_config, FakeEstimator())
    controller.open_folder(sample_folder)
    controller.nudge(1.0)
    controller.nudge(-0.1)
    assert controller.state.get("a_gel.png").applied.value == pytest.approx(0.9)
    assert controller.state.get("a_gel.png").applied.source == "manual"
    controller.zero_angle()
    assert controller.state.get("a_gel.png").applied.source == "none"
    assert controller.state.get("a_gel.png").applied.value == 0.0


# --------------------------------------------------------------------------- #
# Auto crop                                                                     #
# --------------------------------------------------------------------------- #
def test_auto_crop_gated_until_auto_rotate(app_config, sample_folder):
    controller = _make_controller(app_config, FakeEstimator(make_outcome(ok=True, angle=2.0)),
                                  extract_bands=_bands((5, 5, 25, 30)))
    controller.open_folder(sample_folder)
    assert controller.can_auto_crop() is False
    controller.auto_crop()  # refused: must not set a crop
    assert controller.state.get("a_gel.png").crop is None
    controller.trigger_auto_rotate()
    pump(controller)
    assert controller.can_auto_crop() is True


def test_auto_crop_boxes_bands_and_pushes_rect(app_config, sample_folder):
    viewer = _RecordingViewer()
    controller = _make_controller(app_config, FakeEstimator(make_outcome(ok=True, angle=0.0)),
                                  viewer=viewer, extract_bands=_bands((10, 10, 30, 40)))
    controller.open_folder(sample_folder)
    controller.trigger_auto_rotate()
    pump(controller)
    controller.auto_crop()
    st = controller.state.get("a_gel.png")
    assert st.crop is not None
    r = st.crop.as_tuple()
    # display-frame rect is anchor-relative; size is bbox (20x30) + 10px pad each side
    assert (r[2] - r[0], r[3] - r[1]) == (40.0, 50.0)
    assert viewer.crops[-1] == r


def test_auto_crop_reports_when_no_bands(app_config, sample_folder):
    warnings = []
    controller = _make_controller(app_config, FakeEstimator(make_outcome(ok=True, angle=0.0)),
                                  warnings=warnings, extract_bands=_bands(None))
    controller.open_folder(sample_folder)
    controller.trigger_auto_rotate()
    pump(controller)
    controller.auto_crop()
    assert controller.state.get("a_gel.png").crop is None
    assert warnings


def test_clear_crop_resets_and_notifies(app_config, sample_folder):
    viewer = _RecordingViewer()
    controller = _make_controller(app_config, FakeEstimator(make_outcome(ok=True, angle=0.0)),
                                  viewer=viewer, extract_bands=_bands((5, 5, 25, 30)))
    controller.open_folder(sample_folder)
    controller.trigger_auto_rotate()
    pump(controller)
    controller.auto_crop()
    controller.clear_crop()
    assert controller.state.get("a_gel.png").crop is None
    assert viewer.cleared >= 1


def test_crop_persists_and_viewer_drag_updates_state(app_config, sample_folder):
    viewer = _RecordingViewer()
    controller = _make_controller(app_config, FakeEstimator(make_outcome(ok=True, angle=0.0)),
                                  viewer=viewer, extract_bands=_bands((5, 5, 25, 30)))
    controller.open_folder(sample_folder)
    controller.trigger_auto_rotate()
    pump(controller)
    controller.auto_crop()
    # navigating away and back restores the crop into the viewer
    controller.goto(1)
    controller.goto(0)
    st = controller.state.get("a_gel.png")
    assert st.crop is not None
    r = st.crop.as_tuple()
    assert (r[2] - r[0], r[3] - r[1]) == (40.0, 45.0)  # bbox 20x25 + 10px pad each side
    assert viewer.crops[-1] == r  # restored into the viewer on return
    # a viewer drag feeds the new rect back into the per-sample state
    viewer.change_cb((1.0, 2.0, 3.0, 4.0))
    assert controller.state.get("a_gel.png").crop.as_tuple() == (1.0, 2.0, 3.0, 4.0)


def test_crop_state_callback_reports_availability(app_config, sample_folder):
    states = []
    controller = _make_controller(app_config, FakeEstimator(make_outcome(ok=True, angle=0.0)),
                                  extract_bands=_bands((5, 5, 25, 30)), crop_states=states)
    controller.open_folder(sample_folder)
    assert states[-1] == (False, False)  # not yet auto-rotated
    controller.trigger_auto_rotate()
    pump(controller)
    assert states[-1] == (True, False)
    controller.auto_crop()
    assert states[-1] == (True, True)


# --------------------------------------------------------------------------- #
# Export                                                                        #
# --------------------------------------------------------------------------- #
def test_export_source_follows_use_original(app_config, tmp_path):
    from geltilt.imageio import save_array

    folder = tmp_path / "imgs"
    folder.mkdir()
    arr = (np.arange(32 * 24, dtype=np.uint16).reshape(32, 24) * 100).astype(np.uint16)
    save_array(folder / "g.tif", arr)
    controller = _make_controller(app_config, FakeEstimator(make_outcome(ok=True, angle=0.0)))
    controller.open_folder(folder)
    assert controller.export_array(use_original=True).dtype == np.uint16
    assert controller.export_array(use_original=False).dtype == np.uint8


def test_save_as_writes_a_rotated_file(app_config, sample_folder, tmp_path):
    controller = _make_controller(app_config, FakeEstimator(make_outcome(ok=True, angle=0.0)))
    controller.open_folder(sample_folder)
    dest = tmp_path / "a_gel_corrected.tif"
    assert controller.save_as(dest, use_original=True) is True
    assert dest.exists()


def test_copy_to_clipboard_uses_the_export_array(app_config, sample_folder, monkeypatch):
    seen = {}
    monkeypatch.setattr(export_module, "copy_tiff_to_clipboard",
                        lambda arr: seen.setdefault("shape", arr.shape))
    controller = _make_controller(app_config, FakeEstimator(make_outcome(ok=True, angle=0.0)))
    controller.open_folder(sample_folder)
    assert controller.copy_to_clipboard(use_original=True) is True
    assert seen["shape"] == (32, 24)


def test_export_applies_the_crop_box(app_config, tmp_path):
    from geltilt.imageio import save_array

    folder = tmp_path / "imgs"
    folder.mkdir()
    arr = np.zeros((60, 80), np.uint8)
    arr[20:40, 30:60] = 200
    save_array(folder / "g.png", arr)
    controller = _make_controller(app_config, FakeEstimator(make_outcome(ok=True, angle=0.0)),
                                  extract_bands=_bands((30, 20, 60, 40)))
    controller.open_folder(folder)
    controller.trigger_auto_rotate()
    pump(controller)
    controller.auto_crop()
    full = controller.export_array(use_original=True, apply_crop=False)
    cropped = controller.export_array(use_original=True, apply_crop=True)
    assert full.shape == (60, 80)
    assert cropped.shape == (40, 50)  # band bbox 30x20 + 10px pad on each side


def test_export_without_an_image_warns(app_config):
    warnings = []
    controller = _make_controller(app_config, FakeEstimator(), warnings=warnings)
    assert controller.save_as("x.tif", use_original=True) is False
    assert controller.copy_to_clipboard(use_original=True) is False
    assert warnings


# --------------------------------------------------------------------------- #
# Composite crop                                                                #
# --------------------------------------------------------------------------- #
def _composite_folder(tmp_path):
    from geltilt.imageio import save_array

    folder = tmp_path / "imgs"
    folder.mkdir()
    img = np.zeros((60, 80), np.uint8)
    img[20:40, 5:65] = 120
    save_array(folder / "g.png", img)
    mask = np.zeros((60, 80), np.uint8)
    mask[20:40, 5:15] = 1
    mask[20:40, 30:40] = 1
    mask[20:40, 55:65] = 1
    return folder, mask


def test_composite_needs_autocrop(app_config, sample_folder):
    controller = _make_controller(app_config, FakeEstimator(make_outcome(ok=True, angle=0.0)),
                                  extract_bands=_bands((5, 5, 25, 30)))
    controller.open_folder(sample_folder)
    assert controller.composite_selectable() is False
    assert controller.set_composite(True) is False
    assert controller.composite_on is False


def test_composite_gate_export_and_refusal(app_config, tmp_path):
    folder, mask = _composite_folder(tmp_path)
    controller = _make_controller(app_config, FakeEstimator(make_outcome(ok=True, angle=0.0)),
                                  extract_bands=lambda a: (mask, (5, 20, 65, 40)))
    controller.open_folder(folder)
    controller.trigger_auto_rotate()
    pump(controller)
    controller.auto_crop()
    assert controller.composite_selectable() is True
    assert controller.composite_ready() is True

    assert controller.set_composite(True) is True
    assert controller.composite_active() is False  # no request yet
    controller.set_composite_spec("3,1-2")
    assert controller.composite_active() is True

    out = controller.export_array(use_original=True, apply_crop=True)
    edges = controller.layout["edges"]
    expected_w = sum(edges[r[-1]] - edges[r[0] - 1]
                     for r in lane_mod.output_runs(controller.composite_groups))
    assert out.shape == (40, expected_w)  # crop rows x stitched lanes
    assert out.dtype == np.uint8

    # "Rotation only" with composite active is refused
    assert controller.composite_ignored(False) is True
    assert controller.export_array(use_original=True, apply_crop=False) is None
    warnings = []
    controller.attach(None, None, None, None,
                      warn_cb=lambda t, m: warnings.append((t, m)))
    assert controller.save_as("x.tif", use_original=True, apply_crop=False) is False
    assert warnings


def test_composite_click_lane_rewrites_the_flat_order(app_config, tmp_path):
    folder, mask = _composite_folder(tmp_path)
    controller = _make_controller(app_config, FakeEstimator(make_outcome(ok=True, angle=0.0)),
                                  extract_bands=lambda a: (mask, (5, 20, 65, 40)))
    controller.open_folder(folder)
    controller.trigger_auto_rotate()
    pump(controller)
    controller.auto_crop()
    controller.set_composite(True)
    controller.set_composite_spec("1")
    controller.click_lane(3)
    assert controller.composite_text == "1,3"
    assert controller.composite_groups == [[1], [3]]
    controller.click_lane(1)
    assert controller.composite_text == "3"


def test_composite_split_and_join_renumber(app_config, tmp_path):
    folder, mask = _composite_folder(tmp_path)
    controller = _make_controller(app_config, FakeEstimator(make_outcome(ok=True, angle=0.0)),
                                  extract_bands=lambda a: (mask, (5, 20, 65, 40)))
    controller.open_folder(folder)
    controller.trigger_auto_rotate()
    pump(controller)
    controller.auto_crop()
    controller.set_composite(True)
    controller.set_composite_spec("1,2")
    edges_before = list(controller.layout["edges"])
    x = (edges_before[0] + edges_before[1]) / 2.0
    controller.split_lane_at(x)
    assert controller.layout["count"] == 4  # 3 lanes -> 4 after the split
    controller.join_edge(1)
    assert controller.layout["count"] == 3  # back down after the join


def test_composite_window_and_outer_bounds_resize_the_crop(app_config, tmp_path):
    folder, mask = _composite_folder(tmp_path)
    controller = _make_controller(app_config, FakeEstimator(make_outcome(ok=True, angle=0.0)),
                                  extract_bands=lambda a: (mask, (5, 20, 65, 40)))
    controller.open_folder(folder)
    controller.trigger_auto_rotate()
    pump(controller)
    controller.auto_crop()
    controller.set_composite(True)
    kx_before = controller._frame_kx()
    orig = controller.state.get("g.png").crop

    # the right outer boundary may move past the simple crop's right edge
    controller.move_edge(len(controller.layout["edges"]) - 1, 10 ** 6)
    st = controller.state.get("g.png")
    assert st.crop.x1 > orig.x1
    assert st.crop.x1 == pytest.approx(controller.layout["edges"][-1] - kx_before)
    assert controller._frame_kx() == pytest.approx(kx_before)  # offset is constant

    # dragging the top window row moves the crop's top edge
    y0 = controller.state.get("g.png").crop.y0
    controller.set_window("top", y0 + 25)
    assert controller.state.get("g.png").crop.y0 == pytest.approx(y0 + 25)


def test_even_widths_nudges_against_the_widened_mask(app_config, tmp_path):
    folder, mask = _composite_folder(tmp_path)
    controller = _make_controller(app_config, FakeEstimator(make_outcome(ok=True, angle=0.0)),
                                  extract_bands=lambda a: (mask, (5, 20, 65, 40)))
    controller.open_folder(folder)
    controller.trigger_auto_rotate()
    pump(controller)
    controller.auto_crop()
    controller.set_composite(True)
    # a boundary at 114 is clear of the raw blob but cuts the 4px-widened one
    controller.layout["blobs"] = [{"x0": 100, "x1": 110, "y0": 0, "y1": 5, "cx": 105, "cy": 2}]
    controller.layout["nudge_blobs"] = [{"x0": 96, "x1": 116, "y0": 0, "y1": 5, "cx": 106, "cy": 2}]
    controller.layout["edges"] = [40, 114, 160, 260]
    controller.even_widths()
    x = controller.layout["edges"][1]
    assert not any(b["x0"] < x < b["x1"] for b in controller.layout["nudge_blobs"])


# --------------------------------------------------------------------------- #
# Failure isolation and source integrity                                       #
# --------------------------------------------------------------------------- #
def test_corrupt_sample_fails_without_corrupting_others(app_config, sample_folder):
    (sample_folder / "d_broken.png").write_bytes(b"not an image")
    errors = []
    controller = _make_controller(app_config, FakeEstimator(), errors=errors)
    controller.open_folder(sample_folder)
    controller.goto(3)  # d_broken.png (sorted last)
    assert controller.state.get("d_broken.png").failed is True
    assert errors  # surfaced
    # other samples unaffected
    controller.goto(0)
    assert controller.state.get("a_gel.png").failed is False


def test_source_images_are_never_modified(app_config, sample_folder):
    before = {p.name: p.read_bytes() for p in sample_folder.iterdir() if p.is_file()}
    controller = _make_controller(app_config, FakeEstimator(make_outcome(angle=2.0)))
    controller.open_folder(sample_folder)
    controller.goto(1)
    controller.trigger_auto_rotate()
    pump(controller)
    controller.set_manual_angle(3.0)
    controller.zero_angle()
    after = {p.name: p.read_bytes() for p in sample_folder.iterdir() if p.is_file()}
    assert before == after
    # no corrected/cropped image files written into the run directory
    written = {p.suffix.lower() for p in app_config.run_dir.iterdir()}
    assert written.isdisjoint({".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"})