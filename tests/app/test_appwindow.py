from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import pytest

from geltilt.app.estimator import CanonicalEstimator
from geltilt.app.window import AppWindow


def _walk(widget):
    yield widget
    for child in widget.winfo_children():
        yield from _walk(child)


@pytest.fixture
def window(tmp_path):
    try:
        win = AppWindow(run_dir=str(tmp_path / "run"))
    except tk.TclError:  # pragma: no cover - headless
        pytest.skip("no display available for Tk")
    win.withdraw()
    win.update_idletasks()
    try:
        yield win
    finally:
        win.destroy()


def test_window_has_exactly_one_image_surface(window):
    canvases = [w for w in _walk(window) if isinstance(w, tk.Canvas)]
    assert len(canvases) == 1
    assert canvases[0] is window.viewer.canvas


def test_default_estimator_is_canonical(window):
    assert isinstance(window.controller.estimator, CanonicalEstimator)


def _buttons(window):
    return {str(w.cget("text")): w for w in _walk(window) if isinstance(w, ttk.Button)}


def test_reference_layout_buttons_are_present(window):
    texts = set(_buttons(window))
    assert {
        "Load image...", "Auto rotate", "Reset",
        "Auto crop", "Clear crop", "Even widths", "Save as...", "Copy to clipboard",
    } <= texts
    assert "Load folder..." not in texts        # replaced by "Load image..."


def test_load_image_scans_the_picked_images_folder(window, sample_folder, monkeypatch):
    import geltilt.app.window as window_mod

    picked = sample_folder / "b_gel.png"
    monkeypatch.setattr(window_mod.filedialog, "askopenfilename", lambda **k: str(picked))

    window._choose_image()

    assert window.controller.folder == sample_folder          # folder of the pick is scanned
    assert len(window.controller.samples) == 3                # every image in it
    assert window.controller.active_sample().name == "b_gel.png"  # and the pick is shown


def test_load_image_cancelled_leaves_state_untouched(window, monkeypatch):
    import geltilt.app.window as window_mod

    monkeypatch.setattr(window_mod.filedialog, "askopenfilename", lambda **k: "")
    window._choose_image()
    assert window.controller.folder is None


def test_crop_buttons_disabled_until_auto_rotate(window):
    assert str(window.auto_crop_button.cget("state")) == "disabled"
    assert str(window.clear_crop_button.cget("state")) == "disabled"


def test_export_mode_defaults_to_rotation_plus_crop(window):
    assert window.export_mode.get() == "Rotation + crop"
    assert window._export_apply_crop() is True


def test_export_keyboard_shortcuts_are_bound(window):
    assert window.bind("<Control-s>")
    assert window.bind("<Control-c>")
    assert window.bind("<Control-o>")


def test_vertical_guides_are_off_by_default(window):
    assert window.guides_var.get() is False
    assert window.viewer.show_guides is False


def test_gelgenie_masks_overlay_is_off_by_default(window):
    assert window.mask_var.get() is False
    assert window.viewer.show_mask is False


def test_dpi_scale_is_at_least_one(window):
    assert window.app_config.ui_scale >= 1.0


def test_window_applies_a_preferred_ui_font(window):
    from tkinter import font as tkfont

    from geltilt.app import fonts

    family = tkfont.nametofont("TkDefaultFont").actual("family")
    assert family in fonts.PREFERRED_FAMILIES

def test_window_builds_in_default_chinese(monkeypatch, tmp_path):
    from geltilt.app import config as app_config

    monkeypatch.setattr(app_config, "load_language", lambda run_dir=None: "zh")
    try:
        win = AppWindow(run_dir=str(tmp_path / "run_zh"))
    except tk.TclError:  # pragma: no cover - headless
        pytest.skip("no display available for Tk")
    win.withdraw()
    try:
        assert win.lang_box.get() == "\u7b80\u4f53\u4e2d\u6587"
        assert "\u81ea\u52a8\u65cb\u8f6c" in set(_buttons(win))  # Auto rotate
        assert win.export_mode.get() == "\u65cb\u8f6c + \u88c1\u526a"  # Rotation + crop
    finally:
        win.destroy()