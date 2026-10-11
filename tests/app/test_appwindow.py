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
        "Load image...", "Load folder...", "Auto rotate", "Reset",
        "Auto crop", "Clear crop", "Even widths", "Save as...", "Copy to clipboard",
    } <= texts


def test_unimplemented_buttons_are_enabled_noops(window):
    inert = ["Load image..."]
    before = list(window.controller.samples)
    buttons = _buttons(window)
    for text in inert:
        button = buttons[text]
        assert str(button.cget("state")) != "disabled"
        button.invoke()  # no-op: must not raise or change state
    assert list(window.controller.samples) == before


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


def test_dpi_scale_is_at_least_one(window):
    assert window.app_config.ui_scale >= 1.0