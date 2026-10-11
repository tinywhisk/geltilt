from __future__ import annotations

import tkinter as tk

import pytest

from geltilt.app.domain import AppliedRotation
from geltilt.app.widgets.diagnostic_panel import DiagnosticPanel
from geltilt.app.widgets.manual_angle import ManualAngleControl

from .conftest import make_outcome


# --------------------------------------------------------------------------- #
# Manual angle control                                                         #
# --------------------------------------------------------------------------- #
def test_manual_control_clamps_and_rounds(tk_root):
    control = ManualAngleControl(tk_root)
    control.set_value(500.0)
    assert control.get_value() == 180.0
    control.set_value(-12.34)
    assert control.get_value() == pytest.approx(-12.3)


def test_manual_control_numeric_entry_rejects_garbage(tk_root):
    control = ManualAngleControl(tk_root)
    control.set_value(3.0)
    control.entry_var.set("not a number")
    control._on_entry(None)
    assert control.get_value() == pytest.approx(3.0)


def test_manual_control_emits_change(tk_root):
    control = ManualAngleControl(tk_root)
    seen = []
    control.bind_change(seen.append)
    control.nudge(1.5)
    assert seen and seen[-1] == pytest.approx(1.5)


# --------------------------------------------------------------------------- #
# Diagnostic panel                                                             #
# --------------------------------------------------------------------------- #
def test_panel_shows_not_run(tk_root):
    panel = DiagnosticPanel(tk_root)
    panel.show_state("a.png", AppliedRotation(0.0, "none"), None)
    assert panel._vars["status"].get() == "not run"
    assert panel._vars["applied"].get().startswith("+0.0")


def test_panel_shows_accepted_and_refused(tk_root):
    panel = DiagnosticPanel(tk_root)
    panel.show_state("a.png", AppliedRotation(2.0, "auto"), make_outcome(ok=True, angle=2.0))
    assert panel._vars["status"].get() == "accepted"
    assert panel._vars["lanes"].get() == "2 / 3"
    panel.show_state("a.png", AppliedRotation(2.0, "auto"),
                     make_outcome(ok=False, angle=float("nan"), reason="fewer than 4 bands"))
    assert panel._vars["status"].get() == "refused"
    assert "bands" in panel._vars["reason"].get()


def test_panel_reflects_active_sample(tk_root):
    panel = DiagnosticPanel(tk_root)
    panel.show_state("first.png", AppliedRotation(1.0, "manual"), None)
    panel.show_state("second.png", AppliedRotation(0.0, "none"), None)
    assert panel._vars["sample"].get() == "second.png"
    assert panel._vars["applied"].get().startswith("+0.0")

def test_panel_shows_and_clears_the_cursor_readout(tk_root):
    panel = DiagnosticPanel(tk_root)
    panel.set_cursor((12, 34))
    assert panel._vars["cursor"].get() == "12, 34"
    panel.set_cursor(None)
    assert panel._vars["cursor"].get() == "-"        # English is pinned in tests


def test_panel_clears_cursor_on_empty(tk_root):
    panel = DiagnosticPanel(tk_root)
    panel.set_cursor((5, 6))
    panel.show_empty()
    assert panel._vars["cursor"].get() == "-"
