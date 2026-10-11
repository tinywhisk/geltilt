"""Manual rotation override input for the Rotation group.

Layout follows ``ui_layout_reference``: a numeric entry, coarse ``<``/``>`` step
arrows around the slider, and a Reset button.  The control exposes a small port
(``bind_change``/``set_value``/``get_value``/``nudge``) used by the controller.
"""

from __future__ import annotations

from typing import Callable, Optional

import tkinter as tk
from tkinter import ttk

from ..domain import ANGLE_MAX, ANGLE_MIN, ANGLE_RESOLUTION, clamp_angle

_COARSE_STEP = 1.0


class ManualAngleControl(ttk.Frame):
    """Entry + coarse-step arrows + continuous slider + reset."""

    def __init__(self, master: tk.Misc, ui_scale: float = 1.0) -> None:
        super().__init__(master)
        self.ui_scale = ui_scale
        self._value = 0.0
        self._last_valid = 0.0
        self._on_change: Optional[Callable[[float], None]] = None
        self._suspend = False

        ttk.Label(self, text="Manual angle (\u00b0):").pack(side="left")

        self.entry_var = tk.StringVar(value="0.0")
        self.entry = ttk.Entry(self, textvariable=self.entry_var, width=8, justify="right")
        self.entry.pack(side="left", padx=(4, 8))
        self.entry.bind("<Return>", self._on_entry)
        self.entry.bind("<FocusOut>", self._on_entry)

        ttk.Button(self, text="\u25c0", width=3,
                   command=lambda: self.nudge(-_COARSE_STEP)).pack(side="left")

        self.var = tk.DoubleVar(value=0.0)
        self.scale = tk.Scale(
            self, from_=ANGLE_MIN, to=ANGLE_MAX, resolution=ANGLE_RESOLUTION,
            orient="horizontal", showvalue=False, length=int(340 * ui_scale),
            variable=self.var, command=self._on_scale,
        )
        self.scale.pack(side="left")

        ttk.Button(self, text="\u25b6", width=3,
                   command=lambda: self.nudge(_COARSE_STEP)).pack(side="left")
        ttk.Button(self, text="Reset", width=7,
                   command=lambda: self._emit(0.0)).pack(side="left", padx=(8, 0))

    # -- port --------------------------------------------------------------- #
    def get_value(self) -> float:
        return self._value

    def set_value(self, value: float) -> None:
        self._apply(float(value))

    def bind_change(self, callback: Callable[[float], None]) -> None:
        self._on_change = callback

    def nudge(self, delta: float) -> None:
        self._emit(self._value + float(delta))

    # -- internals ---------------------------------------------------------- #
    def _apply(self, value: float) -> None:
        try:
            clamped = clamp_angle(value)
        except ValueError:
            return
        self._value = clamped
        self._last_valid = clamped
        self._suspend = True
        try:
            self.var.set(clamped)
            self.entry_var.set(f"{clamped:.1f}")
        finally:
            self._suspend = False

    def _emit(self, value: float) -> None:
        self._apply(value)
        if self._on_change is not None:
            self._on_change(self._value)

    def _on_scale(self, value: str) -> None:
        if self._suspend:
            return
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return
        self._value = clamp_angle(parsed)
        self._last_valid = self._value
        self.entry_var.set(f"{self._value:.1f}")
        if self._on_change is not None:
            self._on_change(self._value)

    def _on_entry(self, _event: tk.Event) -> None:
        text = self.entry_var.get().strip()
        try:
            parsed = float(text)
        except (TypeError, ValueError):
            self._apply(self._last_valid)  # reject non-numeric, revert
            return
        self._emit(parsed)