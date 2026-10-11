"""Compact diagnostic display for the active sample (bottom bar)."""

from __future__ import annotations

import math
from typing import Optional

import tkinter as tk
from tkinter import ttk

from ..domain import AppliedRotation, EstimateOutcome

_SOURCE_LABEL = {"none": "none (no rotation)", "auto": "auto", "manual": "manual"}


class DiagnosticPanel(ttk.LabelFrame):
    """Always reflects the active sample's diagnostics, laid out as a bar."""

    def __init__(self, master: tk.Misc, ui_scale: float = 1.0) -> None:
        super().__init__(master, text="Diagnostics")
        self.ui_scale = ui_scale
        self._vars = {
            "sample": tk.StringVar(value="-"),
            "applied": tk.StringVar(value="none (no rotation)"),
            "status": tk.StringVar(value="not run"),
            "reason": tk.StringVar(value=""),
            "uncertainty": tk.StringVar(value="-"),
            "bands": tk.StringVar(value="-"),
            "lanes": tk.StringVar(value="-"),
            "warnings": tk.StringVar(value="(none)"),
        }

        pairs = [
            ("Sample", "sample"),
            ("Applied", "applied"),
            ("Status", "status"),
            ("Uncertainty", "uncertainty"),
            ("Bands", "bands"),
            ("Lanes used/found", "lanes"),
        ]
        body = ttk.Frame(self)
        body.pack(fill="x", padx=6, pady=(4, 2))
        for i, (label, key) in enumerate(pairs):
            row, col = divmod(i, 2)  # two key/value pairs per row (narrow side column)
            ttk.Label(body, text=f"{label}:").grid(
                row=row, column=col * 2, sticky="e", padx=(12 if col else 0, 2), pady=1
            )
            ttk.Label(body, textvariable=self._vars[key]).grid(
                row=row, column=col * 2 + 1, sticky="w", pady=1
            )

        wrap = ttk.Frame(self)
        wrap.pack(fill="x", padx=6, pady=(0, 4))
        text_len = int(420 * ui_scale)
        ttk.Label(wrap, text="Reason:").grid(row=0, column=0, sticky="ne")
        ttk.Label(wrap, textvariable=self._vars["reason"], anchor="w", justify="left",
                  wraplength=text_len, foreground="#333333").grid(
            row=0, column=1, sticky="w", padx=(4, 0)
        )
        ttk.Label(wrap, text="Warnings:").grid(row=1, column=0, sticky="ne")
        ttk.Label(wrap, textvariable=self._vars["warnings"], anchor="w", justify="left",
                  wraplength=text_len).grid(row=1, column=1, sticky="w", padx=(4, 0))

    # -- port --------------------------------------------------------------- #
    def show_empty(self) -> None:
        self._vars["sample"].set("-")
        self._vars["applied"].set("none (no rotation)")
        self._vars["status"].set("not run")
        self._vars["reason"].set("")
        self._vars["uncertainty"].set("-")
        self._vars["bands"].set("-")
        self._vars["lanes"].set("-")
        self._set_warnings(())

    def show_state(
        self,
        name: str,
        applied: AppliedRotation,
        estimate: Optional[EstimateOutcome],
        failed: bool = False,
    ) -> None:
        self._vars["sample"].set(name)
        self._vars["applied"].set(
            f"{applied.value:+.1f} deg  [{_SOURCE_LABEL.get(applied.source, applied.source)}]"
        )
        if failed and estimate is None:
            self._vars["status"].set("failed to load")
        if estimate is None:
            self._vars["status"].set("not run")
            self._vars["reason"].set("")
            self._vars["uncertainty"].set("-")
            self._vars["bands"].set("-")
            self._vars["lanes"].set("-")
            self._set_warnings(())
            return

        if estimate.accepted:
            self._vars["status"].set("accepted")
        else:
            self._vars["status"].set("refused")
        self._vars["reason"].set(estimate.reason or "")
        se = estimate.se
        self._vars["uncertainty"].set(f"{se:.2f} deg" if math.isfinite(se) else "-")
        self._vars["bands"].set(str(estimate.n_bands))
        if estimate.lanes_found or estimate.lanes_used:
            self._vars["lanes"].set(f"{estimate.lanes_used} / {estimate.lanes_found}")
        else:
            self._vars["lanes"].set("no detections")
        self._set_warnings(estimate.warnings)

    # -- internals ---------------------------------------------------------- #
    def _set_warnings(self, warnings) -> None:
        text = "; ".join(str(w) for w in warnings) if warnings else "(none)"
        self._vars["warnings"].set(text)