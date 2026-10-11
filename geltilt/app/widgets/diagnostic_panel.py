"""Compact diagnostic display for the active sample (bottom bar)."""

from __future__ import annotations

import math
from typing import Optional

import tkinter as tk
from tkinter import ttk

from .. import i18n
from ..domain import AppliedRotation, EstimateOutcome

_SOURCE_LABEL = {
    "none": "diag.none",
    "auto": "source.auto",
    "manual": "source.manual",
}


class DiagnosticPanel(ttk.LabelFrame):
    """Always reflects the active sample's diagnostics, laid out as a bar."""

    def __init__(self, master: tk.Misc, ui_scale: float = 1.0) -> None:
        super().__init__(master, text=i18n.tr("group.diagnostics"))
        self.ui_scale = ui_scale
        self._vars = {
            "sample": tk.StringVar(value=i18n.tr("diag.dash")),
            "applied": tk.StringVar(value=i18n.tr("diag.none")),
            "status": tk.StringVar(value=i18n.tr("diag.not_run")),
            "reason": tk.StringVar(value=""),
            "uncertainty": tk.StringVar(value=i18n.tr("diag.dash")),
            "bands": tk.StringVar(value=i18n.tr("diag.dash")),
            "lanes": tk.StringVar(value=i18n.tr("diag.dash")),
            "warnings": tk.StringVar(value=i18n.tr("diag.no_warnings")),
        }

        pairs = [
            ("diag.sample", "sample"),
            ("diag.applied", "applied"),
            ("diag.status", "status"),
            ("diag.uncertainty", "uncertainty"),
            ("diag.bands", "bands"),
            ("diag.lanes", "lanes"),
        ]
        body = ttk.Frame(self)
        body.pack(fill="x", padx=6, pady=(4, 2))
        for i, (label_key, key) in enumerate(pairs):
            row, col = divmod(i, 2)  # two key/value pairs per row (narrow side column)
            ttk.Label(body, text=f"{i18n.tr(label_key)}:").grid(
                row=row, column=col * 2, sticky="e", padx=(12 if col else 0, 2), pady=1
            )
            ttk.Label(body, textvariable=self._vars[key]).grid(
                row=row, column=col * 2 + 1, sticky="w", pady=1
            )

        wrap = ttk.Frame(self)
        wrap.pack(fill="x", padx=6, pady=(0, 4))
        text_len = int(420 * ui_scale)
        ttk.Label(wrap, text=i18n.tr("diag.reason")).grid(row=0, column=0, sticky="ne")
        ttk.Label(wrap, textvariable=self._vars["reason"], anchor="w", justify="left",
                  wraplength=text_len, foreground="#333333").grid(
            row=0, column=1, sticky="w", padx=(4, 0)
        )
        ttk.Label(wrap, text=i18n.tr("diag.warnings")).grid(row=1, column=0, sticky="ne")
        ttk.Label(wrap, textvariable=self._vars["warnings"], anchor="w", justify="left",
                  wraplength=text_len).grid(row=1, column=1, sticky="w", padx=(4, 0))

    # -- port --------------------------------------------------------------- #
    def show_empty(self) -> None:
        self._vars["sample"].set(i18n.tr("diag.dash"))
        self._vars["applied"].set(i18n.tr("diag.none"))
        self._vars["status"].set(i18n.tr("diag.not_run"))
        self._vars["reason"].set("")
        self._vars["uncertainty"].set(i18n.tr("diag.dash"))
        self._vars["bands"].set(i18n.tr("diag.dash"))
        self._vars["lanes"].set(i18n.tr("diag.dash"))
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
            f"{applied.value:+.1f} {i18n.tr('diag.degree')}  "
            f"[{i18n.tr(_SOURCE_LABEL.get(applied.source, applied.source))}]"
        )
        if failed and estimate is None:
            self._vars["status"].set(i18n.tr("diag.failed_load"))
        if estimate is None:
            self._vars["status"].set(i18n.tr("diag.not_run"))
            self._vars["reason"].set("")
            self._vars["uncertainty"].set(i18n.tr("diag.dash"))
            self._vars["bands"].set(i18n.tr("diag.dash"))
            self._vars["lanes"].set(i18n.tr("diag.dash"))
            self._set_warnings(())
            return

        self._vars["status"].set(i18n.tr("diag.accepted") if estimate.accepted
                                 else i18n.tr("diag.refused"))
        self._vars["reason"].set(i18n.localize_diagnostic(estimate.reason or ""))
        se = estimate.se
        self._vars["uncertainty"].set(
            f"{se:.2f} {i18n.tr('diag.degree')}" if math.isfinite(se) else i18n.tr("diag.dash")
        )
        self._vars["bands"].set(str(estimate.n_bands))
        if estimate.lanes_found or estimate.lanes_used:
            self._vars["lanes"].set(f"{estimate.lanes_used} / {estimate.lanes_found}")
        else:
            self._vars["lanes"].set(i18n.tr("diag.no_detections"))
        self._set_warnings(estimate.warnings)

    # -- internals ---------------------------------------------------------- #
    def _set_warnings(self, warnings) -> None:
        text = "; ".join(i18n.localize_diagnostic(str(w)) for w in warnings) if warnings \
            else i18n.tr("diag.no_warnings")
        self._vars["warnings"].set(text)