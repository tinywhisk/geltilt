"""The single-window application shell."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ..labeler import apply_tk_scaling, enable_dpi_awareness
from . import i18n, settings
from .about import show_about
from .config import AppConfig
from .fonts import apply_ui_fonts
from .controller import AppController
from .export import SAVE_FILETYPES
from .widgets.diagnostic_panel import DiagnosticPanel
from .widgets.manual_angle import ManualAngleControl
from .widgets.sample_nav import SampleNav
from .widgets.viewer import SingleViewer


class AppWindow(tk.Tk):
    """Single-window shell: file/folder bar, Rotation, Crop, one viewer,
    Export, and a bottom bar holding overlays, diagnostics and status."""

    def __init__(
        self,
        initial_folder: Optional[str] = None,
        run_dir: Optional[str] = None,
        ui_scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.app_config = AppConfig.resolve(run_dir=run_dir, ui_scale=ui_scale)
        i18n.set_language(self.app_config.language)
        scale = self.app_config.ui_scale

        self.title(i18n.tr("app.title"))
        self.geometry(f"{int(1280 * scale)}x{int(840 * scale)}")
        self.minsize(int(940 * scale), int(620 * scale))
        apply_tk_scaling(self, scale)
        apply_ui_fonts(self)

        self._build(scale)
        self.controller = AppController(self, self.app_config)
        self.controller.attach(
            self.nav, self.viewer, self.manual, self.panel,
            status_cb=self._set_status, busy_cb=self._set_busy,
            error_cb=lambda t, m: messagebox.showerror(t, m),
            warn_cb=lambda t, m: messagebox.showwarning(t, m),
            crop_cb=self._set_crop_buttons,
            composite_cb=self._set_composite_controls,
        )
        self.controller.start(Path(initial_folder) if initial_folder else None)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(500, self._periodic_save)
        self._bind_keys()

    # -- layout ------------------------------------------------------------- #
    def _build(self, scale: float) -> None:
        # Row 1: file / folder bar.
        top = ttk.Frame(self, padding=(8, 6))
        top.pack(side="top", fill="x")
        ttk.Button(top, text=i18n.tr("btn.load_image"), width=13,
                   command=self._noop).pack(side="left")
        ttk.Button(top, text=i18n.tr("btn.load_folder"), width=13, command=self._choose_folder).pack(
            side="left", padx=(6, 0)
        )
        # Arrows + current filename, absorbing the leftover width.
        self.nav = SampleNav(top, ui_scale=scale)
        self.nav.pack(side="left", fill="x", expand=True, padx=(12, 0))
        self.lang_box = ttk.Combobox(
            top, values=[i18n.language_label(c) for c in i18n.available_languages()],
            state="readonly", width=10,
        )
        self.lang_box.set(i18n.language_label(i18n.get_language()))
        self.lang_box.pack(side="right")
        self.lang_box.bind("<<ComboboxSelected>>", self._on_language_selected)
        ttk.Label(top, text=i18n.tr("label.language")).pack(side="right", padx=(0, 4))

        # Row 2: Rotation group.
        rot = ttk.LabelFrame(self, text=i18n.tr("group.rotation"), padding=(8, 6))
        rot.pack(side="top", fill="x", padx=8)
        ttk.Button(rot, text=i18n.tr("btn.auto_rotate"), width=13,
                   command=self._auto_rotate).pack(side="left")
        self.manual = ManualAngleControl(rot, ui_scale=scale)
        self.manual.pack(side="left", padx=(12, 0))

        # Rows 3-4: Crop group.
        crop = ttk.LabelFrame(self, text=i18n.tr("group.crop"), padding=(8, 6))
        crop.pack(side="top", fill="x", padx=8, pady=(6, 0))
        crop_row = ttk.Frame(crop)
        crop_row.pack(side="top", fill="x")
        self.auto_crop_button = ttk.Button(crop_row, text=i18n.tr("btn.auto_crop"), width=13,
                                           command=self._auto_crop, state="disabled")
        self.auto_crop_button.pack(side="left")
        self.clear_crop_button = ttk.Button(crop_row, text=i18n.tr("btn.clear_crop"), width=11,
                                            command=self._clear_crop, state="disabled")
        self.clear_crop_button.pack(side="left", padx=(6, 0))
        self.crop_label = ttk.Label(crop_row, text=i18n.tr("label.no_crop"), anchor="w")
        self.crop_label.pack(side="left", padx=(12, 0))
        ttk.Label(crop_row, text=i18n.tr("label.padding")).pack(side="left", padx=(12, 0))
        self.pad_spin = ttk.Spinbox(crop_row, from_=0, to=200, increment=5, width=4,
                                    justify="right")
        self.pad_spin.pack(side="left", padx=(4, 0))
        ttk.Label(crop_row, text=i18n.tr("label.padding_unit")).pack(side="left", padx=(4, 0))

        lane_row = ttk.Frame(crop)
        lane_row.pack(side="top", fill="x", pady=(6, 0))
        self.composite_var = tk.BooleanVar(value=False)
        self.composite_check = ttk.Checkbutton(
            lane_row, text=i18n.tr("check.composite_crop"), variable=self.composite_var,
            command=self._toggle_composite, state="disabled",
        )
        self.composite_check.pack(side="left")
        self.lane_entry = ttk.Entry(lane_row, width=22, state="disabled")
        self.lane_entry.pack(side="left", padx=(6, 0))
        self.lane_entry.bind("<Return>", self._on_composite_spec)
        self.lane_entry.bind("<FocusOut>", self._on_composite_spec)
        self.even_button = ttk.Button(lane_row, text=i18n.tr("btn.even_widths"), width=11,
                                      command=self._even_widths, state="disabled")
        self.even_button.pack(side="right")
        self.composite_hint = ttk.Label(lane_row, text=i18n.tr("hint.run_auto_crop_first"),
                                        anchor="w")
        self.composite_hint.pack(side="left", padx=(10, 8), fill="x", expand=True)

        # Row 5: main area — viewer on the left, controls/status on the right.
        panes = ttk.Frame(self, padding=(8, 6))
        panes.pack(side="top", fill="both", expand=True)
        panes.columnconfigure(0, weight=1, uniform="pane")
        panes.columnconfigure(1, weight=1, uniform="pane")
        panes.rowconfigure(1, weight=1)
        ttk.Label(panes, text=i18n.tr("label.corrected")).grid(row=0, column=0, sticky="w")
        self.viewer = SingleViewer(panes, ui_scale=scale)
        self.viewer.grid(row=1, column=0, sticky="nsew", padx=(0, 4))

        # Right column: everything that used to sit below the viewer.
        right = ttk.Frame(panes)
        right.grid(row=0, column=1, rowspan=2, sticky="new", padx=(4, 0))
        right.columnconfigure(0, weight=1)

        # Export group.
        exp = ttk.LabelFrame(right, text=i18n.tr("group.export"), padding=(8, 6))
        exp.grid(row=0, column=0, sticky="ew")
        exp_row1 = ttk.Frame(exp)
        exp_row1.pack(side="top", fill="x")
        ttk.Label(exp_row1, text=i18n.tr("label.save_copy")).pack(side="left")
        self.export_mode = ttk.Combobox(
            exp_row1,
            values=[i18n.tr("export.mode_rotation_only"), i18n.tr("export.mode_rotation_crop")],
            state="readonly", width=17,
        )
        self.export_mode.set(i18n.tr("export.mode_rotation_crop"))
        self.export_mode.pack(side="left", padx=(6, 0))
        exp_row2 = ttk.Frame(exp)
        exp_row2.pack(side="top", fill="x", pady=(6, 0))
        self.use_original_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(exp_row2, text=i18n.tr("check.use_original"),
                        variable=self.use_original_var).pack(side="left")
        exp_row3 = ttk.Frame(exp)
        exp_row3.pack(side="top", fill="x", pady=(6, 0))
        ttk.Button(exp_row3, text=i18n.tr("btn.save_as"), width=13,
                   command=self._save_as).pack(side="left")
        ttk.Button(exp_row3, text=i18n.tr("btn.copy_clipboard"), width=17,
                   command=self._copy_to_clipboard).pack(side="left", padx=(6, 0))
        ttk.Label(exp_row3, text=i18n.tr("label.8bit")).pack(side="left", padx=(12, 0))

        # Overlays.
        overlays = ttk.LabelFrame(right, text=i18n.tr("group.overlays"), padding=(8, 6))
        overlays.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.guides_var = tk.BooleanVar(value=False)
        self.centroids_var = tk.BooleanVar(value=False)
        self.lane_lines_var = tk.BooleanVar(value=False)
        self.mask_var = tk.BooleanVar(value=False)  # GelGenie masks overlay: off by default
        ttk.Checkbutton(overlays, text=i18n.tr("check.vertical_guides"), variable=self.guides_var,
                        command=lambda: self._toggle_overlay("guides")).pack(anchor="w")
        ttk.Checkbutton(overlays, text=i18n.tr("check.band_centroids"),
                        variable=self.centroids_var,
                        command=lambda: self._toggle_overlay("centroids")).pack(anchor="w")
        ttk.Checkbutton(overlays, text=i18n.tr("check.lane_lines"),
                        variable=self.lane_lines_var,
                        command=lambda: self._toggle_overlay("lane_lines")).pack(anchor="w")
        ttk.Checkbutton(overlays, text=i18n.tr("check.gelgenie_masks"),
                        variable=self.mask_var,
                        command=lambda: self._toggle_overlay("mask")).pack(anchor="w")

        # Diagnostics.
        self.panel = DiagnosticPanel(right, ui_scale=scale)
        self.panel.grid(row=2, column=0, sticky="ew", pady=(6, 0))

        # Status.
        status_line = ttk.Frame(right)
        status_line.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        self.status = ttk.Label(
            status_line, text=i18n.tr("status.initial"),
            anchor="w", justify="left", wraplength=int(430 * scale),
        )
        self.status.pack(side="left", fill="x", expand=True)
        self.busy_label = ttk.Label(status_line, text=i18n.tr("status.idle"),
                                    foreground="#666666")
        self.busy_label.pack(side="right")
        self.detail = ttk.Label(right, text=i18n.tr("status.no_folder"), anchor="w",
                                justify="left", foreground="#444444",
                                wraplength=int(430 * scale))
        self.detail.grid(row=4, column=0, sticky="ew")
        ttk.Button(right, text=i18n.tr("btn.about"), command=self._about).grid(
            row=5, column=0, sticky="ew", pady=(6, 0)
        )

    # -- callbacks ---------------------------------------------------------- #
    def _about(self) -> None:
        show_about(self, self.app_config.ui_scale)

    def _noop(self, *_args, **_kwargs) -> None:
        """Placeholder for buttons whose feature is not implemented yet."""
        return None

    def _on_language_selected(self, _event=None) -> None:
        """Persist the choice; the new language applies after a restart."""
        code = i18n.language_code(str(self.lang_box.get()))
        if code is None or code == i18n.get_language():
            return
        settings.save_language(self.app_config.run_dir, code)
        messagebox.showinfo(
            i18n.tr("lang.restart_title", language=code),
            i18n.tr("lang.restart_message", language=code),
            parent=self,
        )

    def _choose_folder(self) -> None:
        start = self.controller.folder
        if start is None:
            candidate = self.app_config.run_dir / "samples"
            start = candidate if candidate.is_dir() else self.app_config.run_dir
        chosen = filedialog.askdirectory(initialdir=str(start),
                                         title=i18n.tr("fd.select_image_folder"))
        if chosen:
            self.controller.open_folder(Path(chosen))
            self._update_folder_label()

    def _update_folder_label(self) -> None:
        folder = self.controller.folder
        if folder:
            count = len(self.controller.samples)
            self.detail.config(
                text=i18n.tr("status.folder_count", name=folder.name, count=count)
            )
        else:
            self.detail.config(text=i18n.tr("status.no_folder"))

    def _auto_rotate(self) -> None:
        self.controller.trigger_auto_rotate()

    def _auto_crop(self) -> None:
        self.controller.auto_crop()

    def _clear_crop(self) -> None:
        self.controller.clear_crop()

    def _set_crop_buttons(self, can_auto_crop: bool, has_crop: bool) -> None:
        self.auto_crop_button.config(state="normal" if can_auto_crop else "disabled")
        self.clear_crop_button.config(state="normal" if has_crop else "disabled")

    def _toggle_composite(self) -> None:
        on = bool(self.composite_var.get())
        ok = self.controller.set_composite(on)
        if on and ok:
            self.export_mode.set(i18n.tr("export.mode_rotation_crop"))  # R6
        if not ok:
            self.composite_var.set(False)
        self._set_composite_controls()

    def _on_composite_spec(self, _event=None) -> None:
        self.controller.set_composite_spec(self.lane_entry.get())
        self._set_composite_controls()

    def _even_widths(self) -> None:
        self.controller.even_widths()

    def _set_composite_controls(self) -> None:
        selectable = self.controller.composite_selectable()
        mode = self.controller.composite_mode()
        self.composite_check.config(state="normal" if selectable else "disabled")
        self.lane_entry.config(state="normal" if mode else "disabled")
        self.even_button.config(state="normal" if mode else "disabled")
        self.composite_var.set(self.controller.composite_on)
        text = self.controller.composite_text
        if self.lane_entry.get() != text:
            self.lane_entry.delete(0, "end")
            self.lane_entry.insert(0, text)
        if not selectable:
            self.composite_hint.config(text=i18n.tr("hint.run_auto_crop_first"))
        elif mode:
            self.composite_hint.config(text=i18n.tr("hint.composite_click"))
        else:
            self.composite_hint.config(text=i18n.tr("hint.composite_tick"))

    def _export_apply_crop(self) -> bool:
        return str(self.export_mode.get()) == i18n.tr("export.mode_rotation_crop")

    def _save_as(self) -> None:
        path = self.controller.active_sample()
        if path is None:
            messagebox.showinfo(i18n.tr("dlg.save_as"), i18n.tr("msg.open_image_first"))
            return
        dest = filedialog.asksaveasfilename(
            title=i18n.tr("fd.save_corrected"),
            initialdir=str(path.parent),
            initialfile=f"{path.stem}"
                        f"{'_rotated_composite' if self.controller.composite_active() else '_corrected'}",
            defaultextension=".tif",
            filetypes=SAVE_FILETYPES,
        )
        if dest:
            self.controller.save_as(dest, use_original=self.use_original_var.get(),
                                    apply_crop=self._export_apply_crop())

    def _copy_to_clipboard(self) -> None:
        self.controller.copy_to_clipboard(use_original=self.use_original_var.get(),
                                          apply_crop=self._export_apply_crop())

    def _toggle_overlay(self, kind: str) -> None:
        self.controller.toggle_overlay(kind)

    def _reset_view(self) -> None:
        self.controller.reset_view()

    def _prev(self) -> None:
        self.controller.prev_sample()

    def _next(self) -> None:
        self.controller.next_sample()


    def _set_status(self, text: str) -> None:
        self.status.config(text=text)
        self._update_folder_label()

    def _set_busy(self, busy: bool) -> None:
        self.busy_label.config(text=i18n.tr("status.estimating") if busy
                               else i18n.tr("status.idle"))

    def _periodic_save(self) -> None:
        try:
            self.controller.save_session()
        except Exception:  # noqa: BLE001
            pass
        self.after(15000, self._periodic_save)

    def _text_focus(self) -> bool:
        widget = self.focus_get()
        return isinstance(widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Spinbox))

    def _bind_keys(self) -> None:
        self.bind("<Control-o>", lambda e: self._choose_folder())
        self.bind("<Control-s>", lambda e: self._save_as())
        self.bind("<Control-S>", lambda e: self._save_as())
        self.bind("<Control-c>", lambda e: None if self._text_focus() else self._copy_to_clipboard())
        self.bind("<Control-C>", lambda e: None if self._text_focus() else self._copy_to_clipboard())
        self.bind("<Key-a>", lambda e: None if self._text_focus() else self._auto_rotate())
        self.bind("<Key-z>", lambda e: None if self._text_focus() else self.controller.zero_angle())
        self.bind("<Key-n>", lambda e: None if self._text_focus() else self._next())
        self.bind("<Key-p>", lambda e: None if self._text_focus() else self._prev())
        self.bind("<Next>", lambda e: self._next())
        self.bind("<Prior>", lambda e: self._prev())
        self.bind("<Right>", lambda e: None if self._text_focus() else self.controller.nudge(0.1))
        self.bind("<Left>", lambda e: None if self._text_focus() else self.controller.nudge(-0.1))
        self.bind("<Key-r>", lambda e: None if self._text_focus() else self._reset_view())

    def _on_close(self) -> None:
        self.controller.on_close()
        self.destroy()


def _selftest(run_dir_override=None) -> int:
    """Headless-ish check used to validate packaged builds; writes app_selftest.json."""
    import json
    import sys
    import tkinter as tk

    from .config import resolve_run_dir

    run_dir = resolve_run_dir(run_dir_override)
    report = {"ok": True, "run_dir": str(run_dir), "python": sys.version}
    try:
        window = AppWindow(run_dir=str(run_dir))
        window.withdraw()
        window.update_idletasks()
        canvases = [w for w in _walk_widgets(window) if isinstance(w, tk.Canvas)]
        report["image_surfaces"] = len(canvases)
        report["ok"] = report["ok"] and len(canvases) == 1
        window.controller.save_session()
        report["session_written"] = (run_dir / "app_session.json").exists()
        window.destroy()
    except Exception as exc:  # noqa: BLE001 - report, don't crash
        report["ok"] = False
        report["error"] = repr(exc)
    try:
        (run_dir / "app_selftest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    except OSError:
        pass
    return 0 if report.get("ok") else 1


def _walk_widgets(widget):
    yield widget
    for child in widget.winfo_children():
        yield from _walk_widgets(child)


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="geltilt-app",
                                     description="Unified gel rotation desktop app.")
    parser.add_argument("directory", nargs="?", default=None, help="image folder to open at startup")
    parser.add_argument("--run-dir", default=None, help="run directory for session/log files")
    parser.add_argument("--selftest", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest(args.run_dir)

    enable_dpi_awareness()
    window = AppWindow(initial_folder=args.directory, run_dir=args.run_dir)
    window.mainloop()
    return 0