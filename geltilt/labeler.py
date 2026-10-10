"""Tkinter tool for human labelling of the correct rotation (ground truth).

The labeler shows each image from a chosen folder with the existing brightness /
contrast preprocessing (:func:`geltilt.preprocess.robust_limits` followed by
:func:`geltilt.preprocess.to_u8`, gamma fixed at 1.0) and fixed vertical guide
lines. The operator drags to pan, scrolls to zoom and moves the rotation slider
until the lanes run parallel to the guides, then records the angle plus a free
form comment per image.

The image folder is chosen from the interface (``Open folder...``) or given on
the command line. Labels are written as JSON into the *run directory*: the folder
that contains the executable when frozen, otherwise the current working
directory. On startup the last folder and image position are restored from
``labeler_session.json`` so a labelling session can be resumed; the labels
themselves are read back from the existing labels file (``sample_labels.json``,
or a legacy ``<folder>_labels.json``) so previously entered angles, comments and
confirmed flags reappear.

Run with ``geltilt-label [directory]`` or double-click the packaged executable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import tkinter as tk
from tkinter import filedialog, font as tkfont, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from .imageio import load_array, to_gray
from .preprocess import robust_limits, to_u8
from .viz import content_window

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

Point = Tuple[float, float]


# --------------------------------------------------------------------------- #
# Pure geometry / rendering helpers (kept GUI-free so they are easy to test).  #
# --------------------------------------------------------------------------- #
def preprocess_for_display(arr: np.ndarray) -> np.ndarray:
    """Apply the canonical preprocessing, returning a uint8 grayscale image.

    The window is computed on the full frame (never on a crop) and gamma is
    fixed at 1.0, matching the rest of the tool.
    """
    gray = to_gray(arr)
    lo, hi = robust_limits(gray, 0.5, 99.5)
    return to_u8(gray, lo, hi, 1.0)


def content_anchor(gray_u8: np.ndarray) -> Point:
    """Return the centre of the visible band content, or the frame centre."""
    h, w = gray_u8.shape[:2]
    window = content_window(gray_u8)
    if window is None:
        return ((w - 1) / 2.0, (h - 1) / 2.0)
    y0, y1, x0, x1 = window
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def source_to_screen(p: Point, angle_deg: float, scale: float, anchor: Point, center: Point) -> Point:
    """Map a source pixel to screen coordinates.

    Positive ``angle_deg`` rotates the image content counter-clockwise, matching
    :func:`geltilt.core.correct_rotation` / :func:`geltilt.rotation.rotate_image`.
    """
    th = math.radians(angle_deg)
    c, s = math.cos(th), math.sin(th)
    dx, dy = p[0] - anchor[0], p[1] - anchor[1]
    return (
        center[0] + scale * (c * dx + s * dy),
        center[1] + scale * (-s * dx + c * dy),
    )


def screen_to_source(p: Point, angle_deg: float, scale: float, anchor: Point, center: Point) -> Point:
    """Inverse of :func:`source_to_screen`."""
    th = math.radians(angle_deg)
    c, s = math.cos(th), math.sin(th)
    dx, dy = p[0] - center[0], p[1] - center[1]
    return (
        anchor[0] + (c * dx - s * dy) / scale,
        anchor[1] + (s * dx + c * dy) / scale,
    )


def affine_coeffs(angle_deg: float, scale: float, anchor: Point, center: Point) -> Tuple[float, ...]:
    """PIL AFFINE coefficients mapping output pixels back to source pixels."""
    th = math.radians(angle_deg)
    c, s = math.cos(th), math.sin(th)
    a = c / scale
    b = -s / scale
    d = s / scale
    e = c / scale
    cc = anchor[0] + (-c * center[0] + s * center[1]) / scale
    ff = anchor[1] + (-s * center[0] - c * center[1]) / scale
    return (a, b, cc, d, e, ff)


def render_display(
    src: Image.Image,
    size: Tuple[int, int],
    angle_deg: float,
    scale: float,
    anchor: Point,
    center: Point,
    resample: int = Image.BILINEAR,
) -> Image.Image:
    """Render the (rotated, panned, zoomed) view of ``src`` onto a black canvas."""
    return src.transform(size, Image.AFFINE, affine_coeffs(angle_deg, scale, anchor, center),
                         resample=resample, fillcolor=0)


def find_images(directory: Path) -> List[Path]:
    """Return image files in ``directory`` sorted by name."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(files, key=lambda p: p.name.lower())


def run_directory() -> Path:
    """Directory to write labels into.

    Frozen (packaged) builds use the folder that holds the executable; running
    from source uses the current working directory.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path.cwd()


def enable_dpi_awareness() -> None:
    """Opt the process into per-monitor DPI awareness (Windows).

    Without this, Windows bitmap-scales the whole window on a >100% display,
    which makes all text look blurry. Must be called before the first Tk window
    is created.
    """
    if sys.platform != "win32":
        return
    import ctypes

    # PER_MONITOR_AWARE_V2 (Windows 10 1703+).
    try:
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except Exception:  # noqa: BLE001 - older Windows
        pass
    try:  # PROCESS_PER_MONITOR_DPI_AWARE (Windows 8.1+)
        if ctypes.windll.shcore.SetProcessDpiAwareness(2) == 0:
            return
    except Exception:  # noqa: BLE001
        pass
    try:  # legacy system-DPI aware (Vista+)
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:  # noqa: BLE001
        pass


def dpi_scale() -> float:
    """Primary-monitor scale factor (1.0 = 96 dpi), for sizing the window."""
    if sys.platform != "win32":
        return 1.0
    import ctypes

    dpi = 0
    try:
        dpi = int(ctypes.windll.user32.GetDpiForSystem())
    except Exception:  # noqa: BLE001 - pre-1607
        dpi = 0
    if dpi <= 0:
        try:
            hdc = ctypes.windll.user32.GetDC(0)
            dpi = int(ctypes.windll.gdi32.GetDeviceCaps(hdc, 88))  # LOGPIXELSX
            ctypes.windll.user32.ReleaseDC(0, hdc)
        except Exception:  # noqa: BLE001
            dpi = 96
    return max(1.0, dpi / 96.0)


def apply_tk_scaling(root: tk.Tk, scale: float) -> None:
    """Tell Tk the true pixels-per-point so widgets/fonts render crisply."""
    try:
        root.tk.call("tk", "scaling", 96.0 * scale / 72.0)
    except tk.TclError:
        pass


# --------------------------------------------------------------------------- #
# GUI                                                                          #
# --------------------------------------------------------------------------- #
class RotationLabeler:
    """Interactive per-image rotation labelling widget."""

    def __init__(
        self,
        root: tk.Tk,
        images: Optional[Sequence[Path]] = None,
        run_dir: Optional[Path] = None,
        labels_path: Optional[Path] = None,
        ui_scale: float = 1.0,
    ) -> None:
        self.root = root
        self.ui_scale = float(ui_scale) if ui_scale and ui_scale > 0 else 1.0
        self.run_dir = Path(run_dir) if run_dir is not None else run_directory()
        self._fixed_labels = labels_path is not None
        self.labels_path = Path(labels_path) if labels_path else self.run_dir / "labels.json"
        self.session_path = self.run_dir / "labeler_session.json"
        self.folder: Optional[Path] = None
        self.images: List[Path] = list(images) if images else []
        self.labels: Dict[str, dict] = self._load_labels()

        self.index = 0
        self.src_pil: Optional[Image.Image] = None
        self.src_shape: Tuple[int, int] = (0, 0)
        self.anchor: Point = (0.0, 0.0)
        self.center: Point = (0.0, 0.0)
        self.scale = 1.0
        self.angle = 0.0
        self.guide_spacing = 80

        self._imgtk: Optional[ImageTk.PhotoImage] = None
        self._drag_from: Optional[Point] = None
        self._suspend = False
        self._view_ready = False

        self._build_ui()
        self._update_folder_label()
        if self.images:
            self._goto(0)
        else:
            self._show_empty_state()

    # -- data ------------------------------------------------------------- #
    def _load_labels(self) -> Dict[str, dict]:
        if self.labels_path.exists():
            try:
                data = json.loads(self.labels_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except (OSError, json.JSONDecodeError) as exc:
                print(f"warning: could not read {self.labels_path}: {exc}", file=sys.stderr)
        return {}

    def _record_for(self, name: str) -> dict:
        return self.labels.setdefault(name, {"angle": 0.0, "comment": "", "confirmed": False})

    def _is_labelled(self, name: str) -> bool:
        rec = self.labels.get(name)
        return bool(rec) and bool(rec.get("confirmed"))

    # -- folder selection ------------------------------------------------- #
    def _resolve_labels_path(self, directory: Path) -> Path:
        """Choose the labels file for a folder, preferring an existing one.

        Priority: the explicit ``--labels`` path, then a canonical
        ``sample_labels.json`` in the run directory, then the per-folder
        ``<folder>_labels.json``. A brand-new session writes to
        ``sample_labels.json``.
        """
        if self._fixed_labels:
            return self.labels_path
        canonical = self.run_dir / "sample_labels.json"
        if canonical.exists():
            return canonical
        per_folder = self.run_dir / f"{directory.name}_labels.json"
        if per_folder.exists():
            return per_folder
        return canonical

    def load_directory(
        self, directory: Path, labels_path: Optional[Path] = None, index: int = 0
    ) -> None:
        """Load an image folder, resume its labels and jump to ``index``."""
        directory = Path(directory)
        images = find_images(directory)
        if not images:
            messagebox.showwarning(
                "geltilt", f"No supported images found in:\n{directory}\n\n"
                           f"Supported: {', '.join(sorted(IMAGE_EXTS))}"
            )
            return
        self._store_current()
        self.folder = directory
        self.images = images
        self.src_pil = None
        self._view_ready = False
        if labels_path is not None:
            self.labels_path = Path(labels_path)
        else:
            self.labels_path = self._resolve_labels_path(directory)
        self.labels = self._load_labels()
        self.index = max(0, min(int(index), len(images) - 1))
        self._populate_list()
        self._goto(self.index)
        self._update_folder_label()
        self._update_status()

    def _choose_folder(self) -> None:
        start = self.folder
        if start is None:
            candidate = self.run_dir / "samples"
            start = candidate if candidate.is_dir() else self.run_dir
        chosen = filedialog.askdirectory(initialdir=str(start), title="Select image folder")
        if chosen:
            self.load_directory(Path(chosen))

    # -- session (resume) ------------------------------------------------- #
    def _load_session(self) -> Dict[str, object]:
        try:
            data = json.loads(self.session_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_session(self) -> None:
        if self.folder is None:
            return
        try:
            self.session_path.write_text(
                json.dumps({
                    "folder": str(self.folder),
                    "index": int(self.index),
                    "labels": self.labels_path.name,
                }, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

    # -- UI --------------------------------------------------------------- #
    def _build_ui(self) -> None:
        root = self.root
        s = self.ui_scale
        self.guide_spacing = max(10, int(round(80 * s)))
        self.hud_font = tkfont.Font(root=root, family="Consolas", size=10)
        root.title("geltilt - rotation ground-truth labeler")
        root.geometry(f"{int(1280 * s)}x{int(880 * s)}")
        root.minsize(int(960 * s), int(660 * s))

        toolbar = ttk.Frame(root)
        toolbar.pack(fill="x", padx=8, pady=(8, 0))
        ttk.Button(toolbar, text="Open folder...", command=self._choose_folder).pack(side="left")
        self.folder_label = ttk.Label(toolbar, text="no folder selected")
        self.folder_label.pack(side="left", padx=12)
        self.path_label = ttk.Label(toolbar, text="", foreground="#666666")
        self.path_label.pack(side="right")

        main = ttk.Frame(root)
        main.pack(fill="both", expand=True)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # Left: image list.
        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="ns", padx=(8, 4), pady=8)
        ttk.Label(left, text="Images").pack(anchor="w")
        list_wrap = ttk.Frame(left)
        list_wrap.pack(fill="both", expand=True)
        self.listbox = tk.Listbox(list_wrap, width=36, exportselection=False, activestyle="dotbox")
        self.listbox.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(list_wrap, orient="vertical", command=self.listbox.yview)
        scroll.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=scroll.set)
        self.listbox.bind("<<ListboxSelect>>", self._on_list_select)
        self.listbox.bind("<Double-Button-1>", lambda e: self._goto_next_unlabelled())

        # Right: canvas with guides.
        self.canvas = tk.Canvas(main, bg="#101010", highlightthickness=0)
        self.canvas.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)
        self.canvas.bind("<Configure>", self._on_configure)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._zoom(1.1, e.x, e.y))
        self.canvas.bind("<Button-5>", lambda e: self._zoom(1 / 1.1, e.x, e.y))

        bottom = ttk.Frame(root)
        bottom.pack(fill="x", padx=8, pady=(0, 8))

        # Rotation controls.
        rot = ttk.LabelFrame(bottom, text="Rotation (angle applied to the raw image; positive = CCW)")
        rot.pack(fill="x", pady=(0, 6))
        self.angle_var = tk.DoubleVar(value=0.0)
        tk.Scale(
            rot, from_=-15.0, to=15.0, resolution=0.05, orient="horizontal",
            showvalue=False, length=520, variable=self.angle_var,
            command=self._on_scale,
        ).pack(side="left", padx=(8, 4), fill="x", expand=True)
        self.angle_label = ttk.Label(rot, text="+0.00", width=8, anchor="e",
                                     font=("TkDefaultFont", 13, "bold"))
        self.angle_label.pack(side="left", padx=4)
        for text, delta in (("-1", -1.0), ("-0.1", -0.1), ("+0.1", 0.1), ("+1", 1.0)):
            ttk.Button(rot, text=text, width=4,
                       command=lambda d=delta: self._nudge(d)).pack(side="left", padx=1)
        ttk.Button(rot, text="Zero", width=5, command=lambda: self._apply_angle(0.0)).pack(side="left", padx=(6, 8))

        # Guides + navigation + save.
        tools = ttk.Frame(bottom)
        tools.pack(fill="x", pady=(0, 6))
        self.guides_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tools, text="Vertical guides", variable=self.guides_var,
                        command=self._render).pack(side="left", padx=(0, 8))
        ttk.Label(tools, text="spacing:").pack(side="left")
        self.spacing_var = tk.IntVar(value=self.guide_spacing)
        self.spacing_box = ttk.Spinbox(tools, from_=20, to=400, increment=10, width=5,
                                       textvariable=self.spacing_var, command=self._on_spacing)
        self.spacing_box.pack(side="left", padx=(2, 14))
        self.spacing_box.bind("<Return>", lambda e: self._on_spacing())
        self.spacing_box.bind("<FocusOut>", lambda e: self._on_spacing())
        ttk.Button(tools, text="< Prev", command=self._prev).pack(side="left")
        ttk.Button(tools, text="Next >", command=self._next).pack(side="left", padx=4)
        ttk.Button(tools, text="Next unlabelled", command=self._goto_next_unlabelled).pack(side="left", padx=4)
        ttk.Button(tools, text="Fit view", command=self._reset_view).pack(side="left", padx=4)
        ttk.Button(tools, text="Save & confirm", command=lambda: self._save(confirm_current=True)).pack(side="right")
        self.status = ttk.Label(tools, text="")
        self.status.pack(side="right", padx=12)

        # Comment + confirmation.
        comment_frame = ttk.LabelFrame(bottom, text="Comment for this image")
        comment_frame.pack(fill="x")
        self.comment = tk.Text(comment_frame, height=3, wrap="word")
        self.comment.pack(side="left", fill="both", expand=True, padx=(8, 4), pady=6)
        self.confirmed_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(comment_frame, text="Confirmed\n(ground truth)", variable=self.confirmed_var,
                        command=self._on_confirm).pack(side="right", padx=8)

        # Keyboard shortcuts.
        root.bind("<Control-o>", lambda e: self._choose_folder())
        root.bind("<Control-s>", lambda e: self._save(confirm_current=True))
        root.bind("<Prior>", lambda e: self._prev())
        root.bind("<Next>", lambda e: self._next())
        root.bind("<Left>", lambda e: self._key_nudge(e, -0.1))
        root.bind("<Right>", lambda e: self._key_nudge(e, 0.1))
        root.bind("<bracketleft>", lambda e: self._key_nudge(e, -1.0))
        root.bind("<bracketright>", lambda e: self._key_nudge(e, 1.0))
        root.bind("<Key-r>", lambda e: self._reset_view() if not self._text_focus() else None)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _update_folder_label(self) -> None:
        self.folder_label.config(text=f"folder: {self.folder}" if self.folder else "no folder selected")
        self.path_label.config(text=f"labels -> {self.labels_path}")

    def _show_empty_state(self) -> None:
        self.canvas.delete("all")
        w, h = self._canvas_size()
        self.canvas.create_text(w / 2, h / 2, text="Open an image folder to start labelling\n(ctrl+O)",
                                fill="#888888", font=("TkDefaultFont", 14), justify="center")

    def _populate_list(self) -> None:
        self.listbox.delete(0, tk.END)
        for path in self.images:
            mark = "[x]" if self._is_labelled(path.name) else "[ ]"
            self.listbox.insert(tk.END, f"{mark} {path.name}")

    def _select_list(self) -> None:
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(self.index)
        self.listbox.see(self.index)

    # -- loading ---------------------------------------------------------- #
    def _goto(self, index: int) -> None:
        if not self.images:
            return
        self._store_current()
        self.index = max(0, min(index, len(self.images) - 1))
        path = self.images[self.index]
        try:
            arr = load_array(path)
        except Exception as exc:  # noqa: BLE001 - keep the session usable
            messagebox.showerror("Load error", f"could not load {path.name}:\n{exc}")
            return
        u8 = preprocess_for_display(arr)
        self.src_shape = (u8.shape[1], u8.shape[0])
        self.src_pil = Image.fromarray(u8, mode="L")
        self.anchor = content_anchor(u8)

        rec = self._record_for(path.name)
        self._suspend = True
        self.angle = float(rec.get("angle", 0.0))
        self.angle_var.set(self.angle)
        self.comment.delete("1.0", tk.END)
        self.comment.insert("1.0", rec.get("comment", "") or "")
        self.confirmed_var.set(bool(rec.get("confirmed", False)))
        self._suspend = False
        self.angle_label.config(text=f"{self.angle:+.2f}")

        self._view_ready = False
        self._reset_view()
        self._select_list()
        self._update_status()
        self._save_session()

    def _store_current(self) -> None:
        if not self.images or self.src_pil is None or self._suspend:
            return
        name = self.images[self.index].name
        self.labels[name] = {
            "angle": round(float(self.angle), 2),
            "comment": self.comment.get("1.0", tk.END).strip(),
            "confirmed": bool(self.confirmed_var.get()),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    # -- view ------------------------------------------------------------- #
    def _canvas_size(self) -> Tuple[int, int]:
        return max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())

    def _reset_view(self) -> None:
        if self.src_pil is None:
            self._show_empty_state()
            return
        w, h = self._canvas_size()
        if w <= 2 or h <= 2:
            return
        window = content_window(np.asarray(self.src_pil))
        if window is not None:
            y0, y1, x0, x1 = window
            bw, bh = max(1, x1 - x0), max(1, y1 - y0)
        else:
            bw, bh = self.src_shape
        self.scale = min(0.92 * w / bw, 0.92 * h / bh)
        self.center = (w / 2.0, h / 2.0)
        self._view_ready = True
        self._render()

    def _render(self) -> None:
        if self.src_pil is None:
            self._show_empty_state()
            return
        if not self._view_ready:
            return
        w, h = self._canvas_size()
        if w <= 2 or h <= 2:
            return
        img = render_display(self.src_pil, (w, h), self.angle, self.scale, self.anchor, self.center)
        self._imgtk = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._imgtk)

        if self.guides_var.get():
            for gx in range(self.guide_spacing, w, self.guide_spacing):
                self.canvas.create_line(gx, 0, gx, h, fill="#ffe000", dash=(5, 5), width=1)
        self.canvas.create_line(self.center[0], 0, self.center[0], h, fill="#00d0ff", width=1)

        name = self.images[self.index].name if self.images else "-"
        hud = (f"{name}   |   angle {self.angle:+.2f} deg   |   zoom {self.scale:.3f}x   "
               f"|   {self.index + 1}/{len(self.images)}")
        tw = self.hud_font.measure(hud)
        th = self.hud_font.metrics("linespace")
        self.canvas.create_rectangle(8, 8, 8 + tw + 14, 8 + th + 10, fill="#000000", outline="")
        self.canvas.create_text(15, 8 + (th + 10) // 2, anchor="w", text=hud, fill="#00ff66",
                                font=self.hud_font)

    def _zoom(self, factor: float, sx: float, sy: float) -> None:
        if self.src_pil is None:
            return
        px, py = screen_to_source((sx, sy), self.angle, self.scale, self.anchor, self.center)
        new_scale = min(40.0, max(0.02, self.scale * factor))
        cx, cy = source_to_screen((px, py), self.angle, new_scale, self.anchor, self.center)
        self.scale = new_scale
        self.center = (self.center[0] + (sx - cx), self.center[1] + (sy - cy))
        self._render()

    # -- events ----------------------------------------------------------- #
    def _on_configure(self, _event: tk.Event) -> None:
        if self.src_pil is None:
            self._show_empty_state()
        elif not self._view_ready:
            self._reset_view()
        else:
            self._render()

    def _on_press(self, event: tk.Event) -> None:
        self._drag_from = (event.x, event.y)

    def _on_drag(self, event: tk.Event) -> None:
        if self._drag_from is None:
            return
        dx, dy = event.x - self._drag_from[0], event.y - self._drag_from[1]
        self.center = (self.center[0] + dx, self.center[1] + dy)
        self._drag_from = (event.x, event.y)
        self._render()

    def _on_release(self, _event: tk.Event) -> None:
        self._drag_from = None

    def _on_wheel(self, event: tk.Event) -> None:
        self._zoom(1.1 if event.delta > 0 else 1 / 1.1, event.x, event.y)

    def _on_scale(self, value: str) -> None:
        if self._suspend:
            return
        self.angle = float(value)
        self.angle_label.config(text=f"{self.angle:+.2f}")
        self._render()

    def _apply_angle(self, angle: float) -> None:
        self.angle = max(-15.0, min(15.0, float(angle)))
        self._suspend = True
        self.angle_var.set(self.angle)
        self._suspend = False
        self.angle_label.config(text=f"{self.angle:+.2f}")
        self._render()

    def _nudge(self, delta: float) -> None:
        self._apply_angle(self.angle + delta)

    def _key_nudge(self, event: tk.Event, delta: float) -> None:
        if self._text_focus():
            return None
        self._nudge(delta)
        return "break"

    def _text_focus(self) -> bool:
        return isinstance(self.root.focus_get(), tk.Text)

    def _on_spacing(self) -> None:
        try:
            self.guide_spacing = max(10, int(self.spacing_var.get()))
        except (tk.TclError, ValueError):
            return
        self._render()

    def _on_confirm(self) -> None:
        if not self.images:
            return
        name = self.images[self.index].name
        rec = self._record_for(name)
        rec["confirmed"] = bool(self.confirmed_var.get())
        self._populate_list()
        self._select_list()
        self._update_status()

    def _on_list_select(self, _event: tk.Event) -> None:
        sel = self.listbox.curselection()
        if sel and sel[0] != self.index:
            self._goto(int(sel[0]))

    # -- navigation ------------------------------------------------------- #
    def _prev(self) -> None:
        if self.index > 0:
            self._goto(self.index - 1)

    def _next(self) -> None:
        if self.index < len(self.images) - 1:
            self._goto(self.index + 1)

    def _goto_next_unlabelled(self) -> None:
        if not self.images:
            return
        n = len(self.images)
        for step in range(1, n + 1):
            i = (self.index + step) % n
            if not self._is_labelled(self.images[i].name):
                self._goto(i)
                return
        messagebox.showinfo("geltilt", "All images are confirmed.")

    # -- persistence ------------------------------------------------------ #
    def _save(self, confirm_current: bool = False) -> None:
        if confirm_current and self.images and self.src_pil is not None:
            self.confirmed_var.set(True)
            self._on_confirm()
            self._update_status("saved")
        self._store_current()
        try:
            self.labels_path.parent.mkdir(parents=True, exist_ok=True)
            self.labels_path.write_text(json.dumps(self.labels, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Save error", f"could not write {self.labels_path}:\n{exc}")
            return
        self._save_session()
        self._populate_list()
        self._select_list()
        self._update_status("saved")
        self._flash_status(f"saved {len(self.labels)} labels -> {self.labels_path}")

    def _on_close(self) -> None:
        if self.images:
            self._store_current()
            try:
                self._save()
            except Exception:  # noqa: BLE001 - never block closing
                pass
        self.root.destroy()

    def _update_status(self, extra: str = "") -> None:
        done = sum(1 for p in self.images if self._is_labelled(p.name))
        text = f"{done}/{len(self.images)} confirmed" if self.images else "no images loaded"
        if extra:
            text += f"  -  {extra}"
        self.status.config(text=text)

    def _flash_status(self, text: str) -> None:
        self.status.config(text=text)
        self.root.after(2500, self._update_status)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="geltilt-label",
        description="Human labelling of correct rotation (ground truth) for gel images.",
    )
    p.add_argument("directory", nargs="?", default=None,
                   help="image folder to open at startup (optional; can also be chosen in the UI)")
    p.add_argument("--labels", default=None,
                   help="explicit labels JSON path (default: existing labels file, else sample_labels.json)")
    p.add_argument("--selftest", action="store_true", help=argparse.SUPPRESS)
    return p


def _selftest(run_dir: Path, directory: Optional[Path] = None) -> int:
    """Headless-ish check used to validate packaged builds; writes selftest.json.

    Uses throwaway ``_selftest_*`` files so it never reads or overwrites a real
    labels/session file.
    """
    report: Dict[str, object] = {"ok": True, "run_dir": str(run_dir), "python": sys.version}
    cleanup: List[Path] = []
    try:
        import cv2
        import PIL
        report["cv2"] = cv2.__version__
        report["numpy"] = np.__version__
        report["pillow"] = PIL.__version__
        root = tk.Tk()
        root.withdraw()
        app = RotationLabeler(root, run_dir=run_dir)
        root.update()
        app._show_empty_state()
        if directory is not None and directory.is_dir():
            tmp_labels = run_dir / "_selftest_labels.json"
            tmp_session = run_dir / "_selftest_session.json"
            app.session_path = tmp_session
            app.load_directory(directory, labels_path=tmp_labels)
            root.update()
            report["images"] = len(app.images)
            report["labels_path"] = str(tmp_labels)
            app._apply_angle(-2.5)
            app._save()
            report["labels_written"] = tmp_labels.exists()
            cleanup = [tmp_labels, tmp_session]
        root.destroy()
    except Exception as exc:  # noqa: BLE001 - report, don't crash
        report["ok"] = False
        report["error"] = repr(exc)
    for path in cleanup:
        try:
            path.unlink()
        except OSError:
            pass
    try:
        (run_dir / "selftest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    except OSError:
        pass
    return 0 if report.get("ok") else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    enable_dpi_awareness()
    args = build_parser().parse_args(argv)
    run_dir = run_directory()

    if args.selftest:
        return _selftest(run_dir, Path(args.directory) if args.directory else None)

    directory: Optional[Path] = Path(args.directory) if args.directory else None

    scale = dpi_scale()
    root = tk.Tk()
    apply_tk_scaling(root, scale)
    app = RotationLabeler(root, run_dir=run_dir, labels_path=Path(args.labels) if args.labels else None,
                          ui_scale=scale)

    # Resume: an explicit folder wins; otherwise reopen the last session, then
    # fall back to <run dir>/samples. Restore the last image position when the
    # folder matches the previous session.
    session = app._load_session()
    resume_index = 0
    if directory is None:
        sess_folder = session.get("folder")
        if sess_folder and Path(str(sess_folder)).is_dir():
            directory = Path(str(sess_folder))
            try:
                resume_index = int(session.get("index", 0) or 0)
            except (TypeError, ValueError):
                resume_index = 0
        elif (run_dir / "samples").is_dir():
            directory = run_dir / "samples"
    elif sess_folder := session.get("folder"):
        try:
            if Path(str(sess_folder)).resolve() == directory.resolve():
                resume_index = int(session.get("index", 0) or 0)
        except OSError:
            resume_index = 0

    if directory is not None and directory.is_dir():
        app.load_directory(directory, index=resume_index)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())