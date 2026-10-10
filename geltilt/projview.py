"""Interactive gel projection viewer.

A Tkinter tool that lets the operator:

1. mark a **detection boundary** (a rectangular region of the gel) with a mouse
   drag,
2. **rotate the gel manually** with a free +-180 deg slider (positive = CCW,
   matching :func:`geltilt.core.correct_rotation`), and
3. read the **vertical projection intensity** of the boundary as a live line
   graph (one point per x column).

The boundary is anchored to the *gel*, not the screen: it is stored in source
pixel coordinates and therefore rotates with the image. The projection is the
column sum of the **raw** (unstretched) pixel intensities inside that boundary,
binned by the rotated x coordinate. Because the boundary follows the gel, the
only thing that changes the graph as the slider moves is the lane orientation,
so sharpening peaks mark the angle that makes the lanes vertical.

The graph is drawn on a plain :class:`tkinter.Canvas` (no matplotlib) so the
packaged executable stays small and dependency-free.

Run with ``geltilt-proj [directory]`` or double-click the packaged executable.
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
from .labeler import (
    IMAGE_EXTS,
    Point,
    apply_tk_scaling,
    content_anchor,
    dpi_scale,
    enable_dpi_awareness,
    find_images,
    preprocess_for_display,
    render_display,
    run_directory,
    screen_to_source,
    source_to_screen,
)
from .projection import sharpness
from .viz import content_window

Roi = Tuple[float, float, float, float]  # (x0, y0, x1, y1) in source pixels

PLOT_PAD_L = 46
PLOT_PAD_R = 12
PLOT_PAD_T = 28
PLOT_PAD_B = 26


# --------------------------------------------------------------------------- #
# Pure helpers (GUI-free, unit tested).                                        #
# --------------------------------------------------------------------------- #
def normalize_roi(roi: Roi) -> Roi:
    """Return ``roi`` with min/max ordered (x0 <= x1, y0 <= y1)."""
    x0, y0, x1, y1 = (float(v) for v in roi)
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def clamp_roi(roi: Roi, shape: Sequence[int]) -> Roi:
    """Clip a rectangle to a ``(h, w)`` image, keeping it non-empty when possible."""
    h, w = int(shape[0]), int(shape[1])
    x0, y0, x1, y1 = normalize_roi(roi)
    x0 = min(max(0.0, x0), max(0.0, w - 1.0))
    x1 = min(max(x0 + 1.0, x1), float(w))
    y0 = min(max(0.0, y0), max(0.0, h - 1.0))
    y1 = min(max(y0 + 1.0, y1), float(h))
    return (x0, y0, x1, y1)


def roi_corners(roi: Roi) -> List[Point]:
    """The four corners of an axis-aligned rectangle, clockwise from top-left."""
    x0, y0, x1, y1 = normalize_roi(roi)
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def default_roi_from_u8(u8: np.ndarray) -> Roi:
    """A sensible starting boundary: the visible band content, else a 5% inset."""
    gray = to_gray(u8).astype(np.uint8)
    h, w = gray.shape[:2]
    window = content_window(gray)
    if window is None:
        mx, my = max(1, int(w * 0.05)), max(1, int(h * 0.05))
        return (float(mx), float(my), float(w - mx), float(h - my))
    y0, y1, x0, x1 = window
    return (float(x0), float(y0), float(x1), float(y1))


def vertical_projection_roi(
    raw_gray: np.ndarray,
    roi: Roi,
    angle_deg: float,
    anchor: Point,
    max_samples: int = 600_000,
) -> np.ndarray:
    """Column-sum of raw intensity inside ``roi``, binned by rotated x.

    For every sampled source pixel the rotated x coordinate is
    ``c * (x - ax) + s * (y - ay)`` (the same counter-clockwise convention used
    for rendering), and its raw intensity is accumulated into that column. The
    result is the vertical projection of the boundary as it appears after
    ``angle_deg`` rotation.

    Large boundaries are strided so at most ``max_samples`` pixels are read,
    keeping the live update fast on full-resolution gel TIFFs.
    """
    gray = np.asarray(raw_gray)
    if gray.ndim != 2:
        gray = to_gray(gray)
    gray = gray.astype(np.float64, copy=False)
    h, w = gray.shape

    x0, y0, x1, y1 = normalize_roi(roi)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return np.zeros(0, dtype=np.float64)
    x0, y0, x1, y1 = clamp_roi((x0, y0, x1, y1), (h, w))
    x0i, y0i = int(math.floor(x0)), int(math.floor(y0))
    x1i, y1i = int(math.ceil(x1)), int(math.ceil(y1))
    if x1i - x0i < 1 or y1i - y0i < 1:
        return np.zeros(0, dtype=np.float64)

    bw, bh = x1i - x0i, y1i - y0i
    stride = 1
    if max_samples and bw * bh > max_samples:
        stride = int(math.ceil(math.sqrt(bw * bh / float(max_samples))))

    xs = np.arange(x0i, x1i, stride)
    ys = np.arange(y0i, y1i, stride)
    block = gray[np.ix_(ys, xs)]
    yy, xx = np.meshgrid(ys.astype(np.float64), xs.astype(np.float64), indexing="ij")

    th = math.radians(float(angle_deg))
    c, s = math.cos(th), math.sin(th)
    qx = c * (xx - float(anchor[0])) + s * (yy - float(anchor[1]))
    qxi = np.rint(qx).astype(np.int64)

    lo = int(qxi.min())
    ncols = int(qxi.max()) - lo + 1
    idx = (qxi - lo).ravel()
    profile = np.bincount(idx, weights=block.ravel(), minlength=ncols)
    return profile.astype(np.float64)


def profile_sharpness(profile: np.ndarray, method: str = "tv2") -> float:
    """Sharpness of a projection profile, normalised to unit area first."""
    p = np.asarray(profile, dtype=np.float64)
    total = float(p.sum())
    if p.size == 0 or total <= 0:
        return 0.0
    return sharpness(p / total, method=method)


def profile_to_points(
    profile: np.ndarray,
    width: int,
    height: int,
    pad_l: int = PLOT_PAD_L,
    pad_r: int = PLOT_PAD_R,
    pad_t: int = PLOT_PAD_T,
    pad_b: int = PLOT_PAD_B,
    ymax: Optional[float] = None,
) -> List[float]:
    """Flat ``[x0, y0, x1, y1, ...]`` polyline for a profile, canvas-fitted."""
    p = np.asarray(profile, dtype=np.float64)
    if p.size < 2:
        return []
    if ymax is None:
        ymax = float(p.max())
    if not np.isfinite(ymax) or ymax <= 0:
        ymax = 1.0
    x_lo, x_hi = float(pad_l), float(max(pad_l + 1, width - pad_r))
    y_lo, y_hi = float(max(pad_t + 1, height - pad_b)), float(pad_t)
    xs = np.linspace(x_lo, x_hi, p.size)
    ys = y_lo + (p / ymax) * (y_hi - y_lo)
    pts: List[float] = []
    for x, y in zip(xs, ys):
        pts.extend([float(x), float(y)])
    return pts


# --------------------------------------------------------------------------- #
# GUI                                                                          #
# --------------------------------------------------------------------------- #
class ProjectionViewer:
    """Interactive boundary + rotation + live vertical-projection viewer."""

    def __init__(
        self,
        root: tk.Tk,
        images: Optional[Sequence[Path]] = None,
        run_dir: Optional[Path] = None,
        ui_scale: float = 1.0,
    ) -> None:
        self.root = root
        self.ui_scale = float(ui_scale) if ui_scale and ui_scale > 0 else 1.0
        self.run_dir = Path(run_dir) if run_dir is not None else run_directory()
        self.state_path = self.run_dir / "projview_state.json"
        self.folder: Optional[Path] = None
        self.images: List[Path] = list(images) if images else []
        self.image_state: Dict[str, dict] = {}

        self.index = 0
        self.src_pil: Optional[Image.Image] = None
        self.raw_gray: Optional[np.ndarray] = None
        self.src_shape: Tuple[int, int] = (0, 0)
        self.anchor: Point = (0.0, 0.0)
        self.center: Point = (0.0, 0.0)
        self.scale = 1.0
        self.angle = 0.0
        self.roi: Optional[Roi] = None
        self.default_roi: Optional[Roi] = None
        self.profile: Optional[np.ndarray] = None
        self.guide_spacing = 80

        self._imgtk: Optional[ImageTk.PhotoImage] = None
        self._pan_from: Optional[Point] = None
        self._roi_from: Optional[Point] = None
        self._suspend = False
        self._view_ready = False

        self._load_state()
        self._build_ui()
        self._update_folder_label()
        if self.images:
            self._goto(0)
        else:
            self._show_empty_state()

    # -- persistence ------------------------------------------------------ #
    def _load_state(self) -> None:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        if isinstance(data, dict) and isinstance(data.get("images"), dict):
            self.image_state = data["images"]

    def _save_state(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(
                json.dumps({
                    "folder": str(self.folder) if self.folder else None,
                    "index": int(self.index),
                    "images": self.image_state,
                }, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

    def _store_current(self) -> None:
        if not self.images or self.src_pil is None or self._suspend:
            return
        name = self.images[self.index].name
        rec: dict = {"angle": round(float(self.angle), 2), "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if self.roi is not None:
            x0, y0, x1, y1 = normalize_roi(self.roi)
            rec["roi"] = [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]
        self.image_state[name] = rec

    # -- folder / loading ------------------------------------------------- #
    def load_directory(self, directory: Path, index: int = 0) -> None:
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
        self.raw_gray = to_gray(arr)
        self.default_roi = default_roi_from_u8(u8)

        rec = self.image_state.get(path.name, {})
        self._suspend = True
        self.angle = float(rec.get("angle", 0.0))
        self.angle_var.set(self.angle)
        roi = rec.get("roi")
        self.roi = normalize_roi(tuple(roi)) if roi else self.default_roi
        self._suspend = False
        self.angle_label.config(text=f"{self.angle:+.1f}")

        self._view_ready = False
        self._reset_view()
        self._select_list()
        self._update_projection()
        self._update_status()
        self._save_state()

    # -- UI --------------------------------------------------------------- #
    def _build_ui(self) -> None:
        root = self.root
        s = self.ui_scale
        self.guide_spacing = max(10, int(round(80 * s)))
        self.hud_font = tkfont.Font(root=root, family="Consolas", size=10)
        root.title("geltilt - gel projection viewer")
        root.geometry(f"{int(1520 * s)}x{int(900 * s)}")
        root.minsize(int(1120 * s), int(680 * s))

        toolbar = ttk.Frame(root)
        toolbar.pack(fill="x", padx=8, pady=(8, 0))
        ttk.Button(toolbar, text="Open folder...", command=self._choose_folder).pack(side="left")
        self.folder_label = ttk.Label(toolbar, text="no folder selected")
        self.folder_label.pack(side="left", padx=12)
        ttk.Button(toolbar, text="< Prev", command=self._prev).pack(side="left", padx=(12, 2))
        ttk.Button(toolbar, text="Next >", command=self._next).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Fit view", command=self._reset_view).pack(side="left", padx=(12, 2))
        ttk.Button(toolbar, text="Reset boundary", command=self._reset_boundary).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Clear boundary", command=self._clear_boundary).pack(side="left", padx=2)
        self.path_label = ttk.Label(toolbar, text="", foreground="#666666")
        self.path_label.pack(side="right")

        main = ttk.Frame(root)
        main.pack(fill="both", expand=True)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="ns", padx=(8, 4), pady=8)
        ttk.Label(left, text="Images").pack(anchor="w")
        list_wrap = ttk.Frame(left)
        list_wrap.pack(fill="both", expand=True)
        self.listbox = tk.Listbox(list_wrap, width=30, exportselection=False, activestyle="dotbox")
        self.listbox.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(list_wrap, orient="vertical", command=self.listbox.yview)
        scroll.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=scroll.set)
        self.listbox.bind("<<ListboxSelect>>", self._on_list_select)

        self.canvas = tk.Canvas(main, bg="#101010", highlightthickness=0)
        self.canvas.grid(row=0, column=1, sticky="nsew", padx=4, pady=8)
        self.canvas.bind("<Configure>", self._on_configure)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<ButtonPress-3>", self._on_pan_start)
        self.canvas.bind("<B3-Motion>", self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_pan_end)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._zoom(1.1, e.x, e.y))
        self.canvas.bind("<Button-5>", lambda e: self._zoom(1 / 1.1, e.x, e.y))

        plot_wrap = ttk.LabelFrame(main, text="Vertical projection (sum of raw intensity)")
        plot_wrap.grid(row=0, column=2, sticky="nsew", padx=(4, 8), pady=8)
        plot_wrap.rowconfigure(0, weight=1)
        plot_wrap.columnconfigure(0, weight=1)
        self.plot_canvas = tk.Canvas(plot_wrap, bg="#0b0b12", highlightthickness=0,
                                     width=int(380 * s))
        self.plot_canvas.grid(row=0, column=0, sticky="nsew", padx=6, pady=(6, 2))
        self.plot_canvas.bind("<Configure>", lambda e: self._draw_plot())
        self.sharp_label = ttk.Label(plot_wrap, text="sharpness: -")
        self.sharp_label.grid(row=1, column=0, sticky="w", padx=8, pady=(0, 6))

        bottom = ttk.Frame(root)
        bottom.pack(fill="x", padx=8, pady=(0, 8))

        rot = ttk.LabelFrame(
            bottom,
            text="Manual rotation (applied to the raw image; positive = CCW)",
        )
        rot.pack(fill="x", pady=(0, 6))
        self.angle_var = tk.DoubleVar(value=0.0)
        tk.Scale(
            rot, from_=-180.0, to=180.0, resolution=0.1, orient="horizontal",
            showvalue=False, length=760, variable=self.angle_var, command=self._on_scale,
        ).pack(side="left", padx=(8, 4), fill="x", expand=True)
        self.angle_label = ttk.Label(rot, text="+0.0", width=7, anchor="e",
                                     font=("TkDefaultFont", 13, "bold"))
        self.angle_label.pack(side="left", padx=4)
        for text, delta in (("-10", -10.0), ("-1", -1.0), ("-0.1", -0.1),
                            ("+0.1", 0.1), ("+1", 1.0), ("+10", 10.0)):
            ttk.Button(rot, text=text, width=4,
                       command=lambda d=delta: self._nudge(d)).pack(side="left", padx=1)
        ttk.Button(rot, text="Zero", width=5,
                   command=lambda: self._apply_angle(0.0)).pack(side="left", padx=(6, 8))

        tools = ttk.Frame(bottom)
        tools.pack(fill="x")
        self.mode_var = tk.StringVar(value="boundary")
        ttk.Label(tools, text="Left-drag:").pack(side="left")
        ttk.Radiobutton(tools, text="Boundary", value="boundary",
                        variable=self.mode_var).pack(side="left", padx=(4, 0))
        ttk.Radiobutton(tools, text="Pan", value="pan",
                        variable=self.mode_var).pack(side="left", padx=(0, 10))
        ttk.Label(tools, text="(right-drag pans, wheel zooms)", foreground="#666666").pack(side="left")
        self.guides_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tools, text="Vertical guides", variable=self.guides_var,
                        command=self._render).pack(side="left", padx=(16, 6))
        ttk.Label(tools, text="spacing:").pack(side="left")
        self.spacing_var = tk.IntVar(value=self.guide_spacing)
        self.spacing_box = ttk.Spinbox(tools, from_=20, to=400, increment=10, width=5,
                                       textvariable=self.spacing_var, command=self._on_spacing)
        self.spacing_box.pack(side="left", padx=(2, 0))
        self.spacing_box.bind("<Return>", lambda e: self._on_spacing())
        self.spacing_box.bind("<FocusOut>", lambda e: self._on_spacing())
        self.status = ttk.Label(tools, text="")
        self.status.pack(side="right")

        root.bind("<Control-o>", lambda e: self._choose_folder())
        root.bind("<Prior>", lambda e: self._prev())
        root.bind("<Next>", lambda e: self._next())
        root.bind("<Left>", lambda e: self._key_nudge(e, -1.0))
        root.bind("<Right>", lambda e: self._key_nudge(e, 1.0))
        root.bind("<comma>", lambda e: self._key_nudge(e, -0.1))
        root.bind("<period>", lambda e: self._key_nudge(e, 0.1))
        root.bind("<Key-r>", lambda e: self._reset_view() if not self._text_focus() else None)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _update_folder_label(self) -> None:
        self.folder_label.config(text=f"folder: {self.folder}" if self.folder else "no folder selected")
        self.path_label.config(text=f"state -> {self.state_path}")

    def _show_empty_state(self) -> None:
        self.canvas.delete("all")
        w, h = self._canvas_size()
        self.canvas.create_text(w / 2, h / 2,
                                text="Open an image folder to start\n(ctrl+O)",
                                fill="#888888", font=("TkDefaultFont", 14), justify="center")
        self._draw_plot()

    def _populate_list(self) -> None:
        self.listbox.delete(0, tk.END)
        for path in self.images:
            mark = "[x]" if path.name in self.image_state else "[ ]"
            self.listbox.insert(tk.END, f"{mark} {path.name}")

    def _select_list(self) -> None:
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(self.index)
        self.listbox.see(self.index)

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

        if self.roi is not None:
            pts: List[float] = []
            for corner in roi_corners(self.roi):
                sx, sy = source_to_screen(corner, self.angle, self.scale, self.anchor, self.center)
                pts.extend([sx, sy])
            self.canvas.create_polygon(pts, outline="#00d0ff", fill="#00d0ff",
                                       stipple="gray12", width=2)
            for i in range(0, len(pts), 2):
                x, y = pts[i], pts[i + 1]
                self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3,
                                        outline="#00d0ff", fill="#00202a")

        self.canvas.create_line(self.center[0], 0, self.center[0], h,
                                fill="#00d0ff", dash=(2, 6), width=1)

        name = self.images[self.index].name if self.images else "-"
        roi_txt = ""
        if self.roi is not None:
            x0, y0, x1, y1 = normalize_roi(self.roi)
            roi_txt = f"   |   boundary {x1 - x0:.0f}x{y1 - y0:.0f}px"
        hud = (f"{name}   |   angle {self.angle:+.1f} deg   |   zoom {self.scale:.3f}x"
               f"{roi_txt}   |   {self.index + 1}/{len(self.images)}")
        tw = self.hud_font.measure(hud)
        th = self.hud_font.metrics("linespace")
        self.canvas.create_rectangle(8, 8, 8 + tw + 14, 8 + th + 10, fill="#000000", outline="")
        self.canvas.create_text(15, 8 + (th + 10) // 2, anchor="w", text=hud,
                                fill="#00ff66", font=self.hud_font)

    # -- projection graph ------------------------------------------------- #
    def _update_projection(self) -> None:
        if self.raw_gray is None or self.roi is None:
            self.profile = None
        else:
            self.profile = vertical_projection_roi(
                self.raw_gray, self.roi, self.angle, self.anchor
            )
        self._draw_plot()
        if self.profile is None or self.profile.size < 2:
            self.sharp_label.config(text="sharpness: -")
        else:
            self.sharp_label.config(
                text=f"sharpness (tv2): {profile_sharpness(self.profile):.4g}"
                     f"   |   columns: {self.profile.size}"
            )

    def _draw_plot(self) -> None:
        c = self.plot_canvas
        try:
            w, h = c.winfo_width(), c.winfo_height()
        except tk.TclError:
            return
        c.delete("all")
        if w <= 10 or h <= 10:
            return
        p = self.profile
        if p is None or p.size < 2:
            c.create_text(w / 2, h / 2, text="Mark a detection boundary\n(left-drag on the gel)",
                          fill="#8899aa", font=("TkDefaultFont", 11), justify="center")
            return

        x_lo, x_hi = PLOT_PAD_L, max(PLOT_PAD_L + 1, w - PLOT_PAD_R)
        y_hi, y_lo = PLOT_PAD_T, max(PLOT_PAD_T + 1, h - PLOT_PAD_B)
        ymax = float(np.max(p))
        if not np.isfinite(ymax) or ymax <= 0:
            ymax = 1.0

        c.create_rectangle(x_lo, y_hi, x_hi, y_lo, outline="#2a3550")
        for frac in (0.25, 0.5, 0.75):
            gy = y_lo - frac * (y_lo - y_hi)
            c.create_line(x_lo, gy, x_hi, gy, fill="#1b2438", dash=(2, 4))
        c.create_text(x_lo - 6, y_hi, anchor="e", text=f"{ymax:.0f}", fill="#7f8fb0",
                      font=("TkDefaultFont", 9))
        c.create_text(x_lo - 6, y_lo, anchor="e", text="0", fill="#7f8fb0",
                      font=("TkDefaultFont", 9))
        c.create_text((x_lo + x_hi) / 2, y_lo + 14, anchor="n",
                      text=f"x column  (0 .. {p.size - 1})", fill="#7f8fb0",
                      font=("TkDefaultFont", 9))

        draw_p = p
        if p.size > 1500:
            step = int(math.ceil(p.size / 1500.0))
            draw_p = p[::step]
        pts = profile_to_points(draw_p, w, h, PLOT_PAD_L, PLOT_PAD_R, PLOT_PAD_T, PLOT_PAD_B, ymax)
        if len(pts) >= 4:
            c.create_line(*pts, fill="#00d0ff", width=2)
        c.create_text(x_lo, y_hi - 8, anchor="w",
                      text="vertical projection - raw intensity", fill="#c8d4ee",
                      font=("TkDefaultFont", 10, "bold"))

    # -- zoom / pan ------------------------------------------------------- #
    def _zoom(self, factor: float, sx: float, sy: float) -> None:
        if self.src_pil is None:
            return
        px, py = screen_to_source((sx, sy), self.angle, self.scale, self.anchor, self.center)
        new_scale = min(40.0, max(0.02, self.scale * factor))
        cx, cy = source_to_screen((px, py), self.angle, new_scale, self.anchor, self.center)
        self.scale = new_scale
        self.center = (self.center[0] + (sx - cx), self.center[1] + (sy - cy))
        self._render()

    def _pan_start(self, sx: float, sy: float) -> None:
        self._pan_from = (sx, sy)

    def _pan_to(self, sx: float, sy: float) -> None:
        if self._pan_from is None:
            return
        dx, dy = sx - self._pan_from[0], sy - self._pan_from[1]
        self.center = (self.center[0] + dx, self.center[1] + dy)
        self._pan_from = (sx, sy)
        self._render()

    def _on_pan_start(self, event: tk.Event) -> None:
        self._pan_start(event.x, event.y)

    def _on_pan_drag(self, event: tk.Event) -> None:
        self._pan_to(event.x, event.y)

    def _on_pan_end(self, _event: tk.Event) -> None:
        self._pan_from = None

    # -- boundary drawing ------------------------------------------------- #
    def _boundary_mode(self, event: tk.Event) -> bool:
        if self.mode_var.get() == "pan":
            return False
        return not (int(getattr(event, "state", 0)) & 0x0001)  # shift => pan

    def _on_press(self, event: tk.Event) -> None:
        if self.src_pil is None:
            return
        if self._boundary_mode(event):
            self._roi_from = screen_to_source((event.x, event.y), self.angle,
                                              self.scale, self.anchor, self.center)
            self._pan_from = None
        else:
            self._pan_start(event.x, event.y)

    def _on_drag(self, event: tk.Event) -> None:
        if self.src_pil is None:
            return
        if self._roi_from is not None:
            x, y = screen_to_source((event.x, event.y), self.angle,
                                    self.scale, self.anchor, self.center)
            self.roi = normalize_roi((self._roi_from[0], self._roi_from[1], x, y))
            self._render()
            self._update_projection()
        elif self._pan_from is not None:
            self._pan_to(event.x, event.y)

    def _on_release(self, _event: tk.Event) -> None:
        if self._roi_from is not None and self.roi is not None:
            x0, y0, x1, y1 = normalize_roi(self.roi)
            if x1 - x0 < 2 or y1 - y0 < 2:
                self.roi = self.default_roi
            self.roi = clamp_roi(self.roi, self.src_shape[::-1])
            self._store_current()
            self._save_state()
            self._update_projection()
        self._roi_from = None
        self._pan_from = None

    def _reset_boundary(self) -> None:
        if self.default_roi is not None:
            self.roi = self.default_roi
            self._store_current()
            self._save_state()
            self._render()
            self._update_projection()

    def _clear_boundary(self) -> None:
        self.roi = None
        self._store_current()
        self._save_state()
        self._render()
        self._update_projection()

    # -- rotation --------------------------------------------------------- #
    def _on_wheel(self, event: tk.Event) -> None:
        self._zoom(1.1 if event.delta > 0 else 1 / 1.1, event.x, event.y)

    def _on_scale(self, value: str) -> None:
        if self._suspend:
            return
        self.angle = max(-180.0, min(180.0, float(value)))
        self.angle_label.config(text=f"{self.angle:+.1f}")
        self._render()
        self._update_projection()

    def _apply_angle(self, angle: float) -> None:
        self.angle = max(-180.0, min(180.0, float(angle)))
        self._suspend = True
        self.angle_var.set(round(self.angle, 1))
        self._suspend = False
        self.angle_label.config(text=f"{self.angle:+.1f}")
        self._render()
        self._update_projection()

    def _nudge(self, delta: float) -> None:
        self._apply_angle(round(self.angle + delta, 1))

    def _key_nudge(self, event: tk.Event, delta: float) -> Optional[str]:
        if self._text_focus():
            return None
        self._nudge(delta)
        return "break"

    def _text_focus(self) -> bool:
        return isinstance(self.root.focus_get(), tk.Text)

    # -- misc events ------------------------------------------------------ #
    def _on_configure(self, _event: tk.Event) -> None:
        if self.src_pil is None:
            self._show_empty_state()
        elif not self._view_ready:
            self._reset_view()
        else:
            self._render()

    def _on_spacing(self) -> None:
        try:
            self.guide_spacing = max(10, int(self.spacing_var.get()))
        except (tk.TclError, ValueError):
            return
        self._render()

    def _on_list_select(self, _event: tk.Event) -> None:
        sel = self.listbox.curselection()
        if sel and sel[0] != self.index:
            self._goto(int(sel[0]))

    def _prev(self) -> None:
        if self.index > 0:
            self._goto(self.index - 1)

    def _next(self) -> None:
        if self.index < len(self.images) - 1:
            self._goto(self.index + 1)

    def _update_status(self) -> None:
        text = f"{self.index + 1}/{len(self.images)}" if self.images else "no images loaded"
        self.status.config(text=text)

    def _on_close(self) -> None:
        if self.images:
            self._store_current()
            self._save_state()
        self.root.destroy()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="geltilt-proj",
        description="Mark a boundary, rotate the gel manually and read its vertical projection.",
    )
    p.add_argument("directory", nargs="?", default=None,
                   help="image folder to open at startup (optional; can also be chosen in the UI)")
    p.add_argument("--selftest", action="store_true", help=argparse.SUPPRESS)
    return p


def _selftest(run_dir: Path, directory: Optional[Path] = None) -> int:
    """Headless-ish check for packaged builds; writes projview_selftest.json."""
    report: Dict[str, object] = {"ok": True, "run_dir": str(run_dir), "python": sys.version}
    try:
        import cv2
        import PIL
        report["cv2"] = cv2.__version__
        report["numpy"] = np.__version__
        report["pillow"] = PIL.__version__
        root = tk.Tk()
        root.withdraw()
        app = ProjectionViewer(root, run_dir=run_dir)
        app.state_path = run_dir / "_selftest_state.json"
        root.update()
        if directory is not None and directory.is_dir():
            app.load_directory(directory)
            root.update()
            report["images"] = len(app.images)
            report["profile_columns"] = int(app.profile.size) if app.profile is not None else 0
            app._apply_angle(5.0)
            root.update()
            report["profile_columns_rotated"] = int(app.profile.size) if app.profile is not None else 0
        root.destroy()
    except Exception as exc:  # noqa: BLE001 - report, don't crash
        report["ok"] = False
        report["error"] = repr(exc)
    for leftover in (run_dir / "_selftest_state.json",):
        try:
            leftover.unlink()
        except OSError:
            pass
    try:
        (run_dir / "projview_selftest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    except OSError:
        pass
    return 0 if report.get("ok") else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    enable_dpi_awareness()
    args = build_parser().parse_args(argv)
    run_dir = run_directory()

    if args.selftest:
        return _selftest(run_dir, Path(args.directory) if args.directory else None)

    scale = dpi_scale()
    root = tk.Tk()
    apply_tk_scaling(root, scale)
    app = ProjectionViewer(root, run_dir=run_dir, ui_scale=scale)

    directory: Optional[Path] = Path(args.directory) if args.directory else None
    resume_index = 0
    if directory is None:
        try:
            state = json.loads(app.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = {}
        sess_folder = state.get("folder") if isinstance(state, dict) else None
        if sess_folder and Path(str(sess_folder)).is_dir():
            directory = Path(str(sess_folder))
            try:
                resume_index = int(state.get("index", 0) or 0)
            except (TypeError, ValueError):
                resume_index = 0
        elif (run_dir / "samples").is_dir():
            directory = run_dir / "samples"

    if directory is not None and directory.is_dir():
        app.load_directory(directory, index=resume_index)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())