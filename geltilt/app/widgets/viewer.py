"""The single image surface: fit / zoom / pan / rotate + overlays.

There is exactly one image surface (no split or side-by-side view). Overlays are
mapped through the *same* affine used to render the image, so they cannot drift
from the pixels. Rendering never mutates the underlying sample array.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional, Sequence, Tuple

import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from ...labeler import (
    content_anchor,
    render_display,
    screen_to_source,
    source_to_screen,
)
from ..crop import (
    normalize_rect,
    screen_rect,
    screen_to_display,
    source_bbox_to_display_rect,
)
from ..domain import LaneLine

Point = Tuple[float, float]

GUIDE_COLOR = "#ffe000"

CENTROID_COLOR = "#ff40ff"
LANE_COLOR = "#00ff66"

CROP_COLOR = "#ffe000"
CROP_DIM = (150, 0, 0, 130)  # dark red wash over the cropped-out regions
_MIN_NEW_CROP_PX = 4  # a left-click shorter than this does not start a crop
_PAN_MARGIN_FRAC = 0.25  # extra frame rendered around the canvas so a pan can move items


class SingleViewer(ttk.Frame):
    """Hosts exactly one canvas that renders the composited rotated image."""

    def __init__(self, master: tk.Misc, ui_scale: float = 1.0) -> None:
        super().__init__(master)
        self.ui_scale = float(ui_scale) if ui_scale and ui_scale > 0 else 1.0
        self.guide_spacing = max(10, int(round(80 * self.ui_scale)))

        # Small requested size so the packer lets the viewer expand into the
        # leftover space instead of squeezing the fixed bars around it.
        self.canvas = tk.Canvas(
            self, bg="#101010", highlightthickness=0,
            width=int(200 * self.ui_scale), height=int(120 * self.ui_scale),
        )
        self.canvas.pack(fill="both", expand=True)

        self._u8: Optional[np.ndarray] = None
        self._src_pil: Optional[Image.Image] = None
        self._src_shape: Tuple[int, int] = (0, 0)
        self._imgtk: Optional[ImageTk.PhotoImage] = None
        self.anchor: Point = (0.0, 0.0)
        self.center: Point = (0.0, 0.0)
        self.scale = 1.0
        self.angle = 0.0
        self._pan_from: Optional[Point] = None
        self._view_ready = False

        self._crop: Optional[Tuple[float, float, float, float]] = None
        self._crop_drag: Optional[str] = None
        self._crop_new: Optional[Point] = None
        self._crop_new_screen: Optional[Point] = None
        self._crop_prev: Optional[Tuple[float, float, float, float]] = None
        self._crop_on_change = None

        self._batch = 0          # >0 while several setters are coalesced into one render
        self._render_dirty = False
        self._img_item = None    # canvas id of the image, kept during an overlay-only drag
        self._render_pad = 0     # margin baked into the current frame (0 = canvas-sized)
        self._render_center = (0.0, 0.0)  # self.center at the last full render

        self._composite: Optional[dict] = None
        self._composite_edit: dict = {}
        self._edge_drag: Optional[int] = None
        self._window_drag: Optional[str] = None

        self.show_guides = False
        self.show_centroids = False
        self.show_lane_lines = False
        self._centroids: Optional[np.ndarray] = None
        self._lanes: Sequence[LaneLine] = ()

        self.canvas.bind("<Configure>", self._on_configure)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<ButtonPress-2>", self._on_middle_click)
        self.canvas.bind("<ButtonPress-3>", self._on_pan_press)
        self.canvas.bind("<B3-Motion>", self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_pan_release)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", lambda e: self.canvas.config(cursor=""))
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._zoom(1.1, e.x, e.y))
        self.canvas.bind("<Button-5>", lambda e: self._zoom(1 / 1.1, e.x, e.y))

        self._show_empty_state()

    # -- content ------------------------------------------------------------ #
    def set_image(self, display_u8, reset: bool = True) -> None:
        """Set the preprocessed uint8 display image (ndarray or PIL)."""
        if isinstance(display_u8, Image.Image):
            u8 = np.asarray(display_u8)
        else:
            u8 = np.asarray(display_u8)
        self._u8 = u8
        self._src_shape = (int(u8.shape[1]), int(u8.shape[0]))
        self._src_pil = Image.fromarray(u8.astype(np.uint8), mode="L")
        self.anchor = content_anchor(u8)
        self._crop = None
        self._view_ready = False
        if reset:
            self.reset_view()
        else:
            self._render()

    def set_rotation(self, angle: float) -> None:
        self.angle = float(angle)
        self._render()

    def set_overlays(
        self,
        centroids: Optional[np.ndarray] = None,
        lanes: Optional[Sequence[LaneLine]] = None,
        guides: Optional[bool] = None,
    ) -> None:
        self._centroids = None if centroids is None else np.asarray(centroids, float)
        self._lanes = tuple(lanes or ())
        if guides is not None:
            self.show_guides = bool(guides)
        self._render()

    def toggle_overlay(self, kind: str) -> None:
        if kind == "guides":
            self.show_guides = not self.show_guides
        elif kind == "centroids":
            self.show_centroids = not self.show_centroids
        elif kind == "lane_lines":
            self.show_lane_lines = not self.show_lane_lines
        else:
            raise ValueError(f"unknown overlay kind: {kind!r}")
        self._render()

    # -- crop --------------------------------------------------------------- #
    def bind_crop_change(self, callback) -> None:
        self._crop_on_change = callback

    def set_crop(self, rect) -> None:
        self._crop = None if rect is None else tuple(float(v) for v in rect[:4])
        self._render()

    def clear_crop(self) -> None:
        self._crop = None
        self._crop_drag = None
        self._render()

    @property
    def has_crop(self) -> bool:
        return self._crop is not None

    def crop_rect(self):
        return self._crop

    def _crop_screen_rect(self):
        return screen_rect(self._crop, self.center, self.scale)

    def _handle_radius(self) -> float:
        return max(4.0, 4.0 * self.ui_scale)

    def _hit_tolerance(self) -> float:
        return max(9.0, 9.0 * self.ui_scale)

    def _crop_handles(self):
        x0, y0, x1, y1 = self._crop_screen_rect()
        mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        return {
            "nw": (x0, y0), "n": (mx, y0), "ne": (x1, y0),
            "e": (x1, my), "se": (x1, y1), "s": (mx, y1),
            "sw": (x0, y1), "w": (x0, my),
        }

    def _hit_crop(self, sx: float, sy: float):
        """Which crop edge/corner is under the cursor (edges hit along their whole length)."""
        if self._crop is None:
            return None
        x0, y0, x1, y1 = self._crop_screen_rect()
        tol = self._hit_tolerance()
        near_l, near_r = abs(sx - x0) <= tol, abs(sx - x1) <= tol
        near_t, near_b = abs(sy - y0) <= tol, abs(sy - y1) <= tol
        within_x = (x0 - tol) <= sx <= (x1 + tol)
        within_y = (y0 - tol) <= sy <= (y1 + tol)
        if near_t and near_l:
            return "nw"
        if near_t and near_r:
            return "ne"
        if near_b and near_l:
            return "sw"
        if near_b and near_r:
            return "se"
        if near_t and within_x:
            return "n"
        if near_b and within_x:
            return "s"
        if near_l and within_y:
            return "w"
        if near_r and within_y:
            return "e"
        return None

    def _drag_crop(self, name: str, sx: float, sy: float) -> None:
        qx, qy = screen_to_display(sx, sy, self.center, self.scale)
        x0, y0, x1, y1 = self._crop
        if "n" in name:
            y0 = qy
        if "s" in name:
            y1 = qy
        if "w" in name:
            x0 = qx
        if "e" in name:
            x1 = qx
        self._crop = normalize_rect((x0, y0, x1, y1))
        # dragging only moves the border visually; the controller recalculates on release
        self._render(overlay_only=True)

    def _notify_crop_change(self) -> None:
        if self._crop_on_change is not None and self._crop is not None:
            self._crop_on_change(self._crop)

    # -- composite overlay (SPEC-COMPOSITE §12) ----------------------------- #
    def bind_composite_edit(self, lane=None, split=None, join=None, move=None, window=None) -> None:
        self._composite_edit = {"lane": lane, "split": split, "join": join,
                                "move": move, "window": window}

    def set_composite(self, data) -> None:
        if self._composite == data:
            return  # nothing changed (e.g. set_composite(None) on every switch)
        self._composite = data
        self._render()

    @contextmanager
    def batch(self) -> Iterator[None]:
        """Coalesce several setter calls into a single render.

        ``set_image``/``set_rotation``/``set_crop``/``set_overlays``/``set_composite``
        each render; wrapping a group in ``with viewer.batch():`` renders once at exit.
        """
        self._batch += 1
        try:
            yield
        finally:
            self._batch -= 1
            if self._batch == 0 and self._render_dirty:
                self._render_dirty = False
                self._render()

    @property
    def composite_on(self) -> bool:
        return self._composite is not None

    def _screen_to_q(self, sx: float, sy: float):
        return ((sx - self.center[0]) / self.scale, (sy - self.center[1]) / self.scale)

    def _composite_geom(self):
        d = self._composite
        if not d:
            return None
        y0, y1 = d["window"]
        return {
            "sy0": self.center[1] + self.scale * y0,
            "sy1": self.center[1] + self.scale * y1,
            "xs": [self.center[0] + self.scale * e for e in d["q_edges"]],
        }

    def _lane_at_qx(self, qx: float):
        edges = self._composite["q_edges"]
        if qx < edges[0] or qx >= edges[-1]:
            return None
        for i in range(len(edges) - 1):
            if edges[i] <= qx < edges[i + 1]:
                return i + 1
        return len(edges) - 1

    def _hit_edge(self, sx: float, sy: float):
        geo = self._composite_geom()
        if geo is None:
            return None
        tol = self._hit_tolerance()
        if not (geo["sy0"] - tol <= sy <= geo["sy1"] + tol):
            return None
        best, best_d = None, tol
        for i, x in enumerate(geo["xs"]):
            if abs(sx - x) <= best_d:
                best, best_d = i, abs(sx - x)
        return best

    def _hit_window_row(self, sy: float):
        """Which amber window row (if any) the cursor is on."""
        geo = self._composite_geom()
        if geo is None:
            return None
        tol = self._hit_tolerance()
        if abs(sy - geo["sy0"]) <= tol:
            return "top"
        if abs(sy - geo["sy1"]) <= tol:
            return "bottom"
        return None

    def _composite_tint(self, img: Image.Image, w: int, h: int, off: float = 0.0) -> Image.Image:
        d = self._composite
        geo = self._composite_geom()
        selected = set(d["order"])
        edges = d["q_edges"]
        overlay = Image.new("RGBA", (w, h), CROP_DIM)
        draw = ImageDraw.Draw(overlay)
        for i in range(1, len(edges)):
            if i in selected:
                x0 = self.center[0] + self.scale * edges[i - 1] + off
                x1 = self.center[0] + self.scale * edges[i] + off
                draw.rectangle((x0, geo["sy0"] + off, x1, geo["sy1"] + off), fill=(0, 0, 0, 0))
        return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    def _draw_composite(self, w: int, h: int) -> None:
        d = self._composite
        geo = self._composite_geom()
        sy0, sy1 = geo["sy0"], geo["sy1"]
        selected = set(d["order"])
        for sy in (sy0, sy1):
            self.canvas.create_line(0, sy, w, sy, fill="#ff9d00", width=2, dash=(6, 4))
            self.canvas.create_rectangle(7, sy - 4, 16, sy + 5, fill="#ff9d00", outline="#000000")
            self.canvas.create_rectangle(w - 16, sy - 4, w - 7, sy + 5,
                                         fill="#ff9d00", outline="#000000")
        for i, x in enumerate(geo["xs"]):
            lit = (i in selected) or (i + 1 in selected)
            if lit:
                self.canvas.create_line(x, sy0, x, sy1, fill="#00e5ff", width=2)
            else:
                self.canvas.create_line(x, sy0, x, sy1, fill="#00e5ff", width=1, dash=(2, 4))
            for hy in (sy0, sy1):
                if lit:
                    self.canvas.create_rectangle(x - 3, hy - 4, x + 4, hy + 5,
                                                 fill="#00e5ff", outline="#0a5a66")
                else:
                    self.canvas.create_rectangle(x - 3, hy - 4, x + 4, hy + 5,
                                                 fill="", outline="#00e5ff", dash=(2, 2))
        edges = d["q_edges"]
        num_y = max(12.0, sy0 - 22.0)
        for i in range(1, len(edges)):
            cx = self.center[0] + self.scale * (edges[i - 1] + edges[i]) / 2.0
            self.canvas.create_text(cx, num_y, text=str(i),
                                    fill="#00e5ff" if i in selected else "#8a8a8a")
            if i in selected:
                pos = d["order"].index(i) + 1
                self.canvas.create_text(cx, num_y + 15, text=f"#{pos}", fill="#ffd400")

    def reset_view(self) -> None:
        """Default view: fit the *whole* image to the canvas (no zoom-in).

        The previous default fit only the band-content window, which magnified
        the gel; instead we fit the full source frame and never upscale past 1x.
        """
        if self._src_pil is None:
            self._show_empty_state()
            return
        w, h = self._canvas_size()
        if w <= 2 or h <= 2:
            return
        sw, sh = self._src_shape
        x0, y0, x1, y1 = source_bbox_to_display_rect(
            (0.0, 0.0, float(sw), float(sh)), self.angle, self.anchor
        )
        bw = max(1.0, x1 - x0)
        bh = max(1.0, y1 - y0)
        self.scale = min(0.92 * w / bw, 0.92 * h / bh, 1.0)
        self.center = (
            w / 2.0 - self.scale * (x0 + x1) / 2.0,
            h / 2.0 - self.scale * (y0 + y1) / 2.0,
        )
        self._view_ready = True
        self._render()

    # -- rendering ---------------------------------------------------------- #
    def _canvas_size(self) -> Tuple[int, int]:
        return max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())

    def _show_empty_state(self) -> None:
        self.canvas.delete("all")
        w, h = self._canvas_size()
        self.canvas.create_text(
            w / 2, h / 2,
            text="Open an image folder to start\n(ctrl+O)",
            fill="#888888", font=("TkDefaultFont", 14), justify="center",
        )

    def _render(self, overlay_only: bool = False, pad: int = 0) -> None:
        if self._batch:
            self._render_dirty = True  # coalesced: draw once when the batch exits
            return
        if self._src_pil is None:
            self._show_empty_state()
            return
        if not self._view_ready:
            return
        w, h = self._canvas_size()
        if w <= 2 or h <= 2:
            return
        if overlay_only and self._img_item is not None:
            # dragging: keep the rendered image and only move the border items, so no
            # image recomposite / PhotoImage churn happens until the drag is released
            for item in self.canvas.find_all():
                if item != self._img_item:
                    self.canvas.delete(item)
        else:
            # ``pad`` renders a margin around the canvas so a pan can move the items
            W, H = w + 2 * pad, h + 2 * pad
            cx, cy = self.center
            img = render_display(self._src_pil, (W, H), self.angle, self.scale,
                                 self.anchor, (cx + pad, cy + pad))
            if self._composite is not None:
                img = self._composite_tint(img, W, H, pad)
            elif self._crop is not None:
                img = self._composite_crop(img, W, H, pad)
            self._imgtk = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self._img_item = self.canvas.create_image(-pad, -pad, anchor="nw", image=self._imgtk)
            self._render_pad = pad
            self._render_center = self.center

        if self.show_guides:
            for gx in range(self.guide_spacing, w, self.guide_spacing):
                self.canvas.create_line(gx, 0, gx, h, fill=GUIDE_COLOR, dash=(5, 5), width=1)


        if self.show_centroids and self._centroids is not None and len(self._centroids):
            for x, y in self._centroids:
                sx, sy = source_to_screen((float(x), float(y)), self.angle, self.scale,
                                          self.anchor, self.center)
                r = max(2.0, 3.0 * self.ui_scale)
                self.canvas.create_oval(sx - r, sy - r, sx + r, sy + r,
                                        outline=CENTROID_COLOR, width=2)

        if self.show_lane_lines and self._lanes:
            height = self._src_shape[1]
            for lane in self._lanes:
                y_a, y_b = 0.0, float(max(1, height - 1))
                p_a = source_to_screen((lane.x_at(y_a), y_a), self.angle, self.scale,
                                       self.anchor, self.center)
                p_b = source_to_screen((lane.x_at(y_b), y_b), self.angle, self.scale,
                                       self.anchor, self.center)
                self.canvas.create_line(p_a[0], p_a[1], p_b[0], p_b[1],
                                        fill=LANE_COLOR, width=2)

        if self._composite is not None:
            self._draw_composite(w, h)
        elif self._crop is not None:
            self._draw_crop()

    def _composite_crop(self, img: Image.Image, w: int, h: int, off: float = 0.0) -> Image.Image:
        x0, y0, x1, y1 = self._crop_screen_rect()
        x0 = max(0.0, min(float(x0) + off, w))
        x1 = max(0.0, min(float(x1) + off, w))
        y0 = max(0.0, min(float(y0) + off, h))
        y1 = max(0.0, min(float(y1) + off, h))
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for bx0, by0, bx1, by1 in (
            (0, 0, w, y0), (0, y1, w, h), (0, y0, x0, y1), (x1, y0, w, y1),
        ):
            if bx1 > bx0 and by1 > by0:
                draw.rectangle((bx0, by0, bx1, by1), fill=CROP_DIM)
        return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    def _draw_crop(self) -> None:
        x0, y0, x1, y1 = self._crop_screen_rect()
        self.canvas.create_rectangle(x0, y0, x1, y1, outline=CROP_COLOR, width=2)
        r = self._handle_radius()
        for hx, hy in self._crop_handles().values():
            self.canvas.create_rectangle(hx - r, hy - r, hx + r, hy + r,
                                         fill=CROP_COLOR, outline="#000000")

    # -- events ------------------------------------------------------------- #
    def _zoom(self, factor: float, sx: float, sy: float) -> None:
        if self._src_pil is None:
            return
        px, py = screen_to_source((sx, sy), self.angle, self.scale, self.anchor, self.center)
        new_scale = min(40.0, max(0.02, self.scale * factor))
        cx, cy = source_to_screen((px, py), self.angle, new_scale, self.anchor, self.center)
        self.scale = new_scale
        self.center = (self.center[0] + (sx - cx), self.center[1] + (sy - cy))
        self._render()

    def _on_configure(self, _event: tk.Event) -> None:
        if self._src_pil is None:
            self._show_empty_state()
        elif not self._view_ready:
            self.reset_view()
        else:
            self._render()

    def _on_press(self, event: tk.Event) -> None:
        if self._src_pil is None:
            return
        if self._composite is not None:
            edge = self._hit_edge(event.x, event.y)
            if edge is not None:
                self._edge_drag = edge
                return
            side = self._hit_window_row(event.y)
            if side is not None:
                self._window_drag = side
                return
            if self._within_window(event.y):  # only lanes inside the kept strip
                lane = self._lane_at_qx(self._screen_to_q(event.x, event.y)[0])
                cb = self._composite_edit.get("lane")
                if lane is not None and cb is not None:
                    cb(lane)
            return
        if self._crop is not None:
            handle = self._hit_crop(event.x, event.y)
            if handle is not None:
                self._crop_drag = handle
                return
        # Left-drag on empty space starts a new crop rectangle.
        qx, qy = screen_to_display(event.x, event.y, self.center, self.scale)
        self._crop_prev = self._crop
        self._crop_new = (qx, qy)
        self._crop_new_screen = (event.x, event.y)
        self._crop = (qx, qy, qx, qy)
        self._render()

    def _on_drag(self, event: tk.Event) -> None:
        # During a drag the borders move visually only; the controller recalculates on
        # release, so no re-identification / masks / exports happen per motion event.
        if self._window_drag is not None:
            qy = self._screen_to_q(event.x, event.y)[1]
            y0, y1 = self._composite["window"]
            if self._window_drag == "top":
                y0 = qy
            else:
                y1 = qy
            self._composite["window"] = (y0, y1)
            self._render(overlay_only=True)
            return
        if self._edge_drag is not None:
            qx, _ = self._screen_to_q(event.x, event.y)
            self._composite["q_edges"][self._edge_drag] = qx
            self._render(overlay_only=True)
            return
        if self._crop_drag is not None:
            self._drag_crop(self._crop_drag, event.x, event.y)
            return
        if self._crop_new is not None:
            ax, ay = self._crop_new
            qx, qy = screen_to_display(event.x, event.y, self.center, self.scale)
            self._crop = normalize_rect((ax, ay, qx, qy))
            self._render(overlay_only=True)

    def _on_release(self, event: tk.Event) -> None:
        if self._window_drag is not None:
            side = self._window_drag
            self._window_drag = None
            cb = self._composite_edit.get("window")
            if cb is not None:  # commit once, on release
                cb(side, self._composite["window"][0 if side == "top" else 1])
            return
        if self._edge_drag is not None:
            index = self._edge_drag
            self._edge_drag = None
            cb = self._composite_edit.get("move")
            if cb is not None:
                cb(index, self._composite["q_edges"][index] + self._composite["kx"])
            return
        if self._crop_drag is not None:
            self._crop_drag = None
            self._notify_crop_change()  # commit once, on release
            self._render()
            return
        if self._crop_new is not None:
            start = self._crop_new_screen or (event.x, event.y)
            moved = abs(event.x - start[0]) + abs(event.y - start[1])
            if moved < _MIN_NEW_CROP_PX:
                self._crop = self._crop_prev  # ignore an accidental click
            else:
                self._notify_crop_change()
            self._crop_new = None
            self._crop_new_screen = None
            self._crop_prev = None
            self._render()

    def _pan_margin(self) -> int:
        w, h = self._canvas_size()
        return int(_PAN_MARGIN_FRAC * min(w, h))

    def _on_pan_press(self, event: tk.Event) -> None:
        self._pan_from = (event.x, event.y)

    def _on_pan_drag(self, event: tk.Event) -> None:
        if self._pan_from is None:
            return
        dx = event.x - self._pan_from[0]
        dy = event.y - self._pan_from[1]
        new = (self.center[0] + dx, self.center[1] + dy)
        ox, oy = self._render_center
        margin = self._render_pad
        within = (margin > 0 and self._img_item is not None
                  and abs(new[0] - ox) <= margin and abs(new[1] - oy) <= margin)
        self.center = new
        self._pan_from = (event.x, event.y)
        if within:
            # slide the whole frame (image + overlays); no re-render, so it stays smooth
            self.canvas.move("all", dx, dy)
        else:
            # first motion, or the pan outran the rendered margin: re-render with margin
            self._render(pad=self._pan_margin())

    def _on_pan_release(self, _event: tk.Event) -> None:
        self._pan_from = None
        if self._render_pad > 0:  # re-center once the pan finished (canvas-sized frame)
            self._render()

    def _within_window(self, sy: float) -> bool:
        geo = self._composite_geom()
        return geo is not None and geo["sy0"] <= sy <= geo["sy1"]

    _CURSOR_H = "sb_h_double_arrow"   # vertical bound (lane edge)
    _CURSOR_V = "sb_v_double_arrow"   # horizontal bound (window row / crop edge)
    _CURSOR_D = "size_nw_se"

    def _cursor_for(self, x: float, y: float) -> str:
        if self._composite is not None:
            if self._hit_edge(x, y) is not None:
                return self._CURSOR_H
            if self._hit_window_row(y) is not None:
                return self._CURSOR_V
            return ""
        if self._crop is not None:
            handle = self._hit_crop(x, y)
            if handle in ("w", "e"):
                return self._CURSOR_H
            if handle in ("n", "s"):
                return self._CURSOR_V
            if handle is not None:
                return self._CURSOR_D
        return ""

    def _on_motion(self, event: tk.Event) -> None:
        if self._src_pil is None:
            return
        cursor = self._cursor_for(event.x, event.y)
        if self.canvas.cget("cursor") != cursor:
            self.canvas.config(cursor=cursor)

    def _on_middle_click(self, event: tk.Event) -> None:
        if self._composite is None:
            return
        edge = self._hit_edge(event.x, event.y)
        if edge is not None:
            cb = self._composite_edit.get("join")
            if cb is not None:
                cb(edge)
            return
        if self._within_window(event.y):
            qx, _ = self._screen_to_q(event.x, event.y)
            cb = self._composite_edit.get("split")
            if cb is not None:
                cb(qx + self._composite["kx"])

    def _on_wheel(self, event: tk.Event) -> None:
        self._zoom(1.1 if event.delta > 0 else 1 / 1.1, event.x, event.y)