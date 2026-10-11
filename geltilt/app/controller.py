"""Application controller: the single state mutator, owned by the main thread."""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, List, Optional

import tkinter as tk
from tkinter import messagebox

from ..banddetect import band_mask, bbox_from_mask, default_segmenter
from ..labeler import content_anchor, find_images
from . import export, i18n, lanes
from .config import AppConfig, ensure_run_directory
from .crop import display_rect_to_corrected_box, pad_rect, rotate_mask, source_bbox_to_display_rect
from .display import DisplayPreprocessor
from .domain import (
    AppliedRotation,
    CropRect,
    DiagnosticLogEntry,
    EstimateOutcome,
    make_applied_rotation,
)

CROP_PAD = 10
from .estimator import CanonicalEstimator
from .log import JsonlDiagnosticLog
from .service import EstimationService, EstimateResult
from .session import JsonSessionStore, make_session
from .state import SampleStateStore

# Lane-spec keys are rendered through the i18n catalog (SPEC-COMPOSITE §7.2, §13).
def _spec_message(exc) -> str:
    return i18n.tr(exc.key, **exc.kwargs)


def _still_cutting(edges, blobs) -> int:
    return sum(1 for x in edges[1:-1] if any(b["x0"] < x < b["x1"] for b in blobs))


class AppController:
    """Owns application state and orchestrates the widgets and services."""

    def __init__(
        self,
        root: tk.Misc,
        config: AppConfig,
        estimator=None,
        session_store=None,
        diag_log=None,
        preprocessor: Optional[DisplayPreprocessor] = None,
        extract_bands=None,
    ) -> None:
        self.root = root
        self.config = config
        ensure_run_directory(config.run_dir)
        self.estimator = estimator or CanonicalEstimator(
            config.model_path, config.threshold, config.min_area
        )
        self._extract_bands = extract_bands or self._default_extract_bands
        self.service = EstimationService(self.estimator)
        self.session_store = session_store or JsonSessionStore(config.run_dir)
        self.diag_log = diag_log or JsonlDiagnosticLog(config.run_dir)
        self.display = preprocessor or DisplayPreprocessor()
        self.state = SampleStateStore()

        self.folder: Optional[Path] = None
        self.samples: List[Path] = []
        self.index = 0
        self.rotation_epoch = 0

        self.browser = None
        self.viewer = None
        self.manual = None
        self.panel = None
        self._status_cb: Optional[Callable[[str], None]] = None
        self._busy_cb: Optional[Callable[[bool], None]] = None
        self._error_cb: Optional[Callable[[str, str], None]] = None
        self._warn_cb: Optional[Callable[[str, str], None]] = None
        self._crop_cb: Optional[Callable[[bool, bool], None]] = None
        self._composite_cb: Optional[Callable[[], None]] = None

        # Composite-crop state (single active image).
        self.composite_on = False
        self.composite_text = ""
        self.composite_groups: Optional[List[List[int]]] = None
        self.layout: Optional[dict] = None

        self._polling = False
        self._poll_interval = 60

    # -- wiring ------------------------------------------------------------- #
    def attach(self, browser, viewer, manual, panel, status_cb=None, busy_cb=None,
               error_cb=None, warn_cb=None, crop_cb=None, composite_cb=None) -> None:
        self.browser = browser
        self.viewer = viewer
        self.manual = manual
        self.panel = panel
        self._status_cb = status_cb
        self._busy_cb = busy_cb
        self._error_cb = error_cb
        self._warn_cb = warn_cb
        self._crop_cb = crop_cb
        self._composite_cb = composite_cb
        if browser is not None:
            browser.bind_select(self.goto)
        if manual is not None:
            manual.bind_change(self.set_manual_angle)
        if viewer is not None and hasattr(viewer, "bind_crop_change"):
            viewer.bind_crop_change(self.set_crop)
        if viewer is not None and hasattr(viewer, "bind_composite_edit"):
            viewer.bind_composite_edit(
                lane=self.click_lane, split=self.split_lane_at,
                join=self.join_edge, move=self.move_edge, window=self.set_window,
            )

    # -- callbacks ---------------------------------------------------------- #
    def _status(self, text: str) -> None:
        if self._status_cb is not None:
            self._status_cb(text)

    def _busy(self, value: bool) -> None:
        if self._busy_cb is not None:
            self._busy_cb(value)

    def _error(self, title: str, message: str) -> None:
        if self._error_cb is not None:
            self._error_cb(title, message)
        else:
            messagebox.showerror(title, message)

    def _warn(self, title: str, message: str) -> None:
        if self._warn_cb is not None:
            self._warn_cb(title, message)
        else:
            messagebox.showwarning(title, message)

    # -- launch / restore --------------------------------------------------- #
    def start(self, initial_folder: Optional[Path] = None) -> None:
        session = self.session_store.load()
        self.state = SampleStateStore.from_session(session)
        folder: Optional[Path] = None
        if initial_folder is not None:
            folder = Path(initial_folder)
        elif session.folder:
            candidate = Path(session.folder)
            if candidate.is_dir():
                folder = candidate
        if folder is not None:
            self.open_folder(folder)
            if session.folder and str(folder) == session.folder and self.samples:
                self.goto(min(max(session.index, 0), len(self.samples) - 1))
        else:
            if self.browser is not None:
                self.browser.set_samples([])
            if self.panel is not None:
                self.panel.show_empty()
            self._notify_crop()

    # -- folder / navigation ------------------------------------------------ #
    def open_folder(self, path) -> bool:
        path = Path(path)
        if not path.is_dir():
            self._error(i18n.tr("dlg.open_folder"), i18n.tr("msg.not_a_folder", path=path))
            return False
        images = find_images(path)
        if not images:
            exts = ", ".join(sorted(self.config.image_exts))
            self._warn(i18n.tr("dlg.open_folder"),
                       i18n.tr("msg.no_images", path=path, exts=exts))
            return False
        session = self.session_store.load()
        self.state = SampleStateStore.from_session(session)
        self.folder = path
        self.samples = images
        self.index = 0
        if self.browser is not None:
            self.browser.set_samples([p.name for p in images])
        self._activate()
        self.save_session()
        self._status(i18n.tr("status.samples_in", count=len(images), name=path.name))
        return True

    def goto(self, index: int) -> None:
        if not self.samples:
            return
        self.index = max(0, min(int(index), len(self.samples) - 1))
        self._activate()
        self.save_session()

    def next_sample(self) -> None:
        if self.samples and self.index < len(self.samples) - 1:
            self.goto(self.index + 1)

    def prev_sample(self) -> None:
        if self.samples and self.index > 0:
            self.goto(self.index - 1)

    # -- activation --------------------------------------------------------- #
    def _active_name(self) -> Optional[str]:
        if not self.samples:
            return None
        return self.samples[self.index].name

    def _activate(self) -> None:
        if not self.samples:
            return
        path = self.samples[self.index]
        name = path.name
        st = self.state.get_or_create(name)
        try:
            u8 = self.display.display_u8(path)
        except Exception as exc:  # noqa: BLE001 - isolate to the sample
            self.state.mark_failed(name, True)
            if self.panel is not None:
                self.panel.show_state(name, st.applied, None, failed=True)
            self._error(i18n.tr("dlg.load_error"), i18n.tr("msg.could_not_load", name=name, exc=exc))
            self._notify_crop()
            self._notify_composite()
            return
        # Coalesce every viewer mutation below into a single render (fix for switch latency).
        batch = self.viewer.batch() if (self.viewer is not None
                                        and hasattr(self.viewer, "batch")) else nullcontext()
        with batch:
            if self.viewer is not None:
                self.viewer.set_image(u8, reset=True)
                self.viewer.set_rotation(st.applied.value)
                if st.crop is not None:
                    self.viewer.set_crop(st.crop.as_tuple())
                else:
                    self.viewer.clear_crop()
                self._apply_overlays(st.last_estimate)
            if self.manual is not None:
                self.manual.set_value(st.applied.value)
            if self.panel is not None:
                self.panel.show_state(name, st.applied, st.last_estimate, st.failed)
            if self.browser is not None:
                self.browser.set_active(self.index)
            self._notify_crop()
            # Recompute lanes for the new sample; a ticked box with no lanes behind it untick (R3).
            if self.composite_on:
                if not self.reidentify_lanes(announce=False) or not self.composite_ready():
                    self.composite_on = False
                    self.composite_groups = None
            self._notify_composite()

    def _apply_overlays(self, estimate: Optional[EstimateOutcome]) -> None:
        if self.viewer is None:
            return
        if estimate is None:
            self.viewer.set_overlays(None, ())
        else:
            self.viewer.set_overlays(estimate.centroids, estimate.accepted_lanes)

    # -- auto rotation ------------------------------------------------------ #
    def trigger_auto_rotate(self) -> None:
        name = self._active_name()
        if name is None:
            return
        path = self.samples[self.index]
        try:
            arr = self.display.source_array(path)
        except Exception as exc:  # noqa: BLE001
            self._error(i18n.tr("dlg.auto_rotate"),
                        i18n.tr("msg.could_not_load", name=name, exc=exc))
            return
        self.rotation_epoch += 1
        epoch = self.rotation_epoch
        self._busy(True)
        self._status(i18n.tr("status.estimating_named", name=name))
        self.service.start(name, arr, epoch)
        self._start_polling()

    def _start_polling(self) -> None:
        if self._polling:
            return
        self._polling = True
        self.root.after(self._poll_interval, self._poll)

    def _poll(self) -> None:
        result = self.service.poll()
        if result is not None:
            self._handle_result(result)
        if self.service.busy:
            self.root.after(self._poll_interval, self._poll)
        else:
            self._polling = False
            self._busy(False)

    def _handle_result(self, result: EstimateResult) -> None:
        name = result.sample
        if result.error is not None:
            entry = DiagnosticLogEntry(
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), sample=name,
                angle=float("nan"), se=float("nan"), ok=False,
                reason=result.error, warnings=(),
            )
            self.diag_log.append(entry)
            self._error(i18n.tr("dlg.estimate_failed"), f"{name}:\n{result.error}")
            self._status(i18n.tr("status.estimate_failed"))
            return

        outcome = result.outcome
        st = self.state.set_estimate(name, outcome)
        self._append_log(name, outcome)

        active = self._active_name()
        stale = name != active or result.epoch != self.rotation_epoch
        if stale:
            if name == active and self.panel is not None:
                self.panel.show_state(name, st.applied, outcome, st.failed)
            return

        if outcome.accepted:
            applied = AppliedRotation(outcome.angle, "auto")
            self.state.set_applied(name, applied)
            self.rotation_epoch += 1
            if self.viewer is not None:
                self.viewer.set_rotation(applied.value)
            if self.manual is not None:
                self.manual.set_value(applied.value)
        self._apply_overlays(outcome)
        if self.panel is not None:
            self.panel.show_state(name, self.state.get_or_create(name).applied, outcome, st.failed)
        self._status(
            i18n.tr("status.applied", angle=f"{outcome.angle:+.1f}")
            if outcome.accepted else i18n.tr("status.estimate_refused")
        )
        self._notify_crop()
        self.save_session()

    def _append_log(self, name: str, outcome: EstimateOutcome) -> None:
        self.diag_log.append(
            DiagnosticLogEntry(
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                sample=name,
                angle=outcome.angle,
                se=outcome.se,
                ok=bool(outcome.ok),
                reason=outcome.reason,
                warnings=outcome.warnings,
            )
        )

    # -- manual rotation ---------------------------------------------------- #
    def set_manual_angle(self, value: float) -> None:
        name = self._active_name()
        if name is None:
            return
        applied = make_applied_rotation(value, "manual")
        self.state.set_applied(name, applied)
        self.rotation_epoch += 1
        if self.viewer is not None:
            self.viewer.set_rotation(applied.value)
        if self.manual is not None:
            self.manual.set_value(applied.value)
        st = self.state.get_or_create(name)
        if self.panel is not None:
            self.panel.show_state(name, applied, st.last_estimate, st.failed)
        self._notify_crop()
        self.save_session()

    def nudge(self, delta: float) -> None:
        name = self._active_name()
        if name is None:
            return
        current = self.state.get_or_create(name).applied.value
        self.set_manual_angle(current + float(delta))

    def zero_angle(self) -> None:
        name = self._active_name()
        if name is None:
            return
        applied = AppliedRotation(0.0, "none")
        self.state.set_applied(name, applied)
        self.rotation_epoch += 1
        if self.viewer is not None:
            self.viewer.set_rotation(0.0)
        if self.manual is not None:
            self.manual.set_value(0.0)
        st = self.state.get_or_create(name)
        if self.panel is not None:
            self.panel.show_state(name, applied, st.last_estimate, st.failed)
        self._notify_crop()
        self.save_session()

    # -- view / overlays ---------------------------------------------------- #
    def toggle_overlay(self, kind: str) -> None:
        if self.viewer is not None:
            self.viewer.toggle_overlay(kind)

    def reset_view(self) -> None:
        if self.viewer is not None:
            self.viewer.reset_view()

    # -- auto crop ---------------------------------------------------------- #
    def _default_extract_bands(self, image):
        """(band mask, union bbox) from a single GelGenie inference."""
        segmenter = default_segmenter(self.config.model_path)
        mask = band_mask(image, segmenter=segmenter, threshold=self.config.threshold)
        return mask, bbox_from_mask(mask, min_area=self.config.min_area)

    def _crop_available(self, st) -> bool:
        """Auto crop is unlocked only once auto-rotate has succeeded."""
        return st.applied.source == "auto" or (
            st.last_estimate is not None and st.last_estimate.accepted
        )

    def can_auto_crop(self) -> bool:
        name = self._active_name()
        if name is None:
            return False
        return self._crop_available(self.state.get_or_create(name))

    def _notify_crop(self) -> None:
        if self._crop_cb is None:
            return
        name = self._active_name()
        has_crop = bool(name and self.state.get_or_create(name).crop is not None)
        self._crop_cb(self.can_auto_crop(), has_crop)

    def auto_crop(self) -> None:
        """Box the GelGenie bands (10px pad) in the corrected frame."""
        name = self._active_name()
        if name is None:
            return
        st = self.state.get_or_create(name)
        if not self._crop_available(st):
            self._warn(i18n.tr("dlg.auto_crop"), i18n.tr("msg.run_auto_rotate_first"))
            return
        path = self.samples[self.index]
        try:
            arr = self.display.source_array(path)
            u8 = self.display.display_u8(path)
        except Exception as exc:  # noqa: BLE001
            self._error(i18n.tr("dlg.auto_crop"),
                        i18n.tr("msg.could_not_load", name=name, exc=exc))
            return
        try:
            mask, bbox = self._extract_bands(arr)
        except Exception as exc:  # noqa: BLE001
            self._error(i18n.tr("dlg.auto_crop"),
                        i18n.tr("msg.band_detection_failed", name=name, exc=exc))
            return
        st.band_mask = mask  # cached for composite lane re-identification (T-7)
        if bbox is None:
            self._warn(i18n.tr("dlg.auto_crop"), i18n.tr("msg.no_bands", name=name))
            return
        anchor = content_anchor(u8)
        rect = source_bbox_to_display_rect(
            (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
            st.applied.value, anchor,
        )
        rect = pad_rect(rect, CROP_PAD)
        st.crop = CropRect(rect[0], rect[1], rect[2], rect[3], angle=st.applied.value)
        if self.viewer is not None:
            self.viewer.set_crop(rect)
        self._notify_crop()
        self.reidentify_lanes(announce=False)
        self._status(i18n.tr("status.auto_crop_size",
                             w=f"{rect[2] - rect[0]:.0f}", h=f"{rect[3] - rect[1]:.0f}"))

    def clear_crop(self) -> None:
        name = self._active_name()
        if name is None:
            return
        st = self.state.get_or_create(name)
        st.crop = None
        if self.viewer is not None:
            self.viewer.clear_crop()
        self._on_crop_changed()
        self._notify_crop()
        self._status(i18n.tr("status.crop_cleared"))

    def set_crop(self, rect) -> None:
        """Store a crop rect dragged in the viewer (keeps the locked angle)."""
        name = self._active_name()
        if name is None:
            return
        st = self.state.get_or_create(name)
        angle = st.crop.angle if st.crop is not None else st.applied.value
        st.crop = CropRect(rect[0], rect[1], rect[2], rect[3], angle=angle)
        self._on_crop_changed()
        self._notify_crop()

    def _on_crop_changed(self) -> None:
        """Re-identify lanes inside the crop when the box changes (§11)."""
        if not self.composite_on:
            return
        if not self.reidentify_lanes(announce=False):
            return
        if not self.composite_ready():
            self.composite_on = False  # R3: a ticked box with no lanes behind it
            self.composite_groups = None
            self._status(i18n.tr("status.composite_off_few_lanes"))

    # -- composite crop (SPEC-COMPOSITE) ------------------------------------ #
    def _active_state(self):
        name = self._active_name()
        return None if name is None else self.state.get_or_create(name)

    def _lane_frame(self):
        """(corrected-frame band mask, crop region box), or (None, None)."""
        st = self._active_state()
        path = self.active_sample()
        if st is None or path is None or st.band_mask is None or st.crop is None:
            return None, None
        angle = st.applied.value
        mask = rotate_mask(st.band_mask, angle)
        h, w = mask.shape[:2]
        anchor = content_anchor(self.display.display_u8(path))
        box = display_rect_to_corrected_box(
            st.crop.as_tuple(), st.crop.angle, anchor, angle, (h, w)
        )
        if box is None:
            box = (0, 0, w, h)
        return mask, box

    def composite_ready(self) -> bool:
        return bool(self.layout) and int(self.layout.get("count", 0)) >= 2

    def composite_mode(self) -> bool:
        return self.composite_on and self.composite_ready()

    def composite_active(self) -> bool:
        return self.composite_mode() and self.composite_groups is not None

    def composite_selectable(self) -> bool:
        """Ticking can succeed only once Auto crop has cached a band mask."""
        st = self._active_state()
        return st is not None and st.band_mask is not None

    def reidentify_lanes(self, announce: bool = True) -> bool:
        mask, box = self._lane_frame()
        if mask is None:
            if announce:
                self._status(i18n.tr("status.auto_crop_first"))
            return False
        self.layout = lanes.lane_layout(mask, box)
        self._resync_spec()
        if announce:
            self._status(i18n.tr("status.lanes_reidentified",
                                 count=self.layout['count']))
        self._notify_composite()
        return True

    def _resync_spec(self) -> None:
        """Keep the request only if it still parses against the new live count (§11)."""
        count = int(self.layout["count"]) if self.layout else 0
        if self.composite_text.strip() and count >= 2:
            try:
                self.composite_groups = lanes.parse_lane_spec(self.composite_text, count)
                return
            except lanes.LaneSpecError:
                pass
        self.composite_groups = None

    def set_composite(self, on: bool) -> bool:
        if on:
            ok = self.composite_selectable() and self.reidentify_lanes(announce=False)
            if not (ok and self.composite_ready()):
                self.composite_on = False
                self.composite_groups = None
                self._status(i18n.tr("status.auto_crop_first"))
                self._notify_composite()
                return False
            self.composite_on = True
            self.reidentify_lanes(announce=False)  # R7: commit inside the crop as it stands
            self._sync_crop_to_composite()  # crop bbox follows the lane region
        else:
            self.composite_on = False
            self.composite_groups = None
            self._status(i18n.tr("status.composite_off_crop_applies"))
        self._notify_composite()
        return self.composite_on

    def set_composite_spec(self, text: str) -> None:
        self.composite_text = text or ""
        if not self.composite_mode():
            self.composite_groups = None
            self._notify_composite()
            return
        try:
            self.composite_groups = lanes.parse_lane_spec(self.composite_text, int(self.layout["count"]))
            order = lanes.spec_groups(self.composite_groups)
            self._status(i18n.tr("status.composite_spec",
                                 order=",".join(str(n) for n in order), count=len(order)))
        except lanes.LaneSpecError as exc:
            self.composite_groups = None
            self._status(_spec_message(exc))
        self._notify_composite()

    def even_widths(self) -> None:
        if not self.composite_mode() or not self.layout:
            return
        edges = self.layout["edges"]
        # the 4px-widened mask feeds nudging only, so a boundary clears the visible band
        blobs = self.layout.get("nudge_blobs") or self.layout["blobs"]
        evened = lanes.even_edges(edges)
        new_edges, nudged = lanes.nudge_off_bands(evened, blobs, lanes.LANE_MIN_PX)
        changed_even = evened != edges
        self.layout["edges"] = new_edges
        self.layout["spans"] = lanes.spans_from_edges(new_edges)
        still = _still_cutting(new_edges, blobs)
        if not changed_even and nudged == 0:
            status = i18n.tr("status.lanes_already_even")
        elif changed_even and nudged == 0:
            status = i18n.tr("status.lanes_evened")
        elif not changed_even and nudged:
            status = i18n.tr("status.lanes_nudged")
        else:
            status = i18n.tr("status.lanes_evened_nudged")
        if still:
            status += " " + i18n.tr("status.boundaries_still_cut", count=still)
        self._status(status)
        self._notify_composite()

    def click_lane(self, lane: int) -> None:
        if not self.composite_mode():
            return
        order = lanes.spec_groups(self.composite_groups) if self.composite_groups else []
        if lane in order:
            order.remove(lane)
            verb = i18n.tr("status.lane_verb_removed")
        else:
            order.append(lane)
            verb = i18n.tr("status.lane_verb_added")
        self._set_order(order)
        shown = ",".join(str(n) for n in order) if order else i18n.tr("status.none_placeholder")
        self._status(i18n.tr("status.lane_click", lane=lane, verb=verb, shown=shown))

    def _set_order(self, order) -> None:
        deduped = list(dict.fromkeys(int(n) for n in order))
        self.composite_text = ",".join(str(n) for n in deduped)
        self.composite_groups = [[n] for n in deduped]
        self._notify_composite()

    def split_lane_at(self, x: float) -> None:
        if not self.composite_mode() or not self.layout:
            return
        edges, index = lanes.insert_edge(self.layout["edges"], x, lanes.LANE_MIN_PX)
        if index is None:
            self._status(i18n.tr("status.no_border_added"))
            return
        self._apply_edges(edges)
        order = lanes.spec_groups(self.composite_groups) if self.composite_groups else []
        self._set_order(lanes.renumber_after_split(order, index))
        self._status(i18n.tr("status.border_added", count=self.layout['count']))

    def join_edge(self, index: int) -> None:
        if not self.composite_mode() or not self.layout:
            return
        new_edges, removed = lanes.remove_edge(self.layout["edges"], index, lanes.LANE_MIN_PX)
        if removed is None:
            self._status(i18n.tr("status.border_stays"))
            return
        self._apply_edges(new_edges)
        order = lanes.spec_groups(self.composite_groups) if self.composite_groups else []
        self._set_order(lanes.renumber_after_merge(order, removed))
        self._status(i18n.tr("status.border_removed", count=self.layout['count']))

    def move_edge(self, index: int, x: float, width: Optional[int] = None) -> None:
        if not self.composite_mode() or not self.layout:
            return
        if width is None:
            # outer boundaries may go anywhere in the frame, not just the simple crop
            mask, _box = self._lane_frame()
            width = int(mask.shape[1]) if mask is not None else None
        self._apply_edges(lanes.move_edge(self.layout["edges"], index, x, width, lanes.LANE_MIN_PX))
        self._sync_crop_to_composite()

    def set_window(self, side: str, qy: float) -> None:
        """Drag a window row (composite mode): move the crop's top/bottom edge."""
        st = self._active_state()
        if st is None or st.crop is None or not self.composite_mode():
            return
        x0, y0, x1, y1 = st.crop.as_tuple()
        if side == "top":
            y0 = min(float(qy), y1 - lanes.LANE_MIN_PX)
        else:
            y1 = max(float(qy), y0 + lanes.LANE_MIN_PX)
        st.crop = CropRect(x0, y0, x1, y1, angle=st.crop.angle)
        self._sync_crop_to_composite()
        self._notify_composite()

    def _frame_kx(self) -> float:
        """Display-frame -> corrected-frame x offset (a constant translation)."""
        st = self._active_state()
        path = self.active_sample()
        if st is None or st.crop is None or path is None or st.band_mask is None:
            return 0.0
        h, w = st.band_mask.shape[:2]
        anchor = content_anchor(self.display.display_u8(path))
        corr = display_rect_to_corrected_box(
            st.crop.as_tuple(), st.crop.angle, anchor, st.applied.value, (h, w)
        )
        return 0.0 if corr is None else float(corr[0] - st.crop.x0)

    def _sync_crop_to_composite(self) -> None:
        """Resize the simple crop to the composite's bbox (outer edges x window)."""
        st = self._active_state()
        if st is None or st.crop is None or not self.layout or not self.layout.get("edges"):
            return
        kx = self._frame_kx()
        x0 = float(self.layout["edges"][0]) - kx
        x1 = float(self.layout["edges"][-1]) - kx
        st.crop = CropRect(x0, st.crop.y0, x1, st.crop.y1, angle=st.crop.angle)
        if self.viewer is not None and hasattr(self.viewer, "set_crop"):
            self.viewer.set_crop(st.crop.as_tuple())

    def _apply_edges(self, edges) -> None:
        self.layout["edges"] = list(edges)
        self.layout["spans"] = lanes.spans_from_edges(edges)
        self.layout["count"] = len(edges) - 1
        self._notify_composite()

    def _notify_composite(self) -> None:
        if self.viewer is not None and hasattr(self.viewer, "set_composite"):
            self.viewer.set_composite(self._composite_overlay())
        if self._composite_cb is not None:
            self._composite_cb()

    def _composite_overlay(self) -> Optional[dict]:
        """Lane spans and the window in the *display* frame, for the viewer."""
        if not self.composite_mode() or not self.layout:
            return None
        st = self._active_state()
        if st is None or st.crop is None:
            return None
        angle = st.applied.value
        path = self.active_sample()
        anchor = content_anchor(self.display.display_u8(path))
        mask_h = int(st.band_mask.shape[0])
        mask_w = int(st.band_mask.shape[1])
        corr = display_rect_to_corrected_box(
            st.crop.as_tuple(), st.crop.angle, anchor, angle, (mask_h, mask_w)
        )
        qx = list(st.crop.as_tuple())
        kx = (corr[0] - qx[0]) if corr is not None else 0.0
        order = lanes.spec_groups(self.composite_groups) if self.composite_groups else []
        q_edges = [float(e) - kx for e in self.layout["edges"]]
        return {
            "q_edges": q_edges,
            "spans": lanes.spans_from_edges(self.layout["edges"]),
            "kx": kx,
            "count": int(self.layout["count"]),
            "order": order,
            "window": (float(qx[1]), float(qx[3])),
        }

    # -- export ------------------------------------------------------------- #
    def active_sample(self) -> Optional[Path]:
        if not self.samples:
            return None
        return self.samples[self.index]

    def export_array(self, use_original: bool, apply_crop: bool = True):
        """The exported array for the active sample (or ``None``).

        ``use_original`` picks the raw source pixels (no brightness processing)
        over the brightness-processed display image. ``apply_crop`` cuts the box
        drawn on the viewer (:attr:`SampleState.crop`), clamped to the frame.
        """
        path = self.active_sample()
        if path is None:
            return None
        st = self.state.get_or_create(path.name)
        src = self.display.source_array(path) if use_original else self.display.display_u8(path)
        corrected = export.rotated_export(src, st.applied.value)
        if self.composite_active():
            if not apply_crop:
                return None  # §10: "Rotation only" + composite is refused
            # §10: the composite is the crop's rows, cut into the chosen lanes
            mask, box = self._lane_frame()
            if box is None or not self.layout or not self.layout.get("edges"):
                return corrected[:, :0]
            y0, y1 = lanes.composite_rows(box, corrected.shape[0])
            return lanes.stitch_lanes(corrected[y0:y1, :], self.layout["edges"],
                                      self.composite_groups)
        if apply_crop and st.crop is not None:
            anchor = content_anchor(self.display.display_u8(path))
            box = display_rect_to_corrected_box(
                st.crop.as_tuple(), st.crop.angle, anchor, st.applied.value,
                corrected.shape[:2],
            )
            if box is not None:
                corrected = corrected[box[1]:box[3], box[0]:box[2]]
        return corrected

    def composite_ignored(self, apply_crop: bool) -> bool:
        """Composite is active but 'Rotation only' is selected (§10)."""
        return self.composite_active() and not apply_crop

    def save_as(self, dest, use_original: bool, apply_crop: bool = True) -> bool:
        if self.composite_ignored(apply_crop):
            self._warn(i18n.tr("dlg.save_as"), i18n.tr("msg.composite_rotation_only"))
            return False
        arr = self.export_array(use_original, apply_crop)
        if arr is None:
            self._warn(i18n.tr("dlg.save_as"), i18n.tr("msg.no_image_open"))
            return False
        try:
            export.save_image(dest, arr)
        except Exception as exc:  # noqa: BLE001
            self._error(i18n.tr("dlg.save_as"), i18n.tr("msg.could_not_save", exc=exc))
            return False
        self._status(i18n.tr("status.saved", name=Path(dest).name))
        return True

    def copy_to_clipboard(self, use_original: bool, apply_crop: bool = True) -> bool:
        if self.composite_ignored(apply_crop):
            self._warn(i18n.tr("dlg.copy_clipboard"), i18n.tr("msg.composite_rotation_only"))
            return False
        arr = self.export_array(use_original, apply_crop)
        if arr is None:
            self._warn(i18n.tr("dlg.copy_clipboard"), i18n.tr("msg.no_image_open"))
            return False
        try:
            export.copy_tiff_to_clipboard(arr)
        except Exception as exc:  # noqa: BLE001
            self._error(i18n.tr("dlg.copy_clipboard"), i18n.tr("msg.could_not_copy", exc=exc))
            return False
        self._status(i18n.tr("status.copied"))
        return True

    # -- persistence -------------------------------------------------------- #
    def save_session(self) -> None:
        folder = str(self.folder) if self.folder else ""
        self.session_store.save(make_session(folder, self.index, self.state.rotations()))

    def on_close(self) -> None:
        try:
            self.save_session()
        except Exception:  # noqa: BLE001 - never block closing
            pass
        self.service.cancel()