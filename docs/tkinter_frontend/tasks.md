# tkinter_frontend — Implementation Tasks

> Feature: `tkinter_frontend` · Sources: `spec.md` (WHAT), `design.md` (HOW)
> The tasks below translate the technical design into executable, verifiable work items for a NEW
> unified single-window Tkinter desktop app that reuses the canonical in-repo pipeline
> (`band_centroids` → `ladders.estimate`) and the canonical display preprocessing unchanged.
> Delivery target: windowed PyInstaller `.exe` on Windows 10/11, DPI-aware at `dpi/96`.

## Ordering and Dependency Notes
- Groups 1–2 are foundations (canonical seam, typed contracts, run-directory persistence) and must be
  completed first.
- Groups 3–7 are the user-facing business capabilities; they build on Groups 1–2 and may be developed
  in parallel once the foundations land, except where a subtask names a dependency.
- Group 8 assembles the shell that hosts Groups 3–7.
- Groups 9–12 are delivery, integration, and verification.
- Group 13 is a **corrective** workstream against the already-delivered shell (Group 8). It fixes a
  layout-starvation defect observed in the unified Tkinter frontend and must be executed **after**
  the shell of Group 8 exists; it changes only layout/sizing, not behaviour. Its verification task
  (13.4) is the acceptance gate for the correction.
- New code lives in a new package `geltilt/app/`; the only change inside the canonical modules is the
  additive `kept_lanes` output in `geltilt/ladders.py`.

### Reported Defect (Group 13 context)
- **Symptom**: after opening a folder and running Auto-rotate (diagnostics populate correctly), **no
  image surface is visible** — only the sample browser (left) and the controls/diagnostics panel
  (right) are shown; the middle viewer area collapses.
- **Root cause (measured at `ui_scale=2.0`)**: in `geltilt/app/window.py::_build`, only column 1
  (the viewer) is weighted (`self.columnconfigure(1, weight=1)`); columns 0 and 2 have weight 0 and no
  bounded size. The side panel's requested width (~2197 px) is dominated by the `tk.Text` warnings
  widget in `geltilt/app/widgets/diagnostic_panel.py`, which uses Tk's default `width=80` characters
  (~2100 px at 200% scaling). The manual slider (`manual_angle.py`, `length=int(300*ui_scale)`) and the
  browser listbox (`sample_browser.py`, `max(28, int(34*ui_scale))` chars) add ~905 px and the viewer
  requests ~754 px, so the total request (~3856 px) exceeds the window (~2048 px) and Tk shrinks the
  only weighted column — the viewer — to 1×1 px.
- **Required outcome**: the single viewer (spec §5.5.1) must always be visible and receive priority
  space; the browser and the side panel must be bounded so they can never starve it, at `ui_scale` up
  to 2.0, without changing behaviour elsewhere.

---

## 1. Establish the Canonical Estimate Seam and Typed Domain Contracts

### 1.1 Add Accepted-Lane Subset to the Canonical Estimator
- [ ] In `geltilt/ladders.py`, extend the `ladders.estimate` result dict additively with `kept_lanes`: the list of `pooled` lane dicts (each carrying `s`, `ybar`, `idx`), without changing any scoring, selection, or threshold logic (spec §5.4.4, design §1.1.2 / §2.1.3(e)).
- [ ] Ensure the added key does not alter the existing scalar keys (`angle`, `se`, `ok`, `reason`, `warnings`, `n_bands`, `n_ladders`, `kept`, `fallback`, `overdispersion`) so CLI parity is preserved (spec §4.4.2).
- [ ] Add a unit test in `tests/test_ladders.py` asserting `len(kept_lanes) == kept`, that each entry has `s`/`ybar`/`idx`, and that repeated runs on a fixed input remain deterministic (spec §4.2.2).

### 1.2 Define Domain Types
- [ ] Create `geltilt/app/domain.py` with frozen dataclasses `AppliedRotation`, `LaneLine`, `EstimateOutcome`, `SampleState`, `SessionState`, `DiagnosticLogEntry` exactly matching the design contracts (design §2.2.2.1, §2.3.2).
- [ ] Add `AppliedRotation` validation/normalisation helper enforcing clamp to `[-180.0, +180.0]`, rounding to `0.1°`, and `source ∈ {none, auto, manual}` with `none ⇒ value == 0.0` (spec §6.3, §5.3.1).
- [ ] Add `SessionState.schema_version` and a `default()` factory for a fresh session (spec §6.4, design §2.2.1).

### 1.3 Define Ports and Errors
- [ ] Create `geltilt/app/ports.py` with `typing.Protocol` definitions `EstimatorPort.estimate(image) -> EstimateOutcome`, `SessionStore.load()/save(state)`, and `DiagnosticLog.append(entry)` (design §2.2.2.2).
- [ ] Define domain exceptions `ModelUnavailableError` and `EstimateError` for the adapter to raise (design §1.2.5, §2.2.2.2).

### 1.4 Implement the Single Canonical Estimator Adapter
- [ ] Create `geltilt/app/estimator.py` implementing `CanonicalEstimator` (an `EstimatorPort`): call `band_centroids(image, …)` then `ladders.estimate(centroids)` using the same two steps as `estimate_rotation_ladders`, with `threshold=0.5`, `min_area=30` (spec §5.2.1, design §1.2.2, §2.2.2.2).
- [ ] Project the canonical dict into `EstimateOutcome`: `ok`, `angle` (as returned, never negated), `se`, `reason`, `warnings`, `n_bands`, `lanes_used = kept`, `lanes_found = n_ladders`, `fallback`, retained `centroids` and `accepted_lanes`, and the untouched `raw` dict (spec §5.2.6, design §2.2.2.1).
- [ ] Map a missing/undownloadable ONNX model to `ModelUnavailableError` and inference failure to `EstimateError`; a refusal is a normal `ok=False` return, not an exception (spec §5.2.3.2/3, design §2.2.2.2).
- [ ] Add a test asserting the adapter output matches the canonical command-line record fields (angle, accepted/refused status, reason) for the same image (spec §4.4.2).

---

## 2. Run-Directory Configuration and Persistence

### 2.1 Resolve Launch Configuration
- [ ] Create `geltilt/app/config.py` with an `AppConfig` holding `run_dir`, `ui_scale = dpi_scale()`, `model_path` (env `GELTILT_GELGENIE_ONNX` override, else cached), `image_exts`, `threshold = 0.5`, `min_area = 30` (design §2.1.2).
- [ ] Implement run-directory resolution defaulting to `run_directory()` (exe folder when frozen, else CWD) with an optional user-supplied override passed at launch (spec §5.6.1, design §1.2.6).
- [ ] Implement `ensure_run_directory()` creating the directory if missing before the first write (spec §5.6.1.1).

### 2.2 Session Store
- [ ] Create `geltilt/app/session.py` implementing `JsonSessionStore` with a single session file in the run directory (spec §6.5.2).
- [ ] Implement `save(SessionState)` writing the full payload atomically (folder, index, per-sample `{value, source}` rotations, `schema_version`, `updated_at`) (spec §5.6.2, §6.4).
- [ ] Implement `load()` returning a default `SessionState` for a missing or malformed file, and preserving round-trip integrity for a valid file (spec §5.6.3.2, §6.4.5).
- [ ] On write failure (`OSError`), warn once and continue in-memory without persistence (spec §5.6.3.1).

### 2.3 Diagnostic Run Log
- [ ] Create `geltilt/app/log.py` implementing `JsonlDiagnosticLog` appending one JSON line per auto-rotation run (timestamp, sample, angle, se, ok, reason, warnings) to a single append-only file in the run directory (spec §4.4.1, §5.6.4, design §1.1.2-H).
- [ ] Guarantee metadata-only content (no raw pixels) and that the log path never aliases a source image (spec §4.3.4, §5.6.5).

### 2.4 Persistence Tests
- [ ] Add tests for session save→load round-trip (folder, index, rotations) and corrupt-file fallback to a fresh session (spec §6.4.5, §5.6.3.2).
- [ ] Add a test asserting the diagnostic log appends a valid JSON line with the expected fields per run (spec §4.4.1).

---

## 3. Sample Browsing and Navigation

### 3.1 Sample Browser Surface
- [ ] Create `geltilt/app/widgets/sample_browser.py` with a `SampleBrowser` listing supported images via `find_images(directory)` in ascending name order with a stable index from 0 (spec §5.1.1.1/2/5, design §1.1.1).
- [ ] Display the active sample's index and name, and highlight exactly one active entry (spec §5.1.1.4).
- [ ] Populate the list lazily/virtualised (windowed slice) so folders with ≥2000 samples remain responsive, preserving the stable ascending index (spec §5.1.3.4, design §1.1.2).

### 3.2 Folder Open
- [ ] Implement `AppController.open_folder(path)` in `geltilt/app/controller.py`: list samples, activate index 0, update session folder (spec §5.1.2, design §2.2.2.3).
- [ ] On an empty folder, keep the prior folder/selection and warn listing the supported extensions (spec §5.1.3.1).
- [ ] On an unreadable/missing folder, abort the load, keep prior state, and show an error naming the folder and cause (spec §5.1.3.2).

### 3.3 Index Navigation
- [ ] Implement `goto(index)`, `next_sample()`, `prev_sample()` clamping to `[0, count-1]` with no wrap (spec §5.1.1.3).
- [ ] Wire list selection to `goto(k)` so selecting entry k activates sample k (spec §5.1.1.3b).

### 3.4 Per-Sample State Retention and Restore
- [ ] Create `geltilt/app/state.py` `SampleStateStore` keyed by sample name holding applied rotation, last estimate, and failure flag (spec §5.1.1.6, design §2.1.3(d)).
- [ ] On navigation, restore a visited sample's applied rotation + last estimate, or start at `none` / estimate-not-run for a new sample (spec §5.1.1.6, design §2.1.3(c)).
- [ ] Mark a sample that fails to load as failed in the session while keeping navigation and the app usable (spec §5.1.3.3, §4.2.3).

### 3.5 Browsing Tests
- [ ] Add tests for ordered listing, filter of unsupported extensions, clamp-at-ends navigation, and per-sample restore after round-trip (spec §5.1.1).
- [ ] Add a test asserting source files are byte-identical after browsing activity (spec §5.1.1.7).

---

## 4. Single-Viewer Rendering

### 4.1 Canonical Display Preprocessing
- [ ] Create `geltilt/app/display.py` with a `DisplayPreprocessor` that calls the canonical `preprocess_for_display` (`robust_limits(0.5, 99.5)` + `to_u8(…, 1.0)`) on the full frame and caches the uint8 result per sample (spec §4.5.5, design §1.2.4, §2.1.3(c)).

### 4.2 Single Viewer Surface
- [ ] Create `geltilt/app/widgets/viewer.py` with a `SingleViewer` hosting exactly one image surface (no split/side-by-side) (spec §5.5.1, §5.5.7).
- [ ] Implement fit-to-window on load using `viz.content_window` for the fit scale, preserving aspect ratio and handling the no-content-window fallback (spec §5.5.2, §5.5.3.1, design §1.2.7).
- [ ] Implement wheel zoom about the pointer and drag panning using `affine_coeffs` / `source_to_screen` / `screen_to_source` (spec §5.5.3, design §1.1.1).
- [ ] Implement `set_rotation(angle)` to re-render the composited image at the applied rotation (angle as-is) using `correct_rotation`/`rotate_image`, so the viewer reflects the current applied rotation (spec §5.5.4, §5.2.6).
- [ ] Defer rendering when the viewer has no usable canvas size yet (spec §5.5.3.2).
- [ ] Downscale for display only; never mutate the underlying sample array (spec §5.5.6).

### 4.3 Optional Vertical Guides
- [ ] Add a guides toggle drawing evenly spaced vertical lines via `viz.draw_vertical_guides`, spaced using `ui_scale` (spec §5.5.5, design §1.1.2).

### 4.4 Rendering Tests
- [ ] Add tests for aspect-preserving fit, pointer-anchored zoom, pan, and that rotation is applied without negating the sign (spec §5.5.2/3, §5.2.6).
- [ ] Add a test asserting display downscaling leaves the source array unchanged (spec §5.5.6).

---

## 5. Automatic Rotation Estimate

### 5.1 Asynchronous Estimation Service
- [ ] Create `geltilt/app/service.py` with an `EstimationService` running the estimator on a single worker thread, marshalling results through a `queue.Queue`, and exposing `start(sample, image, epoch) -> token`, `cancel()`, and `poll()` for Tk `after` polling (spec §4.1.3, design §2.2.2.4).
- [ ] Enforce at most one in-flight estimate and token-based supersede/cancel of earlier requests (design §2.1.3(a)).
- [ ] Serialise estimates (single worker) because `ladders.P` is a module-global, and keep the UI unblocked by short polls only (spec §4.1.3, design §1.2.1).

### 5.2 Auto-Rotate Flow
- [ ] Implement `AppController.trigger_auto_rotate()`: show a busy indicator, submit the active sample's image, and apply exactly one result on the main thread (spec §5.2.2).
- [ ] On an accepted estimate (`ok=True` and finite angle), apply the angle as-is to the viewer and record `AppliedRotation(value, "auto")` (spec §5.2.3, §5.2.6).
- [ ] On a refusal (`ok=False` or non-finite angle), leave the image unrotated (or at prior angle) and display the reason + warnings; never substitute 0° or another method (spec §5.2.4, §5.2.9).
- [ ] Re-triggering auto-rotate replaces the sample's previous estimate and applied rotation (spec §5.2.8).
- [ ] Record the manual-edit race guard: apply the result only if `rotation_epoch` is unchanged since the trigger, otherwise update diagnostics without overwriting the manual edit (spec §5.3.3.3, design §2.1.3(a)).
- [ ] Append a `DiagnosticLogEntry` to the run log for every auto-rotation run (spec §4.4.1, §5.6.4).
- [ ] Catch and isolate estimator/model errors to the active sample, surface an error dialog, and log the failure (spec §5.2.3.3, §4.2.3, §4.4.3).

### 5.3 Estimation Tests
- [ ] Add tests for accepted→applied, refused→unrotated-with-reason, determinism across repeated runs, and supersede behaviour (spec §5.2.3/4/7/8).
- [ ] Add a test asserting the busy indicator state transitions while the worker runs, with the UI not blocked beyond 250 ms per poll (spec §4.1.3).

---

## 6. Manual Rotation Override

### 6.1 Manual Angle Control
- [ ] Create `geltilt/app/widgets/manual_angle.py` with a `ManualAngleControl` exposing `get_value`, `set_value`, and `bind_change`, combining a continuous control with step buttons and numeric entry (spec §5.3.1.1).
- [ ] Clamp manual input to `[-180.0, +180.0]` at `0.1°` resolution and display the clamped value (spec §5.3.1.2, §5.3.3.2).
- [ ] Reject non-numeric input and revert to the last valid value (spec §5.3.3.1).

### 6.2 Controller Integration
- [ ] Implement `set_manual_angle(value)` marking `source = manual` and superseding the auto estimate for the sample (spec §5.3.1.3, design §2.1.3(b)).
- [ ] Implement `nudge(delta)` for at least ±0.1°, ±1°, and ±10° with additive clamping (spec §5.3.1.6).
- [ ] Implement `zero_angle()` setting `AppliedRotation(0.0, "none")` (spec §5.3.1.5, design §2.1.3(b)).
- [ ] Re-render the viewer live on every change without a separate confirm action (spec §5.3.1.7).
- [ ] Ensure re-running auto-rotate replaces a manual value with source `auto` (spec §5.3.1.4).
- [ ] Guarantee manual override never writes to the source image (spec §5.3.1.8).

### 6.3 Manual Override Tests
- [ ] Add tests for clamp/round, non-numeric rejection, precedence over auto, and auto re-run replacing manual (spec §5.3.1, §5.3.3).

---

## 7. Diagnostic Display and Overlays

### 7.1 Diagnostic Panel
- [ ] Create `geltilt/app/widgets/diagnostic_panel.py` showing, for the active sample, the applied rotation value + source (`none`/`auto`/`manual`), estimate status (`accepted`/`refused`/`not run`), reason, uncertainty, and warnings (spec §5.4.1.1/3).
- [ ] Show band count and "lanes used / lanes found" when an estimate exists (spec §5.4.1.2).
- [ ] Provide a scrollable warning area and neutral "not run"/"no detections" states (spec §5.4.3.1/2/3).
- [ ] Always reflect the active sample; never present another sample's diagnostics as current (spec §5.4.1.5).

### 7.2 Overlay Renderer
- [ ] Add an overlay renderer in `geltilt/app/widgets/viewer.py` mapping band centroids and accepted lane lines through the same affine used for display, so overlays cannot drift from the image (spec §5.4.4, design §1.1.2, §2.1.3(e)).
- [ ] Implement `toggle_overlay(kind)` for `centroids`, `lane_lines`, and `guides`, each independently toggled, with overlays never mutating source data (spec §5.4.4, §5.5.5).
- [ ] Show an explanatory note in the panel when an overlay has no detections (spec §5.4.3.2).

### 7.3 Diagnostic Tests
- [ ] Add tests asserting diagnostics switch with the active sample, reset to "not run" for a fresh sample, and that overlay toggles draw/clear without touching source data (spec §5.4.1.5, §5.4.4).

---

## 8. Application Shell, DPI Scaling, and Launch

### 8.1 DPI Awareness and Scaling
- [ ] Call `enable_dpi_awareness()` before creating the first Tk window, then `apply_tk_scaling(root, ui_scale)` with `ui_scale = dpi_scale()` (spec §4.5.2, design §1.2.6).
- [ ] Apply `ui_scale` to all new-window geometry, fonts, paddings, row heights, control sizes, and guide spacing so content is crisp and hit targets reachable at 192 DPI, and scale = 1.0 at 96 DPI (spec §4.5.2, §5.4.1.6, design §1.1.2).

### 8.2 Shell Assembly and Wiring
- [ ] Create `geltilt/app/window.py` with an `AppWindow` (`tk.Tk`) hosting exactly one viewer, one sample browser, one control strip, and one diagnostic panel (design §1.1.3-A).
- [ ] Wire the `AppController` to all sub-components; keep the controller the only state mutator, on the main thread (design §2.1.2).
- [ ] Accept launch parameters: an optional initial folder path and an optional run-directory path (spec §1.2.2, design §1.1.3-A).

### 8.3 Launch Session Restore
- [ ] On launch, restore the last folder, index, and per-sample applied rotations from the session file and refit the viewer (spec §5.6.2, §4.2.4, design §2.1.3(c)).
- [ ] Implement `reset_view()` refitting to content with preserved aspect (design §2.2.2.3).
- [ ] Implement `save_session()` on close (and periodic) persisting folder + index + rotations (spec §5.6.2, design §2.2.2.3).

### 8.4 Shell Tests
- [ ] Add a smoke test constructing the window off-screen asserting exactly one image surface exists (spec §5.5.1).

---

## 9. Planned-Capability Placeholders (Deferred, Inert)

### 9.1 Lane-Crop and Export Placeholders
- [ ] Add disabled, "planned"-labelled controls for lane cropping and output export that perform no action and produce no crop or file (spec §5.7.1, §5.7.2, design §1.1.3-I).
- [ ] Add a test asserting browsing and rotating produce no corrected/cropped image files in the run directory (spec §5.7.2.2, §5.6.3).

---

## 10. Packaging and Delivery

### 10.1 Entry Point
- [ ] Create `run_app.py` launching `AppWindow`/`AppController` with the DPI and run-directory setup (spec §4.5.1, design §1.1.3-J).

### 10.2 PyInstaller Build
- [ ] Create `build_app_exe.py` and `geltilt-app.spec` modelled on the existing windowed build pattern, producing a single-file `--windowed` (no console) executable (spec §4.5.1, design §1.1.2).
- [ ] Collect `geltilt` and `cv2` submodules and bundle the bundled GelGenie ONNX model path so the packaged app runs offline (spec §4.3.1, design §1.1.3-J).
- [ ] Verify the packaged `.exe` resolves the run directory to the executable folder and writes session/log there (spec §5.6.1, §4.5.1).

---

## 11. Integration, End-to-End, and Parity Testing

### 11.1 Cross-Module Integration
- [ ] Add an integration test for the full path: open folder → navigate → auto-rotate → manual override → session save/restore (spec §5.1–§5.6).
- [ ] Add an integration test verifying failure isolation: a corrupt sample fails without corrupting other samples' session state (spec §4.2.3, §5.1.3.3).

### 11.2 CLI Parity
- [ ] Add a parity test asserting the App's angle, accepted/refused status, and reason equal the canonical command-line result for the same image (spec §4.4.2, §5.2.2).

### 11.3 Performance and Reliability
- [ ] Add a performance check that a 1280×1024 estimate completes within 30 s (target ≤10 s), runs off the UI thread, and manual navigation re-renders within 500 ms (spec §4.1.2/3/4).
- [ ] Add a reliability test confirming determinism of repeated auto-rotate on an unchanged image (spec §4.2.2).
- [ ] Add a source-integrity test confirming no source image bytes change across a full session (spec §4.3.2, §5.1.1.7).

---

## 12. Final Verification and Review

### 12.1 Review Gates
- [ ] Code review of the canonical seam change (`kept_lanes`) confirming additive-only, no scoring change, and CLI parity preserved (spec §4.4.2, design §1.1.2).
- [ ] Design-consistency review confirming the App reuses the single canonical rotation engine and contains no alternative rotation method or silent 0° fallback (spec §1.4.1, §5.2.2, §5.2.9).
- [ ] DPI review at 192 DPI confirming geometry, fonts, and hit targets are scaled by `dpi/96` and diagnostic text is legible without clipping (spec §4.5.2, §5.4.1.6).
- [ ] Scope review confirming lane cropping and export are inert/disabled and no image outputs are written (spec §5.7).
- [ ] Record the verification results and, for any failed item, raise a follow-up action with an owner.

---

## 13. Corrective Fix — Guarantee the Single Viewer Is Always Visible (Layout Starvation at 200% Scaling)

> Corrects the reported defect ("no image surface visible") against the delivered shell of Group 8.
> Scope is **layout and sizing only**; the single-viewer requirement and diagnostics behaviour are
> preserved. Applies at `ui_scale` up to 2.0 (spec §5.5.1, §4.5.2).

### 13.1 Bound the Diagnostics Side-Panel Width
- [ ] In `geltilt/app/widgets/diagnostic_panel.py`, give the warnings `tk.Text` an explicit bounded `width` in characters that is **small and does not grow with `ui_scale`**, so its requested pixel width can no longer balloon to ~2100 px at `ui_scale=2.0` (spec §5.4.3.3, §5.4.1.6).
- [ ] Keep `wrap="word"` and the existing fill/expand + vertical scrollbar so long warning lists still wrap and remain scrollable and legible within the bounded panel; do not reduce diagnostics content or change the neutral "not run"/"no detections" wording (spec §5.4.1, §5.4.3.1/2/3).
- [ ] Verify the panel still expands to the width the side column allocates to it (it must be a consumer of space, never a driver of window width).

### 13.2 Cap the Browser and Control-Strip Widths
- [ ] In `geltilt/app/widgets/sample_browser.py`, cap the listbox character width so `max(28, int(34*ui_scale))` can no longer request ~905 px at `ui_scale=2.0`; keep the ordered listing, stable 0-based index, and lazy population unchanged (spec §5.1.1.1/2, §5.1.3.4).
- [ ] In `geltilt/app/widgets/manual_angle.py`, cap the horizontal slider `length` (currently `int(300*ui_scale)`) so it stays within the bounded side column; keep `[-180.0, +180.0]` range, `0.1°` resolution, step buttons, and entry validation unchanged (spec §5.3.1).
- [ ] In `geltilt/app/window.py::_build`, give the side panel frame an explicit bounded width (overlays + auto-rotate + planned placeholders + manual control + diagnostics combined) so its total requested width cannot exceed the intended cap.
- [ ] Confirm no capping clips or hides any control at `ui_scale=1.0` or `2.0` (all interactive hit targets remain reachable; spec §4.5.2).

### 13.3 Make the Viewer Column the Protected, Expanding Cell
- [ ] In `geltilt/app/window.py::_build`, set the viewer column to `weight=1` with a `minsize` floor (in px, scaled) so it cannot be shrunk below a non-trivial width, and keep the browser (column 0) and side panel (column 2) at `weight=0` with bounded widths (spec §5.5.1).
- [ ] In `geltilt/app/widgets/viewer.py`, give the `tk.Canvas` a **small intrinsic `width`/`height`** so the viewer frame's requested size is small and it grows into the remaining space instead of collapsing; keep `fill="both", expand=True` (spec §5.5.1, §5.5.3.2).
- [ ] Ensure the combined requested width of browser + viewer intrinsic + side panel fits within the window geometry at `ui_scale=2.0`; adjust the geometry/minsize in `window.py` if needed so the total request stays within the window (spec §4.5.2).
- [ ] Confirm the existing "viewer not yet laid out" deferral still works: the canvas reveals the image once it first receives a usable size (`<Configure>` → `reset_view`) (spec §5.5.3.2).

### 13.4 Verification — Headless/Off-Screen Viewer Visibility at 200% Scaling
- [ ] Add a test in `tests/app/test_appwindow.py` that constructs `AppWindow(run_dir=..., ui_scale=2.0)`, makes it off-screen/mapped, and calls `update_idletasks()`/`update()` so the geometry manager assigns real sizes (reuse the `pytest.importorskip("tkinter")` + `TclError` skip pattern from `tests/app/conftest.py`).
- [ ] Assert `window.viewer.canvas.winfo_width()` and `winfo_height()` are **≥ a non-trivial threshold** (e.g. 200 px) at `ui_scale=2.0`, proving the viewer is not collapsed (spec §5.5.1).
- [ ] Assert exactly **one** `tk.Canvas` exists and it is `window.viewer.canvas` at `ui_scale=2.0`; after loading a sample and rendering, assert exactly one image surface is drawn (spec §5.5.1, §5.5.7).
- [ ] Extend the `_selftest` report in `geltilt/app/window.py` to record the measured viewer canvas width/height at the resolved `ui_scale` and set `ok=False` when the viewer is collapsed, so the packaged build self-check also guards this defect (spec §4.5.1).
- [ ] Re-run the existing `tests/app/` suite (window smoke test, widget tests, controller/service tests) at `ui_scale=1.0` and confirm no behaviour change (browsing, auto-rotate, manual override, diagnostics, session) (spec §5.1–§5.6).