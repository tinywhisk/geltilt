# Project Timeline — geltilt

Gel electrophoresis image editor (Python). Two planned features:

1. **Auto rotation correction** — use band smears + vertical projection to make lanes vertical.
2. **Lane-based cropping** — crop individual lanes out of a straightened gel.

---

## 2026-10-08

### 1. Requirements & scope
- User requested a gel electrophoresis image editor; started with **feature 1 (auto rotation correction)**.
- Constraint: correct tilt so **lanes run vertically**, driven by the **band smears** using a **vertical projection**.
- Stack chosen: Python + numpy + Pillow + OpenCV (headless).

### 2. Sample analysis
- Sample: `samples/2026 0508 PPA6-1 PPA6-2 13772.Tif` — 1280×1024, 8-bit grayscale TIFF, 600 dpi.
- Findings:
  - Dark-background fluorescent gel (bright bands on dark); extremely sparse (mean ≈ 2/255).
  - Gel content confined to a small central region (≈ x 415–563, y 436–731); ~4 lanes.
  - Lanes tilted by a few degrees (vision estimate ~2–4°).
  - Background is mostly black with **MAD = 0**, which breaks naive thresholding.

### 3. Algorithm exploration & validation
- Naive variance/entropy of the projection profile were **flat or biased** (dominated by the single super-bright band and by rotation corner artifacts).
- Fixes that made detection reliable:
  - Score on a **rotation-safe central crop** so constant-filled corners never enter the projection.
  - Use **log compression** of the signal, and **`tv2`** (sum of squared first differences of the vertical projection) as the sharpness metric.
- Controlled validation:
  - **Synthetic gels**: injected rotations of ±1.5–8° recovered to ≈0.1°.
  - **Real ROI**: injecting known rotations tracked with slope ≈1.0 and a stable offset — detector returns **≈ −2.70°**, stable across all ROI choices.
- **Visual confirmation**: −2.5° straightens the lanes; −5.0° over-corrects; −2.8° (pipeline) confirmed essentially vertical.

### 4. Implementation
- Created the `geltilt` package:
  - `imageio.py` — load/save preserving 8/16-bit and float dtype, gray/color.
  - `preprocess.py` — background estimate, polarity detection, signal extraction, content ROI.
  - `projection.py` — vertical projection + sharpness metrics (`tv2`, `d2sq`, `var`, `entropy`).
  - `rotation.py` — rotation, expand, safe-crop helpers.
  - `core.py` — `estimate_rotation`, `correct_rotation`, `auto_correct`.
  - `cli.py` + `__main__.py` — CLI with JSON report, batch, expand/crop.
- Packaging: `pyproject.toml` (console script `geltilt`), `requirements.txt`.

### 5. Testing & verification
- 40 pytest tests: synthetic known-tilt recovery, projection metrics, image I/O round-trips, CLI behavior, real-sample checks.
- **40/40 passing.**
- End-to-end CLI on the real sample: detects **−2.8°**, writes corrected TIFF + JSON report.
- Fixed a Pillow deprecation warning and made `content_mask` robust to zero-MAD backgrounds.

### 6. Repository
- Initialized git repo (branch `main`), repo-local identity `tinywhisk <tinywhisk@localhost>`.
- Added `.gitignore` (Python/agent artifacts), initial commit `4674b51`.

### 7. Presentation figures
- Added `geltilt/viz.py`: `auto_contrast`, `content_window`, vertical-guide overlay, and
  side-by-side renderers, so results are presented as before/after figures with
  overlaid info (per user's requested presentation style).
- Added `render_comparison` (BEFORE/AFTER rotation) and `render_preprocessing_comparison`
  (RAW / PREPROCESSED), plus CLI flags `--compare` and `--compare-preprocess`.

### 8. Preprocessing standard
- Adopted the user-mandated preprocessing verbatim: `robust_limits(arr, plo=0.5, phi=99.5)`
  and `to_u8(arr, lo, hi, gamma=1.0)` in `geltilt/preprocess.py`.
- **Gamma is fixed at 1.0** ("Don't change gamma"); gamma is not used to tune brightness.

### 9. Bug fix: display window computed on the full frame
- Symptom: raw vs preprocessed panels looked identical.
- Cause: the window (`robust_limits`) was computed on the **cropped gel**, where a single
  super-bright band drags `hi` to ~242 → gain ~1.08 → near-identity stretch (max change 12/255).
- Fix: compute the window on the **full image** first, then crop for display. On the full
  frame the same percentiles give `lo=1, hi=26` → ×10.2, which visibly brightens the gel.
- Removed the reference scratch script `preproc_compare.py` (and its output) after folding
  the behaviour into `viz.py`.
- Test suite grew to **54 tests, all passing**.

---

## 2026-10-10 (continued session)

### 10. Slab-edge failure mode identified
- The vertical-projection estimator can lock onto the **gel slab edge** — the long, low-contrast,
  column-constant black→gray boundary of the illuminated region — instead of the bands
  (edge ≈ 0° vs bands ≈ +3° on sample #1; the estimator returned ≈ 0°). Per-column background
  subtraction was only marginal (flagged 26 → 23) and not sufficient.

### 11. Slab masking (marker watershed on log-intensity gradient)
- `slab_ws.py` / `slab_edges.py` → **v2** (EDT inside-seed; verified a pure speedup, bit-identical masks)
  → **v3** (`slab_ws_v3.py` adds `halo_depth` + `open_sides`, which sharply cuts `%weak`).
- Mask quality scored by `%weak` (no sharp step) and `%frame` (outline follows the image frame).
- **Adopted the v3 outline, shrunk 10 px, as the slab mask** and wired it into the `smears` engine
  (`--slab-mask-dir`, `--no-slab-mask`). Validation with the mask on: median 0.53°, >1° = 27, max 4.83°
  — it removed the catastrophic outliers but shifted 10 borderline cases. Slab-mask files are retained
  for a later step (not deleted).

### 12. Band-only masking via GelGenie (validated dead-end for the default path)
- Integrated the GelGenie ONNX band segmenter via `cv2.dnn` (`geltilt/bandmask.py`) to down-weight bands.
- Band coverage is only ~1–2 % of pixels, so pixel/neighbourhood/local-max reweighting were neutral or
  worse; the mask only helped where the slab edge dominated. Reweighting approaches abandoned.

### 13. `smears` SPEC engine
- Implemented `SPEC-AUTOROTATE.md` as `geltilt/smears.py` (`gel_mask` → eroded `interior` →
  per-angle `flatten` + `ridge` → `contrast`/`persistence` cues). Acceptance tests T1–T5 pass and it
  reproduced the SPEC's recorded values exactly (`G1-NE` −1.39°, `SLERF2GDNA` +5.19°, `胡椒rna2` +1.76°).

### 14. Cue arbitration + reweighting experiments
- §7.2 arbitration changed to **prefer `contrast` unless its peak is at the range edge (`edge_cv`)`,
  falling back to `persistence` — this fixed the persistence-spurious cases.
- `asinh(resp/k)` compression lowered max error 8.42° → 3.25° but *shifted peaks* (not a real fix);
  band-mask and local-max reweighting did not help. All reweighting options were exhausted.

### 15. Definitive lane-based estimator (`ladders`)
- Adopted `improved-rotation-alg/` — `ladders.estimate(band centroids)` — as the **one and only**
  rotation algorithm; retired `smears` + `projection` from the CLI (`geltilt/cli.py` rewritten).
- Front end implemented per **`BAND_DETECTION.md`**: GelGenie ONNX **tiled at native scale**
  (1024 window, overlap 256, tent-weight feather merge), softmax over the class axis, threshold 0.5 →
  opening 3×3 → 8-connected → area ≥ 30 → centroids. **No slab mask** is used in this path.
- End-to-end over 94 vs `ground_truth.json`: **92/94 coverage, mean 0.709°, median 0.646°, ≤1° 75 %,
  ≤2° 100 %, max 1.93°** — an exact reproduction of the SPEC. The 2 refusals match the documented
  structural refusals (`20260428 15810 990 酵母膜蛋白质粒`, `20260811 62.Tif`).
- Regenerated `rot_output/` (94 comparisons + `summary.json`) and `discrepencies/` (25 figures +
  `index.txt`); test suite **92/92**.

### 16. Repository state
- New modules this session: `bandmask.py`, `banddetect.py`, `ladders.py`, `smears.py`, plus the slab
  watershed scripts. Everything since commit `c6854bc` is still **uncommitted**.

---

## 2026-10-10 (unified Tkinter frontend)

### 17. Spec-driven build of the unified desktop app
- Ran the SDD flow (spec → design → tasks) for a **new unified single-window Tkinter app** under
  `.codeartsdoer/specs/tkinter_frontend/` (`spec.md`, `design.md`, `tasks.md`). Scope agreed with the
  user up front: folder/sample browsing, auto-rotate + manual override, pipeline diagnostics in **one
  viewer**; lane cropping and export **deferred/inert**; delivered as a windowed PyInstaller `.exe`.
- Implemented a new package `geltilt/app/`:
  - `domain.py` / `ports.py` — typed contracts (`AppliedRotation`, `EstimateOutcome`, `SessionState`,
    `LaneLine`, …), protocols and `ModelUnavailableError`/`EstimateError`.
  - `estimator.py` — `CanonicalEstimator`, the **only** rotation seam: the same two canonical calls as
    `estimate_rotation_ladders` (`band_centroids` → `ladders.estimate`), projected to a typed outcome.
  - `config.py` / `session.py` / `log.py` / `state.py` / `display.py` / `service.py` — run-directory
    config, JSON session, JSONL run log, per-sample state, canonical display preprocessing + cache,
    and an async single-worker estimate service (serialised because `ladders.P` is a module global).
  - `controller.py` + `widgets/` (`viewer`, `sample_browser`, `manual_angle`, `diagnostic_panel`) +
    `window.py` — the shell, navigation, auto/manual rotation, overlays and diagnostics.
  - `run_app.py`, `build_app_exe.py`, `geltilt-app.spec`, and a `geltilt-app` console script.
- One additive change inside the canonical module: `ladders.estimate` now also returns `kept_lanes`
  (the accepted `pooled` subset) for the overlay — no scoring/selection/threshold logic changed, so
  determinism and CLI parity are preserved.
- Tests: new `tests/app/` suite (domain, estimator, session, log, state, display, service, viewer,
  widgets, appwindow, controller integration/parity) plus `kept_lanes` cases in `tests/test_ladders.py`.
  Full suite **159/159**; `AppWindow` launch smoke-tested.

### 18. Packaging build
- Built the distributable with `python build_app_exe.py` → `dist/geltilt-app.exe` (single-file,
  `--windowed`, PyInstaller 6.22.3). The 57 MB GelGenie ONNX model is bundled via `--add-data` and
  `run_app.py` exposes it through `GELTILT_GELGENIE_ONNX` (from `_MEIPASS`/exe folder), so the packaged
  app runs offline; the script falls back to the runtime cache/download when no model is found.
- Added a `--selftest` mode to `geltilt.app.window` (mirrors the labeler's) that constructs the window
  hidden and writes `app_selftest.json`. The packaged exe was verified: exit 0, exactly **1 image
  surface**, run directory resolved to the **exe's own folder** (`dist/`), and the session file written
  there. App + ladders tests still **74/74**.

### 19. Viewer display transform
- The viewer's display preprocessing changed from `robust_limits`+`to_u8(gamma=1.0)` to a new
  `preprocess.preview_u8(arr, gamma=0.45, percentile=99.7)` = `clip(arr / p99.7, 0, 1) ** 0.45` scaled
  to uint8 (no low-end subtraction), matching the reference `preview` snippet. Wired in `DisplayPreprocessor`.
- **Display-only:** the estimator still consumes the **raw** source array (`DisplayPreprocessor.source_array`),
  so rotation accuracy is unaffected; the GelGenie segmenter does its own linear peak-scaling in `bandmask._model_input`.
  Verified bit-identical to the snippet's grayscale output on a real sample.

### 20. Viewer-visibility fix + browsing
- The centre viewer collapsed to 1×1 px at 2× DPI: the left browser `Listbox` and the diagnostics `tk.Text`
  (default 80-char width) both requested huge widths and crushed the weighted column. Removed `sample_browser.py`,
  added the top-left filename + `◀`/`▶` navigation, and constrained the diagnostics text width.

### 21. UI matched to `ui_layout_reference.py`
- Rebuilt `window.py` to the reference skeleton: file/folder bar (Load image / Load folder, arrows + filename,
  language picker), **Rotation** group (Auto rotate + entry + coarse arrows + slider + Reset), **Crop** group,
  a **single** viewer, **Export** group, and a bottom bar (overlays + compact horizontal **Diagnostics** + status).
- Unimplemented controls (`Load image...`, `Auto crop`, `Clear crop`, `Even widths`, `Save as...`, `Copy to
  clipboard`) are **enabled no-ops**. Diagnostics moved into the bottom bar and compacted (the old panel
  requested 724 px and crushed the viewer to 42 px / Export to 1 px); the viewer canvas got a small requested
  size so it expands. Verified at 2× scale: viewer **1864×474**, Export and bottom bars full height,
  `--selftest` image_surfaces **1**. Suite **162/162**.

### 22. Auto crop
- Wired the Crop group: **Auto crop** runs the GelGenie band segmenter on the raw frame
  (`banddetect.band_bbox` = union bbox of the kept components, same mask as band detection:
  prob>0.5, open 3×3, min-area 30), transforms that bbox into the **corrected/display frame**
  (`crop.source_bbox_to_display_rect`, anchor-relative `q = M(angle)(p - anchor)`) and pads a fixed
  **10 px** on all sides. The box is always upright and does **not** move when the rotation later
  changes (dragging locks it; zoom/pan keep it on the content).
- Viewer renders a yellow outline + 8 drag handles and a dark-red wash (`CROP_DIM`) over the
  cropped-out regions (composited into the PIL image). Edges/corners are hit-tested in screen space
  and dragged back into the display frame (`crop.screen_to_display`). **Clear crop** resets.
- **Gating:** Auto crop is enabled only once auto-rotate has succeeded for the active sample
  (`applied.source == "auto"` or an accepted estimate); the button starts disabled. Crop is stored
  per sample in `SampleState` (in-memory; not yet in the session file) and restored on navigation.
- New: `geltilt/app/crop.py`, `CropRect` in `domain.py`, `banddetect.band_bbox` (+ `_kept_components`
  refactor shared with `band_centroids`), viewer crop methods. Tests: `tests/app/test_crop.py`,
  viewer/controller/banddetect crop cases. Verified on a real sample: enabled after auto-rotate,
  box **1204×250** (bands + 10 px pad). Suite **182/182**.

### 23. Crop UX + GelGenie debris filter + split viewer
- **Mouse remap:** left-drag in the viewer now **draws/redraws the crop box** (a left click that
  doesn't land on an existing edge/corner handle starts a new box; a sub-4px click is ignored);
  **pan rebound to right-button drag** (`<ButtonPress-3>`/`<B3-Motion>`/`<ButtonRelease-3>`).
  Edges/corners still grab-and-adjust when hit first.
- **GelGenie debris filter:** `banddetect.band_bbox(right_margin_frac=0.03)` drops band components
  whose centroid lies in the rightmost 3% of the frame. Measured across 24 samples: GelGenie reports
  a fixed debris blob at `cx≈1260–1267` (`0.98·width`), while real bands reach at most `cx=1219`
  (`0.95·width`) — a clean separation at `0.97·width`. Applied to the **auto-crop box only**, not the
  rotation centroids (`band_centroids` unchanged, so the validated estimator is unaffected). Effect:
  sample `20206417 55130-G.Tif`'s box shrinks from **1182 px to 459 px** wide.
- **Viewer split:** the panes row is now a 2-column uniform grid; the **viewer occupies the left
  half** (~928 px at 2× scale) and the **right half is intentionally empty** (was two panes in
  `ui_layout_reference.py`), reserved for a future pane. Suite **189/189**.

### 24. Default view fits the whole image
- The viewer's default (and "fit"/reset) view was fitting the **band-content window**
  (`content_window`), which magnified the gel so the frame edges were off-screen. Changed
  `SingleViewer.reset_view` to fit the **whole source frame** — it maps the four image corners into
  the anchor-relative display frame, sizes them to the canvas, centers on the image, and caps the
  scale at **1.0** so the default never zooms in. On a real 1280×1024 sample in the 928×474 viewer
  the default is now scale **0.426**, whole image inside the canvas. Suite **190/190**.

### 25. Right column takes the below-viewer content
- Moved **everything that was below the viewer** into the right-hand column: the **Export** group,
  the **Overlays** checkboxes, the **Diagnostics** panel and the **status / folder** lines. The top
  bars (file/folder, Rotation, Crop) and the left-half viewer are unchanged.
- Readjusted for the narrower column: Export reflowed into three stacked rows, Overlays set
  one-per-line, Diagnostics switched to **two key/value pairs per row** with a tighter wraplength, and
  the folder line now shows the **folder name + count** (a full path could out-measure the half column).
- Because the old bottom bars are gone, the viewer grew from 928×474 to **928×896** at 2× scale; the
  right column requests 739 px and fits the 928 px half (and 324 px vs 628 px at 1×). Suite **190/190**.

### 26. Save as / Copy to clipboard
- New `geltilt/app/export.py`. Both actions export the **rotation-corrected** frame via the canonical
  `correct_rotation(source, angle)` (center pivot, same size, dtype preserved); cropping is **not**
  applied yet (a future addition).
- **Use original image** checkbox drives the pixel source: checked → the **raw** source array
  (`DisplayPreprocessor.source_array`, no brightness processing); unchecked → the brightness-processed
  **display** image (`preview_u8`).
- **Save as...** opens `asksaveasfilename` with TIFF / PNG / JPEG / BMP (+ All files) and picks the
  format from the extension; TIFF/PNG preserve bit depth, JPEG/BMP get a plain 8-bit reduction
  (`to_uint8`, no contrast processing).
- **Copy to clipboard** always puts a **TIFF** on the Windows clipboard (ctypes `CF_TIFF`), plus a
  `CF_DIB` fallback so it pastes in common apps. Verified the clipboard reports both formats.
- Tests: `tests/app/test_export.py` (depth conversions, rotation-preserves-dtype, TIFF/PNG/JPEG save,
  TIFF/DIB encoders) + controller `save_as`/`copy_to_clipboard` cases. Suite **202/202**.

### 27. Export honours the crop box
- The exported frame is now **cropped** as well as rotated. The crop rect lives in the display frame
  `q = M(angle_lock)(p - anchor)`; because `correct_rotation` uses the *same* `M(θ)=[[cos,sin],[-sin,cos]]`
  about the image centre, `q` maps to the corrected image by a pure translation, so an upright crop stays
  upright. New `crop.display_rect_to_corrected_box(rect, angle_lock, anchor, angle_now, size)` handles the
  general case (including a rotation changed after cropping) and clamps to the frame.
- `Controller.export_array(use_original, apply_crop)` rotates then crops; `save_as`/`copy_to_clipboard`
  take `apply_crop`, driven by the **Export mode** combobox (`Rotation + crop` = crop, the default;
  `Rotation only` = no crop). Verified on a real sample: full 1024×1280 → cropped **229×482**, matching
  the on-screen box (481×228). Suite **206/206**.

### 28. Export keyboard shortcuts
- Bound **Ctrl+S → Save as...** and **Ctrl+C → Copy to clipboard** at the window level (alongside the
  existing Ctrl+O). Ctrl+C is guarded by `_text_focus()` so it still copies text when an entry/combobox
  has focus; Ctrl+S is global. Suite **207/207**.

---

## 2026-10-11

### 29. Composite cropping (SPEC-COMPOSITE.md)
- New pure lane model `geltilt/app/lanes.py` (Tk-free, `cv2`+`numpy`): `band_blobs` (area floor 50),
  `label_lanes` (QuPath/GelGenie grouping + the mandatory progress guard, T-1), `fill_empty_lanes`
  (plain-median pitch, `PITCH_TOLERANCE=0.3`), `lane_layout` (region is `[x0,y0,x1,y1]`, T-6; midpoint
  edges derived once, T-5), `move_edge`/`insert_edge`/`remove_edge` + `renumber_after_{split,merge}`,
  `even_edges`/`nudge_off_bands`, `parse_lane_spec` (`LaneSpecError` carries a key + args),
  `spec_groups`/`output_runs`/`stitch_lanes`, `composite_rows`. Empty branch includes `edges: []` (fixes
  the §16 known defect). Acceptance tests **C1–C14** in `tests/app/test_lanes.py` (34 cases).
- Controller integration: Auto crop now runs **one** GelGenie inference, caches the source-frame band
  mask on `SampleState` (T-7) and derives the box from it; the composite lane layout is computed on the
  mask **rotated into the corrected frame** (`crop.rotate_mask`, nearest) with the crop's corrected-frame
  region (the source→corrected mapping is a pure translation). Gate = ready/mode/active; re-identify
  runs on crop change and on tick (R7); the box untickes itself when lanes stop being ready (R3).
- Editing + export: click a lane (append to the order, rewrites the spec as a flat list), middle-click
  splits/joins, left-drag moves a boundary; **Even widths** = `even_edges`+`nudge_off_bands` with §6.3
  reporting. Export = `correct_rotation` frame → crop rows → `stitch_lanes`; composite on + "Rotation
  only" is **refused** (§10); Save-as pre-fills `_rotated_composite` (CC-1). Verified on a real sample:
  9 lanes found, spec `3,1-2` → composite export **229×52**.
- Viewer overlay (§12): amber window rows across the pane, cyan solid/dashed boundary lines with end
  handles, lane numbers (cyan/grey) + amber `#n` badges, and a red tint darkened everywhere except the
  selected lanes × the window. Hit testing is a separate model (no new box, no left/right edges).
- Suite **249/249**.

### 30. Lane-boundary manual adjust + window click guard
- **Fixed a frame bug that broke boundary dragging:** the viewer emits the *display-frame* x, but
  `lanes.move_edge` works in the *corrected* frame; the drag callback now adds `kx`
  (`q + kx`) before moving, so a boundary lands where it was dropped instead of clamping to a neighbour.
- **Outer boundaries are now movable:** `lanes.move_edge` previously refused `index 0` / `last`; it now
  bounds them by `0` / `width` per §5.1, and the controller passes the crop region's right edge as the
  width. Added `move_edge` unit cases.
- **Clicks above the window no longer select:** lane toggles (left-click) and splits (middle-click) now
  require the press to be inside the two amber window rows (`_within_window`); clicks in the dimmed area
  outside do nothing.
- Verified on a real sample: dragging boundary 3 moved the edge 239→272 (+33 px = 15 screen px / scale
  0.451), and a click above the window left the order unchanged. Suite **252/252**.

### 31. Composite bounds: Y window, free outer edges, crop resize, hover cursor
- **Y bounds draggable:** the two amber window rows are now grabbable (`_hit_window_row` → `_window_drag`);
  dragging a row calls `Controller.set_window(side, qy)` which moves the crop's top/bottom edge without
  re-identifying the lanes (manual lane edits survive).
- **Outer lane boundaries are no longer pinned to the simple crop:** `move_edge` now bounds the outer
  boundaries by the **frame** width (`mask.shape[1]`), so they can be pushed past the crop's x edges.
- **The simple crop resizes to the composite bbox:** after any boundary move / window drag the controller
  runs `_sync_crop_to_composite()` — crop x = the outer lane edges (display frame, via the constant
  `_frame_kx()` offset), crop y = the window rows — so leaving composite mode keeps a matching crop.
- **Hover cursor:** `<Motion>` sets `sb_h_double_arrow` over a lane boundary, `sb_v_double_arrow` over a
  window row (or crop edge), `size_nw_se` over a rect-crop corner, else the default.
- Verified on a real sample: top-row drag moved the crop top (−110.7 → −70.8), the right outer boundary
  dragged past the crop (crop x1 157.4 → 246.4, last edge 643), cursors reported correctly.
  Suite **255/255**.

### 32. Rect-crop edges resize along their whole length
- `SingleViewer._hit_crop` now tests the four **edge segments** (whole length) before falling back, not
  just the 8 square handle points: corners still return `nw/ne/sw/se`, but the space between a corner and
  the midpoint handle now returns `n/s/e/w`. So hovering (and dragging) anywhere on the simple crop's
  outline shows the resize cursor and resizes that edge; the cursor is `sb_v_double_arrow` for the
  top/bottom, `sb_h_double_arrow` for the left/right, `size_nw_se` at the corners. Suite **256/256**.

### 33. Even widths: nudge clears bands more reliably
- `lanes.nudge_off_bands` now considers **every band edge** in the move range as a candidate
  (plus `low`/`high`), not just the two edges of the bands currently being cut. The §6.2 two-option
  rule oscillates when a boundary sits between two overlapping bands (each move lands back inside the
  other, e.g. `47↔48`), leaving the boundary intersecting. The new rule verifies the chosen candidate
  cuts **no** band, so it converges in one step.
- Measured on 8 real samples: cutting internal boundaries went **12 → 4** after "Even widths". The
  remaining 4 (on the dense 22-lane `20260403 35160.Tif`) are **genuinely blocked** — the whole
  `[neighbour+8, neighbour−8]` range is covered by bands (e.g. edge 21, x=1002, range [998,1034],
  covered by `(995,1040)`,`(1001,1044)`,`(1000,1042)`), which the spec leaves alone and the status
  line reports ("N boundary(ies) still cut a band"). Suite **258/258**.

### 34. Image-switch latency: coalesce viewer renders
- Profiled a `goto`: one switch triggered **5 full renders** — `reset_view` (set_image), `set_rotation`,
  `clear_crop`/`set_crop`, `set_overlays`, and `set_composite(None)` — each rebuilding the PIL
  `render_display` + a fresh `ImageTk.PhotoImage` + `canvas.delete("all")`. At 2× scale that was
  ~112 ms/switch.
- **Fix 1 — render batching:** `SingleViewer.batch()` is a context manager; `_render` sets a dirty flag
  while a batch is open and draws once at exit. `AppController._activate` wraps its whole viewer
  mutation block in `viewer.batch()`, so a switch is now **1 render**.
- **Fix 2 — skip the no-op:** `SingleViewer.set_composite` returns early when the data is unchanged
  (it was called with `None` on every switch, even with composite off).
- Verified at 2×: actual renders/switch **5 → 1**, latency **112 → 64 ms** (~43% faster), canvas still
  shows exactly one image. Remaining ~40 ms is the uncached display preprocess (`to_display_u8` +
  `content_anchor`), which was option 3 and not requested. Suite **264/264**.

### 35. Drag smoothness: borders move visually, calculations on release
- Rule from the user: "calculations only after the resize is completed — dragging only adjusts the
  borders visually." Measured before: simple crop drag **23 ms/motion**, composite window row
  **111 ms**, composite boundary **145 ms** (each motion did 2 full renders + 2–3 uncached
  `content_anchor` calls + per-motion controller work).
- `SingleViewer._render(overlay_only=True)` keeps the rendered image item (`self._img_item`) and
  redraws only the overlay items — no PIL recomposite, no `ImageTk.PhotoImage`, no recalculation.
  Every drag (`_drag_crop`, composite window row / boundary, new box) now moves the border with
  `_render(overlay_only=True)` and calls **no** controller callback; `_on_release` notifies the
  controller **once** and draws a full render (tint included).
- Measured after: simple crop **0.12 ms/motion**, composite window/boundary **~0.45 ms/motion**,
  **0 controller calls** during a drag, one ~25 ms release. Suite **265/265**.

### 36. Right-click pan: move the frame instead of re-rendering (option A)
- Cause (measured): every `<B3-Motion>` did a full re-render — **15 ms/motion** composite off,
  **34 ms/motion** composite on — because the view offset (`self.center`) is baked into the PIL
  resample, so the frame can't be reused.
- Fix: renders can now carry a **margin** (`_render(pad=…)`): the image is drawn at
  `(w+2·pad, h+2·pad)` with the anchor at `center+pad`, placed at `(-pad, -pad)` (overlay coords stay
  canvas-relative, so all overlay code is unchanged; `_composite_crop`/`_composite_tint` take the same
  offset). During a pan, the first motion renders the margin once; subsequent motions do
  `canvas.move("all", dx, dy)` + update `center` with **no re-render** until the pan outruns the margin.
  `_on_pan_release` re-centers with a canvas-sized frame (`pad=0`).
- Measured after: small pan **1.7 ms/motion**, large pan **6.6 ms/motion** (composite on ~2.3),
  vs 15–34 before; normal switch/render path is `pad=0` and **unchanged** (60–92 ms, dominated by the
  uncached preprocess). Suite **266/266**.

### 37. Smaller exe (step 1) + licensing
- **Distribution step 1 (size):** `collect_submodules('geltilt')` was force-bundling the retired
  `smears.py`, whose only dependency is `scipy` (~30 MB). Added
  `--exclude-module geltilt.smears --exclude-module scipy` to `build_app_exe.py` (and
  `excludes=[...]` in `geltilt-app.spec`); `projection` is still used, so it stays. Verified a
  model-less + scipy-less build runs (`--selftest` ok). Rebuilt with the model still bundled:
  **156 → 124 MB** (−32 MB).
- **Licensing:** added `LICENSE` (MIT — placeholder holder "geltilt contributors", edit me),
  `THIRD_PARTY_NOTICES.md` (Python/PSF, Tcl/Tk, NumPy BSD-3, Pillow MIT-CMU, OpenCV Apache-2.0,
  PyInstaller GPL-2.0-with-exception, GelGenie Apache-2.0 + provenance notes), and `licenses/`
  with the upstream texts copied from the wheels. New `geltilt/app/about.py` +
  an **About / Licenses** button in the right column; both files and `licenses/` are bundled by the
  build. Tests in `tests/app/test_about.py`. Suite **268/268**.
- **`.gitignore`:** added per-run state (`app_session.json`, `app_runs.jsonl`, `labeler_session.json`,
  `projview_state.json`, `sample_labels.json`), `band_mask_diag/`, `_measure/`, and `*.onnx`.

---

## Next up
- Distribution steps 2 (don't bundle the model; download on first run with progress) and 3 (UPX).
- Composite crop: a UI acceptance pass on the two-pane viewer + drag hit-feel polish.
- Commit the ladders integration + BAND_DETECTION front end + composite crop.