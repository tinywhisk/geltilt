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

## Next up
- **Feature 2: lane-based cropping** — detect lane boundaries on the straightened gel and export per-lane crops.