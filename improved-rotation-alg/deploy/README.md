# Gel lane tilt — integration instructions

Estimates the single rotation that makes the lanes of a gel image vertical, from band centroids.

**What this is.** One function, `ladders.estimate(xy)`, that takes detected band positions and
returns an angle in degrees. It reads no pixels, so it is independent of which segmentation model
produced the detections.

**What this is not.** It does not segment bands, does not rotate the image, does not assign every
band to a lane, and does not measure intensity.

---

## 1. Files

| file | what it is |
|---|---|
| `ladders.py` | the core module. **Dependency: numpy only.** Copy this into your project. |
| `SPECIFICATION.md` | the full algorithm specification: parameters, gates, scoring, the final estimator, output contract, and the measured validation (including what was tried and rejected). |
| `example.py` | runnable self-contained example on a synthetic gel with a known tilt. |
| `README.md` | this file. |

Nothing else is required to run it. There is no model file, no config file, no data file.

## 2. Install

```bash
pip install numpy
cp ladders.py <your project>/
```

Python 3.8+. Verified on numpy 2.x. No GPU, no compiled extensions.

Smoke test:

```bash
python example.py
```

Expected: `angle to apply : +1.935 deg   (truth +2.000, error -0.065)` and two lanes found. The
numbers will shift slightly if numpy changes its RNG stream defaults; the error should stay well
under 0.3 deg.

## 3. Interface

```python
import ladders

res = ladders.estimate(xy, params=None)
```

### Input — `xy`

An `(N, 2)` float array of band centroids, **in pixels, in the image's own coordinate frame**:
`(x, y)` with x rightward, **y downward** (the usual image convention). One row per detected band.

Both a list of pairs and a numpy array are accepted; it is passed through `np.asarray(..., float)`,
non-finite rows are dropped, and the input is not modified.

The coordinate frame matters: **the angle is scale-invariant but the parameters are not.** All
thresholds are in pixels (`min_span=30`, `dx_tol=4`, `max_x_std=5`, `min_gap=4`, ...). If your
segmenter ran on a resized image, you may either

* scale the centroids back to the original pixel frame before calling, or
* call with the resized-frame centroids and pass `params={'min_span': 30*s, 'dx_tol': 4*s,
  'dx_tol_wide': 12*s, 'max_x_std': 5*s, 'max_x_dev': 8*s, 'min_gap': 4*s, 'min_span_warn': 70*s}`
  where `s` is the resize factor,

then apply the returned angle to **the image whose frame you used**.

### Output — a dict

| key | type | meaning |
|---|---|---|
| `ok` | bool | **check this first.** An angle was produced |
| `reason` | str | human-readable status; on `ok=False`, why it refused |
| `angle` | float | **the answer**, degrees, rotation to APPLY |
| `se` | float | calibrated uncertainty of `angle`, degrees |
| `warnings` | list[str] | non-vetoing quality notes |
| `n_bands` | int | detections supplied (after dropping non-finite rows) |
| `n_ladders` / `kept` | int | lanes found / lanes pooled into the answer |
| `n_rungs` | int | rungs in the pooled lanes |
| `fallback` | bool | the answer rests on lanes of fewer rungs than `min_rungs` |
| `overdispersion` | float | `≥1`; `>>1` means the lanes disagree by more than their own errors allow |
| `ladders` | list[dict] | every lane, best trustworthiness first; each carries `n`, `span`, `pitch`, `angle`, `se_deg`, `se_deg_cal`, `consistency`, `trust`, `r2`, `half_gap`, `idx` (indices into `xy`) and more |
| `angle_rungs`, `angle_invvar`, `angle_trust_weighted`, `angle_raw` | float | the alternative estimators and the pre-correction value; see §6 |
| `slope` | float | `tan(-angle)`, for callers who want the slope |

If you only need a number:

```python
res = ladders.estimate(xy)
angle = res['angle'] if res['ok'] else 0.0     # 0.0 = leave the image alone
```

### Angle convention — this is the part to get right

`angle` is the value to pass **directly** to OpenCV, in the `cv2.getRotationMatrix2D` convention:
**positive = counter-clockwise on screen**, with y pointing down.

```python
import cv2
h, w = image.shape[:2]
M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), res['angle'], 1.0)
corrected = cv2.warpAffine(image, M, (w, h))
```

Equivalently it is `-atan(s)` for a lane of slope `s = dx/dy`. If your transform library takes the
opposite sign convention, negate it — and **verify with the example before trusting it**, because a
sign error looks plausible (a small tilt either way) and will silently double your error.

`cv2.getRotationMatrix2D` rotates about the image centre, which at a few degrees pulls the corners
in and leaves a dark border. That is the transform, not the estimator.

## 4. Do not ignore `ok`

Coverage on the 94 hand-labelled gels is **92/94 (98%)**; two images refused. A refusal is a normal
outcome, not an error condition — it means no lane-like column of ≥2 rungs was found. Decide what
your application does: leave the image unrotated, prompt the user, or fall back to a different
method. **Do not** treat `angle` as valid when `ok` is False.

Recommended use of the quality signals:

| signal | reading |
|---|---|
| `kept == 1` | one lane only, no independent consensus — the weakest class of answer |
| `se` | compare against your tolerance; `se > 1.0` deg means the answer is not precise |
| `overdispersion > 2` | the lanes disagree more than their errors allow; treat with suspicion |
| `fallback == True` | few-rung lanes supplied the answer (see `warnings`) |
| `warnings` | never vetoes; they explain weakness |

Empirically, answers that pooled **2 or more** lanes averaged 0.630 deg error, versus 0.782 deg for
single-lane answers. If your workflow can supply more detected bands, that is the highest-leverage
improvement on the input side.

## 5. Performance

* **Cost** is driven by the number of detections, `N`. Candidate hypotheses are generated from
  pairs of detections (`O(N²)`), capped at `pair_max_n = 140` live detections per round, and up to
  `max_ladders = 8` rounds. A 94-image batch over cached centroids runs in seconds on CPU.
* **Deterministic.** No RNG anywhere: identical input and parameters give identical output.
* **Thread safety.** The module holds a module-level parameter dict `ladders.P`. `estimate(params=...)`
  applies overrides for the call only and restores `P` in a `finally`, but two threads calling
  `estimate` concurrently with different `params` can interleave. The default path is read-only and
  safe; if you need per-call parameters from multiple threads, give each thread its own loaded copy
  of the module (e.g. `importlib`) or serialise the calls.

## 6. Which number to use

`angle` is the primary output and is what all quoted validation numbers use: a **pooled shared-slope
fit** over the kept lanes' rungs, with one free x-offset per lane.

The alternatives are exposed because they are useful diagnostics, not because you should pick one:

| field | what it is | when it matters |
|---|---|---|
| `angle_rungs` | the same as `angle` unless you change `primary_estimator` | — |
| `angle_invvar` | inverse-variance random-effects pool of the per-lane angles, with MAD outlier rejection and a Cochran-Q overdispersion inflation | use if you want the estimator that propagates a per-lane uncertainty model |
| `angle_trust_weighted` (`contrast`) | trust-weighted mean of the lane angles | an independent cross-check on `angle`; a gap between the two means the lanes disagree |
| `angle_raw` | `angle` before any empirical over-rotation correction | informational; the correction is off unless you enable it |

A large gap between `angle` and `angle_invvar` is itself a signal that the lanes disagree. Measured
over the 94 gels, `angle_rungs` was better on every aggregate measure, which is why it is the
default.

## 7. Tuning

All parameters live in `ladders.P` and are documented in `SPECIFICATION.md` §3. Pass overrides per
call:

```python
res = ladders.estimate(xy, params={'min_rungs': 3, 'agree_tol': 2.0})
```

Useful ones:

| parameter | default | raise it when |
|---|---|---|
| `min_rungs` | `4` | detections are noisy and there are many of them |
| `fallback_min_rungs` | `2` | **lower** it (to 3) if you would rather refuse than answer from 2 bands |
| `agree_tol` | `1.5` | gels are warped, so lanes genuinely disagree |
| `max_tilt` | `8.0` | your gels really do exceed 8 deg (it is a sanity guard, not a limit of the maths) |
| `max_ladders` | `8` | very wide gels with many lanes |
| `pair_max_n` | `140` | **lower** it for speed on dense images, at the cost of recall |
| `se` consumers | — | `lane_se_scale` and `centroid_sigma_px` calibrate `se`; they are empirically fitted on the 94-gel set and should be re-derived for a different scanner/model |

**Do not tune against a handful of images.** The parameters interact, and this project's own
experience is that several plausible changes (slope-aware gating, more lanes, merging band
fragments, tolerating missed bands) all measured *worse* on the labelled set. See
`SPECIFICATION.md` §14 for those results before changing a default.

## 8. Integration checklist

1. Confirm your centroids are in the image frame you will rotate (§3).
2. **Verify the sign convention** with `example.py` and one real image: correct a gel, then look at
   whether the lanes are vertical. A sign error is the most likely integration fault.
3. Handle `ok == False` explicitly (§4).
4. Decide a threshold on `se` (or on `kept`) for "answer is good enough to auto-apply".
5. If your segmenter has a threshold to trade precision for recall, prefer **more bands**: the
   single biggest driver of accuracy here is having ≥2 lanes to pool.
6. Re-derive `lane_se_scale` / `centroid_sigma_px` if your band detector differs from the one these
   were fitted on, so `se` means what it says.

## 9. Measured performance

94 hand-labelled gels, end to end (ONNX band segmentation → centroids → `estimate`), compared with
hand-annotated ground truth:

| metric | value |
|---|---|
| coverage | 92/94 (98%) |
| mean \|error\| | 0.709 deg |
| median \|error\| | 0.646 deg |
| within 1 deg | 75% |
| within 2 deg | 100% |
| worst case | 1.93 deg |
| reported `se` | 0.943 deg (mean) — the error actually made is 0.75 × the reported `se` |
| answers from ≥2 lanes | 0.630 deg mean, 44 of 92 images |
| answers from 1 lane | 0.782 deg mean, 48 of 92 images |

`se` is calibrated so that `mean|error| / mean(se) ≈ 0.8`, the value a well-calibrated normal gives.
Treat `se` as a real standard deviation, not as a confidence score.

**Known limitations**, in the order they are likely to bite (full list in `SPECIFICATION.md` §14):

* **Lane curvature.** Gels where a lane bends cannot be made vertical by any rigid rotation; the
  estimator returns a straight-line fit and no cheap per-lane feature reliably flags this.
* **Single-lane answers** are the bulk of the error. They arise when only one column has enough
  rungs — usually a band-starved image.
* **The top and bottom halves of a two-row-of-wells gel** are kept separate by splitting at gaps
  that are not near-integer multiples of a lane's own spacing. This is robust but not infallible.
* **A systematic over-rotation remains**: fitted over the labelled set, `angle ≈ 0.117 + 1.228·GT`.
  A correction is implemented (`bias_slope`, `bias_offset`) and ships **disabled**, because it is
  fitted on the same set it would be applied to.
