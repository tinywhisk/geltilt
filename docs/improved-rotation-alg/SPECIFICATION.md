# Gel lane tilt — specification

**Source of truth:** [`ladders.py`](ladders.py) (no external deps beyond NumPy).
**Status:** implemented and validated. This document specifies the algorithm *as implemented*.

**Design premise.** A ladder and a loaded sample lane are the same structure: a column of rungs
sharing one x, leaning together with the gel. There is no innate difference between them beyond how
many rungs were detected. The module therefore detects **lanes**, scores each one by **how precisely
it pins its own angle**, and combines them with an **inverse-variance random-effects** estimator
that reports a calibrated uncertainty.

**Angle convention:** a lane is a line `x = a + s·(y − ȳ)` with slope `s = dx/dy`. The **reported angle is `θ = −atan(s)` in degrees**, which is the value to pass to `cv2.getRotationMatrix2D` (positive = counter-clockwise on screen, y downward) so that the lane becomes vertical.

---

## 1. Scope and I/O contract

**Input:** `xy` — an `(N, 2)` float array of band centroids in image pixels, `(x, y)`, from a band segmentation model (GelGenie U-Net mask → connected components). No image pixels are used anywhere in this module.

**Output:** one dictionary per image (§10) containing the rotation angle and diagnostics; plus the list of accepted lanes and their per-lane statistics.

**What it is for:** estimating the single rigid rotation that makes the lanes of a gel vertical, from the *columnar* structure in the detections. A column is either a **ladder** (a size series: many evenly spaced rungs at one x) or a **sample lane** (the same structure arising from the experiment's band pattern). Ladders are tried first only because sample lanes usually carry too few bands to fit — they are not the only valid structure.

**Non-goals:** it does not rotate the image, does not assign every band to a lane, and does not measure band intensity.

---

## 2. Notation

| symbol | meaning |
|---|---|
| `N` | number of detections in the image |
| `p`, `idx` | a candidate rung set and its indices into `xy` |
| `span` | `max(y) − min(y)` over a rung set |
| `gaps` | consecutive differences of the sorted `y` of a rung set, keeping only positive values |
| `pitch` | `median(gaps)` |
| `cv` | `std(gaps) / pitch` (spacing irregularity) |
| `lever` | `sqrt(Σ (y − ȳ)²)` over the rung set — the angular lever arm |
| `rms_perp` | RMS perpendicular distance of the rungs to the fitted line |
| `se_deg` | per-lane slope standard error in degrees (§8) |
| `se_deg_cal` | `se_deg` scaled to be comparable with the error actually made (§8) |
| `trust` | per-lane trustworthiness score (§8) |
| `x_std` | `std(x)` over the rung set (raw) |
| `x_std_line` | RMS of the rung residuals about the fitted line |
| `x_dev` | `max abs(x − x̄)` (raw) |
| `x_dev_line` | `max abs(x − fitted line)` |
| `elong` | `span / (x_std + 0.5)` — the elongation heuristic |
| `θ_i` | per-lane angle `−atan(s_i)` in degrees |
| `w_i` | inverse-variance weight `1 / se_deg_cal_i²` |
| `phi` | Cochran overdispersion factor `max(1, Q/dof)` (§10) |

---

## 3. Parameters (`P`, all tunable)

| parameter | default | unit | role |
|---|---|---|---|
| `min_rungs` | `4` | rungs | minimum rungs for a lane in the **primary** pass |
| `fallback_min_rungs` | `2` | rungs | minimum rungs in the **fallback** pass |
| `dx_tol` | `4.0` | px | how far a rung may sit from the fitted line (strict membership) |
| `dx_tol_wide` | `12.0` | px | first-pass tube width (seed anchor is only approximate) |
| `slope_hi` | `0.20` | tan | slope hypotheses swept in ±this (`0.20` ≈ 11.3°) |
| `n_slope` | `41` | count | number of slope hypotheses |
| `pair_min_dy` | `10.0` | px | min y separation of a seeding pair |
| `pair_max_dx` | `12.0` | px | max x separation of a seeding pair (same lane, not a row) |
| `pair_max_n` | `140` | count | skip pair seeding above this many live detections (cost) |
| `max_seeds` | `4000` | count | cap on pair-seeded hypotheses per extraction round |
| `min_rungs_prior` | `3` | rungs | rungs needed before the core x-cluster is cut and refitted |
| `max_rounds` | `3` | passes | tube→core→refit iterations per candidate |
| `min_span` | `30.0` | px | reject shorter rung sets |
| `min_gap` | `4.0` | px | reject rungs sitting this close (double detections) |
| `split_ratio` | `3.2` | ×pitch | reject when a single gap exceeds this multiple of the pitch |
| `min_elong` | `3.5` | ratio | reject unless `span/(x_std+0.5) ≥` this |
| `max_x_std` | `5.0` | px | reject wider lanes (measured on **raw x** by default — see `gate_raw_x`) |
| `max_x_dev` | `8.0` | px | reject a lane whose worst rung is this far off |
| `gate_raw_x` | `True` | bool | `max_x_std`/`max_x_dev` measured on raw x; `False` measures about the line |
| `max_pitch` | `None` | px | **no pitch ceiling**; set a number to re-enable the old ladder-density gate |
| `max_frac` | `0.9` | fraction | a lane may not claim more than this fraction of all detections |
| `max_frac_min_bands` | `12` | bands | **below this many detections `max_frac` is not applied** (§14.19) |
| `max_ladders` | `8` | lanes | stop extracting after this many (a hard cap — see §14.17) |
| `min_score` | `6.0` | score | stop when the best remaining candidate scores below this |
| `agree_tol` | `1.5` | deg | lanes within this of the **median** angle are pooled in |
| `max_tilt` | `8.0` | deg | refuse when the pooled angle exceeds this |
| `r2_warn` | `0.90` | — | line R² below this raises a **warning only** |
| `min_span_warn` | `70.0` | px | longest lane shorter than this raises a **warning only** |
| `core_win` | `15.0` | px | **declared but inert** (§15) |
| `curve_ref_px` | `1.0` | px | rung-to-line width counting as "straight" in `trust` |
| `curve_weight` | `0.4` | — | weight of the straightness term in `trust` |
| `gap_consistent_ratio` | `3.0` | × | a lane's longest/shortest spacing beyond this is penalised (§14.9) |
| `gap_penalty_power` | `1.0` | — | exponent of that penalty |
| `missed_band_ratio` | `1e9` | ×pitch | **disabled**; tolerated near-multiple gaps measured worse (§14.10) |
| `missed_band_tol` | `0.30` | fraction | tolerance for that (inert while disabled) |
| `lane_se_scale` | `5.5` | × | measured factor making `se_deg` comparable with the error made |
| `centroid_sigma_px` | `0.5` | px | 1σ band-centroid localisation error — the only precision info a 2-rung lane has |
| `lane_se_floor_deg` | `0.05` | deg | numerical guard on a degenerate `se_deg` |
| `mad_k` | `3.0` | MAD | reject a lane beyond this many MAD of the median angle |
| `mad_floor_deg` | `0.35` | deg | never reject for less than this deviation |
| `primary_estimator` | `'rungs'` | — | `'rungs'` or `'invvar'` — which estimator is reported as `angle` |
| `phi_warn` | `2.0` | — | overdispersion above this raises a **warning only** |
| `bias_slope` | `0.0` | deg/deg | optional empirical over-rotation correction (**off**) |
| `bias_offset` | `0.0` | deg | optional empirical over-rotation offset (**off**) |

---

## 4. Lane record

Every accepted lane carries:

```
idx (indices into xy)        n (rung count)        span, pitch, max_gap
x_std, x_std_line, x_dev, x_dev_line, lever        elong, gap_cv
rms, rms_perp, r2            se_deg, se_deg_line, se_deg_cal, half_gap
s, angle, ybar               score, trust
```

`se_deg` is the per-lane slope standard error expressed in degrees (§8), i.e. *how precisely this one line is pinned*. `trust` is the trustworthiness score (§8). `half_gap` is a **diagnostic only** — it is carried but not scored on (§14).

---

## 5. Stage A — hypothesis generation (per extraction round)

Let `A` be the indices still alive (not yet claimed) with `|A| ≥ min_rungs`; `sub = xy[A]`.

1. **Pair seeds** (only if `|A| ≤ pair_max_n`): for every ordered pair `(i, j)` with
   `|y_j − y_i| ≥ pair_min_dy` and `|x_j − x_i| ≤ pair_max_dx`, form the slope
   `s = (x_j − x_i)/(y_j − y_i)`; keep it when `|s| ≤ slope_hi`.
   A pair of rungs on one column defines that column's slope exactly, whereas a slope sweep only hits the true slope by luck — and an unhit slope steers the tube onto a *diagonal slice* of the gel.
   The hypothesis line is forced through the pair (`fix0 = True`) so a distant singleton sharing the tube cannot hijack the first fit.
2. **Slope sweep seeds:** additionally, one hypothesis per grid slope, anchored at the centroid of `sub`. This covers columns whose rungs were poorly detected.
3. **Cap:** if the pair list exceeds `max_seeds`, subsample it uniformly.

---

## 6. Stage B — candidate refinement

`_candidate(sub, x0, y0, s0, a0, yb0, fix0)`:

```
ybar, a, s, xc  ←  (yb0 or y0), (a0 or x0), s0, a
repeat max_rounds times:
    tol  ← dx_tol_wide on the first pass, dx_tol afterwards
    ton  ← {i : |x_i − (a + s·(y_i − ybar))| ≤ tol}          # inlier tube
    core ← _core(sub, ton, xc)                               # tightest x-cluster around anchor
    if |core| < 2: reject
    s'   ← s            if (fix0 and first pass)  else  slope_of(core)
    a'   ← mean(x_core − s'·(y_core − ybar))
    ybar'← ybar         if (fix0 and first pass)  else  mean(y_core)
    xc   ← a' + s'·(median(y_core) − ybar')
    stop early when |s' − s| < 1e-7 after the first pass
idx ← {i : |x_i − (a + s·(y_i − ybar))| ≤ dx_tol}            # strict membership
idx ← _largest_run(sub, idx, s, a, ybar)
if |idx| < min_rungs: reject                                  # (fallback pass: 3)
s   ← slope_of(sub[idx])                                      # refit on the final rung set
return _evaluate(sub, idx, s)
```

**`_core(sub, ton, xc, s, ybar, a)`** — the rung column inside the tube: sort `ton` by **perpendicular distance to the tube line**, `|x − (a + s·(y − ybar))| / sqrt(1 + s²)`; cut at the largest *relative* jump in those distances (a jump `> 2.5 ×` the median of the preceding gaps, provided at least `min_rungs_prior` nearest points survive); then split the winner by y-gaps `> split_ratio × median(gap)` and keep the largest uninterrupted run. The cut is relative rather than a fixed width because the right width differs per gel (a dense lane spreads over ~2 px of x, another over ~12 px). Measuring in the perpendicular direction rather than raw `|x − xc|` matters because raw x counts `|x − line| / cos θ`, so it over-penalises the far end of a leaning lane — exactly the rungs carrying the slope. Measured on the 94 gels it is very slightly **better** than the raw cut (0.698 vs 0.701 deg), so it is the default.

**`_largest_run(sub, idx, s, a, ybar)`** — filter to `|x − line| ≤ dx_tol`, sort by `y`, split where `Δy > split_ratio × median(positive gaps)`, return the largest run. A lane is one *uninterrupted* column: anything past a big hole is a second structure and may not be averaged into the line. This is also what keeps the top and bottom halves of a two-row-of-wells gel from being merged into one lane (§14).

---

## 7. Stage C — acceptance gates (`_evaluate`)

Evaluated in this order; the first failure rejects the candidate.

| # | condition to **reject** |
|---|---|
| 1 | rung set empty |
| 2 | `span < min_span` (30 px) |
| 3 | no positive gaps |
| 4 | `pitch < min_gap` (4 px) — rungs on top of each other |
| 5 | `max(gap) > split_ratio × pitch` (3.2×) — a hole, i.e. two structures |
| 6 | `elong < min_elong` (3.5) |
| 7 | `x_std > max_x_std` (5 px) **or** `x_dev > max_x_dev` (8 px) — a diagonal band row, not a lane |
| 8 | `max_pitch` is set **and** `pitch > max_pitch` |

Gate 7 uses **raw x** by default. Setting `gate_raw_x=False` measures it about the fitted line instead, which is slope-blind and admits leaning lanes; measured on the 94 gels the line version is **worse** (0.707 vs 0.698), so raw is the default. See §14.1 — this is a real limitation, not a tuning choice.

If all pass, the lane is accepted and scored.

---

## 8. Stage D — per-lane geometry, trustworthiness and uncertainty

`lane_geometry(p)` measures one lane, always about the **fitted line**:

```
s, a, ȳ   = fit_line(p)                            # weighted LSQ x = a + s(y − ȳ)
resid     = x − (a + s·(y − ȳ))
lever     = sqrt(Σ (y − ȳ)²)
rms_perp  = sqrt( Σ (resid/sqrt(1+s²))² / dof )     # perpendicular, the correct residual
se_slope  = rms_perp / lever
se_deg    = degrees( se_slope / (1 + s²) )         # linearised slope → angle
r2        = 1 − SS_res/SS_tot
```

`se_deg` is the actionable precision number and combines the rung scatter with the lever arm (rung count × span). `r2` is easy to misread: a genuinely near-vertical lane leans only ~0.5 px over its height, so 0.3 px of jitter can drive `r2` to nearly zero while the implied angle is still within a fraction of a degree. `half_gap` (upper-half slope vs lower-half slope) is recorded as a diagnostic but **not** scored — measured r = +0.08 against per-lane error (§14).

**`trust` — the trustworthiness score.** One quantity is estimated: how precisely this lane pins its own angle.

```
precision    = lever / (rms_perp + curve_ref_px)
straightness = exp(−curve_weight · rms_perp / lever)
consistency  = 1                                if max_gap / min_gap <= gap_consistent_ratio
             = (gap_consistent_ratio / r) ** gap_penalty_power      otherwise, r = max_gap/min_gap
trust        = precision · straightness · consistency
```

`precision` rises with a longer lever arm and a tighter line; `curve_ref_px` (1 px) only stops a zero-scatter lane scoring infinite confidence. `straightness` penalises a line whose scatter grows with its own length, which is no longer a line. Both terms are in px/px, so there is nothing fitted.

`consistency` is the spacing check: a lane is ONE series, so its own spacings must be mutually similar. The reference is the **shortest** spacing, because that is the only quantity a stitched lane cannot fake — adding a distant well row adds a large gap, never a small one, whereas the median is dragged around by the very gaps under test. This is the fix for lanes that span several well rows (below): in `20260403 35160.Tif` the spanning lane has gaps `367 24 302` and is penalised to 0.234, which drops it from rank 0 to rank 3 behind the three real lanes (which score `consistency = 1.0`).

This deliberately **replaces** `elong · exp(−0.3·cv)`: spacing regularity and elongation were measured to be poor proxies (`cv`: r = −0.02 against per-lane error), and they encoded a ladder prior — long, dense, regular — that a valid sample lane need not satisfy. `score` is still computed and stored for continuity, but it was measured to be no better than `trust` (lane-level mean |err| 0.774 vs 0.775), so `trust` is what ranks and weights.

**`se_deg_cal` — the calibrated uncertainty.**

```
a_term = se_deg · lane_se_scale                                  # already in degrees
b_slope = centroid_sigma_px · sqrt(2) / lever                    # slope units
b_term = degrees( b_slope / (1 + s²) )                           # same linearisation as se_deg
se_deg_cal = hypot(a_term, b_term) + lane_se_floor_deg
```

Two independent shortfalls are covered, and they must be converted to the same unit system before
being combined — taking them in quadrature first mixes a degrees term with a slope term and
inflates the result by the `degrees()` factor (a bug that briefly produced `se` ≈ 43°).

**(a) Fitted residual vs error made.** A least-squares residual cannot see error that is not in the
residuals: dropping one or two weak rungs moves a 4–7 rung lane by 0.5–1.2° while its own `se_deg`
stays near 0.1, and every lane in an image shares the gel's warp. Measured over 94 gels that
shortfall is close to a constant **factor**: per-lane error/se is 2.55 (3–4 rungs), 2.88 (5–6) and
4.96 (7+), giving `lane_se_scale ≈ 4.5–5.5`. A corollary worth stating: **a lane's rung count does
not protect it** — 7-rung lanes are as wrong as 4-rung ones (0.69° vs 0.77°), they are just more
confident about it.

**(b) A line through two rungs has zero residual by construction.** `se_deg` is exactly 0, so (a)
alone reports `se ≈ 0.05°` for a lane that is wrong by 1.1°. Its precision is instead set by how
well the two centroids are *localised*: an x error of `centroid_sigma_px` over a lever arm `lever`
is a slope error of `centroid_sigma_px·√2 / lever`. This makes the reported uncertainty fall as
`1/lever` — 1.06° for a 40 px two-rung lane, 0.25° for a 200 px one — which is the physically right
behaviour and the only precision information available in that case. `centroid_sigma_px = 0.5` is
calibrated against the error actually made; larger values fit the two-rung cases better but wreck
the global calibration (|err|/se falls to 0.28 at σ = 3).

---

## 9. Stage E — greedy extraction and masking

```
slopes ← linspace(−slope_hi, slope_hi, n_slope)
alive  ← all True
repeat up to max_ladders times:
    if |alive| < min_rungs: stop
    generate hypotheses (§5) from the alive detections
    best ← argmax trust over _candidate(...) for every hypothesis,
           rejecting any candidate with |idx| > max_frac · N      # a lane is one lane, not half the gel
    if best is None or best.trust ≤ min_score: stop
    emit best; alive[best.idx] ← False                             # mask its rungs
```

Greedy score-and-mask, not joint optimisation: the highest-scoring lane is removed before the next is searched, which keeps two overlapping-but-distinct lanes from splitting each other's rungs.

`detect_ladders(..., slope_window=(c, half))` optionally narrows the slope hypotheses to `c ± half` (tan units). Every lane in a gel shares one tilt, so once a first pass has measured it there is no reason to keep hypothesising 11° diagonals — this makes short, sparse lanes findable that would never win the open sweep. It is a recall aid and is **off by default**: on the 94 gels raising recall beyond the strongest lanes measured *worse* (§14.4).

---

## 10. Stage F — consensus selection and the final angle

```
primary ← detect_ladders(xy, min_rungs = P.min_rungs)              # 4
if primary is empty:
    lanes ← detect_ladders(xy, min_rungs = P.fallback_min_rungs)   # 3
    fallback ← True
else:
    lanes ← primary ; fallback ← False
if lanes is empty: refuse('no lane-like rung column found')

θ_med ← median{ θ_i }                                              # ROBUST reference
keep  ← { L ∈ lanes : |θ_L − θ_med| ≤ agree_tol (1.5°) }
```

The reference is the **median of all detected lanes**, not the top-scoring one. Anchoring the vote on one lane lets that lane decide both the answer and who is allowed to disagree with it, so a wrong winner builds a wrong consensus around itself.

Two estimators are computed over `keep`, and both are reported:

**(a) `angle_rungs` — pooled shared-slope fit** (one free x-offset per lane, one common slope). This is the default `angle`:

```
num = Σ_lanes Σ_rungs (x − x̄_lane)(y − ȳ_lane)    den = Σ_lanes Σ_rungs (y − ȳ_lane)²
s   = num / den                                    angle_rungs = −degrees(atan(s))
```

**(b) `angle_invvar` — inverse-variance random-effects pool of the per-lane angles.** Reported as `angle_invvar`, and `angle` when `primary_estimator='invvar'`:

```
w_i   = 1 / se_deg_cal_i²
θ̂     = Σ w_i θ_i / Σ w_i                          after MAD outlier rejection
Q     = Σ w_i (θ_i − θ̂)² ,  dof = k − 1            (Cochran's Q)
phi   = max(1, Q/dof)                              (1 = lanes agree with their own errors)
se    = sqrt(phi / Σ w_i)
```

**Robust rejection.** Weighted least squares has an unbounded influence function, so one lane off by degrees (a merged structure, a band row that slipped a gate) would be weighted by its own small `se` and drag the answer. Lanes beyond `mad_k` MAD of the median angle are dropped, iterated to convergence, and never rejected for less than `mad_floor_deg`.

**Overdispersion (`phi`) — the correction the old consensus lacked.** If the lanes disagree by *more* than their individual `se` predicts, the excess is real structure that no amount of averaging removes, so the error bar must widen instead of the disagreement being averaged away.

Which one is primary is a **measured** choice, not a preference: over the 94 labelled gels `angle_rungs` beat `angle_invvar` on every measure (mean |err| 0.698 vs 0.706, within 1° 77% vs 73%), so `primary_estimator='rungs'`. The `se` reported is **always** from the lane-level model (b), never from the rung fit's own `se`: a rung-level residual understates the error by `lane_se_scale` and cannot express that lanes err together.

**Guard:** if `abs(angle) > max_tilt` (8°) the estimate is refused with reason *"implausible tilt … grouping suspect"* (with `" (3-rung fallback)"` appended when the fallback pass supplied the lanes).

**Optional empirical correction.** `bias_slope` / `bias_offset` (both `0.0`, i.e. **off**) apply `angle −= bias_offset + bias_slope·|angle|`. The measured over-rotation is `≈ 0.117 + 0.228·|GT|` (§13), so this could remove ~0.09° of mean error — but it is fitted on the labelled set, so it ships disabled and any value set is reported in `warnings`.


---

## 11. Stage G — fallback pass, warnings, refusal reasons

- The fallback pass relaxes **only** `min_rungs` (4 → 2). It is attempted **only** when the primary pass yields nothing, so weak and strong evidence never mix in one average. A two-rung lane answered the two hardest accepted images (a 9-band gel at 1.68° error and a 21-band gel at 1.06°), which the primary pass alone could not. It is reported with the `1/lever` uncertainty of §8, which is what keeps it outvoted whenever a 4-rung lane exists.
- `fallback=True`, `min_rungs_used`, and a warning are attached, naming the rung counts actually in force: *"2-rung fallback: no lane with >=4 rungs in this image; a 2-rung column is a short lever arm (angle se ~X deg, N rungs, Y px span)"*.
- Additional warnings (never vetoes): per-lane `r2 < r2_warn`; `max(span) < min_span_warn`; `phi > phi_warn` (lanes disagree beyond their own errors); a **single-lane** answer ("no independent consensus, the answer is that lane alone"); and the empirical over-rotation correction when it is enabled.
- Refusal reasons: `fewer than 4 bands`; `no lane-like rung column found`; `implausible tilt …`; and *"lanes disagree by more than 1.5 deg about the median …"* is recorded as `reason` **in addition to** a successful `ok=True` when some lanes were dropped by `agree_tol`.

---

## 12. Output contract

| field | type | meaning |
|---|---|---|
| `ok` | bool | an angle was produced |
| `angle` | float° | **primary** answer (§10): `angle_rungs` unless `primary_estimator='invvar'` |
| `se` | float° | **calibrated** uncertainty of `angle`: always from the lane-level model (§8), NOT the rung fit |
| `angle_rungs` | float° | pooled shared-slope fit over kept lanes' rungs |
| `se_rungs` | float° | that fit's own se — reported for comparison, **not** an error bar |
| `angle_invvar` | float° | inverse-variance random-effects pool of the lane angles |
| `se_invvar` | float° | its se, including the Cochran overdispersion inflation |
| `angle_trust_weighted`, `contrast` | float° | trust-weighted mean of the lane angles, an independent cross-check |
| `angle_raw` | float° | `angle` before the optional over-rotation correction |
| `overdispersion` | float | `phi = max(1, Q/dof)`; ≈1 means lanes agree with their own errors |
| `slope` | float | `tan(−angle)` |
| `ladders` | list | every extracted lane, best trust first (per-lane record, §4) |
| `n_ladders` | int | lanes extracted |
| `kept`, `lanes_used` | int | lanes within `agree_tol` of the median; lanes surviving MAD rejection |
| `fallback` | bool | the 3-rung pass supplied the lanes |
| `min_rungs_used` | int | 4 or 3 |
| `n_bands`, `n_rungs` | int | detections in the image; rungs in the kept lanes |
| `best_trust`, `best_score` | float | trustworthiness of the strongest lane |
| `span_max`, `x_std_max`, `pitch_max` | float | summary of the kept lanes |
| `r2_min`, `se_deg` | float | worst kept-lane R²; RMS of per-lane `se_deg_cal` |
| `warnings` | list[str] | non-vetoing quality notes |
| `reason` | str | human-readable status |

**Primary / secondary signal semantics** (as drawn in `final/`): *primary* = rungs of lanes from the ≥`min_rungs` pass that were pooled into the answer; *secondary* = rungs of 3-band lanes (fallback pass) or of detected-but-unpooled lanes.

---

## 13. Validation (measured, 94 hand-labelled gels)

The 94 samples were run end-to-end: ONNX band segmentation → centroids → `ladders.estimate`,
compared against `ground_truth.json` by [`run_eval.py`](run_eval.py).

| configuration | coverage | mean \|diff\| | median | ≤0.5° | ≤1° | ≤2° | max |
|---|---|---|---|---|---|---|---|
| **total (all accepted)** | **92/94** | **0.709°** | **0.646°** | 38% | 75% | **100%** | 1.93° |
| — primary pass (≥4 rungs) | 85/94 | 0.705° | 0.596° | 39% | 76% | 100% | 1.93° |
| — fallback pass (<4 rungs) | 7/94 | 0.756° | 0.660° | 29% | 57% | 100% | 1.68° |
| — answers from 1 lane | 48/94 | 0.782° | 0.708° | 31% | 71% | 100% | 1.93° |
| — answers from ≥2 lanes | 44/94 | 0.630° | 0.565° | 45% | 80% | 100% | 1.68° |
| — \|GT\| < 1.5° | 67/94 | 0.732° | 0.666° | 37% | 73% | 100% | 1.93° |
| — \|GT\| ≥ 1.5° | 25/94 | 0.649° | 0.584° | 40% | 80% | 100% | 1.68° |

All figures are from one run of the shipped defaults (`max_ladders=8`, `min_rungs=4`,
`fallback_min_rungs=2`, `agree_tol=1.5`, `max_tilt=8`, `lane_se_scale=5.5`,
`centroid_sigma_px=0.5`), reproduced by [`report_numbers.py`](report_numbers.py).

**The uncertainty is calibrated.** Mean reported `se` is **0.943°** against a mean \|error\| of
0.709°; the implied se from the error itself is 0.887°. The ratio \|err\|/se is **0.75**, close to
the 0.80 a calibrated normal gives. The reported `se` can therefore be read as a real standard
deviation.

**Where the error lives.** It is concentrated in the single-lane answers (0.782° over 48 images)
rather than the multi-lane ones (0.630° over 44) — pooling independent lanes is the single largest
source of accuracy here, and the module reports `kept` so a caller can see which class of answer it
got. Errors are also systematically positive (signed +0.171°), an over-rotation analysed in §14.8.

**Synthetic gels with a known tilt** (6 lanes, the first an 8-rung ladder, 0.35 px x-jitter,
0.8 px y-jitter, drawn from the module's own noise scale): recovered at 0/+1.5/+3/+5/−4° with
**mean \|err\| 0.14°, max 0.22°**, detecting 1–4 lanes per image. Synthetic error is much smaller
than real-gel error because real rung positions — not the algorithm — carry most of the real error.

**Per-lane ground truth.** Because the whole gel is one rigid rotation, de-rotating the centroids by
the GT angle must make every real lane vertical, so each detected lane's *expected* angle is the
image's GT angle. Used by [`lane_truth.py`](lane_truth.py): 181 lanes, mean per-lane \|err\| 0.759°,
median 0.660°. The detector finds a mean of 1.9–2.3 lanes/image.

---

## 14. Known limitations, and changes measured as NOT helping

Section 14 records approaches that were implemented and measured on the 94 gels. Those that did not
pay are **excluded from the defaults**, and the measurement is given so the work is not repeated.

1. **Slope blindness of the narrowness gate (real; the raw-x measurement remains the default).** `max_x_std` is measured on **raw** `x`, and a lane leaning by `θ` over span `L` necessarily has `x_std ≈ tan(θ)·L/√12` — 6° over 250 px ≈ 6.3 px > `max_x_std = 5`. So the default gate rejects precisely the lanes that lean. Measuring it about the fitted line is available as `gate_raw_x=False` and measured **worse** on this set (0.707° vs 0.698°), which is why raw remains the default. Whether it helps depends on how many gels actually lean past ~5°, and this set mostly does not.
2. **`_core` cuts on perpendicular distance.** The cut is made on the perpendicular distance to the tube line (§6) rather than on raw `|x − xc|`, which is the geometrically correct distance for a leaning lane and measures very slightly better (0.698° vs 0.701°).
3. **No pitch ceiling (deliberate).** A column of ≥4 regularly spaced bands sharing one x is a valid signal whether ladder or sample lane; sample-lane spacing is set by the experiment, so a ladder pitch gate rejects good measurements. `max_pitch` is retained as a switch to re-enable the gate; it is `None` by default.
4. **Recall beyond the strongest lanes does not pay.** Raising `max_ladders` to 12, flooring `min_score` to 0 and adding the consensus `slope_window` recall pass raised the lane count 1.86 → 2.33 per image but made the answer **worse** (image error 0.675° → 0.716°; per-lane error 0.722° → 0.865°). Lanes beyond the strongest are mostly low quality. `slope_window` is implemented and off by default.
5. **`trust` and `score` rank about equally; the estimator choice matters more.** Lane-level mean \|err\| is 0.774 (`score`) vs 0.775 (`trust`) — indistinguishable. `trust` is used because it estimates a quantity with a meaning (angular precision) and carries no ladder prior, not because it measured better.
6. **Curvature is not predictable from these features.** `half_gap` (upper-half vs lower-half slope) correlates only **+0.08** with per-lane error, and a 2-piece-vs-1-piece residual ratio only **+0.03**. A column that is visibly bent (worst case: halves at 1.40° and 4.10°, pooled 3.16°, GT 1.40°) is real, but no cheap per-lane feature found here flags it. This is the largest single unexplained error source.
7. **The human GT is not a local slope.** Restricting each lane to its central 80/70/50% of y was measured **worse** than the full column (0.749/0.754/0.853° vs 0.647°), and did not improve the signed bias. So the residual error is not "the annotator looked at the middle" and cannot be fixed by windowing.
8. **A systematic over-rotation remains.** Fitting `angle = a + b·GT` over the accepted set gives **`a = +0.117°, b = 1.228`** — the estimator over-reads, and more so at larger tilt. The winning lane alone shows the same (`0.074 + 1.206·GT`), so it is not an artefact of aggregation. Correcting with the fitted line removes ≈0.09° of mean error and is available via `bias_slope`/`bias_offset`, but it is fitted on the labelled set and therefore ships **disabled**.
9. **Per-lane errors are correlated within an image, which caps what pooling can do.** This is why inverse-variance pooling gains only ~0.01° over the trust-weighted mean, and why the overdispersion factor `phi` stays near 1: the lanes agree with each other *and* are all wrong together, so `Q/dof` cannot see the shared error. Only `lane_se_scale`, which was calibrated against the error actually made, expresses it.
10. **Duplicate same-y detections are invisible to `min_gap`.** `gaps` keeps only positive values, so a band split into two blobs yields a `0` gap and inflates the rung count / distorts pitch and `cv`. Proposed fix: merge rungs closer than `min_gap` within a lane.
11. **`r2_warn` is weak evidence** (correlates ≈ +0.41 with the actual angle error) and is deliberately a warning, never a veto.
12. **`max_tilt` guards `angle` only.** `contrast` and `angle_invvar` are means of individual lane angles, none individually bounded, so they can exceed `max_tilt` while `angle` does not.
13. **2 refusals remain, both structural.** `20260428 15810 990 酵母膜蛋白质粒` (15 bands) yields no lane of ≥2 rungs at all. `20260811 62.Tif` (GT −6.8°, only 7 bands) *is* detected but at −8.1°, tripping the `max_tilt = 8` sanity guard. Note that `20260509 G1-NE` and `20260513 CR 990 35160 13772P` answer from a single 3-rung lane despite 37 and 26 detections — for those two the limit is candidate formation, not band scarcity (§14.14).
14. **Only ~40% of detected bands are ever grouped into a lane.** This is the largest structural fact about the detector and it is NOT explained by the gates: loosening `dx_tol` (4→8), `dx_tol_wide` (12→25), `max_x_std` (5→12) and `split_ratio` (3.2→15) moves grouping only 36.9% → 37.8%. Lowering `min_rungs` to 3 moves it to 40.1%. The detections that never join a lane are therefore not being rejected by a gate — they do not form a structure the search proposes, or they are blobs that are not rungs. Confirming which needs the review overlays (§19), not more parameter sweeps.
15. **Lanes spanning several well rows — partially addressed, and the residue is instructive.** Observed in `20260403 35160` (one lane spanning 3 well rows), `20260424 11-1 11-2 65-1 65-2 55130-G` (two lanes spanning 2) and `20260430 990-CE…`. The gate `max_gap > split_ratio × pitch` does not catch them because `pitch` is the median, which a stitched lane drags to its smallest gap or hides: the `20260403 35160` case has gaps `367 24 302`, median 302, so 367/302 = 1.22 looks innocuous. The `consistency` term in `trust` (§8) penalises these, and it works as intended on the ranking: that spanning lane drops from rank 0 to rank 3, behind the three real lanes. **But it does not improve image-level accuracy** — mean 0.709°, and the penalty is what sets the run at that figure rather than the 0.698° obtained with the penalty disabled. Note that a *real* geometric ladder has max/min ≈ 2.3 (gaps 19 24 29 36 44), so the ratio must stay ≥3.0 to avoid penalising genuine ladders — which leaves the worst stitched lanes (max/min ≈ 2.9) unpenalised. This is the fundamental limit of the ratio statistic.
16. **Tolerating near-multiple gaps as "missed bands" made things clearly worse.** Rationale was sound — `G1-NE` has gaps 13 15 0 191 where 191 ≈ 13 × 14.7, so one weak band fragments a lane — and it is implemented as `missed_band_ratio`, but measured: mean |err| **0.828°**, max **3.08°**, and grouping did not even rise (36.9% → 37.4%). It dismantles the lane-splitting defence, letting long ragged lanes through that then win on span. **Disabled by default** (`missed_band_ratio = 1e9`).
17. **`max_ladders` is a hard cap and must be set generously.** The extraction loop stops after emitting this many lanes, so a cap that is too low silently truncates multi-lane gels and makes every other parameter look inert. `20260403 3030 1450 825 RNA` holds 16 findable lanes, of which 4 are emitted at a cap of 4, grouping 13.8% of its 116 detections; at a cap of 16 it groups **55.2%**. Set to **8**: mean |err| 0.696°, median 0.628°, lanes 2.07/image, grouping 38.3%. Beyond 8 the extra lanes are real but add nothing (0.697° at 16, 0.699° at 32). The lesson worth keeping: when several independent changes all measure identically, suspect a limit that none of them touch.
18. **A reachable crash, fixed.** `agree_tol` applied to the raw `angle` could reject *every* lane when the median fell between two clusters, leaving `keep` empty and `_pooled` indexing an empty list. It now falls back to the strongest lane.
19. **`max_frac` needs a floor or it refuses the single-lane case — the module's own primary use.** A lone lane IS essentially every detection, so `len(idx) > 0.9·N` rejects it: a perfectly vertical 4-rung or 6-rung column returns *"no lane-like rung column found"*. `max_frac` exists to stop one candidate swallowing **several** lanes, which cannot happen when there are few bands, so it is applied only when `len(xy) >= max_frac_min_bands` (12). Verified: vertical 4- and 6-rung columns return `ok=True` at 0.000° with `se` scaling correctly with pitch (0.28° at 80 px pitch, 0.96° at 20 px), while a diagonal band row and a genuine diagonal are still rejected. No effect on the 94 labelled gels (92/94, 0.709°), so this is a strict improvement on an otherwise unrepresented class of input.
20. **A refusal message must name the threshold actually in force.** Reporting `min_rungs` (4) for an image that, with `fallback_min_rungs = 2`, only ever needed 2 tells the caller a limit exists that does not. The message now reports the achievable threshold.

---

---

## 15. Declared but inert

| name | status |
|---|---|
| `core_win` | declared in `P` and documented as "half-width of the core x-cluster window", but **never read** — `_core` uses a relative cut. Dead parameter. |
| `verbose` | accepted by `estimate` / `_estimate`, never used. |
| `max_pitch` | read, but `None` by default (§14.3) — retained as a switch. |
| `half_gap` | computed on every lane and carried in the record, but **not** used by any score or gate (§14.6). |
| `bias_slope`, `bias_offset` | read and applied, but `0.0` by default (§14.8) — retained as an opt-in calibration. |

---

## 16. Determinism, state, complexity

- **Deterministic:** no RNG anywhere; identical input and parameters give identical output.
- **State:** `P` is a module-level dict. `estimate(params=…)`, `detect_ladders(min_rungs=…, max_pitch=…, slope_window=…)` apply overrides **for the call only** and always restore `P` in a `finally` block.
- **Complexity:** per extraction round, `H` hypotheses (≤ `max_seeds` + `n_slope`, with pair seeding `O(N²)` capped at `pair_max_n = 140`) each costing `max_rounds × O(N)`; up to `max_ladders = 8` rounds. In practice a 94-image batch over cached centroids runs in seconds.

---

## 17. Test surface

| script | what it establishes |
|---|---|
| [`run_bands.py`](run_bands.py) | ONNX band segmentation → cached centroids + component stats (the front half) |
| [`run_eval.py`](run_eval.py) | **the headline evaluation**: all 94 samples vs ground truth, se calibration, splits |
| [`lane_truth.py`](lane_truth.py) | per-lane ground truth from the GT rotation; which lane features predict error |
| [`ablation.py`](ablation.py) | isolates each design axis (score / gate / core / agreement / pooling) |
| [`core_test.py`](core_test.py) | raw vs perpendicular core cut, and the gate, with everything else fixed |
| [`settle.py`](settle.py) | derives `lane_se_scale` and compares the two estimators |
| [`pooling_test.py`](pooling_test.py) | recall variants and pooling schemes; the se-calibration ratio |
| [`window_test.py`](window_test.py) | tests whether the human GT is a local (windowed) slope — it is not (§14.7) |
| [`bias_check.py`](bias_check.py) | the systematic over-rotation (§14.8) |
| [`diagnose.py`](diagnose.py) | per-lane straightness, the aggregation effect, worst-case breakdown |
| [`inspect_case.py`](inspect_case.py) | dumps one sample's rung geometry, per-lane fit and half-slopes |
| [`spacing_check.py`](spacing_check.py) | the spacing signatures of §14.15 — measures `max/min`, log-spread, `cv` against error |
| [`consistency_test.py`](consistency_test.py) | the `consistency` penalty: reference ratios, exponents, robust vs min base |
| [`coverage_check.py`](coverage_check.py) | why ~60% of detections never join a lane (§14.14) |
| [`tune_fixes.py`](tune_fixes.py) | penalty and missed-band tolerance tuning; the negative result of §14.16 |
| [`profile_candidate.py`](profile_candidate.py) | instruments `_candidate`: counts where hypotheses die (gates 1.0%, `_core` 25%, run length 23%) |
| [`seeding_test.py`](seeding_test.py) | 5 seeding strategies — all measure identically (§14.17 explains why) |
| [`tube_test.py`](tube_test.py) | core-only vs whole-tube membership — identical |
| [`true_lanes.py`](true_lanes.py) | reads the TRUE lanes off the de-rotated detections and compares with what was emitted |
| [`look_at_detections.py`](look_at_detections.py) | prints the detections of an image against every gate |
| [`merge_test.py`](merge_test.py) · [`merge_components.py`](merge_components.py) | morphological merging of band components — worse at every radius |
| [`segment_test.py`](segment_test.py) | smallest-gap multiple rule for lanes crossing a missed band — correct but no effect |
| [`caplift_test.py`](caplift_test.py) | the `max_ladders` cap of §14.17 |
| [`render_lanes.py`](render_lanes.py) · [`contact_sheet.py`](contact_sheet.py) · [`panel_index.py`](panel_index.py) · [`review_panels.py`](review_panels.py) · [`hard_panels.py`](hard_panels.py) | the review overlays and sheets of §19 |
| [`render_final.py`](render_final.py) | the `final/` before-and-after comparisons with vertical guides |
| [`report_numbers.py`](report_numbers.py) | prints every figure quoted in §13 from one run of the shipped defaults |

---

## 18. Deployment package

`deploy/` holds everything needed to integrate the estimator into another application:

| file | contents |
|---|---|
| `deploy/ladders.py` | the core module, byte-identical to this folder's (numpy only) |
| `deploy/README.md` | integration instructions: interface, angle convention, coordinate-frame rules, `ok=False` handling, performance, tuning, checklist |
| `deploy/SPECIFICATION.md` | this document |
| `deploy/example.py` | runnable self-contained example on a synthetic gel with a known tilt |

Regenerate after any change to the module:

```
copy ladders.py deploy\ladders.py
copy LADDER_ALGORITHM.md deploy\SPECIFICATION.md
```

---

## 19. Reviewing the grouping

The overlays were produced to check the step everything else rests on. `render_lanes.py` writes
`work/lane_overlays/` (full resolution) and `contact_sheet.py` writes `work/lane_sheets/`:
`overview.png` (all 94 at a glance), `detail_NN.png` (4 numbered panels each), and `index.txt`
(panel number → image, estimate, GT, error, lane count, band count). Each panel shows every
detection coloured by its assigned lane, the polyline joining one lane's rungs in y order, the
lane's fitted line, detected-but-unassigned bands as hollow grey circles, and a strip of the image
de-rotated by the reported angle.

Findings from reviewing these overlays:

| panel | image | observation |
|---|---|---|
| #5, #40 | `20260403 3030 1450 825 RNA` and `20260403 35160 cds+gfp…` | all sample lanes, some 4-band some 3-band, of similar consistency — yet only 4 were taken as candidates |
| #7 | `20260403 35160` | 3 well rows, and L0 spanned all 3 |
| #18 | `20260414AD6-1` | a 3-band lane right of L0 that should have been considered |
| #33 | `20260423 65-1 65-2` | 2 well rows, L0 and L1 spanned both |
| #45 | `20260430 990-CE…` | 3 well rows; the actual 3 ladders were not detected, instead 3 lanes spanning multiple rows |
| #59 | `20260512 胡椒rna` | missed ladder |

Overall assessment: the grouping performed well, with spanning-across-well-rows as the main defect.
The "#5/#40, only 4 candidates" observation is what led to finding the `max_ladders` cap (§14.17).
§14.15 records what was done about the spanning lanes and what it cost.

---
