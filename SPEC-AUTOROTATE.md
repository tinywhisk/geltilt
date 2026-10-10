# GELTILT — default auto-rotation algorithm specification

**Terminology.**

|term|meaning|
|-|-|
|**smear**|the faint vertical streaking that runs down each lane, like ruled lines on a page laid vertically. The cue this algorithm measures|
|**gel body / mask**|the region of the image that is gel rather than black surround|
|**strip / interior**|the gel mask after erosion, cropped to its own bounding box|
|**column profile**|a vector, one value per image column, produced by summing down the rows|
|**angle**|the rotation that levels the smears, in OpenCV's convention: positive rotates content counter-clockwise. Applied **as-is** to the image|

\---

## 1\. Interface

```
estimate(img, mask=None, gates=True, reference=None, min\_delta=0.5,
         split\_tol=1.3, equiv\_tol=0.5) -> (ok: bool, angle: float, info: dict)
```

|||
|-|-|
|`img`|2-D greyscale array, `uint8` or `uint16`, native scale. Not normalised, not windowed|
|`mask`|optional precomputed `gel\_mask`. `None` computes it (§4). Reuse matters: the gates recompute it on pre-rotated images|
|`gates`|run the reliability gates of §7.3. **The interactive default is `False`** (§8)|
|`reference`|an angle already in hand, for the `agrees` short-circuit. **Must be `None` for a decision** (§7.3)|
|`min\_delta`, `split\_tol`, `equiv\_tol`|the three thresholds of §2, as parameters with those defaults|
|`ok`|whether `angle` may be used|
|`angle`|degrees. **Always returned**, even when `ok` is False, so a caller can display a rejected estimate|
|`info`|diagnostics: `angle`, `boundary`, `split`, `equiv`, `reason`, `gated`|

The caller must apply `angle` **as-is**. It is already the rotation that cancels the
tilt; negating it doubles the error.

A bare estimator that omits the gates entirely may be exposed separately
(`smear\_angle(img) -> float`); it is this function with `gates=False` and its `ok`
discarded, and it exists for the gates' own internal use.

\---

## 2\. Constants

Every constant, with its role and what it is load-bearing for.

|constant|value|role|load-bearing because|
|-|-|-|-|
|`BODY\_FRAC`|`0.04`|threshold offset for the gel mask|a single low fraction lets the gel's dimmer parts in while leaving the black surround out|
|`BODY\_MIN\_AREA`|`1500`|px, minimum component kept in the mask|removes small bright details on the surround, which are not gel|
|`BODY\_BLUR`|`9`|px, blur before thresholding|stops one noisy pixel seeding a component|
|`EDGE\_ERODE`|`25`|px, mask erosion before weighting|**removes the slab's own left and right edges**, which are strong near-vertical lines but not the band direction. This is the single reason the estimator measures bands rather than slab geometry|
|`BG\_SIGMA`|`25.0`|px, illumination cutoff|removes gradients slower than the light variation, which otherwise dominate a projection|
|`RIDGE\_SIGMA\_Y`|`6.0`|px, smoothing **along** the smear|the matched filter|
|`RIDGE\_SIGMA\_X`|`2.0`|px, curvature **across** the smear|the matched filter|
|`RANGE\_DEG`|`10.0`|the sweep is `−10° … +10°`|no gel in the reference set is tilted beyond it|
|`STEP`|`0.25`|deg, the sweep step|**one pass, and this step.** A coarse-then-fine search is *rejected*: a 1° first pass misses narrow peaks and reads `+0.97°` on a gel where this step reads `−1.47°`|
|`COARSE\_STEP`|`1.0`|deg, step for the equivariance probes only|the probes only need to see a large deviation|
|`PREBLUR`|`0.8`|px, blur inside the rotated search|every angle is then interpolated equally, so 0° acquires no free sharpness spike|
|`MIN\_SUPPORT`|`800`|mask px below which there is nothing to measure|a nearly-empty mask produces noise, not an angle|
|`SLICE\_HEIGHT`|`36`|rows per persistence slice|**must be shorter than the gap between band rows**, so a band falls inside one slice and cannot correlate with itself. Measured: 28–40 px all agree; 56 px and up start locking onto the bands|
|`SLICE\_MIN\_ROWS`|`24`|rows, trailing slice dropped below this|correlating a short trailing slice on noise is worse than dropping it|
|`EDGE\_GUARD`|`1.0`|deg from the range end|a peak that has *nearly* run off the end is the same runaway one step earlier. Measured: every gel it catches sits at ≥ 9.4°, where no gel in the set really is|
|`AGREE\_TOL`|`1.5`|deg, cue-agreement tolerance|the observed failures disagreed by 8–16°, while every gel the user confirmed agreed to under 1.2°. The line sits between them with room to spare|
|`SPLIT\_TOL`|`1.3`|deg, halves gate|see §7.3|
|`EQUIV\_TOL`|`0.5`|deg, equivariance gate|see §7.3|
|`EQUIV\_DELTAS`|`(−2.0, +2.0)`|deg, the pre-rotations probed|one either side of the answer, far enough to expose a large deviation|

**No constant here is derived from first principles.** Each came from a specific gel
or a specific measured failure. If results drift, suspect this table first.

\---

## 3\. Required helpers

These are dependencies. Their behaviour is part of this specification.

### 3.1 `robust\_limits(arr, plo=0.1, phi=99.9) -> (lo, hi)`

Percentile range of `arr` as floats. If `hi <= lo`, set `hi = lo + 1.0`.

Percentiles rather than `min`/`max` because one saturated pixel would otherwise set
the top of the range and make every threshold built on it meaningless.

### 3.2 `rotate\_about(img, centre, angle, border=CONSTANT, border\_value=0) -> array`

Rigid rotation of `img` about `centre` (given as `(x, y)`), **preserving the canvas
size**, bilinear interpolation, corners filled with constant black.

Requirements that matter here:

* **`scale` is pinned to exactly 1.0.** No shear, no perspective, no non-uniform
scale, ever. Distances and areas are preserved.
* **Canvas size preserved.** Both the input and output are `H×W`, which is what lets
a mask rotated separately land on the same grid.
* **Constant black corners, never replicated.** Replication smears edge pixels around
the frame and invents structure that was never imaged, which this estimator would
then measure.

### 3.3 `rotate\_roi(roi, angle, preblur=0.0) -> array`

Rotate about the ROI's **own centre**, keeping its size. If `preblur > 0`, convert to
`float32` and Gaussian-blur by `preblur` **before** rotating.

The pre-blur is not cosmetic. An affine warp is an exact identity at 0° and
interpolates at every other angle, so without it the 0° sample is sharper than every
other sample and any sharpness score acquires a manufactured peak there.

### 3.4 `map\_chunks(fn, items) -> array`

Apply `fn` to **contiguous chunks** of `items` and concatenate the results in order.
`fn` takes an array of items and returns either an array of results or a tuple of
such arrays (each concatenated separately).

Thread count: `min(8, max(1, cpu\_count() − 1), len(items) // 6)`, and **1 if that is
≤ 1**, in which case call `fn` directly. The chunking must be contiguous so that the
result is **bit-identical** to the serial loop — this is a hard requirement, not an
optimisation note, and a conforming implementation must assert it.

The work is numeric and GIL-releasing (OpenCV, `scipy.ndimage`, numpy ufuncs), so
threads give real parallelism. Threads rather than processes on purpose: a process
pool would add pickling and freeze-fragility for no gain.

\---

## 4\. `gel\_mask()` — the gel body

This is the foundation of everything else, and it is specified in full.

```
gel\_mask(img, frac=0.04, min\_area=1500, blur=9) -> uint8 array, same shape as img,
                                                   values 0 or 1
```

```
f  = float32(img)
b  = GaussianBlur(f, sigma=blur)          # sigma 9 px, kernel size derived
lo, hi = robust\_limits(f, 0.1, 99.9)
if hi <= lo:
    return zeros\_like(img, uint8)

m  = uint8(b > lo + frac \* (hi - lo))
m  = morphologyEx(m, MORPH\_OPEN, ones(3, 3))
n, labels, stats, \_ = connectedComponentsWithStats(m, connectivity=8)

out = zeros\_like(m)
for i in 1 .. n-1:
    if stats\[i, AREA] >= min\_area:
        out\[labels == i] = 1
return out
```

### 4.1 Specification of each step

|step|specification|
|-|-|
|blur σ=9|spatial, px. Stops a single noisy pixel becoming a component|
|`robust\_limits(f, 0.1, 99.9)`|taken from the **unblurred** float image, but applied to the blurred one. Blurring first stops noise seeding components; taking the range from the raw data stops the blur depressing the range|
|threshold `lo + 0.04·(hi − lo)`|**relative to the robust range**, never absolute. This is what makes the mask work on a bright gel and a faint gel, and on 8-bit and 16-bit data alike|
|open 3×3|removes single-pixel and thin-line noise|
|components ≥ 1500 px|keeps the gel; drops small bright details on the black surround|
|output|`uint8`, 0/1, **same shape as the input**|

### 4.2 What it must and must not do

* **It must include the gel's dimmer parts.** A higher threshold returns only the
bright core, and the estimator would then measure the core's edges.
* **It must exclude the black surround.** On a gel whose brightness varies, a
higher-threshold approach returns one bright block of a continuous gel; here the
low fraction keeps the whole body.
* **It must not be a segmentation of the slab boundary.** The mask is allowed to be
loose, because §5 erodes it anyway. Nothing downstream depends on its edge being
accurate.
* **It never raises.** A degenerate image (empty range) returns an all-zero mask, and
every caller must handle that by declining, not by guessing.

### 4.3 Why this and not a threshold on the whole frame

The mask exists to answer "where is the gel", not "where are the bands". It is
deliberately a *brightness* mask with no structural assumptions, because the two
things that would break a structural approach are both present in this data: a gel
whose brightness varies across it, and a frame that may hold more than one gel. It is
also computed once and reused by every sub-estimate in §7.

\---

## 5\. `interior()` — the strip the estimator measures

```
interior(img, mask, erode=25) -> (sub, subm) or None
```

```
m = uint8(mask)
if m.sum() == 0:
    return None

if erode > 0:
    me = erode(m, ones(erode, erode))        # 25 × 25
    if me.sum() >= MIN\_SUPPORT:
        m = me

ys, xs = where(m > 0)
if xs.size == 0:
    return None
y0, y1 = min(ys), max(ys) + 1
x0, x1 = min(xs), max(xs) + 1

sub  = float32(img\[y0:y1, x0:x1])
subm = float32(m\[y0:y1, x0:x1])
if subm.sum() < MIN\_SUPPORT:
    return None
return (sub, subm)
```

### 5.1 Requirements

* **The erosion is mandatory and is the point of the function.** A 25×25 erosion
removes a band of that width around the mask boundary, which is where a slab's own
left and right edges are. Those edges are strong, near-vertical lines; if they were
included in the projection they would overwhelm the faint smears and the estimator
would report the slab's edge angle instead of the band direction.
* **The erosion is conditional.** If eroding would leave fewer than `MIN\_SUPPORT`
pixels, the un-eroded mask is used. Better a noisier estimate than none.
* **The crop to the mask's bounding box is an optimisation and a correctness measure
together:** it keeps the strip small, and it means the rotation in §6 operates on
gel rather than on empty surround, which is what makes the constant-black corner
fill harmless.
* **`MIN\_SUPPORT` is checked twice**, before and after the crop, because an eroded
mask can be non-empty yet still too small to measure.
* Returns `None` for an empty mask, not an exception.

\---

## 6\. The two per-angle responses

Both are computed from the same rotated strip, in one pass. This shared work is why
asking two questions of each angle costs barely more than asking one.

### 6.1 `flatten(roi, sigma=25.0)`

```
clip(float32(roi) - GaussianBlur(roi, sigma), 0, +inf)
```

An unsharp/difference-of-Gaussians high-pass. Everything slower than σ is treated as
illumination and subtracted. The **clip at zero is not cosmetic**: a dark surround
would otherwise go negative and, when summed into a projection, would subtract from
the very structure being measured.

**The flattening must stay inside the per-angle loop.** It is tempting to hoist it
out — the blur is isotropic, so it commutes with a rotation, and hoisting roughly
halves the cost. It is wrong: the clip is a non-linearity, so it does *not* commute.
Applied before the pre-blur it moves the black surround differently, and the halves
gate of §7.3 then reads **6–10°** of disagreement on gels whose halves really agree
to 0.5°. Tried, measured, reverted.

### 6.2 `ridge(roi)`

```
clip(-gaussian\_filter(float32(roi), sigma=(6.0, 2.0), order=(0, 2)), 0, +inf)
```

`order=(0, 2)` is the **second derivative across x** (across the smear); the Gaussian
smooths **along y** with the larger σ = 6. So a vertical streak is smoothed along its
length and sharply peaked across it: a matched filter for "vertical bright line".

The anisotropy is load-bearing. An isotropic ridge detector would respond to any
blob; this one responds specifically to a vertical streak.

### 6.3 The per-angle body

```
def scores\_serial(strip, angles, preblur=0.8):
    sub, subm = strip
    for k, a in enumerate(angles):
        rot  = rotate\_roi(sub,  a, preblur=preblur)     # §3.3, about the strip centre
        rm   = rotate\_roi(subm, a, preblur=0.0)
        supp = rm > 0.5
        resp = ridge(flatten(rot, BG\_SIGMA)) \* supp     # §6.1, §6.2
        p    = resp.sum(axis=0)                         # column profile

        contrast\[k] = p.std() / p.mean() if p.mean() > 0 else 0.0
        persist\[k]  = persistence(resp)                 # §6.4
```

Notes:

* **The mask is rotated separately with no pre-blur**, then thresholded at `> 0.5`.
It must stay binary and must land on the same grid as the rotated image.
* **The support is applied after the response**, not before: `resp = ridge(...) · supp`.
Applying it to the input would let the black surround's clipped values enter the
derivative.
* `contrast` and `persist` are arrays the same length as `angles`.
* **`contrast` is a coefficient of variation**, so it is invariant to any positive
scale factor of the image. This is why the estimator does not care whether the gel
is bright or faint, or whether the data is 8- or 16-bit.

### 6.4 `persistence(resp) -> float in \[−1, 1]`

```
rows = resp.shape\[0]
if rows < SLICE\_HEIGHT + SLICE\_MIN\_ROWS:      # < 36 + 24 = 60
    return 0.0

profs = \[]
for y0 in range(0, rows - SLICE\_MIN\_ROWS + 1, SLICE\_HEIGHT):
    p  = resp\[y0 : y0 + SLICE\_HEIGHT].sum(axis=0)
    sd = p.std()
    if sd <= 0:
        continue                              # a flat slice carries no shape
    profs.append((p - p.mean()) / sd)         # z-scored column profile

if len(profs) < 2:
    return 0.0
m = asarray(profs)                            # slices × columns
corr = m @ m.T / m.shape\[1]                   # pairwise correlation
return mean(corr\[upper\_triangle\_indices])     # mean off-diagonal
```

**What it measures.** Whether the column profile *repeats* down the gel. A band is a
blob: it lands in one slice and contributes a spike nothing else shares. A smear is a
streak: it runs through every slice, so it correlates with itself. That is exactly the
difference between a smear a user calls well defined and one they do not.

**Why the slice is 36 rows.** It must be shorter than the gap between band rows, or a
band would span a whole slice and correlate with itself too, destroying the
discrimination. Measured on five gels: 28–40 px all agree; 56 px and up start locking
onto the bands.

**Why slices are z-scored.** Each slice is normalised before correlating, so a slice
that is merely brighter than its neighbours does not inflate the score. Only the
*shape* is compared.

**The requirement on the fixture for this to be testable:** two images carrying the
same energy in the same number of columns must score differently if and only if the
energy sits in the same columns in every slice. The reference test builds exactly
that pair — a streak that repeats (must score ≥ 0.9) and blobs that never line up
(must score ≤ 0.3) — and requires the first to exceed the second.

\---

## 7\. The estimator

### 7.1 The sweep

```
angles = arange(-RANGE\_DEG, +RANGE\_DEG + 1e-9, STEP)    # -10 … +10 by 0.25, 81 samples
contrast, persist = map\_chunks(scores\_serial, angles)
```

The `1e-9` is there so `+10.0` is included — `arange` excludes the stop value
otherwise. A conforming implementation must sample **81 angles** inclusive of both
ends.

**One pass at 0.25°.** No coarse-to-fine, no downsampling, no window around an
expected answer:

* coarse-then-fine at 1° misses narrow peaks (measured: `+0.97°` where this step
finds `−1.47°`);
* downsampling the ROI is \~4× faster and moves the answer by up to 1.5°;
* **narrowing the sweep to a window is unsound.** A *second* local maximum can sit
inside the window while the true peak is just outside it, and it is not on the
window edge, so nothing flags it. Measured: for the halves it moved 15 of 94 gels
by up to 12° and flipped the verdict on 3; for the equivariance probe it made one
gel read as equivariant when the full sweep has it 5.96° out.

### 7.2 Choosing between the two cues

```
peak(curve, step):
    i = argmax(curve)                       # ties resolve to the lowest angle
    best = angles\[i]
    if 0 < i < len(curve) - 1:
        d = curve\[i-1] - 2\*curve\[i] + curve\[i+1]
        if abs(d) > 1e-15:
            best += 0.5 \* (curve\[i-1] - curve\[i+1]) / d \* step     # parabolic vertex
    boundary = abs(best) >= RANGE\_DEG - EDGE\_GUARD                 # >= 9.0
    return best, boundary

a\_cv, edge\_cv = peak(contrast,    STEP)
a\_ps, edge\_ps = peak(persist,     STEP)

if not edge\_cv and abs(a\_cv - a\_ps) <= AGREE\_TOL:
    angle, boundary = a\_cv, False
else:
    angle, boundary = a\_ps, edge\_ps
```

Requirements:

* **Sub-step refinement is required.** Without the parabolic vertex the answer is
quantised to 0.25°, which is coarser than the sub-degree precision this data needs.
* **Contrast is the primary cue**; it is the sharper of the two and is used whenever
it is trustworthy, i.e. its peak is interior and the two cues agree within 1.5°.
* **Persistence takes over when contrast has been fooled**, which is the failure it
exists for: on a gel whose bands sit in rows, contrast locks onto the *blobs* rather
than the smears and runs off toward the end of the range.
* **`boundary` is recomputed from whichever peak won.** When persistence wins, its own
edge test is reported.

### 7.3 The gates

```
verdict(img, mask, angle, boundary, gates,
        reference=None, min\_delta=0.5, split\_tol=1.3, equiv\_tol=0.5):

    if boundary:
        return ok=False, reason='no\_peak'

    if reference is not None and abs(angle - reference) < min\_delta:
        return ok=False, reason='agrees'          # gates deliberately NOT run

    if not gates:
        return ok=True, gated=False               # caller will fill the caveats in later

    split, dev, reason = gates\_of(img, mask, angle, split\_tol, equiv\_tol)
    if reason:
        return ok=False, reason=reason
    return ok=True, reason=''
```

**Gate 1 — `no\_peak`.** A peak at or within 1° of the range end is not an answer: it
means the sweep ran off the end and its value is an artefact of the limit.

**Gate 2 — `agrees`.** When a reference angle is supplied and the estimate is within
0.5° of it there is nothing to decide, and the two expensive gates are skipped. This
is a *performance* short-circuit. **It must not be used when the caller intends to
apply the angle** — the default auto-rotate path passes no reference, precisely so
that this path is never taken for a decision.

**Gate 3 — `halves\_differ`.**

```
xs  = where(mask.any(axis=0))
if xs.size < 2:  split = None, reason = 'no\_gel'
mid   = (min(xs) + max(xs)) // 2              # integer midpoint
left  = mask with columns >= mid zeroed
right = mask with columns <  mid zeroed
split = abs(estimate(left) - estimate(right))
if split > 1.3:  reason = 'halves\_differ'
```

Both halves are measured with the **full fine sweep** — not a window. The reason for
sampling a whole image twice is that a bent or impossible geometry is invisible to a
single global estimate.

**Gate 4 — `not\_equivariant`.**

```
centre = (w/2, h/2)                            # of the ORIGINAL image
dev = 0
for d in (-2.0, +2.0):
    rot = rotate\_about(img, centre, d)
    a\_d = estimate(rot, mask=gel\_mask(rot), mode='coarse')     # step 1.0
    dev = max(dev, abs(a\_d - (angle - d)))
if dev > 0.5:  reason = 'not\_equivariant'
```

**The property.** Pre-rotating the image by `d` must move the estimate by exactly
`−d`. An estimator whose answer depends on the absolute angle is reading an artefact
of the pixel grid rather than the content. This is the gate that distinguishes a real
estimate from a grid-aligned one, and it is not hypothetical: it is what separated
the two gels this cue was validated against.

Requirements:

* **The probe uses a full coarse sweep, never a window.** A window is exactly the
unsoundness described in §7.1, and measurably: one gel reads as equivariant under a
window when the full sweep has it 5.96° out.
* **The mask is recomputed on the pre-rotated image** (`gel\_mask(rot)`), not rotated.
* `mode='coarse'` differs **only** in `STEP` → `COARSE\_STEP`; the range, the
aggregation and §7.2 are unchanged.

**Gate ordering matters.** `boundary` and `agrees` are checked before any heavy work;
`halves\_differ` is checked before `not\_equivariant`, so a gel that fails the cheaper
gate never pays for the probe.

**The gates cannot move the angle.** They decide *whether it may be used* and what is
reported. An implementation that let a gate adjust the angle would break the
measurement that shows the caveats can be computed behind the rotation (§8).

\---

## 8\. Timing: the caveats are computed behind the rotation

The gates of §7.3 cost more than the estimate they describe (two further full sweeps
for the halves, two coarse sweeps for the probe). Requirements:

1. The **default auto-rotate path calls the estimator with `gates=False`**, so
`verdict` returns as soon as `boundary` and `reference` are settled, with
`info\['gated'] = False`.
2. The caller **applies the angle immediately**.
3. The caller then runs the gates **on a background thread** and merges `split`,
`equiv` and `reason` into the diagnostics when they land, re-rendering whatever
sentence describes them.
4. A late result **must be discarded if the detection it belongs to is no longer the
one on screen** (the user loaded another gel, or re-ran detection).

Measured: the rotation appears **1.6–2.8× sooner** and lands on exactly the same
angle, because the gates cannot influence it.

**`gates=True` must remain available** for the synchronous path — the self-test and
any non-interactive caller use it.

\---

## 9\. Reference pseudocode

```
def estimate(img, gates=False, reference=None):
    mask = gel\_mask(img)                                  # §4
    strip = interior(img, mask)                           # §5
    if strip is None:
        return True, 0.0, {'angle': 0.0, 'boundary': True, 'split': None,
                           'equiv': None, 'reason': 'no\_peak', 'gated': bool(gates)}

    angles = arange(-10.0, 10.0 + 1e-9, 0.25)             # 81 samples
    contrast, persist = map\_chunks(scores\_serial, angles) # §6.3, §3.4

    a\_cv, edge\_cv = peak(contrast, 0.25)                  # §7.2
    a\_ps, edge\_ps = peak(persist,  0.25)

    if not edge\_cv and abs(a\_cv - a\_ps) <= 1.5:
        angle, boundary = a\_cv, False
    else:
        angle, boundary = a\_ps, edge\_ps

    info = {'angle': angle, 'boundary': boundary, 'split': None,
            'equiv': None, 'reason': '', 'gated': bool(gates)}

    if boundary:
        info\['reason'] = 'no\_peak'
        return False, angle, info
    if reference is not None and abs(angle - reference) < 0.5:
        info\['reason'] = 'agrees'
        return False, angle, info
    if not gates:
        return True, angle, info

    split, dev, reason = gates\_of(img, mask, angle, 1.3, 0.5)
    info\['split'], info\['equiv'] = split, dev
    if reason:
        info\['reason'] = reason
        return False, angle, info
    return True, angle, info
```

**Caller contract.** Apply `angle` as-is. Refuse when `ok` is False — do not
substitute another cue, and do not fall back to 0° silently; report the `reason`.
The reasons a user can be shown are `no\_peak`, `halves\_differ`, `not\_equivariant`,
`hard\_gel` (assigned by the caller, §10), `agrees` and `error`.

\---

## 10\. Caller duties (the parts that are not the estimator)

1. **Do not negate the angle.** It is already the rotation that cancels the tilt.
2. **Refuse, do not fall back.** When `ok` is False:

   * `reason == 'no\_peak'` → report "no clear peak";
   * `reason == 'halves\_differ'` → report that the two halves disagree;
   * `reason == 'not\_equivariant'` → report that the answer moved with the image
angle.
3. **One extra refusal rule of the product, not of the estimator:** when the
structural detector declined altogether for this gel (no top edge and no well row),
the smear estimate must **also** be refused, with the reason reported as
`hard\_gel`. It is the one gel in the reference set where the smears were
measurably worse than no rotation at all. A standalone implementation of this
document can omit this rule, but then it will accept that gel.
4. **Report where the angle came from**, and the caveat if there is one.
5. **Run the estimator on the original image**, never on a rotated or windowed one.
6. **Compute `gel\_mask` once per estimate**, not per angle, and reuse it inside §7.2.
7. **Do not window or normalise the image before calling.** The estimator is
scale-invariant by construction (§6.3) and normalising changes nothing except the
cost.

\---

## 11\. Acceptance tests

A conforming implementation must pass these. They are the reference checks from
`gelaalign/smears\_selftest.py`, restated so they can be built from this document.

**Fixture.** A bright slab carrying vertical smears, where the smear direction and
the slab's own long axis are tilted by **different** angles on purpose. The fixture
parameters: a slab polygon, `n = 16` vertical stripes of pitch `span/16` and half
width `max(1.5, pitch·0.16)`, stripe amplitude 55 over a slab at 70 on a surround at
12, Gaussian noise σ = 1.5, everything rotated bodily by a known `tilt`. The
leveling angle the estimator must recover is `−(tilt + smear\_offset)`.

|#|test|criterion|
|-|-|-|
|**T1**|the smears set the angle|for `smear\_offset ∈ {−3.0, −1.5, 0.0, +1.5, +3.0}`, the estimate is within **0.6°** of `−smear\_offset`|
|**T2**|the answer is rotation-equivariant in the body|a ±2.0° body tilt moves the estimate by exactly `−tilt`, within **0.6°** of the `smear\_offset = 1.5` baseline|
|**T3**|**the slab edges do not set the angle**|for `edge\_offset ∈ {−3.0, +3.0}` with vertical smears, the estimate is within **0.8°** of 0. This is the test of `EDGE\_ERODE`, and it is the single most important one|
|**T4**|a split gel is refused|with the two halves' smears offset by ±3.0° (6° apart), `ok` is False and `reason == 'halves\_differ'`|
|**T5**|persistence tells a streak from a band|a streak repeating in the same two columns of every slice scores **≥ 0.9**; blobs of equal energy that never line up score **≤ 0.3**; and the streak exceeds the blobs|

T3 deserves emphasis: it is the property that makes this cue *good*, rather than
merely different. An implementation that omits the erosion passes T1 and T2 and fails
T3 with an error of the full `edge\_offset`.

T5 must be built with the two fixtures carrying **the same energy in the same number
of columns** — the only difference being whether it sits in the same columns in every
slice. Otherwise the test passes for the wrong reason.

\---

## 12\. Measured behaviour (for calibration, not for testing)

Recorded against the 94-gel reference set, at these defaults:

|metric|value|
|-|-|
|gels given an angle|**92 of 94**|
|gels refused|2, by **two different gates** — one defeated every structural cue (`hard\_gel`), one whose persistence peak ran to the edge (`no\_peak`)|
|gels confirmed by eye|**7**, moved by this rule and nothing else|
|mean run time|a few seconds per gel, single-pass threaded|

The seven confirmations are the reason the defaults are what they are. Three
measured examples of what the persistence cue fixes, current (`smears`) versus the
pre-persistence behaviour:

|gel|now|before|
|-|-|-|
|`20260509 G1-NE`|**−1.39°**|+9.84°|
|`2026-519 胡椒rna2`|**+1.76°**|−6.02°|
|`20260520 SLERF2GDNA`|**+5.19°**|+0.27° — the old value had merely aligned the slab's left edge|

And two that must still be refused, caught by different gates: `20260429 990 13772-…`
(the `hard\_gel` case — perfectly equivariant, but every structural cue declined) and
`20260428 15810 990` (its persistence peak sits within `EDGE\_GUARD` of the range end,
where it used to report −9.4°).

**None of these gels are in the tree.** They are lab data. The numbers are the
recorded measurements and cannot be re-run from source.

\---

## 13\. Deliberately rejected

Each of these was implemented, measured and reverted. Do not reintroduce one without
addressing the measurement.

|rejected|measurement|
|-|-|
|**coarse-then-fine search**|at a 1° first pass it misses narrow peaks: `+0.97°` where the fine step reads `−1.47°`|
|**downsampling the ROI**|\~4× faster, moves the answer by up to 1.5°|
|**narrowing a sweep to a window**|15 of 94 gels moved by up to 12° and 3 verdicts flipped for the halves; one gel read equivariant when it is 5.96° out|
|**hoisting the flattening out of the angle loop**|the clip is non-linear and does not commute with the rotation: the halves gate reads 6–10° of disagreement on gels that agree to 0.5°|
|**dividing each column by its mask support**|removes a real silhouette artefact, but is not rotation-equivariant there, and made the left/right agreement noisier on several verified wins. The plain sum stands|
|**an isotropic ridge filter**|responds to any blob; the anisotropic `order=(0,2)` is what makes it a smear detector|
|**a single aggregation cue**|contrast alone runs away on gels whose bands sit in rows (up to 9.8°); persistence alone is less precise. Both, with the 1.5° rule, is what works|
|**enforcing the gates rather than reporting them**|the user reviewed every gel the gates flagged and judged the smear angle right in each case; they are now caveats, not vetoes. (The `no\_peak` and `hard\_gel` refusals are separate and do remain)|

\---

## 14\. Conformance checklist

1. `gel\_mask` thresholds at `lo + 0.04·(hi − lo)` of a 0.1/99.9 percentile range,
opens 3×3, keeps components ≥ 1500 px, returns 0/1 `uint8` of the same shape. (§4)
2. `interior` erodes the mask by 25×25 **unless** that would leave < 800 px, then
crops to the mask's bounding box. (§5)
3. 81 angles: `−10 … +10` inclusive at 0.25°. One pass. (§7.1)
4. Per angle, in this order: pre-blurred rotation → flatten σ=25 → clip → ridge
`(6, 2, order=(0,2))` → clip → multiply by the separately rotated binary mask.
(§6)
5. `contrast = std/mean` of the column profile; `persistence` = mean pairwise
correlation of 36-row z-scored slices, trailing < 24 rows dropped. (§6.3, §6.4)
6. Peak per curve with parabolic sub-step refinement; `boundary` at ≥ 9.0°. (§7.2)
7. Contrast wins iff its peak is interior and it agrees with persistence within 1.5°;
otherwise persistence. (§7.2)
8. Gates in order: `no\_peak`, `agrees` (only with a reference), `halves\_differ` at
1.3° on the full sweep, `not\_equivariant` at 0.5° on a full **coarse** sweep of
±2.0° pre-rotations. (§7.3)
9. Neither gate may alter the angle. (§7.3)
10. `gates=False` returns immediately with `gated=False`; the caveats are filled in
later, and discarded if the detection is stale. (§8)
11. Threading is over **contiguous** chunks and is bit-identical to the serial loop.
(§3.4)
12. The angle is returned for **both** outcomes, and the caller never negates it and
never falls back silently. (§1, §10)
13. T1–T5 of §11 all pass, **including T3**.

