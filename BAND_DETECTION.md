# How the GelGenie band outputs are generated

`ladders.py` consumes band centroids and never touches pixels, so it is independent of how they
were produced. This document specifies the front half that produced them for this project
([`run_bands.py`](run_bands.py)), because the integration contract starts one step earlier: the
centroids are only as good as the segmentation that made them.

Nothing here is a dependency of `ladders.py`. A host application may substitute its own segmenter
as long as it emits band centroids in image pixels.

---

## 1. The model

| property | value |
|---|---|
| file | `gelgenie_1024.onnx`, 54.6 MB |
| producer | PyTorch (`producer_name`); **no custom metadata** in the graph |
| input | `input`, float32, `[1, 1, 1024, 1024]` — one grayscale tile |
| output | `output`, float32, `[1, 2, 1024, 1024]` — **raw logits, not probabilities** |
| providers | `CPUExecutionProvider` (used throughout; no GPU required) |
| class 0 | background — logits ≈ +5.5 |
| class 1 | **Gel Band** — logits ≈ −7.5 |

Because the graph carries no metadata, two input conventions had to be established empirically:
the intensity scaling, and the orientation (bright bands on dark, or the inverse). Both were decided
by probing the model and are recorded in §3. §7 shows the commands and outputs that verify them.

## 2. Loading the images

```python
a = np.asarray(tifffile.imread(path))
if a.ndim == 3:
    a = a.mean(-1)                      # collapse any channel axis
img = a.astype(np.float32) / 255.0      # uint8 -> [0, 1]
```

`uint16` divides by 65535; anything else is normalised by its own maximum. The 94 project samples
are all `uint8`, `(1024, 1280)` — height × width — which is a **non-square** image against a
square model input, so §4 matters.

## 3. The two conventions that matter

**Intensity.** Feed `gray / 255`. Inverting it is wrong: the model expects **bright bands on a dark
background**, so the gel is fed unmodified.

Evidence: with `gray/255` the band channel reaches probability 1.0 and yields sensible band objects
(15–36 components early in the project, 116 on the densest sample). With `1 − gray/255` the band
channel collapses (max logit ≈ 4.2, probability ≈ 0.5) and yields ~4 components, which is
background. The probe in §7 reports both on the same tile.

**Softmax axis.** The output is `(C, H, W)`, so the softmax must be taken over **axis 0**, the class
axis:

```python
out = sess.run(None, {in_name: t[None, None]})[0][0]   # (C, H, W)
e = np.exp(out - out.max(axis=0, keepdims=True))       # axis 0 = classes
p = (e / e.sum(axis=0, keepdims=True))[1]              # class 1 = Gel Band
```

This is worth stating explicitly because the axis is easy to lose. The correct class-1 map on
`20260403 3030 1450 825 RNA` has mean 0.031 and max 1.000, with 32 492 pixels above 0.5. Softmaxing
over a *spatial* axis instead produces a single row rather than a spatial map — mean 0.00056,
max 0.00072, and **no** pixel above 0.5 — which is why the axis is documented here rather than left
implicit.

Two equivalent formulations, both verified to give the same mask: the class-axis softmax above, and
`sigmoid(ch1 − ch0)`, which is the two-class softmax written out. `sigmoid(ch0)` alone is not
equivalent: it is ≈ 1 nearly everywhere (mean 0.96) and thresholds to a single giant blob.

## 4. Tiling and the feather merge

The input window is fixed at 1024×1024, so a 1024×1280 image is covered by overlapping tiles:

```
ys = sorted(set([0] + range(0, H - 1024 + 1, 1024 - overlap) + [H - 1024]))     # overlap = 256
xs = sorted(set([0] + range(0, W - 1024 + 1, 1024 - overlap) + [W - 1024]))
```

For these samples that is **1 tile in y × 2 in x** (origins 0 and 256). Each tile is placed into a
zero-filled 1024×1024 canvas, so an image *smaller* than the window is zero-padded rather than
resized, and the logits are computed at the image's true scale.

Overlaps are merged with a **tent weight** — the distance to the nearest tile edge, clipped at 128:

```python
w = clip(minimum(xx + 1, tw - xx, yy + 1, th - yy), 0, 128)
prob[y0:y1, x0:x1] += p * w
wsum[y0:y1, x0:x1] += w
prob /= maximum(wsum, 1e-6)
```

With `overlap = 256`, only the outer 128 px taper, so a tile contributes at full weight over its
interior and two tiles cross-fade across the seam. Without the merge, the seam is visible as a
vertical edge in the band probability.

## 5. Mask and centroids

```python
mask = cv2.morphologyEx((prob > 0.5).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
n, lab, stats, cent = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
keep = (stats[:, AREA] >= 30) & (np.arange(n) > 0)
xy = cent[keep]          # (N, 2) float64, columns (x, y)
```

* **Threshold 0.5** on the softmax probability.
* **Opening with a 3×3 kernel** removes isolated single-pixel responses.
* **8-connected** components.
* **Minimum area 30 px** discards specks. Measured component sizes on real samples: median width
  25–41 px, median height 11–14 px, median area 248–462 px — comfortably above the floor, so the
  floor is not doing much work on these images.
* **Centroid** of each component is the band position. This is the only geometric output used
  downstream; the component's `(width, height, area)` is also cached for diagnostics.

For reference on the 94 samples: 7–116 centroids per image, mean ≈ 34.

## 6. Caching

Inference is the slow step, so results are cached under `work/bandcache/`:

| file | contents |
|---|---|
| `<key>.npz` | `xy` (N,2) centroids and `wha` (N,3) component width/height/area |
| `<key>_prob.npy` | the probability map, float16 |

`key = md5(abspath | file size | mtime)[:16]` — i.e. **keyed on the image file, not on the code**. If
you change this pipeline, delete `work/bandcache/` or the old centroids will be reused silently.

## 7. Verifying the conventions

`probe_preprocess.py` renders the input and both orientations on production tiles:

```
python probe_preprocess.py
```

It writes `work/probe/<tile>_compare.png`, three panels side by side: the input as fed, the band
probability from the raw scaling, and from the inverted scaling. `softmax_compare.py` tests every
plausible reading of the model output against the mask and band count the pipeline reports:

| candidate | mean | max | pixels > 0.5 | bands |
|---|---|---|---|---|
| class-axis softmax, class 1 **(used)** | 0.0311 | 1.0000 | 32 492 | 115 |
| `sigmoid(ch1 − ch0)` (equivalent) | 0.0311 | 1.0000 | 32 492 | 115 |
| `sigmoid(ch1)` | 0.0286 | 0.9973 | 31 057 | 115 |
| `sigmoid(ch0)` | 0.9636 | 0.9999 | 1 015 068 | 2 |
| softmax over a spatial axis | — | — | — | not a 2-D map |

(Counts are for the top-left tile; the two-tile feather merge yields 116 for the full image.)

## 8. What this pipeline does not do

* **No threshold tuning per image.** A fixed 0.5 is used throughout. This is the most likely place
  for a different scanner or staining protocol to need a change.
* **No band shape or intensity filtering.** Every component above 30 px becomes a centroid; a
  well-line dash or a smear counts as a band. `ladders.py` rejects what it cannot form into a lane,
  which is a different and later filter.
* **No lane assignment.** Grouping is entirely `ladders.py`'s job.
* **No allowance for overlapping bands.** Two bands closer than the 3×3 opening merge into one
  component and become one centroid.
* **No sub-pixel refinement** beyond the component centroid.
