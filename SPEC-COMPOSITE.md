# GELTILT — composite crop specification

**Scope.** The **composite crop** feature: joining lanes of a gel into a single figure,
in a chosen order. Self-contained — the lane model, every formula, every constant, the
spec language, the editing operations, the export contract and the acceptance tests are
all specified here.

**Not in scope.** The angle cues, the rotation, the band segmentation model, the
rectangular crop, the display overlays in general, the batch runner. Those are in
[`SPEC.md`](SPEC.md), [`SPEC-AUTOROTATE.md`](SPEC-AUTOROTATE.md) and
[`DISPLAY.md`](DISPLAY.md).

**Provenance.** `desktop/lanes.py` is the reference implementation (numbering, the spec,
the composition); the composite's interaction lives in `desktop/app.py`; the reference
test is `desktop/lanes_selftest.py`, with `check_composite_cropping` in
`desktop/app_selftest.py` covering the window.

**One-line summary.** Composite crop lets a user say *"output lane 3, then lanes 1–2,
then lanes 4–6, butted together"* and get exactly that — a figure assembled from the
lanes of one gel.

---

## 1. What it is, and what it deliberately is not

**It is a figure, not a crop.** The gel's own left-to-right order is **not** preserved
unless the request asks for it. This is the defining property of the feature and the
source of its value: a ladder lane is rarely the leftmost lane, and a figure that puts
the ladder first is what the user actually wants.

Three consequences follow, and each is a requirement:

| # | requirement | why |
|---|---|---|
| **CC-1** | The exported file MUST be named `…_rotated_composite`, and the UI MUST say the lanes were rearranged | it must never be mistaken for, or overwrite, the quantitative image beside it |
| **CC-2** | The pixels MUST be the gel's own — native bit depth, native intensities, every strip a **rigid copy**. Nothing resampled, rescaled, interpolated or duplicated | the output is a figure for a paper; rearranging columns must not alter the data in them |
| **CC-3** | Rows MUST be taken whole and contiguous | the composite is the crop's vertical window, cut into the chosen lanes — not a re-sampling |

**It is not "crop to these lanes" either.** Cropping to a set of lanes would keep the
gaps between them; the composite removes the gaps, because the point is to bring
non-adjacent lanes next to each other.

---

## 2. Data model

All coordinates are **original-image pixels**, which — because the rotation preserves
the canvas size — are also corrected-image pixels. The feature never transforms
coordinates.

```
edges   : [int]                count + 1 boundaries, strictly increasing
spans   : [(int, int)]         len == count, derived from edges, tiling
groups  : [[int], ...]         the parsed request, e.g. [[3], [1, 2], [4, 5, 6]]
bands   : [{x0, x1, y0, y1, cx, cy, area, name}]   one dict per band blob
layout  : {lanes, edges, spans, blobs, centres, pitch, inserted, dropped, count}
```

The single most important structural decision: **a lane is the gap between two
boundaries.** Nothing stores a lane's own pair of edges. Therefore:

- a gap between neighbouring lanes **cannot be expressed**;
- an overlap between neighbouring lanes **cannot be expressed**;
- the lanes tile the gel by construction;
- `move_edge` cannot reorder lanes.

`spans` is derived, never edited:

```
spans_from_edges(edges) = [(int(a), int(b)) for a, b in zip(edges, edges[1:])]
```

---

## 3. The three-state gate

Composite cropping has three distinct states, and conflating them is the source of
several of the traps in §15. They are separate predicates for a reason.

| predicate | definition | what it controls |
|---|---|---|
| **ready** | `lane_count >= 2` | whether the checkbox and the spec box are enabled at all |
| **mode** | the checkbox is ticked **and** ready | whether the **overlay** is drawn — boundaries, handles, numbers, window |
| **active** | mode **and** the spec parses (`groups is not None`) | whether an **export** is the composite |

```
lane_count = max(0, len(edges) - 1)
```

Requirements:

- **R1.** `lane_count` MUST be read from the **live** boundaries, never from the count
  the segmentation originally reported. The user can split and join lanes, so
  everything that must agree with the live model — the spec's range check, whether the
  feature is available — reads the live value.
- **R2.** Fewer than two lanes means the feature is unavailable: rearranging one lane
  is not a rearrangement. The checkbox and the spec box MUST be **disabled**, and the
  hint MUST say why (`label.composite_needs`: *"run Auto crop first"*).
- **R3.** If the lanes cease to be ready (a crop that no longer contains them), the
  checkbox MUST be unticked and the request cleared — a ticked box with nothing behind
  it is worse than an unticked one.
- **R4.** The overlay MUST follow **mode**, not **active**. It has to be on screen as
  soon as the box is ticked, *including while the spec box is still empty* — that is
  exactly when a user is looking at the gel deciding what to type.
- **R5.** **Even widths** MUST be enabled only in **mode**, because it edits boundaries
  the user can only see while the overlay is drawn.
- **R6.** Ticking the box MUST switch the export dropdown to *Rotation + crop*. The
  composite **is** a crop; leaving the dropdown on *Rotation only* would make the
  export silently ignore the lanes just chosen.
- **R7.** Ticking the box is also the moment to **re-identify the lanes inside the crop
  as it stands** (§11) and to commit the spec, in that order.

---

## 4. Lane identification

Five steps: blobs → groups → fill the gaps → name → boundaries.

### 4.1 `band_blobs(mask, min_area=50) -> (blobs, dropped)`

```
count, labels, stats, centroids = connectedComponentsWithStats(mask, connectivity=8)
for i in 1 .. count-1:
    if area < min_area:  dropped += 1; continue
    blobs.append({x0, x1: x+w, y0, y1: y+h, cx, cy, area})
```

- **`MIN_BLOB_AREA = 50`** is GelGenie's own `minimum_pixel_count`, adopted rather than
  invented, so the two agree on what a band is. Without the floor, specks near the
  wells seed lanes of their own.
- `dropped` MUST be returned: it is the difference between a clean numbering and one
  that has been polluted, and it is worth reporting.

### 4.2 `label_lanes(blobs) -> [[band, ...], ...]` — the grouping rule

The convention is QuPath's, via GelGenie: lanes run left to right from 1; a lane's
window is the x-span of its **leftmost unassigned band**; a band joins a lane if its
centroid falls inside that window, **or** its box straddles either window edge.

```
while remaining:
    left = min(remaining, key=(cx, cy))            # leftmost, then topmost
    lo, hi = left.x0, left.x1
    taken = [b for b in remaining
             if lo < b.cx < hi                     # centroid inside
             or (b.x0 < hi and b.x1 > hi)          # straddles the right edge
             or (b.x0 < lo and b.x1 > lo)]         # straddles the left edge
    if not taken:
        taken = [left]                             # ← THE PROGRESS GUARD
    remove taken from remaining
    taken.sort(key=cy)                             # top to bottom
    lanes.append(taken)
    if len(lanes) > len(blobs):
        raise RuntimeError(...)
lanes.sort(key=mean(cx))                           # left to right
```

**The progress guard is mandatory and is a porting hazard.** The Java window test is
strict at both ends (`xCent < upperX && xCent > lowerX`), which is safe there because a
QuPath ROI centroid sits at the centre of its pixels — `x + 0.5` for a one-pixel-wide
object. `connectedComponentsWithStats` returns the mean of the pixel **indices**
instead, so a one-pixel-wide blob's centroid is exactly its left edge, **all three
membership tests fail, and the loop consumes nothing**. The observed result was
gigabytes of empty lanes and a pegged CPU with no error at all.

Taking the leftmost band when nothing matched guarantees progress. The
`len(lanes) > len(blobs)` check makes that guarantee impossible to lose silently by a
later edit — and a regression test for it would **hang** rather than fail, which is why
the invariant is a hard `raise` rather than an assertion.

Three other signals were measured on 14 gels and **rejected** before this rule was
chosen: mask column-gaps alone (bands are narrower than lanes — 9 "lanes" where 7 are
real), well-row peaks (noisy — 3 peaks where 8 lanes are visible), and an even-comb fit
to the well profile (aliases to a harmonic — pitch 18 where the truth is ~58).

### 4.3 `fill_empty_lanes(centres) -> (full, pitch, inserted)`

A lane with no bands leaves no blob, so **empty lanes must be inferred from the comb's
spacing**.

```
pitch = median(diff(centres))                     # the plain median
for each consecutive pair (a, b):
    k = max(1, round((b - a) / pitch))
    if k > 1 and abs((b - a)/pitch - k) > PITCH_TOLERANCE:   # 0.3
        k = 1                                     # not a whole number of pitches
    insert k-1 evenly spaced centres between a and b
```

| constant | value | load-bearing because |
|---|---|---|
| `PITCH_TOLERANCE` | **0.3** | a real empty lane sits at **1.91** pitches; the seam between two pasted slabs sits at **2.62**. A gap counts as missing lanes only when it is a whole number of pitches to within this tolerance, which separates those two measured cases |

**The pitch is the plain median of all gaps, and that is a measured decision.** A
cleverer estimator — the median of only the "single-pitch-sized" gaps — was written and
then measured away:

- it fixes a synthetic case of three detected lanes with an empty one between them;
- on the real gels it is **worse**: it put `MU55130` back to **19 lanes** (its two
  pasted slabs have pitches of ~51 and ~38, so the smaller figure makes the 2.62-pitch
  seam read as three lanes) and it inserted **ten phantom lanes** on `20230402`.

The three-lane case it misses is what the `count` override is for (§4.4). That
limitation MUST be pinned by a test, not fixed by the cleverer estimator.

### 4.4 `lane_layout(mask, region, min_area=50, count=None) -> layout`

```
confine mask to region = [x0, y0, x1, y1]        # the same convention as bands.content_box_from_mask
blobs, dropped = band_blobs(confined, min_area)
lanes          = label_lanes(blobs)
centres        = [mean(cx) for each lane]

if count and count > 1 and len(centres) >= 2:    # the override
    lo, hi = min(centres), max(centres)
    full   = linspace(lo, hi, count)
    pitch  = (hi - lo) / (count - 1)
    inserted = max(0, count - len(centres))
else:
    full, pitch, inserted = fill_empty_lanes(centres)

if not full:
    return {lanes: [], spans: [], centres: [], pitch: 0.0,
            inserted: 0, dropped: dropped, count: 0}      # ← note: NO 'edges' key

# assign each band to its nearest filled centre, name it, sort top to bottom
for i, lane in enumerate(named, 1):
    lane.sort(key=cy)
    for k, band in enumerate(lane, 1):
        band['name'] = f'L{i}-{k}'

# boundaries: half a pitch beyond the outer centres, midpoints between
edges = [full[0] - pitch/2] + [(a+b)/2 for consecutive pairs] + [full[-1] + pitch/2]
edges = [clamp(round(e), x0, x1) for e in edges]
for i in 1..len(edges)-1:
    edges[i] = max(edges[i], edges[i-1] + 1)     # force strictly increasing
```

Requirements:

- **The `region` MUST be `[x0, y0, x1, y1]`.** It was `(x, y, w, h)` at first and the
  two disagreed: the right edge became a **width**, stretching the search past the gel
  and finding specks in the black surround (a dropped blob at x≈1250 on a gel 878 px
  wide).
- **The `count` override** divides the span the bands actually occupy evenly instead of
  trusting the spacing. It is the escape hatch for a gel the heuristics over-count
  (`MU13772`: a band row runs on past the wells and adds **four** lanes) and it is the
  one thing the user reliably knows about a gel.
- **Edges are clamped to the region and then forced strictly increasing**, so a
  degenerate pitch can never produce a zero- or negative-width lane.
- **Band naming is `L<lane>-<band>`**, bands top to bottom. That is the QuPath
  vocabulary, so the numbers on the pane mean what the user means when they type them.
- **Acceptance: 11 of 11** gels the user reviewed (an earlier revision matched 10 of 11,
  over-counting four lanes where a band row runs on past the wells).

---

## 5. Editing the boundaries

Four operations, all of which preserve the model in §2 — none can create a gap, an
overlap or a reordering.

### 5.1 `move_edge(edges, index, x, width=None, min_side=1)`

Moves **one** boundary. Because a lane is the gap between two boundaries, this moves
the right edge of one lane and the left edge of the next **together** — which is the
whole point: the model cannot express a gap, so dragging cannot make one.

```
low  = edges[index-1] + side   if index > 0        else 0
high = edges[index+1] - side   if index < last     else width (or edges[index])
if low > high:  return edges unchanged             # nowhere to go
edges[index] = clamp(round(x), low, high)
```

- The new position is clamped between its neighbours, so a boundary cannot cross them.
- `width` is the frame width, bounding the outermost boundary; without it an outer
  boundary can only move inwards.
- "Nowhere to go" leaves the list **exactly** as it was, rather than nudging.

### 5.2 `insert_edge(edges, x, min_side) -> (edges, index)` — split

Adds a boundary at `x`. Returns the index it landed at, or `None` if refused.

**Refused, not clamped**, when `x` is outside the lane region or either resulting piece
would be narrower than `min_side`. Refusing keeps the click honest: a border appears
where it was asked for, or not at all.

### 5.3 `remove_edge(edges, index, min_side) -> (edges, index)` — join

Drops an **inner** boundary, joining the two lanes it separates. Returns the removed
index, or `None`.

Refused when: fewer than 4 boundaries (two lanes are the fewest that can be
rearranged), or the index is one of the two **outer** boundaries — those hold the lane
region, and joining a lane to nothing would only lose gel.

### 5.4 Renumbering after an edit — mandatory

**Both edits renumber the selection**, because the lanes after the change shift and a
number left alone would quietly come to mean a different piece of gel.

```
renumber_after_split(order, lane) = [n if n <= lane else n + 1 for n in order]
renumber_after_merge(order, lane) = dedup([lane if n == lane+1 else
                                           (n - 1 if n > lane + 1 else n)
                                           for n in order])
```

- After a **split** at boundary index `lane`: the split lane keeps its number (it is
  now the **left half** of what was there, which is what the number meant), and
  everything after shifts up by one.
- After a **merge**: both numbers become `lane`, and an order naming **both** keeps one
  of them — the merged lane is a single piece of gel, and listing it twice would
  duplicate it. `dedup` preserves first-occurrence order.

### 5.5 `LANE_MIN_PX = 8`

The narrowest a lane may be made by splitting or joining, in image pixels. Lanes are
38–56 px across on the gels this was built for, so 8 leaves room to correct a boundary
without allowing a lane too narrow to hold a band.

---

## 6. Even widths

A convenience for a heavily hand-edited gel where boundaries have drifted and no two
lanes match. **Two steps, in this order.**

### 6.1 `even_edges(edges)` — average the internal boundaries

```
n = len(edges) - 1
left, right = edges[0], edges[-1]
if right - left < n:  return edges unchanged        # not even a pixel each
return [left + round(i * (right - left) / n) for i in 0..n]
```

- **The two outer boundaries do not move.** They are where the user put the gel's
  edges.
- The formula is exact, so **a boundary that does not need to move does not move** —
  on an already even gel this is a no-op, and pressing the button twice changes nothing
  the second time.

### 6.2 `nudge_off_bands(edges, blobs, min_side) -> (edges, nudged)`

Averaging knows nothing about where the bands are, so it can drop a boundary straight
through one — **invisible in the numbering, very visible in the figure**.

```
for each INTERNAL boundary index:
    low, high = edges[index-1] + side, edges[index+1] - side
    for up to 4 passes:
        x = edges[index]
        cutting = [boxes with left < x < right]          # bands this boundary cuts
        if none: break
        options = [min(left of cutting), max(right of cutting)]
                  filtered to those within [low, high]
        if no options: break                             # cannot clear without crossing
        edges[index] = the option nearest to x
        nudged += 1
```

- **Only internal boundaries move.** The outer two are the gel's edges, and a boundary
  that cannot be cleared without crossing a neighbour is **left alone** rather than
  traded for a worse fault.
- Up to 4 passes, because a move can land inside another band.
- The clearing move is computed **past the leftmost left edge or the rightmost right
  edge of all the bands being cut**, so one move can clear several at once.

### 6.3 Reporting

Both steps are idempotent, and the status line MUST say which of them did something,
rather than leaving the user to wonder whether the button worked:

| situation | key |
|---|---|
| nothing changed | `status.lanes_already_even` |
| averaged only | `status.lanes_evened` |
| nudged only (already even) | `status.lanes_nudged` |
| both | `status.lanes_evened_nudged` |

A band **wider than a lane** cannot be cleared by moving a boundary — clearing it would
have to cross a neighbour and undo a lane. The count of such boundaries MUST be
reported (`status.still_cutting`, interpolated as `note`) rather than left for the user
to find in the figure.

---

## 7. The spec language

### 7.1 Grammar

```
spec   := term (',' term)*
term   := lane | range
lane   := [Ll]? DIGITS
range  := [Ll]? DIGITS \s*-\s* [Ll]? DIGITS
```

```
3,1-2,4-6      lane 3, then lanes 1-2, then lanes 4-6, butted together
L3,L1-L2       the same thing in the QuPath vocabulary
3              just lane 3
```

Each comma-separated term is a lane or an inclusive range; **the groups are joined left
to right in the output.** The usual use is to put the ladder first, which is rarely the
leftmost lane.

**The fan-out form is not supported yet** and MUST say so rather than guess:

```
3,{1,2,4}      three separate outputs, lane 3 with each of 1, 2 and 4
```

### 7.2 `parse_lane_spec(text, n_lanes) -> [[int], ...]`

Checks in this order, each raising `LaneSpecError(key, **kw)`:

| # | condition | key | arguments |
|---|---|---|---|
| 1 | `text` is None or empty after strip | `lane_spec.empty` | — |
| 2 | `{` or `}` appears anywhere | `lane_spec.braces` | — |
| 3 | a comma term is empty | `lane_spec.empty_term` | — |
| 4 | a term does not match the grammar | `lane_spec.unparsed` | `term` |
| 5 | `first < 1` or `last < 1` | `lane_spec.not_positive` | `term` |
| 6 | `first > n_lanes` or `last > n_lanes` | `lane_spec.out_of_range` | `term`, `count` |
| 7 | `last < first` | `lane_spec.reversed` | `term` |

Note the consequences of that order: braces win over everything; `-2` is `unparsed`
(the grammar requires digits, and the leading `-` fails the match) rather than
`not_positive`; and `0` is `not_positive`.

**`LaneSpecError` carries a locale key and its arguments, never a sentence**, because
the app must explain the problem in the user's language and the tests assert on the
key.

### 7.3 Failure behaviour

**A spec that does not parse leaves the composite OFF.** The spec is a promise about
which lanes go in which order; guessing at it is how a figure ends up in a paper with
the wrong lanes in it.

```
on LaneSpecError:
    groups = None            # -> _composite_active() is False -> no composite export
    spec_bad = True
    status = the error key and its arguments
```

The box keeps the user's text — they are mid-edit — but the figure is not built from a
guess.

### 7.4 `spec_groups(groups)` — the flat order

```
[[3], [1, 2], [4, 5]] -> [3, 1, 2, 4, 5]
```

This flat list is the single source of truth for *order*. It is what the badges number,
what the tint follows, and what the click handlers edit.

---

## 8. The vertical window

The composite keeps a **band of rows**, given by the rectangular crop's `y0`/`y1` —
everything between them, across whichever columns the lanes select.

```
composite_rows():
    height = image height
    if crop is None:  return (0, height)
    y0, y1 = int(crop[1]), int(crop[3])
    y0 = clamp(y0, 0, height)
    y1 = clamp(max(y0, y1), 0, height)
    return (y0, y1)
```

Requirements:

- **Rows are in IMAGE pixels and the function takes no size argument.** It previously
  took a height argument and was called with the **pane's** height, so at small window
  sizes the window was clamped to the scaled-down pane: the drawn boxes and the dimming
  stopped early while the export, which works at full resolution, was correct. The user
  saw boxes that "end too early" until the window was made fullscreen. **There is no
  argument to get wrong now — the frame is the only thing it can clamp to.**
- **The window is exactly where the automatic crop ends**, not the whole lane and not
  the whole gel. An earlier attempt made it "the whole lane", down to where the gel's
  own signal fades — 516 px on the sample gel against the crop's 213 — and the user
  asked for the automatic crop's window back: a figure *of the lanes*, not of the gel's
  empty lower smear.
- With no crop box the window is the whole frame's height.

---

## 9. Composition

### 9.1 `output_runs(groups) -> [[int], ...]`

The flat order, grouped into **maximal blocks of consecutive lane numbers**:

```
[[3], [1, 2], [4]]  ->  [[3], [1, 2], [4]]
[[1, 2, 3]]         ->  [[1, 2, 3]]
[[1], [2], [3]]     ->  [[1, 2, 3]]        # consecutive, so one block
[[1], [3]]          ->  [[1], [3]]         # a real skip
```

This is a **description of the result, not a different way of building it** — see
§9.3.

### 9.2 `stitch_lanes(arr, edges, groups) -> array`

```
if arr.ndim != 2:
    raise ValueError(f'expected a 2-D grey image, got shape {arr.shape}')

parts = []
for run in output_runs(groups):
    parts.append(arr[:, edges[run[0] - 1] : edges[run[-1]]])

if not parts:
    return arr[:, :0]
return concatenate(parts, axis=1)
```

Note the slice: for a run `[a..b]`, the columns are `edges[a-1]` to `edges[b]` — the
whole contiguous block, taken in **one** slice.

- **Rows are taken whole**, so the result is the input's own data: native bit depth,
  native intensities, every strip a rigid copy.
- The caller passes the already-row-sliced array (§8), so this function is purely a
  column rearrangement.
- **A 2-D grey input is required** — the rearrangement is defined on columns, and
  silently handling a 3-channel array would change what "lane" means.

### 9.3 Why neighbouring lanes must be one slice

Because neighbouring spans **share their boundary** (§4.4's midpoint edges), slicing
`[1, 2]` lane-by-lane and butting the pieces gives **identical pixels** to slicing the
block once. This is not a coincidence to be relied on loosely; it is the property the
shared-boundary model exists to guarantee.

The measured contrast: rounding each span's two edges independently from its own centre
made neighbours overlap by a pixel wherever the fractional pitch rounded up — on a real
seven-lane gel, **three of six seams** each duplicated a column, giving **367 px of
composite where the gel's own block is 364**. Bounding the edges once, at the midpoint,
makes the lanes tile exactly.

A corollary worth knowing: moving a boundary that **both sides of a request include**
cannot change the composite's width, because the runs together are one contiguous
range. It changes the width only where the request skips a lane, and it always changes
which lane a band belongs to.

---

## 10. Export

The composite uses the **crop** export path, not a path of its own.

```
if mode == CROP and composite_active:
    rotated = final_image(arr, angle, crop=None)     # the full rotated NATIVE-DEPTH frame
    y0, y1  = composite_rows()
    return stitch_lanes(rotated[y0:y1, :], edges, groups)

if mode != CROP and composite_active:
    refuse: 'status.composite_ignored'
    return None
```

Requirements:

- **The composite is built from the rotated native-depth frame**, not from a display
  copy. Rearranging columns must not change the data.
- **The rows come from the crop**, so a hand-adjusted crop box still applies: the
  composite is that window, cut into the lanes the user chose.
- **Ticking composite and then exporting "Rotation only" MUST be refused with a
  message**, not quietly ignored. Silently dropping the lanes the user just chose is the
  worse failure.
- **The filename suffix is `_rotated_composite`** (CC-1), and the mode detail string is
  `mode.detail.composite`: *"rotation + composite crop, lanes rearranged"*.
- The **Use original image** checkbox and the clipboard reduction behave exactly as for
  any other export; the composite is not special-cased there.

---

## 11. Re-identification inside the crop

**Auto crop finds the lanes of the whole corrected frame, but the figure being built is
the crop.** After the user adjusts the crop by hand — a second gel stripped off the
edge — the lanes of the whole frame are the wrong answer: the numbering starts
somewhere off-screen, and the spans are pitched for lanes no longer in the picture.

```
reidentify_lanes(announce=True):
    if crop is None or band_mask is None:  return False
    layout = lane_layout(band_mask, crop)          # the CROP as the region
    set lanes, edges from layout
    if the existing spec still parses against the new live count:
        keep it                                    # same count -> same lanes
    else:
        clear the spec box and the groups, saying so
```

Requirements:

- **The segmentation model is NOT re-run.** Auto crop already computed the band mask for
  this image; this re-**groups** that mask inside the crop. The mask is a property of
  the image; "which lanes are in this crop" is a question about the grouping. It takes
  milliseconds, which is why a crop drag can afford it.
- **The selection is kept if it still parses** — a same count means the numbers still
  name the same lanes — and otherwise dropped with a message, rather than left pointing
  at lanes that no longer exist.
- **It runs when a crop drag ENDS**, not on every motion event: the work is fast but the
  *point* of it is a stable answer, and a moving answer is no use while dragging.
- It also runs when the box is ticked (§3 R7) and from the crop-application hook, which
  covers Auto crop and the padding control in one place.

---

## 12. Interaction contract

### 12.1 What is drawn

The overlay follows **mode** (§3 R4). Per lane, left to right:

| element | rule |
|---|---|
| **lane number** | at each lane's horizontal centre, just **above the window** (22 px above the top row, floored so it never leaves the pane). Cyan `#00e5ff` when the lane is selected, grey `#8a8a8a` otherwise |
| **position badge** | `#1`, `#2`, … **15 px below** the number, amber `#ffd400`, only for selected lanes, giving the lane's position **in the output** |

Per boundary, at every index:

| state | boundary line | handle (7×9 px) at each end |
|---|---|---|
| **lit** — either neighbouring lane is selected | solid, width 2, cyan | filled cyan, darker outline |
| **unlit** | `dash=(2,4)`, width 1, cyan | hollow, cyan outline, `dash=(2,2)` |

Boundaries are drawn only between the window's two rows, and there is **one handle at
each end of the window** — so the boundary reads as a division of the kept strip, not of
the whole gel.

**The window's two rows are drawn across the whole image** (not merely across the crop
box's width) in orange `#ff9d00`, `width=2`, `dash=(6,4)`, with a 9×9 handle 7 px in
from each side.

### 12.2 The crop box becomes a window

In composite mode the crop box is **only its top and bottom edges**, spanning the whole
pane, in amber. Its **left and right bounds are neither drawn nor grabbable.**

This is a requirement, not a simplification. With the box in rectangle mode, the left
and right edges described **nothing about the output** — the width comes from the lane
boundaries — yet they looked like handles and, if dragged, silently moved the
rectangular crop behind the composite's back while the composite looked unchanged. The
user's words: *"only the top ever moves"*.

### 12.3 Hit testing and clicks

In composite mode the hit test is a **different model** from the rectangle one (which
is in [`DISPLAY.md`](DISPLAY.md) §10.3). There is no `left`, `right`, `move` or `new`,
and **no new box is ever started** — the crop box is what a composite keeps vertically,
and a press that quietly replaced it is how the window used to jump to wherever the user
clicked.

| action | target | effect |
|---|---|---|
| **left-click** a lane | `lane n` | toggle it in the figure, **appended at the end** of the order |
| **middle-click** a lane | `lane n` | **split** it: insert a boundary at the click |
| **middle-click** a boundary | `edge i` | **join**: remove that boundary |
| left-drag a boundary | `edge i` | move it (§5.1) |
| anything else | — | nothing |

Requirements:

- **A lane click edits and does not drag** (the drag mode is set to none first).
- **Added at the end of the order**, so the figure is composed in the order the lanes
  are clicked — which is the whole point of putting the ladder first.
- Any click edit **rewrites the spec box as a flat comma list** (`3,1,2,4,5`). That is
  the same figure pixel for pixel as the compact ranges the user may have typed,
  because neighbouring lanes are one slice either way (§9.3); the *order* is what the
  clicks are composing.
- The clicks **renumber the selection** (§5.4) — mandatory, not cosmetic.

### 12.4 The tint

The pane dims everything outside the chosen lanes: for each lane in the flat order, its
`(x0…x1)` × the window's rows is kept untouched, and all else is dimmed and pushed
toward red. The lanes are tinted in **request order**, which the pane cannot show
spatially — the badge is what conveys the order.

---

## 13. Localisation keys

Every string the feature shows, from `desktop/locales/*.json`:

| key | text (en) |
|---|---|
| `check.composite` | Composite crop |
| `btn.even_lanes` | Even widths |
| `label.composite_hint` | click lanes to join, or type e.g. 3,1-2,4-6 |
| `label.composite_needs` | run Auto crop first |
| `mode.detail.composite` | rotation + composite crop, lanes rearranged |
| `info.composite` | composite: {spec} |
| `info.composite_clicks` | click a lane to add or remove it · middle-click a lane to split it, a border to join it |
| `status.composite_on` | Composite crop: {spec} — {count} lane(s) joined in that order, {w} px wide. |
| `status.composite_off` | Composite crop off — the crop box applies again. |
| `status.composite_needs_autocrop` | Auto crop first — composite cropping joins the lanes it finds. |
| `status.composite_ignored` | Composite crop is set, but Save/copy is "Rotation only". Choose "Rotation + crop", or untick composite cropping. |
| `status.lane_added` / `status.lane_removed` | Lane {lane} added/removed — the figure is lanes {order}. |
| `status.lanes_none` | Nothing selected — click lanes to add them. |
| `status.border_added` | Border added at x={x} — {count} lanes now. |
| `status.border_removed` | Border removed — {count} lanes now. |
| `status.border_here` | No border added: too close to the border beside it, or outside the lanes. |
| `status.border_outer` | That border stays: the two outer ones hold the lane region, and two lanes are the fewest that can be rearranged. |
| `status.lanes_evened` / `_nudged` / `_evened_nudged` / `_already_even` | see §6.3 |
| `status.still_cutting` | (interpolated note) |
| `status.lanes_reidentified` | Lanes re-identified from the crop — {count} lane(s). |
| `lane_spec.empty` / `_empty_term` / `_unparsed` / `_not_positive` / `_out_of_range` / `_reversed` / `_braces` | see §7.2 |

`LaneSpecError` arguments MUST be interpolated by the app, and the **keys** of
`LaneSpecError` MUST NOT be translated.

---

## 14. Acceptance tests

A conforming implementation must pass these. They are the reference checks from
`desktop/lanes_selftest.py` and `desktop/app_selftest.py`, restated so they can be
built from this document.

| # | test | criterion |
|---|---|---|
| **C1** | the grouping rule always makes progress | a mask with one-pixel-wide blobs produces lanes without hanging, and the count never exceeds the blob count |
| **C2** | the area floor | blobs below 50 px are dropped and counted in `dropped` |
| **C3** | the pitch and empty-lane insertion | a gap of exactly 2 pitches inserts one lane; a gap of 2.62 pitches inserts none; the three-detected-lane case is pinned as a **known limitation** |
| **C4** | names and spans | bands are named `L<lane>-<band>` top to bottom, and `spans` tiles `edges` exactly |
| **C5** | spans are centred on the lanes | each span's midpoint lies within its lane, not in the gap beside it |
| **C6** | **neighbours share their boundary** | for every pair of adjacent lanes, the left lane's `x1` equals the right lane's `x0` — no gap, no overlap, no duplicated column |
| **C7** | **adjacent lanes are one block** | stitching a run of neighbouring lanes equals the gel's own contiguous slice, pixel for pixel |
| **C8** | splitting and joining | `insert_edge` refuses too-close and out-of-region positions; `remove_edge` refuses the two outer boundaries and the last available join; both renumber a selection correctly |
| **C9** | averaging the widths | the two outer bounds are unchanged, the internal ones are evenly spaced, a second press changes nothing, and a boundary inside a band is nudged clear |
| **C10** | boundaries are nudged off the bands | no internal boundary lands inside a blob box after `nudge_off_bands`, unless clearing it would cross a neighbour |
| **C11** | the count override | `count=2` and `count=6` produce that many lanes with even spacing regardless of what was detected |
| **C12** | empty and degenerate inputs | an empty mask, a region outside the mask, an impossible `min_area`, and a one-lane-wide region all return without raising (`count == 0` / a sane layout) |
| **C13** | parsing the lane spec | every error case of §7.2 raises its exact key, and the valid cases produce the written order |
| **C14** | **the composite is the written order, the right pixels, the original depth** | `stitch_lanes` on a `uint16` array with `[[3],[1,2],[4]]` gives shape `(10, 40)` and dtype `uint16`; a duplicated lane `[[2,2]]` yields one copy; a skip `[[1],[3]]` gives two slices; a 3-D input raises `ValueError` |
| **C15** | the window in the UI | the composite's rows follow the crop box, in image pixels, at any window size; the tint, the drawn window lines and the export all agree |

C6 and C7 are the pair that protects §9.3, and C14 is the one that protects CC-2. An
implementation that rounds each lane's edges independently will pass C4 and C5 and fail
C6 and C7 — with the duplicated columns showing up as a too-wide composite.

---

## 15. Traps already paid for

| # | trap | rule |
|---|---|---|
| **T-1** | **The grouping loop can run away.** `connectedComponentsWithStats` centroids are the mean of pixel *indices*, so a one-pixel blob's centroid equals its left edge and every membership test fails | always take the leftmost band when nothing matched, and keep the `lanes > blobs` raise |
| **T-2** | **A pane-sized window.** The window was clamped to the pane, so boxes stopped early at small window sizes while the export was right | measure the window in **image** pixels, and take no size argument |
| **T-3** | **A press started an invisible crop box.** The overlay stopped drawing the box in composite mode but the hit test still tested its edges | in composite mode there is no `new`, and no left/right edges to grab |
| **T-4** | **The crop box's left/right edges silently edited the rectangle crop** while the composite looked unchanged | draw and test the top and bottom only |
| **T-5** | **Per-span rounding duplicates columns** at seams where the fractional pitch rounds up | bound the edges once, at the midpoint; derive spans from edges |
| **T-6** | **`lane_layout` took `(x, y, w, h)` and the caller passed a box**, so the right edge became a width and the search ran into the black surround | the region is `[x0, y0, x1, y1]` |
| **T-7** | **The mask was thrown away with the cached detection**, so re-identification silently had nothing to re-group | the cached Auto crop result must carry the band mask |
| **T-8** | **Averaging alone cuts bands in two** — invisible in the numbering, very visible in the figure | always follow `even_edges` with `nudge_off_bands` |
| **T-9** | **A ticked box with no lanes behind it** | uncheck and clear when the lanes stop being ready |
| **T-10** | **Ticking composite with the dropdown on "Rotation only"** would ignore the lanes | force the dropdown to crop mode, and refuse the inconsistent export with a message (§10) |
| **T-11** | **The fan-out form `3,{1,2,4}`** would silently produce something the user did not ask for | refuse it explicitly with `lane_spec.braces` |

---

## 16. Known defect

**`lane_layout` returns no `edges` key when there are no bands, and a caller indexes
it.** The empty branch (§4.4) returns a dict of seven keys — `lanes`, `spans`,
`centres`, `pitch`, `inserted`, `dropped`, `count` — with **no `edges`**.

`desktop/app.py:2294`, in `reidentify_lanes`, does:

```python
if not layout or not layout['edges']:
```

The empty dict is **truthy** (it has keys), so the first test does not short-circuit and
the second raises `KeyError: 'edges'`.

Reproduced:

```
returns: {'lanes': [], 'spans': [], 'centres': [], 'pitch': 0.0,
          'inserted': 0, 'dropped': 0, 'count': 0}
keys   : ['centres', 'count', 'dropped', 'inserted', 'lanes', 'pitch', 'spans']
has edges key? False
caller expression RAISES: KeyError 'edges'
```

**Reachable in the UI:** tick *Composite crop*, then drag the crop box onto a region
containing no bands — blank gel or the black surround — and release. The crop-end hook
calls `reidentify_lanes`, which raises. In Tk an exception inside a callback prints a
traceback and **aborts that callback**, so the re-render that follows it does not run:
the pane keeps the stale overlay until the next event.

**Specified behaviour:** `lane_layout`'s empty branch SHOULD include `edges: []` (so
the returned shape is constant), and callers MUST use `layout.get('edges')`. The
`count == 0` and `spans`/`centres` shape of the empty branch is already asserted by C12;
`edges` is not, which is why this was not caught.

---

## 17. Conformance checklist

1. `region` is `[x0, y0, x1, y1]`. (§4.4)
2. Bands below `MIN_BLOB_AREA = 50` are dropped, and the count is reported. (§4.1)
3. The grouping rule takes the leftmost band when no band matches its window, and raises
   if lanes ever exceed blobs. (§4.2)
4. `pitch` is the **plain median** of all gaps; a gap is missing lanes only when it is a
   whole number of pitches within `PITCH_TOLERANCE = 0.3`. (§4.3)
5. An empty lane's `pitch` (1.91) and a slab seam's (2.62) are distinguished. (§4.3)
6. Edges are midpoints between centres, half a pitch beyond the outer ones, clamped to
   the region and forced strictly increasing. (§4.4)
7. `spans` is derived from `edges`, never stored; a lane is the gap between two
   boundaries. (§2, §4.4)
8. Bands are named `L<lane>-<band>` top to bottom. (§4.4)
9. `move_edge` clamps between neighbours and cannot reorder lanes. (§5.1)
10. `insert_edge` and `remove_edge` **refuse** rather than clamp, and both renumber the
    selection. (§5.2–5.4)
11. `even_edges` holds the two outer boundaries and is idempotent; `nudge_off_bands`
    moves only internal boundaries and only when it can clear without crossing. (§6)
12. The status line says which of averaging/nudging actually did something. (§6.3)
13. The spec grammar accepts `3`, `1-3`, `L3`, `L1-L2`, and refuses braces, gaps,
    out-of-range, reversed and non-numeric terms with the exact keys of §7.2. (§7)
14. A spec that does not parse leaves the composite **off** and keeps the user's text.
    (§7.3)
15. The vertical window is in image pixels, follows the crop, and takes no size
    argument. (§8)
16. `output_runs` groups consecutive lanes, and `stitch_lanes` takes each run as one
    slice of the native-depth data. (§9)
17. The export builds from the **rotated native-depth** frame and the crop's rows, and
    "Rotation only" with composite active is **refused with a message**. (§10)
18. Re-identification re-groups the cached mask inside the crop, at drag **end**, and
    keeps the selection only if it still parses. (§11)
19. The overlay follows **mode**, not **active**; the checkbox, spec box and Even-widths
    button follow ready / ready / mode respectively. (§3)
20. The crop box draws as two amber full-width rows, with no left, right or move
    targets. (§12.2, §12.3)
21. Left-click toggles a lane (appended last); middle-click splits or joins; click edits
    rewrite the spec as a flat comma list. (§12.3)
22. The export is named `…_rotated_composite`. (CC-1)
23. C1–C15 of §14 pass, **including C6, C7 and C14**.
