"""Gel lane tilt from band centroids: find the lane-like columns of detections and estimate the
single rigid rotation that makes the lanes vertical.

WHAT IT ESTIMATES. One number: the rotation to APPLY to the image, in degrees, in the
`cv2.getRotationMatrix2D` convention (positive = counter-clockwise on screen, y downward). For a
lane of slope s = dx/dy, that is -atan(s). This module does not rotate anything, does not assign
every band to a lane, and does not measure intensity.

INPUT. An (N, 2) array of band centroids in image pixels, (x, y) - nothing else. No pixels are
read anywhere in this module, so it is independent of which segmentation model produced them.

LANES, NOT LADDERS. A ladder and a loaded sample lane are the same structure: a column of rungs
sharing one x, leaning together with the gel. There is no innate difference beyond how many rungs
were detected, so the module does not privilege ladders. A lane is a set of detections on a line
x = a + s*(y - ybar) whose y extent is large compared with its x scatter. The slope is shared by
every lane in the image (all lanes tilt together) with one free x-offset per lane, so two lanes in
the same x column at different heights stay two lanes instead of one impossible line.

DETECTION is greedy score-and-mask: propose candidate lines, refine each against its inlier tube,
score, accept the best, remove its points, repeat. Because a lane is one uninterrupted column,
lanes are split at gaps that are not part of the same spacing series - this is what stops the top
and bottom halves of a two-row-of-wells gel from being averaged into one lane.

SCORING is per-lane trustworthiness: how precisely that lane pins its own angle
(lever arm over rung-to-line scatter, times a spacing-consistency factor). It carries no ladder
prior, so a clean sample lane scores like a clean ladder.

COMBINATION is a random-effects inverse-variance pool of the per-lane angles, with MAD outlier
rejection and a Cochran-Q overdispersion factor, against a robust median reference. The reported
`se` is calibrated: on 94 hand-labelled gels the error actually made is 0.75 x the reported se.

Measured on 94 hand-labelled gels: coverage 92/94, mean |error| 0.709 deg, median 0.646 deg,
100% within 2 deg, worst case 1.93 deg. See LADDER_ALGORITHM.md for the full specification, the
per-lane statistics, and the list of approaches that were measured and rejected.
"""
import numpy as np

# ------------------------------------------------------------------ defaults (all tunable)
# Extraction as implemented today: greedy score-and-mask. Each round proposes lines from PAIRS of
# detections in the same x neighbourhood, keeps the best-elongation candidate whose rungs are
# uninterrupted in y, removes its points, and repeats. `estimate` then reports the shared-slope
# fit over the ladders that agree with the strongest.
#
# What this deliberately does NOT use yet (next stage): sample lanes with >=3 bands. Those are a
# real tilt signal but a weaker one - they are usually 2-3 bands, and unlike a ladder their
# spacing carries no expectation - so they belong in the pooled fit as SUPPORT once ladder
# detection is solid, never as the primary candidate. The seam for that is `_pooled(xy, keep)`:
# accept a second kind of group into `keep` and the angle follows.
P = dict(
    min_rungs=4,          # a lane needs at least this many rungs
    dx_tol=4.0,           # px: how far a rung may sit from the fitted line
    dx_tol_wide=12.0,     # px: first-pass window, tightened after the first fit
    slope_hi=0.20,        # tan: slope hypotheses are swept in +-this range
    n_slope=41,           # number of slope hypotheses
    max_seeds=4000,       # cap on pair-seeded hypotheses per extraction round
    pair_min_dy=10.0,     # px: min y separation for a seeding pair (a dense lane has short rungs)
    pair_max_dx=12.0,     # px: max x separation for a seeding pair (same column, not a row)
    pair_max_n=140,       # skip pair seeding above this many live detections (cost)
    core_win=15.0,        # px: declared but inert (see doc)
    min_rungs_prior=3,    # rungs on the hypothesis line needed before re-fitting it
    max_rounds=3,         # proposal -> fit -> re-select passes
    min_span=30.0,        # px: rung extent below which a candidate is not a lane
    min_gap=4.0,          # px: reject candidates whose rungs sit this close (double detections)
    split_ratio=3.2,      # reject when a gap exceeds this multiple of the median rung pitch
    min_elong=3.5,        # min y-spread / x-spread for a candidate to count as a lane
    max_frac=0.9,         # a lane may not claim essentially every detection in the image
    # ...but only when there are enough detections for "essentially every detection" to mean
    # anything. On a sparse image the ONE real lane IS every detection, so the rule unconditionally
    # refused the only answer available: a perfectly vertical 4-rung or 6-rung column returned
    # "no lane-like rung column found". Below this many bands the rule is not applied. It exists to
    # stop a candidate swallowing several lanes, which cannot happen when there are this few bands.
    max_frac_min_bands=12,
    max_x_std=5.0,        # px: measured about the FITTED LINE (see `gate_line`) or about raw x
    max_x_dev=8.0,        # px: max |x - fitted line| for a rung (same idea, worst case)
    # Which x the narrowness gate measures. True (default) = RAW x, i.e. the behaviour every
    # validated result was produced with: on the 94 labelled gels switching to the line-relative
    # measure moved the image error from 0.683 to 0.719 deg, so it is NOT the default. Set False
    # for strongly tilted gels: a lane leaning by theta has a raw x-spread of tan(theta)*span/12^0.5
    # (~6 px for 6 deg over 250 px), so the raw gate rejects precisely the lanes that lean, and
    # that is a real limitation of this default rather than a tuning choice.
    gate_raw_x=True,
    # NO pitch ceiling. A column of >= min_rungs regularly spaced bands sharing one x is a valid
    # tilt signal whether it is a ladder or a SAMPLE LANE: sample lanes are highly regular too,
    # and their spacing is set by the experiment rather than by a ladder's size series, so judging
    # them against a ladder pitch (13-26 px here) rejects good measurements. Ladders are merely
    # tried FIRST because most sample lanes carry too few bands to fit; they are not the only
    # valid structure. The gates that do the work are min_rungs, regularity (split_ratio against
    # the gaps), narrowness (max_x_std / max_x_dev), elongation, and agreement between columns.
    max_pitch=None,       # None = no pitch ceiling; set a number to re-enable the old gate
    # FALLBACK when no column with >= min_rungs exists at all: try 3-rung columns. A real but
    # weak measurement - short lever arm, only two gaps - so it is flagged (fallback=True, warning
    # carrying the se) and only attempted when the primary pass finds nothing.
    fallback_min_rungs=2,
    r2_warn=0.90,         # lane-line R^2 below this raises a warning (never a veto)
    min_span_warn=70.0,   # px: a lane shorter than this is a short lever arm -> warning
    # How many lanes may be emitted per image. This is a hard cap: the extraction loop stops after
    # emitting this many, which is why it silently truncates multi-lane gels. `20260403 3030 1450
    # 825 RNA` holds 16 findable lanes, of which 4 were emitted at a cap of 4, grouping 13.8% of
    # its detections; at a cap of 16 it groups 55.2%. Measured over the 94 labelled gels, 4 -> 8
    # improves mean |err| 0.705 -> 0.696 and the median 0.663 -> 0.628; beyond 8 the extra lanes
    # are real but add nothing to the estimate (0.697 at 16, 0.699 at 32). What stops the loop
    # early enough in practice is `min_score`, not this cap.
    max_ladders=8,
    min_score=6.0,        # score floor for a candidate lane
    agree_tol=1.5,        # deg: other lanes within this of the best are pooled in
    max_tilt=8.0,         # deg: beyond this the grouping is suspect, not the gel
    # ---- trustworthiness scoring and the pooled estimate -------------------------------
    # `trust` is the score used to rank lanes and to weight the final answer. It measures ONE
    # thing: how precisely this lane pins its own angle. Everything is in px or deg, no fitted
    # constants, except the floor `lane_se_floor_deg` below.
    curve_ref_px=1.0,     # px: rung-to-line width that counts as "straight" for a given span
    curve_weight=0.4,     # weight of the straightness term relative to the precision term
    # Spacing consistency. A lane is one series, so its own spacings must be mutually similar: a
    # geometric ladder size series grows ~1.2x per step, so even 8 rungs span only ~3.5x, while a
    # lane stitched from several well rows has gaps many times longer than its shortest. The old
    # defence was `max_gap > split_ratio * pitch`, and `pitch` is the MEDIAN - which a stitched lane
    # drags to its smallest gap or hides entirely. Measured on the labelled gels, a lane whose
    # longest spacing exceeds 2x its shortest is much more likely to be stitched (from
    # `20260403 35160`: L0 gaps 367 24 302 spans three well rows, while the real lanes have gaps
    # 16 20 16 20 25 19). It is a SOFT penalty, not a reject: a stitched lane stays available but
    # ranks below the tighter lanes.
    gap_consistent_ratio=3.0,
    gap_penalty_power=1.0,
    # A gap that is a near-integer multiple of the pitch COULD be missed bands in one lane rather
    # than a hole: `G1-NE` is y = 404 417 432 432 623, i.e. gaps 13 15 0 191, where 191 ~ 13 * 14.7.
    # Tolerating that was measured and made things clearly WORSE - mean |err| 0.698 -> 0.828, max
    # 1.93 -> 3.08 - and did not even raise grouping (36.9% -> 37.4% of detections). The reason is
    # that this tolerance is exactly what the lane-splitting rule exists to stop: a hole
    # accidentally near a multiple lets a lane span two wells, and a long ragged lane then wins on
    # span. It is therefore DISABLED by default (a huge ratio accepts nothing) and kept only as a
    # documented switch. See LADDER_ALGORITHM.md 14.
    missed_band_ratio=1e9,
    missed_band_tol=0.30,
    # Per-lane uncertainty. A lane's own fit residual understates its error: it cannot see the
    # jitter from dropping one or two weak rungs, nor the centroid bias, nor the gel warp shared by
    # every lane. Measured on the 94 labelled gels, the error actually made per lane is 0.72 deg
    # (3-4 rungs), 0.72 (5-6) and 0.69 (7+) against per-lane `se_deg` of 0.30 / 0.25 / 0.14 - so the
    # understatement is close to a constant FACTOR, which is what `lane_se_scale` is. It is the
    # single number that turns the fitted se into one comparable with the error actually made.
    lane_se_scale=5.5,
    lane_se_floor_deg=0.05,   # numerical guard only; the scale does the work
    # Assumed 1-sigma localisation error of a detected band centroid, in px. This is the ONLY
    # precision information available for a lane fitted to two rungs, whose residual is identically
    # zero; it makes the reported se scale as 1/lever, so a two-rung lane spanning 40 px is reported
    # as far less certain than one spanning 300 px. Measured band components are ~25-45 px wide and
    # ~11-14 px tall, so a sub-pixel centroid is plausible; the value is calibrated against the
    # error actually made on the 94 labelled gels.
    centroid_sigma_px=0.5,
    # Robust rejection of a lane whose angle disagrees with the others: drop beyond mad_k MAD of
    # the median angle, but never reject for less than mad_floor_deg (a set of lanes agreeing to
    # 0.05 deg must not reject a lane 0.2 deg away just because its own MAD is tiny).
    mad_k=3.0,
    mad_floor_deg=0.35,
    # Which estimator is reported as `angle`. The pooled shared-slope fit over the kept lanes'
    # rungs ('rungs') and the inverse-variance pool of their angles ('invvar') are two estimators
    # of the same quantity; over the 94 labelled gels 'rungs' was better on every measure
    # (mean |err| 0.698 vs 0.706, within 1 deg 77% vs 73%), so it is the default. 'invvar' is the
    # one that propagates a per-lane uncertainty, so it is still computed and reported.
    primary_estimator='rungs',
    # Random-effects threshold: `phi` above this is reported as a warning. phi is Cochran's Q/dof
    # from the inverse-variance pool, i.e. how much more the lanes disagree than their own errors
    # allow.
    phi_warn=2.0,
    # Optional empirical correction for the measured over-rotation bias. OFF by default: it is
    # fitted on the labelled set, so it is a calibration parameter, not a mechanism.
    bias_slope=0.0,       # deg of bias per deg of |angle|
    bias_offset=0.0,      # deg of constant bias
)


def fit_line(p, w=None):
    """Weighted least squares x = a + s*(y - ybar). Returns (s, a, ybar)."""
    if w is None:
        w = np.ones(len(p))
    w = np.asarray(w, float)
    yb = float((w * p[:, 1]).sum() / w.sum())
    dy = p[:, 1] - yb
    den = float((w * dy * dy).sum())
    a = float((w * p[:, 0]).sum() / w.sum())
    s = float((w * (p[:, 0] - a) * dy).sum() / den) if den > 0 else 0.0
    return s, a, yb


def slope_of(p):
    return fit_line(p)[0]


def lane_geometry(p):
    """Everything about ONE lane that says how precisely it pins its own angle.

    The angle of a lane is the slope of a straight line through its rungs, so its uncertainty is
    governed by the perpendicular rung-to-line scatter and the lever arm sqrt(sum (y-ybar)^2).
    Both are returned, plus the scatter measured about the FITTED LINE rather than about raw x:
    a lane leaning by theta necessarily has a raw x-spread of about tan(theta)*span/sqrt(12)
    (~6 px for 6 deg over 250 px), which raw-x measures misread as a wide, non-columnar lane.

    `se_deg` is the slope standard error in degrees - the actionable precision number. `half_gap`
    (upper-half slope vs lower-half slope) is carried as a diagnostic only: measured over the 94
    labelled gels it does NOT predict a lane's angular error (r = +0.08), so it is not scored on.
    """
    n = len(p)
    s, a, yb = fit_line(p)
    dy = p[:, 1] - yb
    resid = p[:, 0] - (a + s * dy)
    dof = max(n - 2, 1)
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((p[:, 0] - p[:, 0].mean()) ** 2).sum())
    scale = np.sqrt(1.0 + s * s)
    # perpendicular distance to the fitted line: the geometrically correct residual
    dist = resid / scale
    rms_perp = float(np.sqrt((dist ** 2).sum() / dof))
    lever = float(np.sqrt(max(float((dy ** 2).sum()), 1e-9)))
    se_slope = rms_perp / lever
    st = dict(
        s=float(s), angle=float(-np.degrees(np.arctan(s))), ybar=float(yb),
        rms=float(np.sqrt(ss_res / dof)), rms_perp=rms_perp,
        se_deg=float(np.degrees(se_slope / (1.0 + s * s))),
        se_deg_line=float(np.degrees(se_slope)),
        r2=float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 1.0,
        x_std=float(p[:, 0].std()), x_std_line=float(np.sqrt((resid ** 2).mean())),
        x_dev=float(np.abs(p[:, 0] - p[:, 0].mean()).max()), x_dev_line=float(np.abs(resid).max()),
        lever=lever,
    )
    if n >= 4:
        h = n // 2
        st['half_gap'] = float(abs(-np.degrees(np.arctan(slope_of(p[:h])))
                                  + np.degrees(np.arctan(slope_of(p[h:])))))
    else:
        st['half_gap'] = float('nan')
    return st


def trustworthiness(st):
    """How much this lane's own angle can be believed, as a number to rank and weight lanes by.

    ONE quantity is being estimated: the precision of this lane's angle, in px of rung scatter
    per px of lever arm. Two terms, both in those natural units and neither fitted:

      precision    = lever / (rms_perp + curve_ref_px)
                     a longer lane and a tighter line both pin the angle better. `curve_ref_px`
                     (1 px) only stops a zero-scatter lane from scoring infinite confidence.
      straightness = exp(-curve_weight * rms_perp / lever)
                     a line whose rung scatter grows with its own length is no longer a line, so
                     its "slope" is a summary of a curve rather than a tilt. This term is the only
                     place a lane's span and its straightness interact, and it is scale-free.

    This is deliberately NOT the old `elong * exp(-0.3*cv)`: spacing regularity (cv) and elongation
    were both measured to be poor proxies (`cv`: r = -0.02 against per-lane angular error), and
    they encoded a ladder prior - long, dense, regular - that a valid sample lane need not satisfy.
    """
    prec = st['lever'] / (st['rms_perp'] + P['curve_ref_px'])
    straight = float(np.exp(-P['curve_weight'] * st['rms_perp'] / max(st['lever'], 1e-9)))
    return float(prec * straight)


def missed_band_gaps(y):
    """Gaps in this lane that are near-integer multiples of its own median spacing.

    A ladder with a weak rung - or a sample lane with a faint band - shows a gap of 2x, 3x ...
    its usual pitch, not a hole. `G1-NE` (y = 404 417 432 432 623) has gaps 13 15 0 191, and
    191 ~ 13 * 14.7, so it is missed bands rather than two structures. Returns the boolean mask
    over `np.diff(sorted(y))`, and the median pitch.
    """
    y = np.sort(np.asarray(y, float))
    g = np.diff(y)
    if len(g) == 0:
        return np.zeros(0, bool), float('nan')
    pos = g[g > 0]
    pitch = float(np.median(pos)) if len(pos) else float('nan')
    out = np.zeros(len(g), bool)
    if not np.isfinite(pitch) or pitch <= 0:
        return out, pitch
    for i, v in enumerate(g):
        if v <= P['missed_band_ratio'] * pitch:
            continue
        m = int(round(v / pitch))
        if m >= 2 and abs(v - m * pitch) <= P['missed_band_tol'] * m * pitch:
            out[i] = True
    return out, pitch


def largest_run_keep_missed(o, y):
    """Largest run of rungs uninterrupted in y, but NOT broken by a missed band.

    Same as the old `_largest_run` split, except a gap that is a near-integer multiple of the
    pitch counts as a missed band inside the lane instead of a hole between two structures.
    """
    if len(o) < 2:
        return list(o)
    missed, pitch = missed_band_gaps(y)
    is_break = np.zeros(max(len(y) - 1, 0), bool)
    gi = 0
    for i in range(len(y) - 1):
        d = y[i + 1] - y[i]
        if d <= 0:
            continue
        brk = d > P['split_ratio'] * pitch if np.isfinite(pitch) and pitch > 0 else False
        if brk and gi < len(missed) and missed[gi]:
            brk = False
        is_break[i] = brk
        gi += 1
    runs = np.split(np.asarray(o), np.nonzero(is_break)[0] + 1)
    return list(max(runs, key=len))


def spacing_consistency_penalty(y):
    """Soft factor in (0, 1]: 1.0 when this lane's own spacings are mutually consistent.

    `max_gap / min_gap` measures how far a lane is from one regularly spaced series. The reference
    is the SHORTEST spacing, because that is the only quantity a stitched lane cannot fake: adding
    a distant well row adds a large gap, never a small one, whereas the median is dragged around by
    the very gaps under test. A gap beyond `gap_consistent_ratio` x the shortest spacing is treated
    as evidence of a hole and the score is scaled by `(ratio / r) ** gap_penalty_power`.
    """
    g = np.diff(np.sort(np.asarray(y, float)))
    g = g[g > 0]
    if len(g) < 2:
        return 1.0
    base = max(float(g.min()), 1e-9)
    r = float(g.max()) / base
    if r <= P['gap_consistent_ratio']:
        return 1.0
    return float((P['gap_consistent_ratio'] / r) ** P['gap_penalty_power'])


def calibrate_se(se_deg, lever=0.0, s=0.0):
    """Turn a lane's fitted slope standard error into one comparable with the error actually made.

    Two independent shortfalls are covered:

    (a) FITTED RESIDUAL vs ERROR MADE. A least-squares residual cannot see error that is not in the
        residuals: dropping one or two weak rungs moves a 4-7 rung lane by 0.5-1.2 deg while its
        own `se_deg` stays near 0.1, and every lane in an image shares the gel's warp. Measured over
        the 94 labelled gels that shortfall is close to a constant factor (`lane_se_scale`).

    (b) A LINE FITTED TO ONLY TWO RUNGS HAS ZERO RESIDUAL BY CONSTRUCTION. Its `se_deg` is exactly
        0, so (a) alone reports se = 0.05 deg for a lane that is wrong by 1.1 deg. The precision of
        a two-point line is not set by its residuals - there are none - but by how well the two
        centroids are LOCALISED: an x error of `centroid_sigma_px` over a lever arm `lever` is a
        slope error of `centroid_sigma_px * sqrt(2) / lever`. Adding that term in quadrature makes
        the reported uncertainty a function of the lever arm, which is the physically right
        behaviour and is what the residual-based term cannot provide.

    Returns the slope standard error in degrees, converted back through the same linearisation used
    for `se_deg`.
    """
    # (a) is already a slope standard error in DEGREES, so it is used as-is.
    a_term = max(se_deg, 0.0) * P['lane_se_scale']
    # (b) is a slope error in slope units; convert it with the same linearisation `lane_geometry`
    # uses for se_deg, i.e. divide by (1 + s^2), and only then take it in quadrature with (a).
    # (Taking it in quadrature BEFORE converting mixes the two unit systems.)
    b_slope = (P['centroid_sigma_px'] * np.sqrt(2.0) / lever) if lever > 0 else 0.0
    b_term = np.degrees(b_slope / max(1.0 + s * s, 1e-9))
    return float(np.hypot(a_term, b_term) + P['lane_se_floor_deg'])


def _evaluate(xy, idx, s):
    """Acceptance test + score for a candidate rung set. Returns (idx, s, ybar, score, stats)."""
    p = xy[idx]
    if len(p) == 0:
        return None
    y = p[:, 1]
    span = float(np.ptp(y))
    if span < P['min_span']:
        return None
    g = np.diff(np.sort(y))
    g = g[g > 0]
    if len(g) == 0:
        return None
    pitch = float(np.median(g))
    if pitch < P['min_gap']:                        # rungs on top of each other: not a lane
        return None
    # A gap that is a near-integer multiple of the pitch is MISSED BANDS in one lane, not a hole
    # between two structures, so it is tolerated here. Only a large NON-multiple gap is a hole.
    missed, _ = missed_band_gaps(y)
    gap_is_hole = g > P['split_ratio'] * pitch
    if len(missed) == len(gap_is_hole):
        gap_is_hole = gap_is_hole & ~missed
    if gap_is_hole.any():                           # hole in the column: two lanes, not one
        return None
    cv = float(g.std() / max(pitch, 1e-9))
    if P.get('max_pitch') is not None and pitch > P['max_pitch']:
        return None
    # Geometry measured about the FITTED LINE, never about raw x. See `lane_geometry`: a lane
    # leaning by theta has a raw x-spread of about tan(theta)*span/sqrt(12) (~6 px for 6 deg over
    # 250 px), so a raw-x narrowness gate rejects exactly the lanes that lean - which is the
    # signal we are trying to measure. Measuring about the line makes the gate slope-blind in the
    # right direction: it still rejects a diagonal band ROW, for which the rungs are not collinear.
    st = lane_geometry(p)
    x_spread = st['x_std'] if P['gate_raw_x'] else st['x_std_line']
    x_worst = st['x_dev'] if P['gate_raw_x'] else st['x_dev_line']
    elong = span / (x_spread + 0.5)
    if elong < P['min_elong']:
        return None
    if x_spread > P['max_x_std'] or x_worst > P['max_x_dev']:
        return None                                  # the "column" is a diagonal band row
    st.update(n=int(len(idx)), span=span, pitch=pitch, max_gap=float(g.max()),
              gap_cv=cv, elong=elong,
              consistency=spacing_consistency_penalty(y),
              trust=trustworthiness(st) * spacing_consistency_penalty(y),
              se_deg_cal=calibrate_se(st['se_deg'], st['lever'], st['s']))
    # Two summaries of the same lane, kept side by side because they answer different questions.
    # `trust` is the trustworthiness score (angular precision) and is what ranks and weights lanes.
    # `score` is the elongation-and-regularity heuristic, an independent second opinion: it is the
    # only other signal available for a caller that wants to compare rankings.
    st['score'] = elong * float(np.exp(-0.3 * cv))
    return idx, float(s), float(np.mean(y)), float(st['trust']), st


def _largest_run(sub, idx, s, a, ybar):
    """Largest contiguous run of points (sorted by y) that is both geometrically on the line
    (|x - line| <= dx_tol) and uninterrupted in y (no gap above split_ratio * median gap).

    A ladder is one uninterrupted column. Anything off the line is a different structure that
    happened to fall inside the tube, and anything past a big hole is a second gel or a band
    row - neither may be averaged into the line.
    """
    idx = np.asarray(idx)
    if len(idx) == 0:
        return idx
    on = np.abs(sub[idx, 0] - (a + s * (sub[idx, 1] - ybar))) <= P['dx_tol']
    idx = idx[on]
    if len(idx) < 2:
        return idx
    o = idx[np.argsort(sub[idx, 1])]
    y = sub[o, 1]
    if len(np.unique(y)) < 2:
        return o
    return np.asarray(largest_run_keep_missed(o, y))


def _core(sub, ton, xc, s=0.0, ybar=0.0, a=0.0):
    """The rung column inside the inlier tube: the tightest x-cluster around the anchor `xc`,
    found by cutting the sorted distances at the largest relative jump.

    Distances are measured PERPENDICULAR TO THE TUBE LINE, not as raw |x - xc|. For a lane leaning
    by theta the perpendicular direction is cos(theta) narrower than the horizontal one, so a raw
    |x - xc| ranking is systematically skewed: it counts `|x - line| / cos(theta)` and therefore
    over-penalises the rungs at the far end of a leaning lane, which is exactly the part carrying
    the slope information. Measured effect of the fix together with the slope-aware narrowness
    gate: the slope-blind versions cost nothing on near-vertical gels but reject real leaning ones.

    The cut is relative (a jump of 2.5x the typical spacing before it) rather than a fixed width,
    because the right width differs per gel: a dense lane is one rung per 13 px of y with rungs
    spread over 2 px of x, while another is spread over 12 px, and a fixed window either clips
    the wide one or swallows the isolated sample bands next to the narrow one. Finally the
    largest uninterrupted run in y is kept, so a band row is not averaged in with the rungs."""
    if len(ton) < 2:
        return list(ton)
    ton = np.asarray(ton)
    scale = np.sqrt(1.0 + s * s)
    d_all = np.abs(sub[ton, 0] - (a + s * (sub[ton, 1] - ybar))) / scale
    order = np.argsort(d_all)
    d = d_all[order]
    gaps = np.diff(d)
    cut = len(d)
    if len(gaps):
        base = np.median(gaps[:max(3, len(gaps) // 2)])
        j = int(np.argmax(gaps / np.maximum(base, 0.5)))
        if gaps[j] > 2.5 * max(base, 0.5) and j + 1 >= P['min_rungs_prior']:
            cut = j + 1
    win = list(ton[order][:cut])
    o = np.array(win)[np.argsort(sub[win, 1])]
    y = sub[o, 1]
    if len(np.unique(y)) < 2:
        return list(o)
    return largest_run_keep_missed(o, y)


def _candidate(sub, x0, y0, s0, a0=None, yb0=None, fix0=False):
    """Refine the hypothesis line through (x0, y0) with slope s0, then evaluate the rung set.

    The first pass uses the wide window so every rung of the column gets in even though the seed
    anchor is only approximate; for a pair-seeded hypothesis (fix0) the line is forced through
    the two rungs that defined it, so a distant singleton sharing the tube cannot hijack the
    first fit. Later passes fit the CORE x-cluster only, and the final membership is the largest
    uninterrupted run inside the strict window."""
    ybar = float(yb0 if yb0 is not None else y0)
    a = float(a0 if a0 is not None else x0)
    s = float(s0)
    xc = float(a)
    for k in range(P['max_rounds']):
        fixed = fix0 and k == 0
        tol = P['dx_tol_wide'] if k == 0 else P['dx_tol']
        ton = np.nonzero(np.abs(sub[:, 0] - (a + s * (sub[:, 1] - ybar))) <= tol)[0]
        core = _core(sub, ton, xc, s, ybar, a)
        if len(core) < 2:
            return None
        s_new = s if fixed else slope_of(sub[core])
        a_new = float(np.mean(sub[core, 0] - s_new * (sub[core, 1] - ybar)))
        y_new = ybar if fixed else float(sub[core, 1].mean())
        xc = float(a_new + s_new * (np.median(sub[core, 1]) - y_new))
        if k > 0 and abs(s_new - s) < 1e-7:
            s, a, ybar = s_new, a_new, y_new
            break
        s, a, ybar = s_new, a_new, y_new
    idx = np.nonzero(np.abs(sub[:, 0] - (a + s * (sub[:, 1] - ybar))) <= P['dx_tol'])[0]
    idx = _largest_run(sub, idx, s, a, ybar)
    if len(idx) < P['min_rungs']:
        return None
    s = slope_of(sub[idx])                              # refit on the final rung set
    return _evaluate(sub, idx, s)


def detect_ladders(xy, score_floor=None, min_rungs=None, max_pitch=None, slope_window=None):
    """Greedy score-and-mask lane extraction.

    Returns a list of dicts, best score first: idx (indices into xy), s, angle, ybar, score,
    trust, and the stats from _evaluate. `min_rungs` / `max_pitch` override P for this call only
    (used by the 3-rung fallback); `slope_window = (centre, half_width)` in tan units narrows the
    slope hypotheses to a consensus +- half_width for a second-pass recall of weak lanes that
    agree with the tilt already measured. P is always restored.
    """
    xy = np.asarray(xy, float)
    floor = P['min_score'] if score_floor is None else score_floor
    old = (P['min_rungs'], P['max_pitch'])
    if min_rungs is not None:
        P['min_rungs'] = int(min_rungs)
    if max_pitch is not None:
        P['max_pitch'] = float(max_pitch)
    try:
        return _detect_ladders(xy, floor, slope_window=slope_window)
    finally:
        P['min_rungs'], P['max_pitch'] = old


def _detect_ladders(xy, floor, slope_window=None):
    if slope_window is not None:
        # Every lane in one gel shares the tilt, so once a first pass has measured it there is no
        # reason to keep hypothesising 11-deg diagonals. Sweeping the hypotheses around that
        # consensus instead makes short, sparse lanes findable: a lane with 3-4 bands rarely wins
        # the open sweep against a long one, and the fixed slope grid hits its true slope only by
        # luck. This is a RECALL pass - it adds lanes, it does not re-rank the ones already found.
        c = float(np.clip(slope_window[0], -P['slope_hi'], P['slope_hi']))
        half = max(float(slope_window[1]), 1e-4)
        slopes = np.unique(np.concatenate([
            np.linspace(c - half, c + half, max(P['n_slope'] // 2, 5)),
            np.linspace(-P['slope_hi'], P['slope_hi'], P['n_slope'])]))
    else:
        slopes = np.linspace(-P['slope_hi'], P['slope_hi'], P['n_slope'])
    alive = np.ones(len(xy), bool)
    lanes = []
    for _ in range(P['max_ladders']):
        idx_alive = np.nonzero(alive)[0]
        if len(idx_alive) < P['min_rungs']:
            break
        sub = xy[idx_alive]
        # Hypothesis lines are seeded from PAIRS of detections in the same x neighbourhood: a
        # pair of rungs on one column defines that column's true slope exactly, whereas a slope
        # sweep only hits it by luck (and an unhit slope steers the tube onto a diagonal slice
        # of the gel instead). A flat centroid plus a sweep of slopes is added for coverage of
        # columns whose rungs were not detected.
        pairs = []
        n = len(sub)
        if n <= P['pair_max_n']:
            for i in range(n):
                dy = np.abs(sub[:, 1] - sub[i, 1])
                dx = np.abs(sub[:, 0] - sub[i, 0])
                for j in np.nonzero((dy >= P['pair_min_dy']) & (dx <= P['pair_max_dx']))[0]:
                    if j <= i:
                        continue
                    s = (sub[j, 0] - sub[i, 0]) / (sub[j, 1] - sub[i, 1])
                    if abs(s) <= P['slope_hi']:
                        yb = 0.5 * (sub[i, 1] + sub[j, 1])
                        a = float(sub[i, 0] - s * (sub[i, 1] - yb))
                        pairs.append((float(sub[i, 0]), float(sub[i, 1]), float(s), a, float(yb)))
        if len(pairs) > P['max_seeds']:
            sel = np.unique(np.linspace(0, len(pairs) - 1, P['max_seeds']).astype(int))
            pairs = [pairs[i] for i in sel]
        best = None
        for hyp in list(pairs) + [(float(sub[:, 0].mean()), float(sub[:, 1].mean()), float(s), None, None)
                                  for s in slopes]:
            x0, y0, s0 = hyp[0], hyp[1], hyp[2]
            cand = _candidate(sub, x0, y0, s0, a0=hyp[3], yb0=hyp[4], fix0=hyp[3] is not None)
            if cand is None:
                continue
            idx, s, ybar, score, st = cand
            if len(idx) > P['max_frac'] * len(xy) and len(xy) >= P['max_frac_min_bands']:
                continue
            if best is None or score > best[3]:
                best = (idx, s, ybar, score, st)
        if best is None or best[3] <= floor:
            break
        idx, s, ybar, score, st = best
        entry = dict(st)
        entry.update(idx=idx_alive[idx], ybar=ybar, score=score, trust=score)
        lanes.append(entry)
        alive[idx_alive[idx]] = False
    return lanes


def pooled_lane_angles(lanes):
    """The final angle: a random-effects meta-analysis of the per-lane angles.

    Each lane contributes an independent estimate of the SAME quantity - the gel's tilt - so the
    statistically sound way to combine them is inverse-variance weighting on its own uncertainty:

        theta_hat = sum(w_i * theta_i) / sum(w_i),      w_i = 1 / se_i^2

    with `se_i` the lane's calibrated slope standard error (`se_deg_cal`). Two corrections make
    this honest rather than merely tidy:

    ROBUST REJECTION. Weighted least squares has an unbounded influence function, so one lane
    whose angle is off by degrees (a merged structure, a band row that slipped through a gate)
    would be weighted by its own - small - `se` and drag the answer. Lanes outside `mad_k` MAD of
    the median angle are dropped, iterated to convergence.

    OVERDISPERSION (Cochran's Q). If the lanes disagree by MORE than their individual `se`
    predicts, the excess is real structure - lane curvature, per-lane detection bias - that no
    amount of averaging removes, and the error bar must widen instead of the disagreement being
    averaged away:

        Q   = sum(w_i (theta_i - theta_hat)^2),  dof = k - 1
        phi = max(1, Q / dof)                    (1 = the lanes agree with their own errors)
        se  = sqrt(phi / sum(w_i))

    Returns (angle_deg, se_deg, k_used, phi, kept_lanes).
    """
    if not lanes:
        return float('nan'), float('nan'), 0, float('nan'), []
    ang = np.array([l['angle'] for l in lanes], float)
    se = np.array([max(l.get('se_deg_cal', l.get('se_deg', 0.35)), 1e-6) for l in lanes], float)
    keep = np.ones(len(ang), bool)
    for _ in range(4):
        w = 1.0 / se[keep] ** 2
        med = float(np.median(ang[keep]))
        mad = 1.4826 * float(np.median(np.abs(ang[keep] - med)))
        tol = max(P['mad_k'] * max(mad, 1e-6), P['mad_floor_deg'])
        good = np.abs(ang - med) <= tol
        if good.sum() == keep.sum() or good.sum() < 1:
            break
        keep = good
    w = 1.0 / se[keep] ** 2
    mu = float((w * ang[keep]).sum() / w.sum())
    k = int(keep.sum())
    if k > 1:
        Q = float((w * (ang[keep] - mu) ** 2).sum())
        phi = max(1.0, Q / (k - 1))
    else:
        phi = 1.0
    se_mu = float(np.sqrt(phi / w.sum()))
    return mu, se_mu, k, phi, [l for l, g in zip(lanes, keep) if g]


def pool_lane_geometry(xy, keep):
    """`contrast`: the trust-weighted mean of the kept lanes' own angles.

    An independent second aggregation, reported alongside the primary `angle` rather than used as
    it. It weights a lane by `trust`, a precision proxy, which is the right shape of weight; it
    just ignores each lane's measured `se`, and it never inflates for the lanes disagreeing more
    than their errors allow. `angle` from `pooled_lane_angles` does both.
    """
    wsum = sum(l.get('trust', l.get('score', 0.0)) for l in keep)
    if wsum <= 0:
        return float(np.mean([l['angle'] for l in keep]))
    return float(sum(l.get('trust', l.get('score', 0.0)) * l['angle'] for l in keep) / wsum)


def _pooled(xy, keep):
    """Shared-slope fit over the accepted lanes (one free x-offset each). Cross-check only.

    This pools the rung POSITIONS rather than the per-lane angles, so with independent rung noise
    it is the more efficient estimator. It is kept as a cross-check precisely because it assumes
    what the data shows is not quite true - that rung errors are independent - so when it and
    `pooled_lane_angles` disagree, the disagreement is the diagnostic.
    """
    num = den = 0.0
    for l in keep:
        p = xy[l['idx']]
        w = p[:, 1] - p[:, 1].mean()
        num += float((p[:, 0] - p[:, 0].mean()) @ w)
        den += float(w @ w)
    s = num / den if den > 0 else float(keep[0]['s'])
    ss, dof = 0.0, 0
    for l in keep:
        p = xy[l['idx']]
        w = p[:, 1] - p[:, 1].mean()
        r = (p[:, 0] - p[:, 0].mean()) - s * w
        ss += float(r @ r); dof += max(len(p) - 2, 1)
    dof = max(dof - (len(keep) - 1), 1)
    se = float(np.degrees(np.arctan(np.sqrt(ss / dof / den)))) if den > 0 else np.nan
    return float(-np.degrees(np.arctan(s))), float(s), se


def estimate(xy, params=None, verbose=False):
    """The entry point. Returns `angle` (the primary answer, in degrees), the uncertainty, the
    accepted lanes, and diagnostics including two independent cross-check aggregations.

    `params` overrides P FOR THIS CALL ONLY: P is restored afterwards, so one caller passing a
    modified dict cannot silently change the defaults seen by every later call."""
    old = dict(P)
    try:
        return _estimate(xy, params, verbose)
    finally:
        P.clear(); P.update(old)


def _estimate(xy, params=None, verbose=False):
    if params:
        P.update(params)
    xy = np.asarray(xy, float)
    if len(xy):
        xy = xy[np.isfinite(xy).all(1)]
    out = dict(n_bands=len(xy), angle=np.nan, contrast=np.nan, se=np.nan, ok=False,
               ladders=[], n_ladders=0, reason='')
    if len(xy) < P['fallback_min_rungs']:
        # No lane of any kind is possible. Report against the achievable threshold, not min_rungs:
        # saying "fewer than 4 bands" for an image that only ever needed 2 is misleading.
        out['reason'] = 'fewer than %d bands' % P['fallback_min_rungs']
        return out
    lanes = detect_ladders(xy)
    fallback = False
    if not lanes:
        # No proper lane (>= min_rungs) anywhere in the image. Fall back to 3-rung lanes:
        # a real but weak measurement - short lever arm, only two gaps - so it is flagged and
        # never mixed with proper lanes (it is only attempted when the primary pass is empty).
        lanes = detect_ladders(xy, min_rungs=P['fallback_min_rungs'])
        fallback = bool(lanes)
    if not lanes:
        out['reason'] = 'no lane-like rung column found'
        return out
    if fallback:
        out['reason'] = ('no lane with >=%d rungs; used %d-rung column(s) (weak)'
                         % (P['min_rungs'], P['fallback_min_rungs']))
    # Which lanes get a vote. The reference is the ROBUST MEDIAN of the already-detected lanes,
    # not the single top-scoring one: anchoring the vote on one lane means that lane decides both
    # the answer and who is allowed to disagree with it, so a wrong winner builds a wrong
    # consensus around itself. Detection order (by `trust`) only breaks ties here.
    med = float(np.median([l['angle'] for l in lanes]))
    keep = [l for l in lanes if abs(l['angle'] - med) <= P['agree_tol']]
    if not keep:
        # agree_tol can reject every lane when the median falls between two clusters of lanes.
        # The strongest lane is then the only defensible answer, and `_pooled` would otherwise
        # index an empty list.
        keep = [max(lanes, key=lambda l: l.get('trust', l.get('score', 0.0)))]
    if len(keep) < len(lanes):
        out['reason'] = ('lanes disagree by more than %.1f deg about the median %.3f; '
                         'used the %d agreeing one(s)' % (P['agree_tol'], med, len(keep)))
    # PRIMARY. Two estimators of the same quantity - see `primary_estimator`. `angle_rungs` pools
    # the rung POSITIONS with one free x-offset per lane (a shared-slope fit); `angle_invvar`
    # pools the per-lane ANGLES by inverse variance with MAD outlier rejection and a Cochran-Q
    # overdispersion inflation. The first is more efficient when rung errors are independent; the
    # second is the one that carries a per-lane uncertainty model and is robust to one bad lane.
    ang_rung, s, se_rung = _pooled(xy, keep)
    angle_invvar, se_invvar, k_used, phi, pooled = pooled_lane_angles(keep)
    if k_used == 0:
        pooled = keep
        angle_invvar = float(keep[0]['angle'])
        se_invvar = float(keep[0].get('se_deg_cal', np.nan))
        k_used, phi = 1, 1.0
    if P['primary_estimator'] == 'invvar':
        angle, se = float(angle_invvar), float(se_invvar)
    else:
        angle = float(ang_rung)
        # The reported UNCERTAINTY always comes from the lane-level model, never from the rung
        # fit's own se: a rung-level residual is the scatter of detections about one line, which
        # understates the error actually made by the factor `lane_se_scale` (measured), and it has
        # no way to express that the lanes in one image err together. `se_rungs` is still reported
        # for comparison, but it is not the error bar.
        se = float(se_invvar) if np.isfinite(se_invvar) else float(se_rung)
    # Second cross-check: the trust-weighted mean of the lane angles.
    contrast = pool_lane_geometry(xy, pooled)
    # Empirical over-rotation correction, OFF by default (bias_slope = bias_offset = 0). It is a
    # calibration fitted on the 94 labelled gels, not a mechanism, so it is opt-in and reported.
    angle_raw = angle
    if P['bias_slope'] or P['bias_offset']:
        angle = float(angle - (P['bias_offset'] + P['bias_slope'] * abs(angle)))
    out.update(slope=float(-np.tan(np.radians(angle))), angle=float(angle),
               angle_raw=float(angle_raw),
               angle_rungs=float(ang_rung), se_rungs=float(se_rung),
               angle_invvar=float(angle_invvar), se_invvar=float(se_invvar),
               angle_trust_weighted=float(contrast), contrast=float(contrast),
               se=float(se), ladders=lanes,
               kept=len(pooled), n_ladders=len(lanes), fallback=fallback,
               overdispersion=float(phi), lanes_used=int(k_used),
               min_rungs_used=(P['fallback_min_rungs'] if fallback else P['min_rungs']),
               n_rungs=int(sum(len(l['idx']) for l in pooled)),
               best_trust=float(lanes[0].get('trust', lanes[0]['score'])),
               best_score=float(lanes[0].get('trust', lanes[0]['score'])),
               span_max=float(max(l['span'] for l in pooled)),
               x_std_max=float(max(l['x_std_line'] for l in pooled)),
               pitch_max=float(max(l['pitch'] for l in pooled)))
    if abs(angle) > P['max_tilt']:
        out['reason'] = ('implausible tilt %+.1f deg exceeds +-%.1f deg; grouping suspect%s'
                         % (angle, P['max_tilt'], ' (3-rung fallback)' if fallback else ''))
        return out
    out['ok'] = True
    if not out['reason']:
        out['reason'] = ('ok: %d lane(s) pooled of %d, %d rungs, se %.2f deg'
                         % (len(pooled), len(lanes), out['n_rungs'], se))
    # Fit-quality warnings. NOT vetoes: these are reported and the caller decides. Calibration
    # worth knowing: across the labelled gels a low R^2 correlates only ~+0.41 with the actual
    # angle error, so it means "this lane is a weak measurement", not a confidence. `se_deg_cal`
    # is the number to compare against the error actually made.
    warn = []
    if fallback:
        warn.append('%d-rung fallback: no lane with >=%d rungs in this image; a %d-rung column is '
                    'a short lever arm (angle se ~%.2f deg, %d rungs, %.0f px span)'
                    % (P['fallback_min_rungs'], P['min_rungs'], P['fallback_min_rungs'],
                       out['se'], out['n_rungs'], max(l['span'] for l in pooled)))
    for i, l in enumerate(pooled):
        if l.get('r2', 1.0) < P['r2_warn']:
            warn.append('lane %d: line R2 %.2f < %.2f (%d rungs, %.1f px of lean, angle '
                        '+-%.2f deg)'
                        % (i, l['r2'], P['r2_warn'], l['n'], l['x_std'] * np.sqrt(l['n']),
                           l.get('se_deg_cal', float('nan'))))
    if max(l['span'] for l in pooled) < P['min_span_warn']:
        warn.append('longest lane spans only %.0f px (< %.0f): short lever arm'
                    % (max(l['span'] for l in pooled), P['min_span_warn']))
    if phi > P['phi_warn']:
        warn.append('lanes disagree %.1fx more than their own errors predict (Q/dof %.1f): the '
                    'pooled se already carries this inflation' % (phi, phi))
    if len(pooled) == 1 and not fallback:
        warn.append('single lane: no independent consensus, the answer is that lane alone '
                    '(se %.2f deg)' % out['se'])
    if out['angle_raw'] != out['angle']:
        warn.append('empirical over-rotation correction applied: %+.3f -> %+.3f deg'
                    % (out['angle_raw'], out['angle']))
    out['warnings'] = warn
    out['r2_min'] = float(min(l.get('r2', 1.0) for l in pooled))
    out['se_deg'] = float(np.sqrt(sum(l.get('se_deg_cal', 0.0) ** 2 for l in pooled))
                          / max(len(pooled), 1))
    return out
