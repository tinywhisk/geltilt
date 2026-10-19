from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")  # retired engine; optional test dependency

from geltilt.smears import (
    COARSE_STEP,
    STEP,
    estimate,
    gel_mask,
    interior,
    map_chunks,
    persistence,
    rotate_about,
)

H = W = 320
N_STRIPES = 16


def _stripes(smear_offset=0.0, half=None):
    span = W * 0.8
    pitch = span / N_STRIPES
    hw = max(1.5, pitch * 0.16)
    field = np.zeros((H, W), np.float32)
    x0 = (W - span) / 2.0
    for i in range(N_STRIPES):
        cx = x0 + (i + 0.5) * pitch
        a, b = int(round(cx - hw)), int(round(cx + hw)) + 1
        field[:, max(0, a):min(W, b)] = 1.0
    if half is not None:
        lo, hi = half
        keep = np.zeros_like(field)
        keep[:, lo:hi] = 1.0
        field = field * keep
    if smear_offset:
        field = np.clip(rotate_about(field, (W / 2, H / 2), smear_offset), 0, None)
    return field


def make_fixture(tilt=0.0, smear_offset=0.0, edge_offset=0.0, seed=0):
    rng = np.random.default_rng(seed)
    stripes = _stripes(smear_offset)
    slab = np.zeros((H, W), np.float32)
    slab[int(H * 0.1):int(H * 0.9), int(W * 0.12):int(W * 0.88)] = 1.0
    if edge_offset:
        slab = np.clip(rotate_about(slab, (W / 2, H / 2), edge_offset), 0, 1)
    img = 12.0 + 58.0 * slab + 55.0 * slab * stripes
    img = img + rng.normal(0.0, 1.5, img.shape)
    if tilt:
        img = rotate_about(img, (W / 2, H / 2), tilt, border_value=12.0)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_split_fixture(smear_offset=3.0, seed=0):
    rng = np.random.default_rng(seed)
    left = _stripes(smear_offset, half=(0, W // 2))
    right = _stripes(-smear_offset, half=(W // 2, W))
    stripes = np.clip(left + right, 0, 1)
    slab = np.zeros((H, W), np.float32)
    slab[int(H * 0.1):int(H * 0.9), int(W * 0.12):int(W * 0.88)] = 1.0
    img = 12.0 + 58.0 * slab + 55.0 * slab * stripes
    return np.clip(img + rng.normal(0, 1.5, img.shape), 0, 255).astype(np.uint8)


# --- T1: the smears set the angle ----------------------------------------------
@pytest.mark.parametrize("offset", [-3.0, -1.5, 0.0, 1.5, 3.0])
def test_T1_smears_set_the_angle(offset):
    from geltilt.smears import smear_angle

    angle = smear_angle(make_fixture(smear_offset=offset))
    assert abs(angle - (-offset)) <= 0.6, (offset, angle)


# --- T2: rotation equivariance in the body -------------------------------------
@pytest.mark.parametrize("tilt", [-2.0, 2.0])
def test_T2_rotation_equivariant_in_body(tilt):
    from geltilt.smears import smear_angle

    baseline = smear_angle(make_fixture(smear_offset=1.5, tilt=0.0))
    moved = smear_angle(make_fixture(smear_offset=1.5, tilt=tilt))
    assert abs(moved - (baseline - tilt)) <= 0.6, (tilt, baseline, moved)


# --- T3: the slab edges do not set the angle (the key test for the erosion) ----
@pytest.mark.parametrize("edge", [-3.0, 3.0])
def test_T3_slab_edges_do_not_set_the_angle(edge):
    from geltilt.smears import smear_angle

    angle = smear_angle(make_fixture(smear_offset=0.0, edge_offset=edge))
    assert abs(angle) <= 0.8, (edge, angle)


# --- T4: a split gel is refused ------------------------------------------------
def test_T4_split_gel_is_refused():
    ok, angle, info = estimate(make_split_fixture(3.0), gates=True)
    assert ok is False
    assert info["reason"] == "halves_differ"


# --- T5: persistence tells a streak from blobs ---------------------------------
def test_T5_persistence_streak_vs_blobs():
    rows = 36 * 8
    w = 64
    streak = np.zeros((rows, w), np.float32)
    streak[:, [20, 21]] = 1.0
    blobs = np.zeros((rows, w), np.float32)
    for k in range(8):
        c = (2 + k * 7) % (w - 2)
        blobs[k * 36:(k + 1) * 36, [c, c + 1]] = 1.0
    ps, pb = persistence(streak), persistence(blobs)
    assert ps >= 0.9
    assert pb <= 0.3
    assert ps > pb


# --- supporting conformance ----------------------------------------------------
def test_map_chunks_is_bit_identical():
    items = np.linspace(0.0, 1.0, 81)

    def fn(a):
        return np.sin(a).astype(np.float64), np.cos(a).astype(np.float64)

    serial = fn(items)
    parallel = map_chunks(fn, items)
    assert np.array_equal(serial[0], parallel[0])
    assert np.array_equal(serial[1], parallel[1])


def test_gel_mask_and_interior_basic():
    img = make_fixture()
    mask = gel_mask(img)
    assert mask.dtype == np.uint8
    assert mask.shape == img.shape
    assert mask.sum() > 0
    strip = interior(img, mask)
    assert strip is not None
    sub, subm = strip
    assert sub.shape == subm.shape
    assert subm.sum() >= 800