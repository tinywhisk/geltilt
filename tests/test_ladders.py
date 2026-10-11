from __future__ import annotations

import numpy as np
import pytest

from geltilt import ladders


def synthetic_gel(tilt_deg=2.0, n_lanes=6, seed=0):
    """A ladder lane plus sample lanes of 7-9 regularly spaced bands (from the module example)."""
    rng = np.random.default_rng(seed)
    slope = -np.tan(np.radians(tilt_deg))
    pts = []
    for lane in range(n_lanes):
        x0 = 90.0 + lane * 190.0
        if lane == 0:
            ys = np.linspace(100.0, 760.0, 8)
        else:
            n = int(rng.integers(7, 10))
            top = float(rng.uniform(90.0, 170.0))
            pitch = float(rng.uniform(55.0, 85.0))
            ys = np.cumsum(rng.normal(pitch, pitch * 0.06, n)) + top
        for y in ys:
            pts.append([x0 + slope * (y - 430.0) + rng.normal(0, 0.35), y + rng.normal(0, 0.8)])
    return np.array(pts, float), tilt_deg


@pytest.mark.parametrize("tilt", [0.0, 1.5, 2.0, -2.0])
def test_ladders_recovers_known_tilt(tilt):
    xy, truth = synthetic_gel(tilt)
    res = ladders.estimate(xy)
    assert res["ok"] is True
    assert abs(res["angle"] - truth) <= 0.6
    assert res["n_bands"] == len(xy)


def test_ladders_refuses_too_few_bands():
    res = ladders.estimate(np.array([[10.0, 10.0]]))
    assert res["ok"] is False
    assert "bands" in res["reason"]


def test_ladders_angle_sign_convention():
    # a lane with slope s = dx/dy; the angle to apply is -atan(s)
    s = 0.05
    ys = np.linspace(0.0, 300.0, 10)
    x = 100.0 + s * ys
    res = ladders.estimate(np.stack([x, ys], 1))
    assert res["ok"] is True
    assert res["angle"] == pytest.approx(-np.degrees(np.arctan(s)), abs=0.4)


def test_ladders_is_deterministic():
    xy, _ = synthetic_gel(2.5, seed=3)
    assert ladders.estimate(xy)["angle"] == ladders.estimate(xy)["angle"]

def test_ladders_exposes_kept_lanes_additively():
    xy, _ = synthetic_gel(1.5, seed=1)
    res = ladders.estimate(xy)
    assert res["ok"] is True
    assert len(res["kept_lanes"]) == res["kept"]
    for lane in res["kept_lanes"]:
        assert {"s", "ybar", "idx"} <= set(lane)
    # the scalar keys CLI parity relies on are unchanged
    for key in ("angle", "se", "ok", "reason", "n_bands", "n_ladders", "kept", "fallback"):
        assert key in res


def test_ladders_kept_lanes_empty_on_refusal():
    res = ladders.estimate(np.array([[10.0, 10.0]]))
    assert res["ok"] is False
    assert res["kept_lanes"] == []