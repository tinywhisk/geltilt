from __future__ import annotations

import numpy as np
import pytest

from geltilt.app import estimator as estimator_mod
from geltilt.app.estimator import CanonicalEstimator
from geltilt.app.ports import EstimateError, ModelUnavailableError


def _fake_xy():
    return np.array([[10.0, 100.0], [12.0, 200.0], [30.0, 100.0], [32.0, 200.0]], float)


def _canonical_ok_dict(xy):
    kept_lanes = [
        {"s": 0.05, "angle": -2.86, "ybar": 150.0, "idx": np.array([0, 1])},
        {"s": 0.05, "angle": -2.86, "ybar": 150.0, "idx": np.array([2, 3])},
    ]
    return {
        "ok": True, "angle": 2.4, "se": 0.31, "reason": "ok",
        "warnings": ["short lever arm"], "n_bands": len(xy), "n_ladders": 3,
        "kept": 2, "fallback": False, "kept_lanes": kept_lanes,
    }


def test_adapter_projects_canonical_dict(monkeypatch):
    xy = _fake_xy()
    monkeypatch.setattr(estimator_mod, "default_segmenter", lambda path=None: object())
    monkeypatch.setattr(estimator_mod, "band_centroids", lambda image, **kw: xy)
    monkeypatch.setattr(estimator_mod.ladders, "estimate", lambda points: _canonical_ok_dict(points))

    est = CanonicalEstimator()
    outcome = est.estimate(np.zeros((8, 8), np.uint8))

    assert outcome.ok is True
    assert outcome.angle == pytest.approx(2.4)
    assert outcome.se == pytest.approx(0.31)
    assert outcome.n_bands == 4
    assert outcome.lanes_used == 2
    assert outcome.lanes_found == 3
    assert outcome.fallback is False
    assert outcome.warnings == ("short lever arm",)
    assert len(outcome.accepted_lanes) == 2
    assert outcome.accepted_lanes[0].slope == pytest.approx(0.05)
    assert outcome.accepted_lanes[0].points.shape == (2, 2)
    assert outcome.centroids.shape == (4, 2)


def test_adapter_refusal_is_not_an_exception(monkeypatch):
    monkeypatch.setattr(estimator_mod, "default_segmenter", lambda path=None: object())
    monkeypatch.setattr(estimator_mod, "band_centroids", lambda image, **kw: np.array([[1.0, 1.0]]))
    monkeypatch.setattr(
        estimator_mod.ladders, "estimate",
        lambda points: {"ok": False, "angle": float("nan"), "reason": "fewer than 4 bands",
                        "n_bands": 1},
    )
    outcome = CanonicalEstimator().estimate(np.zeros((4, 4), np.uint8))
    assert outcome.ok is False
    assert outcome.accepted is False
    assert "bands" in outcome.reason


def test_adapter_maps_missing_model(monkeypatch):
    def _boom(_path=None):
        raise FileNotFoundError("model missing")

    monkeypatch.setattr(estimator_mod, "default_segmenter", _boom)
    with pytest.raises(ModelUnavailableError):
        CanonicalEstimator().estimate(np.zeros((4, 4), np.uint8))


def test_adapter_maps_segmentation_failure(monkeypatch):
    monkeypatch.setattr(estimator_mod, "default_segmenter", lambda path=None: object())

    def _boom(image, **kw):
        raise RuntimeError("inference blew up")

    monkeypatch.setattr(estimator_mod, "band_centroids", _boom)
    with pytest.raises(EstimateError):
        CanonicalEstimator().estimate(np.zeros((4, 4), np.uint8))