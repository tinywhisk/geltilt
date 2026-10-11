from __future__ import annotations

import time

import numpy as np

from geltilt.app.ports import EstimateError, ModelUnavailableError
from geltilt.app.service import EstimationService

from .conftest import FakeEstimator, make_outcome


def _wait(service, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = service.poll()
        if result is not None:
            return result
        if not service.busy:
            return None
        time.sleep(0.005)
    raise AssertionError("no result in time")


def test_service_returns_outcome():
    service = EstimationService(FakeEstimator(make_outcome(angle=1.2)))
    service.start("a.png", np.zeros((4, 4)), 0)
    result = _wait(service)
    assert result is not None
    assert result.outcome.angle == 1.2
    assert result.sample == "a.png"


def test_service_maps_model_error():
    service = EstimationService(FakeEstimator(exc=ModelUnavailableError("no model")))
    service.start("a.png", np.zeros((4, 4)), 0)
    result = _wait(service)
    assert result.error_kind == "model"


def test_service_maps_estimate_error():
    service = EstimationService(FakeEstimator(exc=EstimateError("bad inference")))
    service.start("a.png", np.zeros((4, 4)), 0)
    result = _wait(service)
    assert result.error_kind == "estimate"


def test_service_maps_unexpected_error():
    service = EstimationService(FakeEstimator(exc=ValueError("surprise")))
    service.start("a.png", np.zeros((4, 4)), 0)
    result = _wait(service)
    assert result.error_kind == "estimate"
    assert "surprise" in result.error


def test_service_supersedes_earlier_request():
    estimator = FakeEstimator(make_outcome(angle=9.0), delay=0.2)
    service = EstimationService(estimator)
    service.start("first.png", np.zeros((4, 4)), 0)
    time.sleep(0.02)
    service.start("second.png", np.zeros((4, 4)), 0)
    result = _wait(service)
    assert result is not None
    assert result.sample == "second.png"


def test_service_cancel_drops_result():
    service = EstimationService(FakeEstimator(make_outcome(), delay=0.2))
    service.start("a.png", np.zeros((4, 4)), 0)
    service.cancel()
    time.sleep(0.3)
    assert service.poll() is None


def test_service_poll_returns_none_while_running():
    service = EstimationService(FakeEstimator(make_outcome(), delay=0.2))
    service.start("a.png", np.zeros((4, 4)), 0)
    assert service.poll() is None
    result = _wait(service)
    assert result is not None