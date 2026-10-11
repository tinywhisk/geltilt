from __future__ import annotations

import numpy as np
import pytest

from geltilt.app.domain import (
    ANGLE_MAX,
    AppliedRotation,
    EstimateOutcome,
    LaneLine,
    SessionState,
    clamp_angle,
    make_applied_rotation,
)


def test_clamp_angle_bounds_and_rounding():
    assert clamp_angle(200.0) == ANGLE_MAX
    assert clamp_angle(-500.0) == ANGLE_MAX * -1
    assert clamp_angle(2.34) == 2.3
    assert clamp_angle(2.36) == 2.4


def test_clamp_angle_rejects_non_finite():
    with pytest.raises(ValueError):
        clamp_angle(float("nan"))


def test_source_none_implies_zero():
    rot = AppliedRotation(5.0, "none")
    assert rot.value == 0.0
    assert rot.source == "none"


def test_invalid_source_rejected():
    with pytest.raises(ValueError):
        AppliedRotation(0.0, "bogus")  # type: ignore[arg-type]


def test_make_applied_rotation_helper():
    rot = make_applied_rotation(999.0, "manual")
    assert rot.value == ANGLE_MAX
    assert rot.source == "manual"
    none_rot = make_applied_rotation(12.0, "none")
    assert none_rot.value == 0.0


def test_lane_line_x_at():
    lane = LaneLine(slope=0.1, angle=-5.0, ybar=100.0, x_offset=50.0,
                    points=np.zeros((0, 2)))
    assert lane.x_at(100.0) == pytest.approx(50.0)
    assert lane.x_at(200.0) == pytest.approx(60.0)


def test_estimate_outcome_accepted_requires_finite():
    ok = EstimateOutcome(True, 2.0, 0.1, "ok", (), 5, 1, 1, False, np.zeros((0, 2)), (), {})
    assert ok.accepted is True
    refused = EstimateOutcome(True, float("nan"), 0.1, "ok", (), 5, 1, 1, False,
                              np.zeros((0, 2)), (), {})
    assert refused.accepted is False


def test_session_default():
    s = SessionState.default()
    assert s.schema_version >= 1
    assert s.folder == "" and s.index == 0 and s.rotations == {}