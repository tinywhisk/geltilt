"""Immutable domain contracts for the geltilt desktop app.

These dataclasses are the typed vocabulary shared by the controller, the
estimation service and the widgets. Nothing here touches Tk or the canonical
pipeline; :class:`geltilt.app.estimator.CanonicalEstimator` is the only place an
untyped canonical result dict is read and projected into :class:`EstimateOutcome`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple

import numpy as np

RotationSource = Literal["none", "auto", "manual"]

SOURCES: Tuple[str, ...] = ("none", "auto", "manual")

ANGLE_MIN = -180.0
ANGLE_MAX = 180.0
ANGLE_RESOLUTION = 0.1

SCHEMA_VERSION = 1


def clamp_angle(value: float) -> float:
    """Clamp ``value`` to ``[-180.0, +180.0]`` and round to ``0.1`` degrees."""
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"angle must be finite, got {value!r}")
    v = max(ANGLE_MIN, min(ANGLE_MAX, v))
    return round(v, 1)


def is_valid_source(source: object) -> bool:
    return isinstance(source, str) and source in SOURCES


@dataclass(frozen=True)
class AppliedRotation:
    """What the single viewer currently shows, plus its provenance."""

    value: float = 0.0
    source: RotationSource = "none"

    def __post_init__(self) -> None:
        if not is_valid_source(self.source):
            raise ValueError(f"invalid rotation source: {self.source!r}")
        clamped = clamp_angle(self.value)
        if self.source == "none":
            clamped = 0.0
        object.__setattr__(self, "value", clamped)

    def is_valid_for_auto(self) -> bool:
        return self.source in ("none", "auto", "manual")

    def to_dict(self) -> Dict[str, object]:
        return {"value": float(self.value), "source": self.source}


def make_applied_rotation(value: float, source: str) -> AppliedRotation:
    """Validation/normalisation helper enforcing the domain rules.

    Clamps to ``[-180.0, +180.0]``, rounds to ``0.1``, validates the source and
    forces ``source == "none"`` to imply ``value == 0.0``.
    """
    if not is_valid_source(source):
        raise ValueError(f"invalid rotation source: {source!r}")
    v = clamp_angle(value)
    if source == "none":
        v = 0.0
    return AppliedRotation(v, source)


@dataclass(frozen=True)
class LaneLine:
    """One accepted lane, projected from the canonical ``kept_lanes`` subset."""

    slope: float
    angle: float
    ybar: float
    x_offset: float
    points: np.ndarray

    def x_at(self, y: float) -> float:
        return self.x_offset + self.slope * (y - self.ybar)


@dataclass(frozen=True)
class EstimateOutcome:
    """Typed projection of the canonical estimator result."""

    ok: bool
    angle: float
    se: float
    reason: str
    warnings: Tuple[str, ...]
    n_bands: int
    lanes_used: int
    lanes_found: int
    fallback: bool
    centroids: np.ndarray
    accepted_lanes: Tuple[LaneLine, ...]
    raw: Dict[str, object] = field(default_factory=dict, compare=False, repr=False)

    @property
    def accepted(self) -> bool:
        return bool(self.ok) and math.isfinite(float(self.angle))


@dataclass(frozen=True)
class CropRect:
    """A crop rectangle in the corrected/display frame (anchor-relative pixels).

    ``angle`` records the rotation the frame was locked to (for a future
    crop-aware export); it is ignored when drawing, so the box stays upright and
    fixed when the applied rotation changes.
    """

    x0: float
    y0: float
    x1: float
    y1: float
    angle: float = 0.0

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    def to_dict(self) -> Dict[str, float]:
        return {"x0": float(self.x0), "y0": float(self.y0),
                "x1": float(self.x1), "y1": float(self.y1), "angle": float(self.angle)}


@dataclass
class SampleState:
    """Per-sample in-memory state retained for the session."""

    name: str
    applied: AppliedRotation = field(default_factory=AppliedRotation)
    last_estimate: Optional[EstimateOutcome] = None
    failed: bool = False
    crop: Optional[CropRect] = None
    band_mask: object = None  # cached source-frame binary band mask (T-7), in-memory


@dataclass(frozen=True)
class SessionState:
    """Persisted session payload (exactly one file per run directory)."""

    schema_version: int = SCHEMA_VERSION
    folder: str = ""
    index: int = 0
    rotations: Dict[str, AppliedRotation] = field(default_factory=dict)
    updated_at: str = ""

    @classmethod
    def default(cls) -> "SessionState":
        return cls()


@dataclass(frozen=True)
class DiagnosticLogEntry:
    """One structured auto-rotation run record (metadata only)."""

    timestamp: str
    sample: str
    angle: float
    se: float
    ok: bool
    reason: str
    warnings: Tuple[str, ...] = ()