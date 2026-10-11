"""Ports (protocols) and domain errors for the geltilt desktop app.

The controller depends only on these protocols, so the canonical adapter can be
swapped for a test double without inheritance.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .domain import DiagnosticLogEntry, EstimateOutcome, SessionState


class ModelUnavailableError(RuntimeError):
    """The GelGenie ONNX band segmenter could not be located or loaded."""


class EstimateError(RuntimeError):
    """Segmentation or estimation failed after the model was available."""


class EstimatorPort(Protocol):
    """The single rotation seam: image -> typed estimate outcome."""

    def estimate(self, image: np.ndarray) -> EstimateOutcome: ...


class SessionStore(Protocol):
    """Load/save the persisted session payload."""

    def load(self) -> SessionState: ...

    def save(self, state: SessionState) -> None: ...


class DiagnosticLog(Protocol):
    """Append one structured run-log record."""

    def append(self, entry: DiagnosticLogEntry) -> None: ...