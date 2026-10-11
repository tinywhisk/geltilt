"""Unified single-window Tkinter frontend for geltilt.

The app is a thin presentation layer over the canonical in-repo pipeline
(:func:`geltilt.banddetect.band_centroids` -> :func:`geltilt.ladders.estimate`).
It introduces no parallel rotation engine: every estimate goes through
:class:`geltilt.app.estimator.CanonicalEstimator`.
"""

from __future__ import annotations

__all__ = [
    "domain",
    "ports",
    "estimator",
    "config",
    "session",
    "log",
    "state",
    "display",
    "service",
    "controller",
]