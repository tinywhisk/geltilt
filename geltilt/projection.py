"""Vertical projection of band signal and sharpness scoring."""

from __future__ import annotations

import numpy as np

MAXIMIZE = {"tv2": True, "d2sq": True, "var": True, "entropy": False}


def vertical_projection(signal: np.ndarray, log: bool = True) -> np.ndarray:
    """Column-sum profile of the band signal (sum over rows), normalised to sum 1.

    When lanes run vertically this profile shows sharp peaks at the lane
    positions and deep valleys between lanes.
    """
    x = np.log1p(signal) if log else signal.astype(np.float64)
    profile = x.sum(axis=0).astype(np.float64)
    total = profile.sum()
    if total <= 0:
        return profile
    return profile / total


def sharpness(profile: np.ndarray, method: str = "tv2") -> float:
    """Score how sharply a projection profile is peaked (higher is sharper)."""
    if method == "tv2":
        return float((np.abs(np.diff(profile)) ** 2).sum())
    if method == "d2sq":
        return float((np.diff(profile, 2) ** 2).sum())
    if method == "var":
        return float(profile.var())
    if method == "entropy":
        p = profile[profile > 0]
        return -float((p * np.log(p)).sum())
    if method == "max":
        return float(profile.max())
    raise ValueError(f"unknown sharpness method: {method!r}")


def orientation_score(signal: np.ndarray, method: str = "tv2", log: bool = True) -> float:
    """Projection sharpness of an image; maximised (or minimised) at vertical lanes."""
    return sharpness(vertical_projection(signal, log=log), method=method)