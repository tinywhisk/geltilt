"""The single canonical estimator adapter.

``CanonicalEstimator`` calls exactly the same two canonical steps as
:func:`geltilt.core.estimate_rotation_ladders` -- ``band_centroids`` then
``ladders.estimate`` -- and projects the untyped result dict into an
:class:`~geltilt.app.domain.EstimateOutcome`. This is the only seam to the
rotation engine; no alternative engine may be registered.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .. import ladders
from ..banddetect import band_centroids, default_segmenter
from .domain import EstimateOutcome, LaneLine
from .ports import EstimateError, ModelUnavailableError


def _lane_from_dict(lane: dict, xy: np.ndarray) -> LaneLine:
    """Project one canonical kept-lane dict into a :class:`LaneLine`."""
    slope = float(lane.get("s", 0.0))
    ybar = float(lane.get("ybar", 0.0))
    idx = lane.get("idx")
    pts = np.asarray(xy, float)[idx] if idx is not None and len(xy) else np.zeros((0, 2), float)
    if pts.size:
        x_offset = float(np.mean(pts[:, 0] - slope * (pts[:, 1] - ybar)))
    else:
        x_offset = float(lane.get("a", 0.0))
    return LaneLine(
        slope=slope,
        angle=float(lane.get("angle", -np.degrees(np.arctan(slope)))),
        ybar=ybar,
        x_offset=x_offset,
        points=pts,
    )


class CanonicalEstimator:
    """Adapter around ``band_centroids`` -> ``ladders.estimate``."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        threshold: float = 0.5,
        min_area: int = 30,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else None
        self.threshold = float(threshold)
        self.min_area = int(min_area)
        self._segmenter = None

    def _segmenter_or_raise(self):
        if self._segmenter is None:
            try:
                self._segmenter = default_segmenter(self.model_path)
            except Exception as exc:  # noqa: BLE001 - surfaced as a typed error
                raise ModelUnavailableError(
                    f"could not load the GelGenie band segmenter: {exc}"
                ) from exc
        return self._segmenter

    def estimate(self, image: np.ndarray) -> EstimateOutcome:
        segmenter = self._segmenter_or_raise()
        try:
            xy = band_centroids(
                image,
                segmenter=segmenter,
                threshold=self.threshold,
                min_area=self.min_area,
            )
        except ModelUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001 - inference failure
            raise EstimateError(f"band detection failed: {exc}") from exc
        try:
            raw = ladders.estimate(xy)
        except Exception as exc:  # noqa: BLE001 - estimation failure
            raise EstimateError(f"rotation estimation failed: {exc}") from exc
        return self._project(raw, xy)

    @staticmethod
    def _project(raw: dict, xy: np.ndarray) -> EstimateOutcome:
        angle = raw.get("angle", float("nan"))
        try:
            angle = float(angle)
        except (TypeError, ValueError):
            angle = float("nan")
        se = raw.get("se", float("nan"))
        try:
            se = float(se)
        except (TypeError, ValueError):
            se = float("nan")
        warnings: Tuple[str, ...] = tuple(str(w) for w in raw.get("warnings", ()) or ())
        kept_lanes = raw.get("kept_lanes", ()) or ()
        lanes = tuple(_lane_from_dict(lane, xy) for lane in kept_lanes)
        return EstimateOutcome(
            ok=bool(raw.get("ok", False)),
            angle=angle,
            se=se,
            reason=str(raw.get("reason", "")),
            warnings=warnings,
            n_bands=int(raw.get("n_bands", len(xy))),
            lanes_used=int(raw.get("kept", len(kept_lanes))),
            lanes_found=int(raw.get("n_ladders", 0)),
            fallback=bool(raw.get("fallback", False)),
            centroids=np.asarray(xy, float),
            accepted_lanes=lanes,
            raw=raw,
        )