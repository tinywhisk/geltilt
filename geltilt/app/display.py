"""Canonical display preprocessing with a per-sample cache."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..imageio import load_array, to_gray
from ..preprocess import preview_u8


def _to_display_u8(arr: np.ndarray) -> np.ndarray:
    """Apply the viewer display transform to a native-dtype source array."""
    return preview_u8(to_gray(arr))


class DisplayPreprocessor:
    """Applies the viewer ``preview_u8`` display transform, cached.

    The transform divides by the 99.7th percentile of the full frame, clips and
    applies ``gamma=0.45`` — matching the reference ``preview`` snippet. It is
    display-only: the rotation estimator still consumes the raw source array via
    :meth:`source_array`. Only the active sample is held in the cache.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, np.ndarray] = {}
        self._arrays: Dict[str, np.ndarray] = {}

    def display_u8(self, path: Path) -> np.ndarray:
        """Return the cached uint8 display image for ``path`` (loading if needed)."""
        key = Path(path).name
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        arr = load_array(path)
        u8 = _to_display_u8(arr)
        self._arrays[key] = arr
        self._cache[key] = u8
        return u8

    def source_array(self, path: Path) -> np.ndarray:
        """Return the cached native-dtype source array for ``path``."""
        key = Path(path).name
        arr = self._arrays.get(key)
        if arr is None:
            arr = load_array(path)
            self._arrays[key] = arr
            if key not in self._cache:
                self._cache[key] = _to_display_u8(arr)
        return arr

    def get_cached_u8(self, name: str) -> Optional[np.ndarray]:
        return self._cache.get(name)

    def clear(self) -> None:
        self._cache.clear()
        self._arrays.clear()