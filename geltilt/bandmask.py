"""GelGenie ONNX band segmentation (run with OpenCV's DNN module).

Implements the band-detection front end specified in ``BAND_DETECTION.md``: tiled inference at the
image's true scale (the model window is a fixed 1024x1024) with a tent-weight feather merge over the
overlaps, softmax over the class axis, and the class-1 ("Gel Band") map. The centroids derived from
this mask are the input to ``geltilt.ladders``. No slab mask is used here.

No new runtime dependency: inference uses ``cv2.dnn`` with the fixed-size (1024) ONNX export. The
model is fetched once from HuggingFace and cached.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .imageio import to_gray

MODEL_URL = (
    "https://huggingface.co/mattaq/GelGenie-Universal-Dec-2023/resolve/main/"
    "onnx_checkpoints/unet_dec_21_epoch_579_fixed_dim_1024.onnx"
)
MODEL_FILENAME = "gelgenie_universal_1024.onnx"
OVERLAP = 256
FEATHER = 128


def _origins(length: int, win: int, overlap: int) -> List[int]:
    """Tile origins along one axis: native scale, zero-padded if the image is smaller than the window."""
    if length <= win:
        return [0]
    step = win - overlap
    return sorted(set([0] + list(range(0, length - win + 1, step)) + [length - win]))


def default_model_path() -> Path:
    """Where the ONNX model is looked up (override with GELTILT_GELGENIE_ONNX)."""
    env = os.environ.get("GELTILT_GELGENIE_ONNX")
    if env:
        return Path(env)
    cache = Path(os.environ.get("GELTILT_CACHE", str(Path.home() / ".geltilt")))
    return cache / MODEL_FILENAME


def ensure_model(model_path: Optional[Path] = None, download: bool = True) -> Path:
    """Return a path to the ONNX model, downloading it once if necessary."""
    path = Path(model_path) if model_path else default_model_path()
    if path.exists():
        return path
    if not download:
        raise FileNotFoundError(
            f"GelGenie ONNX model not found at {path}. Download it or set "
            f"GELTILT_GELGENIE_ONNX to its location."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, path)
    return path


def _model_input(gray: np.ndarray) -> np.ndarray:
    arr = to_gray(gray)
    peak = float(arr.max()) if arr.size else 0.0
    if peak <= 255.0:
        scale = 255.0
    elif peak <= 65535.0:
        scale = 65535.0
    else:
        scale = peak
    return (arr / max(scale, 1.0)).astype(np.float32)


class BandSegmenter:
    """Wraps the GelGenie ONNX model to produce a band-probability map."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        input_size: int = 1024,
        download: bool = True,
    ) -> None:
        self.model_path = ensure_model(model_path, download)
        self.input_size = int(input_size)
        self._net = cv2.dnn.readNetFromONNX(str(self.model_path))

    def probability(self, gray: np.ndarray) -> np.ndarray:
        """Return a float32 band-probability map (0..1) the same size as ``gray``.

        Tiled at the image's true scale with ``OVERLAP`` and a tent-weight feather merge, so a
        non-square image (e.g. 1024x1280) is covered without resizing and the tile seams cross-fade.
        """
        image = _model_input(gray)
        h, w = image.shape
        s = self.input_size
        prob = np.zeros((h, w), np.float32)
        wsum = np.zeros((h, w), np.float32)
        yy, xx = np.mgrid[0:s, 0:s]
        tent = np.clip(np.minimum(np.minimum(xx + 1, s - xx), np.minimum(yy + 1, s - yy)),
                       0, FEATHER).astype(np.float32)
        for y0 in _origins(h, s, OVERLAP):
            for x0 in _origins(w, s, OVERLAP):
                y1, x1 = min(y0 + s, h), min(x0 + s, w)
                tile = np.zeros((s, s), np.float32)
                tile[:y1 - y0, :x1 - x0] = image[y0:y1, x0:x1]
                self._net.setInput(tile[None, None])
                out = np.asarray(self._net.forward())[0]           # (C, s, s) logits
                if out.ndim == 3 and out.shape[0] >= 2:
                    e = np.exp(out - out.max(axis=0, keepdims=True))
                    band = (e / e.sum(axis=0, keepdims=True))[1]
                else:
                    band = out.reshape(s, s)
                tw = tent[:y1 - y0, :x1 - x0]
                prob[y0:y1, x0:x1] += band[:y1 - y0, :x1 - x0] * tw
                wsum[y0:y1, x0:x1] += tw
        return prob / np.maximum(wsum, 1e-6)


_CACHE: dict = {}


def band_probability(
    gray: np.ndarray,
    segmenter: Optional[BandSegmenter] = None,
    model_path: Optional[Path] = None,
) -> np.ndarray:
    """Convenience wrapper that caches a default :class:`BandSegmenter`."""
    if segmenter is None:
        key = str(Path(model_path) if model_path else default_model_path())
        segmenter = _CACHE.get(key)
        if segmenter is None:
            segmenter = BandSegmenter(model_path)
            _CACHE[key] = segmenter
    return segmenter.probability(gray)