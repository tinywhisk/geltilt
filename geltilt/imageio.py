"""Image loading and saving that preserves bit depth (8-bit, 16-bit, float)."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

PathLike = Union[str, Path]


def to_gray(arr: np.ndarray) -> np.ndarray:
    """Convert an array (2D gray or HxWxC color) to a float32 2D grayscale array."""
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            return arr[:, :, 0].astype(np.float32)
        rgb = arr[:, :, :3].astype(np.float32)
        return (
            0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        ).astype(np.float32)
    raise ValueError(f"unsupported array shape: {arr.shape}")


def load_array(path: PathLike) -> np.ndarray:
    """Load an image from disk, preserving its native dtype and channel count."""
    with Image.open(path) as im:
        im.load()
        mode = im.mode
        arr = np.asarray(im)
    if mode in ("I;16", "I;16B", "I;16L") and arr.dtype != np.uint16:
        arr = np.clip(arr, 0, 65535).astype(np.uint16)
    if mode == "I" and arr.dtype not in (np.uint16, np.int32):
        arr = arr.astype(np.int32)
    return arr


def load_gray(path: PathLike) -> np.ndarray:
    """Load an image and return a float32 2D grayscale array."""
    return to_gray(load_array(path))


def save_array(path: PathLike, arr: np.ndarray) -> None:
    """Save an array to disk, choosing a PIL mode that preserves its dtype."""
    arr = np.asarray(arr)
    if arr.ndim == 2:
        if arr.dtype == np.uint8 or arr.dtype == np.uint16:
            img = Image.fromarray(arr)
        elif np.issubdtype(arr.dtype, np.floating):
            img = Image.fromarray(arr.astype(np.float32))
        else:
            img = Image.fromarray(arr.astype(np.int32))
    elif arr.ndim == 3 and arr.shape[2] in (3, 4):
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    else:
        raise ValueError(f"unsupported array shape for saving: {arr.shape}")
    img.save(path)