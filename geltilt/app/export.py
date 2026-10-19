"""Export the active sample: rotation-correct it, then save or copy it.

The exported image is the canonically corrected frame
(:func:`geltilt.core.correct_rotation`, center pivot, same size, dtype-preserved).
Cropping is not applied yet (a future addition). Whether the pixels come from the
**original** source array or the brightness-processed **display** image is chosen
by the caller (the "Use original image" checkbox).
"""

from __future__ import annotations

import io
import struct
import sys
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

from ..core import correct_rotation
from ..imageio import save_array

# File-dialog format list (name, glob). Order = preference.
SAVE_FILETYPES = [
    ("TIFF image", "*.tif *.tiff"),
    ("PNG image", "*.png"),
    ("JPEG image", "*.jpg *.jpeg"),
    ("BMP image", "*.bmp"),
    ("All files", "*.*"),
]

# Open-dialog format list; the first entry mirrors the supported input types.
OPEN_FILETYPES = [
    ("Image files", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"),
    ("TIFF image", "*.tif *.tiff"),
    ("PNG image", "*.png"),
    ("JPEG image", "*.jpg *.jpeg"),
    ("BMP image", "*.bmp"),
    ("All files", "*.*"),
]

# extension -> (preserves the source bit depth?)
_PRESERVES_DEPTH: Dict[str, bool] = {
    ".tif": True, ".tiff": True, ".png": True,
    ".jpg": False, ".jpeg": False, ".bmp": False,
}

_CF_TIFF = 6   # Windows clipboard: TIFF
_CF_DIB = 8    # Windows clipboard: device-independent bitmap


def rotated_export(source: np.ndarray, angle: float) -> np.ndarray:
    """Return the rotation-corrected source (same dtype and size as the input)."""
    return correct_rotation(np.asarray(source), float(angle))


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """Reduce an array to 8-bit for formats that cannot hold the native depth.

    A plain linear bit-depth reduction — no contrast/brightness processing.
    """
    a = np.asarray(arr)
    if a.dtype == np.uint8:
        return a
    if a.dtype == np.uint16:
        return np.clip(a.astype(np.float32) / 257.0, 0, 255).round().astype(np.uint8)
    if np.issubdtype(a.dtype, np.floating):
        peak = float(np.nanmax(a)) if a.size else 0.0
        scale = 255.0 if peak <= 1.0 else 1.0
        return np.clip(a.astype(np.float32) * scale, 0, 255).round().astype(np.uint8)
    return np.clip(a, 0, 255).astype(np.uint8)


def save_image(path, arr: np.ndarray) -> None:
    """Save ``arr`` to ``path``, choosing the format from the file extension."""
    p = Path(path)
    preserve = _PRESERVES_DEPTH.get(p.suffix.lower(), True)
    save_array(p, np.asarray(arr) if preserve else to_uint8(arr))


def tiff_bytes(arr: np.ndarray) -> bytes:
    """Encode ``arr`` as a TIFF (bit depth preserved)."""
    buf = io.BytesIO()
    Image.fromarray(np.asarray(arr)).save(buf, format="TIFF")
    return buf.getvalue()


def dib_bytes(arr: np.ndarray) -> bytes:
    """Encode ``arr`` as a Windows ``CF_DIB`` blob (24-bit BGR, bottom-up)."""
    a = to_uint8(arr)
    if a.ndim == 2:
        rgb = np.repeat(a[:, :, None], 3, axis=2)
    else:
        rgb = a[:, :, :3]
    h, w = rgb.shape[:2]
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])[::-1]  # BGR, bottom-up
    row_bytes = w * 3
    pad = (-row_bytes) % 4  # DIB rows are padded to a 4-byte boundary
    rows = np.zeros((h, row_bytes + pad), np.uint8)
    rows[:, :row_bytes] = bgr.reshape(h, row_bytes)
    pixels = rows.tobytes()
    header = struct.pack("<IiiHHIIiiII", 40, w, h, 1, 24, 0, len(pixels), 2835, 2835, 0, 0)
    return header + pixels


def copy_tiff_to_clipboard(arr: np.ndarray) -> None:
    """Put the image on the Windows clipboard as a TIFF (plus a DIB fallback)."""
    if sys.platform != "win32":
        raise RuntimeError("Copying an image to the clipboard is Windows-only.")
    _win_set_clipboard({_CF_TIFF: tiff_bytes(arr), _CF_DIB: dib_bytes(arr)})


def _win_set_clipboard(formats: Dict[int, bytes]) -> None:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalLock.restype = wintypes.LPVOID
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    user32.SetClipboardData.restype = wintypes.HANDLE
    user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]

    gmem_moveable = 0x0002
    if not user32.OpenClipboard(None):
        raise RuntimeError("could not open the clipboard")
    try:
        user32.EmptyClipboard()
        for fmt, data in formats.items():
            handle = kernel32.GlobalAlloc(gmem_moveable, len(data))
            if not handle:
                raise RuntimeError("GlobalAlloc failed")
            ptr = kernel32.GlobalLock(handle)
            if not ptr:
                kernel32.GlobalFree(handle)
                raise RuntimeError("GlobalLock failed")
            ctypes.memmove(ptr, data, len(data))
            kernel32.GlobalUnlock(handle)
            if not user32.SetClipboardData(fmt, handle):
                kernel32.GlobalFree(handle)
                raise RuntimeError(f"SetClipboardData failed for format {fmt}")
    finally:
        user32.CloseClipboard()