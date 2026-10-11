"""PyInstaller entry point for the unified geltilt desktop app.

Build with ``python build_app_exe.py`` (produces ``dist/geltilt-app.exe``).

If a GelGenie ONNX model was bundled at build time it is located next to the
executable (or in the PyInstaller extraction dir) and exposed through
``GELTILT_GELGENIE_ONNX`` so the packaged app runs fully offline.
"""

from __future__ import annotations

import os
from pathlib import Path

MODEL_FILENAME = "gelgenie_universal_1024.onnx"


def _configure_bundled_model() -> None:
    """Point the canonical model lookup at a bundled model, if one exists."""
    if os.environ.get("GELTILT_GELGENIE_ONNX"):
        return
    import sys

    candidates = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / MODEL_FILENAME)
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / MODEL_FILENAME)
    for candidate in candidates:
        if candidate.exists():
            os.environ["GELTILT_GELGENIE_ONNX"] = str(candidate)
            return


def _run() -> int:
    try:
        _configure_bundled_model()
        from geltilt.app.window import main
        return main()
    except Exception:  # noqa: BLE001 - windowed builds hide the console
        import traceback

        try:
            import tkinter.messagebox as messagebox
            messagebox.showerror("geltilt-app", traceback.format_exc())
        except Exception:
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(_run())