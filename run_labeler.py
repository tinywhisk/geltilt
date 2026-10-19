"""PyInstaller entry point for the geltilt rotation labeller GUI.

Build with ``python build_exe.py`` (produces ``dist/geltilt-label.exe``).
"""

from __future__ import annotations


def _run() -> int:
    try:
        from geltilt.labeler import main
        return main()
    except Exception:  # noqa: BLE001 - windowed builds hide the console
        import traceback
        try:
            import tkinter.messagebox as messagebox
            messagebox.showerror("geltilt-label", traceback.format_exc())
        except Exception:
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(_run())