"""PyInstaller entry point for the geltilt projection viewer GUI.

Build with ``python build_projview_exe.py`` (produces ``dist/geltilt-projview.exe``).
"""

from __future__ import annotations


def _run() -> int:
    try:
        from geltilt.projview import main
        return main()
    except Exception:  # noqa: BLE001 - windowed builds hide the console
        import traceback
        try:
            import tkinter.messagebox as messagebox
            messagebox.showerror("geltilt-projview", traceback.format_exc())
        except Exception:
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(_run())