"""About / Licenses view: shows the app license and the third-party notices."""

from __future__ import annotations

import sys
from pathlib import Path

import tkinter as tk
from tkinter import ttk


def bundle_root() -> Path:
    """Directory holding LICENSE / THIRD_PARTY_NOTICES.md (frozen bundle or repo root)."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[2]


def licenses_text() -> str:
    """The app LICENSE followed by the third-party notices (missing files are skipped)."""
    chunks = []
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        path = bundle_root() / name
        if path.exists():
            chunks.append(path.read_text(encoding="utf-8", errors="replace").strip())
    separator = "\n\n" + "-" * 72 + "\n\n"
    return separator.join(chunks) if chunks else "(license text not found)"


def show_about(parent: tk.Misc, ui_scale: float = 1.0) -> tk.Toplevel:
    """Open a scrollable window with the license + third-party notices."""
    win = tk.Toplevel(parent)
    win.title("About geltilt")
    win.geometry(f"{int(760 * ui_scale)}x{int(560 * ui_scale)}")
    win.transient(parent)

    frame = ttk.Frame(win, padding=8)
    frame.pack(fill="both", expand=True)
    text = tk.Text(frame, wrap="word", font=("TkDefaultFont", max(8, int(9 * ui_scale))))
    scroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
    text.configure(yscrollcommand=scroll.set)
    scroll.pack(side="right", fill="y")
    text.pack(side="left", fill="both", expand=True)
    text.insert("1.0", "geltilt \u2014 gel electrophoresis image editor\n\n" + licenses_text())
    text.configure(state="disabled")

    ttk.Button(win, text="Close", command=win.destroy).pack(pady=6)
    return win