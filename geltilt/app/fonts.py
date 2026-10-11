"""UI font selection for the desktop app.

Tk's default UI font on Windows is ``Segoe UI``, which has **no CJK glyphs** — so
Simplified Chinese text falls back to whatever font Windows picks, producing
mismatched shapes and uneven metrics.  We instead reconfigure Tk's named fonts to
a single modern, CJK-capable family (Microsoft YaHei UI on Windows), which also
renders Latin text well.

The family is chosen from a preference list of whatever is actually installed, so
the app degrades gracefully on machines/OSes without those fonts.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont
from typing import Optional, Tuple

# First family that is installed wins.  Windows-first, then cross-platform.
PREFERRED_FAMILIES: Tuple[str, ...] = (
    "Microsoft YaHei UI",
    "Microsoft YaHei",
    "Noto Sans SC",
    "Segoe UI",
    "Noto Sans",
    "DejaVu Sans",
    "Helvetica Neue",
    "Helvetica",
    "Arial",
)

# Named Tk fonts that drive widget text.  ``TkFixedFont`` is deliberately left
# alone so monospaced uses stay monospaced.
_THEMED_FONTS = (
    "TkDefaultFont",
    "TkTextFont",
    "TkMenuFont",
    "TkHeadingFont",
    "TkCaptionFont",
    "TkSmallCaptionFont",
    "TkIconFont",
    "TkTooltipFont",
)

_selected_family: Optional[str] = None


def choose_ui_family(root: tk.Misc) -> Optional[str]:
    """Return the first preferred family installed, or ``None`` if none are."""
    try:
        available = set(tkfont.families(root))
    except tk.TclError:  # pragma: no cover - no interpreter
        return None
    for family in PREFERRED_FAMILIES:
        if family in available:
            return family
    return None


def selected_family() -> Optional[str]:
    """The family applied by the last :func:`apply_ui_fonts` call."""
    return _selected_family


def apply_ui_fonts(root: tk.Misc) -> Optional[str]:
    """Point Tk's named UI fonts at the best available family.

    Only the *family* is changed; each font keeps its own size/weight/slant.  The
    chosen family is returned (and remembered), or ``None`` when nothing in the
    preference list is installed (in which case Tk's defaults are untouched).
    """
    global _selected_family
    family = choose_ui_family(root)
    if family is None:
        return None
    for name in _THEMED_FONTS:
        try:
            tkfont.nametofont(name, root=root).configure(family=family)
        except tk.TclError:  # pragma: no cover - font not defined in this build
            continue
    _selected_family = family
    return family

def ui_font(root: Optional[tk.Misc] = None, size: Optional[int] = None,
            weight: Optional[str] = None, slant: Optional[str] = None) -> tkfont.Font:
    """A font derived from the themed default, with optional overrides.

    Use this instead of ``("TkDefaultFont", N)``: a tuple treats ``TkDefaultFont``
    as a *family* name (which does not exist, so Tk silently falls back to Arial),
    whereas this copies the actual named font and so tracks :func:`apply_ui_fonts`.
    """
    font = tkfont.nametofont("TkDefaultFont", root=root).copy()
    overrides = {}
    if size is not None:
        overrides["size"] = int(size)
    if weight is not None:
        overrides["weight"] = weight
    if slant is not None:
        overrides["slant"] = slant
    if overrides:
        font.configure(**overrides)
    return font