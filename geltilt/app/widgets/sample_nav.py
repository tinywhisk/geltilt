"""Top-bar sample navigation: prev/next arrows plus the current filename.

Sits in the file/folder bar (see ``ui_layout_reference``): the two arrow buttons
step through the folder and the expanding label shows the active filename.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import tkinter as tk
from tkinter import ttk

from .. import i18n

_PREV = "\u25c0"
_NEXT = "\u25b6"


class SampleNav(ttk.Frame):
    """Shows the active filename with arrows that step through the folder."""

    def __init__(self, master: tk.Misc, ui_scale: float = 1.0) -> None:
        super().__init__(master)
        self.ui_scale = ui_scale
        self._names: List[str] = []
        self._active: Optional[int] = None
        self._on_select: Optional[Callable[[int], None]] = None

        self.prev_button = ttk.Button(self, text=_PREV, width=3, command=self._prev)
        self.prev_button.pack(side="left")
        self.next_button = ttk.Button(self, text=_NEXT, width=3, command=self._next)
        self.next_button.pack(side="left", padx=(4, 0))
        self.name_label = ttk.Label(self, text=i18n.tr("nav.no_image"), anchor="w")
        self.name_label.pack(side="left", fill="x", expand=True, padx=(12, 0))
        self._update_buttons()

    # -- population --------------------------------------------------------- #
    def bind_select(self, callback: Callable[[int], None]) -> None:
        self._on_select = callback

    def set_samples(self, names: List[str]) -> None:
        self._names = list(names)
        self._active = None
        if self._names:
            self.name_label.config(text=self._names[0])
        else:
            self.name_label.config(text=i18n.tr("nav.no_image"))
        self._update_buttons()

    def set_active(self, index: int) -> None:
        if not self._names:
            self.name_label.config(text=i18n.tr("nav.no_image"))
            self._update_buttons()
            return
        index = max(0, min(int(index), len(self._names) - 1))
        self._active = index
        self.name_label.config(text=self._names[index])
        self._update_buttons()

    def active_index(self) -> Optional[int]:
        return self._active

    def count(self) -> int:
        return len(self._names)

    # -- navigation --------------------------------------------------------- #
    def _prev(self) -> None:
        if self._active is not None:
            self._go(self._active - 1)

    def _next(self) -> None:
        if self._active is not None:
            self._go(self._active + 1)

    def _go(self, index: int) -> None:
        if not self._names or self._on_select is None:
            return
        index = max(0, min(int(index), len(self._names) - 1))
        if index == self._active:
            return
        self._on_select(index)

    def _update_buttons(self) -> None:
        n = len(self._names)
        can_prev = self._active is not None and self._active > 0
        can_next = self._active is not None and self._active < n - 1
        self.prev_button.config(state="normal" if can_prev else "disabled")
        self.next_button.config(state="normal" if can_next else "disabled")