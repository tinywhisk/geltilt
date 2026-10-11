from __future__ import annotations

from tkinter import font as tkfont

from geltilt.app import fonts


def test_apply_ui_fonts_uses_an_installed_preferred_family(tk_root):
    family = fonts.apply_ui_fonts(tk_root)
    assert family in fonts.PREFERRED_FAMILIES
    assert family in set(tkfont.families(tk_root))
    assert tkfont.nametofont("TkDefaultFont").actual("family") == family
    assert fonts.selected_family() == family


def test_apply_ui_fonts_leaves_fixed_font_monospaced(tk_root):
    fonts.apply_ui_fonts(tk_root)
    assert "TkFixedFont" not in fonts._THEMED_FONTS


def test_ui_font_follows_the_themed_family(tk_root):
    family = fonts.apply_ui_fonts(tk_root)
    chosen = fonts.ui_font(tk_root, size=14, weight="bold")
    assert chosen.actual("family") == family
    assert chosen.actual("size") == 14
    assert chosen.actual("weight") == "bold"


def test_choose_ui_family_returns_none_when_nothing_matches(tk_root, monkeypatch):
    monkeypatch.setattr(fonts, "PREFERRED_FAMILIES", ("Definitely Not A Font",))
    assert fonts.choose_ui_family(tk_root) is None