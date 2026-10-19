#!/usr/bin/env python3
"""GELTILT UI layout only -- window and widget positions, nothing else.

    python ui_layout.py            # opens the window
    python ui_layout.py --dump     # prints every widget's realized geometry, exits

Recreates the *placement* of GELTILT's window: the window size, the vertical order
of the control rows, every widget's width and packing side, and the two-pane grid.
It draws no image, computes no angle, loads no gel -- every widget is an inert
stand-in in the right place.

Every number and every pack order below is taken from ``desktop/app.py``
(``_build_widgets``).  The comments name the Tk option that produces each effect, so
the same skeleton can be translated to another toolkit.  For the *rendered* overlays
(crop box, lane boundaries, tint) see ``docs/DISPLAY.md``; this file is the widget
skeleton only.

Dependencies: the standard library only.
"""
from __future__ import annotations

import argparse
import sys
import tkinter as tk
from tkinter import ttk

# --------------------------------------------------------------------------- config
# desktop/app.py: window geometry

WINDOW = '1280x840'          # root.geometry
MINSIZE = (940, 620)         # root.minsize

PAGE_PAD = 8                 # the padx used by every group frame
GROUP_PAD = (8, 6)           # padx/pady inside every LabelFrame

# desktop/app.py: the angle slider.  ANGLE_MIN/MAX/STEP are the app's constants.
ANGLE_MIN, ANGLE_MAX, ANGLE_STEP, SLIDER_LEN = -20.0, 20.0, 0.01, 340


def label(parent, text, **kw):
    return tk.Label(parent, text=text, **kw)


# ------------------------------------------------------------------------------ UI

def build(root: tk.Tk) -> dict:
    """Build the layout.  Returns the widgets, in case a caller wants them."""
    root.title('Gel Tilt - layout only')
    root.geometry(WINDOW)                      # 1280x840
    root.minsize(*MINSIZE)                     # cannot shrink below 940x620

    w = {}

    # --- row 1: file / folder -------------------------------------------------
    # Packed TOP with fill X.  The filename label is packed LAST with
    # expand=True, so it absorbs the leftover width.
    top = tk.Frame(root, padx=PAGE_PAD, pady=6)
    top.pack(side='top', fill='x')

    w['load_image'] = label(top, 'Load image...', width=13)     # width is in CHARACTERS
    w['load_image'].pack(side='left')
    w['load_folder'] = label(top, 'Load folder...', width=13)
    w['load_folder'].pack(side='left', padx=(6, 0))
    w['prev'] = label(top, '\u25c0', width=3)                   # the glyph IS the label
    w['prev'].pack(side='left', padx=(12, 0))
    w['next'] = label(top, '\u25b6', width=3)
    w['next'].pack(side='left', padx=(4, 0))

    # The language picker is packed BEFORE the expanding filename label.  Tk hands
    # an expanding widget the whole remaining cavity, so anything packed after it
    # is squeezed out of sight -- this order is load-bearing, not style.
    w['lang_box'] = ttk.Combobox(top, values=['简体中文', 'English'],
                                 state='readonly', width=10)
    w['lang_box'].pack(side='right')
    label(top, 'Language:').pack(side='right', padx=(0, 4))

    w['file_label'] = label(top, 'gel1.tif', anchor='w')        # anchor w: left-justified
    w['file_label'].pack(side='left', padx=(12, 0), fill='x', expand=True)

    # --- row 2: Rotation ------------------------------------------------------
    # A LabelFrame, so the title doubles as the group label.
    rot = tk.LabelFrame(root, text=' Rotation ', padx=PAGE_PAD, pady=6)
    rot.pack(side='top', fill='x', padx=PAGE_PAD)

    label(rot, 'Auto rotate', width=13).pack(side='left')
    label(rot, 'Manual angle (°):').pack(side='left', padx=(12, 0))
    w['angle_entry'] = tk.Entry(rot, width=8, justify='right')  # chars, right-aligned
    w['angle_entry'].insert(0, '0.00')
    w['angle_entry'].pack(side='left', padx=(4, 8))
    label(rot, '\u25c0', width=3).pack(side='left')             # coarse-step arrows
    w['scale'] = tk.Scale(rot, from_=ANGLE_MIN, to=ANGLE_MAX,
                          resolution=ANGLE_STEP, orient='horizontal',
                          showvalue=False, length=SLIDER_LEN)   # length is in PIXELS
    w['scale'].pack(side='left')
    label(rot, '\u25b6', width=3).pack(side='left')
    label(rot, 'Reset', width=7).pack(side='left', padx=(8, 0))

    # --- rows 3 and 4: Crop ---------------------------------------------------
    crop = tk.LabelFrame(root, text=' Crop ', padx=PAGE_PAD, pady=6)
    crop.pack(side='top', fill='x', padx=PAGE_PAD, pady=(6, 0))

    crop_row = tk.Frame(crop)                                   # crop row 1 of 2
    crop_row.pack(side='top', fill='x')
    label(crop_row, 'Auto crop', width=13).pack(side='left')
    label(crop_row, 'Clear crop', width=11).pack(side='left', padx=(6, 0))
    label(crop_row, 'no crop', anchor='w').pack(side='left', padx=(12, 0))
    label(crop_row, 'Padding:').pack(side='left', padx=(12, 0))
    w['pad'] = tk.Spinbox(crop_row, from_=0, to=200, increment=5,
                          width=4, justify='right')
    w['pad'].pack(side='left', padx=(4, 0))
    label(crop_row, 'px  left/right/bottom').pack(side='left', padx=(4, 0))

    lane_row = tk.Frame(crop)                                   # crop row 2 of 2
    lane_row.pack(side='top', fill='x', pady=(6, 0))
    w['composite'] = tk.Checkbutton(lane_row, text='Composite crop')
    w['composite'].pack(side='left')
    w['lane_spec'] = tk.Entry(lane_row, width=22)
    w['lane_spec'].pack(side='left', padx=(6, 0))

    # Button right, then the hint with expand=True: the hint fills everything
    # between the entry and the button, and the button stays flush right.
    w['even'] = label(lane_row, 'Even widths', width=11)
    w['even'].pack(side='right')
    w['hint'] = label(lane_row, 'click lanes to join, or type e.g. 3,1-2,4-6',
                      anchor='w')
    w['hint'].pack(side='left', padx=(10, 8), fill='x', expand=True)

    # --- status strip ---------------------------------------------------------
    # Packed BOTTOM, and BEFORE the panes.  Tk gives a bottom-packed widget its
    # natural height only if it is claimed before the expanding widget, so this
    # order is what stops the strip being squeezed to nothing.
    status = tk.Frame(root)
    status.pack(side='bottom', fill='x', padx=PAGE_PAD, pady=(0, 8))
    label(status, 'Load an image, or a folder of gels, to begin.',
          anchor='w', justify='left', wraplength=1240).pack(side='top', fill='x')
    label(status, '8-bit   |   1280×1024   |   angle +0.00°',
          anchor='w', justify='left', wraplength=1240, fg='#444444').pack(side='top',
                                                                         fill='x')

    # --- row 5: the two panes -------------------------------------------------
    # TOP with fill BOTH and expand, so it absorbs the leftover height.
    panes = tk.Frame(root, padx=PAGE_PAD, pady=6)
    panes.pack(side='top', fill='both', expand=True)
    panes.columnconfigure(0, weight=1, uniform='pane')   # equal width, always
    panes.columnconfigure(1, weight=1, uniform='pane')
    panes.rowconfigure(1, weight=1)                      # row 1 expands; row 0 is labels

    label(panes, 'Original').grid(row=0, column=0, sticky='w')
    label(panes, 'Corrected').grid(row=0, column=1, sticky='w')

    # 4 px gap between the panes: padx (0,4) on the left, (4,0) on the right
    w['left_canvas'] = tk.Canvas(panes, bg='#1b1b1b', highlightthickness=1,
                                 highlightbackground='#666666')
    w['left_canvas'].grid(row=1, column=0, sticky='nsew', padx=(0, 4))
    w['right_canvas'] = tk.Canvas(panes, bg='#1b1b1b', highlightthickness=1,
                                  highlightbackground='#666666')
    w['right_canvas'].grid(row=1, column=1, sticky='nsew', padx=(4, 0))

    # --- row 6: Export --------------------------------------------------------
    exp = tk.LabelFrame(root, text=' Export ', padx=PAGE_PAD, pady=6)
    exp.pack(side='top', fill='x', padx=PAGE_PAD, pady=(0, 6))

    label(exp, 'Save / copy:').pack(side='left')
    w['mode'] = ttk.Combobox(exp, values=['Rotation only', 'Rotation + crop'],
                             state='readonly', width=17)
    w['mode'].set('Rotation + crop')                    # the default entry
    w['mode'].pack(side='left', padx=(6, 12))
    w['original'] = tk.Checkbutton(
        exp, text='Use original image (no contrast/brightness)')
    w['original'].select()                              # ticked by default
    w['original'].pack(side='left', padx=(0, 12))
    label(exp, 'Save as...', width=13).pack(side='left')
    label(exp, 'Copy to clipboard', width=19).pack(side='left', padx=(6, 0))
    w['depth'] = label(exp, '8-bit', anchor='w')
    w['depth'].pack(side='left', padx=(12, 0))

    return w


# ----------------------------------------------------------------------- geometry

def dump(root: tk.Tk) -> None:
    """Print the realized geometry of every widget, so the layout can be checked."""
    root.update_idletasks()
    root.update()

    def walk(widget, depth=0):
        try:
            geo = (f'x={widget.winfo_x():5d} y={widget.winfo_y():4d} '
                   f'w={widget.winfo_width():5d} h={widget.winfo_height():4d}')
        except tk.TclError:
            geo = 'n/a'
        text = ''
        try:
            if 'text' in widget.keys():
                text = str(widget.cget('text'))[:30]
        except tk.TclError:
            pass
        print(f'{"  " * depth}{widget.winfo_class():<12} {geo}  {text}')
        for child in widget.winfo_children():
            walk(child, depth + 1)

    print(f'root  {root.winfo_width()}x{root.winfo_height()}   '
          f'(geometry {WINDOW}, minsize {MINSIZE[0]}x{MINSIZE[1]})')
    print()
    walk(root)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='ui_layout',
                                 description='GELTILT UI layout only.')
    ap.add_argument('--dump', action='store_true',
                    help='print every widget geometry and exit')
    ap.add_argument('--pos', default=None, metavar='X,Y',
                    help='window position, e.g. --pos 40,40')
    ap.add_argument('--keep', action='store_true',
                    help='with --dump, leave the window open')
    args = ap.parse_args(argv)

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f'cannot open a window: {exc}', file=sys.stderr)
        return 2

    build(root)
    if args.pos:
        root.geometry(f'+{args.pos.replace(",", "+")}')
    root.update_idletasks()
    root.update()

    if args.dump:
        dump(root)
        if not args.keep:
            root.destroy()
        return 0

    root.mainloop()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
