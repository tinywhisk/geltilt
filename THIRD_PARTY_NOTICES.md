# Third-Party Notices

`geltilt` itself is licensed under the MIT License (see [`LICENSE`](LICENSE)).
The packaged application (`dist/geltilt-app.exe`) is a PyInstaller bundle that
redistributes the third-party components below. Keep this file and the
[`licenses/`](licenses/) folder with any distribution of the app or the exe.

| Component | Version | License | License text |
|---|---|---|---|
| Python runtime | 3.14 | PSF-2.0 | [`licenses/python-LICENSE.txt`](licenses/python-LICENSE.txt) |
| Tcl/Tk (Tkinter) | 9.0 | Tcl/Tk (BSD-style) | see note 1 |
| NumPy | 2.5.3 | BSD-3-Clause | [`licenses/numpy-LICENSE.txt`](licenses/numpy-LICENSE.txt) |
| Pillow | 12.3.0 | MIT-CMU (HPND) | [`licenses/pillow-LICENSE.txt`](licenses/pillow-LICENSE.txt) |
| OpenCV (`opencv-python-headless`) | 5.0.0.93 | Apache-2.0 | [`licenses/opencv-apache-2.0-LICENSE.txt`](licenses/opencv-apache-2.0-LICENSE.txt) |
| PyInstaller (bootloader) | 6.22.3 | GPL-2.0-or-later **with the PyInstaller exception** | [`licenses/pyinstaller-COPYING.txt`](licenses/pyinstaller-COPYING.txt) |
| GelGenie ONNX band-segmentation weights | Dec-2023 build | Apache-2.0 | see note 2 |

Notes
-----

1. **Tcl/Tk** is bundled by the Python runtime. Redistribution requires its
   license text; drop the official `license.terms` from the Tcl/Tk distribution
   into `licenses/tcl-tk-license.terms` and add the row's link.

2. **GelGenie** (github.com/mattaq31/GelGenie) machine-learning model weights are
   Apache-2.0. They are redistributed **unmodified**. Apache-2.0 requires you to
   include the license text (the same text as
   `licenses/opencv-apache-2.0-LICENSE.txt`), to preserve any upstream `NOTICE`
   file, and to state that the work is redistributed unmodified. Add the
   upstream `LICENSE`/`NOTICE` here as `licenses/gelgenie-LICENSE.txt` /
   `licenses/gelgenie-NOTICE.txt`, and record the exact source revision (the
   HuggingFace repo + commit) used to produce the bundled `.onnx`.
   If instead the model is downloaded from the original repository at first run
   (not bundled), you are not redistributing it — but do not imply it is yours.

3. **PyInstaller's** bootloader is GPL-2.0-or-later **with a special exception**
   that explicitly permits building and distributing non-free (including
   commercial) programs; the frozen bundle is therefore not required to be
   GPL-licensed.

4. Everything above except PyInstaller's bootloader is a permissive license
   (MIT / BSD / Apache / PSF / Tcl), so the app may be distributed, including
   commercially and/or as closed source, provided these notices are retained.

Regenerating
------------

The `licenses/` texts were copied from the installed wheels' `dist-info`
license files and the Python install. After a dependency upgrade, re-copy them
and update the version column above.