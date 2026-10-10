"""Build the standalone projection-viewer executable with PyInstaller.

Usage::

    python build_projview_exe.py

Produces ``dist/geltilt-projview.exe`` (single file, no console window).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAME = "geltilt-projview"


def main() -> int:
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean",
        "--onefile",
        "--windowed",
        "--name", NAME,
        "--collect-submodules", "geltilt",
        "--collect-submodules", "cv2",
        str(ROOT / "run_projview.py"),
    ]
    print("running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())