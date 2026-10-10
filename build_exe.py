"""Build the standalone labeller executable with PyInstaller.

Usage::

    python build_exe.py

Produces ``dist/geltilt-label.exe`` (single file, no console window).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAME = "geltilt-label"


def main() -> int:
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean",
        "--onefile",
        "--windowed",
        "--name", NAME,
        "--collect-submodules", "geltilt",
        "--collect-submodules", "cv2",
        str(ROOT / "run_labeler.py"),
    ]
    print("running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())