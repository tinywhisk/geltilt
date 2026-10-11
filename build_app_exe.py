"""Build the standalone unified desktop app with PyInstaller.

Usage::

    python build_app_exe.py

Produces ``dist/geltilt-app.exe`` (single file, no console window).

If a GelGenie ONNX model is available (``GELTILT_GELGENIE_ONNX`` or the cache at
``~/.geltilt``) it is bundled so the packaged app runs offline; otherwise the app
falls back to the runtime cache/download.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAME = "geltilt-app"
MODEL_FILENAME = "gelgenie_universal_1024.onnx"


def _model_file() -> Path | None:
    env = os.environ.get("GELTILT_GELGENIE_ONNX")
    if env and Path(env).exists():
        return Path(env)
    cache = Path(os.environ.get("GELTILT_CACHE", str(Path.home() / ".geltilt")))
    candidate = cache / MODEL_FILENAME
    return candidate if candidate.exists() else None


def main() -> int:
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean",
        "--onefile",
        "--windowed",
        "--name", NAME,
        "--collect-submodules", "geltilt",
        "--collect-submodules", "cv2",
        # `collect-submodules geltilt` otherwise drags in the retired `smears`
        # engine, whose only dependency is the heavy `scipy` (~30 MB). Nothing
        # the app imports needs either, so drop them from the bundle.
        "--exclude-module", "geltilt.smears",
        "--exclude-module", "scipy",
    ]
    for data in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        path = ROOT / data
        if path.exists():
            cmd += ["--add-data", f"{path}{os.pathsep}."]
    licenses = ROOT / "licenses"
    if licenses.is_dir():
        cmd += ["--add-data", f"{licenses}{os.pathsep}licenses"]
    model = _model_file()
    if model is not None:
        cmd += ["--add-data", f"{model}{os.pathsep}."]
        print(f"bundling GelGenie model: {model}")
    else:
        print("no GelGenie model found; the packaged app will use the runtime cache/download")
    cmd.append(str(ROOT / "run_app.py"))
    print("running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())