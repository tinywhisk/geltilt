"""Launch-time configuration and run-directory resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import FrozenSet, Optional

from . import i18n
from .settings import load_language

DEFAULT_IMAGE_EXTS: FrozenSet[str] = frozenset(
    {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
)


def run_directory() -> Path:
    """Resolve the default run directory (exe folder when frozen, else CWD)."""
    from ..labeler import run_directory as _run_directory

    return _run_directory()


def resolve_run_dir(override: Optional[object] = None) -> Path:
    """Return the run directory, honouring an explicit launch override."""
    if override:
        return Path(str(override)).expanduser()
    return run_directory()


def default_model_path() -> Optional[Path]:
    """The env-overridden GelGenie ONNX path, or ``None`` for the cached lookup."""
    env = os.environ.get("GELTILT_GELGENIE_ONNX")
    return Path(env) if env else None


def ensure_run_directory(path: Path) -> Path:
    """Create ``path`` if it does not exist yet; return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class AppConfig:
    """Configuration resolved once at launch and held for the session."""

    run_dir: Path
    ui_scale: float = 1.0
    model_path: Optional[Path] = None
    language: str = i18n.DEFAULT_LANGUAGE
    image_exts: FrozenSet[str] = field(default_factory=lambda: DEFAULT_IMAGE_EXTS)
    threshold: float = 0.5
    min_area: int = 30

    @classmethod
    def resolve(cls, run_dir: Optional[object] = None, ui_scale: Optional[float] = None) -> "AppConfig":
        from ..labeler import dpi_scale

        scale = float(ui_scale) if ui_scale else dpi_scale()
        resolved = resolve_run_dir(run_dir)
        return cls(
            run_dir=resolved,
            ui_scale=max(1.0, scale),
            model_path=default_model_path(),
            language=load_language(resolved),
        )