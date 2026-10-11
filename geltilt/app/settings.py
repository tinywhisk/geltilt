"""Persisted user preferences for the desktop app.

Unlike :mod:`geltilt.app.session` (per-run browsing/rotation state), settings are
durable preferences that survive even when no session file exists.  Today the
only setting is the UI language; it is read once at launch so the window can be
built in the right language.
"""

from __future__ import annotations

import json
from pathlib import Path


from . import i18n

SETTINGS_FILENAME = "app_settings.json"


def settings_path(run_dir: object) -> Path:
    return Path(str(run_dir)) / SETTINGS_FILENAME


def _read(run_dir: object) -> dict:
    try:
        text = settings_path(run_dir).read_text(encoding="utf-8")
        data = json.loads(text)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def load_language(run_dir: object) -> str:
    """The persisted UI language, or the default when unset/unknown."""
    code = _read(run_dir).get("language")
    return code if code in i18n.LANGUAGES else i18n.DEFAULT_LANGUAGE


def save_language(run_dir: object, code: str) -> bool:
    """Persist ``code`` (validated) into ``run_dir``; returns success."""
    if code not in i18n.LANGUAGES:
        return False
    path = settings_path(run_dir)
    data = _read(run_dir)
    data["language"] = code
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        return False
    return True