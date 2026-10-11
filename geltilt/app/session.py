"""JSON-backed session persistence (folder + index + per-sample rotations)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from .domain import SCHEMA_VERSION, AppliedRotation, SessionState


class JsonSessionStore:
    """A single session file inside the run directory."""

    def __init__(self, run_dir: Path, filename: str = "app_session.json") -> None:
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / filename
        self._warned = False

    # -- (de)serialisation -------------------------------------------------- #
    @staticmethod
    def _to_payload(state: SessionState) -> dict:
        return {
            "schema_version": int(state.schema_version),
            "folder": str(state.folder),
            "index": int(state.index),
            "rotations": {
                name: {"value": float(rot.value), "source": str(rot.source)}
                for name, rot in state.rotations.items()
            },
            "updated_at": str(state.updated_at),
        }

    @staticmethod
    def _from_payload(data: object) -> SessionState:
        if not isinstance(data, dict):
            return SessionState.default()
        rotations = {}
        raw_rotations = data.get("rotations")
        if isinstance(raw_rotations, dict):
            for name, rec in raw_rotations.items():
                try:
                    if isinstance(rec, dict):
                        rotations[str(name)] = AppliedRotation(
                            float(rec.get("value", 0.0)), str(rec.get("source", "none"))
                        )
                    else:
                        rotations[str(name)] = AppliedRotation(float(rec), "manual")
                except (TypeError, ValueError):
                    continue
        try:
            index = int(data.get("index", 0) or 0)
        except (TypeError, ValueError):
            index = 0
        return SessionState(
            schema_version=int(data.get("schema_version", SCHEMA_VERSION) or SCHEMA_VERSION),
            folder=str(data.get("folder", "") or ""),
            index=index,
            rotations=rotations,
            updated_at=str(data.get("updated_at", "") or ""),
        )

    # -- port --------------------------------------------------------------- #
    def load(self) -> SessionState:
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError:
            return SessionState.default()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return SessionState.default()
        return self._from_payload(data)

    def save(self, state: SessionState) -> None:
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(self._to_payload(state), indent=2), encoding="utf-8"
            )
            tmp.replace(self.path)
            self._warned = False
        except OSError:
            if not self._warned:
                self._warned = True
                import sys

                print(
                    f"warning: could not write session to {self.path}; "
                    "continuing without persistence",
                    file=sys.stderr,
                )


def make_session(folder: str, index: int, rotations, updated_at: Optional[str] = None) -> SessionState:
    """Build a :class:`SessionState` with a fresh timestamp."""
    stamp = updated_at or time.strftime("%Y-%m-%dT%H:%M:%S")
    return SessionState(
        schema_version=SCHEMA_VERSION,
        folder=str(folder or ""),
        index=int(index),
        rotations=dict(rotations),
        updated_at=stamp,
    )