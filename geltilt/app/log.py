"""Append-only JSON-lines diagnostic run log (metadata only)."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .domain import DiagnosticLogEntry


class JsonlDiagnosticLog:
    """Appends one JSON line per auto-rotation run inside the run directory."""

    def __init__(self, run_dir: Path, filename: str = "app_runs.jsonl") -> None:
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / filename

    @staticmethod
    def _to_record(entry: DiagnosticLogEntry) -> dict:
        angle = float(entry.angle)
        se = float(entry.se)
        return {
            "timestamp": str(entry.timestamp),
            "sample": str(entry.sample),
            "angle": angle if math.isfinite(angle) else None,
            "se": se if math.isfinite(se) else None,
            "ok": bool(entry.ok),
            "reason": str(entry.reason),
            "warnings": [str(w) for w in entry.warnings],
        }

    def append(self, entry: DiagnosticLogEntry) -> None:
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            # Never alias a source image: the log path always ends in .jsonl.
            record = self._to_record(entry)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        except OSError:
            pass