from __future__ import annotations

import json
from pathlib import Path

from geltilt.app.domain import DiagnosticLogEntry
from geltilt.app.log import JsonlDiagnosticLog


def test_log_appends_one_json_line_per_run(run_dir: Path):
    log = JsonlDiagnosticLog(run_dir)
    log.append(DiagnosticLogEntry("2026-01-01T00:00:00", "a_gel.png", 2.4, 0.3, True, "ok",
                                  ("warn",)))
    log.append(DiagnosticLogEntry("2026-01-01T00:01:00", "b_gel.png", float("nan"),
                                  float("nan"), False, "fewer than 4 bands", ()))
    lines = log.path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first == {
        "timestamp": "2026-01-01T00:00:00",
        "sample": "a_gel.png",
        "angle": 2.4,
        "se": 0.3,
        "ok": True,
        "reason": "ok",
        "warnings": ["warn"],
    }
    second = json.loads(lines[1])
    assert second["angle"] is None and second["se"] is None
    assert second["ok"] is False


def test_log_is_metadata_only_and_never_an_image_path(run_dir: Path):
    log = JsonlDiagnosticLog(run_dir)
    assert log.path.suffix == ".jsonl"
    log.append(DiagnosticLogEntry("t", "s", 1.0, 0.1, True, "ok", ()))
    record = json.loads(log.path.read_text(encoding="utf-8").strip())
    assert set(record) == {"timestamp", "sample", "angle", "se", "ok", "reason", "warnings"}