from __future__ import annotations

from pathlib import Path

from geltilt.app.domain import AppliedRotation
from geltilt.app.session import JsonSessionStore, make_session


def test_session_round_trip(run_dir: Path):
    store = JsonSessionStore(run_dir)
    state = make_session(
        folder="C:/gels/batch7",
        index=5,
        rotations={
            "a_gel.png": AppliedRotation(2.3, "manual"),
            "b_gel.png": AppliedRotation(-1.0, "auto"),
            "c_gel.png": AppliedRotation(0.0, "none"),
        },
        updated_at="2026-01-01T00:00:00",
    )
    store.save(state)
    loaded = store.load()
    assert loaded.folder == "C:/gels/batch7"
    assert loaded.index == 5
    assert loaded.rotations["a_gel.png"].value == 2.3
    assert loaded.rotations["a_gel.png"].source == "manual"
    assert loaded.rotations["b_gel.png"].value == -1.0
    assert loaded.rotations["b_gel.png"].source == "auto"
    assert loaded.updated_at == "2026-01-01T00:00:00"


def test_session_missing_file_returns_default(run_dir: Path):
    store = JsonSessionStore(run_dir)
    state = store.load()
    assert state.folder == ""
    assert state.index == 0
    assert state.rotations == {}


def test_session_corrupt_file_returns_default(run_dir: Path):
    store = JsonSessionStore(run_dir)
    store.path.write_text("{not valid json", encoding="utf-8")
    state = store.load()
    assert state.folder == ""
    assert state.rotations == {}


def test_session_ignores_malformed_rotation_entries(run_dir: Path):
    store = JsonSessionStore(run_dir)
    store.path.write_text(
        '{"schema_version": 1, "folder": "x", "index": 2, '
        '"rotations": {"good.png": {"value": 1.5, "source": "manual"}, "bad.png": {"value": "oops"}}}',
        encoding="utf-8",
    )
    state = store.load()
    assert "good.png" in state.rotations
    assert "bad.png" not in state.rotations


def test_session_write_failure_warns_once(run_dir: Path, capsys):
    store = JsonSessionStore(run_dir)
    # Point the store at a path whose parent is a file, forcing OSError.
    blocker = run_dir / "blocker"
    blocker.write_text("x", encoding="utf-8")
    store.run_dir = blocker
    store.path = blocker / "app_session.json"
    store.save(make_session("f", 0, {}))
    store.save(make_session("f", 0, {}))
    err = capsys.readouterr().err
    assert err.count("could not write session") == 1