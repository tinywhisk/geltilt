from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pytest
from PIL import Image

from geltilt.app.config import AppConfig
from geltilt.app.domain import EstimateOutcome, LaneLine


# --------------------------------------------------------------------------- #
# Test doubles                                                                 #
# --------------------------------------------------------------------------- #
class FakeRoot:
    """Minimal stand-in for ``tk.Tk`` that no-ops ``after`` scheduling."""

    def __init__(self) -> None:
        self.after_calls = 0

    def after(self, _ms: int, _fn=None):
        self.after_calls += 1
        return f"after-{self.after_calls}"


class FakeEstimator:
    """Deterministic estimator test double (no ONNX, no pixels read)."""

    def __init__(self, outcome: Optional[EstimateOutcome] = None, exc: Optional[Exception] = None,
                 delay: float = 0.0) -> None:
        self.outcome = outcome or make_outcome()
        self.exc = exc
        self.delay = delay
        self.calls = 0

    def estimate(self, image: np.ndarray) -> EstimateOutcome:
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.exc is not None:
            raise self.exc
        return self.outcome


def make_outcome(
    ok: bool = True,
    angle: float = 2.0,
    se: float = 0.3,
    reason: str = "ok",
    warnings: Sequence[str] = (),
    n_bands: int = 20,
    lanes_used: int = 2,
    lanes_found: int = 3,
    fallback: bool = False,
    centroids: Optional[np.ndarray] = None,
    lanes: Sequence[LaneLine] = (),
) -> EstimateOutcome:
    if centroids is None:
        centroids = np.array([[10.0, 20.0], [30.0, 40.0]], float)
    return EstimateOutcome(
        ok=ok,
        angle=angle,
        se=se,
        reason=reason,
        warnings=tuple(warnings),
        n_bands=n_bands,
        lanes_used=lanes_used,
        lanes_found=lanes_found,
        fallback=fallback,
        centroids=centroids,
        accepted_lanes=tuple(lanes),
        raw={},
    )


def pump(controller, timeout: float = 5.0) -> None:
    """Drive the controller's poll loop until it goes idle."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        controller._poll()
        if not controller.service.busy and not controller._polling:
            return
        time.sleep(0.005)
    raise AssertionError("controller did not become idle in time")


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    path = tmp_path / "run"
    path.mkdir()
    return path


@pytest.fixture
def app_config(run_dir: Path) -> AppConfig:
    return AppConfig(run_dir=run_dir, ui_scale=1.0, model_path=None)


@pytest.fixture
def sample_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "samples"
    folder.mkdir()
    rng = np.random.default_rng(0)
    for name in ("b_gel.png", "a_gel.png", "c_gel.png"):
        arr = rng.integers(0, 255, size=(32, 24), dtype=np.uint8)
        Image.fromarray(arr, mode="L").save(folder / name)
    (folder / "notes.txt").write_text("not an image", encoding="utf-8")
    return folder


@pytest.fixture
def tk_root():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:  # pragma: no cover - headless environment
        pytest.skip("no display available for Tk")
    root.withdraw()
    try:
        yield root
    finally:
        root.destroy()