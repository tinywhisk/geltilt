from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from geltilt.bandmask import default_model_path
from geltilt.cli import main
from geltilt.imageio import load_array, save_array
from geltilt.rotation import rotate_image

from conftest import make_gel


def _model_ready() -> bool:
    env = os.environ.get("GELTILT_GELGENIE_ONNX")
    return bool(env and Path(env).exists()) or default_model_path().exists()


requires_model = pytest.mark.skipif(not _model_ready(), reason="GelGenie ONNX model not available")


def _write(path, tilt=0.0):
    img = make_gel(seed=7)
    if tilt:
        img = rotate_image(img, tilt)
    save_array(path, np.clip(img, 0, 255).astype(np.uint8))


@requires_model
def test_cli_writes_output_and_report(tmp_path):
    src = tmp_path / "gel.tif"
    out = tmp_path / "gel_fixed.tif"
    report = tmp_path / "report.json"
    _write(src, tilt=5.0)

    rc = main([str(src), "-o", str(out), "--report", str(report), "--quiet"])
    assert rc == 0
    assert out.exists()

    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["input"] == str(src)
    assert data["engine"] == "ladders"
    assert isinstance(data["ok"], bool)
    assert "angle" in data and "reason" in data and "se" in data
    assert load_array(out).shape == load_array(src).shape


@requires_model
def test_cli_dry_run_writes_nothing(tmp_path):
    src = tmp_path / "gel.tif"
    _write(src, tilt=3.0)
    rc = main([str(src), "--dry-run", "--quiet"])
    assert rc == 0
    assert not (tmp_path / "gel_corrected.tif").exists()


@requires_model
def test_cli_default_output_name(tmp_path):
    src = tmp_path / "gel.tif"
    _write(src)
    rc = main([str(src), "--quiet"])
    assert rc == 0
    assert (tmp_path / "gel_corrected.tif").exists()


@requires_model
def test_cli_output_dir_multiple_inputs(tmp_path):
    src1 = tmp_path / "a.tif"
    src2 = tmp_path / "b.tif"
    outdir = tmp_path / "out"
    _write(src1, tilt=4.0)
    _write(src2, tilt=-4.0)
    rc = main([str(src1), str(src2), "--output-dir", str(outdir), "--quiet"])
    assert rc == 0
    assert (outdir / "a_corrected.tif").exists()
    assert (outdir / "b_corrected.tif").exists()


@requires_model
def test_cli_compare_writes_figure(tmp_path):
    src = tmp_path / "gel.tif"
    fig = tmp_path / "compare.png"
    _write(src, tilt=5.0)
    rc = main([str(src), "--dry-run", "--compare", str(fig), "--quiet"])
    assert rc == 0
    assert fig.exists()
    assert fig.stat().st_size > 0


@requires_model
def test_cli_dry_run_compare_dir_writes_only_figures(tmp_path):
    src = tmp_path / "gel.tif"
    outdir = tmp_path / "rot"
    _write(src, tilt=5.0)
    rc = main([str(src), "--dry-run", "--compare-dir", str(outdir), "--quiet"])
    assert rc == 0
    assert (outdir / "gel_compare.png").exists()
    assert not list(outdir.glob("*_corrected*"))


def test_cli_missing_input(tmp_path):
    rc = main([str(tmp_path / "nope.tif"), "--quiet"])
    assert rc == 2


@requires_model
def test_cli_output_requires_single_input(tmp_path):
    src1 = tmp_path / "a.tif"
    src2 = tmp_path / "b.tif"
    _write(src1)
    _write(src2)
    rc = main([str(src1), str(src2), "-o", str(tmp_path / "x.tif"), "--quiet"])
    assert rc == 2