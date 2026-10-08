from __future__ import annotations

import numpy as np

from geltilt.imageio import to_gray
from geltilt.rotation import rotate_image
from geltilt.viz import (
    auto_contrast,
    content_window,
    display_crop,
    render_comparison,
    render_preprocessing_comparison,
    save_comparison,
)

from conftest import make_gel


def test_auto_contrast_expands_range():
    img = (np.random.default_rng(0).random((32, 32)) * 40 + 10).astype(np.uint8)
    out = auto_contrast(img)
    assert out.dtype == np.uint8
    assert out.max() - out.min() > img.max() - img.min()


def test_auto_contrast_handles_flat_image():
    out = auto_contrast(np.full((8, 8), 7, np.uint8))
    assert out.shape == (8, 8)


def test_content_window_and_crop_around_bands():
    img = np.zeros((100, 100), np.float32)
    img[40:60, 30:70] = 200.0
    window = content_window(img, margin_frac=0.0, min_margin=0)
    assert window is not None
    cropped = display_crop(img, margin_frac=0.0, min_margin=0)
    assert cropped.shape[0] < img.shape[0]
    assert cropped.shape[1] < img.shape[1]


def test_render_comparison_shape_and_channels(gel_factory):
    original = gel_factory()
    corrected = rotate_image(original, 3.0)
    fig = render_comparison(
        original, corrected, -3.0,
        info={"method": "tv2", "polarity": "bright"},
        source_name="gel.tif",
        panel_height=200,
    )
    assert fig.ndim == 3 and fig.shape[2] == 3
    assert fig.dtype == np.uint8
    assert fig.shape[0] > 200
    assert fig.shape[1] > 100


def test_save_comparison_roundtrip(tmp_path, gel_factory):
    original = gel_factory()
    fig = render_comparison(original, original, 0.0, panel_height=120)
    path = tmp_path / "cmp.png"
    save_comparison(path, fig)
    assert path.exists()


def test_to_gray_smoke():
    assert to_gray(np.zeros((4, 4), np.uint8)).shape == (4, 4)

def test_render_preprocessing_comparison_shape(gel_factory):
    img = gel_factory()
    fig = render_preprocessing_comparison(img, source_name="gel.tif", panel_height=200)
    assert fig.ndim == 3 and fig.shape[2] == 3
    assert fig.dtype == np.uint8
    assert fig.shape[0] > 200
    assert fig.shape[1] > 100