"""Command-line interface for geltilt.

The one and only rotation algorithm: GelGenie band mask -> band centroids -> ``ladders.estimate``.
The reported ``angle`` is passed as-is to ``cv2.getRotationMatrix2D`` (its convention), and a
refusal (``ok=False``) leaves the image unrotated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from . import ladders
from .banddetect import band_centroids, default_segmenter
from .core import correct_rotation
from .imageio import load_array, save_array


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="geltilt",
        description="Auto-correct gel electrophoresis image rotation so that lanes are vertical "
                    "(lane-based estimator).",
    )
    p.add_argument("inputs", nargs="+", help="input image file(s)")
    p.add_argument("-o", "--output", help="output file (only with a single input)")
    p.add_argument("--output-dir", help="output directory (for one or more inputs)")
    p.add_argument("--suffix", default="_corrected", help="suffix for generated filenames")
    p.add_argument("--gelgenie-model", help="GelGenie ONNX model path (default: cached/downloaded)")
    p.add_argument("--threshold", type=float, default=0.5, help="band-probability cutoff")
    p.add_argument("--min-area", type=int, default=30, help="min component area (px) for a band")
    p.add_argument("--expand", action="store_true",
                   help="grow the canvas so no content is cropped")
    p.add_argument("--crop", action="store_true", help="crop output to band content")
    p.add_argument("--dry-run", action="store_true", help="only estimate, do not write images")
    p.add_argument("--report", help="write a JSON report of the estimate to this path")
    p.add_argument("--compare", help="write a side-by-side before/after PNG with vertical guides")
    p.add_argument("--compare-dir",
                   help="write a per-input before/after comparison PNG into this directory")
    p.add_argument("--gamma", type=float, default=1.0, help="display gamma for comparison figures")
    p.add_argument("--quiet", action="store_true", help="suppress per-file output")
    return p


def _output_path(src: Path, args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output)
    name = f"{src.stem}{args.suffix}{src.suffix}"
    if args.output_dir:
        return Path(args.output_dir) / name
    return src.with_name(name)


def _process_one(src: Path, args: argparse.Namespace) -> dict:
    original = load_array(src)
    seg = default_segmenter(args.gelgenie_model)
    xy = band_centroids(original, segmenter=seg, threshold=args.threshold, min_area=args.min_area)
    res = ladders.estimate(xy)

    ok = bool(res.get("ok"))
    angle = float(res.get("angle", 0.0))
    apply_angle = angle if (ok and np.isfinite(angle)) else 0.0
    corrected = correct_rotation(original, apply_angle, expand=args.expand,
                                 crop_to_content_flag=args.crop)

    se = res.get("se")
    se_txt = f"{se:.2f}" if isinstance(se, (int, float)) and np.isfinite(se) else "-"
    status = (f"engine ladders | ok={ok} | se={se_txt} | kept={res.get('kept', 0)}/"
              f"{res.get('n_ladders', 0)} | {res.get('reason', '')}")
    figure_of = {"method": "ladders", "polarity": "-", "status": status,
                 "max_angle": "-", "coarse_step": "-", "fine_step": "-"}
    record = {
        "input": str(src), "engine": "ladders", "ok": ok,
        "angle": round(apply_angle, 3),
        "angle_reported": round(angle, 3) if np.isfinite(angle) else None,
        "se": se, "reason": res.get("reason", ""), "warnings": res.get("warnings", []),
        "kept": res.get("kept"), "n_ladders": res.get("n_ladders"),
        "n_bands": res.get("n_bands"), "fallback": res.get("fallback"),
        "overdispersion": res.get("overdispersion"),
    }

    if not args.dry_run:
        out = _output_path(src, args)
        out.parent.mkdir(parents=True, exist_ok=True)
        save_array(out, corrected)
        record["output"] = str(out)

    targets = []
    if args.compare:
        targets.append(Path(args.compare))
    if args.compare_dir:
        dest = Path(args.compare_dir) / f"{src.stem}_compare.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        targets.append(dest)
    if targets:
        from .viz import render_comparison, save_comparison

        for dest in targets:
            figure = render_comparison(original, corrected, apply_angle,
                                       info=figure_of, source_name=src.name, gamma=args.gamma)
            save_comparison(dest, figure)
            record["compare"] = str(dest)

    if not args.quiet:
        print(f"{src.name}: [ladders] angle {apply_angle:+.3f} deg  ok={ok}  {res.get('reason', '')}")
    return record


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = [Path(p) for p in args.inputs]
    for src in inputs:
        if not src.exists():
            print(f"error: input not found: {src}", file=sys.stderr)
            return 2
    if args.output and len(inputs) != 1:
        print("error: --output is only valid with a single input", file=sys.stderr)
        return 2
    if args.compare and len(inputs) != 1:
        print("error: --compare is only valid with a single input", file=sys.stderr)
        return 2

    records: List[dict] = []
    for src in inputs:
        try:
            records.append(_process_one(src, args))
        except Exception as exc:  # noqa: BLE001 - keep the batch going
            print(f"error: {src.name}: {exc}", file=sys.stderr)
            records.append({"input": str(src), "error": str(exc)})

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = records[0] if len(records) == 1 else records
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0