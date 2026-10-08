"""Command-line interface for geltilt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .core import auto_correct
from .imageio import load_array, save_array
from .projection import MAXIMIZE


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="geltilt",
        description="Auto-correct gel electrophoresis image rotation so that lanes are vertical.",
    )
    p.add_argument("inputs", nargs="+", help="input image file(s)")
    p.add_argument("-o", "--output", help="output file (only with a single input)")
    p.add_argument("--output-dir", help="output directory (for one or more inputs)")
    p.add_argument("--suffix", default="_corrected", help="suffix for generated filenames")
    p.add_argument("--max-angle", type=float, default=15.0, help="search range +/- degrees")
    p.add_argument("--coarse-step", type=float, default=0.5, help="coarse sweep step (deg)")
    p.add_argument("--fine-step", type=float, default=0.05, help="fine sweep step (deg)")
    p.add_argument("--method", default="tv2", choices=sorted(MAXIMIZE),
                   help="projection sharpness metric")
    p.add_argument("--polarity", default="auto", choices=["auto", "bright", "dark"],
                   help="band polarity (auto-detected by default)")
    p.add_argument("--no-log", action="store_true", help="disable log compression of signal")
    p.add_argument("--no-roi", action="store_true", help="analyse the whole frame")
    p.add_argument("--downsample", type=int, default=1200,
                   help="max working dimension during estimation (0 disables)")
    p.add_argument("--expand", action="store_true",
                   help="grow the canvas so no content is cropped")
    p.add_argument("--crop", action="store_true", help="crop output to band content")
    p.add_argument("--dry-run", action="store_true", help="only estimate, do not write images")
    p.add_argument("--report", help="write a JSON report of the estimate to this path")
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
    corrected, estimate = auto_correct(
        original,
        expand=args.expand,
        crop=args.crop,
        max_angle=args.max_angle,
        coarse_step=args.coarse_step,
        fine_step=args.fine_step,
        method=args.method,
        polarity=args.polarity,
        log=not args.no_log,
        use_roi=not args.no_roi,
        downsample_to=args.downsample or None,
    )
    record = {"input": str(src), **estimate.to_dict()}
    if not args.dry_run:
        out = _output_path(src, args)
        out.parent.mkdir(parents=True, exist_ok=True)
        save_array(out, corrected)
        record["output"] = str(out)
    if not args.quiet:
        detail = f"{src.name}: angle {estimate.angle:+.3f} deg " \
                 f"(method={estimate.method}, polarity={estimate.polarity})"
        if "output" in record:
            detail += f" -> {record['output']}"
        print(detail)
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


    records: List[dict] = [_process_one(src, args) for src in inputs]

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = records[0] if len(records) == 1 else records
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0