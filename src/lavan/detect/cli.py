"""CLI entry point: ``tune`` (interactive GUI) and ``detect`` (single-image)."""

import argparse
import json
from pathlib import Path


def _add_tune_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "tune",
        help="Open the interactive GUI to tune pupil + glint thresholds per image.",
    )
    p.add_argument(
        "--input",
        required=True,
        type=Path,
        help="A single PNG image, or a directory containing PNGs (walked recursively).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON path to save thresholds to (default: <input>/thresholds.json for directory input, or <input>.thresholds.json for single image).",
    )
    p.add_argument(
        "--pattern",
        default="*.png",
        help="Glob pattern for directory input (default: '*.png').",
    )
    p.set_defaults(handler=_handle_tune)


def _handle_tune(args: argparse.Namespace) -> None:
    from .gui import tune_thresholds  # noqa: PLC0415  (defer matplotlib until tune runs)

    if args.input.is_file():
        images = [args.input]
        output = args.output or args.input.with_suffix(".thresholds.json")
    elif args.input.is_dir():
        images = sorted(p for p in args.input.rglob(args.pattern) if p.is_file())
        if not images:
            raise SystemExit(f"no images matching {args.pattern!r} under {args.input}")
        output = args.output or args.input / "thresholds.json"
    else:
        raise SystemExit(f"--input not a file or directory: {args.input}")

    keys = [str(p.relative_to(args.input)) if args.input.is_dir() else p.name for p in images]
    tune_thresholds(images=images, output_path=output, keys=keys)


def _add_detect_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "detect",
        help="Run pupil + glint detection on one image and print/save the result.",
    )
    p.add_argument("--input", required=True, type=Path, help="Grayscale eye-image PNG.")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON path to save the detection result. Default: print to stdout.",
    )
    p.add_argument("--pupil-threshold", type=int, default=30, help="(default: 30).")
    p.add_argument("--glint-threshold", type=int, default=240, help="(default: 240).")
    p.add_argument("--glint-margin", type=int, default=10, help="px tolerance around pupil (default: 10).")
    p.add_argument("--glints-target", type=int, default=1, help="number of IR LEDs (default: 1).")
    p.add_argument(
        "--glint-max-area-ratio",
        type=float,
        default=0.1,
        help="reject glint candidates exceeding this fraction of pupil area (default: 0.1).",
    )
    p.add_argument(
        "--pupil-center-method",
        choices=("convex_hull_centroid", "center_of_mass"),
        default="convex_hull_centroid",
        help="(default: convex_hull_centroid).",
    )
    p.set_defaults(handler=_handle_detect)


def _handle_detect(args: argparse.Namespace) -> None:
    import cv2  # noqa: PLC0415  (defer cv2 import to subcommand dispatch)

    from .core import detect_pupil_and_glints  # noqa: PLC0415

    img = cv2.imread(str(args.input), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"failed to read {args.input}")

    det = detect_pupil_and_glints(
        img,
        pupil_threshold=args.pupil_threshold,
        glint_threshold=args.glint_threshold,
        glint_margin=args.glint_margin,
        glints_target=args.glints_target,
        glint_max_area_ratio=args.glint_max_area_ratio,
        pupil_center_method=args.pupil_center_method,
    )
    serialisable = {
        "pupil_center": det["pupil_center"],
        "pupil_ellipse": det["pupil_ellipse"],
        "glints": [{"center": g["center"], "ellipse": g["ellipse"]} for g in det["glints"]],
    }
    payload = json.dumps(serialisable, indent=2)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(payload)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``pupil-glint-detector`` CLI."""
    parser = argparse.ArgumentParser(
        prog="pupil-glint-detector",
        description="Pupil + glint detection on grayscale eye images.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _add_tune_parser(sub)
    _add_detect_parser(sub)

    args = parser.parse_args(argv)
    args.handler(args)
