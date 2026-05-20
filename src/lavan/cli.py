"""CLI entry point: ``detect`` — run pupil + glint detection on one image.

Prints the combined JSON-serialisable result. Only the most-used knobs
are surfaced as flags; advanced refiners keep their library defaults —
callers who need them invoke the API directly.
"""

import argparse
import json
from pathlib import Path

import cv2

from lavan.glint_detectors.Simple import detect_glints
from lavan.pupil_detectors.Simple import detect_pupil

CENTER_METHOD_CHOICES = (
    "convex_hull_centroid",
    "hull_moments_centroid",
    "center_of_mass",
    "ellipse_fit_center",
    "min_area_rect_center",
)


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
    p.add_argument(
        "--pupil-center-method",
        choices=CENTER_METHOD_CHOICES,
        default="convex_hull_centroid",
        help="(default: convex_hull_centroid).",
    )
    p.add_argument(
        "--min-ellipse-fit-ratio",
        type=float,
        default=None,
        help=(
            "Reject pupils whose contour area divided by fitted-ellipse area "
            "is below this fraction (0..1). Catches fragmented or under-filled "
            "detections. Default: off (no gate)."
        ),
    )
    p.add_argument(
        "--min-roundness-ratio",
        type=float,
        default=None,
        help=(
            "Reject pupils whose 4*pi*area / perimeter^2 is below this fraction "
            "(0..1). 1.0 = perfect circle. Use only with on-axis cameras; off-axis "
            "pupils look elliptical and score low. Default: off (no gate)."
        ),
    )
    p.add_argument("--glint-threshold", type=int, default=240, help="(default: 240).")
    p.add_argument(
        "--search-radius-factor",
        type=float,
        default=2.0,
        help="Glint search disk radius as a multiple of the pupil radius (default: 2.0).",
    )
    p.add_argument(
        "--glint-center-method",
        choices=CENTER_METHOD_CHOICES,
        default="min_area_rect_center",
        help="(default: min_area_rect_center).",
    )
    p.add_argument("--glints-target", type=int, default=1, help="number of IR LEDs (default: 1).")
    p.add_argument(
        "--glint-min-ellipse-fit-ratio",
        type=float,
        default=None,
        help=(
            "Reject glints whose contour area divided by fitted-ellipse area "
            "is below this fraction (0..1). Default: off (no gate)."
        ),
    )
    p.add_argument(
        "--glint-min-roundness-ratio",
        type=float,
        default=None,
        help=(
            "Reject glints whose 4*pi*area / perimeter^2 is below this fraction "
            "(0..1). 1.0 = perfect circle. Default: off (no gate)."
        ),
    )
    p.set_defaults(handler=_handle_detect)


def _handle_detect(args: argparse.Namespace) -> None:
    img = cv2.imread(str(args.input), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"failed to read {args.input}")

    pupil = detect_pupil(
        img,
        pupil_threshold=args.pupil_threshold,
        pupil_center_method=args.pupil_center_method,
        min_ellipse_fit_ratio=args.min_ellipse_fit_ratio,
        min_roundness_ratio=args.min_roundness_ratio,
    )
    if pupil is None:
        raise SystemExit("pupil detection produced no result at the given parameters.")

    (ecx, ecy), (w, h), angle = pupil["ellipse"]
    pupil_radius = max(w, h) / 2.0
    glints_result = detect_glints(
        img,
        pupil_center=pupil["center"],
        pupil_radius=pupil_radius,
        glint_threshold=args.glint_threshold,
        search_radius_factor=args.search_radius_factor,
        glint_center_method=args.glint_center_method,
        glints_target=args.glints_target,
        min_ellipse_fit_ratio=args.glint_min_ellipse_fit_ratio,
        min_roundness_ratio=args.glint_min_roundness_ratio,
    )

    serialisable = {
        "pupil_center": list(pupil["center"]),
        "pupil_ellipse": {
            "center": [float(ecx), float(ecy)],
            "size": [float(w), float(h)],
            "angle": float(angle),
        },
        "glints": [{"center": list(g["center"])} for g in glints_result["glints"]],
    }
    payload = json.dumps(serialisable, indent=2)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(payload)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``lavan-detect`` CLI."""
    parser = argparse.ArgumentParser(
        prog="lavan-detect",
        description="Pupil + glint detection on grayscale eye images.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _add_detect_parser(sub)

    args = parser.parse_args(argv)
    args.handler(args)
