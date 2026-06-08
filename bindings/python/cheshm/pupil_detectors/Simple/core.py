"""Simple pupil detector — Python wrapper over the nanobind extension.

Threshold-based detector: pixels below ``pupil_threshold`` form the
candidate mask, ``cv::findContours`` walks the dark regions, a shape-
quality filter picks the first that looks pupil-shaped, ``cv::fitEllipse``
runs on the convex hull, and the pupil centre comes from one of five
methods (default: periodic interpolating cubic spline through the convex
hull, with Green's-theorem centroid of the enclosed area).
"""

from typing import Literal

import numpy as np

from cheshm._protocols import PupilResult

from . import _core

# GUI metadata. Defaults / types come from `detect_pupil`'s signature.
_UI = {
    "pupil_threshold": {
        "min": 0,
        "max": 255,
        "help": "Intensity below which a pixel is considered pupil.",
    },
    "pupil_center_method": {"label": "Centre method"},
    "fourier_smoothing": {
        "label": "Fourier smoothing",
        "help": "Fit a robust low-order Fourier pupil-form to the contour, bridging glint / eyelash intrusions.",
    },
    "fourier_harmonics": {
        "min": 2,
        "max": 8,
        "label": "Fourier harmonics",
        "help": "Number of Fourier terms (K). Higher = more shape detail, lower = rounder.",
    },
    "fourier_samples": {
        "min": 90,
        "max": 720,
        "label": "Fourier samples",
        "help": "Number of points on the smoothed margin.",
    },
    "fourier_iterations": {
        "min": 1,
        "max": 8,
        "label": "Fourier iterations",
        "help": "Robust refinement passes (IRLS).",
    },
    "fourier_inward_rejection": {
        "min": 0.3,
        "max": 3.0,
        "label": "Fourier inward rejection",
        "help": "How hard inward intrusions (glints / eyelashes) are ignored. Higher = bridge more aggressively.",
    },
    "glint_merge": {
        "label": "Glint merge",
        "help": "Merge very-white glints (and their halo) into the pupil so they don't carve the contour inward.",
    },
    "glint_threshold": {
        "min": 150,
        "max": 255,
        "label": "Glint threshold",
        "help": "Brightness above which a pixel is treated as a glint to merge into the pupil.",
    },
    "glint_boost_pct": {
        "min": 0.0,
        "max": 100.0,
        "label": "Glint boost %",
        "help": "How much to relax the pupil threshold near a glint, to recover halo-brightened pupil pixels.",
    },
    "glint_reach_px": {
        "min": 0,
        "max": 40,
        "label": "Glint reach px",
        "help": "How far from a glint the relaxed threshold applies.",
    },
    "pupil_roi": {
        "widget": "roi",
        "label": "Pupil ROI",
        "help": "Optional (x, y, w, h) rectangle. None = whole image.",
        "hidden": True,
    },
    "min_ellipse_fit_ratio": {
        "min": 0.0,
        "max": 1.0,
        "label": "Min ellipse-fit ratio",
        "help": "Reject pupils whose contour-to-ellipse area ratio is below this. None = off.",
    },
    "min_roundness_ratio": {
        "min": 0.0,
        "max": 1.0,
        "help": "Reject pupils with 4·π·area / perimeter² below this. None = off.",
    },
}

_OVERLAYS = (
    ("contour", "line"),
    ("ellipse", "line"),
    ("center", "point"),
    ("mask", "fill"),
)

_CENTER_METHOD_CODE = {
    "convex_hull_centroid": 0,
    "center_of_mass": 1,
    "ellipse_fit_center": 2,
    "min_area_rect_center": 3,
    "hull_moments_centroid": 4,
}


def detect_pupil(
    img: np.ndarray,
    pupil_threshold: int = _core.PUPIL_THRESHOLD,
    pupil_center_method: Literal[
        "convex_hull_centroid",
        "center_of_mass",
        "ellipse_fit_center",
        "min_area_rect_center",
        "hull_moments_centroid",
    ] = "convex_hull_centroid",
    pupil_roi: tuple[int, int, int, int] | None = None,
    *,
    fourier_smoothing: bool = _core.FOURIER_SMOOTHING,
    fourier_harmonics: int = _core.FOURIER_HARMONICS,
    fourier_samples: int = _core.FOURIER_SAMPLES,
    fourier_iterations: int = _core.FOURIER_ITERATIONS,
    fourier_inward_rejection: float = _core.FOURIER_INWARD_REJECTION,
    glint_merge: bool = _core.GLINT_MERGE,
    glint_threshold: int = _core.GLINT_THRESHOLD,
    glint_boost_pct: float = _core.GLINT_BOOST_PCT,
    glint_reach_px: int = _core.GLINT_REACH_PX,
    min_ellipse_fit_ratio: float | None = None,
    min_roundness_ratio: float | None = None,
    max_contour_points: int = _core.MAX_CONTOUR_POINTS,
) -> PupilResult | None:
    """Detect the pupil contour, centre, and fitted ellipse in a grayscale image.

    Returns ``{contour, center, ellipse, mask}`` on success, or ``None``
    when no pupil can be produced at the current parameters.

    ``ellipse`` is ``((cx, cy), (w, h), angle)`` from ``cv::fitEllipse``
    on the convex hull of the pupil contour. The pupil centre is chosen
    by ``pupil_center_method``:

      - ``convex_hull_centroid`` — periodic interpolating cubic spline
        through the convex hull, sampled at 200 points; Green's-theorem
        centroid of the enclosed region.
      - ``center_of_mass`` — moments centroid of the contour-masked
        pupil region. The glint hole stays cut out and biases the
        centroid away from the glint side.
      - ``ellipse_fit_center`` — centre of ``cv::fitEllipse``.
      - ``min_area_rect_center`` — centre of ``cv::minAreaRect``.
      - ``hull_moments_centroid`` — moments centroid of the filled
        convex-hull polygon (no spline). Cheaper than
        ``convex_hull_centroid`` but less sub-pixel-stable.

    Border-touching dark contours are rejected so the pupil is always an
    interior region — unless ``pupil_roi=(x, y, w, h)`` is set, in which
    case the detector runs on the cropped sub-image and contours touching
    the crop edge are accepted (an explicit ROI is treated as "the pupil
    is here"). Output coordinates are always in full-image space.

    Two optional post-fit shape-quality gates reject detections that
    don't look pupil-shaped:

    - ``min_ellipse_fit_ratio`` (0..1): ``contour_area /
      fitted_ellipse_area``.
    - ``min_roundness_ratio`` (0..1): ``4·π·area / perimeter²``, the
      isoperimetric quotient. ``1.0`` = perfect circle.

    Both gates default to ``None`` (off). When either is set, candidate
    contours are walked from largest to smallest and the first one that
    satisfies both active gates is chosen. ``None`` is returned only if
    every candidate fails.
    """
    if pupil_roi is None:
        roi_x = roi_y = roi_w = roi_h = 0
    else:
        roi_x, roi_y, roi_w, roi_h = (int(v) for v in pupil_roi)

    result = _core.detect(
        img,
        roi_x,
        roi_y,
        roi_w,
        roi_h,
        pupil_threshold,
        _CENTER_METHOD_CODE[pupil_center_method],
        -1.0 if min_ellipse_fit_ratio is None else float(min_ellipse_fit_ratio),
        -1.0 if min_roundness_ratio is None else float(min_roundness_ratio),
        fourier_smoothing,
        fourier_harmonics,
        fourier_samples,
        fourier_iterations,
        fourier_inward_rejection,
        glint_merge,
        glint_threshold,
        glint_boost_pct,
        glint_reach_px,
        max_contour_points,
    )
    if result is None:
        return None
    (cx, cy), (ecx, ecy, ew, eh, angle), contour_xy, mask = result

    contour = contour_xy.astype(np.int32).reshape(-1, 1, 2)

    return {
        "contour": contour,
        "center": (round(cx), round(cy)),
        "ellipse": ((ecx, ecy), (ew, eh), angle),
        "mask": mask,
    }
