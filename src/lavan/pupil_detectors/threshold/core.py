"""Threshold-based pupil detector.

  - :func:`detect_pupil` — pupil contour, center, fitted ellipse, mask.

One-shot and stateless: no temporal tracking, no calibration, no model
fitting across frames.
"""

from typing import Literal

import cv2
import numpy as np

from lavan._common import _contour_center, _passes_shape_quality, _roi_mask

from ..centers import fit_convex_hull_spline, pupil_center_of_mass

# GUI metadata. Defaults / types / choices come from the function
# signature; this dict carries the bits that don't fit in Python's type
# system (slider bounds, per-param help, widget hints, label overrides
# where the auto-derived label would be wrong).
_UI = {
    "pupil_threshold": {
        "min": 0,
        "max": 255,
        "help": "Intensity below which a pixel is considered pupil.",
    },
    "pupil_center_method": {"label": "Centre method"},
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

# Overlay elements this detector produces. Each tuple is (key_in_result, element_type).
# element_type ∈ {"line", "point", "fill"} drives the per-element widget set.
_OVERLAYS = (
    ("contour", "line"),
    ("ellipse", "line"),
    ("center", "point"),
    ("mask", "fill"),
)


def _touches_border(contour: np.ndarray, shape: tuple[int, ...]) -> bool:
    h, w = shape[:2]
    x, y, cw, ch = cv2.boundingRect(contour)
    return x == 0 or y == 0 or x + cw == w or y + ch == h


def detect_pupil(
    img: np.ndarray,
    pupil_threshold: int = 30,
    pupil_center_method: Literal[
        "convex_hull_centroid",
        "center_of_mass",
        "ellipse_fit_center",
        "min_area_rect_center",
    ] = "convex_hull_centroid",
    pupil_roi: tuple[int, int, int, int] | None = None,
    *,
    min_ellipse_fit_ratio: float | None = None,
    min_roundness_ratio: float | None = None,
) -> dict | None:
    """Detect the pupil contour, centre, and fitted ellipse in a grayscale image.

    Returns ``{contour, center, ellipse, mask}`` on success, or ``None``
    when no pupil can be produced at the current parameters (no candidate
    contour, hull with fewer than 5 points, zero-mass mask, etc.).

    ``ellipse`` is ``((cx, cy), (w, h), angle)`` from ``cv2.fitEllipse``
    on the convex hull of the pupil contour. The pupil centre is chosen
    by ``pupil_center_method``; see :func:`lavan._common._contour_center`
    for the four contour-based methods, plus ``"center_of_mass"`` which
    uses :func:`lavan.pupil_detectors.centers.pupil_center_of_mass` (the
    glint hole stays cut out) and ``"convex_hull_centroid"`` which uses
    the spline centroid for sub-pixel stability.

    Border-touching dark contours are rejected so the pupil is always an
    interior region — unless ``pupil_roi`` is set, in which case the
    largest contour inside the ROI is accepted regardless of border
    contact (an explicit ROI is treated as "the pupil is here").

    Two optional post-fit shape-quality gates reject detections that
    don't look pupil-shaped:

    - ``min_ellipse_fit_ratio`` (0..1): ``contour_area /
      fitted_ellipse_area``. Catches detections whose contour is
      fragmented or under-fills its fit (irregular blobs, multi-component
      masks).
    - ``min_roundness_ratio`` (0..1): ``4·π·area / perimeter²``, the
      isoperimetric quotient. ``1.0`` = perfect circle. Catches
      jagged or elongated contours; only useful when the camera views
      the eye on-axis (off-axis pupils are legitimately elliptical and
      score well below 1.0).

    Both gates default to ``None`` (off). When either is set, candidate
    contours are walked from largest to smallest and the first one that
    satisfies both active gates is chosen. ``None`` is returned only if
    every candidate fails. With both gates off, behaviour matches the
    no-gate baseline: the largest contour wins without any shape check.
    """
    _, pupil_mask = cv2.threshold(img, pupil_threshold, 255, cv2.THRESH_BINARY_INV)
    if pupil_roi is not None:
        pupil_mask = cv2.bitwise_and(pupil_mask, _roi_mask(img.shape, pupil_roi))
    contours, _ = cv2.findContours(pupil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if pupil_roi is not None:
        # An explicit ROI overrides the border-rejection filter — the
        # filter only makes sense to guard against frame vignette.
        candidates = [c for c in contours if cv2.contourArea(c) > 0]
    else:
        candidates = [c for c in contours if not _touches_border(c, img.shape)]
    if not candidates:
        return None
    candidates.sort(key=cv2.contourArea, reverse=True)
    pupil_contour = None
    ellipse_fit = None
    for candidate in candidates:
        candidate_hull = cv2.convexHull(candidate)
        if len(candidate_hull) < 5:
            continue
        candidate_fit = cv2.fitEllipse(candidate_hull)
        if not _passes_shape_quality(
            candidate,
            candidate_fit,
            min_ellipse_fit_ratio=min_ellipse_fit_ratio,
            min_roundness_ratio=min_roundness_ratio,
        ):
            continue
        pupil_contour = candidate
        ellipse_fit = candidate_fit
        break
    if pupil_contour is None:
        return None

    if pupil_center_method == "center_of_mass":
        com = pupil_center_of_mass(pupil_mask, pupil_contour)
        if com is None:
            return None
        cx, cy = com
    elif pupil_center_method == "convex_hull_centroid":
        cx, cy = fit_convex_hull_spline(pupil_contour)["center"]
    elif pupil_center_method == "ellipse_fit_center":
        (cx, cy), _, _ = ellipse_fit
    else:
        cx, cy = _contour_center(pupil_contour, pupil_center_method)

    _, (w, h), angle = ellipse_fit
    return {
        "contour": pupil_contour,
        "center": (round(cx), round(cy)),
        "ellipse": ((cx, cy), (w, h), angle),
        "mask": pupil_mask,
    }
