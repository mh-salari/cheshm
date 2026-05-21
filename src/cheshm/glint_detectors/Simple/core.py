"""Simple glint detector — Python wrapper over the nanobind extension.

Threshold-based detector: bright pixels inside the search disk around
the pupil centre form the candidate region, contours are filtered by
area / half-plane / shape-quality gates, optionally split when the rig
has one more LED than contours found, sorted left-to-right, and each
glint's centre is computed via one of five methods.
"""

from typing import Literal

import cv2
import numpy as np

from cheshm._protocols import GlintResult

from . import _core

_OVERLAYS = (
    ("contour", "line"),
    ("center", "point"),
    ("mask", "fill"),
)

_UI = {
    "pupil_center": {"hidden": True},
    "pupil_radius": {"hidden": True},
    "glint_threshold": {
        "min": 0,
        "max": 255,
        "help": "Intensity above which a pixel is considered glint.",
    },
    "search_radius_factor": {
        "min": 0.1,
        "max": 10.0,
        "help": "Multiplied by pupil_radius to define the glint search disk. Ignored when pupil_center / pupil_radius are not supplied.",
    },
    "search_radius_max_px": {
        "min": 1,
        "max": 4096,
        "label": "Search radius max (px)",
        "help": "Optional upper bound on the search disk radius. None = no cap.",
    },
    "glint_roi": {
        "widget": "roi",
        "label": "Glint ROI",
        "help": "Optional (x, y, w, h) rectangle.",
        "hidden": True,
    },
    "glint_center_method": {"label": "Centre method"},
    "max_area_px": {
        "min": 1,
        "max": 100000,
        "label": "Max area (px)",
        "help": "Reject glint blobs larger than this. None = no cap.",
    },
    "keep_above": {"label": "Keep above pupil"},
    "keep_below": {"label": "Keep below pupil"},
    "keep_left": {"label": "Keep left of pupil"},
    "keep_right": {"label": "Keep right of pupil"},
    "filter_margin_px": {
        "min": 0,
        "max": 100,
        "label": "Half-plane margin (px)",
    },
    "glints_target": {
        "min": 1,
        "max": 8,
        "label": "Target glint count",
        "help": "Number of IR LEDs expected.",
    },
    "split_widest_for_target": {
        "label": "Split widest blob",
        "help": "When the target count is N but only one blob is found, split it into N along its long axis.",
    },
    "min_ellipse_fit_ratio": {
        "min": 0.0,
        "max": 1.0,
        "label": "Min ellipse-fit ratio",
    },
    "min_roundness_ratio": {
        "min": 0.0,
        "max": 1.0,
    },
}

_CENTER_METHOD_CODE = {
    "convex_hull_centroid": 0,
    "center_of_mass": 1,
    "ellipse_fit_center": 2,
    "min_area_rect_center": 3,
    "hull_moments_centroid": 4,
}


def detect_glints(
    img: np.ndarray,
    *,
    pupil_center: tuple[float, float] | None = None,
    pupil_radius: float | None = None,
    glint_threshold: int = _core.GLINT_THRESHOLD,
    search_radius_factor: float = _core.SEARCH_RADIUS_FACTOR,
    search_radius_max_px: int | None = None,
    glint_roi: tuple[int, int, int, int] | None = None,
    glint_center_method: Literal[
        "convex_hull_centroid",
        "hull_moments_centroid",
        "center_of_mass",
        "ellipse_fit_center",
        "min_area_rect_center",
    ] = "min_area_rect_center",
    max_area_px: int | None = None,
    keep_above: bool = _core.KEEP_ABOVE,
    keep_below: bool = _core.KEEP_BELOW,
    keep_left: bool = _core.KEEP_LEFT,
    keep_right: bool = _core.KEEP_RIGHT,
    filter_margin_px: int = _core.FILTER_MARGIN_PX,
    glints_target: int = _core.GLINTS_TARGET,
    split_widest_for_target: bool = _core.SPLIT_WIDEST_FOR_TARGET,
    min_ellipse_fit_ratio: float | None = None,
    min_roundness_ratio: float | None = None,
) -> GlintResult:
    """Detect bright glint blobs near ``pupil_center``.

    Search region: a filled circle centred at ``pupil_center`` with radius
    ``search_radius_factor * pupil_radius``, optionally clamped from above
    by ``search_radius_max_px``. ``glint_roi=(x, y, w, h)`` runs the
    algorithm on the cropped sub-image and translates outputs back.

    Pipeline:

      1. Threshold the image at ``glint_threshold`` (bright pixels = 255).
      2. Intersect with the circular search region (when pupil is supplied).
      3. Find contours; optionally drop any whose area exceeds
         ``max_area_px``.
      4. Optionally drop contours whose centroid fails the half-plane
         filter described by ``keep_above`` / ``keep_below`` /
         ``keep_left`` / ``keep_right`` (default = all four True =
         no filter). ``filter_margin_px`` softens the boundary.
      5. Apply the opt-in shape-quality gates
         (``min_ellipse_fit_ratio`` / ``min_roundness_ratio``).
      6. If ``split_widest_for_target`` and exactly ``glints_target - 1``
         contours survive, split the widest in half horizontally.
      7. Keep the ``glints_target`` largest remaining contours by area.
      8. Compute each surviving glint's centre via
         ``glint_center_method``; the returned glints are ordered
         left-to-right by bounding-box x.

    Returns ``{"glints": [...], "search_area": np.ndarray}``. Each glint
    is ``{"contour", "center": (cx, cy), "ellipse" | None}``. The mask
    payload is the post-filter (threshold ∧ search) image — useful as a
    live overlay for tuning.
    """
    if img.ndim != 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.ascontiguousarray(img, dtype=np.uint8)
    height, width = img.shape

    if glint_roi is None:
        roi_x = roi_y = roi_w = roi_h = 0
    else:
        roi_x, roi_y, roi_w, roi_h = (int(v) for v in glint_roi)

    has_pupil = pupil_center is not None and pupil_radius is not None
    cx = float(pupil_center[0]) if has_pupil else 0.0
    cy = float(pupil_center[1]) if has_pupil else 0.0
    pr = float(pupil_radius) if has_pupil else 0.0

    result = _core.detect(
        img,
        roi_x,
        roi_y,
        roi_w,
        roi_h,
        1 if has_pupil else 0,
        cx,
        cy,
        pr,
        glint_threshold,
        float(search_radius_factor),
        -1 if search_radius_max_px is None else int(search_radius_max_px),
        _CENTER_METHOD_CODE[glint_center_method],
        -1 if max_area_px is None else int(max_area_px),
        1 if keep_above else 0,
        1 if keep_below else 0,
        1 if keep_left else 0,
        1 if keep_right else 0,
        int(filter_margin_px),
        int(glints_target),
        1 if split_widest_for_target else 0,
        -1.0 if min_ellipse_fit_ratio is None else float(min_ellipse_fit_ratio),
        -1.0 if min_roundness_ratio is None else float(min_roundness_ratio),
    )
    if result is None:
        return {"glints": [], "search_area": np.zeros((height, width), dtype=np.uint8)}
    glint_tuples, search_area = result

    glints = []
    for gcx, gcy, ellipse_t, contour_xy in glint_tuples:
        if ellipse_t is None:
            ellipse = None
        else:
            ecx, ecy, ew, eh, angle = ellipse_t
            ellipse = ((ecx, ecy), (ew, eh), angle)
        contour = contour_xy.astype(np.int32).reshape(-1, 1, 2)
        glints.append({"contour": contour, "center": (round(gcx), round(gcy)), "ellipse": ellipse})

    return {"glints": glints, "search_area": search_area}
