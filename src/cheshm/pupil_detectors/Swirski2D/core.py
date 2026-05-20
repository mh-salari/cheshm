"""Swirski 2D pupil detector — Python wrapper over the nanobind extension.

Reference: Swirski, L., Bulling, A., Dodgson, N. (2012). "Robust
real-time pupil tracking in highly off-axis images." *ETRA 2012*.
<https://doi.org/10.1145/2168556.2168585>

The kernel runs a Haar surround feature to localise the pupil, k-means
segments the resulting ROI's histogram into pupil/iris, Canny detects
edges, starburst rays from three seed centres collect candidate edge
points, and a RANSAC ellipse fit with image-aware support produces the
final pupil ellipse.
"""

import cv2
import numpy as np

from . import _core

# GUI metadata. Defaults / types come from `detect_pupil`'s signature.
_UI = {
    "pupil_roi": {
        "widget": "roi",
        "label": "Pupil ROI",
        "help": "Optional (x, y, w, h) rectangle. None = whole image.",
        "hidden": True,
    },
    "radius_min": {
        "min": 1,
        "max": 1024,
        "help": "Lower bound on pupil radius (pixels). Used by the Haar feature sweep.",
    },
    "radius_max": {
        "min": 1,
        "max": 1024,
        "help": "Upper bound on pupil radius (pixels).",
    },
    "canny_blur": {
        "min": 0.0,
        "max": 10.0,
        "help": "Gaussian σ (px) before Canny edge detection inside the pupil ROI.",
    },
    "canny_threshold_1": {
        "min": 0.0,
        "max": 500.0,
        "help": "Canny hysteresis lower threshold.",
    },
    "canny_threshold_2": {
        "min": 0.0,
        "max": 500.0,
        "help": "Canny hysteresis upper threshold.",
    },
    "starburst_points": {
        "min": 4,
        "max": 360,
        "help": "Number of rays cast from each of the three seed centres.",
    },
    "percentage_inliers": {
        "min": 1,
        "max": 100,
        "label": "Percentage inliers",
        "help": "Target inlier fraction (percent of edge points). Drives the RANSAC sample size.",
    },
    "inlier_iterations": {
        "min": 1,
        "max": 20,
        "help": "How many inlier-refit cycles per RANSAC hypothesis.",
    },
    "image_aware_support": {
        "help": "Use image-gradient strength as the goodness score (otherwise inlier count).",
    },
    "early_termination_percentage": {
        "min": 0,
        "max": 100,
        "help": "Stop RANSAC once inliers exceed this percentage of edge points. 0 disables.",
    },
    "early_rejection": {
        "help": "Reject a hypothesis whose sampled edge gradients point the wrong way.",
    },
    "seed": {
        "min": -1,
        "max": 1 << 30,
        "help": "RANSAC random seed. -1 uses an internal counter (less reproducible).",
    },
    "max_inliers": {
        "min": 16,
        "max": 8192,
        "help": "Cap on inlier points returned to Python (algorithmic upper bound).",
    },
}

_OVERLAYS = (
    ("contour", "line"),
    ("ellipse", "line"),
    ("center", "point"),
)


def detect_pupil(
    img: np.ndarray,
    pupil_roi: tuple[int, int, int, int] | None = None,
    *,
    radius_min: int = 20,
    radius_max: int = 80,
    canny_blur: float = 1.6,
    canny_threshold_1: float = 30.0,
    canny_threshold_2: float = 50.0,
    starburst_points: int = 30,
    percentage_inliers: int = 30,
    inlier_iterations: int = 2,
    image_aware_support: bool = True,
    early_termination_percentage: int = 95,
    early_rejection: bool = True,
    seed: int = 0,
    max_inliers: int = 1024,
) -> dict | None:
    """Detect the pupil ellipse via Swirski et al. 2012.

    Returns ``{"contour", "ellipse", "center"}`` matching the rest of
    cheshm's pupil detectors, or ``None`` when no ellipse could be fit:

      - ``contour`` — ``(N, 1, 2)`` int32 array of RANSAC inlier points
        (cv2-contour-shaped).
      - ``ellipse`` — ``((cx, cy), (w, h), angle_deg)`` from the fit.
      - ``center`` — ``(cx, cy)`` rounded ints.

    No seed point is needed — the Haar feature step localises the
    pupil automatically before the RANSAC fit kicks in.
    ``pupil_roi=(x, y, w, h)`` runs the algorithm on the cropped
    sub-image; the crop and all coordinate translations happen in the
    C++ kernel so the caller always sees full-image coordinates.
    """
    if img.ndim != 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.ascontiguousarray(img, dtype=np.uint8)

    if pupil_roi is None:
        roi_x = roi_y = roi_w = roi_h = 0
    else:
        roi_x, roi_y, roi_w, roi_h = (int(v) for v in pupil_roi)

    result = _core.detect(
        img,
        roi_x, roi_y, roi_w, roi_h,
        radius_min, radius_max,
        canny_blur, canny_threshold_1, canny_threshold_2,
        starburst_points,
        percentage_inliers, inlier_iterations,
        int(image_aware_support),
        early_termination_percentage, int(early_rejection),
        seed,
        max_inliers,
    )
    if result is None:
        return None
    (cx, cy, w, h, angle_deg), inliers_xy = result
    contour = inliers_xy.astype(np.int32).reshape(-1, 1, 2)
    return {
        "contour": contour,
        "center": (round(cx), round(cy)),
        "ellipse": ((cx, cy), (w, h), angle_deg),
    }
