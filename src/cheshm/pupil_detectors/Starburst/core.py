"""Starburst pupil detector — Python wrapper over the nanobind extension.

Reference: Li, D., Winfield, D., Parkhurst, D.J. (2005). "Starburst: A
hybrid algorithm for video-based eye tracking combining feature-based
and model-based approaches." *CVPR Workshops 2005*, vol. 3, 79-79.

The kernel runs Starburst rays from a seed centre, locates per-ray
intensity-jump edges, then RANSAC-fits an ellipse to the survivors. The
seed is either supplied by the caller or auto-derived from the image
centroid of pixels below a threshold (a rough dark-blob centre).
"""

import cv2
import numpy as np

from cheshm._protocols import PupilResult

from . import _core

# GUI metadata. Defaults / types come from `detect_pupil`'s signature.
_UI = {
    "pupil_roi": {
        "widget": "roi",
        "label": "Pupil ROI",
        "help": "Optional (x, y, w, h) rectangle. None = whole image.",
        "hidden": True,
    },
    "edge_threshold": {
        "min": 5,
        "max": 100,
        "help": "Intensity jump along a ray that counts as an edge. Lower = catches softer edges, more noise.",
    },
    "rays": {
        "min": 4,
        "max": 360,
        "help": "Number of starburst rays shot from the seed.",
    },
    "min_feature_candidates": {
        "min": 5,
        "max": 200,
        "help": "Minimum edge-point count before RANSAC kicks in.",
    },
    "corneal_reflection_window": {
        "min": 0,
        "max": 999,
        "label": "CR search window (px)",
        "help": "Side of the corneal-reflection search window around the seed (odd integer). 0 disables CR removal.",
    },
    "corneal_reflection_ratio": {
        "min": 0,
        "max": 20,
        "label": "CR max-radius ratio",
        "help": "Largest accepted CR radius = image_height / this value. 0 disables CR removal.",
    },
    "max_edge_points": {
        "min": 16,
        "max": 4096,
        "help": "Cap on edge points returned to Python (algorithm internal vector size).",
    },
    "seed_threshold": {
        "min": 0,
        "max": 255,
        "help": "When pupil_center is None, the seed is the centroid of pixels below this intensity.",
    },
}

_OVERLAYS = (
    ("contour", "line"),
    ("ellipse", "line"),
    ("center", "point"),
)


def _touches_border(contour: np.ndarray, shape: tuple[int, ...]) -> bool:
    h, w = shape[:2]
    x, y, cw, ch = cv2.boundingRect(contour)
    return x == 0 or y == 0 or x + cw == w or y + ch == h


def _auto_seed(img: np.ndarray, seed_threshold: int) -> tuple[float, float]:
    """Initial pupil-seed centre: centroid of the largest interior dark blob.

    Thresholds the image at ``seed_threshold``, drops contours that
    touch the image border (eyelashes and frame vignette), and takes
    the largest remaining contour's moments centroid. Falls back to
    the image centre when no candidate is found.
    """
    h, w = img.shape
    _, mask = cv2.threshold(img, seed_threshold, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    interior = [c for c in contours if cv2.contourArea(c) > 0 and not _touches_border(c, img.shape)]
    if not interior:
        return (w / 2.0, h / 2.0)
    largest = max(interior, key=cv2.contourArea)
    m = cv2.moments(largest)
    if m["m00"] <= 0:
        return (w / 2.0, h / 2.0)
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


def detect_pupil(
    img: np.ndarray,
    pupil_center: tuple[float, float] | None = None,
    pupil_roi: tuple[int, int, int, int] | None = None,
    *,
    edge_threshold: int = _core.EDGE_THRESHOLD,
    rays: int = _core.RAYS,
    min_feature_candidates: int = _core.MIN_FEATURE_CANDIDATES,
    corneal_reflection_window: int = _core.CR_WINDOW_SIZE,
    corneal_reflection_ratio: int = _core.CR_RATIO_TO_IMAGE_HEIGHT,
    max_edge_points: int = _core.MAX_EDGE_POINTS,
    seed_threshold: int = _core.SEED_THRESHOLD,
) -> PupilResult | None:
    """Detect the pupil ellipse via Starburst (Li et al. 2005).

    Returns ``{"contour", "ellipse", "center"}`` matching the rest of
    cheshm's pupil detectors, or ``None`` when no ellipse could be fit:

      - ``contour`` — ``(N, 1, 2)`` int32 array of edge points produced
        by the ray search (cv2-contour-shaped).
      - ``ellipse`` — ``((cx, cy), (w, h), angle_deg)`` from the RANSAC
        fit.
      - ``center`` — ``(cx, cy)`` rounded ints.

    ``pupil_center`` is optional. When omitted, a seed is auto-derived
    by thresholding the image at ``seed_threshold`` and taking the
    centroid of the dark region. ``pupil_roi=(x, y, w, h)`` runs the
    algorithm on the cropped sub-image; the crop and all coordinate
    translations happen in the C++ kernel so the caller always sees
    full-image coordinates. Set ``corneal_reflection_window=0`` to skip
    the corneal-reflection pre-pass (faster, less robust on bright
    glints).
    """
    if img.ndim != 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.ascontiguousarray(img, dtype=np.uint8)

    if pupil_center is None:
        seed_x, seed_y = _auto_seed(img, seed_threshold)
    else:
        seed_x, seed_y = float(pupil_center[0]), float(pupil_center[1])

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
        seed_x,
        seed_y,
        edge_threshold,
        rays,
        min_feature_candidates,
        corneal_reflection_window,
        corneal_reflection_ratio,
        max_edge_points,
    )
    if result is None:
        return None
    (a, b, cx, cy, theta_rad), edge_xy = result
    angle_deg = float(np.degrees(theta_rad))
    contour = edge_xy.astype(np.int32).reshape(-1, 1, 2)
    return {
        "contour": contour,
        "center": (round(cx), round(cy)),
        "ellipse": ((cx, cy), (2 * a, 2 * b), angle_deg),
    }
