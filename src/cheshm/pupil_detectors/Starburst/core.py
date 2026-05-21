"""Starburst pupil detector — Python wrapper over the nanobind extension.

Reference: Li, D., Winfield, D., Parkhurst, D.J. (2005). "Starburst: A
hybrid algorithm for video-based eye tracking combining feature-based
and model-based approaches." *CVPR Workshops 2005*, vol. 3, 79-79.

The kernel runs Starburst rays from a seed centre, locates per-ray
intensity-jump edges, then RANSAC-fits an ellipse to the survivors. The
seed is either supplied by the caller or auto-derived from the image
centroid of pixels below a threshold (a rough dark-blob centre).
"""

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

    use_auto_seed = pupil_center is None
    seed_x = 0.0 if use_auto_seed else float(pupil_center[0])
    seed_y = 0.0 if use_auto_seed else float(pupil_center[1])

    if pupil_roi is None:
        roi_x = roi_y = roi_w = roi_h = 0
    else:
        roi_x, roi_y, roi_w, roi_h = (int(v) for v in pupil_roi)

    result = _core.detect(
        np.ascontiguousarray(img, dtype=np.uint8),
        roi_x,
        roi_y,
        roi_w,
        roi_h,
        use_auto_seed,
        int(seed_threshold),
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
