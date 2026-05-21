"""ExCuSe pupil detector — Python wrapper over the nanobind extension.

Reference: Fuhl, W., Kübler, T., Sippel, K., Rosenstiel, W., Kasneci, E.
(2015). "ExCuSe: Robust Pupil Detection in Real-World Scenarios."
*CAIP 2015*, 39-51.

The kernel runs an adaptive-threshold-driven angular histogram to pick a
coarse pupil seed, custom Canny edge detection, ray-based contour
collection, and ellipse fitting with quality validation.
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
    "max_ellipse_radi": {
        "min": 5,
        "max": 1024,
        "label": "Max ellipse radius",
        "help": "Upper bound on accepted ellipse semi-axis length (pixels).",
    },
    "good_ellipse_threshold": {
        "min": 1,
        "max": 200,
        "label": "Good-ellipse threshold",
        "help": "Pixel-count threshold for the goodness test on the candidate ellipse.",
    },
}

_OVERLAYS = (
    ("ellipse", "line"),
    ("center", "point"),
)


def detect_pupil(
    img: np.ndarray,
    pupil_roi: tuple[int, int, int, int] | None = None,
    *,
    max_ellipse_radi: int = 50,
    good_ellipse_threshold: int = 15,
) -> PupilResult | None:
    """Detect the pupil ellipse via ExCuSe (Fuhl et al. 2015).

    Returns ``{"ellipse", "center"}`` matching the rest of cheshm's pupil
    detectors, or ``None`` when no ellipse could be produced:

      - ``ellipse`` — ``((cx, cy), (w, h), angle_deg)`` from the fit.
      - ``center`` — ``(cx, cy)`` rounded ints.

    ``pupil_roi=(x, y, w, h)`` runs the algorithm on the cropped
    sub-image and translates outputs back to full-image coordinates.
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
        max_ellipse_radi,
        good_ellipse_threshold,
    )
    if result is None:
        return None
    cx, cy, w, h, angle_deg = result
    return {
        "center": (round(cx), round(cy)),
        "ellipse": ((cx, cy), (w, h), angle_deg),
    }
