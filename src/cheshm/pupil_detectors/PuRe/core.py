"""PuRe pupil detector — Python wrapper over the nanobind extension.

Reference: Santini, T., Fuhl, W., Kasneci, E. (2018). "PuRe: Robust
pupil detection for real-time pervasive eye tracking." *Computer
Vision and Image Understanding*, 170, 40-50.

The kernel downscales the frame to a 320x240 working size, runs a
custom Canny edge detector, filters spurs and over-connected edges,
extracts contour curves, validates each as a pupil candidate (size,
curvature, ellipse fit, anchor distribution, outline contrast), tries
pair-wise candidate combinations, then picks the highest-scoring
candidate. Returns the ellipse plus a confidence score from the
outline-contrast vote ratio.
"""

import numpy as np

from cheshm._protocols import PupilResult

from . import _core

_UI = {
    "pupil_roi": {
        "widget": "roi",
        "label": "Pupil ROI",
        "help": "Optional (x, y, w, h) rectangle. None = whole image.",
        "hidden": True,
    },
    "min_pupil_diameter_mm": {
        "min": 0.5,
        "max": 10.0,
        "step": 0.1,
        "label": "Min pupil diameter (mm)",
        "help": "Minimum expected physical pupil diameter.",
    },
    "max_pupil_diameter_mm": {
        "min": 1.0,
        "max": 20.0,
        "step": 0.1,
        "label": "Max pupil diameter (mm)",
        "help": "Maximum expected physical pupil diameter.",
    },
    "canthi_distance_mm": {
        "min": 15.0,
        "max": 50.0,
        "step": 0.1,
        "label": "Canthi distance (mm)",
        "help": "Mean palpebral fissure width used to convert mm to pixels.",
    },
    "outline_bias": {
        "min": 0,
        "max": 30,
        "label": "Outline bias",
        "help": "Intensity gap (uchar) gating outline-contrast votes.",
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
    min_pupil_diameter_mm: float = _core.MIN_PUPIL_DIAMETER_MM,
    max_pupil_diameter_mm: float = _core.MAX_PUPIL_DIAMETER_MM,
    canthi_distance_mm: float = _core.CANTHI_DISTANCE_MM,
    outline_bias: int = _core.OUTLINE_BIAS,
) -> PupilResult | None:
    """Detect the pupil via PuRe (Santini et al. 2018).

    Returns ``{"center", "ellipse", "confidence"}`` or ``None`` when no
    candidate passes the validity gates:

      - ``center`` — ``(cx, cy)`` rounded ints.
      - ``ellipse`` — ``((cx, cy), (w, h), angle_deg)`` from the fit.
      - ``confidence`` — outline-contrast vote ratio in ``[0, 1]``.

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
        float(min_pupil_diameter_mm),
        float(max_pupil_diameter_mm),
        float(canthi_distance_mm),
        int(outline_bias),
    )
    if result is None:
        return None

    cx, cy, w, h, angle_deg, confidence = result
    return {
        "center": (round(cx), round(cy)),
        "ellipse": ((cx, cy), (w, h), angle_deg),
        "confidence": float(confidence),
    }
