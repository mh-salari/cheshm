"""PupilLabs2D pupil detector — Python wrapper.

Reference: Kassner, M., Patera, W., Bulling, A. (2014). "Pupil: an open
source platform for pervasive eye tracking and mobile gaze-based
interaction." *UbiComp 2014 Adjunct*, 1151-1160.

The Pupil Core 2D detector, vendored from
<https://github.com/pupil-labs/pupil-detectors> (LGPL-3.0). The
algorithm runs an optional Haar-based coarse-pupil seed, an intensity-
range threshold, Canny edges + Guo-Hall thinning, contour walking with
strong/final perimeter and area ratio gates, and ellipse fitting with
support-pixel scoring.
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
    "intensity_range": {
        "min": 1,
        "max": 80,
        "label": "Intensity range",
    },
    "blur_size": {
        "min": 1,
        "max": 21,
        "label": "Blur kernel size",
    },
    "canny_threshold": {
        "min": 1.0,
        "max": 400.0,
        "step": 1.0,
        "label": "Canny threshold",
    },
    "canny_ratio": {
        "min": 1.0,
        "max": 5.0,
        "step": 0.1,
        "label": "Canny ratio",
    },
    "canny_aperture": {
        "min": 3,
        "max": 7,
        "label": "Canny aperture",
    },
    "pupil_size_max": {
        "min": 10,
        "max": 400,
        "label": "Max pupil size (px)",
    },
    "pupil_size_min": {
        "min": 1,
        "max": 100,
        "label": "Min pupil size (px)",
    },
    "contour_size_min": {
        "min": 3,
        "max": 30,
        "label": "Min contour size",
    },
    "ellipse_roundness_ratio": {
        "min": 0.01,
        "max": 1.0,
        "step": 0.01,
        "label": "Ellipse roundness ratio",
    },
    "initial_ellipse_fit_threshold": {
        "min": 0.1,
        "max": 10.0,
        "step": 0.1,
        "label": "Initial fit threshold",
    },
    "ellipse_true_support_min_dist": {
        "min": 0.5,
        "max": 10.0,
        "step": 0.1,
        "label": "Support min dist",
    },
    "support_pixel_ratio_exponent": {
        "min": 0.5,
        "max": 5.0,
        "step": 0.1,
        "label": "Support exponent",
    },
    "coarse_detection": {
        "label": "Coarse detection",
    },
    "coarse_filter_min": {
        "min": 8,
        "max": 400,
        "label": "Coarse filter min",
    },
    "coarse_filter_max": {
        "min": 8,
        "max": 600,
        "label": "Coarse filter max",
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
    intensity_range: int = _core.INTENSITY_RANGE,
    blur_size: int = _core.BLUR_SIZE,
    canny_threshold: float = _core.CANNY_THRESHOLD,
    canny_ratio: float = _core.CANNY_RATIO,
    canny_aperture: int = _core.CANNY_APERTURE,
    pupil_size_max: int = _core.PUPIL_SIZE_MAX,
    pupil_size_min: int = _core.PUPIL_SIZE_MIN,
    contour_size_min: int = _core.CONTOUR_SIZE_MIN,
    ellipse_roundness_ratio: float = _core.ELLIPSE_ROUNDNESS_RATIO,
    initial_ellipse_fit_threshold: float = _core.INITIAL_ELLIPSE_FIT_THRESHOLD,
    ellipse_true_support_min_dist: float = _core.ELLIPSE_TRUE_SUPPORT_MIN_DIST,
    support_pixel_ratio_exponent: float = _core.SUPPORT_PIXEL_RATIO_EXPONENT,
    coarse_detection: bool = _core.COARSE_DETECTION,
    coarse_filter_min: int = _core.COARSE_FILTER_MIN,
    coarse_filter_max: int = _core.COARSE_FILTER_MAX,
) -> PupilResult | None:
    """Detect the pupil via Pupil Labs' 2D detector (Kassner et al. 2014).

    Returns ``{"center", "ellipse", "confidence"}`` or ``None`` when the
    detector produces no valid fit.

    ``pupil_roi=(x, y, w, h)`` constrains the search region. ``None`` =
    whole image.
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
        int(intensity_range),
        int(blur_size),
        float(canny_threshold),
        float(canny_ratio),
        int(canny_aperture),
        int(pupil_size_max),
        int(pupil_size_min),
        int(contour_size_min),
        float(ellipse_roundness_ratio),
        float(initial_ellipse_fit_threshold),
        float(ellipse_true_support_min_dist),
        float(support_pixel_ratio_exponent),
        bool(coarse_detection),
        int(coarse_filter_min),
        int(coarse_filter_max),
    )
    if result is None:
        return None

    cx, cy, w, h, angle_deg, confidence = result
    return {
        "center": (round(cx), round(cy)),
        "ellipse": ((cx, cy), (w, h), angle_deg),
        "confidence": float(confidence),
    }
