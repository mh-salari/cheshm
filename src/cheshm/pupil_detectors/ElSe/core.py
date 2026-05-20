"""ElSe pupil detector — Python wrapper over the nanobind extension.

Reference: Fuhl, W., Santini, T., Kübler, T., Kasneci, E. (2016).
"ElSe: Ellipse Selection for Robust Pupil Detection in Real-World
Environments." *ETRA 2016*, vol. 14, 123-130.

The kernel runs a custom Canny edge detector, filters spurs and
over-connected pixels, fits ellipses to surviving contour curves,
gates them by an inner / outer intensity-ratio test, and picks the
best candidate by an inner-darkness score. If no ellipse passes the
gate, a morphological-blob fallback returns a coarse pupil position
only.
"""

import cv2
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
    "min_area_ratio": {
        "min": 0.0001,
        "max": 0.5,
        "step": 0.0005,
        "label": "Min pupil area ratio",
        "help": "Minimum pupil area as a fraction of the working-frame area.",
    },
    "max_area_ratio": {
        "min": 0.001,
        "max": 0.9,
        "step": 0.005,
        "label": "Max pupil area ratio",
        "help": "Maximum pupil area as a fraction of the working-frame area.",
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
    min_area_ratio: float = _core.MIN_AREA_RATIO,
    max_area_ratio: float = _core.MAX_AREA_RATIO,
) -> PupilResult | None:
    """Detect the pupil via ElSe (Fuhl et al. 2016).

    Returns a dict that always carries ``method`` and ``center``, plus
    ``ellipse`` only on the primary path:

      - ``method`` — ``"ellipse"`` when the edge-based ellipse fit passed
        the intensity-ratio gate, ``"blob_fallback"`` when the
        morphological-blob fallback fired instead.
      - ``center`` — ``(cx, cy)`` rounded ints, in full-image coordinates.
      - ``ellipse`` — ``((cx, cy), (w, h), angle_deg)`` from the fit,
        only present when ``method == "ellipse"``.

    Returns ``None`` if both detection paths fail.

    ``pupil_roi=(x, y, w, h)`` runs the algorithm on the cropped
    sub-image and translates outputs back to full-image coordinates.
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
        float(min_area_ratio),
        float(max_area_ratio),
    )
    if result is None:
        return None

    method = result[0]
    if method == "ellipse":
        _, cx, cy, w, h, angle_deg = result
        return {
            "method": "ellipse",
            "center": (round(cx), round(cy)),
            "ellipse": ((cx, cy), (w, h), angle_deg),
        }

    _, cx, cy = result
    return {
        "method": "blob_fallback",
        "center": (round(cx), round(cy)),
    }
