"""PuReST pupil detector + tracker — Python wrapper.

Reference: Santini, T., Fuhl, W., Kasneci, E. (2018). "PuReST: Robust
pupil tracking for real-time pervasive eye tracking." *ETRA 2018*.

PuReST is stateful: the first call runs the full PuRe detection and
stores the result; subsequent calls reuse it as a seed for a local
greedy + outline search. If both tracking paths fail, the tracker
falls back to the full PuRe detection and re-seeds.

Typical use:

    tracker = PuReST()
    for frame in frames:
        result = tracker.detect(frame)
        ...

Use ``tracker.reset()`` to clear state between sequences.

The module also exposes a stateless ``detect_pupil(img, ...)`` helper
that creates a fresh tracker per call — it loses the temporal-tracking
benefit but matches the single-image-per-call contract the GUI and the
other detectors use.
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


class PuReST:
    """Stateful PuReST pupil tracker.

    Construct once, call :meth:`detect` per frame in a sequence. The
    tracker carries the previous detection across calls and uses it as
    a seed for a local search. Use :meth:`reset` to clear the state.
    """

    def __init__(
        self,
        min_pupil_diameter_mm: float = _core.MIN_PUPIL_DIAMETER_MM,
        max_pupil_diameter_mm: float = _core.MAX_PUPIL_DIAMETER_MM,
        canthi_distance_mm: float = _core.CANTHI_DISTANCE_MM,
        outline_bias: int = _core.OUTLINE_BIAS,
    ) -> None:
        self._tracker = _core.Tracker(
            float(min_pupil_diameter_mm),
            float(max_pupil_diameter_mm),
            float(canthi_distance_mm),
            int(outline_bias),
        )

    def detect(
        self,
        img: np.ndarray,
        pupil_roi: tuple[int, int, int, int] | None = None,
    ) -> PupilResult | None:
        """Run one detection on ``img`` using the tracker's accumulated state."""
        if img.ndim != 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.ascontiguousarray(img, dtype=np.uint8)

        if pupil_roi is None:
            roi_x = roi_y = roi_w = roi_h = 0
        else:
            roi_x, roi_y, roi_w, roi_h = (int(v) for v in pupil_roi)

        result = self._tracker.detect(img, roi_x, roi_y, roi_w, roi_h)
        if result is None:
            return None

        cx, cy, w, h, angle_deg, confidence = result
        return {
            "center": (round(cx), round(cy)),
            "ellipse": ((cx, cy), (w, h), angle_deg),
            "confidence": float(confidence),
        }

    def reset(self) -> None:
        """Clear the tracker's state. Use at the start of a new sequence."""
        self._tracker.reset()


def detect_pupil(
    img: np.ndarray,
    pupil_roi: tuple[int, int, int, int] | None = None,
    *,
    min_pupil_diameter_mm: float = _core.MIN_PUPIL_DIAMETER_MM,
    max_pupil_diameter_mm: float = _core.MAX_PUPIL_DIAMETER_MM,
    canthi_distance_mm: float = _core.CANTHI_DISTANCE_MM,
    outline_bias: int = _core.OUTLINE_BIAS,
) -> PupilResult | None:
    """Stateless single-frame shim for the GUI and other single-image callers.

    Creates a fresh :class:`PuReST` tracker per call — equivalent to a
    PuRe full detection (no tracking benefit). Use the :class:`PuReST`
    class directly to get the stateful tracker behaviour.
    """
    tracker = PuReST(
        min_pupil_diameter_mm=min_pupil_diameter_mm,
        max_pupil_diameter_mm=max_pupil_diameter_mm,
        canthi_distance_mm=canthi_distance_mm,
        outline_bias=outline_bias,
    )
    return tracker.detect(img, pupil_roi=pupil_roi)