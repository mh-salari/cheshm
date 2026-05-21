"""Daugman's integro-differential operator for iris boundary localization.

Reference: Daugman, J. (2004). "How Iris Recognition Works." IEEE Trans.
Circuits and Systems for Video Technology, 14(1), 21-30, eq. (1). The same
operator was introduced in Daugman, J. (1993). "High Confidence Visual
Recognition of Persons by a Test of Statistical Independence." IEEE Trans.
PAMI, 15(11), 1148-1161.
"""

import numpy as np

from cheshm._protocols import LimbusResult

from . import _core

_OVERLAYS = (
    ("curve", "line"),
    ("center", "point"),
    ("mask", "fill"),
)

_UI = {
    "r_min": {
        "min": 1,
        "max": 1024,
        "help": "Lower bound on candidate iris radius (pixels).",
    },
    "r_max": {
        "min": 1,
        "max": 1024,
        "help": "Upper bound on candidate iris radius (pixels).",
    },
    "range_": {
        "min": 0,
        "max": 200,
        "label": "Search range (px)",
        "help": "Half-width of the centre-sweep grid around the seed (±range, in pixels).",
    },
    "step": {
        "min": 1,
        "max": 20,
        "label": "Search step (px)",
        "help": "Grid step for the centre sweep (pixels). Smaller = finer + slower.",
    },
}

DEFAULT_R_MIN = _core.R_MIN
DEFAULT_R_MAX = _core.R_MAX
DEFAULT_RANGE = _core.RANGE
DEFAULT_STEP = _core.STEP


def detect_limbus(
    img: np.ndarray,
    seed_center: tuple[float, float],
    *,
    r_min: int = DEFAULT_R_MIN,
    r_max: int = DEFAULT_R_MAX,
    range_: int = DEFAULT_RANGE,
    step: int = DEFAULT_STEP,
) -> LimbusResult | None:
    """One-shot integro-differential limbus localization around ``seed_center``.

    Runs a single grid search of ``(±range_, step)`` around ``seed_center``,
    scoring each candidate centre by the Gaussian-smoothed derivative of the
    mean circle intensity. Returns ``{"center": (cx, cy), "radius": r}`` or
    ``None`` if the search produced no candidate.
    """
    result = _core.detect_limbus(
        np.ascontiguousarray(img, dtype=np.uint8),
        float(seed_center[0]),
        float(seed_center[1]),
        int(r_min),
        int(r_max),
        int(range_),
        int(step),
    )
    if result is None:
        return None
    cx, cy, radius = result
    return {"center": (int(cx), int(cy)), "radius": int(radius)}
