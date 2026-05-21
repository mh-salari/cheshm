"""Daugman 2007 Fourier-series active contour for iris boundary localization.

Reference: Daugman, J. (2007). "New Methods in Iris Recognition." IEEE Trans.
Systems, Man, and Cybernetics, Part B, 37(5), 1167-1175. Section II
("Active Contours and Generalized Coordinates"), equations (1) and (2).
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
    "N": {
        "min": 8,
        "max": 1440,
        "label": "Angular samples (N)",
        "help": "Number of angles θ around the seed (paper notation: N).",
    },
    "M": {
        "min": 1,
        "max": 32,
        "label": "Fourier harmonics (M)",
        "help": "Number of Fourier coefficients kept. Daugman 2007 recommends 5 for iris.",
    },
    "gradient_sigma": {"min": 0.0, "max": 10.0, "help": "Gaussian σ (px) for pre-gradient blur."},
    "radial_smoothing": {
        "min": 0.0,
        "max": 10.0,
        "help": "Gaussian σ (radial samples) applied to the radial-gradient profile.",
    },
    "skip_eyelid_wedges": {"label": "Skip eyelid wedges", "help": "Mask out the upper-lid angular wedge (240°–300°)."},
    "r_min": {"min": 1.0, "max": 1024.0, "help": "Lower bound on iris radius (pixels)."},
    "r_max": {"min": 1.0, "max": 1024.0, "help": "Upper bound on iris radius (pixels)."},
}

DEFAULT_N = _core.N
DEFAULT_M = _core.M
DEFAULT_GRADIENT_SIGMA = _core.GRADIENT_SIGMA
DEFAULT_RADIAL_SMOOTHING = _core.RADIAL_SMOOTHING
DEFAULT_SKIP_EYELID_WEDGES = _core.SKIP_EYELID_WEDGES
DEFAULT_R_MIN = _core.R_MIN
DEFAULT_R_MAX = _core.R_MAX


def detect_limbus(
    img: np.ndarray,
    seed_center: tuple[float, float],
    *,
    N: int = DEFAULT_N,
    M: int = DEFAULT_M,
    gradient_sigma: float = DEFAULT_GRADIENT_SIGMA,
    radial_smoothing: float = DEFAULT_RADIAL_SMOOTHING,
    skip_eyelid_wedges: bool = DEFAULT_SKIP_EYELID_WEDGES,
    r_min: float = DEFAULT_R_MIN,
    r_max: float = DEFAULT_R_MAX,
) -> LimbusResult | None:
    """One-shot Daugman 2007 active-contour limbus fit around ``seed_center``.

    Returns ``{"center": (cx, cy), "thetas": np.ndarray, "R_theta": np.ndarray}``,
    or ``None`` if too few angular samples produced a radial-gradient peak.
    Boundary points are at ``(cx + R_theta cos θ, cy + R_theta sin θ)``.
    """
    result = _core.detect_limbus(
        np.ascontiguousarray(img, dtype=np.uint8),
        float(seed_center[0]),
        float(seed_center[1]),
        int(N),
        int(M),
        float(gradient_sigma),
        float(radial_smoothing),
        bool(skip_eyelid_wedges),
        float(r_min),
        float(r_max),
    )
    if result is None:
        return None
    cx, cy, thetas, R_theta = result
    return {"center": (float(cx), float(cy)), "thetas": thetas, "R_theta": R_theta}
