"""Pupil-shape-guided active contour for iris boundary localization.

Extends the Daugman 2007 radial-gradient framework with anisotropic
per-angle search bounds derived from the pupil ellipse. The pupil's
position is intentionally not used — the search is centred at the
externally supplied seed (typically the integro-differential operator
output), so pupil/limbus decentration is preserved as signal.
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
    "N": {"min": 8, "max": 1440, "label": "Angular samples (N)",
          "help": "Number of angles θ around the seed (paper notation: N)."},
    "M": {"min": 1, "max": 32, "label": "Fourier harmonics (M)",
          "help": "Number of Fourier coefficients kept. Default 3 keeps the mean + ellipse harmonic."},
    "gradient_sigma": {"min": 0.0, "max": 10.0,
                       "help": "Gaussian σ (px) for pre-gradient blur."},
    "radial_smoothing": {"min": 0.0, "max": 10.0,
                         "help": "Gaussian σ (radial samples) applied to the radial-gradient profile."},
    "k_min": {"min": 0.1, "max": 20.0,
              "help": "Lower radius factor: search starts at k_min · pupil_radius(θ)."},
    "k_max": {"min": 0.1, "max": 20.0,
              "help": "Upper radius factor: search ends at k_max · pupil_radius(θ)."},
}

DEFAULT_N = _core.N
DEFAULT_M = _core.M
DEFAULT_GRADIENT_SIGMA = _core.GRADIENT_SIGMA
DEFAULT_RADIAL_SMOOTHING = _core.RADIAL_SMOOTHING
DEFAULT_K_MIN = _core.K_MIN
DEFAULT_K_MAX = _core.K_MAX


def detect_limbus(
    img: np.ndarray,
    seed_center: tuple[float, float],
    pupil_ellipse: tuple[tuple[float, float], tuple[float, float], float],
    *,
    N: int = DEFAULT_N,
    M: int = DEFAULT_M,
    gradient_sigma: float = DEFAULT_GRADIENT_SIGMA,
    radial_smoothing: float = DEFAULT_RADIAL_SMOOTHING,
    k_min: float = DEFAULT_K_MIN,
    k_max: float = DEFAULT_K_MAX,
) -> LimbusResult | None:
    """One-shot pupil-shape-prior active-contour limbus fit.

    ``pupil_ellipse`` is the ``cv2.fitEllipse``-style tuple
    ``((cx_p, cy_p), (2a_p, 2b_p), angle_deg)``. Returns
    ``{"center": (cx, cy), "thetas": np.ndarray, "R_theta": np.ndarray}``
    or ``None`` if too few angular samples produced a peak.
    """
    (pcx, pcy), (pw, ph), p_angle_deg = pupil_ellipse
    result = _core.detect_limbus(
        np.ascontiguousarray(img, dtype=np.uint8),
        float(seed_center[0]),
        float(seed_center[1]),
        float(pcx),
        float(pcy),
        float(pw),
        float(ph),
        float(p_angle_deg),
        int(N),
        int(M),
        float(gradient_sigma),
        float(radial_smoothing),
        float(k_min),
        float(k_max),
    )
    if result is None:
        return None
    cx, cy, thetas, R_theta = result
    return {"center": (float(cx), float(cy)), "thetas": thetas, "R_theta": R_theta}
