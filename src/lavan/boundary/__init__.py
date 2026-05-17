"""Iris boundary detectors.

- ``IntegroDifferentialOperator`` (Daugman 1993 / 2004) — fits a perfect
  **circle** by maximising the Gaussian-smoothed derivative of the mean
  intensity around a circle, over (centre, radius). C kernel in
  ``integro_differential_operator_core.c``.
- ``DaugmanActiveContour`` (Daugman 2007) — Fourier-series active contour
  around a seed centre; fits a smooth **non-circular** boundary using
  isotropic radial search. C kernel in ``active_contour_core.c``.
- ``PupilGuidedContour`` — same Fourier-series framework but with per-angle
  anisotropic search bounds derived from the pupil ellipse shape. Used by
  the PSA pipeline to find the limbus on cluttered head-mounted images
  without locking onto eyelashes at high radii. C kernel in
  ``pupil_guided_contour_core.c``.
"""

from .daugman_active_contour import DaugmanActiveContour
from .integro_differential_operator import IntegroDifferentialOperator
from .pupil_guided_contour import PupilGuidedContour

__all__ = [
    "DaugmanActiveContour",
    "IntegroDifferentialOperator",
    "PupilGuidedContour",
]
