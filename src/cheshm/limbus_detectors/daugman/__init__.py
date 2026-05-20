"""Daugman-derived limbus boundary detectors.

- :class:`IntegroDifferentialOperator` — Daugman 1993 / 2004 operator.
- :class:`DaugmanActiveContour` — Daugman 2007 Fourier active contour.
- :class:`PupilGuidedContour` — pupil-shape-prior variant.
"""

from .active_contour import DaugmanActiveContour
from .integro_differential import IntegroDifferentialOperator
from .pupil_guided import PupilGuidedContour

__all__ = [
    "DaugmanActiveContour",
    "IntegroDifferentialOperator",
    "PupilGuidedContour",
]
