"""Pupil detectors.

- :mod:`.threshold` — threshold-based detector (``detect_pupil``).
- :mod:`.centers` — pupil-center helpers (``pupil_center_of_mass``,
``fit_convex_hull_spline``).
"""

from .centers import fit_convex_hull_spline, pupil_center_of_mass
from .threshold import detect_pupil

__all__ = [
    "detect_pupil",
    "fit_convex_hull_spline",
    "pupil_center_of_mass",
]
