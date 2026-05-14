"""Pupil + glint detection on grayscale eye images.

Public surface re-exported here for convenience::

    from pupil_glint_detector import (
        crop_side,
        detect_glints,
        detect_limbus,
        detect_pupil_and_glints,
        fit_convex_hull_spline,
        plot_detections,
        pupil_center_of_mass,
    )

Pair this package with an alignment library such as
``eye-image-alignment`` to register two eye images using the detected
pupil + limbus geometry.
"""

from .core import (
    crop_side,
    detect_glints,
    detect_limbus,
    detect_pupil_and_glints,
    fit_convex_hull_spline,
    pupil_center_of_mass,
)
from .gui import tune_thresholds
from .plots import plot_detections

__all__ = [
    "crop_side",
    "detect_glints",
    "detect_limbus",
    "detect_pupil_and_glints",
    "fit_convex_hull_spline",
    "plot_detections",
    "pupil_center_of_mass",
    "tune_thresholds",
]
