"""Pupil + glint + limbus detection on grayscale eye images.

Public surface re-exported here for convenience::

    from lavan.detect import (
        detect_glints,
        detect_limbus,
        detect_pupil,
        fit_convex_hull_spline,
        pupil_center_of_mass,
    )

``detect_pupil`` and ``detect_glints`` are independent entry points
that callers compose explicitly — the glint detector takes the pupil
centre + radius as inputs. There is intentionally no convenience
wrapper that chains the two so the cost of each pass is visible at
the call site.

``detect_limbus`` wraps :mod:`lavan.boundary.integro_differential_operator`;
once a pupil + limbus pair is in hand, :mod:`lavan.align` registers a
target eye image onto a reference by their iris texture.
"""

from .glint import detect_glints
from .limbus import detect_limbus
from .pupil import detect_pupil, fit_convex_hull_spline, pupil_center_of_mass

__all__ = [
    "detect_glints",
    "detect_limbus",
    "detect_pupil",
    "fit_convex_hull_spline",
    "pupil_center_of_mass",
]
