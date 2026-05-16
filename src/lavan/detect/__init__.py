"""Pupil + glint detection on grayscale eye images.

Public surface re-exported here for convenience::

    from pupil_glint_detector import (
        crop_side,
        detect_glints,
        detect_limbus,
        detect_pupil,
        fit_convex_hull_spline,
        pupil_center_of_mass,
    )

Pair this package with an alignment library such as
``eye-image-alignment`` to register two eye images using the detected
pupil + limbus geometry.

``detect_pupil`` and ``detect_glints`` are independent entry points
that callers compose explicitly — the glint detector takes the pupil
centre + radius as inputs. There is intentionally no convenience
wrapper that chains the two so the cost of each pass is visible at
the call site.

The ``cli`` and ``gui`` submodules are not re-exported from the
package root because both call into the old monolithic API and will
be rewritten against ``detect_pupil`` / ``detect_glints`` before the
next release.
"""

from .core import (
    crop_side,
    detect_glints,
    detect_limbus,
    detect_pupil,
    fit_convex_hull_spline,
    pupil_center_of_mass,
)

__all__ = [
    "crop_side",
    "detect_glints",
    "detect_limbus",
    "detect_pupil",
    "fit_convex_hull_spline",
    "pupil_center_of_mass",
]
