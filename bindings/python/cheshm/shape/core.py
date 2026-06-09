"""Pupil-shape helpers usable on a detector contour or a handful of manual points.

Both entry points operate on a set of boundary points, so the same code serves
the image detectors (dense contour) and the manual-annotation path (a few
clicks). The image-only ``center_of_mass`` estimator is intentionally not
exposed here, since it needs pixel intensities rather than geometry alone.
"""

import numpy as np

from . import _core

# Centre estimators that work on bare boundary points.
_CENTER_CODE = {
    "convex_hull_centroid": _core.CENTER_CONVEX_HULL_CENTROID,
    "ellipse_fit_center": _core.CENTER_ELLIPSE_FIT,
    "min_area_rect_center": _core.CENTER_MIN_AREA_RECT,
    "hull_moments_centroid": _core.CENTER_HULL_MOMENTS,
}
CENTER_METHODS = tuple(_CENTER_CODE)


def fit_pupil_form(
    points: np.ndarray,
    harmonics: int,
    samples: int = 360,
    iterations: int = 1,
    inward_rejection: float = 1.0,
) -> tuple[np.ndarray, tuple[float, float]] | None:
    """Polar-Fourier boundary fitted to ``points`` (needs N >= 2*harmonics + 2).

    ``iterations=1`` is a plain least-squares fit; higher values add the robust
    inward-rejection reweighting used to bridge glint/eyelash intrusions on a
    dense detector contour. Returns ``(boundary (M, 2) float64, (cx, cy))`` or
    ``None`` if the fit could not be solved.
    """
    pts = np.ascontiguousarray(points, dtype=np.float64)
    return _core.fit_pupil_form(pts, int(harmonics), int(samples), int(iterations), float(inward_rejection))


def pupil_center(points: np.ndarray, method: str) -> tuple[float, float] | None:
    """Pupil centre from boundary ``points`` by ``method`` (see ``CENTER_METHODS``).

    Returns ``(cx, cy)`` or ``None`` (e.g. fewer than 5 points).
    """
    if method not in _CENTER_CODE:
        raise ValueError(f"unknown center method {method!r}; choose from {CENTER_METHODS}")
    pts = np.ascontiguousarray(points, dtype=np.float64)
    return _core.pupil_center(pts, _CENTER_CODE[method])


def smoothing_spline(points: np.ndarray, smoothness: float, n_samples: int = 360) -> np.ndarray | None:
    """Periodic cubic smoothing spline through ``points`` (needs N >= 4).

    ``smoothness`` 0 passes through every point (interpolation); larger values
    trade point-closeness for lower curvature. ``smoothness`` is scale-invariant.
    Returns the sampled closed boundary as ``(n_samples, 2)`` float64, or ``None``.
    """
    pts = np.ascontiguousarray(points, dtype=np.float64)
    return _core.smoothing_spline(pts, float(smoothness), int(n_samples))
