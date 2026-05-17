"""Limbus detection on grayscale eye images.

Public surface:

  - :func:`detect_limbus` — Daugman integro-differential operator limbus
    circle. Thin wrapper around
    :class:`lavan.boundary.IntegroDifferentialOperator` that translates a
    pupil-centre + pupil-radius seed into the centre / radius search
    ranges the operator expects.

The Daugman sweep is the slowest detector in :mod:`lavan.detect` (orders
of magnitude over :func:`~lavan.detect.detect_pupil` and
:func:`~lavan.detect.detect_glints`), so callers should treat it as a
deliberate, user-triggered step rather than a slider-rate live pass.
"""

import numpy as np

from lavan.boundary import IntegroDifferentialOperator


def detect_limbus(
    img: np.ndarray,
    pupil_center: tuple[float, float],
    pupil_radius: float,
    *,
    r_min_factor: float = 1.5,
    r_max_factor: float = 5.0,
    search_window_px: int = 15,
) -> dict | None:
    """Daugman integro-differential operator limbus circle.

    Returns ``{"center": (lx, ly), "radius": r}`` on success, or ``None``
    when the operator finds no iris candidates.

    The centre search is seeded at the pupil centre and swept over a
    ``±search_window_px`` window. The iris radius is searched between
    ``r_min_factor`` and ``r_max_factor`` times the pupil radius.
    """
    pcx, pcy = pupil_center
    r_min = max(round(pupil_radius * r_min_factor), 1)
    r_max = max(round(pupil_radius * r_max_factor), r_min + 1)
    op = IntegroDifferentialOperator(img, r_min=r_min, r_max=r_max)
    results = op.search(cen_x=round(pcy), cen_y=round(pcx), range_=int(search_window_px), step=1)
    if len(results) == 0:
        return None
    ly, lx, _score, lr = results[-1]
    return {"center": (float(lx), float(ly)), "radius": float(lr)}
