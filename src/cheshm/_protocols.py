"""Public protocols and TypedDicts for cheshm detector callables.

A plugin package or downstream consumer can rely on these types to:

  - Declare a detector function's contract (``PupilDetector``,
    ``GlintDetector``, ``LimbusDetector``).
  - Get editor autocomplete on detector return values
    (``PupilResult``, ``GlintResult``, ``LimbusResult``).

Each ``*Result`` lists the keys every in-category detector returns,
plus ``NotRequired`` keys some detectors add (e.g. ``contour`` for
pupil detectors that produce one, ``mask`` for those that produce one).
Concrete detectors are free to add further extra keys; ``TypedDict``
permits extras by default.
"""

from __future__ import annotations

from typing import NotRequired, Protocol, TypedDict

import numpy as np

Ellipse = tuple[tuple[float, float], tuple[float, float], float]
"""``((cx, cy), (w, h), angle_deg)`` — ``cv::RotatedRect`` convention."""


class PupilResult(TypedDict):
    """One-shot pupil detection result.

    ``center`` is the only key every detector sets. ``ellipse`` is
    present whenever the detector produced a fit. ``method`` is set by
    detectors with more than one detection path (e.g. ElSe distinguishes
    its primary ellipse path from a blob-position fallback); other
    detectors omit it.
    """

    center: tuple[int, int]
    ellipse: NotRequired[Ellipse]
    method: NotRequired[str]
    contour: NotRequired[np.ndarray]
    mask: NotRequired[np.ndarray]


class Glint(TypedDict):
    """One glint inside a ``GlintResult``."""

    center: tuple[int, int]
    contour: np.ndarray
    ellipse: Ellipse | None


class GlintResult(TypedDict):
    """One-shot glint detection result."""

    glints: list[Glint]
    search_area: np.ndarray


class LimbusResult(TypedDict):
    """One-shot limbus detection result.

    ``radius`` is set by circular-fit detectors (Daugman integro-
    differential operator); ``thetas`` / ``R_theta`` are set by
    angular-Fourier detectors (Daugman 2007 active contour and
    pupil-guided active contour).
    """

    center: tuple[int, int]
    radius: NotRequired[int]
    thetas: NotRequired[np.ndarray]
    R_theta: NotRequired[np.ndarray]


class PupilDetector(Protocol):
    """A callable that detects the pupil in a grayscale image.

    Detector-specific keyword arguments are not part of the protocol —
    consult each implementation's docstring for tunables.
    """

    def __call__(self, img: np.ndarray, /, *args: object, **kwargs: object) -> PupilResult | None: ...


class GlintDetector(Protocol):
    """A callable that detects glints in a grayscale image."""

    def __call__(self, img: np.ndarray, /, *args: object, **kwargs: object) -> GlintResult: ...


class LimbusDetector(Protocol):
    """A callable that detects the limbus / iris boundary in a grayscale image."""

    def __call__(self, img: np.ndarray, /, *args: object, **kwargs: object) -> LimbusResult | None: ...
