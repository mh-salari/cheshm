"""Pupil detection on grayscale eye images.

Public surface:

  - :func:`detect_pupil` — pupil contour, center, fitted ellipse, mask.
  - :func:`fit_convex_hull_spline` — periodic cubic B-spline through the
    convex hull of a contour, with centroid + equivalent diameter. Used
    internally by ``detect_pupil`` and exposed standalone.
  - :func:`pupil_center_of_mass` — moments centroid of the thresholded
    pupil region with the glint hole preserved.

The detector is intentionally one-shot and stateless: no temporal
tracking, no calibration, no model fitting across frames. See
:mod:`lavan.detect.glint` for the bright-blob glint detector that
consumes ``detect_pupil``'s centre + radius.
"""

import cv2
import numpy as np
from scipy.interpolate import splev, splprep

from .utils import _contour_center, _passes_shape_quality, _roi_mask


def fit_convex_hull_spline(contour: np.ndarray, n_points: int = 200) -> dict:
    """Fit a smooth closed cubic B-spline to the convex hull of a contour.

    Steps:
      1. Take the convex hull of the input contour points.
      2. Fit a periodic cubic B-spline through the hull vertices.
      3. Sample `n_points` evenly along the spline.
      4. Compute the enclosed area and centroid via Green's theorem on the
         sampled curve.

    Returns:
        points: (n_points, 2) sampled spline boundary.
        center: (cx, cy) centroid of the enclosed region.
        equiv_diam: diameter of a circle with the same enclosed area.

    """
    hull_indices = cv2.convexHull(contour, returnPoints=False).squeeze()
    hull_pts = contour.squeeze()[np.sort(hull_indices)]

    pts = np.vstack([hull_pts, hull_pts[0]])
    x, y = pts[:, 0].astype(float), pts[:, 1].astype(float)

    tck, _ = splprep([x, y], s=0, per=True, k=3)
    t = np.linspace(0, 1, n_points)
    sx, sy = splev(t, tck)

    sx_c = np.append(sx, sx[0])
    sy_c = np.append(sy, sy[0])
    cross = sx_c[:-1] * sy_c[1:] - sx_c[1:] * sy_c[:-1]
    signed_area = 0.5 * np.sum(cross)
    area = abs(signed_area)
    if signed_area != 0:
        cx = np.sum((sx_c[:-1] + sx_c[1:]) * cross) / (6 * signed_area)
        cy = np.sum((sy_c[:-1] + sy_c[1:]) * cross) / (6 * signed_area)
    else:
        cx, cy = float(np.mean(sx)), float(np.mean(sy))

    equiv_diam = 2 * np.sqrt(area / np.pi)
    return {
        "points": np.column_stack([sx, sy]),
        "center": (float(cx), float(cy)),
        "equiv_diam": float(equiv_diam),
    }


def pupil_center_of_mass(
    pupil_mask: np.ndarray,
    pupil_contour: np.ndarray,
) -> tuple[float, float] | None:
    """Centre-of-mass of the thresholded pupil area, with glint cutouts preserved.

    The glint creates a hole in the dark pupil region; that hole biases the
    centroid away from the glint side. A convex-hull-based centroid fills the
    hole in and therefore lands somewhere different.

    Returns ``None`` if the pupil mass is zero (degenerate input).
    """
    contour_mask = np.zeros_like(pupil_mask)
    cv2.drawContours(contour_mask, [pupil_contour], -1, 255, thickness=cv2.FILLED)
    pupil_only = cv2.bitwise_and(pupil_mask, contour_mask)
    m = cv2.moments(pupil_only, binaryImage=True)
    if m["m00"] == 0:
        return None
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


def _touches_border(contour: np.ndarray, shape: tuple[int, ...]) -> bool:
    h, w = shape[:2]
    x, y, cw, ch = cv2.boundingRect(contour)
    return x == 0 or y == 0 or x + cw == w or y + ch == h


def detect_pupil(
    img: np.ndarray,
    pupil_threshold: int = 30,
    pupil_center_method: str = "convex_hull_centroid",
    pupil_roi: tuple[int, int, int, int] | None = None,
    *,
    min_ellipse_fit_ratio: float | None = None,
    min_roundness_ratio: float | None = None,
) -> dict | None:
    """Detect the pupil contour, centre, and fitted ellipse in a grayscale image.

    Returns ``{contour, center, ellipse, mask}`` on success, or ``None``
    when no pupil can be produced at the current parameters (no candidate
    contour, hull with fewer than 5 points, zero-mass mask, etc.).

    ``ellipse`` is ``((cx, cy), (w, h), angle)`` from ``cv2.fitEllipse``
    on the convex hull of the pupil contour. The pupil centre is chosen
    by ``pupil_center_method``; see :func:`._contour_center` for the four
    contour-based methods, plus ``"center_of_mass"`` which uses
    :func:`pupil_center_of_mass` (the glint hole stays cut out) and
    ``"convex_hull_centroid"`` which uses the spline centroid for
    sub-pixel stability.

    Border-touching dark contours are rejected so the pupil is always an
    interior region — unless ``pupil_roi`` is set, in which case the
    largest contour inside the ROI is accepted regardless of border
    contact (an explicit ROI is treated as "the pupil is here").

    Two optional post-fit shape-quality gates reject detections that
    don't look pupil-shaped:

    - ``min_ellipse_fit_ratio`` (0..1): ``contour_area /
      fitted_ellipse_area``. Catches detections whose contour is
      fragmented or under-fills its fit (irregular blobs, multi-component
      masks).
    - ``min_roundness_ratio`` (0..1): ``4·π·area / perimeter²``, the
      isoperimetric quotient. ``1.0`` = perfect circle. Catches
      jagged or elongated contours; only useful when the camera views
      the eye on-axis (off-axis pupils are legitimately elliptical and
      score well below 1.0).

    Both gates default to ``None`` (off). When either is set, candidate
    contours are walked from largest to smallest and the first one that
    satisfies both active gates is chosen. ``None`` is returned only if
    every candidate fails. With both gates off, behaviour matches the
    no-gate baseline: the largest contour wins without any shape check.
    """
    _, pupil_mask = cv2.threshold(img, pupil_threshold, 255, cv2.THRESH_BINARY_INV)
    if pupil_roi is not None:
        pupil_mask = cv2.bitwise_and(pupil_mask, _roi_mask(img.shape, pupil_roi))
    contours, _ = cv2.findContours(pupil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if pupil_roi is not None:
        # An explicit ROI overrides the border-rejection filter — the
        # filter only makes sense to guard against frame vignette.
        candidates = [c for c in contours if cv2.contourArea(c) > 0]
    else:
        candidates = [c for c in contours if not _touches_border(c, img.shape)]
    if not candidates:
        return None
    # Walk candidates largest -> smallest; with shape gates active, skip
    # contours that fail the gates and try the next one. Without gates,
    # the first iteration matches the prior "max by area" pick.
    candidates.sort(key=cv2.contourArea, reverse=True)
    pupil_contour = None
    ellipse_fit = None
    for candidate in candidates:
        candidate_hull = cv2.convexHull(candidate)
        if len(candidate_hull) < 5:
            continue
        candidate_fit = cv2.fitEllipse(candidate_hull)
        if not _passes_shape_quality(
            candidate,
            candidate_fit,
            min_ellipse_fit_ratio=min_ellipse_fit_ratio,
            min_roundness_ratio=min_roundness_ratio,
        ):
            continue
        pupil_contour = candidate
        ellipse_fit = candidate_fit
        break
    if pupil_contour is None:
        return None

    if pupil_center_method == "center_of_mass":
        com = pupil_center_of_mass(pupil_mask, pupil_contour)
        if com is None:
            return None
        cx, cy = com
    elif pupil_center_method == "convex_hull_centroid":
        cx, cy = fit_convex_hull_spline(pupil_contour)["center"]
    elif pupil_center_method == "ellipse_fit_center":
        (cx, cy), _, _ = ellipse_fit
    else:
        cx, cy = _contour_center(pupil_contour, pupil_center_method)

    _, (w, h), angle = ellipse_fit
    return {
        "contour": pupil_contour,
        "center": (round(cx), round(cy)),
        "ellipse": ((cx, cy), (w, h), angle),
        "mask": pupil_mask,
    }
