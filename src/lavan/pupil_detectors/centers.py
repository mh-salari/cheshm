"""Pupil-center methods.

- :func:`pupil_center_of_mass` — moments centroid of the thresholded
pupil region, with the glint hole preserved.
- :func:`fit_convex_hull_spline` — periodic cubic B-spline through the
convex hull of a contour; returns the sampled boundary, the centroid
of the enclosed region, and the equivalent diameter.
"""

import cv2
import numpy as np
from scipy.interpolate import splev, splprep


def fit_convex_hull_spline(contour: np.ndarray, n_points: int = 200) -> dict:
    """Fit a smooth closed cubic B-spline to the convex hull of a contour.

    Steps:
      1. Take the convex hull of the input contour points.
      2. Fit a periodic cubic B-spline through the hull vertices.
      3. Sample ``n_points`` evenly along the spline.
      4. Compute the enclosed area and centroid via Green's theorem on the
         sampled curve.

    Returns:
        points: ``(n_points, 2)`` sampled spline boundary.
        center: ``(cx, cy)`` centroid of the enclosed region.
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
