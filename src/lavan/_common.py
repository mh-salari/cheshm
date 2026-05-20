"""Shared internal helpers used by the lavan detector modules.

  - :func:`_roi_mask` — build a binary mask from a rectangle.
  - :func:`_crop_to_roi` / :func:`_translate_pupil_result` — the crop +
    shift-back pair every pupil detector uses to honour ``pupil_roi``.
  - :func:`_contour_center` — four ways to take a contour's centre.
  - :func:`_passes_shape_quality` — two opt-in gates (ellipse-fit ratio
    and isoperimetric roundness) used to reject contours whose shape
    doesn't match a clean pupil / glint blob.
"""

import math

import cv2
import numpy as np


def _roi_mask(shape: tuple[int, ...], roi: tuple[int, int, int, int]) -> np.ndarray:
    """Build a uint8 mask with the ``(x, y, w, h)`` rectangle set to 255."""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    rx, ry, rw, rh = (int(v) for v in roi)
    rx, ry = max(rx, 0), max(ry, 0)
    rw, rh = max(min(rw, w - rx), 0), max(min(rh, h - ry), 0)
    if rw and rh:
        mask[ry : ry + rh, rx : rx + rw] = 255
    return mask


def _crop_to_roi(
    img: np.ndarray,
    roi: tuple[int, int, int, int],
) -> tuple[np.ndarray, tuple[int, int]]:
    """Clamp ``roi=(x, y, w, h)`` to ``img`` bounds and return ``(crop, (x0, y0))``.

    ``crop`` is a numpy view over ``img`` (no copy); ``(x0, y0)`` is the
    clamped top-left to add back to any crop-local coordinate to recover
    the full-image coordinate. ``cv2.Rect`` semantics — integer pixels,
    so no half-pixel offset is introduced.

    Raises ``ValueError`` if the ROI has zero overlap with the image.
    """
    h, w = img.shape[:2]
    rx, ry, rw, rh = (int(v) for v in roi)
    x0 = max(rx, 0)
    y0 = max(ry, 0)
    x1 = min(rx + rw, w)
    y1 = min(ry + rh, h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"ROI {tuple(roi)} has zero overlap with {(w, h)} image")
    return img[y0:y1, x0:x1], (x0, y0)


def _translate_pupil_result(
    result: dict | None,
    dx: int,
    dy: int,
    full_shape: tuple[int, ...] | None = None,
) -> dict | None:
    """Shift a pupil-detector result from crop-local back to full-image coords.

    Fields are translated only when present in ``result``:

      - ``center``: ``(cx, cy) + (dx, dy)``
      - ``ellipse``: ``((cx, cy), size, angle)`` — centre shifted
      - ``contour``: ``(N, 1, 2)`` integer array — each point shifted
      - ``mask``: pasted into a zero canvas of ``full_shape`` at ``(dy, dx)``;
        skipped when ``full_shape`` is None.

    Returns ``None`` if ``result`` is ``None``.
    """
    if result is None:
        return None
    out = dict(result)
    if out.get("center") is not None:
        cx, cy = out["center"]
        out["center"] = (cx + dx, cy + dy)
    if out.get("ellipse") is not None:
        (cx, cy), size, angle = out["ellipse"]
        out["ellipse"] = ((cx + dx, cy + dy), size, angle)
    if out.get("contour") is not None:
        contour = out["contour"]
        out["contour"] = contour + np.array([[[dx, dy]]], dtype=contour.dtype)
    if out.get("mask") is not None and full_shape is not None:
        crop_mask = out["mask"]
        full_mask = np.zeros(full_shape[:2], dtype=crop_mask.dtype)
        mh, mw = crop_mask.shape
        full_mask[dy : dy + mh, dx : dx + mw] = crop_mask
        out["mask"] = full_mask
    return out


def _contour_center(contour: np.ndarray, method: str) -> tuple[float, float]:
    """Geometric centre of ``contour`` chosen by ``method``.

    Methods:
      - ``convex_hull_centroid``: moments centroid of the filled convex hull.
      - ``center_of_mass``: moments centroid of the contour itself.
      - ``ellipse_fit_center``: centre of ``cv2.fitEllipse`` (needs >= 5 points).
      - ``min_area_rect_center``: centre of the minimum-area rotated rect.

    """
    if method == "convex_hull_centroid":
        hull = cv2.convexHull(contour)
        m = cv2.moments(hull)
        if m["m00"] <= 0:
            raise ValueError("convex_hull_centroid: zero-area hull")
        return (m["m10"] / m["m00"], m["m01"] / m["m00"])
    if method == "center_of_mass":
        m = cv2.moments(contour)
        if m["m00"] <= 0:
            raise ValueError("center_of_mass: zero-area contour")
        return (m["m10"] / m["m00"], m["m01"] / m["m00"])
    if method == "ellipse_fit_center":
        if len(contour) < 5:
            raise ValueError("ellipse_fit_center: contour needs >= 5 points")
        (cx, cy), _, _ = cv2.fitEllipse(contour)
        return (cx, cy)
    if method == "min_area_rect_center":
        (cx, cy), _, _ = cv2.minAreaRect(contour)
        return (cx, cy)
    raise ValueError(
        f"unknown center method {method!r}; expected one of "
        "'convex_hull_centroid', 'center_of_mass', 'ellipse_fit_center', 'min_area_rect_center'",
    )


def _passes_shape_quality(
    contour: np.ndarray,
    ellipse_fit: tuple[tuple[float, float], tuple[float, float], float] | None,
    *,
    min_ellipse_fit_ratio: float | None,
    min_roundness_ratio: float | None,
) -> bool:
    """Return ``True`` iff ``contour`` satisfies the active shape-quality gates.

    Each gate is skipped when its threshold is ``None``. With both gates
    inactive the function always returns ``True`` and the upstream
    pipeline runs unchanged. ``ellipse_fit`` may be ``None`` (small
    contours with < 5 points have no fittable ellipse); in that case
    the ellipse-fit gate fails by construction while the roundness gate
    still applies since it only needs the contour itself.
    """
    if min_ellipse_fit_ratio is None and min_roundness_ratio is None:
        return True
    contour_area = float(cv2.contourArea(contour))
    if min_ellipse_fit_ratio is not None:
        if ellipse_fit is None:
            return False
        _, (w, h), _ = ellipse_fit
        ellipse_area = math.pi * (w / 2.0) * (h / 2.0)
        if ellipse_area <= 0:
            return False
        if contour_area / ellipse_area < float(min_ellipse_fit_ratio):
            return False
    if min_roundness_ratio is not None:
        perimeter = float(cv2.arcLength(contour, closed=True))
        if perimeter <= 0:
            return False
        roundness = 4.0 * math.pi * contour_area / (perimeter * perimeter)
        if roundness < float(min_roundness_ratio):
            return False
    return True
