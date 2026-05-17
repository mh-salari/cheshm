"""Pupil + glint detection on grayscale eye images.

Public surface:

  - :func:`detect_pupil` — pupil contour, center, fitted ellipse.
  - :func:`detect_glints` — bright glints near a known pupil center,
    with simple, opt-in refiners (position filter, area cap, target
    coalescing, 4-LED split-the-widest).
  - :func:`detect_limbus` — Daugman integro-differential operator limbus
    circle (slow; only the limbus plugin should call this).
  - :func:`pupil_center_of_mass`, :func:`fit_convex_hull_spline` —
    building blocks that also serve as standalone helpers.
  - :func:`crop_side` — convenience: return left/right half of a binocular frame.

The pupil and glint detectors are intentionally separate. The glint
detector takes the pupil center + radius as inputs so callers compose
the two explicitly; there is no convenience wrapper that chains them.
"""

import math

import cv2
import numpy as np
from daugman_derived_boundary_detectors import IntegroDifferentialOperator
from scipy.interpolate import splev, splprep


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


def crop_side(img: np.ndarray, side: str | None) -> np.ndarray:
    """Return the left or right half of a binocular frame, or the whole image."""
    if side is None:
        return img
    _h, w = img.shape[:2]
    mid = w // 2
    if side == "left":
        return img[:, :mid]
    if side == "right":
        return img[:, mid:]
    raise ValueError(f"unknown side: {side!r} (expected 'left', 'right', or None)")


def _touches_border(contour: np.ndarray, shape: tuple[int, ...]) -> bool:
    h, w = shape[:2]
    x, y, cw, ch = cv2.boundingRect(contour)
    return x == 0 or y == 0 or x + cw == w or y + ch == h


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
    by ``pupil_center_method``; see :func:`_contour_center` for the four
    contour-based methods, plus ``"center_of_mass"`` which uses
    :func:`pupil_center_of_mass` (the glint hole stays cut out — matches
    EyeLink Centroid mode) and ``"convex_hull_centroid"`` which uses the
    spline centroid for sub-pixel stability.

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


def detect_glints(
    img: np.ndarray,
    pupil_center: tuple[float, float],
    pupil_radius: float,
    *,
    glint_threshold: int = 240,
    search_radius_factor: float = 2.0,
    search_radius_max_px: int | None = None,
    glint_roi: tuple[int, int, int, int] | None = None,
    glint_center_method: str = "min_area_rect_center",
    max_area_px: int | None = None,
    keep_above: bool = True,
    keep_below: bool = True,
    keep_left: bool = True,
    keep_right: bool = True,
    filter_margin_px: int = 5,
    glints_target: int = 1,
    split_widest_for_target: bool = False,
    min_ellipse_fit_ratio: float | None = None,
    min_roundness_ratio: float | None = None,
) -> dict:
    """Detect bright glint blobs near ``pupil_center``.

    Search region: a filled circle centred at ``pupil_center`` with radius
    ``search_radius_factor * pupil_radius``, optionally clamped from above
    by ``search_radius_max_px``. No iris / limbus geometry is involved —
    the search is a plain pixel-distance test from the pupil centre.

    Pipeline:

      1. Threshold the image at ``glint_threshold`` (bright pixels = 255).
      2. Intersect with the circular search region.
      3. Find contours; optionally drop any whose area exceeds
         ``max_area_px``.
      4. Optionally drop contours whose centroid fails the half-plane
         filter described by ``keep_above`` / ``keep_below`` /
         ``keep_left`` / ``keep_right`` (default = all four True =
         no filter). ``filter_margin_px`` softens the boundary so a
         glint sitting a few pixels past the pupil-centre line still
         counts as belonging to the chosen half.
      5. Apply the opt-in shape-quality gates
         (``min_ellipse_fit_ratio`` / ``min_roundness_ratio``) per
         contour. Contours that fail any active gate are dropped.
      6. If ``split_widest_for_target`` and exactly ``glints_target - 1``
         contours survive, split the widest in half horizontally — covers
         the 4-LED rig case where two LEDs merge into one blob.
      7. Keep the ``glints_target`` largest remaining contours by area
         (or fewer when not enough candidates survived).
      8. Compute each surviving glint's centre via
         ``glint_center_method``; the returned glints are ordered
         left-to-right by bounding-box x.

    Returns ``{"glints": [...], "search_area": np.ndarray}``. Each glint
    is ``{"contour", "center": (cx, cy), "ellipse" | None}``. The mask
    payload is the post-filter (threshold ∧ search) image — useful as a
    live overlay for tuning.
    """
    # 1+2: bright pixels inside the pupil-centred disk, optionally
    # intersected with a user-supplied rectangle (``glint_roi``).
    _, glint_mask = cv2.threshold(img, glint_threshold, 255, cv2.THRESH_BINARY)
    radius = search_radius_factor * pupil_radius
    if search_radius_max_px is not None:
        radius = min(radius, float(search_radius_max_px))
    radius_int = max(round(radius), 0)
    search_mask = np.zeros_like(glint_mask)
    cv2.circle(
        search_mask,
        (round(pupil_center[0]), round(pupil_center[1])),
        radius_int,
        255,
        thickness=-1,
    )
    if glint_roi is not None:
        search_mask = cv2.bitwise_and(search_mask, _roi_mask(img.shape, glint_roi))
    candidates_mask = cv2.bitwise_and(glint_mask, search_mask)

    contours, _ = cv2.findContours(candidates_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 3: area cap (absolute pixels) — optional.
    if max_area_px is not None:
        contours = [c for c in contours if cv2.contourArea(c) <= max_area_px]

    # 4: half-plane filter on each contour's moments centroid.
    if not (keep_above and keep_below and keep_left and keep_right):
        contours = [
            c
            for c in contours
            if _passes_half_plane_filter(
                c,
                pupil_center,
                keep_above=keep_above,
                keep_below=keep_below,
                keep_left=keep_left,
                keep_right=keep_right,
                margin_px=filter_margin_px,
            )
        ]

    # 5: drop contours whose shape doesn't match a clean glint reflection.
    accepted: list[tuple[np.ndarray, tuple]] = []
    for c in contours:
        ellipse = cv2.fitEllipse(c) if len(c) >= 5 else None
        if not _passes_shape_quality(
            c,
            ellipse,
            min_ellipse_fit_ratio=min_ellipse_fit_ratio,
            min_roundness_ratio=min_roundness_ratio,
        ):
            continue
        accepted.append((c, ellipse))

    # 6: split the widest blob in half when the rig has one more LED than
    # contours found — the 4-LED 3-found special case generalised to any
    # ``target == found + 1``. Operates on the contour list only; the
    # split halves get fresh ellipse fits below.
    if split_widest_for_target and glints_target > 1 and len(accepted) == glints_target - 1 and accepted:
        split_contours = _split_widest_blob([c for c, _ in accepted], glint_mask)
        accepted = [(c, cv2.fitEllipse(c) if len(c) >= 5 else None) for c in split_contours]

    # 7: keep the N largest by area; fewer when we don't have enough.
    accepted.sort(key=lambda pair: cv2.contourArea(pair[0]), reverse=True)
    accepted = accepted[: max(glints_target, 0)]

    # Output ordering: left-to-right by bounding-box x.
    accepted.sort(key=lambda pair: cv2.boundingRect(pair[0])[0])

    glints = []
    for c, ellipse in accepted:
        try:
            cx, cy = _contour_center(c, glint_center_method)
        except ValueError:
            # Per-glint method failure: drop this blob, keep the others.
            # The user can switch ``glint_center_method`` if too many are
            # rejected. No automatic fallback to another method.
            continue
        glints.append({"contour": c, "center": (round(cx), round(cy)), "ellipse": ellipse})

    return {"glints": glints, "search_area": candidates_mask}


def _passes_half_plane_filter(
    contour: np.ndarray,
    pupil_center: tuple[float, float],
    *,
    keep_above: bool,
    keep_below: bool,
    keep_left: bool,
    keep_right: bool,
    margin_px: int,
) -> bool:
    """Return True iff the contour's centroid sits in the kept half-planes.

    Vertical (above/below) and horizontal (left/right) checks are
    independent. Each axis has three meaningful states:

      - both halves checked: no filter on that axis.
      - one half checked: that half plus ``margin_px`` across the
        boundary is admitted (a real glint sitting a few pixels off the
        pupil-centre line still counts as belonging to the chosen half).
      - neither half checked: nothing passes that axis.
    """
    m = cv2.moments(contour)
    if m["m00"] <= 0:
        return False
    gx = m["m10"] / m["m00"]
    gy = m["m01"] / m["m00"]
    cx, cy = pupil_center

    if keep_above and keep_below:
        vertical_ok = True
    elif keep_above:
        vertical_ok = gy < cy + margin_px
    elif keep_below:
        vertical_ok = gy > cy - margin_px
    else:
        vertical_ok = False

    if keep_left and keep_right:
        horizontal_ok = True
    elif keep_left:
        horizontal_ok = gx < cx + margin_px
    elif keep_right:
        horizontal_ok = gx > cx - margin_px
    else:
        horizontal_ok = False

    return vertical_ok and horizontal_ok


def _split_widest_blob(contours: list[np.ndarray], mask: np.ndarray) -> list[np.ndarray]:
    """Split the widest contour in two halves horizontally.

    Used when the rig has one more LED than the threshold + filter
    pipeline produced contours — typical case is two adjacent LEDs whose
    glints merged into a single bright blob.
    """
    widths = [cv2.boundingRect(c)[2] for c in contours]
    widest_idx = int(np.argmax(widths))
    median_w = sorted(widths)[len(widths) // 2]
    # Only split when the widest is meaningfully wider than the others —
    # a uniform set of LEDs is not a candidate for splitting.
    if widths[widest_idx] <= 1.3 * median_w:
        return contours
    survivors = [c for i, c in enumerate(contours) if i != widest_idx]
    wide = contours[widest_idx]
    bx, by, bw, bh = cv2.boundingRect(wide)
    mid_x = bx + bw // 2
    halves: list[np.ndarray] = []
    for x_lo, x_hi in ((bx, mid_x), (mid_x, bx + bw)):
        half = np.zeros_like(mask)
        half[by : by + bh, x_lo:x_hi] = mask[by : by + bh, x_lo:x_hi]
        half_contours, _ = cv2.findContours(half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        halves.extend(half_contours)
    return survivors + halves
