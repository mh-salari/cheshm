"""Pupil + glint detection: thresholding, contour fitting, marker selection.

Public surface:

  - :func:`detect_pupil_and_glints` — main entry point: takes a grayscale eye image,
    returns ``{pupil_contour, pupil_center, pupil_ellipse, glints: [{contour, center, ellipse}, ...]}``.
  - :func:`detect_limbus` — Daugman integro-differential operator limbus circle.
  - :func:`detect_glints` — bright-spot detection inside a mask.
  - :func:`pupil_center_of_mass` — centre-of-mass of the thresholded pupil area.
  - :func:`fit_convex_hull_spline` — periodic cubic B-spline through the convex hull of a contour.
  - :func:`crop_side` — convenience: return the left/right half of a binocular frame.
"""

import operator

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
    r_min_factor: float = 1.5,
    r_max_factor: float = 5.0,
) -> tuple[tuple[float, float], float]:
    """Daugman integro-differential operator limbus circle: ``(center_xy, radius)``.

    The centre search is seeded at the pupil centre and swept over a ±15 px
    window. The iris radius is searched between ``r_min_factor`` and
    ``r_max_factor`` times the pupil radius.
    """
    pcx, pcy = pupil_center
    r_min = max(round(pupil_radius * r_min_factor), 1)
    r_max = max(round(pupil_radius * r_max_factor), r_min + 1)
    op = IntegroDifferentialOperator(img, r_min=r_min, r_max=r_max)
    results = op.search(cen_x=round(pcy), cen_y=round(pcx), range_=15, step=1)
    if len(results) == 0:
        raise ValueError("integro-differential operator: no iris candidates")
    ly, lx, _score, lr = results[-1]
    return (float(lx), float(ly)), float(lr)


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


def detect_glints(img: np.ndarray, mask: np.ndarray, glint_threshold: int = 200) -> list[tuple[float, float]]:
    """Return ``[(x, y), ...]`` glint centroids inside ``mask`` (left-to-right by ``x``).

    Thresholds bright spots above ``glint_threshold`` inside ``mask``, splits
    merged blobs when only 3 are found, returns centroids in image coordinates.
    """
    masked = img.copy()
    masked[mask == 0] = 0

    _, glint_mask = cv2.threshold(masked, glint_threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(glint_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []
    for c in contours:
        if c.squeeze().ndim < 2:
            continue
        m = cv2.moments(c)
        if m["m00"] > 0:
            blobs.append({
                "contour": c,
                "cx": m["m10"] / m["m00"],
                "cy": m["m01"] / m["m00"],
                "w": cv2.boundingRect(c)[2],
            })
    blobs.sort(key=operator.itemgetter("cx"))

    # if 3 blobs, split the widest one at its horizontal midpoint
    if len(blobs) == 3:
        widths = [b["w"] for b in blobs]
        median_w = sorted(widths)[1]
        widest_idx = np.argmax(widths)
        if widths[widest_idx] > 1.3 * median_w:
            wide = blobs.pop(widest_idx)
            bx, by, bw, bh = cv2.boundingRect(wide["contour"])
            mid_x = bx + bw // 2
            for x_range in [(bx, mid_x), (mid_x, bx + bw)]:
                half_mask = np.zeros_like(glint_mask)
                half_mask[by : by + bh, x_range[0] : x_range[1]] = glint_mask[by : by + bh, x_range[0] : x_range[1]]
                half_contours, _ = cv2.findContours(half_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for hc in half_contours:
                    hm = cv2.moments(hc)
                    if hm["m00"] > 0:
                        blobs.append({
                            "contour": hc,
                            "cx": hm["m10"] / hm["m00"],
                            "cy": hm["m01"] / hm["m00"],
                            "w": cv2.boundingRect(hc)[2],
                        })
            blobs.sort(key=operator.itemgetter("cx"))

    return [(b["cx"], b["cy"]) for b in blobs]


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


def detect_pupil_and_glints(
    img: np.ndarray,
    pupil_threshold: int = 30,
    glint_threshold: int = 240,
    glint_margin_ratio: float = 0.1,
    glints_target: int = 1,
    glint_max_area_ratio: float = 0.1,
    pupil_center_method: str = "convex_hull_centroid",
    glint_center_method: str = "min_area_rect_center",
    pupil_roi: tuple[int, int, int, int] | None = None,
    glint_roi: tuple[int, int, int, int] | None = None,
) -> dict | None:
    """Detect the pupil contour, the limbus circle, and glint contours in a grayscale eye image.

    Returns ``{pupil_contour, pupil_center, pupil_ellipse, glints, limbus,
    pupil_mask, glint_search_area}`` on success. Returns ``None`` when the
    chosen pupil method cannot produce a result at the current thresholds
    (no contour, hull with fewer than 5 points, zero-mass mask, etc.).

    ``pupil_ellipse`` is ``((cx, cy), (w, h), angle)``; the centre is chosen
    by ``pupil_center_method`` and ``(w, h, angle)`` come from
    ``cv2.fitEllipse`` on the convex hull of the pupil contour. Border-
    touching dark contours are rejected so the pupil is always an interior
    region — unless ``pupil_roi`` is set, in which case the largest contour
    inside the ROI is accepted regardless of border contact. ``limbus`` is
    ``{"center": [lx, ly], "radius": r}`` or ``None`` if Daugman IDO did
    not converge.

    ``pupil_center_method`` selects the pupil centre. The supported values are
    the four methods of :func:`_contour_center`, plus ``"center_of_mass"``
    which uses :func:`pupil_center_of_mass` (which preserves the glint hole
    in the pupil mask — matches EyeLink Centroid mode) and
    ``"convex_hull_centroid"`` which uses the spline centroid for higher
    sub-pixel stability than the raw hull moment.

    ``glint_margin_ratio`` is signed: positive expands the glint search
    region outward into the iris, negative shrinks it inward.
      - 0.0  -> search region = pupil boundary
      - +X   -> dilate by ``X * (limbus_radius - pupil_radius)`` pixels,
                so +1.0 reaches the limbus. Falls back to scaling by
                ``pupil_radius`` when limbus detection fails.
      - -X   -> erode by ``X * pupil_radius`` pixels, so -1.0 collapses
                to the pupil centre.

    ``glints_target`` is the number of physical IR LEDs in the rig. When 1
    (default), every bright blob inside the search region is unioned into a
    single centroid so a saturated reflection split across contours still
    yields one glint.

    ``glint_max_area_ratio`` rejects bright contours whose area exceeds this
    fraction of the pupil area — a guard against skin / eyelid bleed-through
    above ``glint_threshold``.

    ``glint_center_method`` chooses how each retained glint contour is
    reduced to a single point — see :func:`_contour_center`. The default
    ``"min_area_rect_center"`` is robust against the asymmetric brightness
    distribution that biases plain moment centroids on rectangular glints.

    ``pupil_roi`` and ``glint_roi`` are optional ``(x, y, w, h)`` rectangles.
    When set, the corresponding search is confined to that rectangle;
    ``glint_roi`` also overrides the pupil-mask + margin constraint and
    disables the ``glint_max_area_ratio`` filter (an explicit ROI is treated
    as "the glint is here").
    """
    _, pupil_mask = cv2.threshold(img, pupil_threshold, 255, cv2.THRESH_BINARY_INV)
    if pupil_roi is not None:
        pupil_mask = cv2.bitwise_and(pupil_mask, _roi_mask(img.shape, pupil_roi))
    contours, _ = cv2.findContours(pupil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if pupil_roi is not None:
        # ROI is an explicit "the pupil is here" signal — take any non-empty
        # contour. The border-rejection filter only makes sense without ROI,
        # where it guards against picking up the frame's vignette.
        candidates = [c for c in contours if cv2.contourArea(c) > 0]
    else:
        candidates = [c for c in contours if not _touches_border(c, img.shape)]
    if not candidates:
        return None
    pupil_contour = max(candidates, key=cv2.contourArea)
    hull = cv2.convexHull(pupil_contour)
    if len(hull) < 5:
        return None
    # Computed once and reused for the (w, h, angle) tuple and the
    # ``ellipse_fit_center`` method.
    ellipse_fit = cv2.fitEllipse(hull)

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
    pupil_center = (round(cx), round(cy))
    _, (w, h), angle = ellipse_fit
    pupil_ellipse = ((cx, cy), (w, h), angle)
    pupil_area = float(cv2.contourArea(pupil_contour))
    glint_max_area = pupil_area * glint_max_area_ratio

    _, glint_mask = cv2.threshold(img, glint_threshold, 255, cv2.THRESH_BINARY)

    # Limbus is only consumed when the glint search region expands outward
    # from the pupil (positive glint_margin_ratio with no explicit glint_roi).
    # In every other case — pupil-only callers passing a negative margin,
    # callers supplying an explicit glint_roi, or callers that never look
    # at the limbus output — running Daugman IDO would be pure cost on the
    # main thread. Skip it.
    pupil_radius = max(w, h) / 2
    needs_limbus = glint_roi is None and glint_margin_ratio > 0
    limbus: dict | None = None
    if needs_limbus:
        try:
            (lcx, lcy), lr = detect_limbus(img, (cx, cy), pupil_radius)
            limbus = {"center": [float(lcx), float(lcy)], "radius": float(lr)}
        except Exception:  # noqa: S110 - optional; Daugman IDO can fail at extreme thresholds
            pass

    # Glint search region: explicit ROI overrides the pupil-mask + dilation default.
    # Dilation is expressed as a fraction of the iris ring (limbus_r - pupil_r)
    # so the tuning value transfers across image resolutions.
    if glint_roi is not None:
        glint_search_mask = _roi_mask(img.shape, glint_roi)
    else:
        glint_search_mask = np.zeros_like(glint_mask)
        cv2.drawContours(glint_search_mask, [pupil_contour], -1, 255, thickness=cv2.FILLED)
        if glint_margin_ratio > 0:
            # Expand outward in iris-ring units so +100% reaches the limbus.
            # If limbus detection failed, fall back to pupil_radius as the scale.
            ring_px = max(limbus["radius"] - pupil_radius, 0.0) if limbus is not None else pupil_radius
            glint_margin_px = round(glint_margin_ratio * ring_px)
            if glint_margin_px > 0:
                k = 2 * glint_margin_px + 1
                glint_search_mask = cv2.dilate(
                    glint_search_mask,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
                )
        elif glint_margin_ratio < 0:
            # Shrink inward in pupil-radius units so -100% collapses to the centre.
            erosion_px = round(-glint_margin_ratio * pupil_radius)
            if erosion_px > 0:
                k = 2 * erosion_px + 1
                glint_search_mask = cv2.erode(
                    glint_search_mask,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
                )

    # An explicit glint_roi means "the glint is here" — disable the area
    # filter that exists to reject skin/eyelid bleed in the no-ROI case.
    apply_area_filter = glint_roi is None

    if glints_target == 1:
        # Single-LED rig: union every bright blob inside the search region into
        # one centroid (a saturated, irregular reflection can split into multiple
        # contours that still belong to the same physical LED).
        inside_mask = cv2.bitwise_and(glint_mask, glint_search_mask)
        inside_contours, _ = cv2.findContours(inside_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if apply_area_filter:
            inside_contours = [c for c in inside_contours if cv2.contourArea(c) <= glint_max_area]
        if inside_contours:
            union = np.zeros_like(glint_mask)
            cv2.drawContours(union, inside_contours, -1, 255, thickness=cv2.FILLED)
            u_contours, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered = [max(u_contours, key=cv2.contourArea)] if u_contours else []
        else:
            filtered = []
    else:
        # Multi-LED: keep every bright blob whose centroid lands inside the
        # search region. The area filter only kicks in without glint_roi.
        glint_contours, _ = cv2.findContours(glint_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = []
        for c in glint_contours:
            if apply_area_filter and cv2.contourArea(c) > glint_max_area:
                continue
            gm = cv2.moments(c)
            if gm["m00"] > 0:
                cx = int(gm["m10"] / gm["m00"])
                cy = int(gm["m01"] / gm["m00"])
                if (
                    0 <= cy < glint_search_mask.shape[0]
                    and 0 <= cx < glint_search_mask.shape[1]
                    and glint_search_mask[cy, cx] == 255
                ):
                    filtered.append(c)

    if glints_target == 4 and len(filtered) == 3:
        # 4-LED rig fallback: if one blob spans two LEDs horizontally, split it.
        widths = [cv2.boundingRect(c)[2] for c in filtered]
        median_w = sorted(widths)[1]
        widest_idx = np.argmax(widths)
        if widths[widest_idx] > 1.5 * median_w:
            wide_c = filtered.pop(widest_idx)
            x, y, w, h = cv2.boundingRect(wide_c)
            roi = glint_mask[y : y + h, x : x + w].copy()
            left_roi = roi.copy()
            left_roi[:, w // 2 :] = 0
            right_roi = roi.copy()
            right_roi[:, : w // 2] = 0
            for half_roi, offset_x in [(left_roi, x), (right_roi, x)]:
                half_contours, _ = cv2.findContours(half_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for hc in half_contours:
                    hc[:, :, 0] += offset_x
                    hc[:, :, 1] += y
                    filtered.append(hc)

    glints = []
    for c in filtered:
        try:
            cx, cy = _contour_center(c, glint_center_method)
        except ValueError:
            # Per-glint failure under the chosen method (e.g. fewer than 5
            # points for ellipse_fit_center). Drop this glint and keep the
            # rest; the user can pick a different method if too many are lost.
            continue
        ellipse = cv2.fitEllipse(c) if len(c) >= 5 else None
        glints.append({"contour": c, "center": (round(cx), round(cy)), "ellipse": ellipse})

    # Surface the binary masks so callers can show the user what the current
    # thresholds actually select. ``glint_search_area`` is the intersection of
    # the bright mask with the glint search region — exactly the pixels that
    # are evaluated for glint candidates.
    glint_search_area = cv2.bitwise_and(glint_mask, glint_search_mask)
    return {
        "pupil_contour": pupil_contour,
        "pupil_center": pupil_center,
        "pupil_ellipse": pupil_ellipse,
        "glints": glints,
        "limbus": limbus,
        "pupil_mask": pupil_mask,
        "glint_search_area": glint_search_area,
    }
