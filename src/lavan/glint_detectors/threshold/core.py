"""Bright-glint detection on grayscale eye images.

Public surface:

  - :func:`detect_glints` — bright glint blobs near a known pupil centre,
    with opt-in refiners (position half-plane filter, area cap, target
    coalescing via widest-blob split, shape-quality gates).

The detector takes ``pupil_center`` and ``pupil_radius`` as inputs —
typically the output of :func:`lavan.detect.detect_pupil`. There is no
convenience wrapper that chains pupil + glint; callers compose the two
explicitly so the cost of each pass is visible at the call site.
"""

from typing import Literal

import cv2
import numpy as np

from lavan._common import _contour_center, _passes_shape_quality, _roi_mask

# GUI metadata. Defaults / types / choices come from the function
# signature; this dict carries slider bounds, per-param help, widget
# hints, and label overrides where the auto-derived label would be
# wrong (acronyms, units in parens, UK English).
# Overlays this detector produces.
_OVERLAYS = (
    ("contour", "line"),
    ("center", "point"),
    ("mask", "fill"),
)

_UI = {
    "pupil_center": {"hidden": True},
    "pupil_radius": {"hidden": True},
    "glint_threshold": {
        "min": 0,
        "max": 255,
        "help": "Intensity above which a pixel is considered glint.",
    },
    "search_radius_factor": {
        "min": 0.1,
        "max": 10.0,
        "help": "Multiplied by pupil_radius to define the glint search disk. Ignored when pupil_center / pupil_radius are not supplied.",
    },
    "search_radius_max_px": {
        "min": 1,
        "max": 4096,
        "label": "Search radius max (px)",
        "help": "Optional upper bound on the search disk radius. None = no cap.",
    },
    "glint_roi": {
        "widget": "roi",
        "label": "Glint ROI",
        "help": "Optional (x, y, w, h) rectangle. Intersected with the search disk.",
        "hidden": True,
    },
    "glint_center_method": {"label": "Centre method"},
    "max_area_px": {
        "min": 1,
        "max": 100000,
        "label": "Max area (px)",
        "help": "Reject glint blobs larger than this. None = no cap.",
    },
    "keep_above": {"label": "Keep above pupil"},
    "keep_below": {"label": "Keep below pupil"},
    "keep_left": {"label": "Keep left of pupil"},
    "keep_right": {"label": "Keep right of pupil"},
    "filter_margin_px": {
        "min": 0,
        "max": 100,
        "label": "Half-plane margin (px)",
    },
    "glints_target": {
        "min": 1,
        "max": 8,
        "label": "Target glint count",
        "help": "Number of IR LEDs expected.",
    },
    "split_widest_for_target": {
        "label": "Split widest blob",
        "help": "When the target count is N but only one blob is found, split it into N along its long axis.",
    },
    "min_ellipse_fit_ratio": {
        "min": 0.0,
        "max": 1.0,
        "label": "Min ellipse-fit ratio",
    },
    "min_roundness_ratio": {
        "min": 0.0,
        "max": 1.0,
    },
}


def detect_glints(
    img: np.ndarray,
    *,
    pupil_center: tuple[float, float] | None = None,
    pupil_radius: float | None = None,
    glint_threshold: int = 240,
    search_radius_factor: float = 2.0,
    search_radius_max_px: int | None = None,
    glint_roi: tuple[int, int, int, int] | None = None,
    glint_center_method: Literal[
        "convex_hull_centroid",
        "hull_moments_centroid",
        "center_of_mass",
        "ellipse_fit_center",
        "min_area_rect_center",
    ] = "min_area_rect_center",
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
    # 1+2: bright pixels inside the candidate region. The candidate
    # region is the intersection of (a) the pupil-centred disk when
    # ``pupil_center`` and ``pupil_radius`` are both supplied, otherwise
    # the whole image, and (b) ``glint_roi`` when supplied.
    _, glint_mask = cv2.threshold(img, glint_threshold, 255, cv2.THRESH_BINARY)
    if pupil_center is not None and pupil_radius is not None:
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
    else:
        search_mask = np.full_like(glint_mask, 255)
    if glint_roi is not None:
        search_mask = cv2.bitwise_and(search_mask, _roi_mask(img.shape, glint_roi))
    candidates_mask = cv2.bitwise_and(glint_mask, search_mask)

    contours, _ = cv2.findContours(candidates_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 3: area cap (absolute pixels) — optional.
    if max_area_px is not None:
        contours = [c for c in contours if cv2.contourArea(c) <= max_area_px]

    # 4: half-plane filter on each contour's moments centroid. The
    # filter is anchored at ``pupil_center``, so it is a no-op when no
    # pupil centre is supplied.
    if pupil_center is not None and not (keep_above and keep_below and keep_left and keep_right):
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
