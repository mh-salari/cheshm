"""Simple glint detector — ctypes binding to the C++ kernel in ``core.dylib``.

Threshold-based detector: bright pixels inside the search disk around
the pupil centre form the candidate region, contours are filtered by
area / half-plane / shape-quality gates, optionally split when the rig
has one more LED than contours found, sorted left-to-right, and each
glint's centre is computed via one of five methods.
"""

import ctypes
import math
import pathlib
import platform
from typing import Literal

import cv2
import numpy as np

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
        "help": "Optional (x, y, w, h) rectangle.",
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

_CENTER_METHOD_CODE = {
    "convex_hull_centroid": 0,
    "center_of_mass": 1,
    "ellipse_fit_center": 2,
    "min_area_rect_center": 3,
    "hull_moments_centroid": 4,
}


_LIB_DIR = pathlib.Path(__file__).parent
_lib_ext = {"Darwin": ".dylib", "Linux": ".so", "Windows": ".dll"}[platform.system()]
_lib = ctypes.CDLL(str(_LIB_DIR / f"core{_lib_ext}"))
_lib.Simple_glint_detect.restype = ctypes.c_int
_lib.Simple_glint_detect.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),  # img_data
    ctypes.c_int,                    # width
    ctypes.c_int,                    # height
    ctypes.c_int,                    # roi_x
    ctypes.c_int,                    # roi_y
    ctypes.c_int,                    # roi_w
    ctypes.c_int,                    # roi_h
    ctypes.c_int,                    # has_pupil
    ctypes.c_double,                 # pupil_cx
    ctypes.c_double,                 # pupil_cy
    ctypes.c_double,                 # pupil_radius
    ctypes.c_int,                    # glint_threshold
    ctypes.c_double,                 # search_radius_factor
    ctypes.c_int,                    # search_radius_max_px (-1 = no cap)
    ctypes.c_int,                    # glint_center_method
    ctypes.c_int,                    # max_area_px (-1 = no cap)
    ctypes.c_int,                    # keep_above
    ctypes.c_int,                    # keep_below
    ctypes.c_int,                    # keep_left
    ctypes.c_int,                    # keep_right
    ctypes.c_int,                    # filter_margin_px
    ctypes.c_int,                    # glints_target
    ctypes.c_int,                    # split_widest_for_target
    ctypes.c_double,                 # min_ellipse_fit_ratio (< 0 = off)
    ctypes.c_double,                 # min_roundness_ratio   (< 0 = off)
    ctypes.POINTER(ctypes.c_int),    # out_n_glints
    ctypes.POINTER(ctypes.c_double), # out_centers_xy (2 * max_glints)
    ctypes.POINTER(ctypes.c_double), # out_ellipse_params (6 * max_glints)
    ctypes.POINTER(ctypes.c_int),    # out_contour_lengths (max_glints)
    ctypes.POINTER(ctypes.c_double), # out_contours_xy (2 * max_glints * max_pts)
    ctypes.c_int,                    # max_glints
    ctypes.c_int,                    # max_contour_points_per_glint
    ctypes.POINTER(ctypes.c_uint8),  # out_search_mask (width * height)
]


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
    max_glints: int = 16,
    max_contour_points_per_glint: int = 512,
) -> dict:
    """Detect bright glint blobs near ``pupil_center``.

    Search region: a filled circle centred at ``pupil_center`` with radius
    ``search_radius_factor * pupil_radius``, optionally clamped from above
    by ``search_radius_max_px``. ``glint_roi=(x, y, w, h)`` runs the
    algorithm on the cropped sub-image and translates outputs back.

    Pipeline:

      1. Threshold the image at ``glint_threshold`` (bright pixels = 255).
      2. Intersect with the circular search region (when pupil is supplied).
      3. Find contours; optionally drop any whose area exceeds
         ``max_area_px``.
      4. Optionally drop contours whose centroid fails the half-plane
         filter described by ``keep_above`` / ``keep_below`` /
         ``keep_left`` / ``keep_right`` (default = all four True =
         no filter). ``filter_margin_px`` softens the boundary.
      5. Apply the opt-in shape-quality gates
         (``min_ellipse_fit_ratio`` / ``min_roundness_ratio``).
      6. If ``split_widest_for_target`` and exactly ``glints_target - 1``
         contours survive, split the widest in half horizontally.
      7. Keep the ``glints_target`` largest remaining contours by area.
      8. Compute each surviving glint's centre via
         ``glint_center_method``; the returned glints are ordered
         left-to-right by bounding-box x.

    Returns ``{"glints": [...], "search_area": np.ndarray}``. Each glint
    is ``{"contour", "center": (cx, cy), "ellipse" | None}``. The mask
    payload is the post-filter (threshold ∧ search) image — useful as a
    live overlay for tuning.
    """
    if img.ndim != 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.ascontiguousarray(img, dtype=np.uint8)
    height, width = img.shape

    if glint_roi is None:
        roi_x = roi_y = roi_w = roi_h = 0
    else:
        roi_x, roi_y, roi_w, roi_h = (int(v) for v in glint_roi)

    has_pupil = pupil_center is not None and pupil_radius is not None
    cx = float(pupil_center[0]) if has_pupil else 0.0
    cy = float(pupil_center[1]) if has_pupil else 0.0
    pr = float(pupil_radius) if has_pupil else 0.0

    out_n = ctypes.c_int(0)
    centers_buf = (ctypes.c_double * (2 * max_glints))()
    ellipse_buf = (ctypes.c_double * (6 * max_glints))()
    lengths_buf = (ctypes.c_int * max_glints)()
    contours_buf = (ctypes.c_double * (2 * max_glints * max_contour_points_per_glint))()
    search_area = np.zeros((height, width), dtype=np.uint8)

    ok = _lib.Simple_glint_detect(
        img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        width,
        height,
        roi_x,
        roi_y,
        roi_w,
        roi_h,
        1 if has_pupil else 0,
        cx,
        cy,
        pr,
        glint_threshold,
        float(search_radius_factor),
        -1 if search_radius_max_px is None else int(search_radius_max_px),
        _CENTER_METHOD_CODE[glint_center_method],
        -1 if max_area_px is None else int(max_area_px),
        1 if keep_above else 0,
        1 if keep_below else 0,
        1 if keep_left else 0,
        1 if keep_right else 0,
        int(filter_margin_px),
        int(glints_target),
        1 if split_widest_for_target else 0,
        -1.0 if min_ellipse_fit_ratio is None else float(min_ellipse_fit_ratio),
        -1.0 if min_roundness_ratio is None else float(min_roundness_ratio),
        ctypes.byref(out_n),
        centers_buf,
        ellipse_buf,
        lengths_buf,
        contours_buf,
        max_glints,
        max_contour_points_per_glint,
        search_area.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    )
    if not ok:
        return {"glints": [], "search_area": search_area}

    n = out_n.value
    glints = []
    stride = 2 * max_contour_points_per_glint
    for i in range(n):
        gcx = centers_buf[2 * i]
        gcy = centers_buf[2 * i + 1]
        has_fit = ellipse_buf[6 * i + 5] != 0.0
        if has_fit:
            ellipse = (
                (ellipse_buf[6 * i + 0], ellipse_buf[6 * i + 1]),
                (ellipse_buf[6 * i + 2], ellipse_buf[6 * i + 3]),
                ellipse_buf[6 * i + 4],
            )
        else:
            ellipse = None
        n_pts = lengths_buf[i]
        flat = np.array(contours_buf[i * stride : i * stride + 2 * n_pts], dtype=np.float64).reshape(n_pts, 2)
        contour = flat.astype(np.int32).reshape(-1, 1, 2)
        glints.append({"contour": contour, "center": (round(gcx), round(gcy)), "ellipse": ellipse})

    return {"glints": glints, "search_area": search_area}
