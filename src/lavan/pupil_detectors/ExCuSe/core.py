"""ExCuSe pupil detector — ctypes binding to the C++ kernel in ``core.dylib``.

Reference: Fuhl, W., Kübler, T., Sippel, K., Rosenstiel, W., Kasneci, E.
(2015). "ExCuSe: Robust Pupil Detection in Real-World Scenarios."
*CAIP 2015*, 39-51.

The kernel runs an adaptive-threshold-driven angular histogram to pick a
coarse pupil seed, custom Canny edge detection, ray-based contour
collection, and ellipse fitting with quality validation.
"""

import ctypes
import pathlib
import platform

import cv2
import numpy as np

# GUI metadata. Defaults / types come from `detect_pupil`'s signature.
_UI = {
    "pupil_roi": {
        "widget": "roi",
        "label": "Pupil ROI",
        "help": "Optional (x, y, w, h) rectangle. None = whole image.",
        "hidden": True,
    },
    "max_ellipse_radi": {
        "min": 5,
        "max": 1024,
        "label": "Max ellipse radius",
        "help": "Upper bound on accepted ellipse semi-axis length (pixels).",
    },
    "good_ellipse_threshold": {
        "min": 1,
        "max": 200,
        "label": "Good-ellipse threshold",
        "help": "Pixel-count threshold for the goodness test on the candidate ellipse.",
    },
}

_OVERLAYS = (
    ("ellipse", "line"),
    ("center", "point"),
)


_LIB_DIR = pathlib.Path(__file__).parent
_lib_ext = {"Darwin": ".dylib", "Linux": ".so", "Windows": ".dll"}[platform.system()]
_lib = ctypes.CDLL(str(_LIB_DIR / f"core{_lib_ext}"))
_lib.ExCuSe_detect.restype = ctypes.c_int
_lib.ExCuSe_detect.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),  # img_data
    ctypes.c_int,                    # width
    ctypes.c_int,                    # height
    ctypes.c_int,                    # roi_x
    ctypes.c_int,                    # roi_y
    ctypes.c_int,                    # roi_w  (<= 0 = no ROI)
    ctypes.c_int,                    # roi_h
    ctypes.c_int,                    # max_ellipse_radi
    ctypes.c_int,                    # good_ellipse_threshold
    ctypes.POINTER(ctypes.c_double), # out_ellipse_params[5]
]


def detect_pupil(
    img: np.ndarray,
    pupil_roi: tuple[int, int, int, int] | None = None,
    *,
    max_ellipse_radi: int = 50,
    good_ellipse_threshold: int = 15,
) -> dict | None:
    """Detect the pupil ellipse via ExCuSe (Fuhl et al. 2015).

    Returns ``{"ellipse", "center"}`` matching the rest of lavan's pupil
    detectors, or ``None`` when no ellipse could be produced:

      - ``ellipse`` — ``((cx, cy), (w, h), angle_deg)`` from the fit.
      - ``center`` — ``(cx, cy)`` rounded ints.

    ``pupil_roi=(x, y, w, h)`` runs the algorithm on the cropped
    sub-image and translates outputs back to full-image coordinates.
    """
    if img.ndim != 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.ascontiguousarray(img, dtype=np.uint8)
    height, width = img.shape

    if pupil_roi is None:
        roi_x = roi_y = roi_w = roi_h = 0
    else:
        roi_x, roi_y, roi_w, roi_h = (int(v) for v in pupil_roi)

    out_params = (ctypes.c_double * 5)()

    ok = _lib.ExCuSe_detect(
        img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        width,
        height,
        roi_x,
        roi_y,
        roi_w,
        roi_h,
        max_ellipse_radi,
        good_ellipse_threshold,
        out_params,
    )
    if not ok:
        return None

    cx, cy, w, h, angle_deg = out_params[0], out_params[1], out_params[2], out_params[3], out_params[4]
    if cx == 0.0 and cy == 0.0 and w == 0.0 and h == 0.0:
        return None
    return {
        "center": (round(cx), round(cy)),
        "ellipse": ((cx, cy), (w, h), angle_deg),
    }
