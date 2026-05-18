"""Daugman's integro-differential operator for iris boundary localization.

Reference: Daugman, J. (2004). "How Iris Recognition Works." IEEE Trans.
Circuits and Systems for Video Technology, 14(1), 21-30, eq. (1). The same
operator was introduced in Daugman, J. (1993). "High Confidence Visual
Recognition of Persons by a Test of Statistical Independence." IEEE Trans.
PAMI, 15(11), 1148-1161.

The Python wrapper here is a thin ``ctypes`` binding around the C kernel in
``integro_differential_operator_core.c``.
"""

import ctypes
import pathlib
import platform

import cv2
import numpy as np

# Equivalent of skimage.morphology.disk(3); avoids the skimage dependency.
_DISK3 = np.array(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ],
    dtype=np.uint8,
)

# 1-D Gaussian kernel used by the C kernel to smooth the radius-integral
# before its derivative (size 3, sigma 1, normalised).
_GAUSSIAN_KERNEL_1D = np.array([0.27406862, 0.45186276, 0.27406862], dtype=np.float64)

_LIB_DIR = pathlib.Path(__file__).parent
_LIB_NAME = "integro_differential_operator_core"
_lib_ext = {"Darwin": ".dylib", "Linux": ".so", "Windows": ".dll"}[platform.system()]
_lib = ctypes.CDLL(str(_LIB_DIR / f"{_LIB_NAME}{_lib_ext}"))
_lib.integro_differential_operator_search.restype = ctypes.c_int
_lib.integro_differential_operator_search.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
]


def _circle_perimeter(x: int, y: int, r: int, half: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Bresenham circle perimeter (Python fallback used by ``result()`` to draw)."""
    aa, bb = [], []
    dp = 3 - 2 * r
    a, b = 0, r
    while a <= b:
        if half:
            aa.extend([a, a, -a, -a])
            bb.extend([b, -b, b, -b])
        else:
            aa.extend([a, -a, a, -a, b, -b, b, -b])
            bb.extend([b, b, -b, -b, a, a, -a, -a])
        if dp > 0:
            b -= 1
            dp += 4 * (a - b) + 10
        else:
            dp += 4 * a + 6
        a += 1
    return x + np.array(aa, dtype=np.int16), y + np.array(bb, dtype=np.int16)


class IntegroDifferentialOperator:
    """Daugman's integro-differential operator (1993 / 2004) wrapping a C kernel.

    For each candidate ``(x_0, y_0, r)`` the operator integrates image
    intensity around the circle, takes a Gaussian-smoothed derivative with
    respect to ``r``, and selects the ``(x_0, y_0, r*)`` that maximises that
    derivative magnitude (paper eq. 1).
    """

    def __init__(self, image: np.ndarray, r_min: int = 40, r_max: int = 62, c_type: str = "half") -> None:
        """Convert ``image`` to grayscale, apply the morphological open, store params."""
        if image.ndim == 3:
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            self.image = image
        self.r_min, self.r_max = r_min, r_max
        self.image_open = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, _DISK3)
        self.c_type = c_type

    def search(self, cen_x: int, cen_y: int, range_: int, step: int) -> np.ndarray:
        """Sweep centres over a ``(±range_, step)`` grid; return responses sorted by score."""
        img = np.ascontiguousarray(self.image_open)
        h, w = img.shape
        grid_size = (2 * range_ // step + 1) ** 2
        out = np.empty((grid_size, 4), dtype=np.float64)

        n = _lib.integro_differential_operator_search(
            img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            h,
            w,
            cen_x,
            cen_y,
            range_,
            step,
            self.r_min,
            self.r_max,
            _GAUSSIAN_KERNEL_1D.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(_GAUSSIAN_KERNEL_1D),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        results = out[:n]
        return results[results[:, 2].argsort()].astype(np.int16)

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        """Run a coarse-then-fine search and return the best iris circle + the marked image."""
        range_ = int(np.min([int(s / 2) for s in self.image.shape]) * 0.40)
        cen_x, cen_y = [int(s / 2) for s in self.image.shape]

        fuzzy_max = self.search(cen_x, cen_y, range_, 3)[-1]
        cen_x, cen_y = fuzzy_max[0], fuzzy_max[1]

        info = np.array([])
        for i in np.arange(2):
            try:
                circle_max = self.search(cen_x, cen_y, 2 + i, 1)[-1]
                circle = _circle_perimeter(circle_max[0], circle_max[1], circle_max[3], False)
                self.image[circle] = 255
            except Exception:
                info = np.append(info, [0, 0, 0, 0])
                return self.image, info

            if i == 0:
                info = np.append(info, circle_max)
                self.r_min, self.r_max = int(np.ceil(0.1 * circle_max[3])), int(np.ceil(0.8 * circle_max[3]))

        return self.image, info
