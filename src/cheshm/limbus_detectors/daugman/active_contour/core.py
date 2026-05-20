"""Daugman 2007 Fourier-series active contour for iris boundary localization.

Reference: Daugman, J. (2007). "New Methods in Iris Recognition." IEEE Trans.
Systems, Man, and Cybernetics, Part B, 37(5), 1167-1175. Section II
("Active Contours and Generalized Coordinates"), equations (1) and (2).

Algorithm:

  1. A seed centre is supplied externally (typically from
     ``IntegroDifferentialOperator``).
  2. The image is Gaussian-blurred (``gradient_sigma``) and Sobel gradients
     ``G_x``, ``G_y`` are computed once.
  3. For each of ``N`` angular samples ``theta`` around the seed, the radial-
     direction gradient ``g_r(r, theta) = G_x cos theta + G_y sin theta`` is
     sampled at ``n_r`` radii in ``[r_min, r_max]`` (bilinear interpolation),
     smoothed across the radial coordinate with a 1-D Gaussian
     (``radial_smoothing``), and ``r_theta`` is the radius of maximum
     smoothed gradient. The inner loop runs in C (``core.c``).
  4. The discrete Fourier transform of ``{r_theta}`` (paper eq. 1) is
     truncated to ``M`` complex coefficients with monotonically-decreasing
     weights. The inverse DFT (paper eq. 2) gives the smooth, possibly non-
     circular boundary ``{R_theta}``.

Daugman recommends ``M = 17`` for the pupil (strong data) and ``M = 5`` for
the iris outer boundary (weaker data, more smoothing). The paper handles
eyelid occlusion per-image with spline contours (Section III); we provide
an optional fixed angular wedge for the upper lid as a pragmatic
approximation suitable for head-mounted eye-tracker geometry.
"""

import ctypes
import math
import pathlib
import platform

import cv2
import numpy as np

from .._common import _gaussian_kernel_1d

# Overlays this detector produces.
_OVERLAYS = (
    ("curve", "line"),
    ("center", "point"),
    ("mask", "fill"),
)

# GUI metadata. Defaults / types come from `detect_limbus`'s signature;
# this dict carries slider bounds and per-param help only.
_UI = {
    "N": {
        "min": 8,
        "max": 1440,
        "label": "Angular samples (N)",
        "help": "Number of angles θ around the seed (paper notation: N).",
    },
    "M": {
        "min": 1,
        "max": 32,
        "label": "Fourier harmonics (M)",
        "help": "Number of Fourier coefficients kept (paper notation: M). Daugman 2007 recommends 5 for iris, 17 for pupil.",
    },
    "gradient_sigma": {
        "min": 0.0,
        "max": 10.0,
        "help": "Gaussian σ (px) for pre-gradient blur. 0 = no blur.",
    },
    "radial_smoothing": {
        "min": 0.0,
        "max": 10.0,
        "help": "Gaussian σ (radial samples) applied to the radial-gradient profile before argmax.",
    },
    "skip_eyelid_wedges": {
        "label": "Skip eyelid wedges",
        "help": "Mask out the upper-lid angular wedge (240°–300° in image coords) during the radial search.",
    },
    "r_min": {
        "min": 1.0,
        "max": 1024.0,
        "help": "Lower bound on iris radius (pixels).",
    },
    "r_max": {
        "min": 1.0,
        "max": 1024.0,
        "help": "Upper bound on iris radius (pixels).",
    },
}

# Image-coordinate angle range (y-down) typically occupied by the upper
# eyelid in head-on eye-tracker frames; skipped during the radial search
# and filled by cyclic linear interpolation before the FFT.
_EYELID_TOP_DEG = set(range(240, 301))

_LIB_DIR = pathlib.Path(__file__).parent
_LIB_NAME = "core"
_lib_ext = {"Darwin": ".dylib", "Linux": ".so", "Windows": ".dll"}[platform.system()]
_lib = ctypes.CDLL(str(_LIB_DIR / f"{_LIB_NAME}{_lib_ext}"))
_lib.active_contour_radial_search.restype = ctypes.c_int
_lib.active_contour_radial_search.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # gx
    ctypes.POINTER(ctypes.c_float),  # gy
    ctypes.c_int,  # h
    ctypes.c_int,  # w
    ctypes.c_float,  # cx
    ctypes.c_float,  # cy
    ctypes.c_int,  # n_angles
    ctypes.c_float,  # r_min
    ctypes.c_float,  # r_max
    ctypes.c_int,  # n_r
    ctypes.POINTER(ctypes.c_ubyte),  # eyelid_mask
    ctypes.POINTER(ctypes.c_double),  # smoothing_kernel
    ctypes.c_int,  # k_len
    ctypes.POINTER(ctypes.c_double),  # r_theta_out
]


class DaugmanActiveContour:
    """Daugman 2007 active-contour iris boundary detector.

    Lifecycle::

        dac = DaugmanActiveContour(image, N=360, M=5)
        (cx, cy), thetas, R_theta = dac.fit(seed_center=(sx, sy),
                                            r_min=r_lo, r_max=r_hi)

    Boundary points are at ``(cx + R_theta cos theta, cy + R_theta sin theta)``.
    """

    def __init__(
        self,
        image: np.ndarray,
        *,
        N: int = 360,
        M: int = 5,
        gradient_sigma: float = 1.0,
        radial_smoothing: float = 2.0,
        skip_eyelid_wedges: bool = True,
    ) -> None:
        """Pre-compute the gradient fields for ``image``; ``N`` angles, ``2M+1`` Fourier harmonics."""
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if M < 1:
            raise ValueError("M must be >= 1")
        if N < 8:
            raise ValueError("N must be >= 8")
        self.image = image
        self.N = N
        self.M = M
        self.skip_eyelid_wedges = skip_eyelid_wedges

        # Sobel on a Gaussian-blurred copy is the standard noise-resilient
        # gradient pre-computation; both components are needed every angle
        # for the radial projection g_r = G_x cos θ + G_y sin θ.
        smoothed = cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=gradient_sigma, sigmaY=gradient_sigma)
        self._G_x = np.ascontiguousarray(cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3), dtype=np.float32)
        self._G_y = np.ascontiguousarray(cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3), dtype=np.float32)
        self._kernel = np.ascontiguousarray(_gaussian_kernel_1d(radial_smoothing), dtype=np.float64)

        mask = np.zeros(N, dtype=np.uint8)
        if skip_eyelid_wedges:
            for i in range(N):
                deg = round(360.0 * i / N) % 360
                if deg in _EYELID_TOP_DEG:
                    mask[i] = 1
        self._eyelid_mask = np.ascontiguousarray(mask)

    def _fourier_truncate(self, r_theta: np.ndarray) -> np.ndarray:
        """Low-pass r_theta to the first M Fourier coefficients (paper eqs. 1-2).

        The weights ``1 / (1 + k/M)`` give a monotone decay across kept
        harmonics, matching Daugman's recipe for the active-contour boundary.
        """
        N = self.N
        coeffs = np.fft.fft(r_theta)
        mask = np.zeros(N, dtype=bool)
        mask[: self.M] = True
        if self.M > 1:
            mask[-(self.M - 1) :] = True
        weights = np.zeros(N, dtype=np.float64)
        for k in range(self.M):
            w = 1.0 / (1.0 + k / self.M)
            weights[k] = w
            if k > 0:
                weights[N - k] = w
        coeffs *= weights
        coeffs[~mask] = 0
        return np.real(np.fft.ifft(coeffs))

    def fit(
        self,
        seed_center: tuple[float, float],
        *,
        r_min: float,
        r_max: float,
    ) -> tuple[tuple[float, float], np.ndarray, np.ndarray]:
        """Fit the boundary around ``seed_center``.

        Returns ``((cx, cy), thetas, R_theta)``. ``cx, cy`` are the seed (the
        active contour parameterises radii from a fixed centre, so the centre
        does not move during the fit).
        """
        if r_max <= r_min:
            raise ValueError("r_max must be > r_min")
        cx, cy = float(seed_center[0]), float(seed_center[1])
        h, w = self._G_x.shape
        # One radial sample per ~pixel of search range; clamped so very small
        # ranges still get enough resolution for the Gaussian smoothing.
        n_r = max(round(r_max - r_min) + 1, 16)

        r_theta = np.full(self.N, -1.0, dtype=np.float64)
        n_valid = _lib.active_contour_radial_search(
            self._G_x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._G_y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(h),
            int(w),
            ctypes.c_float(cx),
            ctypes.c_float(cy),
            int(self.N),
            ctypes.c_float(float(r_min)),
            ctypes.c_float(float(r_max)),
            int(n_r),
            self._eyelid_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            self._kernel.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(self._kernel),
            r_theta.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if n_valid < int(self.N * 0.3):
            raise ValueError(
                f"DaugmanActiveContour: only {n_valid}/{self.N} angular samples produced a radial-gradient peak",
            )

        thetas = np.linspace(0.0, 2.0 * math.pi, self.N, endpoint=False, dtype=np.float64)

        # Masked / failed angles get filled cyclically so the FFT input is a
        # regular ring of length N. The Fourier truncation below smooths
        # away any kinks the linear fill introduces.
        valid = r_theta >= 0
        if not valid.all():
            valid_idx = np.where(valid)[0]
            valid_vals = r_theta[valid]
            ext_idx = np.concatenate([valid_idx - self.N, valid_idx, valid_idx + self.N])
            ext_vals = np.concatenate([valid_vals, valid_vals, valid_vals])
            r_theta = np.interp(np.arange(self.N), ext_idx, ext_vals)

        R_theta = self._fourier_truncate(r_theta)
        return (cx, cy), thetas, R_theta


def detect_limbus(
    img: np.ndarray,
    seed_center: tuple[float, float],
    *,
    N: int = 360,
    M: int = 5,
    gradient_sigma: float = 1.0,
    radial_smoothing: float = 2.0,
    skip_eyelid_wedges: bool = True,
    r_min: float = 30.0,
    r_max: float = 80.0,
) -> dict:
    """One-shot Daugman 2007 active-contour limbus fit around ``seed_center``.

    Builds a :class:`DaugmanActiveContour` for ``img`` and runs a single
    radial-gradient search + Fourier truncation. Returns
    ``{"center": (cx, cy), "thetas": np.ndarray, "R_theta": np.ndarray}``.
    Boundary points are at ``(cx + R_theta cos θ, cy + R_theta sin θ)``.
    """
    op = DaugmanActiveContour(
        img,
        N=N,
        M=M,
        gradient_sigma=gradient_sigma,
        radial_smoothing=radial_smoothing,
        skip_eyelid_wedges=skip_eyelid_wedges,
    )
    (cx, cy), thetas, R_theta = op.fit(seed_center, r_min=r_min, r_max=r_max)
    return {"center": (cx, cy), "thetas": thetas, "R_theta": R_theta}
