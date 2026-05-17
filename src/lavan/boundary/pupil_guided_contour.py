"""Pupil-shape-guided active contour for iris boundary localization.

Extends the Daugman 2007 radial-gradient framework with anisotropic, per-
angle search bounds derived from the pupil ellipse. The pupil and limbus
are projections of (nearly) co-planar circles on the front of the eye, so
they share elongation direction and aspect ratio; using that shape as a
search-window prior keeps the radial search inside the iris and avoids
locking onto eyelash gradients at high radii.

The pupil's *position* is intentionally not used — the search is centred at
the externally supplied seed (typically the integro-differential operator
output). The limbus centre is therefore allowed to differ from the pupil
centre, which is the signal that pupil-size artifact (PSA) analysis needs.

Algorithm:

  1. Caller supplies the pupil ellipse ``((cx_p, cy_p), (2a_p, 2b_p), α_p)``.
     Only the shape ``(a_p, b_p, α_p)`` is used.
  2. For each of N angles θ, compute the pupil-ellipse radius at that
     angle: ``r_p(θ) = a_p b_p / sqrt((b_p cos(θ−α))² + (a_p sin(θ−α))²)``.
  3. Per-angle search bounds: ``r ∈ [k_min · r_p(θ), k_max · r_p(θ)]``.
  4. Radial-gradient argmax per angle (bilinear-sampled Sobel gradients,
     1-D Gaussian smoothing across radii) runs in C
     (``pupil_guided_contour_core.c``).
  5. Cyclic linear interpolation fills any missing angles, then the M-term
     Fourier truncation gives the smooth boundary R_θ.

``M = 3`` is the default: keeps the k=0 (mean radius) and k=2 (ellipse
harmonic) Fourier components plus k=1 — enough to express a centred
ellipse, not enough for the eyelash-fitting wiggle that larger M allows on
cluttered head-mounted images.
"""

import ctypes
import math
import pathlib

import cv2
import numpy as np

from daugman_derived_boundary_detectors.daugman_active_contour import _gaussian_kernel_1d

_LIB_DIR = pathlib.Path(__file__).parent
_LIB_NAME = "pupil_guided_contour_core"
_lib_ext = ".dylib" if (_LIB_DIR / f"{_LIB_NAME}.dylib").exists() else ".so"
_lib = ctypes.CDLL(str(_LIB_DIR / f"{_LIB_NAME}{_lib_ext}"))
_lib.pupil_guided_radial_search.restype = ctypes.c_int
_lib.pupil_guided_radial_search.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # gx
    ctypes.POINTER(ctypes.c_float),  # gy
    ctypes.c_int,  # h
    ctypes.c_int,  # w
    ctypes.c_float,  # cx
    ctypes.c_float,  # cy
    ctypes.c_int,  # n_angles
    ctypes.POINTER(ctypes.c_float),  # r_min[n_angles]
    ctypes.POINTER(ctypes.c_float),  # r_max[n_angles]
    ctypes.c_int,  # n_r
    ctypes.POINTER(ctypes.c_double),  # smoothing_kernel
    ctypes.c_int,  # k_len
    ctypes.POINTER(ctypes.c_double),  # r_theta_out
]


def _pupil_radial_profile(
    a_p: float,
    b_p: float,
    angle_rad: float,
    thetas: np.ndarray,
) -> np.ndarray:
    """Pupil-ellipse radius along each θ in ``thetas`` (centre-at-origin form)."""
    phi = thetas - angle_rad
    return (a_p * b_p) / np.hypot(b_p * np.cos(phi), a_p * np.sin(phi))


class PupilGuidedContour:
    """Iris boundary detector with pupil-shape-prior search bounds.

    Lifecycle::

        pgc = PupilGuidedContour(image, pupil_ellipse, N=360, M=3)
        (cx, cy), thetas, R_theta = pgc.fit(seed_center=(sx, sy),
                                            k_min=2.0, k_max=4.0)

    ``pupil_ellipse`` is the cv2.fitEllipse-style tuple
    ``((cx_p, cy_p), (2a_p, 2b_p), angle_deg)``; only the axes and angle
    are used. ``k_min`` / ``k_max`` are dimensionless scale factors against
    the pupil's per-angle radius.
    """

    def __init__(
        self,
        image: np.ndarray,
        pupil_ellipse: tuple[tuple[float, float], tuple[float, float], float],
        *,
        N: int = 360,
        M: int = 3,
        gradient_sigma: float = 1.0,
        radial_smoothing: float = 2.0,
    ) -> None:
        """Pre-compute the gradient fields + pupil radial profile; ``N`` angles, ``2M+1`` Fourier harmonics."""
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if M < 1:
            raise ValueError("M must be >= 1")
        if N < 8:
            raise ValueError("N must be >= 8")
        self.image = image
        self.N = N
        self.M = M

        # Pupil ellipse: keep shape only (axes + angle), discard centre. The
        # search will be centred at the caller-supplied seed instead, so any
        # pupil/limbus decentration is preserved as signal.
        (_, _), (pw, ph), p_angle_deg = pupil_ellipse
        self._a_p = max(pw, ph) / 2.0
        self._b_p = min(pw, ph) / 2.0
        # cv2.fitEllipse measures the angle of the *width* axis. When width
        # is the minor axis, the major-axis angle is rotated by 90°.
        self._angle_rad = math.radians(p_angle_deg if pw >= ph else p_angle_deg + 90.0)

        smoothed = cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=gradient_sigma, sigmaY=gradient_sigma)
        self._G_x = np.ascontiguousarray(cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3), dtype=np.float32)
        self._G_y = np.ascontiguousarray(cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3), dtype=np.float32)
        self._kernel = np.ascontiguousarray(_gaussian_kernel_1d(radial_smoothing), dtype=np.float64)

        self._thetas = np.linspace(0.0, 2.0 * math.pi, N, endpoint=False, dtype=np.float64)
        self._r_p = _pupil_radial_profile(self._a_p, self._b_p, self._angle_rad, self._thetas)

    def _fourier_truncate(self, r_theta: np.ndarray) -> np.ndarray:
        """Low-pass r_theta to the first M Fourier coefficients (Daugman 2007 eqs. 1-2)."""
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
        k_min: float,
        k_max: float,
    ) -> tuple[tuple[float, float], np.ndarray, np.ndarray]:
        """Fit the limbus boundary around ``seed_center``.

        ``k_min`` and ``k_max`` scale the per-angle pupil radius
        ``r_p(θ)`` to define the radial search window
        ``[k_min · r_p(θ), k_max · r_p(θ)]``. Returns
        ``((cx, cy), thetas, R_theta)``.
        """
        if not (k_max > k_min > 0):
            raise ValueError("Require 0 < k_min < k_max")
        cx, cy = float(seed_center[0]), float(seed_center[1])
        h, w = self._G_x.shape

        r_min_arr = np.ascontiguousarray(k_min * self._r_p, dtype=np.float32)
        r_max_arr = np.ascontiguousarray(k_max * self._r_p, dtype=np.float32)
        # n_r scales with the worst-case (widest) annulus so every angle has
        # at least ~1-pixel radial resolution; the C kernel reuses it.
        n_r = max(round((r_max_arr - r_min_arr).max()) + 1, 16)

        r_theta = np.full(self.N, -1.0, dtype=np.float64)
        n_valid = _lib.pupil_guided_radial_search(
            self._G_x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._G_y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(h),
            int(w),
            ctypes.c_float(cx),
            ctypes.c_float(cy),
            int(self.N),
            r_min_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            r_max_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(n_r),
            self._kernel.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(self._kernel),
            r_theta.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if n_valid < int(self.N * 0.3):
            raise ValueError(
                f"PupilGuidedContour: only {n_valid}/{self.N} angular samples produced a radial-gradient peak",
            )

        valid = r_theta >= 0
        if not valid.all():
            valid_idx = np.where(valid)[0]
            valid_vals = r_theta[valid]
            ext_idx = np.concatenate([valid_idx - self.N, valid_idx, valid_idx + self.N])
            ext_vals = np.concatenate([valid_vals, valid_vals, valid_vals])
            r_theta = np.interp(np.arange(self.N), ext_idx, ext_vals)

        R_theta = self._fourier_truncate(r_theta)
        return (cx, cy), self._thetas, R_theta
