"""Internal helpers shared between the Daugman-derived limbus detectors."""

import cv2
import numpy as np


def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    """Odd-length 1-D Gaussian kernel sized to roughly ±3σ."""
    if sigma <= 0:
        return np.array([1.0], dtype=np.float64)
    k = max(round(sigma * 6) | 1, 3)
    return cv2.getGaussianKernel(k, sigma).ravel().astype(np.float64)
