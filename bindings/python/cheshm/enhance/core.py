"""Image-enhancement preprocessing (CLAHE, percentile-stretch, gamma, bilateral, unsharp)."""

from typing import Literal

import numpy as np

from . import _core

Method = Literal["clahe", "percentile_stretch", "gamma", "bilateral", "unsharp"]


def _u8(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img, dtype=np.uint8)


def clahe(img: np.ndarray, clip_limit: float = _core.CLAHE_CLIP_LIMIT, tile: int = _core.CLAHE_TILE) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization (local contrast)."""
    return _core.clahe(_u8(img), float(clip_limit), int(tile))


def percentile_stretch(
    img: np.ndarray, lo_pct: float = _core.STRETCH_LO_PCT, hi_pct: float = _core.STRETCH_HI_PCT
) -> np.ndarray:
    """Linear contrast stretch between the lo/hi intensity percentiles."""
    return _core.percentile_stretch(_u8(img), float(lo_pct), float(hi_pct))


def gamma(img: np.ndarray, g: float = _core.GAMMA) -> np.ndarray:
    """Power-law mapping; g < 1 brightens shadows, g > 1 darkens."""
    return _core.gamma(_u8(img), float(g))


def bilateral(
    img: np.ndarray,
    d: int = _core.BILATERAL_D,
    sigma_color: float = _core.BILATERAL_SIGMA_COLOR,
    sigma_space: float = _core.BILATERAL_SIGMA_SPACE,
) -> np.ndarray:
    """Edge-preserving denoise."""
    return _core.bilateral(_u8(img), int(d), float(sigma_color), float(sigma_space))


def unsharp(img: np.ndarray, sigma: float = _core.UNSHARP_SIGMA, amount: float = _core.UNSHARP_AMOUNT) -> np.ndarray:
    """Unsharp-mask edge sharpening."""
    return _core.unsharp(_u8(img), float(sigma), float(amount))


_DISPATCH = {
    "clahe": clahe,
    "percentile_stretch": percentile_stretch,
    "gamma": gamma,
    "bilateral": bilateral,
    "unsharp": unsharp,
}


def apply(img: np.ndarray, method: Method, **params: float) -> np.ndarray:
    """Apply one enhancement by name; ``params`` override that method's defaults."""
    if method not in _DISPATCH:
        raise ValueError(f"unknown method {method!r}; valid: {sorted(_DISPATCH)}")
    return _DISPATCH[method](img, **params)
