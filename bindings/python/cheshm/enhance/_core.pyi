import numpy as np

CLAHE_CLIP_LIMIT: float
CLAHE_TILE: int
STRETCH_LO_PCT: float
STRETCH_HI_PCT: float
GAMMA: float
BILATERAL_D: int
BILATERAL_SIGMA_COLOR: float
BILATERAL_SIGMA_SPACE: float
UNSHARP_SIGMA: float
UNSHARP_AMOUNT: float

def clahe(img: np.ndarray, clip_limit: float = ..., tile: int = ...) -> np.ndarray: ...
def percentile_stretch(img: np.ndarray, lo_pct: float = ..., hi_pct: float = ...) -> np.ndarray: ...
def gamma(img: np.ndarray, g: float = ...) -> np.ndarray: ...
def bilateral(img: np.ndarray, d: int = ..., sigma_color: float = ..., sigma_space: float = ...) -> np.ndarray: ...
def unsharp(img: np.ndarray, sigma: float = ..., amount: float = ...) -> np.ndarray: ...
