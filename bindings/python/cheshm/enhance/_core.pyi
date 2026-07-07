from typing import Annotated

import numpy
from numpy.typing import NDArray


def clahe(img: Annotated[NDArray[numpy.uint8], dict(order='C', device='cpu', writable=False)], clip_limit: float = 2.0, tile: int = 8) -> object: ...

def percentile_stretch(img: Annotated[NDArray[numpy.uint8], dict(order='C', device='cpu', writable=False)], lo_pct: float = 1.0, hi_pct: float = 99.0) -> object: ...

def gamma(img: Annotated[NDArray[numpy.uint8], dict(order='C', device='cpu', writable=False)], g: float = 1.0) -> object: ...

def bilateral(img: Annotated[NDArray[numpy.uint8], dict(order='C', device='cpu', writable=False)], d: int = 5, sigma_color: float = 50.0, sigma_space: float = 50.0) -> object: ...

def unsharp(img: Annotated[NDArray[numpy.uint8], dict(order='C', device='cpu', writable=False)], sigma: float = 1.0, amount: float = 1.0) -> object: ...

CLAHE_CLIP_LIMIT: float = 2.0

CLAHE_TILE: int = 8

STRETCH_LO_PCT: float = 1.0

STRETCH_HI_PCT: float = 99.0

GAMMA: float = 1.0

BILATERAL_D: int = 5

BILATERAL_SIGMA_COLOR: float = 50.0

BILATERAL_SIGMA_SPACE: float = 50.0

UNSHARP_SIGMA: float = 1.0

UNSHARP_AMOUNT: float = 1.0
