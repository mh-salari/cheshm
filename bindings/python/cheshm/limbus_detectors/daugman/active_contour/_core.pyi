from typing import Annotated

import numpy
from numpy.typing import NDArray


def detect_limbus(img: Annotated[NDArray[numpy.uint8], dict(order='C', device='cpu', writable=False)], seed_x: float, seed_y: float, n_angles: int, m_harmonics: int, gradient_sigma: float, radial_smoothing: float, skip_eyelid_wedges: bool, r_min: float, r_max: float) -> object: ...

N: int = 360

M: int = 5

GRADIENT_SIGMA: float = 1.0

RADIAL_SMOOTHING: float = 2.0

SKIP_EYELID_WEDGES: bool = True

R_MIN: float = 30.0

R_MAX: float = 80.0
