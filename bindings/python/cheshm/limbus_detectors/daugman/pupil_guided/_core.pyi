from typing import Annotated

import numpy
from numpy.typing import NDArray


def detect_limbus(img: Annotated[NDArray[numpy.uint8], dict(order='C', device='cpu', writable=False)], seed_x: float, seed_y: float, pupil_cx: float, pupil_cy: float, pupil_w: float, pupil_h: float, pupil_angle_deg: float, n_angles: int, m_harmonics: int, gradient_sigma: float, radial_smoothing: float, k_min: float, k_max: float) -> object: ...

N: int = 360

M: int = 3

GRADIENT_SIGMA: float = 1.0

RADIAL_SMOOTHING: float = 2.0

K_MIN: float = 2.0

K_MAX: float = 4.0
