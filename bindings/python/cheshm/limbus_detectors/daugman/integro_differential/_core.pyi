from typing import Annotated

import numpy
from numpy.typing import NDArray


def detect_limbus(img: Annotated[NDArray[numpy.uint8], dict(order='C', device='cpu', writable=False)], seed_x: float, seed_y: float, r_min: int, r_max: int, range_: int, step: int) -> object: ...

R_MIN: int = 40

R_MAX: int = 62

RANGE: int = 5

STEP: int = 1
