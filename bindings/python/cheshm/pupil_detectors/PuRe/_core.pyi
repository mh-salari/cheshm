from typing import Annotated

import numpy
from numpy.typing import NDArray


def detect(img: Annotated[NDArray[numpy.uint8], dict(order='C', device='cpu', writable=False)], roi_x: int, roi_y: int, roi_w: int, roi_h: int, min_pupil_diameter_mm: float, max_pupil_diameter_mm: float, canthi_distance_mm: float, outline_bias: int) -> object: ...

MIN_PUPIL_DIAMETER_MM: float = 2.0

MAX_PUPIL_DIAMETER_MM: float = 8.0

CANTHI_DISTANCE_MM: float = 27.600000381469727

OUTLINE_BIAS: int = 5
