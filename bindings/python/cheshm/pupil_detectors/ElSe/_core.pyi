from typing import Annotated

import numpy
from numpy.typing import NDArray


def detect(img: Annotated[NDArray[numpy.uint8], dict(order='C', device='cpu', writable=False)], roi_x: int, roi_y: int, roi_w: int, roi_h: int, min_area_ratio: float, max_area_ratio: float) -> object: ...

MIN_AREA_RATIO: float = 0.004999999888241291

MAX_AREA_RATIO: float = 0.20000000298023224
