from typing import Annotated

import numpy
from numpy.typing import NDArray


def detect(img: Annotated[NDArray[numpy.uint8], dict(order='C', device='cpu', writable=False)], roi_x: int, roi_y: int, roi_w: int, roi_h: int, pupil_threshold: int, pupil_center_method: int, min_ellipse_fit_ratio: float, min_roundness_ratio: float, max_contour_points: int) -> object: ...

PUPIL_THRESHOLD: int = 30

MAX_CONTOUR_POINTS: int = 4096
