from typing import Annotated

import numpy
from numpy.typing import NDArray


def detect(img: Annotated[NDArray[numpy.uint8], dict(order='C', device='cpu', writable=False)], roi_x: int, roi_y: int, roi_w: int, roi_h: int, radius_min: int, radius_max: int, canny_blur: float, canny_thresh1: float, canny_thresh2: float, starburst_points: int, percentage_inliers: int, inlier_iterations: int, image_aware_support: int, early_termination_percentage: int, early_rejection: int, seed: int, max_inliers: int) -> object: ...

RADIUS_MIN: int = 20

RADIUS_MAX: int = 80

CANNY_BLUR: float = 1.600000023841858

CANNY_THRESHOLD_1: float = 30.0

CANNY_THRESHOLD_2: float = 50.0

STARBURST_POINTS: int = 30

PERCENTAGE_INLIERS: int = 30

INLIER_ITERATIONS: int = 2

IMAGE_AWARE_SUPPORT: bool = True

EARLY_TERMINATION_PERCENTAGE: int = 95

EARLY_REJECTION: bool = True

SEED: int = 0

MAX_INLIERS: int = 1024
