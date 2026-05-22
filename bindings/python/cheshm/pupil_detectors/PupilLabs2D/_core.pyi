from typing import Annotated

import numpy
from numpy.typing import NDArray


def detect(img: Annotated[NDArray[numpy.uint8], dict(order='C', device='cpu', writable=False)], roi_x: int, roi_y: int, roi_w: int, roi_h: int, intensity_range: int, blur_size: int, canny_threshold: float, canny_ratio: float, canny_aperture: int, pupil_size_max: int, pupil_size_min: int, contour_size_min: int, ellipse_roundness_ratio: float, initial_ellipse_fit_threshold: float, ellipse_true_support_min_dist: float, support_pixel_ratio_exponent: float, coarse_detection: bool, coarse_filter_min: int, coarse_filter_max: int) -> object: ...

INTENSITY_RANGE: int = 23

BLUR_SIZE: int = 5

CANNY_THRESHOLD: float = 160.0

CANNY_RATIO: float = 2.0

CANNY_APERTURE: int = 5

PUPIL_SIZE_MAX: int = 100

PUPIL_SIZE_MIN: int = 10

CONTOUR_SIZE_MIN: int = 5

ELLIPSE_ROUNDNESS_RATIO: float = 0.10000000149011612

INITIAL_ELLIPSE_FIT_THRESHOLD: float = 1.7999999523162842

ELLIPSE_TRUE_SUPPORT_MIN_DIST: float = 2.5

SUPPORT_PIXEL_RATIO_EXPONENT: float = 2.0

COARSE_DETECTION: bool = True

COARSE_FILTER_MIN: int = 128

COARSE_FILTER_MAX: int = 280
