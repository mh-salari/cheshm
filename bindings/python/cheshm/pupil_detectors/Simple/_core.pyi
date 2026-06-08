from typing import Annotated

import numpy
from numpy.typing import NDArray


def detect(img: Annotated[NDArray[numpy.uint8], dict(order='C', device='cpu', writable=False)], roi_x: int, roi_y: int, roi_w: int, roi_h: int, pupil_threshold: int, pupil_center_method: int, min_ellipse_fit_ratio: float, min_roundness_ratio: float, fourier_smoothing: bool, fourier_harmonics: int, fourier_samples: int, fourier_iterations: int, fourier_inward_rejection: float, glint_merge: bool, glint_threshold: int, glint_boost_pct: float, glint_reach_px: int, max_contour_points: int) -> object: ...

PUPIL_THRESHOLD: int = 30

MAX_CONTOUR_POINTS: int = 4096

FOURIER_SMOOTHING: bool = True

FOURIER_HARMONICS: int = 5

FOURIER_SAMPLES: int = 360

FOURIER_ITERATIONS: int = 4

FOURIER_INWARD_REJECTION: float = 1.0

GLINT_MERGE: bool = False

GLINT_THRESHOLD: int = 230

GLINT_BOOST_PCT: float = 25.0

GLINT_REACH_PX: int = 12
