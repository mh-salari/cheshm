from typing import Annotated

import numpy
from numpy.typing import NDArray


def fit_pupil_form(points: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu', writable=False)], harmonics: int, samples: int, iterations: int, inward_rejection: float) -> object: ...

def pupil_center(points: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu', writable=False)], method: int) -> object: ...

CENTER_CONVEX_HULL_CENTROID: int = 0

CENTER_ELLIPSE_FIT: int = 2

CENTER_MIN_AREA_RECT: int = 3

CENTER_HULL_MOMENTS: int = 4
