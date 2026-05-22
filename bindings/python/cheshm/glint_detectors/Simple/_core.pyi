from typing import Annotated

import numpy
from numpy.typing import NDArray


def detect(img: Annotated[NDArray[numpy.uint8], dict(order='C', device='cpu', writable=False)], roi_x: int, roi_y: int, roi_w: int, roi_h: int, has_pupil: int, pupil_cx: float, pupil_cy: float, pupil_radius: float, glint_threshold: int, search_radius_factor: float, search_radius_max_px: int, glint_center_method: int, max_area_px: int, keep_above: int, keep_below: int, keep_left: int, keep_right: int, filter_margin_px: int, glints_target: int, split_widest_for_target: int, min_ellipse_fit_ratio: float, min_roundness_ratio: float) -> object: ...

GLINT_THRESHOLD: int = 240

SEARCH_RADIUS_FACTOR: float = 2.0

FILTER_MARGIN_PX: int = 5

GLINTS_TARGET: int = 1

KEEP_ABOVE: bool = True

KEEP_BELOW: bool = True

KEEP_LEFT: bool = True

KEEP_RIGHT: bool = True

SPLIT_WIDEST_FOR_TARGET: bool = False
