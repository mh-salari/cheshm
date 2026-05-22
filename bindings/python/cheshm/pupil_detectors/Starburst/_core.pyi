from typing import Annotated

import numpy
from numpy.typing import NDArray


def detect(img: Annotated[NDArray[numpy.uint8], dict(order='C', device='cpu', writable=False)], roi_x: int, roi_y: int, roi_w: int, roi_h: int, use_auto_seed: bool, seed_threshold: int, seed_x: float, seed_y: float, edge_threshold: int, rays: int, min_feature_candidates: int, cr_window_size: int, cr_ratio_to_image_height: int, max_edge_points: int) -> object: ...

EDGE_THRESHOLD: int = 16

RAYS: int = 18

MIN_FEATURE_CANDIDATES: int = 10

CR_WINDOW_SIZE: int = 301

CR_RATIO_TO_IMAGE_HEIGHT: int = 2

MAX_EDGE_POINTS: int = 1024

SEED_THRESHOLD: int = 30
