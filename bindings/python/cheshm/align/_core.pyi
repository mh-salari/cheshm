from collections.abc import Sequence
from typing import Annotated

import numpy
from numpy.typing import NDArray


def make_iris_mask(height: int, width: int, lcx: float, lcy: float, limbus_r: float, pupil_r: float, exclude_top: float, exclude_bottom: float, inner_margin: float) -> object: ...

def make_barrel_mask(height: int, width: int, lcx: float, lcy: float, limbus_r: float, pupil_r: float, exclude_top: float, exclude_bottom: float, inner_margin: float) -> object: ...

def align_by_translation(ref_x: float, ref_y: float, mov_x: float, mov_y: float) -> object: ...

def apply_transform(img: Annotated[NDArray[numpy.uint8], dict(order='C', device='cpu', writable=False)], dx: float, dy: float, theta: float, center_x: float | None, center_y: float | None) -> object: ...

def align_by_min_diff(img_ref: Annotated[NDArray[numpy.uint8], dict(order='C', device='cpu', writable=False)], img_mov: Annotated[NDArray[numpy.uint8], dict(order='C', device='cpu', writable=False)], mask: Annotated[NDArray[numpy.uint8], dict(order='C', device='cpu', writable=False)], dx_lo: int, dx_hi: int, dy_lo: int, dy_hi: int, rot_start: float, rot_end: float, rot_step: float, center_x: float | None, center_y: float | None) -> object: ...

def align_eye_images(ref_img: Annotated[NDArray[numpy.uint8], dict(order='C', device='cpu', writable=False)], tgt_img: Annotated[NDArray[numpy.uint8], dict(order='C', device='cpu', writable=False)], ref_pupil_cx: float, ref_pupil_cy: float, ref_pupil_radius: float, ref_glints: Sequence[tuple[float, float]], ref_limbus_cx: float, ref_limbus_cy: float, ref_limbus_radius: float, tgt_pupil_cx: float, tgt_pupil_cy: float, tgt_pupil_radius: float, tgt_glints: Sequence[tuple[float, float]], tgt_limbus_cx: float, tgt_limbus_cy: float, tgt_limbus_radius: float, step1_code: int, step2: bool, exclude_top: float = 60.0, exclude_bottom: float = 45.0, inner_margin: float = 15.0) -> object: ...

def match_glints(reference: Sequence[tuple[float, float]], moving: Sequence[tuple[float, float]], tol_fraction: float = 0.5) -> object: ...

DX_LO: int = -10

DX_HI: int = 11

DY_LO: int = -10

DY_HI: int = 11

ROT_START: float = -2.0

ROT_END: float = 2.0

ROT_STEP: float = 0.05

EXCLUDE_TOP: float = 60.0

EXCLUDE_BOTTOM: float = 45.0

INNER_MARGIN: float = 15.0

GLINT_MATCH_TOL_FRACTION: float = 0.5
