from collections.abc import Sequence
from typing import Annotated

import numpy
from numpy.typing import NDArray


def save_diff_heatmap(out_path: str, ref: Annotated[NDArray[numpy.uint8], dict(order='C', device='cpu', writable=False)], aligned: Annotated[NDArray[numpy.uint8], dict(order='C', device='cpu', writable=False)], vmax: float) -> float: ...

def save_alignment_overlay(out_path: str, ref_img: Annotated[NDArray[numpy.uint8], dict(order='C', device='cpu', writable=False)], aligned: Annotated[NDArray[numpy.uint8], dict(order='C', device='cpu', writable=False)], ref_weight: float, label: str | None) -> None: ...

def save_alignment_comparison(out_path: str, ref_img: Annotated[NDArray[numpy.uint8], dict(order='C', device='cpu', writable=False)], target_img: Annotated[NDArray[numpy.uint8], dict(order='C', device='cpu', writable=False)], aligned: Annotated[NDArray[numpy.uint8], dict(order='C', device='cpu', writable=False)], ref_label: str, target_label: str, vmax: float) -> None: ...

def save_detection_overlay(out_path: str, img: Annotated[NDArray[numpy.uint8], dict(order='C', device='cpu', writable=False)], pupil_contour: Annotated[NDArray[numpy.int32], dict(order='C', device='cpu', writable=False)] | None, pupil_ellipse: tuple[float, float, float, float, float] | None, pupil_center: tuple[float, float] | None, pupil_mask: Annotated[NDArray[numpy.uint8], dict(order='C', device='cpu', writable=False)] | None, glints: Sequence[tuple[Annotated[NDArray[numpy.int32], dict(order='C', device='cpu', writable=False)] | None, tuple[float, float, float, float, float] | None, tuple[float, float] | None]], limbus_curve: Annotated[NDArray[numpy.int32], dict(order='C', device='cpu', writable=False)] | None, limbus_center: tuple[float, float] | None, pupil_contour_style: tuple[bool, tuple[float, float, float], int, float] | None, pupil_ellipse_style: tuple[bool, tuple[float, float, float], int, float] | None, pupil_center_style: tuple[bool, tuple[float, float, float], int, float] | None, pupil_mask_style: tuple[bool, tuple[float, float, float], int, float] | None, glint_contour_style: tuple[bool, tuple[float, float, float], int, float] | None, glint_ellipse_style: tuple[bool, tuple[float, float, float], int, float] | None, glint_center_style: tuple[bool, tuple[float, float, float], int, float] | None, limbus_curve_style: tuple[bool, tuple[float, float, float], int, float] | None, limbus_center_style: tuple[bool, tuple[float, float, float], int, float] | None, label: str | None) -> None: ...

ALIGNMENT_OVERLAY_REF_WEIGHT: float = 0.5
