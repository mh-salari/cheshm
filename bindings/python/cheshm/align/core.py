"""Iris-based rigid alignment of two eye images."""

from typing import Literal

import numpy as np

from . import _core

Step1Method = Literal["glint", "pupil"] | None

_STEP1_CODE = {None: 0, "glint": 1, "pupil": 2}


def make_iris_mask(
    img_shape: tuple[int, ...],
    limbus_center: tuple[float, float],
    limbus_r: float,
    pupil_r: float,
    exclude_top: float = _core.EXCLUDE_TOP,
    exclude_bottom: float = _core.EXCLUDE_BOTTOM,
    inner_margin: float = _core.INNER_MARGIN,
) -> np.ndarray:
    """Annular ring mask covering the iris texture, excluding eyelid zones."""
    return _core.make_iris_mask(
        int(img_shape[0]),
        int(img_shape[1]),
        float(limbus_center[0]),
        float(limbus_center[1]),
        float(limbus_r),
        float(pupil_r),
        float(exclude_top),
        float(exclude_bottom),
        float(inner_margin),
    )


def make_barrel_mask(
    img_shape: tuple[int, ...],
    limbus_center: tuple[float, float],
    limbus_r: float,
    pupil_r: float,
    exclude_top: float = _core.EXCLUDE_TOP,
    exclude_bottom: float = _core.EXCLUDE_BOTTOM,
    inner_margin: float = _core.INNER_MARGIN,
) -> np.ndarray:
    """Filled barrel mask: iris ring + per-row gap fill."""
    return _core.make_barrel_mask(
        int(img_shape[0]),
        int(img_shape[1]),
        float(limbus_center[0]),
        float(limbus_center[1]),
        float(limbus_r),
        float(pupil_r),
        float(exclude_top),
        float(exclude_bottom),
        float(inner_margin),
    )


def align_by_translation(
    ref_point: tuple[float, float],
    mov_point: tuple[float, float],
) -> tuple[float, float, float]:
    """Return ``(dx, dy, 0.0)`` so ``mov_point`` lands on ``ref_point``."""
    return _core.align_by_translation(
        float(ref_point[0]), float(ref_point[1]), float(mov_point[0]), float(mov_point[1])
    )


def apply_transform(
    img: np.ndarray,
    params: tuple[float, float, float],
    center: tuple[float, float] | None = None,
) -> np.ndarray:
    """Apply ``(dx, dy, theta)`` rigid transform around ``center``."""
    cx = float(center[0]) if center is not None else None
    cy = float(center[1]) if center is not None else None
    return _core.apply_transform(
        np.ascontiguousarray(img, dtype=np.uint8), float(params[0]), float(params[1]), float(params[2]), cx, cy
    )


def align_by_min_diff(
    img_ref: np.ndarray,
    img_mov: np.ndarray,
    mask: np.ndarray,
    dx_range: tuple[int, int] = (_core.DX_LO, _core.DX_HI),
    dy_range: tuple[int, int] = (_core.DY_LO, _core.DY_HI),
    rot_range: tuple[float, float, float] = (_core.ROT_START, _core.ROT_END, _core.ROT_STEP),
    rotation_center: tuple[float, float] | None = None,
) -> tuple[tuple[float, float, float], float]:
    """Coarse + fine grid search for the rigid transform that minimises iris-MAE."""
    cx = float(rotation_center[0]) if rotation_center is not None else None
    cy = float(rotation_center[1]) if rotation_center is not None else None
    return _core.align_by_min_diff(
        np.ascontiguousarray(img_ref, dtype=np.uint8),
        np.ascontiguousarray(img_mov, dtype=np.uint8),
        np.ascontiguousarray(mask, dtype=np.uint8),
        int(dx_range[0]),
        int(dx_range[1]),
        int(dy_range[0]),
        int(dy_range[1]),
        float(rot_range[0]),
        float(rot_range[1]),
        float(rot_range[2]),
        cx,
        cy,
    )


def _pupil_radius(detection: dict) -> float:
    _, (w, h), _ = detection["pupil_ellipse"]
    return max(float(w), float(h)) / 2.0


def align_eye_images(
    ref_img: np.ndarray,
    tgt_img: np.ndarray,
    ref_det: dict,
    tgt_det: dict,
    *,
    step1: Step1Method = "glint",
    step2: bool = True,
    exclude_top: float = _core.EXCLUDE_TOP,
    exclude_bottom: float = _core.EXCLUDE_BOTTOM,
    inner_margin: float = _core.INNER_MARGIN,
) -> dict:
    """Rigid-align ``tgt_img`` onto ``ref_img`` using cached detections.

    Step 1 (translation): glint-centroid match, pupil-centre match, or off.
    Step 2 (refinement): iris-barrel min-MAE search over ``(dx, dy, theta)``.

    ``exclude_top`` / ``exclude_bottom`` set the half-angle (degrees) of the
    eyelid wedges cut from the step-2 barrel mask, and ``inner_margin`` the
    pupil-edge gap; widen the wedges to keep eyelashes out of the rotation
    search when the lid/lashes occlude the iris.
    """
    if step1 is None and not step2:
        return {
            "aligned": tgt_img.copy(),
            "step1_translation": None,
            "step2_transform": None,
            "rotation_center": None,
        }

    if step2 and (ref_det.get("limbus") is None or tgt_det.get("limbus") is None):
        raise ValueError("step2=True requires non-None limbus in both detections")
    if step1 == "glint" and (not ref_det.get("glints") or not tgt_det.get("glints")):
        raise ValueError("step1='glint' requires at least one glint in both detections")

    ref_lim = ref_det.get("limbus") or {"center": [0.0, 0.0], "radius": 0.0}
    tgt_lim = tgt_det.get("limbus") or {"center": [0.0, 0.0], "radius": 0.0}

    aligned, step1_out, step2_out, center_out = _core.align_eye_images(
        np.ascontiguousarray(ref_img, dtype=np.uint8),
        np.ascontiguousarray(tgt_img, dtype=np.uint8),
        float(ref_det["pupil_center"][0]),
        float(ref_det["pupil_center"][1]),
        _pupil_radius(ref_det),
        [(float(g[0]), float(g[1])) for g in (ref_det.get("glints") or [])],
        float(ref_lim["center"][0]),
        float(ref_lim["center"][1]),
        float(ref_lim["radius"]),
        float(tgt_det["pupil_center"][0]),
        float(tgt_det["pupil_center"][1]),
        _pupil_radius(tgt_det),
        [(float(g[0]), float(g[1])) for g in (tgt_det.get("glints") or [])],
        float(tgt_lim["center"][0]),
        float(tgt_lim["center"][1]),
        float(tgt_lim["radius"]),
        _STEP1_CODE[step1],
        bool(step2),
        float(exclude_top),
        float(exclude_bottom),
        float(inner_margin),
    )

    return {
        "aligned": aligned,
        "step1_translation": tuple(step1_out) if step1_out is not None else None,
        "step2_transform": tuple(step2_out) if step2_out is not None else None,
        "rotation_center": tuple(center_out) if center_out is not None else None,
    }
