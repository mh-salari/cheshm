"""cv2-based image-save helpers for visualising cheshm's outputs."""

from pathlib import Path

import numpy as np

from . import _core

BGRColor = tuple[int, int, int]


def save_diff_heatmap(
    out_path: str | Path,
    ref: np.ndarray,
    aligned: np.ndarray,
    *,
    vmax: float | None = None,
) -> float:
    """Write ``|ref - aligned|`` as a hot-colour-mapped PNG."""
    return _core.save_diff_heatmap(
        str(out_path),
        np.ascontiguousarray(ref, dtype=np.uint8),
        np.ascontiguousarray(aligned, dtype=np.uint8),
        -1.0 if vmax is None else float(vmax),
    )


def save_alignment_overlay(
    out_path: str | Path,
    ref_img: np.ndarray,
    aligned: np.ndarray,
    *,
    label: str | None = None,
    ref_weight: float = _core.ALIGNMENT_OVERLAY_REF_WEIGHT,
) -> None:
    """Write a blended overlay of ``ref_img`` and ``aligned`` as a PNG."""
    _core.save_alignment_overlay(
        str(out_path),
        np.ascontiguousarray(ref_img, dtype=np.uint8),
        np.ascontiguousarray(aligned, dtype=np.uint8),
        float(ref_weight),
        label,
    )


def save_alignment_comparison(
    out_path: str | Path,
    ref_img: np.ndarray,
    target_img: np.ndarray,
    aligned: np.ndarray,
    *,
    ref_label: str = "reference",
    target_label: str = "aligned",
    vmax: float | None = None,
) -> None:
    """Write a 4-panel PNG showing the alignment outcome."""
    _core.save_alignment_comparison(
        str(out_path),
        np.ascontiguousarray(ref_img, dtype=np.uint8),
        np.ascontiguousarray(target_img, dtype=np.uint8),
        np.ascontiguousarray(aligned, dtype=np.uint8),
        ref_label,
        target_label,
        -1.0 if vmax is None else float(vmax),
    )


_DEFAULT_COLORS = {
    "pupil_contour": _core.PUPIL_CONTOUR_COLOR,
    "pupil_ellipse": _core.PUPIL_ELLIPSE_COLOR,
    "pupil_center": _core.PUPIL_CENTER_COLOR,
    "pupil_mask": _core.PUPIL_MASK_COLOR,
    "glint_contour": _core.GLINT_CONTOUR_COLOR,
    "glint_ellipse": _core.GLINT_ELLIPSE_COLOR,
    "glint_center": _core.GLINT_CENTER_COLOR,
}

_DEFAULT_SHOW = {
    "pupil_contour": _core.SHOW_PUPIL_CONTOUR,
    "pupil_ellipse": _core.SHOW_PUPIL_ELLIPSE,
    "pupil_center": _core.SHOW_PUPIL_CENTER,
    "pupil_mask": _core.SHOW_PUPIL_MASK,
    "glints": _core.SHOW_GLINTS,
}


def _contour_to_xy(contour: np.ndarray | None) -> np.ndarray | None:
    """Reshape (N, 1, 2) int32 contour to (N, 2) int32 for the C++ binding."""
    if contour is None:
        return None
    return np.ascontiguousarray(np.asarray(contour, dtype=np.int32).reshape(-1, 2))


def _ellipse_to_tuple(ellipse: tuple | None) -> tuple[float, float, float, float, float] | None:
    if ellipse is None:
        return None
    (cx, cy), (w, h), angle = ellipse
    return (float(cx), float(cy), float(w), float(h), float(angle))


def _center_to_tuple(center: tuple | None) -> tuple[float, float] | None:
    if center is None:
        return None
    return (float(center[0]), float(center[1]))


def save_detection_overlay(
    out_path: str | Path,
    img: np.ndarray,
    detections: dict,
    *,
    colors: dict[str, BGRColor] | None = None,
    show: dict[str, bool] | None = None,
    mask_alpha: float = _core.MASK_ALPHA,
    label: str | None = None,
) -> None:
    """Draw a pupil + glint detection overlay on ``img`` and save it."""
    palette = {**_DEFAULT_COLORS, **(colors or {})}
    visible = {**_DEFAULT_SHOW, **(show or {})}

    glints_payload = []
    for glint in detections.get("glints", []) or []:
        glints_payload.append((
            _contour_to_xy(glint.get("contour")),
            _ellipse_to_tuple(glint.get("ellipse")),
            _center_to_tuple(glint.get("center")),
        ))

    pupil_mask = detections.get("mask")
    if pupil_mask is not None:
        pupil_mask = np.ascontiguousarray(pupil_mask, dtype=np.uint8)

    _core.save_detection_overlay(
        str(out_path),
        np.ascontiguousarray(img, dtype=np.uint8),
        _contour_to_xy(detections.get("contour")),
        _ellipse_to_tuple(detections.get("ellipse")),
        _center_to_tuple(detections.get("center")),
        pupil_mask,
        glints_payload,
        tuple(float(c) for c in palette["pupil_contour"]),
        tuple(float(c) for c in palette["pupil_ellipse"]),
        tuple(float(c) for c in palette["pupil_center"]),
        tuple(float(c) for c in palette["pupil_mask"]),
        tuple(float(c) for c in palette["glint_contour"]),
        tuple(float(c) for c in palette["glint_ellipse"]),
        tuple(float(c) for c in palette["glint_center"]),
        bool(visible["pupil_contour"]),
        bool(visible["pupil_ellipse"]),
        bool(visible["pupil_center"]),
        bool(visible["pupil_mask"]),
        bool(visible["glints"]),
        float(mask_alpha),
        label,
    )
