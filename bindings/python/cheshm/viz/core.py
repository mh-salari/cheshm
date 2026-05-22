"""cv2-based image-save helpers for visualising cheshm's outputs."""

from pathlib import Path

import numpy as np

from . import _core

BGRColor = tuple[int, int, int]
ElementStyle = dict  # {"show": bool, "color": (B, G, R), "thickness": int, "alpha": float}


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


def _contour_to_xy(contour: np.ndarray | None) -> np.ndarray | None:
    """Reshape a (..., 2) int32 contour/curve to (N, 2) for the C++ binding."""
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


def _style_tuple(style: ElementStyle | None) -> tuple[bool, tuple[float, float, float], int, float] | None:
    """Convert a user style dict ``{show, color, thickness, alpha}`` to a 4-tuple."""
    if style is None:
        return None
    return (
        bool(style["show"]),
        tuple(float(c) for c in style["color"]),
        int(style["thickness"]),
        float(style["alpha"]),
    )


_VALID_KEYS = (
    "pupil_contour",
    "pupil_ellipse",
    "pupil_center",
    "pupil_mask",
    "glint_contour",
    "glint_ellipse",
    "glint_center",
    "limbus_curve",
    "limbus_center",
)


def save_detection_overlay(
    out_path: str | Path,
    img: np.ndarray,
    detections: dict,
    *,
    style: dict[str, ElementStyle] | None = None,
    label: str | None = None,
) -> None:
    """Draw pupil + glint + limbus overlays on ``img`` and save it.

    ``detections`` recognised keys (all optional):

      - ``contour`` / ``ellipse`` / ``center`` / ``mask`` — pupil fields.
      - ``glints`` — list of dicts, each with ``contour`` / ``ellipse`` / ``center``.
      - ``limbus_curve`` — closed polyline as ``np.ndarray`` of shape ``(N, 2)``.
      - ``limbus_center`` — ``(cx, cy)``.

    ``style`` mirrors the GUI overlay state: a dict keyed by element name
    (``pupil_contour``, ``pupil_ellipse``, ``pupil_center``, ``pupil_mask``,
    ``glint_contour``, ``glint_ellipse``, ``glint_center``, ``limbus_curve``,
    ``limbus_center``) with values of ``{"show", "color", "thickness", "alpha"}``.
    Missing element keys fall back to C++ defaults.
    """
    style = style or {}
    unknown = set(style) - set(_VALID_KEYS)
    if unknown:
        raise ValueError(f"unknown style keys: {sorted(unknown)}")

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
        _contour_to_xy(detections.get("limbus_curve")),
        _center_to_tuple(detections.get("limbus_center")),
        _style_tuple(style.get("pupil_contour")),
        _style_tuple(style.get("pupil_ellipse")),
        _style_tuple(style.get("pupil_center")),
        _style_tuple(style.get("pupil_mask")),
        _style_tuple(style.get("glint_contour")),
        _style_tuple(style.get("glint_ellipse")),
        _style_tuple(style.get("glint_center")),
        _style_tuple(style.get("limbus_curve")),
        _style_tuple(style.get("limbus_center")),
        label,
    )
