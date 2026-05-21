"""Save an eye image with pupil + glint detection overlays drawn on top."""

from pathlib import Path

import cv2
import numpy as np

from ._common import _add_label, _to_bgr

BGRColor = tuple[int, int, int]

DEFAULT_COLORS: dict[str, BGRColor] = {
    "pupil_contour": (0, 255, 0),  # green
    "pupil_ellipse": (0, 255, 255),  # yellow
    "pupil_center": (0, 255, 0),  # green
    "pupil_mask": (0, 60, 0),  # dark-green tint
    "glint_contour": (0, 0, 255),  # red
    "glint_ellipse": (0, 165, 255),  # orange
    "glint_center": (0, 0, 255),  # red
}

DEFAULT_SHOW: dict[str, bool] = {
    "pupil_contour": True,
    "pupil_ellipse": True,
    "pupil_center": True,
    "pupil_mask": False,
    "glints": True,
}


def save_detection_overlay(
    out_path: str | Path,
    img: np.ndarray,
    detections: dict,
    *,
    colors: dict[str, BGRColor] | None = None,
    show: dict[str, bool] | None = None,
    mask_alpha: float = 0.3,
    label: str | None = None,
) -> None:
    """Draw a pupil + glint detection overlay on ``img`` and save it.

    ``detections`` is the dict produced by cheshm's detectors. Recognised
    keys:

      - ``contour`` — pupil contour (``np.ndarray``, ``(N, 1, 2)``).
      - ``ellipse`` — pupil ellipse ``((cx, cy), (w, h), angle)``.
      - ``center`` — pupil centre ``(cx, cy)``.
      - ``mask`` — pupil binary mask (``np.ndarray``, ``uint8``).
      - ``glints`` — list of dicts, each with ``contour`` / ``ellipse`` /
        ``center``.

    Missing keys are skipped silently. ``colors`` overrides per-element
    BGR colours; ``show`` overrides per-element visibility. Both fall
    back to :data:`DEFAULT_COLORS` / :data:`DEFAULT_SHOW`.
    """
    palette = {**DEFAULT_COLORS, **(colors or {})}
    visible = {**DEFAULT_SHOW, **(show or {})}
    canvas = _to_bgr(img)

    if visible["pupil_mask"] and detections.get("mask") is not None:
        mask = detections["mask"]
        if mask.ndim == 2:
            tint = np.zeros_like(canvas)
            tint[mask > 0] = palette["pupil_mask"]
            canvas = cv2.addWeighted(canvas, 1.0, tint, mask_alpha, 0.0)

    if visible["pupil_contour"] and detections.get("contour") is not None:
        cv2.drawContours(canvas, [detections["contour"]], -1, palette["pupil_contour"], thickness=1)

    if visible["pupil_ellipse"] and detections.get("ellipse") is not None:
        (cx, cy), (w, h), angle = detections["ellipse"]
        cv2.ellipse(
            canvas,
            (round(cx), round(cy)),
            (max(round(w / 2), 1), max(round(h / 2), 1)),
            float(angle),
            0,
            360,
            palette["pupil_ellipse"],
            thickness=1,
        )

    if visible["pupil_center"] and detections.get("center") is not None:
        cx, cy = detections["center"]
        cv2.circle(canvas, (round(cx), round(cy)), radius=3, color=palette["pupil_center"], thickness=-1)

    if visible["glints"]:
        for glint in detections.get("glints", []) or []:
            if glint.get("contour") is not None:
                cv2.drawContours(canvas, [glint["contour"]], -1, palette["glint_contour"], thickness=1)
            if glint.get("ellipse") is not None:
                (gx, gy), (gw, gh), gangle = glint["ellipse"]
                cv2.ellipse(
                    canvas,
                    (round(gx), round(gy)),
                    (max(round(gw / 2), 1), max(round(gh / 2), 1)),
                    float(gangle),
                    0,
                    360,
                    palette["glint_ellipse"],
                    thickness=1,
                )
            if glint.get("center") is not None:
                gcx, gcy = glint["center"]
                cv2.circle(canvas, (round(gcx), round(gcy)), radius=2, color=palette["glint_center"], thickness=-1)

    if label is not None:
        canvas = _add_label(canvas, label)

    if not cv2.imwrite(str(out_path), canvas):
        raise OSError(f"failed to write {out_path}")
