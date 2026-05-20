"""Shared helpers used by cheshm's Python pupil detectors.

  - :func:`_crop_to_roi` / :func:`_translate_pupil_result` — the crop +
    shift-back pair every Python pupil detector uses to honour
    ``pupil_roi``.
"""

import numpy as np


def _crop_to_roi(
    img: np.ndarray,
    roi: tuple[int, int, int, int],
) -> tuple[np.ndarray, tuple[int, int]]:
    """Clamp ``roi=(x, y, w, h)`` to ``img`` bounds and return ``(crop, (x0, y0))``.

    ``crop`` is a numpy view over ``img`` (no copy); ``(x0, y0)`` is the
    clamped top-left to add back to any crop-local coordinate to recover
    the full-image coordinate. ``cv2.Rect`` semantics — integer pixels,
    so no half-pixel offset is introduced.

    Raises ``ValueError`` if the ROI has zero overlap with the image.
    """
    h, w = img.shape[:2]
    rx, ry, rw, rh = (int(v) for v in roi)
    x0 = max(rx, 0)
    y0 = max(ry, 0)
    x1 = min(rx + rw, w)
    y1 = min(ry + rh, h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"ROI {tuple(roi)} has zero overlap with {(w, h)} image")
    return img[y0:y1, x0:x1], (x0, y0)


def _translate_pupil_result(
    result: dict | None,
    dx: int,
    dy: int,
    full_shape: tuple[int, ...] | None = None,
) -> dict | None:
    """Shift a pupil-detector result from crop-local back to full-image coords.

    Fields are translated only when present in ``result``:

      - ``center``: ``(cx, cy) + (dx, dy)``
      - ``ellipse``: ``((cx, cy), size, angle)`` — centre shifted
      - ``contour``: ``(N, 1, 2)`` integer array — each point shifted
      - ``mask``: pasted into a zero canvas of ``full_shape`` at ``(dy, dx)``;
        skipped when ``full_shape`` is None.

    Returns ``None`` if ``result`` is ``None``.
    """
    if result is None:
        return None
    out = dict(result)
    if out.get("center") is not None:
        cx, cy = out["center"]
        out["center"] = (cx + dx, cy + dy)
    if out.get("ellipse") is not None:
        (cx, cy), size, angle = out["ellipse"]
        out["ellipse"] = ((cx + dx, cy + dy), size, angle)
    if out.get("contour") is not None:
        contour = out["contour"]
        out["contour"] = contour + np.array([[[dx, dy]]], dtype=contour.dtype)
    if out.get("mask") is not None and full_shape is not None:
        crop_mask = out["mask"]
        full_mask = np.zeros(full_shape[:2], dtype=crop_mask.dtype)
        mh, mw = crop_mask.shape
        full_mask[dy : dy + mh, dx : dx + mw] = crop_mask
        out["mask"] = full_mask
    return out
