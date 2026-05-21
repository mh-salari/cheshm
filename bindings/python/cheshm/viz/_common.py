"""Internal helpers shared by the viz savers."""

import cv2
import numpy as np


def _to_bgr(img: np.ndarray) -> np.ndarray:
    """Return a BGR copy of ``img``; passes through if already 3-channel."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def _add_label(img: np.ndarray, text: str, *, height: int = 32) -> np.ndarray:
    """Prepend a dark label bar with centred white text."""
    label_bar = np.zeros((height, img.shape[1], 3), dtype=np.uint8)
    label_bar[:] = (40, 40, 40)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = max((img.shape[1] - text_w) // 2, 0)
    y = (height + text_h) // 2
    cv2.putText(label_bar, text, (x, y), font, font_scale, (255, 255, 255), thickness)
    return cv2.vconcat([label_bar, img])


def _diff_hot(a: np.ndarray, b: np.ndarray, vmax: float) -> np.ndarray:
    """Absolute difference of two grayscale images, colour-mapped to HOT (BGR)."""
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    diff_u8 = np.clip(diff / float(vmax) * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(diff_u8, cv2.COLORMAP_HOT)
