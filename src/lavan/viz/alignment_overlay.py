"""Save a blended overlay of two grayscale images (reference + aligned)."""

from pathlib import Path

import cv2
import numpy as np

from ._common import _add_label, _to_bgr


def save_alignment_overlay(
    out_path: str | Path,
    ref_img: np.ndarray,
    aligned: np.ndarray,
    *,
    label: str | None = None,
    ref_weight: float = 0.5,
) -> None:
    """Write a 50/50 blend of ``ref_img`` and ``aligned`` as a PNG.

    ``ref_weight`` controls the reference's contribution (default 0.5
    for a balanced blend). If ``label`` is given, a title bar is added on
    top of the blend.
    """
    if ref_img.shape != aligned.shape:
        raise ValueError(f"ref shape {ref_img.shape} differs from aligned shape {aligned.shape}")
    blend = cv2.addWeighted(_to_bgr(ref_img), ref_weight, _to_bgr(aligned), 1.0 - ref_weight, 0.0)
    if label is not None:
        blend = _add_label(blend, label)
    if not cv2.imwrite(str(out_path), blend):
        raise OSError(f"failed to write {out_path}")
