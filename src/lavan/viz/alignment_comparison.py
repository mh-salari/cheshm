"""Save a 4-panel alignment comparison: reference | aligned | diff-before | diff-after."""

from pathlib import Path

import cv2
import numpy as np

from ._common import _add_label, _diff_hot, _to_bgr


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
    """Write a 4-panel PNG showing the alignment outcome.

    Panels (left to right):
      1. ``ref_img`` with label ``ref_label``.
      2. ``aligned`` with label ``target_label``.
      3. ``|ref_img - target_img|`` colour-mapped — diff before alignment.
      4. ``|ref_img - aligned|`` colour-mapped — diff after alignment.

    Diff panels use ``cv2.COLORMAP_HOT`` clipped at ``vmax`` (defaults to
    the 99th percentile of the larger of the two diffs).
    """
    if not (ref_img.shape == target_img.shape == aligned.shape):
        raise ValueError("ref_img, target_img and aligned must share the same shape")
    diff_before = np.abs(ref_img.astype(np.float32) - target_img.astype(np.float32))
    diff_after = np.abs(ref_img.astype(np.float32) - aligned.astype(np.float32))
    if vmax is None:
        vmax = max(float(np.percentile(np.maximum(diff_before, diff_after), 99)), 1.0)
    panels = [
        _add_label(_to_bgr(ref_img), ref_label),
        _add_label(_to_bgr(aligned), target_label),
        _add_label(_diff_hot(ref_img, target_img, vmax), "diff (before)"),
        _add_label(_diff_hot(ref_img, aligned, vmax), "diff (after)"),
    ]
    composite = cv2.hconcat(panels)
    if not cv2.imwrite(str(out_path), composite):
        raise OSError(f"failed to write {out_path}")
