"""Save ``|ref - aligned|`` as a colour-mapped PNG."""

from pathlib import Path

import cv2
import numpy as np

from ._common import _diff_hot


def save_diff_heatmap(
    out_path: str | Path,
    ref: np.ndarray,
    aligned: np.ndarray,
    *,
    vmax: float | None = None,
) -> float:
    """Write ``|ref - aligned|`` as a hot-colour-mapped PNG sized to the inputs.

    ``vmax=None`` clips at the per-image 99th percentile. Returns the
    value actually used so callers can record it.
    """
    if ref.shape != aligned.shape:
        raise ValueError(f"ref shape {ref.shape} differs from aligned shape {aligned.shape}")
    diff = np.abs(ref.astype(np.float32) - aligned.astype(np.float32))
    if vmax is None:
        vmax = max(float(np.percentile(diff, 99)), 1.0)
    if not cv2.imwrite(str(out_path), _diff_hot(ref, aligned, vmax)):
        raise OSError(f"failed to write {out_path}")
    return float(vmax)
