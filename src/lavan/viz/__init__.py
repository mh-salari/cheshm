"""cv2-based image-save helpers for visualising lavan's outputs.

- :func:`save_diff_heatmap` — colour-mapped ``|ref - aligned|`` PNG.
- :func:`save_alignment_comparison` — 4-panel: ``ref | aligned | diff-before | diff-after``.
- :func:`save_alignment_overlay` — blended overlay of ``ref`` and ``aligned``.
- :func:`save_detection_overlay` — image with pupil + glint detection overlays drawn on top.
"""

from .alignment_comparison import save_alignment_comparison
from .alignment_overlay import save_alignment_overlay
from .detection_overlay import save_detection_overlay
from .diff_heatmap import save_diff_heatmap

__all__ = [
    "save_alignment_comparison",
    "save_alignment_overlay",
    "save_detection_overlay",
    "save_diff_heatmap",
]
