"""Iris-based rigid alignment of two grayscale eye images.

Given a reference and a target eye image plus the pupil + limbus geometry
detected on each, fit a small rigid transform ``(dx, dy, theta)`` that
warps the target onto the reference. The cost function is the mean
absolute intensity difference inside a barrel-shaped iris mask (top and
bottom eyelid zones excluded), so the alignment is driven by iris texture
rather than by eyelashes or specular glints.

Public surface re-exported here for convenience::

    from lavan.align import (
        align_by_translation,
        align_by_min_diff,
        align_by_min_diff_plain,
        align_eye_images,
        apply_transform,
        make_barrel_mask,
        make_iris_mask,
        plot_blend,
        plot_diff,
        plot_mask_overlay,
        save_aligned_pair_images,
        save_diff_heatmap,
    )
"""

from .core import (
    align_by_translation,
    align_by_min_diff,
    align_by_min_diff_plain,
    apply_transform,
    make_barrel_mask,
    make_iris_mask,
)
from .pair import align_eye_images
from .plots import (
    plot_blend,
    plot_diff,
    plot_mask_overlay,
    save_aligned_pair_images,
    save_diff_heatmap,
)

__all__ = [
    "align_by_translation",
    "align_by_min_diff",
    "align_by_min_diff_plain",
    "align_eye_images",
    "apply_transform",
    "make_barrel_mask",
    "make_iris_mask",
    "plot_blend",
    "plot_diff",
    "plot_mask_overlay",
    "save_aligned_pair_images",
    "save_diff_heatmap",
]
