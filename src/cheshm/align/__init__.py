"""Iris-based rigid alignment of one eye image onto a reference.

Given a reference and a target eye image plus the pupil + limbus geometry
detected on each, fit a small rigid transform ``(dx, dy, theta)`` that
warps the target onto the reference. The cost function is the mean
absolute intensity difference inside a barrel-shaped iris mask (top and
bottom eyelid zones excluded), so the alignment is driven by iris texture
rather than by eyelashes or specular glints.
"""

from .core import (
    align_by_min_diff,
    align_by_min_diff_plain,
    align_by_translation,
    apply_transform,
    make_barrel_mask,
    make_iris_mask,
)
from .pair import align_eye_images

__all__ = [
    "align_by_min_diff",
    "align_by_min_diff_plain",
    "align_by_translation",
    "align_eye_images",
    "apply_transform",
    "make_barrel_mask",
    "make_iris_mask",
]
