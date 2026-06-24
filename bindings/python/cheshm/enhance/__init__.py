"""Image-enhancement preprocessing for eye images.

Display-friendly contrast/denoise filters that also help threshold-based
detectors on low-contrast pupils. Every function takes and returns a uint8
grayscale image. ``apply`` dispatches one method by name.
"""

from .core import apply, bilateral, clahe, gamma, percentile_stretch, unsharp

__all__ = [
    "apply",
    "bilateral",
    "clahe",
    "gamma",
    "percentile_stretch",
    "unsharp",
]
