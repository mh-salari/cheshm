"""Pupil Labs 2D pupil detector, vendored from pupil-labs/pupil-detectors.

Re-exports the upstream public API under ``lavan.pupil_detector_2d``:

  - :class:`Detector2D` — threshold + ellipse-fitting detector from Pupil Core
  - :class:`DetectorBase` — abstract base shared by all upstream detectors
  - :class:`Roi` — region-of-interest helper

The vendored sources retain their original LGPL-3.0-or-later license; see
``COPYING.LESSER`` and ``COPYING`` in this directory and ``NOTICE.md`` at the
repo root.
"""

from .detector_2d import Detector2D
from .detector_base import DetectorBase
from .roi import Roi

__all__ = ["Detector2D", "DetectorBase", "Roi"]
