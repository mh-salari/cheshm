"""Cheshm — eye-image primitives for pupil, glint, limbus, and alignment.

Every public function operates on a **single eye per image**.

Sub-packages, each owning its own surface:

- :mod:`cheshm.pupil_detectors` — pupil detectors (``detect_pupil``):
  ``Simple`` (threshold-based), ``Starburst``, ``Swirski2D``, ``ExCuSe``.
- :mod:`cheshm.glint_detectors` — glint detectors (``detect_glints``).
- :mod:`cheshm.limbus_detectors` — limbus / iris boundary detectors,
  currently the Daugman family (``IntegroDifferentialOperator``,
  ``DaugmanActiveContour``, ``PupilGuidedContour``).
- :mod:`cheshm.align` — iris-texture rigid alignment of a target eye
  image onto a reference. Returns ``(dx, dy, theta)``.
- :mod:`cheshm.viz` — cv2 image-save helpers (diff heatmap, alignment
  comparison + overlay, detection overlay).

Cheshm (چشم) is the Persian word for "eye".
"""
