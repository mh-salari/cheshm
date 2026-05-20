"""Lavan — eye-image primitives for pupil, glint, limbus, and alignment.

Every public function operates on a **single eye per image**.

Sub-packages, each owning its own surface:

- :mod:`lavan.pupil_detectors` — pupil detectors (``detect_pupil``)
  and shared pupil-center methods (``pupil_center_of_mass``,
  ``fit_convex_hull_spline``).
- :mod:`lavan.glint_detectors` — glint detectors (``detect_glints``).
- :mod:`lavan.limbus_detectors` — limbus / iris boundary detectors,
  currently the Daugman family (``IntegroDifferentialOperator``,
  ``DaugmanActiveContour``, ``PupilGuidedContour``).
- :mod:`lavan.align` — iris-texture rigid alignment of a target eye
  image onto a reference. Returns ``(dx, dy, theta)``.
- :mod:`lavan.viz` — cv2 image-save helpers (diff heatmap, alignment
  comparison + overlay, detection overlay).

Lavan (لاوان) is an island in the Persian Gulf.
"""
