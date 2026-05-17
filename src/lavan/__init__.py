"""Lavan — eye-image primitives for pupil, limbus, glint, and alignment.

Three sub-packages, each owning its own surface:

- :mod:`lavan.boundary` — Daugman-derived boundary detectors. C-accelerated
  integro-differential operator (circular fit) plus Fourier-series active
  contour and pupil-guided variant (non-circular fits).
- :mod:`lavan.detect` — pupil + glint + limbus detection on grayscale eye
  images. ``detect_pupil`` and ``detect_glints`` are independent entry points;
  ``detect_limbus`` wraps the integro-differential operator.
- :mod:`lavan.align` — iris-texture rigid alignment of two eye images given
  pupil + limbus geometry on each. Returns ``(dx, dy, theta)``.

Typical use::

    from lavan.detect import detect_pupil, detect_glints, detect_limbus
    from lavan.align import align_eye_images

The three names refer to islands in the Persian Gulf; ``lavan`` is the one
this package borrows.
"""
