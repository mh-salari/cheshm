# `lavan` — eye-image primitives

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
<!-- PyPI / DOI badges added after first release tag. -->

Pupil, glint, and limbus detection on grayscale eye images, plus the C-accelerated
Daugman boundary detectors they're built on, plus iris-texture rigid alignment of
two eye images — packaged as one library.

Lavan is an island in the Persian Gulf. The package name is the island; the
contents are tools for finding what's in an eye image and lining two of them up.

## What's inside

Three sub-packages, each owning its own surface:

- [`lavan.boundary`](src/lavan/boundary/README.md) — Daugman-derived boundary
  detectors. C-accelerated integro-differential operator (1993/2004,
  circle fit) plus Fourier-series active contour (2007, non-circular) and an
  in-house pupil-shape-prior variant.
- `lavan.detect` — pupil + glint + limbus detection on a single grayscale eye
  image. Stateless single-frame detector — no temporal tracking, no calibration,
  no cross-frame model fitting.
- `lavan.align` — rigid alignment of two eye images by iris texture, given pupil
  + limbus geometry on each. Returns `(dx, dy, theta)`.

## Installation

```bash
pip install lavan
```

PyPI wheels ship a pre-compiled C kernel for the host OS (macOS / Linux /
Windows). For local development:

```bash
git clone https://github.com/mh-salari/lavan
cd lavan
python scripts/build_c.py
pip install -e .
```

## Quick start

```python
import cv2
from lavan.detect import detect_pupil, detect_glints, detect_limbus
from lavan.align import align_eye_images

img = cv2.imread("eye.png", cv2.IMREAD_GRAYSCALE)

pupil = detect_pupil(img, pupil_threshold=30)
glints = detect_glints(
    img,
    pupil_center=pupil["center"],
    pupil_radius=pupil["radius"],
    glint_threshold=240,
)
limbus = detect_limbus(img, pupil_center=pupil["center"], pupil_radius=pupil["radius"])

# Align two eye images by iris texture, given pupil + limbus geometry on each:
dx, dy, theta = align_eye_images(ref_img, ref_geom, tgt_img, tgt_geom)
```

A command-line wrapper for single-image detection is installed as `lavan-detect`.

## Acknowledgments

This work received funding from the European Union's Horizon Europe research and
innovation funding program under grant agreement No 101072410, Eyes4ICU project.
