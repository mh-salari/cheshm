# `lavan`

[![License: LGPL v3](https://img.shields.io/badge/license-LGPL--3.0--or--later-blue.svg)](LICENSE)
<!-- PyPI / DOI badges added after first release tag. -->

Pupil, glint, and limbus detection on grayscale eye images, plus iris-texture rigid alignment of two eye images. Includes C-accelerated Daugman boundary detectors (used by the limbus detector) and a vendored Pupil Labs 2D pupil detector.

## Name

Lavan (لاوان) is an island in the Persian Gulf.

## What's inside

Four sub-packages, each owning its own surface:

- [`lavan.boundary`](src/lavan/boundary/README.md) — Daugman-derived boundary detectors: C-accelerated integro-differential operator (1993/2004, circle fit), Fourier-series active contour (2007, non-circular), and an in-house pupil-shape-prior variant.
- `lavan.detect` — threshold-based pupil and glint detection, and a Daugman-operator limbus detector, for grayscale eye images.
- `lavan.pupil_detector_2d` — Pupil Labs' 2D pupil detector from the [Pupil Core](https://github.com/pupil-labs/pupil/) eye-tracking platform, vendored under LGPL v3 from [pupil-labs/pupil-detectors](https://github.com/pupil-labs/pupil-detectors) and built natively for every wheel platform. Exposes `Detector2D`, `DetectorBase`, and `Roi`. See [NOTICE.md](NOTICE.md) for attribution.
- `lavan.align` — rigid alignment of two eye images by iris texture, given pupil + limbus geometry on each. Returns `(dx, dy, theta)`. Includes iris-centre alignment, minimum-difference search, barrel/iris-mask helpers, and matplotlib-based blend / diff / overlay plotting.

### `lavan.detect` settings

| Function | Settings |
|---|---|
| `detect_pupil(img, ...)` | `pupil_threshold` (0–255, default 30), `pupil_center_method` (`convex_hull_centroid` / `center_of_mass` / `ellipse_fit_center` / `min_area_rect_center`), `pupil_roi` (optional `(x, y, w, h)`), shape gates `min_ellipse_fit_ratio` / `min_roundness_ratio` (both 0–1, off by default) |
| `detect_glints(img, pupil_center, pupil_radius, ...)` | `glint_threshold` (0–255, default 240), `search_radius_factor` (× pupil radius, default 2.0), `glint_roi`, `glint_center_method` (same four methods), `max_area_px`, half-plane filters `keep_above` / `keep_below` / `keep_left` / `keep_right` + `filter_margin_px`, `glints_target` (expected count), `split_widest_for_target` (4-LED merge case), shape gates as in `detect_pupil` |
| `detect_limbus(img, pupil_center, pupil_radius, ...)` | `r_min_factor` / `r_max_factor` (limbus radius range as × pupil radius), `search_window_px` (centre-search half-window) |

### `lavan.pupil_detector_2d.Detector2D` settings

`Detector2D` exposes 22 properties via `get_properties()` / `update_properties()`. The practical ones to start tuning:

| Property | Meaning | Default |
|---|---|---|
| `intensity_range` | Pupil-vs-iris brightness gap (0–255 grey levels) | 23 |
| `pupil_size_min` / `pupil_size_max` | Pupil diameter bounds in pixels | 10 / 100 |
| `blur_size` | Pre-detection Gaussian kernel (odd integer) | 5 |
| `coarse_detection` | Enable Pupil Core's coarse-pupil pre-pass | `True` |
| `canny_treshold` / `canny_ration` / `canny_aperture` | Canny edge-detection params | 160 / 2 / 5 |

Run `Detector2D().get_default_properties()` for the full set + advanced gates.

### `lavan.align` strategies + settings

`align_eye_images` is the high-level entry point. It runs up to two steps. The caller picks both:

| Strategy | `step1` | `step2` | Notes |
|---|---|---|---|
| Glint translation only | `"glint"` (default) | `False` | Pure translation so glint centroids match. Needs `glints` in both detections. |
| Pupil translation only | `"pupil"` | `False` | Pure translation so pupil centres match. Needs `pupil_center` or `pupil_ellipse` in both. |
| Iris-texture search only | `None` | `True` | No pre-translation; iris-barrel min-diff search covers the full `(dx, dy, theta)` range. Needs `limbus` in both. |
| Glint translation, then iris search | `"glint"` (default) | `True` (default) | Default behaviour. Step 1 brings the eyes close; step 2 refines inside the iris barrel. |
| Pupil translation, then iris search | `"pupil"` | `True` | Same as the default but anchored on the pupil instead of the glint. Useful when no usable glint is present. |
| No-op | `None` | `False` | Returns the target unchanged. |

Returned dict::

    {
        "aligned":            <warped target image, same shape as ref_img>,
        "step1_translation":  (dx, dy)        or None when step1 is None,
        "step2_transform":    (dx, dy, theta) or None when step2 is False,
        "rotation_center":    (cx, cy)       or None when step2 is False,
    }

Lower-level settings on the individual functions:

| Function | Settings |
|---|---|
| `align_by_min_diff(img_ref, img_mov, mask, ...)` | `dx_range` / `dy_range` (integer-pixel search bounds, default ±10), `rot_range` (`(start, end, step)` degrees, default ±2° at 0.05° steps), `rotation_center` (defaults to image centre). Coarse integer pass + sub-pixel refinement. |
| `align_by_translation(ref_point, mov_point)` | Translation-only; no settings. Generic point-to-point translation — give it any two points (glint centroids, pupil centres, iris centres) and it returns the `(dx, dy, 0.0)` that maps `mov_point` onto `ref_point`. |
| `make_iris_mask(img_shape, limbus_center, limbus_r, pupil_r, ...)` | `exclude_top` (degrees of upper eyelid arc to drop, default 60), `exclude_bottom` (default 45), `inner_margin` (pixels added to the pupil radius to avoid pupil-edge bleed, default 15). |
| `make_barrel_mask(...)` | Same parameters as `make_iris_mask`; fills the gap between the left/right crescents to produce a filled barrel rather than a ring. |

## Installation

```bash
pip install lavan
```

### Building from source

If you want to build a wheel yourself (or develop against `lavan`), the
build is driven by [scikit-build-core] and needs CMake plus OpenCV 4 and
Eigen 3 development packages on the host:

- **macOS:** `brew install cmake eigen opencv`
- **Linux:** `apt-get install cmake libeigen3-dev libopencv-dev` (or your
  distro's equivalent; manylinux CI builds OpenCV from source via
  `scripts/manylinux-before-all.sh`)
- **Windows:** `choco install -y cmake eigen opencv`

Then:

```bash
git clone https://github.com/mh-salari/lavan
cd lavan
pip install .
```

`lavan` ships compiled C/C++ extensions and is **not intended to be installed
editable** — `pip install -e .` would leave the compiled binaries outside the
source tree, where the ctypes loader can't find them. Rebuild + reinstall after
edits instead.

[scikit-build-core]: https://scikit-build-core.readthedocs.io

## Quick start

### Single-image detection

```python
import cv2
from lavan.detect import detect_pupil, detect_glints, detect_limbus

img = cv2.imread("eye.png", cv2.IMREAD_GRAYSCALE)

pupil = detect_pupil(img, pupil_threshold=30)
glints = detect_glints(
    img,
    pupil_center=pupil["center"],
    pupil_radius=pupil["radius"],
    glint_threshold=240,
)
limbus = detect_limbus(img, pupil_center=pupil["center"], pupil_radius=pupil["radius"])
```

A command-line wrapper for single-image detection is installed as `lavan-detect`.

### Pupil Labs 2D detector

```python
import cv2
from lavan.pupil_detector_2d import Detector2D

detector = Detector2D()
img = cv2.imread("eye.png", cv2.IMREAD_GRAYSCALE)
result = detector.detect(img)
print(result["ellipse"], "confidence:", result["confidence"])
```

### Iris-texture alignment

```python
from lavan.align import align_eye_images

# ``ref_geom`` / ``tgt_geom`` each carry the detected pupil + limbus geometry
# for that image. align_eye_images returns the rigid transform that warps
# the target onto the reference, measured by iris-texture similarity inside
# a barrel-shaped iris mask (eyelid zones excluded).
dx, dy, theta = align_eye_images(ref_img, ref_geom, tgt_img, tgt_geom)
```

## License

`lavan` is released under the [GNU Lesser General Public License v3 or later (LGPL-3.0-or-later)](LICENSE). The LGPL incorporates the [GPL](LICENSE.GPL) by reference; both are shipped with the package. See [NOTICE.md](NOTICE.md) for vendored components and their attribution.

## Acknowledgments

This work received funding from the European Union's Horizon Europe research and innovation funding program under grant agreement No 101072410, Eyes4ICU project.

<p align="center">
<img src="https://raw.githubusercontent.com/mh-salari/lavan/main/resources/Funded_by_EU_Eyes4ICU.png" alt="Funded by EU Eyes4ICU" width="500">
</p>
