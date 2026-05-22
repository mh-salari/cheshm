# <img src="https://raw.githubusercontent.com/mh-salari/cheshm/main/bindings/python/cheshm/gui/icon.png" alt="" width="48"> `cheshm`

[![PyPI version](https://img.shields.io/pypi/v/cheshm)](https://pypi.org/project/cheshm/)
[![Downloads](https://static.pepy.tech/badge/cheshm)](https://pepy.tech/project/cheshm)
[![License](https://img.shields.io/pypi/l/cheshm)](https://github.com/mh-salari/cheshm/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/1242928603.svg)](https://doi.org/10.5281/zenodo.20308209)

Cheshm is a cross-platform (Linux, macOS, Windows) C++ library with Python bindings. It packages pupil, glint, and limbus detectors for grayscale eye images, plus rigid alignment of two eye images (glint, pupil, or iris-texture based) and helpers for saving the resulting visualizations to PNG.

To annotate eye images with cheshm's detectors, see [EyE Annotation Tool](https://github.com/mh-salari/eye_annotation_tool).

<p align="center">
<img src="https://raw.githubusercontent.com/mh-salari/cheshm/main/resources/cheshm-gui.png" alt="cheshm GUI" width="800">
</p>

## Install

```
uv add cheshm
```

or with pip:

```
pip install cheshm
```

## Detectors

| Kind | Detector | Paper | Licence |
|---|---|---|---|
| pupil | Simple | — | MIT |
| pupil | Starburst | Li, Winfield, Parkhurst 2005 | GPL |
| pupil | Swirski2D | Świrski, Bulling, Dodgson 2012 | MIT |
| pupil | ExCuSe | Fuhl et al. 2015 | non-commercial |
| pupil | ElSe | Fuhl et al. 2016 | non-commercial |
| pupil | PuRe | Santini, Fuhl, Kasneci 2018 | non-commercial |
| pupil | PuReST | Santini, Fuhl, Kasneci 2018 | non-commercial |
| pupil | PupilLabs2D | Kassner, Patera, Bulling 2014 | LGPL-3.0-or-later |
| glint | Simple | — | MIT |
| limbus | Daugman integro-differential | Daugman 1993 | MIT |
| limbus | Daugman active contour | Daugman 2007 | MIT |
| limbus | Pupil-guided active contour | Daugman 2007 variant | MIT |

The top-level [`LICENSE`](LICENSE) is MIT and covers the framework code; each detector ships its own `LICENSE` file with the detector's terms. Installing the project from PyPI installs all detectors, but only the ones you import are loaded into your process — so the licence that governs your use is the licence of the detectors you imported:

```python
from cheshm.pupil_detectors.Simple import detect_pupil       # MIT
from cheshm.pupil_detectors.Starburst import detect_pupil    # GPL
from cheshm.pupil_detectors.ExCuSe import detect_pupil       # non-commercial
from cheshm.pupil_detectors.ElSe import detect_pupil         # non-commercial
from cheshm.pupil_detectors.PuRe import detect_pupil         # non-commercial
from cheshm.pupil_detectors.PuReST import PuReST             # non-commercial (stateful tracker)
from cheshm.pupil_detectors.PupilLabs2D import detect_pupil  # LGPL-3.0
from cheshm.glint_detectors.Simple import detect_glints      # MIT
from cheshm.limbus_detectors.daugman.integro_differential import detect_limbus  # MIT
from cheshm.limbus_detectors.daugman.active_contour import detect_limbus        # MIT
from cheshm.limbus_detectors.daugman.pupil_guided import detect_limbus          # MIT
```

## Single-eye contract

Every public function operates on **one eye at a time** — a single grayscale image. Callers with two eyes call cheshm twice and combine the results.

## Alignment

`cheshm.align.align_eye_images(ref_img, tgt_img, ref_det, tgt_det, *, step1, step2)` registers `tgt_img` onto `ref_img` with up to a two-step rigid transform. Either step can be enabled independently:

- **Step 1 (translation)** anchors on glint centroids (`step1="glint"`), pupil centres (`step1="pupil"`), or is skipped (`step1=None`).
- **Step 2 (iris-texture refinement, optional)** refines `(dx, dy, theta)` by minimising mean absolute intensity difference inside an iris-barrel mask built from the limbus + pupil geometry.

Set `step1="glint", step2=False` for pure glint alignment, `step1="pupil", step2=False` for pure pupil-centre alignment, `step1=None, step2=True` for iris-only, or any combination.

## Visualization

`cheshm.viz` writes PNGs that show detector and alignment outputs:

- `save_detection_overlay(out_path, img, detections, *, style, label)` — draws pupil, glint, and limbus overlays on `img`. `style` is a dict keyed by element (`pupil_contour`, `pupil_ellipse`, `pupil_center`, `pupil_mask`, `glint_contour`, `glint_ellipse`, `glint_center`, `limbus_curve`, `limbus_center`) where each value is `{show, color, thickness, alpha}`.
- `save_diff_heatmap(out_path, ref, aligned)` — colour-mapped `|ref − aligned|`.
- `save_alignment_overlay(out_path, ref, aligned)` — blended reference + aligned image.
- `save_alignment_comparison(out_path, ref, target, aligned)` — 4-panel: ref | aligned | diff-before | diff-after.

## GUI

`cheshm-gui` opens a Dear PyGui workbench for tuning detector parameters interactively on a folder of images.

```
cheshm-gui                       # empty workbench
cheshm-gui path/to/folder        # open a folder of eye images
cheshm-gui path/to/a.png p2.png  # open specific files
```

## Development

Cheshm ships precompiled wheels on PyPI for Linux, macOS, and Windows on Python 3.10–3.13, so end users never need to build anything. This section is for users who want to build from source. Building from source needs CMake, a C++20 compiler, OpenCV, and (for `PupilLabs2D`) Eigen3.

The project uses `uv` for environment management and `scikit-build-core` to drive CMake. First-time setup:

```
cd path/to/cheshm
uv sync
```

This creates `.venv/`, installs runtime + build deps, and builds the C++ extensions in editable mode (per `[tool.scikit-build]` in `pyproject.toml`).

After changing any C/C++ source, force a recompile:

```
uv sync --reinstall-package cheshm
```

Plain `uv sync` will not notice C/C++ source edits — only `pyproject.toml` changes.

After changing a binding's signature (a `core.cpp` under `bindings/python/src/`), regenerate the matching `_core.pyi`:

```
./scripts/regen_stubs.sh
```

Commit the updated `.pyi` alongside the `.cpp`. CI verifies the committed stubs match what stubgen would emit and fails the PR if they drift.

### Repo layout

```
include/cheshm/         public C++ headers (libcheshm_cpp)
  helpers/              shared image-processing helpers (edges, ellipses, image, shape)
  pupil/<Det>/          per-pupil-detector headers
  glint/Simple/         per-glint-detector headers
  limbus/Daugman/       Daugman family (active_contour, integro_differential, pupil_guided)
  align/                rigid alignment headers
  viz/                  visualization headers
src/                    C++ algorithm implementations
bindings/python/
  cheshm/               Python package (PyPI)
  src/                  nanobind C++ glue, one CMake per detector
third_party/poolstl/    vendored parallel-STL backend
```

## Name

In Persian (Farsi), Cheshm (چشم) literally means "eye".

## Logo

The nazar / cheshm amulet image is from [pngegg](https://www.pngegg.com/en/png-klwpz).

## Acknowledgments

This work received funding from the European Union's Horizon Europe research and innovation funding program under grant agreement No 101072410, Eyes4ICU project.

<p align="center">
<img src="https://raw.githubusercontent.com/mh-salari/cheshm/main/resources/Funded_by_EU_Eyes4ICU.png" alt="Funded by EU Eyes4ICU" width="500">
</p>
