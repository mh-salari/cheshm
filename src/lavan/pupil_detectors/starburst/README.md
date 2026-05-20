# Starburst — pupil detector

Hybrid feature-based + model-based pupil detector: shoot rays from a
seed centre, find the first strong intensity rise on each ray, RANSAC-fit
an ellipse to the surviving edge points.

## Paper

Li, D., Winfield, D., Parkhurst, D.J. (2005). "Starburst: A hybrid
algorithm for video-based eye tracking combining feature-based and
model-based approaches." *CVPR Workshops 2005*, vol. 3, 79-79.
<https://doi.org/10.1109/CVPR.2005.531>

## Origin & licence

The algorithm and the bulk of this source come from **cvEyeTracker
1.2.5 / openEyes ToolKit** (Iowa State University, 2004-2006), authored
by Dongheng Li, Derrick Parkhurst, Jason Babcock, David Winfield.
Licensed under the **GNU General Public License v2 or later**. See
[LICENSE](LICENSE) in this subdirectory.

The C++ has been reorganised into the lavan per-detector layout
(`include/starburst/` + `src/`) and put through pass 1 of the
[three-pass modernisation](../../../REFACTOR.md): namespaced under
`lavan::starburst`, macros and raw heap allocations replaced with
`std::`-equivalents, anonymous-namespace internal helpers, no behaviour
changes. Deep + speed passes are pending.

## Distribution note

Because Starburst is GPL, lavan ships it inside the main `lavan` wheel
during development — `import lavan.pupil_detectors.starburst` works
straight off `pip install lavan`. Before lavan v2.0 ships to PyPI, this
subdir migrates to its own `lavan-starburst` plugin package; the main
`lavan` wheel stays MIT-only and `pip install lavan[starburst]` opts in
to the GPL plugin (see Distribution model in `REFACTOR.md`).

## Layout

```
starburst/
├── include/starburst/
│   ├── corneal_reflection.hpp   # remove_corneal_reflection
│   └── ransac_ellipse.hpp       # class RansacEllipse
├── src/
│   ├── core.cpp                 # extern "C" public surface
│   ├── corneal_reflection.cpp   # CR removal (locate/fit/interpolate)
│   ├── contour_detection.cpp    # ray search + edge accumulation
│   ├── ransac_ellipse.cpp       # RANSAC + conic-to-ellipse solve
│   └── svd.cpp                  # Householder reduction + QR iteration
├── core.py                      # ctypes wrapper, declares _UI and _OVERLAYS
├── CMakeLists.txt
├── __init__.py                  # re-exports detect_pupil
├── LICENSE                      # upstream GPLv2-or-later
└── README.md
```
