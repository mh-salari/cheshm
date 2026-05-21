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

## Layout

```
Starburst/
├── include/Starburst/
│   ├── corneal_reflection.hpp   # remove_corneal_reflection
│   └── ransac_ellipse.hpp       # class RansacEllipse
├── src/
│   ├── core.cpp                 # nanobind binding
│   ├── corneal_reflection.cpp   # CR removal (locate/fit/interpolate)
│   ├── contour_detection.cpp    # ray search + edge accumulation
│   ├── ransac_ellipse.cpp       # RANSAC + conic-to-ellipse solve
│   └── svd.cpp                  # Householder reduction + QR iteration
├── core.py                      # Python wrapper, declares _UI and _OVERLAYS
├── CMakeLists.txt
├── __init__.py                  # re-exports detect_pupil
├── LICENSE                      # upstream GPLv2-or-later
└── README.md
```
