# Swirski 2D — pupil detector

Pupil tracker for off-axis eye images. Haar surround feature localises
the pupil → k-means histogram segmentation → Canny edges → starburst
rays from three seed centres → RANSAC ellipse fit with image-aware
support.

## Paper

Swirski, L., Bulling, A., Dodgson, N. (2012). "Robust real-time pupil
tracking in highly off-axis images." *ETRA 2012*, 173-176.
<https://doi.org/10.1145/2168556.2168585>

## Origin & licence

The algorithm and the bulk of this source come from Lech Swirski's
**pupiltracker** library (2014). Licensed under the **MIT licence**;
see [LICENSE](LICENSE) in this subdirectory.

## Layout

```
Swirski2D/
├── include/Swirski2D/
│   ├── conic_section.hpp      # template ellipse / conic helpers
│   ├── cvx.hpp                # cv2 helpers (resize, ROI, fitEllipse, ...)
│   ├── pupil_tracker.hpp      # public C++ API
│   └── utils.hpp              # random, randomSubset, MakeString
├── src/
│   ├── core.cpp               # nanobind binding
│   ├── cvx.cpp
│   ├── pupil_tracker.cpp      # the algorithm
│   └── utils.cpp
├── core.py                    # Python wrapper, declares _UI and _OVERLAYS
├── CMakeLists.txt
├── __init__.py                # re-exports detect_pupil
├── LICENSE                    # upstream MIT (Lech Swirski, 2014)
├── README.md
└── third_party/poolstl/       # vendored poolSTL header-only parallel
                               # STL (MIT). See LICENSE in that dir.
```
