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

## Pass-1 modernisation

The upstream C++ has been reorganised into lavan's per-detector
"complex detector" layout (`include/Swirski2D/` + `src/`). Pass-1
modernisation per [REFACTOR.md](../../../REFACTOR.md):

  - Namespace `pupiltracker` → `lavan::Swirski2D`.
  - Header guards → `#pragma once`.
  - Dropped the upstream `tracker_log` debug helper (with its
    `boost::lexical_cast` dependency) and the matching parameter
    on `findPupilEllipse`. Also dropped the `SECTION(name, log)`
    timing macro; the algorithm body is unchanged.
  - Replaced `BOOST_FOREACH` (7 sites) with C++11 range-for.
  - Replaced `boost::math::isnormal` with `std::isnormal`.
  - Replaced Intel TBB at all 3 parallel sites (Haar `parallel_reduce`,
    starburst `parallel_for` + `concurrent_vector`, RANSAC
    `parallel_reduce`) with vendored header-only
    [poolSTL](https://github.com/alugowski/poolSTL) (MIT, under
    `third_party/poolstl/`). `poolstl::par` is used as the execution
    policy with `std::transform_reduce` (Haar) and `std::for_each`
    (starburst, RANSAC). Apple Clang's libc++ ships without a parallel
    STL backend, so `std::execution::par_unseq` would silently fall
    back to sequential there — poolSTL fills that gap portably.
  - Dropped the unused `<opencv2/highgui/highgui.hpp>` and
    `<opencv2/core/core_c.h>` includes; replaced the deprecated
    `CV_AA` constant with `cv::LINE_AA`.

Pass 2 (deep modernisation — `std::span`, `std::optional`, drop raw
pointers / arrays, smart pointers) and Pass 3 (speed) are pending.

## Layout

```
Swirski2D/
├── include/Swirski2D/
│   ├── conic_section.hpp      # template ellipse / conic helpers
│   ├── cvx.hpp                # cv2 helpers (resize, ROI, fitEllipse, ...)
│   ├── pupil_tracker.hpp      # public C++ API
│   └── utils.hpp              # random, randomSubset, MakeString
├── src/
│   ├── core.cpp               # extern "C" public surface
│   ├── cvx.cpp
│   ├── pupil_tracker.cpp      # the algorithm
│   └── utils.cpp
├── core.py                    # ctypes wrapper, declares _UI and _OVERLAYS
├── CMakeLists.txt
├── __init__.py                # re-exports detect_pupil
├── LICENSE                    # upstream MIT (Lech Swirski, 2014)
├── README.md
└── third_party/poolstl/       # vendored poolSTL header-only parallel
                               # STL (MIT). See LICENSE in that dir.
```
