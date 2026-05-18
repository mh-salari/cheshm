# Third-party notices

`lavan` is licensed under the GNU Lesser General Public License v3.0 or later
(LGPL-3.0-or-later). See [LICENSE](LICENSE) for the LGPL text and
[LICENSE.GPL](LICENSE.GPL) for the GPL text the LGPL incorporates by
reference.

## Vendored components

### src/lavan/pupil_detector_2d/

Adapted from [pupil-labs/pupil-detectors](https://github.com/pupil-labs/pupil-detectors),
the standalone 2D pupil detector from the Pupil Core eye-tracking platform.

- Original copyright (C) 2012-2019 Pupil Labs
- Original license: LGPL-3.0-or-later (preserved)
- Source repo: <https://github.com/pupil-labs/pupil-detectors>
- The `COPYING` and `COPYING.LESSER` files inside that directory are the
  upstream license files, kept verbatim.

Modifications by Mohammadhossein Salari are released under the same LGPL-3.0-or-later
license. Significant modifications include build-system integration (scikit-build-core
replaces scikit-build / setup.py), arm64 macOS build support, and a thin Python
wrapper exposed as `lavan.pupil_detector_2d.Detector2D`.
