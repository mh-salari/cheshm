# `lavan`

[![PyPI version](https://img.shields.io/pypi/v/lavan)](https://pypi.org/project/lavan/)
[![Downloads](https://static.pepy.tech/badge/lavan)](https://pepy.tech/project/lavan)
[![License](https://img.shields.io/pypi/l/lavan)](https://github.com/mh-salari/lavan/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/1242928603.svg)](https://doi.org/10.5281/zenodo.20284443)


## Single-eye contract

Every public function in lavan operates on **one eye at a time** — a single grayscale image.

## License

Lavan's core code is MIT-licensed (see [`LICENSE`](LICENSE)). Each detector ships its own LICENSE file in its subdirectory and may carry a different licence depending on the upstream source it was ported from.

| Detector | Subdirectory | Licence |
|---|---|---|
| Daugman integro-differential operator | `src/lavan/limbus_detectors/daugman/integro_differential/` | MIT (© 2023 Fatih BAŞATEMUR — carried from his MIT port) |
| Daugman 2007 active contour | `src/lavan/limbus_detectors/daugman/active_contour/` | MIT |
| Pupil-shape-prior active contour | `src/lavan/limbus_detectors/daugman/pupil_guided/` | MIT |

## Name

Lavan (لاوان) is an island in the Persian Gulf.

## Acknowledgments

This work received funding from the European Union's Horizon Europe research and innovation funding program under grant agreement No 101072410, Eyes4ICU project.

<p align="center">
<img src="https://raw.githubusercontent.com/mh-salari/lavan/main/resources/Funded_by_EU_Eyes4ICU.png" alt="Funded by EU Eyes4ICU" width="500">
</p>
