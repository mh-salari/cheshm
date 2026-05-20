# `cheshm`

[![PyPI version](https://img.shields.io/pypi/v/cheshm)](https://pypi.org/project/cheshm/)
[![Downloads](https://static.pepy.tech/badge/cheshm)](https://pepy.tech/project/cheshm)
[![License](https://img.shields.io/pypi/l/cheshm)](https://github.com/mh-salari/cheshm/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/1242928603.svg)](https://doi.org/10.5281/zenodo.20293526)


## Single-eye contract

Every public function in cheshm operates on **one eye at a time** — a single grayscale image.

## License

Cheshm's core code is MIT-licensed (see [`LICENSE`](LICENSE)). Each detector ships its own LICENSE file in its subdirectory and may carry a different licence depending on the upstream source it was ported from.

| Detector | Subdirectory | Licence |
|---|---|---|
| Daugman integro-differential operator | `src/cheshm/limbus_detectors/daugman/integro_differential/` | MIT (© 2023 Fatih BAŞATEMUR — carried from his MIT port) |
| Daugman 2007 active contour | `src/cheshm/limbus_detectors/daugman/active_contour/` | MIT |
| Pupil-shape-prior active contour | `src/cheshm/limbus_detectors/daugman/pupil_guided/` | MIT |

## Name

In Persian (Farsi), Cheshm (چشم) literally means "eye".

## Acknowledgments

This work received funding from the European Union's Horizon Europe research and innovation funding program under grant agreement No 101072410, Eyes4ICU project.

<p align="center">
<img src="https://raw.githubusercontent.com/mh-salari/cheshm/main/resources/Funded_by_EU_Eyes4ICU.png" alt="Funded by EU Eyes4ICU" width="500">
</p>
