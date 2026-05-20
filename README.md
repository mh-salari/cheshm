# <img src="https://raw.githubusercontent.com/mh-salari/cheshm/main/src/cheshm/gui/icon.png" alt="" width="48"> `cheshm`

[![PyPI version](https://img.shields.io/pypi/v/cheshm)](https://pypi.org/project/cheshm/)
[![Downloads](https://static.pepy.tech/badge/cheshm)](https://pepy.tech/project/cheshm)
[![License](https://img.shields.io/pypi/l/cheshm)](https://github.com/mh-salari/cheshm/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/1242928603.svg)](https://doi.org/10.5281/zenodo.20308209)


## Single-eye contract

Every public function in cheshm operates on **one eye at a time** — a single grayscale image.

## License

Cheshm's framework code is MIT-licensed (see [`LICENSE`](LICENSE)). Each detector ships its own LICENSE file in its subdirectory and may carry a different licence depending on the upstream source it was ported from.

| Detector | Subdirectory | Licence |
|---|---|---|
| Simple (pupil) | `src/cheshm/pupil_detectors/Simple/` | MIT |
| Starburst | `src/cheshm/pupil_detectors/Starburst/` | GPL (Li, Winfield, Parkhurst 2005 — cvEyeTracker / openEyes ToolKit) |
| Swirski2D | `src/cheshm/pupil_detectors/Swirski2D/` | MIT (Świrski, Bulling, Dodgson 2012 — pupiltracker by Lech Świrski) |
| ExCuSe | `src/cheshm/pupil_detectors/ExCuSe/` | Non-commercial (Fuhl et al. 2015, University of Tübingen) |
| ElSe | `src/cheshm/pupil_detectors/ElSe/` | Non-commercial (Fuhl et al. 2016, University of Tübingen) |
| PuRe | `src/cheshm/pupil_detectors/PuRe/` | Non-commercial (Santini et al. 2018, University of Tübingen) |
| PuReST | `src/cheshm/pupil_detectors/PuReST/` | Non-commercial (Santini et al. 2018, University of Tübingen) |
| PupilLabs2D | `src/cheshm/pupil_detectors/PupilLabs2D/` | LGPL-3.0-or-later (Kassner et al. 2014 — Pupil Core / pupil-labs/pupil-detectors) |
| Simple (glint) | `src/cheshm/glint_detectors/Simple/` | MIT |
| Daugman integro-differential operator | `src/cheshm/limbus_detectors/daugman/integro_differential/` | MIT (© 2023 Fatih BAŞATEMUR — carried from his MIT port) |
| Daugman 2007 active contour | `src/cheshm/limbus_detectors/daugman/active_contour/` | MIT |
| Pupil-shape-prior active contour | `src/cheshm/limbus_detectors/daugman/pupil_guided/` | MIT |

Each detector is imported explicitly — the licence of the detectors you import is the licence that governs your use. `pip install cheshm` installs all of them, but only the ones you `import` are loaded into your process:

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
