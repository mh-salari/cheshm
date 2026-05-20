# PuReST — pupil detector + tracker

Santini, T., Fuhl, W., Kasneci, E. (2018). "PuReST: Robust pupil
tracking for real-time pervasive eye tracking." *ETRA 2018*, 61:1-61:5.

PuReST is the stateful tracker built on top of PuRe (see
`pupil_detectors/PuRe/`). The first call runs full PuRe detection;
subsequent calls reuse the previous pupil's location to constrain a
local greedy + outline search, falling back to full PuRe detection if
both tracking paths fail.

## Original implementation source

- Source repos: <https://github.com/tcsantini/EyeRecToo>,
  <https://github.com/openPupil/Open-PupilEXT>
- Licence: non-commercial / academic only (see [LICENSE](LICENSE))