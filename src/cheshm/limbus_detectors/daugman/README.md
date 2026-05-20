# `cheshm.limbus_detectors.daugman` — Daugman-derived limbus boundary detectors

Two methods from John Daugman, plus a pupil-shape-prior extension of the
2007 active-contour framework. Importable as `cheshm.limbus_detectors.daugman`.

## 1. `IntegroDifferentialOperator` (1993 / 2004)

The classic Daugman operator. For each candidate `(x₀, y₀, r)` it integrates
image intensity around the corresponding circle, takes a Gaussian-smoothed
derivative with respect to `r`, and selects the `(x₀, y₀, r*)` that
maximises that derivative magnitude:

```
max  | Gσ(r) ∗ ∂/∂r   ∮ I(x, y) / (2πr) ds |
(r, x₀, y₀)                  r, x₀, y₀
```

Source paper: Daugman, J. (2004). "How Iris Recognition Works."
*IEEE Trans. Circuits and Systems for Video Technology*, 14(1), 21-30,
eq. (1). The same operator was introduced in Daugman, J. (1993).
"High Confidence Visual Recognition of Persons by a Test of Statistical
Independence." *IEEE Trans. PAMI*, 15(11), 1148-1161.

**Files:**

- [`integro_differential/core.py`](integro_differential/core.py) —
  `IntegroDifferentialOperator` class, Python wrapper.
- [`integro_differential/core.c`](integro_differential/core.c) +
  compiled `core.{so,dylib,dll}` — Bresenham circle perimeter +
  intensity integral + Gaussian-smoothed `∂/∂r` derivative.

### Provenance

The Python implementation was adapted from Fatih BAŞATEMUR's MIT-licensed
reference port at
<https://github.com/fbasatemur/daugman_iris_detection>.

**What we changed relative to the upstream implementation:**

- Rewrote the per-`(x₀, y₀)` integral-of-mean-around-the-circle and its
  Gaussian-smoothed `∂/∂r` derivative in C (`integro_differential/core.c`)
  for roughly a 100× speed-up. The Python class is now a thin `ctypes`
  wrapper around `integro_differential_operator_search` plus the
  morphological-open preprocessing and the coarse-to-fine search wrapper
  (`result`).
- Dropped the notebook scaffolding; kept the algorithmic core only.
- Renamed from the upstream `DIED` (acronym) to the paper's term.
- `_DISK3` inlined as a NumPy array instead of importing
  `skimage.morphology.disk`.

## 2. `DaugmanActiveContour` — Fourier-series active contour (2007)

A generalisation of the integro-differential operator that drops the
"perfect circle" assumption. The boundary is parameterised as a Fourier
series in the radial coordinate, allowing ellipses (k = 2 Fourier
component) and higher-order non-circular shapes (off-axis foreshortening,
slightly aspherical limbi, partial corneal distortion).

Per the paper:

1. A seed centre is supplied (we pass the integro-differential operator's centre).
2. The radial-direction image gradient
   `g_r(r, θ) = ∂I/∂x cos θ + ∂I/∂y sin θ` is sampled at `N` regularly
   spaced angles around the seed; for each angle, the radius `r_θ` of
   maximum radial gradient is recorded.
3. The discrete Fourier transform of `{r_θ}` (paper eq. 1) is truncated to
   `M` complex coefficients (with monotonically-decreasing weights). The
   inverse DFT (paper eq. 2) gives the smooth, possibly non-circular
   boundary `{R_θ}`. `M = 1` enforces a circle; `M = 5` is Daugman's
   choice for the iris outer boundary; `M = 17` for the pupil.

Source paper: Daugman, J. (2007). "New Methods in Iris Recognition."
*IEEE Trans. Systems, Man, and Cybernetics — Part B*, 37(5), 1167-1175,
Section II ("Active Contours and Generalized Coordinates"), eqs. (1) and
(2).

**Files:**

- [`active_contour/core.py`](active_contour/core.py) —
  `DaugmanActiveContour` class. Python wrapper that computes the Sobel
  gradients via OpenCV and dispatches the per-angle radial-gradient
  search + Gaussian-smoothing + argmax to the C kernel.
- [`active_contour/core.c`](active_contour/core.c) + compiled
  `core.{so,dylib,dll}` — bilinear sampling of the Sobel gradients along
  each radial ray, radial-direction projection, 1-D Gaussian smoothing
  across radii, argmax per angle.

## 3. `PupilGuidedContour` — pupil-shape-prior active contour

Same Fourier-series framework as `DaugmanActiveContour`, but the radial
search uses **per-angle** bounds derived from the pupil ellipse rather
than isotropic scalar `[r_min, r_max]`. The pupil and limbus are
projections of (nearly) co-planar circles on the eye, so the pupil's
elongation direction and aspect ratio are a good prior on the limbus
shape. Using that prior to bound where the search looks keeps the radial
argmax inside the iris and stops it locking onto eyelash gradients at
high radii.

Per angle `θ`, the pupil ellipse `(a_p, b_p, α_p)` gives a radial profile
`r_p(θ) = a_p b_p / sqrt((b_p cos(θ−α))² + (a_p sin(θ−α))²)`. The radial
search then runs over `r ∈ [k_min · r_p(θ), k_max · r_p(θ)]`. Only the
pupil's shape is used — its centre is intentionally discarded, so the
recovered limbus centre is free to differ from the pupil centre.

`M = 3` is the default truncation: it keeps the k=0 (mean radius) and k=2
(ellipse harmonic) components plus k=1, enough to express a centred
ellipse and not enough to admit the higher-frequency wiggle that fits
eyelashes on cluttered images.

**Files:**

- [`pupil_guided/core.py`](pupil_guided/core.py) — `PupilGuidedContour`
  class.
- [`pupil_guided/core.c`](pupil_guided/core.c) + compiled
  `core.{so,dylib,dll}` — same bilinear-sample + radial-projection +
  Gaussian-smoothed argmax structure as `active_contour/core.c`, but
  reads `r_min` / `r_max` as length-`N` arrays.

## API

```python
from cheshm.limbus_detectors.daugman import (
    DaugmanActiveContour,
    IntegroDifferentialOperator,
    PupilGuidedContour,
)

# 1993 / 2004 circular operator
operator = IntegroDifferentialOperator(image, r_min=..., r_max=...)
results = operator.search(cen_x, cen_y, range_, step)
# → array of (y, x, score, r) rows sorted by score; best is results[-1].

# 2007 Fourier active contour, paper-faithful (isotropic search)
dac = DaugmanActiveContour(image, N=360, M=5)
(cx, cy), thetas, R_theta = dac.fit(seed_center=(sx, sy), r_min=..., r_max=...)

# Pupil-shape-prior variant (anisotropic search)
pgc = PupilGuidedContour(image, pupil_ellipse, N=360, M=3)
(cx, cy), thetas, R_theta = pgc.fit(seed_center=(sx, sy), k_min=..., k_max=...)

# Boundary points in either case: (cx + R_θ cos θ, cy + R_θ sin θ).
```

