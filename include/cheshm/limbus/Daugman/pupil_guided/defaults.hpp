// Pupil-shape-guided active contour for iris boundary — tunable defaults.
//
// Radial-gradient framework from:
//   Daugman, J. (2007). "New Methods in Iris Recognition." IEEE Trans.
//   Systems, Man, and Cybernetics, Part B, 37(5), 1167-1175.

#pragma once

namespace cheshm::Daugman::pupil_guided::defaults
{

// Number of angular samples theta taken around the seed centre.
inline constexpr int N = 360;

// Number of Fourier coefficients kept by the boundary low-pass. Three
// keeps the mean radius plus the ellipse harmonic.
inline constexpr int M = 3;

// Gaussian sigma (pixels) for the pre-Sobel image blur.
inline constexpr double GRADIENT_SIGMA = 1.0;

// Gaussian sigma (radial samples) applied to each angle's radial
// gradient profile before the argmax.
inline constexpr double RADIAL_SMOOTHING = 2.0;

// Lower scale factor: per-angle search starts at k_min * pupil_radius(theta).
inline constexpr double K_MIN = 2.0;

// Upper scale factor: per-angle search ends at k_max * pupil_radius(theta).
inline constexpr double K_MAX = 4.0;

} // namespace cheshm::Daugman::pupil_guided::defaults
