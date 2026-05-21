// Daugman 2007 active-contour radial search — tunable defaults.
//
//   Daugman, J. (2007). "New Methods in Iris Recognition." IEEE Trans.
//   Systems, Man, and Cybernetics, Part B, 37(5), 1167-1175.

#pragma once

namespace cheshm::Daugman::active_contour::defaults
{

// Number of angular samples theta taken around the seed centre.
inline constexpr int N = 360;

// Number of Fourier coefficients kept by the boundary low-pass. The
// 2007 paper recommends 5 for the iris outer boundary.
inline constexpr int M = 5;

// Gaussian sigma (pixels) for the pre-Sobel image blur.
inline constexpr double GRADIENT_SIGMA = 1.0;

// Gaussian sigma (radial samples) applied to each angle's radial
// gradient profile before the argmax.
inline constexpr double RADIAL_SMOOTHING = 2.0;

// Mask out the upper-eyelid angular wedge during the radial search.
inline constexpr bool SKIP_EYELID_WEDGES = true;

// Lower bound (pixels) on candidate iris radius.
inline constexpr double R_MIN = 30.0;

// Upper bound (pixels) on candidate iris radius.
inline constexpr double R_MAX = 80.0;

} // namespace cheshm::Daugman::active_contour::defaults
