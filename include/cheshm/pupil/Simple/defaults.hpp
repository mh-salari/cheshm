// Simple pupil detector defaults.

#pragma once

namespace cheshm::Simple::defaults
{

inline constexpr int PUPIL_THRESHOLD = 30;      // intensity below which a pixel is "pupil"
inline constexpr int MAX_CONTOUR_POINTS = 4096; // cap on contour points returned to Python

// Fourier pupil-form smoothing (Wyatt 1995): fit a robust low-order polar
// Fourier series to the contour and return the smoothed margin.
inline constexpr bool FOURIER_SMOOTHING = true;         // on by default
inline constexpr int FOURIER_HARMONICS = 5;             // K terms (shape detail)
inline constexpr int FOURIER_SAMPLES = 360;             // points on the output margin
inline constexpr int FOURIER_ITERATIONS = 4;            // robust IRLS passes
inline constexpr double FOURIER_INWARD_REJECTION = 1.0; // higher -> bridge intrusions harder

// Adaptive thresholding around glints: where the image is very bright (glint)
// and within reach of it, relax the pupil threshold so the glint + halo merge
// into the pupil instead of carving the contour inward.
inline constexpr bool GLINT_MERGE = false;      // off by default
inline constexpr int GLINT_THRESHOLD = 230;     // brightness above which a pixel is a glint
inline constexpr double GLINT_BOOST_PCT = 25.0; // pupil-threshold relaxation near the glint (%)
inline constexpr int GLINT_REACH_PX = 12;       // how far "near the glint" reaches (px)

} // namespace cheshm::Simple::defaults
