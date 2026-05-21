// Swirski2D pupil detector defaults.

#pragma once

namespace cheshm::Swirski2D::defaults
{

inline constexpr int RADIUS_MIN = 20;                   // Haar pupil-radius lower bound (px)
inline constexpr int RADIUS_MAX = 80;                   // Haar pupil-radius upper bound (px)
inline constexpr float CANNY_BLUR = 1.6f;               // Gaussian σ (px) before Canny
inline constexpr float CANNY_THRESHOLD_1 = 30.0f;       // Canny hysteresis low threshold
inline constexpr float CANNY_THRESHOLD_2 = 50.0f;       // Canny hysteresis high threshold
inline constexpr int STARBURST_POINTS = 30;             // rays cast from each of the three seed centres
inline constexpr int PERCENTAGE_INLIERS = 30;           // RANSAC target inlier fraction (%)
inline constexpr int INLIER_ITERATIONS = 2;             // inlier-refit cycles per RANSAC hypothesis
inline constexpr bool IMAGE_AWARE_SUPPORT = true;       // use image-gradient strength as goodness
inline constexpr int EARLY_TERMINATION_PERCENTAGE = 95; // stop RANSAC at this inlier fraction
inline constexpr bool EARLY_REJECTION = true;           // reject samples whose gradients point wrong
inline constexpr int SEED = 0;                          // RANSAC random seed; -1 uses an internal counter
inline constexpr int MAX_INLIERS = 1024;                // cap on inlier points returned to Python

} // namespace cheshm::Swirski2D::defaults
