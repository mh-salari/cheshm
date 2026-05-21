// Simple pupil detector defaults.

#pragma once

namespace cheshm::Simple::defaults {

inline constexpr int PUPIL_THRESHOLD = 30;        // intensity below which a pixel is "pupil"
inline constexpr int MAX_CONTOUR_POINTS = 4096;   // cap on contour points returned to Python

}  // namespace cheshm::Simple::defaults
