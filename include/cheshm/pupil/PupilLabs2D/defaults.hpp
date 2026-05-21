// PupilLabs2D pupil detector defaults.

#pragma once

namespace cheshm::PupilLabs2D::defaults
{

inline constexpr bool COARSE_DETECTION = true;
inline constexpr int COARSE_FILTER_MIN = 128;
inline constexpr int COARSE_FILTER_MAX = 280;
inline constexpr int INTENSITY_RANGE = 23;
inline constexpr int BLUR_SIZE = 5;
inline constexpr float CANNY_THRESHOLD = 160.0f;
inline constexpr float CANNY_RATIO = 2.0f;
inline constexpr int CANNY_APERTURE = 5;
inline constexpr int PUPIL_SIZE_MAX = 100;
inline constexpr int PUPIL_SIZE_MIN = 10;
inline constexpr float STRONG_PERIMETER_RATIO_RANGE_MIN = 0.8f;
inline constexpr float STRONG_PERIMETER_RATIO_RANGE_MAX = 1.1f;
inline constexpr float STRONG_AREA_RATIO_RANGE_MIN = 0.6f;
inline constexpr float STRONG_AREA_RATIO_RANGE_MAX = 1.1f;
inline constexpr int CONTOUR_SIZE_MIN = 5;
inline constexpr float ELLIPSE_ROUNDNESS_RATIO = 0.1f;
inline constexpr float INITIAL_ELLIPSE_FIT_THRESHOLD = 1.8f;
inline constexpr float FINAL_PERIMETER_RATIO_RANGE_MIN = 0.6f;
inline constexpr float FINAL_PERIMETER_RATIO_RANGE_MAX = 1.2f;
inline constexpr float ELLIPSE_TRUE_SUPPORT_MIN_DIST = 2.5f;
inline constexpr float SUPPORT_PIXEL_RATIO_EXPONENT = 2.0f;

} // namespace cheshm::PupilLabs2D::defaults
