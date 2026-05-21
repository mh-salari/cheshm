// PuRe pupil detector defaults.

#pragma once

namespace cheshm::PuRe::defaults
{

inline constexpr int BASE_WIDTH = 320;  // working-frame width cap (px)
inline constexpr int BASE_HEIGHT = 240; // working-frame height cap (px)

inline constexpr float CANTHI_DISTANCE_MM = 27.6f; // mean palpebral fissure width
inline constexpr float MIN_PUPIL_DIAMETER_MM = 2.0f;
inline constexpr float MAX_PUPIL_DIAMETER_MM = 8.0f;

inline constexpr int OUTLINE_BIAS = 5; // intensity gap (uchar) gating outline-contrast votes

inline constexpr int CANNY_BINS = 64;               // bins in the magnitude histogram used for threshold selection
inline constexpr float CANNY_NON_EDGE_RATIO = 0.7f; // fraction of pixels considered non-edge
inline constexpr float CANNY_LOW_HIGH_RATIO = 0.4f; // low_th = ratio * high_th

} // namespace cheshm::PuRe::defaults