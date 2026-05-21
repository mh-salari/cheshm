// Simple glint detector defaults.

#pragma once

namespace cheshm::SimpleGlint::defaults
{

inline constexpr int GLINT_THRESHOLD = 240;            // intensity above which a pixel is "glint"
inline constexpr float SEARCH_RADIUS_FACTOR = 2.0f;    // search disk radius = factor × pupil_radius
inline constexpr int FILTER_MARGIN_PX = 5;             // half-plane filter margin (px)
inline constexpr int GLINTS_TARGET = 1;                // expected number of IR LEDs
inline constexpr bool KEEP_ABOVE = true;               // keep contours above pupil centre line
inline constexpr bool KEEP_BELOW = true;               // keep contours below pupil centre line
inline constexpr bool KEEP_LEFT = true;                // keep contours left of pupil centre line
inline constexpr bool KEEP_RIGHT = true;               // keep contours right of pupil centre line
inline constexpr bool SPLIT_WIDEST_FOR_TARGET = false; // split widest blob when N = target − 1

} // namespace cheshm::SimpleGlint::defaults
