#pragma once

namespace cheshm::align::defaults
{

inline constexpr int DX_LO = -10;
inline constexpr int DX_HI = 11;
inline constexpr int DY_LO = -10;
inline constexpr int DY_HI = 11;

inline constexpr double ROT_START = -2.0;
inline constexpr double ROT_END = 2.0;
inline constexpr double ROT_STEP = 0.05;

inline constexpr double FINE_ROT_HALF = 0.05;
inline constexpr int FINE_ROT_N = 11;
inline constexpr double FINE_SHIFT_HALF = 1.0;
inline constexpr int FINE_SHIFT_N = 21;

inline constexpr double EXCLUDE_TOP = 60.0;
inline constexpr double EXCLUDE_BOTTOM = 45.0;
inline constexpr double INNER_MARGIN = 15.0;
inline constexpr double OUTER_MARGIN = 10.0;

// Glint-match tolerance as a fraction of the glint spacing: above detection
// noise, below the gap to a neighbouring glint.
inline constexpr double GLINT_MATCH_TOL_FRACTION = 0.5;

} // namespace cheshm::align::defaults
