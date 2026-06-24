#pragma once

namespace cheshm::enhance::defaults
{

inline constexpr double CLAHE_CLIP_LIMIT = 2.0;
inline constexpr int CLAHE_TILE = 8;

inline constexpr double STRETCH_LO_PCT = 1.0;
inline constexpr double STRETCH_HI_PCT = 99.0;

inline constexpr double GAMMA = 1.0;

inline constexpr int BILATERAL_D = 5;
inline constexpr double BILATERAL_SIGMA_COLOR = 50.0;
inline constexpr double BILATERAL_SIGMA_SPACE = 50.0;

inline constexpr double UNSHARP_SIGMA = 1.0;
inline constexpr double UNSHARP_AMOUNT = 1.0;

} // namespace cheshm::enhance::defaults
