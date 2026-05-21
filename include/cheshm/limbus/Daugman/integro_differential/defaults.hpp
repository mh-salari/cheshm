// Daugman integro-differential operator — tunable defaults.
//
//   Daugman, J. (2004). "How Iris Recognition Works." IEEE Trans. Circuits
//   and Systems for Video Technology, 14(1), 21-30, eq. (1).

#pragma once

namespace cheshm::Daugman::integro_differential::defaults
{

// Lower bound (pixels) on candidate iris radius.
inline constexpr int R_MIN = 40;

// Upper bound (pixels) on candidate iris radius.
inline constexpr int R_MAX = 62;

// Half-width (pixels) of the centre-sweep grid around the seed.
inline constexpr int RANGE = 5;

// Grid step (pixels) for the centre sweep.
inline constexpr int STEP = 1;

} // namespace cheshm::Daugman::integro_differential::defaults
