// ElSe pupil detector defaults.

#pragma once

namespace cheshm::ElSe::defaults {

// Working size cap. Frames larger than this on either axis are
// downscaled before detection runs; results are scaled back to
// full-image coordinates afterwards.
inline constexpr int IMG_SIZE = 640;

// Pupil-area bounds expressed as fractions of the working frame area.
// Gate candidate ellipses in find_best_edge and the blob fallback.
inline constexpr float MIN_AREA_RATIO = 0.005f;
inline constexpr float MAX_AREA_RATIO = 0.2f;

}  // namespace cheshm::ElSe::defaults
