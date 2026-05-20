// Corneal-reflection removal — locate the specular highlight inside a
// search window around an initial seed and replace it with a smooth
// interpolant of the surrounding intensities, so the pupil edge search
// isn't distracted by it.

#pragma once

#include <opencv2/core/types.hpp>

namespace cheshm::Starburst {

void remove_corneal_reflection(
    cv::Mat &image,
    int sx,
    int sy,
    int window_size,
    int biggest_crr,
    int &crx,
    int &cry,
    int &crr);

}  // namespace cheshm::Starburst
