// Corneal-reflection (CR) removal — locate the specular highlight inside
// a search window around an initial seed and replace it with a smooth
// interpolant of the surrounding intensities, so that the pupil edge
// search isn't distracted by the bright CR.
//
// Original algorithm: cvEyeTracker / openEyes ToolKit (Iowa State,
// 2004-2006). See LICENSE (GPL) in this subdirectory.

#pragma once

#include <opencv2/core/types.hpp>

namespace lavan::starburst {

void remove_corneal_reflection(
    cv::Mat &image,
    int sx,
    int sy,
    int window_size,
    int biggest_crr,
    int &crx,
    int &cry,
    int &crr);

}  // namespace lavan::starburst
