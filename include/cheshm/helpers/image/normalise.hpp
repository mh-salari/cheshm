// Rescale a numeric matrix to the full [0, 255] uint8 range.

#pragma once

#include <opencv2/core.hpp>

namespace cheshm
{

inline void normalise_to_u8(cv::InputArray src, cv::InputOutputArray dst)
{
    cv::normalize(src, dst, 0, 255, cv::NORM_MINMAX, CV_8U);
}

} // namespace cheshm
