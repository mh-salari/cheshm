// Shared ROI helpers for cheshm C++ detectors.

#pragma once

#include <opencv2/core.hpp>

namespace cheshm {

inline cv::Rect clamp_roi(int roi_x, int roi_y, int roi_w, int roi_h, int img_w, int img_h)
{
    return cv::Rect(roi_x, roi_y, roi_w, roi_h) & cv::Rect(0, 0, img_w, img_h);
}

// Returns true when the caller-supplied ROI rectangle is non-empty.
// A non-positive width or height is the "no ROI" sentinel.
inline bool roi_is_active(int roi_w, int roi_h)
{
    return roi_w > 0 && roi_h > 0;
}

}  // namespace cheshm
