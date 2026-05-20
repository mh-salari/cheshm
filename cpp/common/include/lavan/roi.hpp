// Shared ROI helpers for lavan C++ pupil/glint/limbus detectors.
// Header-only — pulled in via the `lavan_common` CMake INTERFACE target.
//
// The pair `clamp_roi` + `crop_view` is the C++ equivalent of the
// Python `_crop_to_roi` helper in `lavan._common`. Same clamping rules,
// same integer-pixel coordinates, so a detector running on the cropped
// view produces bit-identical output to the same detector running on
// the equivalent Python crop.

#pragma once

#include <opencv2/core.hpp>

namespace lavan {

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

}  // namespace lavan
