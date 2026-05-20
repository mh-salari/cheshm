// ExCuSe pupil detector.
//
// Reference: Fuhl, W., Kübler, T., Sippel, K., Rosenstiel, W., Kasneci, E.
// (2015). "ExCuSe: Robust Pupil Detection in Real-World Scenarios."
// *CAIP 2015*, 39-51.
//
// Algorithm: adaptive thresholding via angular histogram, custom Canny
// edge detection, ray-based contour collection from a coarse seed,
// iterative region growing, ellipse fitting with quality validation.

#pragma once

#include <opencv2/core.hpp>

namespace cheshm::ExCuSe {

cv::RotatedRect findPupilEllipse(
    const cv::Mat &frame,
    int max_ellipse_radi = 50,
    int good_ellipse_threshold = 15);

}  // namespace cheshm::ExCuSe
