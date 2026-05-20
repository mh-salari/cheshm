// PuRe pupil detector.
//
// Reference: Santini, T., Fuhl, W., Kasneci, E. (2018). "PuRe: Robust
// pupil detection for real-time pervasive eye tracking." *Computer
// Vision and Image Understanding*, 170, 40-50.
//
// Algorithm: downscale to a working frame, custom Canny → edge filter
// → contour extraction → per-curve pupil-candidate validation (size,
// curvature, ellipse fit, anchor distribution, outline contrast) →
// candidate combination → score-based selection.

#pragma once

#include <opencv2/core.hpp>
#include <optional>

namespace cheshm::PuRe {

struct DetectResult {
    cv::RotatedRect ellipse;
    float confidence;  // outline-contrast vote ratio in [0, 1]
};

std::optional<DetectResult> detect(
    const cv::Mat &frame,
    float min_pupil_diameter_mm,
    float max_pupil_diameter_mm,
    float canthi_distance_mm,
    int outline_bias);

}  // namespace cheshm::PuRe