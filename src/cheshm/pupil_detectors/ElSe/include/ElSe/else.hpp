// ElSe pupil detector.
//
// Reference: Fuhl, W., Santini, T., Kübler, T., Kasneci, E. (2016).
// "ElSe: Ellipse Selection for Robust Pupil Detection in Real-World
// Environments." *ETRA 2016*, vol. 14, 123-130.
//
// Algorithm: custom Canny → edge filter → contour curve extraction →
// ellipse fit → inner/outer intensity-ratio selection. If no ellipse
// passes the selection gate, a morphological-blob fallback returns a
// coarse pupil position only.

#pragma once

#include <opencv2/core.hpp>
#include <optional>

namespace cheshm::ElSe
{

enum class DetectionMethod
{
    Ellipse,      // primary path produced a fitted ellipse
    BlobFallback, // primary failed; only a coarse blob position is known
};

struct DetectResult
{
    DetectionMethod method;
    cv::Point2f center;                     // always set
    std::optional<cv::RotatedRect> ellipse; // set only when method == Ellipse
};

std::optional<DetectResult> detect(const cv::Mat& frame, float min_area_ratio, float max_area_ratio);

} // namespace cheshm::ElSe
