// Two-stage shape-quality gate for candidate contours: contour-to-fitted-
// ellipse area ratio, then isoperimetric quotient (4*pi*area / perimeter^2).

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <vector>

namespace cheshm {

// Returns true when ``contour`` clears both opt-in gates.
//
// A negative ``min_ellipse_fit_ratio`` disables the area-ratio gate; a
// negative ``min_roundness_ratio`` disables the roundness gate. When
// the area-ratio gate is active, ``ellipse_fit`` must hold a value —
// passing ``std::nullopt`` signals "no ellipse could be fit" and the
// candidate is rejected.
inline bool passes_shape_quality(
    const std::vector<cv::Point> &contour,
    const std::optional<cv::RotatedRect> &ellipse_fit,
    double min_ellipse_fit_ratio,
    double min_roundness_ratio)
{
    if (min_ellipse_fit_ratio < 0.0 && min_roundness_ratio < 0.0) {
        return true;
    }
    const double area = cv::contourArea(contour);
    if (min_ellipse_fit_ratio >= 0.0) {
        if (!ellipse_fit) {
            return false;
        }
        const double ellipse_area =
            CV_PI * (ellipse_fit->size.width / 2.0) * (ellipse_fit->size.height / 2.0);
        if (ellipse_area <= 0.0) {
            return false;
        }
        if (area / ellipse_area < min_ellipse_fit_ratio) {
            return false;
        }
    }
    if (min_roundness_ratio >= 0.0) {
        const double perimeter = cv::arcLength(contour, true);
        if (perimeter <= 0.0) {
            return false;
        }
        const double roundness = 4.0 * CV_PI * area / (perimeter * perimeter);
        if (roundness < min_roundness_ratio) {
            return false;
        }
    }
    return true;
}

}  // namespace cheshm
