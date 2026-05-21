// PupilLabs2D pupil detector — Pupil Labs / Pupil Core 2D detector.
//
// Reference: Kassner, M., Patera, W., Bulling, A. (2014). "Pupil: an open
// source platform for pervasive eye tracking and mobile gaze-based
// interaction." *UbiComp 2014 Adjunct*, 1151-1160.

#pragma once

#include <opencv2/core.hpp>
#include <optional>

namespace cheshm::PupilLabs2D
{

struct Properties
{
    int intensity_range;
    int blur_size;
    float canny_threshold;
    float canny_ratio;
    int canny_aperture;
    int pupil_size_max;
    int pupil_size_min;
    float strong_perimeter_ratio_range_min;
    float strong_perimeter_ratio_range_max;
    float strong_area_ratio_range_min;
    float strong_area_ratio_range_max;
    int contour_size_min;
    float ellipse_roundness_ratio;
    float initial_ellipse_fit_threshold;
    float final_perimeter_ratio_range_min;
    float final_perimeter_ratio_range_max;
    float ellipse_true_support_min_dist;
    float support_pixel_ratio_exponent;
    bool coarse_detection;
    int coarse_filter_min;
    int coarse_filter_max;
};

Properties default_properties();

struct DetectResult
{
    cv::RotatedRect ellipse;
    float confidence;
};

std::optional<DetectResult> detect(const cv::Mat& gray, const cv::Rect& roi, const Properties& props);

} // namespace cheshm::PupilLabs2D
