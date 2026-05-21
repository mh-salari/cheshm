// Daugman 1993 / 2004 integro-differential operator for iris boundary
// localization. For each candidate centre (x0, y0) in a grid around the
// seed, integrates image intensity around concentric circles, takes a
// Gaussian-smoothed derivative with respect to radius, and selects the
// (x0, y0, r) that maximises that derivative.

#pragma once

#include "integro_differential/defaults.hpp"

#include <cstdint>
#include <opencv2/core.hpp>
#include <optional>

namespace cheshm::Daugman::integro_differential
{

struct DetectResult
{
    cv::Point2i center;
    int radius;
};

std::optional<DetectResult> detect_limbus(const cv::Mat& image,
                                          cv::Point2d seed_center,
                                          int r_min = defaults::R_MIN,
                                          int r_max = defaults::R_MAX,
                                          int range_ = defaults::RANGE,
                                          int step = defaults::STEP);

} // namespace cheshm::Daugman::integro_differential
