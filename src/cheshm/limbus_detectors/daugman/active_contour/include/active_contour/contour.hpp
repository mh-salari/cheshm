// Daugman 2007 Fourier-series active contour for iris boundary
// localization.
//
//   Daugman, J. (2007). "New Methods in Iris Recognition." IEEE Trans.
//   Systems, Man, and Cybernetics, Part B, 37(5), 1167-1175, Section II
//   ("Active Contours and Generalized Coordinates"), eqs. (1) and (2).

#pragma once

#include "active_contour/defaults.hpp"

#include <opencv2/core.hpp>
#include <optional>
#include <vector>

namespace cheshm::Daugman::active_contour
{

struct DetectResult
{
    cv::Point2d center;
    std::vector<double> thetas;
    std::vector<double> R_theta;
};

std::optional<DetectResult> detect_limbus(const cv::Mat& image,
                                          cv::Point2d seed_center,
                                          int n_angles = defaults::N,
                                          int m_harmonics = defaults::M,
                                          double gradient_sigma = defaults::GRADIENT_SIGMA,
                                          double radial_smoothing = defaults::RADIAL_SMOOTHING,
                                          bool skip_eyelid_wedges = defaults::SKIP_EYELID_WEDGES,
                                          double r_min = defaults::R_MIN,
                                          double r_max = defaults::R_MAX);

} // namespace cheshm::Daugman::active_contour
