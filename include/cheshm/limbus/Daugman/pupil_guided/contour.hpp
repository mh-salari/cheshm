// Pupil-shape-guided active contour for iris boundary localization.
//
// Extends Daugman 2007 radial-gradient search with anisotropic
// per-angle search bounds derived from the pupil ellipse. The pupil
// and limbus are projections of (nearly) co-planar circles on the
// front of the eye, so they share elongation direction and aspect
// ratio; using that shape as a search-window prior keeps the radial
// search inside the iris.

#pragma once

#include "cheshm/limbus/Daugman/pupil_guided/defaults.hpp"

#include <opencv2/core.hpp>
#include <optional>
#include <vector>

namespace cheshm::Daugman::pupil_guided
{

struct PupilEllipse
{
    cv::Point2d center; // unused for the limbus fit, kept for API parity
    cv::Size2d size;    // full width and height (2 * a_p, 2 * b_p)
    double angle_deg;   // cv::fitEllipse-style: angle of the width axis
};

struct DetectResult
{
    cv::Point2d center;
    std::vector<double> thetas;
    std::vector<double> R_theta;
};

std::optional<DetectResult> detect_limbus(const cv::Mat& image,
                                          cv::Point2d seed_center,
                                          const PupilEllipse& pupil,
                                          int n_angles = defaults::N,
                                          int m_harmonics = defaults::M,
                                          double gradient_sigma = defaults::GRADIENT_SIGMA,
                                          double radial_smoothing = defaults::RADIAL_SMOOTHING,
                                          double k_min = defaults::K_MIN,
                                          double k_max = defaults::K_MAX);

} // namespace cheshm::Daugman::pupil_guided
