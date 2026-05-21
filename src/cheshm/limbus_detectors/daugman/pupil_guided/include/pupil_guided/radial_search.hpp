// Per-angle radial-gradient argmax with anisotropic search bounds.
// Each angle gets its own [r_min[i], r_max[i]] range, derived by the
// caller from the pupil ellipse.

#pragma once

#include <opencv2/core.hpp>
#include <span>
#include <vector>

namespace cheshm::Daugman::pupil_guided
{

std::vector<double> radial_search(const cv::Mat& gx,
                                  const cv::Mat& gy,
                                  cv::Point2f seed_center,
                                  int n_angles,
                                  std::span<const float> r_min_per_angle,
                                  std::span<const float> r_max_per_angle,
                                  int n_r,
                                  std::span<const double> smoothing_kernel);

} // namespace cheshm::Daugman::pupil_guided
