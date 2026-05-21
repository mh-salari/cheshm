// Per-angle radial gradient argmax used by both the Daugman 2007
// active contour and its pupil-shape-prior variant. For each angle in
// the range [0, n_angles), bilinearly samples the radial-direction
// gradient g_r = G_x cos theta + G_y sin theta at n_r radii, smooths
// the profile with the supplied 1-D Gaussian kernel, and writes the
// argmax radius. Angles fully out of bounds, or flagged in
// eyelid_mask, are written as -1.

#pragma once

#include <cstdint>
#include <opencv2/core.hpp>
#include <span>
#include <vector>

namespace cheshm::Daugman::active_contour
{

std::vector<double> radial_search(const cv::Mat& gx,
                                  const cv::Mat& gy,
                                  cv::Point2f seed_center,
                                  int n_angles,
                                  float r_min,
                                  float r_max,
                                  int n_r,
                                  std::span<const std::uint8_t> eyelid_mask,
                                  std::span<const double> smoothing_kernel);

} // namespace cheshm::Daugman::active_contour
