#pragma once

#include "cheshm/enhance/defaults.hpp"

#include <opencv2/core.hpp>

namespace cheshm::enhance
{

// Contrast Limited Adaptive Histogram Equalization (local contrast).
cv::Mat clahe(const cv::Mat& gray, double clip_limit = defaults::CLAHE_CLIP_LIMIT, int tile = defaults::CLAHE_TILE);

// Linear contrast stretch between the lo/hi intensity percentiles. Robust to
// saturated glints because it clips outliers by percentile rather than min/max.
cv::Mat percentile_stretch(const cv::Mat& gray,
                           double lo_pct = defaults::STRETCH_LO_PCT,
                           double hi_pct = defaults::STRETCH_HI_PCT);

// Power-law mapping out = 255*(in/255)^g. g < 1 lifts shadows (brightens),
// g > 1 deepens them (darkens).
cv::Mat gamma(const cv::Mat& gray, double g = defaults::GAMMA);

// Edge-preserving denoise (smooths sensor noise, keeps the boundary sharp).
cv::Mat bilateral(const cv::Mat& gray,
                  int d = defaults::BILATERAL_D,
                  double sigma_color = defaults::BILATERAL_SIGMA_COLOR,
                  double sigma_space = defaults::BILATERAL_SIGMA_SPACE);

// Unsharp mask: out = (1+amount)*gray - amount*GaussianBlur(gray, sigma).
cv::Mat unsharp(const cv::Mat& gray, double sigma = defaults::UNSHARP_SIGMA, double amount = defaults::UNSHARP_AMOUNT);

} // namespace cheshm::enhance
