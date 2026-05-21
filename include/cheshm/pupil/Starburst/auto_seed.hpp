// Initial pupil-seed centre: centroid of the largest interior dark
// blob. Thresholds the image at ``seed_threshold``, drops contours
// that touch the image border (eyelashes / frame vignette), and takes
// the largest remaining contour's moments centroid. Falls back to the
// image centre when no candidate is found.

#pragma once

#include <opencv2/core.hpp>

namespace cheshm::Starburst
{

cv::Point2d auto_seed(const cv::Mat& image, int seed_threshold);

} // namespace cheshm::Starburst
