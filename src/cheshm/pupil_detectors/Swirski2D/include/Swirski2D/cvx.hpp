// OpenCV helpers used by Swirski 2D — small wrappers + bounded-ROI
// access. Header-only declarations; implementations in src/cvx.cpp.

#pragma once

#include "Swirski2D/utils.hpp"

#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace cheshm::Swirski2D
{

const double SQRT_2 = std::sqrt(2.0);
const double PI = CV_PI;

namespace cvx
{

template <typename T>
inline cv::Rect_<T> roiAround(T x, T y, T radius)
{
    return cv::Rect_<T>(x - radius, y - radius, 2 * radius + 1, 2 * radius + 1);
}
template <typename T>
inline cv::Rect_<T> roiAround(const cv::Point_<T>& centre, T radius)
{
    return roiAround(centre.x, centre.y, radius);
}

inline cv::Rect boundingBox(const cv::Mat& img)
{
    return cv::Rect(0, 0, img.cols, img.rows);
}

void getROI(const cv::Mat& src, cv::Mat& dst, const cv::Rect& roi, int borderType = cv::BORDER_REPLICATE);

float histKmeans(const cv::Mat_<float>& hist,
                 int bin_min,
                 int bin_max,
                 int K,
                 float init_centres[],
                 cv::Mat_<uchar>& labels,
                 cv::TermCriteria termCriteria);

cv::RotatedRect fitEllipse(const cv::Moments& m);
cv::Vec2f majorAxis(const cv::RotatedRect& ellipse);

} // namespace cvx

} // namespace cheshm::Swirski2D
