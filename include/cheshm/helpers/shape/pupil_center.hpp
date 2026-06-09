// Pupil-centre estimators, shared by the image detectors and the
// manual-annotation path.
//
// All methods but CENTER_OF_MASS are pure geometry: they work from the
// contour / convex hull / fitted ellipse alone, so they apply equally to
// a detector's dense contour and to a handful of hand-placed boundary
// points. CENTER_OF_MASS additionally needs an intensity mask to weight
// the centroid, so it is only meaningful when an image is available.

#pragma once

#include "cheshm/helpers/shape/spline.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <vector>

namespace cheshm
{

constexpr int CENTER_CONVEX_HULL_CENTROID = 0;
constexpr int CENTER_OF_MASS = 1;
constexpr int CENTER_ELLIPSE_FIT = 2;
constexpr int CENTER_MIN_AREA_RECT = 3;
constexpr int CENTER_HULL_MOMENTS = 4;

// Pupil centre from one contour/hull/ellipse, by the requested method.
// ``intensity_mask`` is the crop-local mask the centre-of-mass method
// intersects the filled contour with (the threshold mask for the raw
// contour, or the filled smoothed margin when Fourier smoothing is on).
inline std::optional<cv::Point2d> pupil_center(int method,
                                               const std::vector<cv::Point>& contour,
                                               const std::vector<cv::Point>& hull,
                                               const cv::RotatedRect& ellipse_fit,
                                               const cv::Mat& intensity_mask)
{
    switch (method)
    {
        case CENTER_CONVEX_HULL_CENTROID:
        {
            const cheshm::SplineCentroid sc = cheshm::spline_polygon_centroid(hull, 200);
            if (sc.area == 0.0)
            {
                return std::nullopt;
            }
            return cv::Point2d(sc.cx, sc.cy);
        }
        case CENTER_OF_MASS:
        {
            cv::Mat contour_mask = cv::Mat::zeros(intensity_mask.size(), CV_8U);
            cv::drawContours(
                contour_mask, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(255), cv::FILLED);
            cv::Mat region;
            cv::bitwise_and(contour_mask, intensity_mask, region);
            const cv::Moments m = cv::moments(region, true);
            if (m.m00 == 0.0)
            {
                return std::nullopt;
            }
            return cv::Point2d(m.m10 / m.m00, m.m01 / m.m00);
        }
        case CENTER_ELLIPSE_FIT:
            return cv::Point2d(ellipse_fit.center.x, ellipse_fit.center.y);
        case CENTER_MIN_AREA_RECT:
        {
            const cv::RotatedRect rect = cv::minAreaRect(contour);
            return cv::Point2d(rect.center.x, rect.center.y);
        }
        case CENTER_HULL_MOMENTS:
        {
            const cv::Moments m = cv::moments(hull);
            if (m.m00 == 0.0)
            {
                return std::nullopt;
            }
            return cv::Point2d(m.m10 / m.m00, m.m01 / m.m00);
        }
        default:
            return std::nullopt;
    }
}

} // namespace cheshm
