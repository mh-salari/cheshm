// Public C surface for the Simple pupil detector — a single extern "C"
// entry point that cheshm's Python wrapper loads via ctypes. Threshold
// → contour-find → border / area filter → shape-quality walk →
// convex-hull-ellipse fit → centre via the requested method. ROI is
// applied as a zero-copy view via the shared cpp/common helpers.

#include "cheshm/roi.hpp"
#include "cheshm/spline.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace cheshm::Simple {
namespace {

constexpr int CENTER_CONVEX_HULL_CENTROID = 0;
constexpr int CENTER_OF_MASS = 1;
constexpr int CENTER_ELLIPSE_FIT = 2;
constexpr int CENTER_MIN_AREA_RECT = 3;
constexpr int CENTER_HULL_MOMENTS = 4;

bool touches_border(const std::vector<cv::Point> &contour, int view_w, int view_h)
{
    const cv::Rect br = cv::boundingRect(contour);
    return br.x == 0 || br.y == 0 || br.x + br.width == view_w || br.y + br.height == view_h;
}

bool passes_shape_quality(
    const std::vector<cv::Point> &contour,
    const cv::RotatedRect &ellipse_fit,
    double min_ellipse_fit_ratio,
    double min_roundness_ratio)
{
    if (min_ellipse_fit_ratio < 0.0 && min_roundness_ratio < 0.0) {
        return true;
    }
    const double area = cv::contourArea(contour);
    if (min_ellipse_fit_ratio >= 0.0) {
        const double ellipse_area = CV_PI * (ellipse_fit.size.width / 2.0) * (ellipse_fit.size.height / 2.0);
        if (ellipse_area <= 0.0) {
            return false;
        }
        if (area / ellipse_area < min_ellipse_fit_ratio) {
            return false;
        }
    }
    if (min_roundness_ratio >= 0.0) {
        const double perimeter = cv::arcLength(contour, true);
        if (perimeter <= 0.0) {
            return false;
        }
        const double roundness = 4.0 * CV_PI * area / (perimeter * perimeter);
        if (roundness < min_roundness_ratio) {
            return false;
        }
    }
    return true;
}

// Returns the convex hull of `contour` as points, walked in the source
// contour's traversal order. This anchors the spline parameter origin
// at the contour's starting vertex.
std::vector<cv::Point> hull_in_contour_order(const std::vector<cv::Point> &contour)
{
    std::vector<int> indices;
    cv::convexHull(contour, indices, false, false);
    std::sort(indices.begin(), indices.end());
    std::vector<cv::Point> hull;
    hull.reserve(indices.size());
    for (int i : indices) {
        hull.push_back(contour[i]);
    }
    return hull;
}

}  // namespace
}  // namespace cheshm::Simple

extern "C" {

// Simple_detect — threshold-based pupil detection.
//
//  img_data                 grayscale uint8 buffer, ``width * height``
//                           bytes, row-major.
//  roi_x, roi_y,            ROI rectangle in full-image coordinates.
//  roi_w, roi_h             When ``roi_w > 0 && roi_h > 0`` the algorithm
//                           runs on the cropped sub-image and contours
//                           touching the crop edge are accepted (an
//                           explicit ROI means "the pupil is here");
//                           otherwise the full frame is processed and
//                           border-touching contours are rejected.
//  pupil_threshold          Pixels below this intensity become pupil.
//  pupil_center_method      0=convex_hull_centroid (periodic cubic
//                           spline through the hull + Green's-theorem
//                           centroid of the sampled curve),
//                           1=center_of_mass (moments of the contour-
//                           masked pupil region with glint cut-outs
//                           preserved), 2=ellipse_fit_center,
//                           3=min_area_rect_center, 4=hull_moments_
//                           centroid (moments of the filled convex
//                           hull polygon, no spline).
//  min_ellipse_fit_ratio    Reject candidates whose contour-to-fitted-
//                           ellipse area ratio is below this. Negative
//                           = gate off.
//  min_roundness_ratio      Reject candidates whose isoperimetric
//                           quotient ``4 pi area / perimeter^2`` is
//                           below this. Negative = gate off.
//  out_center_xy            Two doubles: the pupil centre in full-image
//                           coordinates.
//  out_ellipse_params       Five doubles: centre from the chosen
//                           centring method, width / height / angle
//                           from ``cv::fitEllipse`` on the convex hull.
//                           Centre is in full-image coordinates.
//  out_n_contour_points     Number of contour points written.
//  contour_xy               Caller-allocated buffer of
//                           ``2 * max_contour_points`` doubles for
//                           ``(x, y)`` pairs in full-image coordinates.
//  max_contour_points       Capacity of ``contour_xy`` in points.
//  out_mask                 Caller-allocated ``width * height`` uint8
//                           buffer. Receives the thresholded binary
//                           mask (255 = pupil). When the ROI path is
//                           taken, regions outside the ROI are zeroed.
//
// Returns 1 on success, 0 on failure.
int Simple_detect(
    const std::uint8_t *img_data,
    int width,
    int height,
    int roi_x,
    int roi_y,
    int roi_w,
    int roi_h,
    int pupil_threshold,
    int pupil_center_method,
    double min_ellipse_fit_ratio,
    double min_roundness_ratio,
    double *out_center_xy,
    double *out_ellipse_params,
    int *out_n_contour_points,
    double *contour_xy,
    int max_contour_points,
    std::uint8_t *out_mask)
{
    using namespace cheshm::Simple;  // NOLINT(google-build-using-namespace)

    if (img_data == nullptr || width <= 0 || height <= 0 || out_mask == nullptr) {
        if (out_n_contour_points != nullptr) {
            *out_n_contour_points = 0;
        }
        return 0;
    }

    const cv::Mat full(height, width, CV_8U, const_cast<std::uint8_t *>(img_data));
    cv::Rect crop(0, 0, width, height);
    bool roi_active = false;
    if (cheshm::roi_is_active(roi_w, roi_h)) {
        crop = cheshm::clamp_roi(roi_x, roi_y, roi_w, roi_h, width, height);
        if (crop.area() == 0) {
            *out_n_contour_points = 0;
            return 0;
        }
        roi_active = true;
    }
    const cv::Mat view = full(crop);

    if (roi_active) {
        std::memset(out_mask, 0, static_cast<size_t>(width) * static_cast<size_t>(height));
    }
    cv::Mat mask_canvas(height, width, CV_8U, out_mask);
    cv::Mat pupil_mask = mask_canvas(crop);
    cv::threshold(view, pupil_mask, pupil_threshold, 255, cv::THRESH_BINARY_INV);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(pupil_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<int> indices;
    indices.reserve(contours.size());
    for (int i = 0; i < static_cast<int>(contours.size()); ++i) {
        if (cv::contourArea(contours[i]) <= 0.0) {
            continue;
        }
        if (!roi_active && touches_border(contours[i], view.cols, view.rows)) {
            continue;
        }
        indices.push_back(i);
    }
    if (indices.empty()) {
        *out_n_contour_points = 0;
        return 0;
    }

    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return cv::contourArea(contours[a]) > cv::contourArea(contours[b]);
    });

    int winning = -1;
    std::vector<cv::Point> hull;
    cv::RotatedRect ellipse_fit;
    for (int idx : indices) {
        std::vector<cv::Point> candidate_hull = hull_in_contour_order(contours[idx]);
        if (candidate_hull.size() < 5) {
            continue;
        }
        const cv::RotatedRect candidate_fit = cv::fitEllipse(candidate_hull);
        if (!passes_shape_quality(contours[idx], candidate_fit, min_ellipse_fit_ratio, min_roundness_ratio)) {
            continue;
        }
        winning = idx;
        hull = std::move(candidate_hull);
        ellipse_fit = candidate_fit;
        break;
    }
    if (winning < 0) {
        *out_n_contour_points = 0;
        return 0;
    }

    const std::vector<cv::Point> &pupil_contour = contours[winning];

    double cx_local = 0.0;
    double cy_local = 0.0;
    switch (pupil_center_method) {
        case CENTER_CONVEX_HULL_CENTROID: {
            const cheshm::SplineCentroid sc = cheshm::spline_polygon_centroid(hull, 200);
            if (sc.area == 0.0) {
                *out_n_contour_points = 0;
                return 0;
            }
            cx_local = sc.cx;
            cy_local = sc.cy;
            break;
        }
        case CENTER_OF_MASS: {
            cv::Mat contour_mask = cv::Mat::zeros(pupil_mask.size(), CV_8U);
            cv::drawContours(contour_mask, std::vector<std::vector<cv::Point>>{pupil_contour}, -1, cv::Scalar(255), cv::FILLED);
            cv::Mat pupil_only;
            cv::bitwise_and(pupil_mask, contour_mask, pupil_only);
            const cv::Moments m = cv::moments(pupil_only, true);
            if (m.m00 == 0.0) {
                *out_n_contour_points = 0;
                return 0;
            }
            cx_local = m.m10 / m.m00;
            cy_local = m.m01 / m.m00;
            break;
        }
        case CENTER_ELLIPSE_FIT:
            cx_local = ellipse_fit.center.x;
            cy_local = ellipse_fit.center.y;
            break;
        case CENTER_MIN_AREA_RECT: {
            const cv::RotatedRect rect = cv::minAreaRect(pupil_contour);
            cx_local = rect.center.x;
            cy_local = rect.center.y;
            break;
        }
        case CENTER_HULL_MOMENTS: {
            const cv::Moments m = cv::moments(hull);
            if (m.m00 == 0.0) {
                *out_n_contour_points = 0;
                return 0;
            }
            cx_local = m.m10 / m.m00;
            cy_local = m.m01 / m.m00;
            break;
        }
        default:
            *out_n_contour_points = 0;
            return 0;
    }

    // The ellipse tuple's centre comes from the selected centring method;
    // its size and angle come from cv::fitEllipse on the convex hull.
    out_ellipse_params[0] = cx_local + crop.x;
    out_ellipse_params[1] = cy_local + crop.y;
    out_ellipse_params[2] = ellipse_fit.size.width;
    out_ellipse_params[3] = ellipse_fit.size.height;
    out_ellipse_params[4] = ellipse_fit.angle;
    out_center_xy[0] = cx_local + crop.x;
    out_center_xy[1] = cy_local + crop.y;

    const int n = std::min(static_cast<int>(pupil_contour.size()), max_contour_points);
    for (int i = 0; i < n; ++i) {
        contour_xy[2 * i] = static_cast<double>(pupil_contour[i].x) + crop.x;
        contour_xy[2 * i + 1] = static_cast<double>(pupil_contour[i].y) + crop.y;
    }
    *out_n_contour_points = n;

    return 1;
}

}  // extern "C"
