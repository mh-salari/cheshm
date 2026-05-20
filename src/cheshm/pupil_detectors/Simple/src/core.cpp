// Simple pupil detector — nanobind extension. Threshold → contour-find
// → border / area filter → shape-quality walk → convex-hull-ellipse fit
// → centre via the requested method. ROI is applied as a zero-copy view
// via the shared cpp/common helpers.

#include "cheshm/roi.hpp"
#include "cheshm/spline.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

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

// Convex hull walked in source-contour traversal order. Anchors the
// periodic-spline parameterisation at the contour's starting vertex,
// matching the scipy reference exactly.
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

struct DetectResult {
    double cx;       // centre, full-image coords, from the chosen method
    double cy;
    double e_w;      // ellipse width / height / angle from cv::fitEllipse
    double e_h;
    double e_angle;
    std::vector<cv::Point> contour;  // crop-local; caller shifts by crop.tl()
    cv::Mat mask;                    // full-image canvas, ROI region populated
    cv::Point crop_tl;
};

std::optional<DetectResult> detect_impl(
    const cv::Mat &full,
    int roi_x, int roi_y, int roi_w, int roi_h,
    int pupil_threshold,
    int pupil_center_method,
    double min_ellipse_fit_ratio,
    double min_roundness_ratio)
{
    const int width = full.cols;
    const int height = full.rows;

    cv::Rect crop(0, 0, width, height);
    bool roi_active = false;
    if (cheshm::roi_is_active(roi_w, roi_h)) {
        crop = cheshm::clamp_roi(roi_x, roi_y, roi_w, roi_h, width, height);
        if (crop.area() == 0) {
            return std::nullopt;
        }
        roi_active = true;
    }
    const cv::Mat view = full(crop);

    cv::Mat mask_canvas = cv::Mat::zeros(height, width, CV_8U);
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
        return std::nullopt;
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
        return std::nullopt;
    }

    const std::vector<cv::Point> &pupil_contour = contours[winning];

    double cx_local = 0.0;
    double cy_local = 0.0;
    switch (pupil_center_method) {
        case CENTER_CONVEX_HULL_CENTROID: {
            const cheshm::SplineCentroid sc = cheshm::spline_polygon_centroid(hull, 200);
            if (sc.area == 0.0) {
                return std::nullopt;
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
                return std::nullopt;
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
                return std::nullopt;
            }
            cx_local = m.m10 / m.m00;
            cy_local = m.m01 / m.m00;
            break;
        }
        default:
            return std::nullopt;
    }

    return DetectResult{
        .cx = cx_local + crop.x,
        .cy = cy_local + crop.y,
        .e_w = ellipse_fit.size.width,
        .e_h = ellipse_fit.size.height,
        .e_angle = ellipse_fit.angle,
        .contour = pupil_contour,
        .mask = mask_canvas,
        .crop_tl = crop.tl(),
    };
}

}  // namespace
}  // namespace cheshm::Simple

namespace {

// nanobind binding. Returns ``None`` on failure or a 4-tuple on
// success: ``((cx, cy), (e_cx, e_cy, e_w, e_h, e_angle), contour, mask)``.
//   contour: ``(N, 2)`` float64 ndarray, full-image coords.
//   mask:    ``(H, W)`` uint8 ndarray, full-image canvas with non-ROI zeroed.
nb::object detect(
    nb::ndarray<const std::uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> img,
    int roi_x, int roi_y, int roi_w, int roi_h,
    int pupil_threshold,
    int pupil_center_method,
    double min_ellipse_fit_ratio,
    double min_roundness_ratio,
    int max_contour_points)
{
    const int height = static_cast<int>(img.shape(0));
    const int width = static_cast<int>(img.shape(1));
    const cv::Mat full(height, width, CV_8U,
                       const_cast<std::uint8_t *>(img.data()));

    auto result = cheshm::Simple::detect_impl(
        full,
        roi_x, roi_y, roi_w, roi_h,
        pupil_threshold,
        pupil_center_method,
        min_ellipse_fit_ratio,
        min_roundness_ratio);
    if (!result) {
        return nb::none();
    }

    // contour: copy crop-local int points into a fresh contiguous double
    // buffer in full-image coords; wrap as numpy with capsule-managed
    // ownership so Python frees the buffer on GC.
    const int n_pts = std::min(static_cast<int>(result->contour.size()), max_contour_points);
    auto contour_owner = std::make_unique<std::vector<double>>(2 * n_pts);
    for (int i = 0; i < n_pts; ++i) {
        (*contour_owner)[2 * i] = static_cast<double>(result->contour[i].x) + result->crop_tl.x;
        (*contour_owner)[2 * i + 1] = static_cast<double>(result->contour[i].y) + result->crop_tl.y;
    }
    double *contour_data = contour_owner->data();
    nb::capsule contour_cap(contour_owner.release(),
                            [](void *p) noexcept { delete static_cast<std::vector<double> *>(p); });
    const std::size_t contour_shape[2] = {static_cast<std::size_t>(n_pts), 2};
    nb::ndarray<nb::numpy, double, nb::ndim<2>> contour_arr(
        contour_data, 2, contour_shape, contour_cap);

    // mask: clone into a fresh buffer so the cv::Mat (which owns the
    // memory) can drop here. Same capsule pattern.
    auto mask_owner = std::make_unique<std::vector<std::uint8_t>>(
        static_cast<std::size_t>(height) * static_cast<std::size_t>(width));
    std::memcpy(mask_owner->data(), result->mask.data, mask_owner->size());
    std::uint8_t *mask_data = mask_owner->data();
    nb::capsule mask_cap(mask_owner.release(),
                         [](void *p) noexcept { delete static_cast<std::vector<std::uint8_t> *>(p); });
    const std::size_t mask_shape[2] = {static_cast<std::size_t>(height), static_cast<std::size_t>(width)};
    nb::ndarray<nb::numpy, std::uint8_t, nb::ndim<2>> mask_arr(
        mask_data, 2, mask_shape, mask_cap);

    return nb::make_tuple(
        nb::make_tuple(result->cx, result->cy),
        nb::make_tuple(result->cx, result->cy, result->e_w, result->e_h, result->e_angle),
        std::move(contour_arr),
        std::move(mask_arr));
}

}  // namespace

NB_MODULE(_core, m)
{
    m.def("detect", &detect,
          "img"_a, "roi_x"_a, "roi_y"_a, "roi_w"_a, "roi_h"_a,
          "pupil_threshold"_a, "pupil_center_method"_a,
          "min_ellipse_fit_ratio"_a, "min_roundness_ratio"_a,
          "max_contour_points"_a);
}
