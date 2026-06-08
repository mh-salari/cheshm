// Simple pupil detector. Threshold → contour-find → border / area
// filter → shape-quality walk → convex-hull-ellipse fit → centre via
// the requested method.

#include "cheshm/helpers/image/roi.hpp"
#include "cheshm/helpers/shape/pupil_form.hpp"
#include "cheshm/helpers/shape/shape_quality.hpp"
#include "cheshm/helpers/shape/spline.hpp"
#include "cheshm/pupil/Simple/defaults.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace cheshm::Simple
{
namespace
{

constexpr int CENTER_CONVEX_HULL_CENTROID = 0;
constexpr int CENTER_OF_MASS = 1;
constexpr int CENTER_ELLIPSE_FIT = 2;
constexpr int CENTER_MIN_AREA_RECT = 3;
constexpr int CENTER_HULL_MOMENTS = 4;

bool touches_border(const std::vector<cv::Point>& contour, int view_w, int view_h)
{
    const cv::Rect br = cv::boundingRect(contour);
    return br.x == 0 || br.y == 0 || br.x + br.width == view_w || br.y + br.height == view_h;
}

// Convex hull walked in source-contour traversal order. Anchors the
// periodic-spline parameterisation at the contour's starting vertex.
std::vector<cv::Point> hull_in_contour_order(const std::vector<cv::Point>& contour)
{
    std::vector<int> indices;
    cv::convexHull(contour, indices, false, false);
    std::sort(indices.begin(), indices.end());
    std::vector<cv::Point> hull;
    hull.reserve(indices.size());
    for (int i : indices)
    {
        hull.push_back(contour[i]);
    }
    return hull;
}

// Pupil centre from one contour/hull/ellipse, by the requested method.
// ``intensity_mask`` is the crop-local mask the centre-of-mass method
// intersects the filled contour with (the threshold mask for the raw
// contour, or the filled smoothed margin when Fourier smoothing is on).
std::optional<cv::Point2d> compute_center(int method,
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

struct DetectResult
{
    double cx; // centre, full-image coords, from the chosen method
    double cy;
    double e_w; // ellipse width / height / angle from cv::fitEllipse
    double e_h;
    double e_angle;
    std::vector<cv::Point> contour; // crop-local; caller shifts by crop.tl()
    cv::Mat mask;                   // full-image canvas, ROI region populated
    cv::Point crop_tl;
};

std::optional<DetectResult> detect_impl(const cv::Mat& full,
                                        int roi_x,
                                        int roi_y,
                                        int roi_w,
                                        int roi_h,
                                        int pupil_threshold,
                                        int pupil_center_method,
                                        double min_ellipse_fit_ratio,
                                        double min_roundness_ratio,
                                        bool fourier_smoothing,
                                        int fourier_harmonics,
                                        int fourier_samples,
                                        int fourier_iterations,
                                        double fourier_inward_rejection)
{
    const int width = full.cols;
    const int height = full.rows;

    cv::Rect crop(0, 0, width, height);
    bool roi_active = false;
    if (cheshm::roi_is_active(roi_w, roi_h))
    {
        crop = cheshm::clamp_roi(roi_x, roi_y, roi_w, roi_h, width, height);
        if (crop.area() == 0)
        {
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
    for (int i = 0; i < static_cast<int>(contours.size()); ++i)
    {
        if (cv::contourArea(contours[i]) <= 0.0)
        {
            continue;
        }
        if (!roi_active && touches_border(contours[i], view.cols, view.rows))
        {
            continue;
        }
        indices.push_back(i);
    }
    if (indices.empty())
    {
        return std::nullopt;
    }

    std::sort(indices.begin(),
              indices.end(),
              [&](int a, int b) { return cv::contourArea(contours[a]) > cv::contourArea(contours[b]); });

    int winning = -1;
    std::vector<cv::Point> hull;
    cv::RotatedRect ellipse_fit;
    for (int idx : indices)
    {
        std::vector<cv::Point> candidate_hull = hull_in_contour_order(contours[idx]);
        if (candidate_hull.size() < 5)
        {
            continue;
        }
        const cv::RotatedRect candidate_fit = cv::fitEllipse(candidate_hull);
        if (!cheshm::passes_shape_quality(contours[idx], candidate_fit, min_ellipse_fit_ratio, min_roundness_ratio))
        {
            continue;
        }
        winning = idx;
        hull = std::move(candidate_hull);
        ellipse_fit = candidate_fit;
        break;
    }
    if (winning < 0)
    {
        return std::nullopt;
    }

    std::vector<cv::Point> out_contour = contours[winning];
    std::vector<cv::Point> out_hull = hull;
    cv::RotatedRect out_ellipse = ellipse_fit;

    if (fourier_smoothing)
    {
        const cheshm::PupilForm form = cheshm::fit_pupil_form(
            contours[winning], fourier_harmonics, fourier_samples, fourier_iterations, fourier_inward_rejection);
        if (form.ok && form.boundary.size() >= 5)
        {
            out_contour = form.boundary;
            out_hull = hull_in_contour_order(out_contour);
            out_ellipse = cv::fitEllipse(out_hull);
            // Refill the returned mask with the smoothed margin so the mask
            // overlay and centre-of-mass match the reported contour.
            pupil_mask.setTo(0);
            cv::drawContours(
                pupil_mask, std::vector<std::vector<cv::Point>>{out_contour}, -1, cv::Scalar(255), cv::FILLED);
        }
    }

    const std::optional<cv::Point2d> center =
        compute_center(pupil_center_method, out_contour, out_hull, out_ellipse, pupil_mask);
    if (!center)
    {
        return std::nullopt;
    }

    return DetectResult{
        .cx = center->x + crop.x,
        .cy = center->y + crop.y,
        .e_w = out_ellipse.size.width,
        .e_h = out_ellipse.size.height,
        .e_angle = out_ellipse.angle,
        .contour = out_contour,
        .mask = mask_canvas,
        .crop_tl = crop.tl(),
    };
}

} // namespace
} // namespace cheshm::Simple

namespace
{

// Returns ``None`` on failure or a 4-tuple on success:
//   ``((cx, cy), (e_cx, e_cy, e_w, e_h, e_angle), contour, mask)``
// where contour is ``(N, 2)`` float64 in full-image coords and mask is
// ``(H, W)`` uint8 with non-ROI zeroed.
nb::object detect(nb::ndarray<const std::uint8_t, nb::c_contig, nb::device::cpu> img,
                  int roi_x,
                  int roi_y,
                  int roi_w,
                  int roi_h,
                  int pupil_threshold,
                  int pupil_center_method,
                  double min_ellipse_fit_ratio,
                  double min_roundness_ratio,
                  bool fourier_smoothing,
                  int fourier_harmonics,
                  int fourier_samples,
                  int fourier_iterations,
                  double fourier_inward_rejection,
                  int max_contour_points)
{
    const int height = static_cast<int>(img.shape(0));
    const int width = static_cast<int>(img.shape(1));
    const bool is_color = (img.ndim() == 3 && img.shape(2) == 3);
    const cv::Mat raw(height, width, is_color ? CV_8UC3 : CV_8U, const_cast<std::uint8_t*>(img.data()));
    cv::Mat full;
    if (is_color)
        cv::cvtColor(raw, full, cv::COLOR_BGR2GRAY);
    else
        full = raw;

    auto result = cheshm::Simple::detect_impl(full,
                                              roi_x,
                                              roi_y,
                                              roi_w,
                                              roi_h,
                                              pupil_threshold,
                                              pupil_center_method,
                                              min_ellipse_fit_ratio,
                                              min_roundness_ratio,
                                              fourier_smoothing,
                                              fourier_harmonics,
                                              fourier_samples,
                                              fourier_iterations,
                                              fourier_inward_rejection);
    if (!result)
    {
        return nb::none();
    }

    // contour: copy crop-local int points into a fresh contiguous double
    // buffer in full-image coords; wrap as numpy with capsule-managed
    // ownership so Python frees the buffer on GC.
    const int n_pts = std::min(static_cast<int>(result->contour.size()), max_contour_points);
    auto contour_owner = std::make_unique<std::vector<double>>(2 * n_pts);
    for (int i = 0; i < n_pts; ++i)
    {
        (*contour_owner)[2 * i] = static_cast<double>(result->contour[i].x) + result->crop_tl.x;
        (*contour_owner)[2 * i + 1] = static_cast<double>(result->contour[i].y) + result->crop_tl.y;
    }
    double* contour_data = contour_owner->data();
    nb::capsule contour_cap(contour_owner.release(),
                            [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
    const std::size_t contour_shape[2] = {static_cast<std::size_t>(n_pts), 2};
    nb::ndarray<nb::numpy, double, nb::ndim<2>> contour_arr(contour_data, 2, contour_shape, contour_cap);

    // mask: clone into a fresh buffer so the cv::Mat (which owns the
    // memory) can drop here. Same capsule pattern.
    auto mask_owner = std::make_unique<std::vector<std::uint8_t>>(static_cast<std::size_t>(height) *
                                                                  static_cast<std::size_t>(width));
    std::memcpy(mask_owner->data(), result->mask.data, mask_owner->size());
    std::uint8_t* mask_data = mask_owner->data();
    nb::capsule mask_cap(mask_owner.release(),
                         [](void* p) noexcept { delete static_cast<std::vector<std::uint8_t>*>(p); });
    const std::size_t mask_shape[2] = {static_cast<std::size_t>(height), static_cast<std::size_t>(width)};
    nb::ndarray<nb::numpy, std::uint8_t, nb::ndim<2>> mask_arr(mask_data, 2, mask_shape, mask_cap);

    return nb::make_tuple(nb::make_tuple(result->cx, result->cy),
                          nb::make_tuple(result->cx, result->cy, result->e_w, result->e_h, result->e_angle),
                          std::move(contour_arr),
                          std::move(mask_arr));
}

} // namespace

NB_MODULE(_core, m)
{
    namespace d = cheshm::Simple::defaults;

    m.def("detect",
          &detect,
          "img"_a,
          "roi_x"_a,
          "roi_y"_a,
          "roi_w"_a,
          "roi_h"_a,
          "pupil_threshold"_a,
          "pupil_center_method"_a,
          "min_ellipse_fit_ratio"_a,
          "min_roundness_ratio"_a,
          "fourier_smoothing"_a,
          "fourier_harmonics"_a,
          "fourier_samples"_a,
          "fourier_iterations"_a,
          "fourier_inward_rejection"_a,
          "max_contour_points"_a);

    m.attr("PUPIL_THRESHOLD") = d::PUPIL_THRESHOLD;
    m.attr("MAX_CONTOUR_POINTS") = d::MAX_CONTOUR_POINTS;
    m.attr("FOURIER_SMOOTHING") = d::FOURIER_SMOOTHING;
    m.attr("FOURIER_HARMONICS") = d::FOURIER_HARMONICS;
    m.attr("FOURIER_SAMPLES") = d::FOURIER_SAMPLES;
    m.attr("FOURIER_ITERATIONS") = d::FOURIER_ITERATIONS;
    m.attr("FOURIER_INWARD_REJECTION") = d::FOURIER_INWARD_REJECTION;
}
