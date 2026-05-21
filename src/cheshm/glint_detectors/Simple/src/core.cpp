// Simple glint detector. Threshold → pupil-centred search disk →
// ROI / half-plane filtering → shape-quality walk → optional
// widest-blob split → centre per glint.

#include "Simple/defaults.hpp"
#include "cheshm/roi.hpp"
#include "cheshm/spline.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace cheshm::SimpleGlint {
namespace {

constexpr int CENTER_CONVEX_HULL_CENTROID = 0;
constexpr int CENTER_OF_MASS = 1;
constexpr int CENTER_ELLIPSE_FIT = 2;
constexpr int CENTER_MIN_AREA_RECT = 3;
constexpr int CENTER_HULL_MOMENTS = 4;

struct Candidate {
    std::vector<cv::Point> contour;
    bool has_fit;
    cv::RotatedRect fit;
};

bool passes_shape_quality(
    const std::vector<cv::Point> &contour,
    bool has_fit,
    const cv::RotatedRect &ellipse_fit,
    double min_ellipse_fit_ratio,
    double min_roundness_ratio)
{
    if (min_ellipse_fit_ratio < 0.0 && min_roundness_ratio < 0.0) {
        return true;
    }
    const double area = cv::contourArea(contour);
    if (min_ellipse_fit_ratio >= 0.0) {
        if (!has_fit) {
            return false;
        }
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

bool passes_half_plane(
    const std::vector<cv::Point> &contour,
    double cx,
    double cy,
    bool keep_above,
    bool keep_below,
    bool keep_left,
    bool keep_right,
    int margin_px)
{
    const cv::Moments m = cv::moments(contour);
    if (m.m00 <= 0.0) {
        return false;
    }
    const double gx = m.m10 / m.m00;
    const double gy = m.m01 / m.m00;

    bool vertical_ok;
    if (keep_above && keep_below) {
        vertical_ok = true;
    } else if (keep_above) {
        vertical_ok = gy < cy + margin_px;
    } else if (keep_below) {
        vertical_ok = gy > cy - margin_px;
    } else {
        vertical_ok = false;
    }

    bool horizontal_ok;
    if (keep_left && keep_right) {
        horizontal_ok = true;
    } else if (keep_left) {
        horizontal_ok = gx < cx + margin_px;
    } else if (keep_right) {
        horizontal_ok = gx > cx - margin_px;
    } else {
        horizontal_ok = false;
    }

    return vertical_ok && horizontal_ok;
}

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

// Splits the widest contour's bounding-box region into two halves in the
// source `glint_mask` and replaces the widest contour with the contours
// found in each half. The source mask is the pre-search-disk threshold
// image so the bright region's full extent is visible.
std::vector<Candidate> split_widest(
    std::vector<Candidate> &&accepted,
    const cv::Mat &glint_mask)
{
    if (accepted.empty()) {
        return std::move(accepted);
    }

    std::vector<int> widths;
    widths.reserve(accepted.size());
    for (const Candidate &c : accepted) {
        widths.push_back(cv::boundingRect(c.contour).width);
    }
    const int widest_idx = static_cast<int>(std::max_element(widths.begin(), widths.end()) - widths.begin());
    std::vector<int> sorted_widths = widths;
    std::sort(sorted_widths.begin(), sorted_widths.end());
    const int median_w = sorted_widths[sorted_widths.size() / 2];
    if (widths[widest_idx] <= 1.3 * median_w) {
        return std::move(accepted);
    }

    const cv::Rect wide_rect = cv::boundingRect(accepted[widest_idx].contour);
    const int mid_x = wide_rect.x + wide_rect.width / 2;

    std::vector<Candidate> result;
    result.reserve(accepted.size() + 1);
    for (int i = 0; i < static_cast<int>(accepted.size()); ++i) {
        if (i != widest_idx) {
            result.push_back(std::move(accepted[i]));
        }
    }

    const cv::Rect halves[2] = {
        cv::Rect(wide_rect.x, wide_rect.y, mid_x - wide_rect.x, wide_rect.height),
        cv::Rect(mid_x, wide_rect.y, wide_rect.x + wide_rect.width - mid_x, wide_rect.height),
    };
    for (const cv::Rect &half_rect : halves) {
        if (half_rect.width <= 0 || half_rect.height <= 0) {
            continue;
        }
        cv::Mat half_canvas = cv::Mat::zeros(glint_mask.size(), CV_8U);
        glint_mask(half_rect).copyTo(half_canvas(half_rect));
        std::vector<std::vector<cv::Point>> half_contours;
        cv::findContours(half_canvas, half_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (std::vector<cv::Point> &hc : half_contours) {
            Candidate cand;
            cand.has_fit = hc.size() >= 5;
            if (cand.has_fit) {
                cand.fit = cv::fitEllipse(hc);
            }
            cand.contour = std::move(hc);
            result.push_back(std::move(cand));
        }
    }
    return result;
}

bool compute_center(
    const Candidate &cand,
    int method,
    double &cx,
    double &cy)
{
    switch (method) {
        case CENTER_CONVEX_HULL_CENTROID: {
            const std::vector<cv::Point> hull = hull_in_contour_order(cand.contour);
            const cheshm::SplineCentroid sc = cheshm::spline_polygon_centroid(hull, 200);
            if (sc.area == 0.0) {
                return false;
            }
            cx = sc.cx;
            cy = sc.cy;
            return true;
        }
        case CENTER_OF_MASS: {
            const cv::Moments m = cv::moments(cand.contour);
            if (m.m00 <= 0.0) {
                return false;
            }
            cx = m.m10 / m.m00;
            cy = m.m01 / m.m00;
            return true;
        }
        case CENTER_ELLIPSE_FIT:
            if (!cand.has_fit) {
                return false;
            }
            cx = cand.fit.center.x;
            cy = cand.fit.center.y;
            return true;
        case CENTER_MIN_AREA_RECT: {
            const cv::RotatedRect rect = cv::minAreaRect(cand.contour);
            cx = rect.center.x;
            cy = rect.center.y;
            return true;
        }
        case CENTER_HULL_MOMENTS: {
            std::vector<cv::Point> hull;
            cv::convexHull(cand.contour, hull);
            const cv::Moments m = cv::moments(hull);
            if (m.m00 <= 0.0) {
                return false;
            }
            cx = m.m10 / m.m00;
            cy = m.m01 / m.m00;
            return true;
        }
        default:
            return false;
    }
}

struct Glint {
    double cx;
    double cy;
    bool has_fit;
    double e_cx;
    double e_cy;
    double e_w;
    double e_h;
    double e_angle;
    std::vector<cv::Point> contour;  // already in full-image coords
};

struct DetectResult {
    std::vector<Glint> glints;
    cv::Mat search_mask;  // full-image canvas, ROI region populated
};

std::optional<DetectResult> detect_impl(
    const cv::Mat &full,
    int roi_x, int roi_y, int roi_w, int roi_h,
    bool has_pupil, double pupil_cx, double pupil_cy, double pupil_radius,
    int glint_threshold,
    double search_radius_factor, int search_radius_max_px,
    int glint_center_method,
    int max_area_px,
    bool keep_above, bool keep_below, bool keep_left, bool keep_right,
    int filter_margin_px,
    int glints_target,
    bool split_widest_for_target,
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

    cv::Mat glint_mask;
    cv::threshold(view, glint_mask, glint_threshold, 255, cv::THRESH_BINARY);

    cv::Mat search_mask(view.size(), CV_8U);
    if (has_pupil) {
        double radius = search_radius_factor * pupil_radius;
        if (search_radius_max_px >= 0) {
            radius = std::min(radius, static_cast<double>(search_radius_max_px));
        }
        const int radius_int = std::max(cvRound(radius), 0);
        search_mask.setTo(cv::Scalar(0));
        const cv::Point centre_local(cvRound(pupil_cx) - crop.x, cvRound(pupil_cy) - crop.y);
        cv::circle(search_mask, centre_local, radius_int, cv::Scalar(255), -1);
    } else {
        search_mask.setTo(cv::Scalar(255));
    }

    cv::Mat candidates_mask;
    cv::bitwise_and(glint_mask, search_mask, candidates_mask);

    cv::Mat search_canvas = cv::Mat::zeros(height, width, CV_8U);
    candidates_mask.copyTo(search_canvas(crop));

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(candidates_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (max_area_px >= 0) {
        const double cap = static_cast<double>(max_area_px);
        std::vector<std::vector<cv::Point>> kept;
        kept.reserve(contours.size());
        for (auto &c : contours) {
            if (cv::contourArea(c) <= cap) {
                kept.push_back(std::move(c));
            }
        }
        contours = std::move(kept);
    }

    const bool half_plane_active =
        has_pupil && !(keep_above && keep_below && keep_left && keep_right);
    if (half_plane_active) {
        const double cx_local = pupil_cx - crop.x;
        const double cy_local = pupil_cy - crop.y;
        std::vector<std::vector<cv::Point>> kept;
        kept.reserve(contours.size());
        for (auto &c : contours) {
            if (passes_half_plane(c, cx_local, cy_local,
                                  keep_above, keep_below, keep_left, keep_right,
                                  filter_margin_px)) {
                kept.push_back(std::move(c));
            }
        }
        contours = std::move(kept);
    }

    std::vector<Candidate> accepted;
    accepted.reserve(contours.size());
    for (auto &c : contours) {
        Candidate cand;
        cand.has_fit = c.size() >= 5;
        if (cand.has_fit) {
            cand.fit = cv::fitEllipse(c);
        }
        if (!passes_shape_quality(c, cand.has_fit, cand.fit, min_ellipse_fit_ratio, min_roundness_ratio)) {
            continue;
        }
        cand.contour = std::move(c);
        accepted.push_back(std::move(cand));
    }

    if (split_widest_for_target
        && glints_target > 1
        && static_cast<int>(accepted.size()) == glints_target - 1
        && !accepted.empty()) {
        accepted = split_widest(std::move(accepted), glint_mask);
    }

    std::sort(accepted.begin(), accepted.end(), [](const Candidate &a, const Candidate &b) {
        return cv::contourArea(a.contour) > cv::contourArea(b.contour);
    });
    if (static_cast<int>(accepted.size()) > std::max(glints_target, 0)) {
        accepted.resize(std::max(glints_target, 0));
    }

    std::sort(accepted.begin(), accepted.end(), [](const Candidate &a, const Candidate &b) {
        return cv::boundingRect(a.contour).x < cv::boundingRect(b.contour).x;
    });

    DetectResult result;
    result.glints.reserve(accepted.size());
    result.search_mask = std::move(search_canvas);
    for (const Candidate &cand : accepted) {
        double cx_local = 0.0;
        double cy_local = 0.0;
        if (!compute_center(cand, glint_center_method, cx_local, cy_local)) {
            continue;
        }
        Glint g{};
        g.cx = cx_local + crop.x;
        g.cy = cy_local + crop.y;
        g.has_fit = cand.has_fit;
        if (cand.has_fit) {
            g.e_cx = cand.fit.center.x + crop.x;
            g.e_cy = cand.fit.center.y + crop.y;
            g.e_w = cand.fit.size.width;
            g.e_h = cand.fit.size.height;
            g.e_angle = cand.fit.angle;
        }
        g.contour.reserve(cand.contour.size());
        for (const cv::Point &p : cand.contour) {
            g.contour.emplace_back(p.x + crop.x, p.y + crop.y);
        }
        result.glints.push_back(std::move(g));
    }
    return result;
}

}  // namespace
}  // namespace cheshm::SimpleGlint

namespace {

// Pack a single glint into the tuple ``(cx, cy, ellipse_or_None, contour)``.
// ``ellipse_or_None`` is a 5-tuple ``(cx, cy, w, h, angle)`` or ``None``
// when the contour had fewer than five points. ``contour`` is an
// ``(N, 2)`` float64 ndarray in full-image coordinates.
nb::tuple pack_glint(const cheshm::SimpleGlint::Glint &g)
{
    const int n_pts = static_cast<int>(g.contour.size());
    auto contour_owner = std::make_unique<std::vector<double>>(2 * n_pts);
    for (int i = 0; i < n_pts; ++i) {
        (*contour_owner)[2 * i] = static_cast<double>(g.contour[i].x);
        (*contour_owner)[2 * i + 1] = static_cast<double>(g.contour[i].y);
    }
    double *contour_data = contour_owner->data();
    nb::capsule contour_cap(contour_owner.release(),
                            [](void *p) noexcept { delete static_cast<std::vector<double> *>(p); });
    const std::size_t contour_shape[2] = {static_cast<std::size_t>(n_pts), 2};
    nb::ndarray<nb::numpy, double, nb::ndim<2>> contour_arr(
        contour_data, 2, contour_shape, contour_cap);

    nb::object ellipse_obj;
    if (g.has_fit) {
        ellipse_obj = nb::make_tuple(g.e_cx, g.e_cy, g.e_w, g.e_h, g.e_angle);
    } else {
        ellipse_obj = nb::none();
    }
    return nb::make_tuple(g.cx, g.cy, std::move(ellipse_obj), std::move(contour_arr));
}

// Returns ``None`` on error (e.g. ROI fully outside the image),
// otherwise a 2-tuple ``(list_of_glints, search_mask)`` where
// search_mask is ``(H, W)`` uint8 with the post-filter mask.
nb::object detect(
    nb::ndarray<const std::uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> img,
    int roi_x, int roi_y, int roi_w, int roi_h,
    int has_pupil,
    double pupil_cx, double pupil_cy, double pupil_radius,
    int glint_threshold,
    double search_radius_factor, int search_radius_max_px,
    int glint_center_method,
    int max_area_px,
    int keep_above, int keep_below, int keep_left, int keep_right,
    int filter_margin_px,
    int glints_target,
    int split_widest_for_target,
    double min_ellipse_fit_ratio,
    double min_roundness_ratio)
{
    const int height = static_cast<int>(img.shape(0));
    const int width = static_cast<int>(img.shape(1));
    const cv::Mat full(height, width, CV_8U,
                       const_cast<std::uint8_t *>(img.data()));

    auto result = cheshm::SimpleGlint::detect_impl(
        full,
        roi_x, roi_y, roi_w, roi_h,
        has_pupil != 0, pupil_cx, pupil_cy, pupil_radius,
        glint_threshold,
        search_radius_factor, search_radius_max_px,
        glint_center_method,
        max_area_px,
        keep_above != 0, keep_below != 0, keep_left != 0, keep_right != 0,
        filter_margin_px,
        glints_target,
        split_widest_for_target != 0,
        min_ellipse_fit_ratio,
        min_roundness_ratio);
    if (!result) {
        return nb::none();
    }

    nb::list glint_list;
    for (const cheshm::SimpleGlint::Glint &g : result->glints) {
        glint_list.append(pack_glint(g));
    }

    auto mask_owner = std::make_unique<std::vector<std::uint8_t>>(
        static_cast<std::size_t>(height) * static_cast<std::size_t>(width));
    std::memcpy(mask_owner->data(), result->search_mask.data, mask_owner->size());
    std::uint8_t *mask_data = mask_owner->data();
    nb::capsule mask_cap(mask_owner.release(),
                         [](void *p) noexcept { delete static_cast<std::vector<std::uint8_t> *>(p); });
    const std::size_t mask_shape[2] = {static_cast<std::size_t>(height), static_cast<std::size_t>(width)};
    nb::ndarray<nb::numpy, std::uint8_t, nb::ndim<2>> mask_arr(
        mask_data, 2, mask_shape, mask_cap);

    return nb::make_tuple(std::move(glint_list), std::move(mask_arr));
}

}  // namespace

NB_MODULE(_core, m)
{
    namespace d = cheshm::SimpleGlint::defaults;

    m.def("detect", &detect,
          "img"_a, "roi_x"_a, "roi_y"_a, "roi_w"_a, "roi_h"_a,
          "has_pupil"_a, "pupil_cx"_a, "pupil_cy"_a, "pupil_radius"_a,
          "glint_threshold"_a,
          "search_radius_factor"_a, "search_radius_max_px"_a,
          "glint_center_method"_a,
          "max_area_px"_a,
          "keep_above"_a, "keep_below"_a, "keep_left"_a, "keep_right"_a,
          "filter_margin_px"_a,
          "glints_target"_a,
          "split_widest_for_target"_a,
          "min_ellipse_fit_ratio"_a,
          "min_roundness_ratio"_a);

    m.attr("GLINT_THRESHOLD") = d::GLINT_THRESHOLD;
    m.attr("SEARCH_RADIUS_FACTOR") = d::SEARCH_RADIUS_FACTOR;
    m.attr("FILTER_MARGIN_PX") = d::FILTER_MARGIN_PX;
    m.attr("GLINTS_TARGET") = d::GLINTS_TARGET;
    m.attr("KEEP_ABOVE") = d::KEEP_ABOVE;
    m.attr("KEEP_BELOW") = d::KEEP_BELOW;
    m.attr("KEEP_LEFT") = d::KEEP_LEFT;
    m.attr("KEEP_RIGHT") = d::KEEP_RIGHT;
    m.attr("SPLIT_WIDEST_FOR_TARGET") = d::SPLIT_WIDEST_FOR_TARGET;
}
