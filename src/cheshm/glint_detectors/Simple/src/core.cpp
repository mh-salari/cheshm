// Public C surface for the Simple glint detector — a single extern "C"
// entry point that cheshm's Python wrapper loads via ctypes. Threshold
// → pupil-centred search disk → ROI / half-plane filtering →
// shape-quality walk → optional widest-blob split → centre per glint.
//
// Variable-length output is laid out into caller-allocated buffers:
// each glint occupies a fixed slot for its centre and ellipse, plus a
// fixed-stride slice in the contour buffer whose actual length is
// reported in ``out_contour_lengths``.

#include "cheshm/roi.hpp"
#include "cheshm/spline.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

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
    const cv::Mat &candidates_mask,
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

}  // namespace
}  // namespace cheshm::SimpleGlint

extern "C" {

// Simple_glint_detect — threshold-based glint detection near a pupil.
//
// Inputs:
//  img_data, width, height        grayscale uint8 buffer, row-major.
//  roi_x, roi_y, roi_w, roi_h     ROI rectangle in full-image coords.
//                                 ``roi_w > 0 && roi_h > 0`` activates a
//                                 crop; outputs are translated back.
//  has_pupil                      Non-zero when a pupil centre + radius
//                                 are supplied; gates the search-disk
//                                 mask and the half-plane filter.
//  pupil_cx, pupil_cy, pupil_radius   Pupil position and scale in
//                                 full-image coords.
//  glint_threshold                Pixels above this intensity become
//                                 candidate glint.
//  search_radius_factor, search_radius_max_px
//                                 Search-disk radius around the pupil
//                                 centre = ``min(factor * radius,
//                                 max_px)``. ``max_px = -1`` disables
//                                 the cap.
//  glint_center_method            0..4 matching Simple_pupil.
//  max_area_px                    Reject contours whose area exceeds
//                                 this. -1 = no cap.
//  keep_above / keep_below /      Half-plane filter around the pupil
//  keep_left / keep_right         centre; ignored when all four are set
//                                 or no pupil is supplied.
//  filter_margin_px               Slack on the half-plane boundary.
//  glints_target                  Number of glints to keep.
//  split_widest_for_target        Non-zero enables the widest-blob
//                                 split when ``len(accepted) ==
//                                 glints_target - 1``.
//  min_ellipse_fit_ratio,         Opt-in shape-quality gates. Negative
//  min_roundness_ratio            = gate off.
//
// Outputs:
//  out_n_glints                   Number of glints written.
//  out_centers_xy                 ``2 * max_glints`` doubles. Each
//                                 (cx, cy) in full-image coordinates.
//  out_ellipse_params             ``6 * max_glints`` doubles per glint:
//                                 (cx, cy, w, h, angle, has_fit).
//                                 ``has_fit = 0.0`` when the contour
//                                 had fewer than five points.
//  out_contour_lengths            ``max_glints`` ints; how many points
//                                 the corresponding contour slot uses.
//  out_contours_xy                ``2 * max_glints * max_contour_points
//                                 _per_glint`` doubles; each glint i
//                                 occupies a stride of
//                                 ``2 * max_contour_points_per_glint``
//                                 starting at ``i * stride``.
//  max_glints                     Capacity of the per-glint output
//                                 arrays.
//  max_contour_points_per_glint   Capacity of one glint's contour slot.
//  out_search_mask                ``width * height`` uint8 canvas; the
//                                 ``threshold & search-disk & ROI``
//                                 intersection. Regions outside the
//                                 ROI are zeroed.
//
// Returns 1 on success (even when zero glints are found), 0 on error.
int Simple_glint_detect(
    const std::uint8_t *img_data,
    int width,
    int height,
    int roi_x,
    int roi_y,
    int roi_w,
    int roi_h,
    int has_pupil,
    double pupil_cx,
    double pupil_cy,
    double pupil_radius,
    int glint_threshold,
    double search_radius_factor,
    int search_radius_max_px,
    int glint_center_method,
    int max_area_px,
    int keep_above,
    int keep_below,
    int keep_left,
    int keep_right,
    int filter_margin_px,
    int glints_target,
    int split_widest_for_target,
    double min_ellipse_fit_ratio,
    double min_roundness_ratio,
    int *out_n_glints,
    double *out_centers_xy,
    double *out_ellipse_params,
    int *out_contour_lengths,
    double *out_contours_xy,
    int max_glints,
    int max_contour_points_per_glint,
    std::uint8_t *out_search_mask)
{
    using namespace cheshm::SimpleGlint;  // NOLINT(google-build-using-namespace)

    if (img_data == nullptr || width <= 0 || height <= 0 || out_search_mask == nullptr) {
        if (out_n_glints != nullptr) {
            *out_n_glints = 0;
        }
        return 0;
    }

    const cv::Mat full(height, width, CV_8U, const_cast<std::uint8_t *>(img_data));
    cv::Rect crop(0, 0, width, height);
    bool roi_active = false;
    if (cheshm::roi_is_active(roi_w, roi_h)) {
        crop = cheshm::clamp_roi(roi_x, roi_y, roi_w, roi_h, width, height);
        if (crop.area() == 0) {
            *out_n_glints = 0;
            return 0;
        }
        roi_active = true;
    }
    const cv::Mat view = full(crop);

    cv::Mat glint_mask;
    cv::threshold(view, glint_mask, glint_threshold, 255, cv::THRESH_BINARY);

    cv::Mat search_mask(view.size(), CV_8U);
    if (has_pupil != 0) {
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

    if (roi_active) {
        std::memset(out_search_mask, 0, static_cast<size_t>(width) * static_cast<size_t>(height));
    }
    cv::Mat search_canvas(height, width, CV_8U, out_search_mask);
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
        has_pupil != 0
        && !(keep_above != 0 && keep_below != 0 && keep_left != 0 && keep_right != 0);
    if (half_plane_active) {
        const double cx_local = pupil_cx - crop.x;
        const double cy_local = pupil_cy - crop.y;
        std::vector<std::vector<cv::Point>> kept;
        kept.reserve(contours.size());
        for (auto &c : contours) {
            if (passes_half_plane(c, cx_local, cy_local,
                                  keep_above != 0, keep_below != 0,
                                  keep_left != 0, keep_right != 0,
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

    if (split_widest_for_target != 0
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

    int written = 0;
    const int stride = 2 * max_contour_points_per_glint;
    for (const Candidate &cand : accepted) {
        if (written >= max_glints) {
            break;
        }
        double cx_local = 0.0;
        double cy_local = 0.0;
        if (!compute_center(cand, candidates_mask, glint_center_method, cx_local, cy_local)) {
            continue;
        }
        const double cx = cx_local + crop.x;
        const double cy = cy_local + crop.y;
        out_centers_xy[2 * written] = cx;
        out_centers_xy[2 * written + 1] = cy;
        if (cand.has_fit) {
            out_ellipse_params[6 * written + 0] = cand.fit.center.x + crop.x;
            out_ellipse_params[6 * written + 1] = cand.fit.center.y + crop.y;
            out_ellipse_params[6 * written + 2] = cand.fit.size.width;
            out_ellipse_params[6 * written + 3] = cand.fit.size.height;
            out_ellipse_params[6 * written + 4] = cand.fit.angle;
            out_ellipse_params[6 * written + 5] = 1.0;
        } else {
            for (int i = 0; i < 5; ++i) {
                out_ellipse_params[6 * written + i] = std::numeric_limits<double>::quiet_NaN();
            }
            out_ellipse_params[6 * written + 5] = 0.0;
        }
        const int n_pts = std::min(static_cast<int>(cand.contour.size()), max_contour_points_per_glint);
        out_contour_lengths[written] = n_pts;
        double *slot = out_contours_xy + static_cast<std::size_t>(written) * stride;
        for (int i = 0; i < n_pts; ++i) {
            slot[2 * i] = static_cast<double>(cand.contour[i].x) + crop.x;
            slot[2 * i + 1] = static_cast<double>(cand.contour[i].y) + crop.y;
        }
        ++written;
    }
    *out_n_glints = written;

    return 1;
}

}  // extern "C"
