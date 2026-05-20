// PupilLabs2D pupil detector algorithm body.

#include "PupilLabs2D/pupil_labs_2d.hpp"
#include "PupilLabs2D/defaults.hpp"

#include "detect_2d.hpp"

#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace cheshm::PupilLabs2D {

Properties default_properties()
{
    using namespace defaults;
    return Properties{
        INTENSITY_RANGE,
        BLUR_SIZE,
        CANNY_THRESHOLD,
        CANNY_RATIO,
        CANNY_APERTURE,
        PUPIL_SIZE_MAX,
        PUPIL_SIZE_MIN,
        STRONG_PERIMETER_RATIO_RANGE_MIN,
        STRONG_PERIMETER_RATIO_RANGE_MAX,
        STRONG_AREA_RATIO_RANGE_MIN,
        STRONG_AREA_RATIO_RANGE_MAX,
        CONTOUR_SIZE_MIN,
        ELLIPSE_ROUNDNESS_RATIO,
        INITIAL_ELLIPSE_FIT_THRESHOLD,
        FINAL_PERIMETER_RATIO_RANGE_MIN,
        FINAL_PERIMETER_RATIO_RANGE_MAX,
        ELLIPSE_TRUE_SUPPORT_MIN_DIST,
        SUPPORT_PIXEL_RATIO_EXPONENT,
        COARSE_DETECTION,
        COARSE_FILTER_MIN,
        COARSE_FILTER_MAX,
    };
}

namespace {

// Haar-like coarse pupil seed. Walks the integral image with a
// 3×inner outer box and returns the bounding box of the top-scoring
// responses, used to narrow the search ROI before the main detector.
cv::Rect coarse_detect(const cv::Mat &gray, const cv::Rect &roi, int min_w, int max_w)
{
    constexpr int scale = 2;
    cv::Mat sub = gray(roi);
    cv::Mat downsampled;
    cv::resize(sub, downsampled, cv::Size(), 1.0 / scale, 1.0 / scale, cv::INTER_NEAREST);
    cv::Mat integral;
    cv::integral(downsampled, integral, CV_32S);

    const int rows = integral.rows;
    const int cols = integral.cols;
    const int min_h = (min_w / scale) / 3;
    const int max_h = (max_w / scale) / 3;
    constexpr int h_step = 4;
    constexpr int step = 5;

    struct Response {
        int x;
        int y;
        int w;
        float score;
    };
    std::vector<Response> results;
    float best_response = -1.0e6f;

    auto area_at = [&](int r0, int c0, int r1, int c1) {
        return integral.at<int>(r1, c1) + integral.at<int>(r0, c0)
             - integral.at<int>(r0, c1) - integral.at<int>(r1, c0);
    };

    for (int h = min_h; h < max_h; h += h_step) {
        const int w = 3 * h;
        const float outer_norm = 1.0f / static_cast<float>(w * w);
        const float inner_norm = -1.0f / static_cast<float>(h * h);
        for (int i = 0; i < rows - 1 - w; i += step) {
            for (int j = 0; j < cols - 1 - w; j += step) {
                const float resp =
                    outer_norm * static_cast<float>(area_at(i, j, i + w, j + w))
                  + inner_norm * static_cast<float>(area_at(i + h, j + h, i + 2 * h, j + 2 * h));
                if (resp > best_response) {
                    best_response = resp;
                    results.push_back({j, i, w, resp});
                    if (results.size() > 30) results.erase(results.begin());
                }
            }
        }
    }

    // Drop weak responses + responses fully containing another.
    std::vector<Response> kept;
    for (auto it = results.rbegin(); it != results.rend(); ++it) {
        if (it->score < 0.4f * best_response) continue;
        bool dominated = false;
        for (const auto &g : results) {
            if (it->x < g.x && it->y < g.y
                && it->x + it->w > g.x + g.w
                && it->y + it->w > g.y + g.w) {
                dominated = true;
                break;
            }
        }
        if (!dominated) kept.push_back(*it);
    }

    if (kept.empty()) return roi;

    int x_b = kept[0].x;
    int y_b = kept[0].y;
    int x2_b = 1;
    int y2_b = 1;
    for (const auto &v : kept) {
        x_b = std::min(v.x, x_b);
        y_b = std::min(v.y, y_b);
        if (x2_b < v.x + v.w) x2_b = v.x + v.w;
        if (y2_b < v.y + v.w) y2_b = v.y + v.w;
    }

    // Scale back + offset into original-image coordinates.
    cv::Rect refined{
        x_b * scale + roi.x,
        y_b * scale + roi.y,
        (x2_b - x_b) * scale,
        (y2_b - y_b) * scale,
    };
    refined &= cv::Rect{0, 0, gray.cols, gray.rows};
    if (refined.area() == 0) return roi;
    return refined;
}

}  // namespace

std::optional<DetectResult> detect(
    const cv::Mat &gray,
    const cv::Rect &roi_in,
    const Properties &props)
{
    if (gray.empty()) return std::nullopt;

    cv::Rect roi = roi_in;
    if (roi.area() == 0) roi = cv::Rect{0, 0, gray.cols, gray.rows};
    roi &= cv::Rect{0, 0, gray.cols, gray.rows};
    if (roi.area() == 0) return std::nullopt;

    if (props.coarse_detection && roi.width * roi.height > 320 * 240) {
        roi = coarse_detect(gray, roi, props.coarse_filter_min, props.coarse_filter_max);
    }

    singleeyefitter::Detector2DProperties up_props{};
    up_props.intensity_range = props.intensity_range;
    up_props.blur_size = props.blur_size;
    up_props.canny_treshold = props.canny_threshold;
    up_props.canny_ration = props.canny_ratio;
    up_props.canny_aperture = props.canny_aperture;
    up_props.pupil_size_max = props.pupil_size_max;
    up_props.pupil_size_min = props.pupil_size_min;
    up_props.strong_perimeter_ratio_range_min = props.strong_perimeter_ratio_range_min;
    up_props.strong_perimeter_ratio_range_max = props.strong_perimeter_ratio_range_max;
    up_props.strong_area_ratio_range_min = props.strong_area_ratio_range_min;
    up_props.strong_area_ratio_range_max = props.strong_area_ratio_range_max;
    up_props.contour_size_min = props.contour_size_min;
    up_props.ellipse_roundness_ratio = props.ellipse_roundness_ratio;
    up_props.initial_ellipse_fit_treshhold = props.initial_ellipse_fit_threshold;
    up_props.final_perimeter_ratio_range_min = props.final_perimeter_ratio_range_min;
    up_props.final_perimeter_ratio_range_max = props.final_perimeter_ratio_range_max;
    up_props.ellipse_true_support_min_dist = props.ellipse_true_support_min_dist;
    up_props.support_pixel_ratio_exponent = props.support_pixel_ratio_exponent;

    // detect() takes non-const cv::Mat references; bind named copies.
    cv::Mat image = gray;
    cv::Mat color_image;
    cv::Mat debug_image;
    cv::Rect roi_for_upstream = roi;

    Detector2D detector;
    auto res = detector.detect(up_props, image, color_image, debug_image,
                               roi_for_upstream, /*visualize=*/false,
                               /*use_debug_image=*/false);
    if (!res) return std::nullopt;

    const auto &e = res->ellipse;
    if (e.major_radius <= 0.0 || e.minor_radius <= 0.0 || res->confidence <= 0.0) {
        return std::nullopt;
    }

    // Upstream Detector2D::detect shifts ellipse.center to full-image
    // coordinates at the end of its pipeline; pass through unchanged.
    // Axes are diameters (radius × 2). Angle is radians, converted to
    // degrees with -90° offset to match cv::RotatedRect's
    // long-axis-at-angle / width-along-short-axis convention.
    cv::RotatedRect rect(
        cv::Point2f(
            static_cast<float>(e.center[0]),
            static_cast<float>(e.center[1])),
        cv::Size2f(
            static_cast<float>(e.minor_radius * 2.0),
            static_cast<float>(e.major_radius * 2.0)),
        static_cast<float>(e.angle * 180.0 / CV_PI - 90.0));

    return DetectResult{rect, static_cast<float>(res->confidence)};
}

}  // namespace cheshm::PupilLabs2D
