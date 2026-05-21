// Swirski 2D pupil detector — Python binding.

#include "cheshm/image/roi.hpp"

#include "Swirski2D/defaults.hpp"
#include "Swirski2D/pupil_tracker.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <opencv2/core.hpp>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace
{

// Returns ``None`` on failure (RANSAC could not converge, ROI empty)
// or a 2-tuple ``((cx, cy, w, h, angle_deg), inliers)`` on success.
//   ellipse params: cv::RotatedRect with centre in full-image coords.
//   inliers: ``(N, 2)`` float64 ndarray of RANSAC inlier points in
//            full-image coords.
nb::object detect(nb::ndarray<const std::uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> img,
                  int roi_x,
                  int roi_y,
                  int roi_w,
                  int roi_h,
                  int radius_min,
                  int radius_max,
                  double canny_blur,
                  double canny_thresh1,
                  double canny_thresh2,
                  int starburst_points,
                  int percentage_inliers,
                  int inlier_iterations,
                  int image_aware_support,
                  int early_termination_percentage,
                  int early_rejection,
                  int seed,
                  int max_inliers)
{
    using namespace cheshm::Swirski2D; // NOLINT(google-build-using-namespace)

    const int height = static_cast<int>(img.shape(0));
    const int width = static_cast<int>(img.shape(1));
    const cv::Mat full(height, width, CV_8U, const_cast<std::uint8_t*>(img.data()));

    cv::Rect crop(0, 0, width, height);
    if (cheshm::roi_is_active(roi_w, roi_h))
    {
        crop = cheshm::clamp_roi(roi_x, roi_y, roi_w, roi_h, width, height);
        if (crop.area() == 0)
        {
            return nb::none();
        }
    }
    // Swirski2D does not mutate the input, so a zero-copy ROI view is safe.
    const cv::Mat input = full(crop);

    TrackerParams params{};
    params.Radius_Min = radius_min;
    params.Radius_Max = radius_max;
    params.CannyBlur = canny_blur;
    params.CannyThreshold1 = canny_thresh1;
    params.CannyThreshold2 = canny_thresh2;
    params.StarburstPoints = starburst_points;
    params.PercentageInliers = percentage_inliers;
    params.InlierIterations = inlier_iterations;
    params.ImageAwareSupport = image_aware_support != 0;
    params.EarlyTerminationPercentage = early_termination_percentage;
    params.EarlyRejection = early_rejection != 0;
    params.Seed = seed;

    findPupilEllipse_out result;
    const bool ok = findPupilEllipse(params, input, result);
    if (!ok)
    {
        return nb::none();
    }

    const double cx = result.elPupil.center.x + crop.x;
    const double cy = result.elPupil.center.y + crop.y;
    const double w = result.elPupil.size.width;
    const double h = result.elPupil.size.height;
    const double angle = result.elPupil.angle;

    const int n = std::min(static_cast<int>(result.inliers.size()), max_inliers);
    auto inliers_owner = std::make_unique<std::vector<double>>(2 * n);
    for (int i = 0; i < n; ++i)
    {
        (*inliers_owner)[2 * i] = result.inliers[i].x + crop.x;
        (*inliers_owner)[2 * i + 1] = result.inliers[i].y + crop.y;
    }
    double* inliers_data = inliers_owner->data();
    nb::capsule inliers_cap(inliers_owner.release(),
                            [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
    const std::size_t inliers_shape[2] = {static_cast<std::size_t>(n), 2};
    nb::ndarray<nb::numpy, double, nb::ndim<2>> inliers_arr(inliers_data, 2, inliers_shape, inliers_cap);

    return nb::make_tuple(nb::make_tuple(cx, cy, w, h, angle), std::move(inliers_arr));
}

} // namespace

NB_MODULE(_core, m)
{
    namespace d = cheshm::Swirski2D::defaults;

    m.def("detect",
          &detect,
          "img"_a,
          "roi_x"_a,
          "roi_y"_a,
          "roi_w"_a,
          "roi_h"_a,
          "radius_min"_a,
          "radius_max"_a,
          "canny_blur"_a,
          "canny_thresh1"_a,
          "canny_thresh2"_a,
          "starburst_points"_a,
          "percentage_inliers"_a,
          "inlier_iterations"_a,
          "image_aware_support"_a,
          "early_termination_percentage"_a,
          "early_rejection"_a,
          "seed"_a,
          "max_inliers"_a);

    m.attr("RADIUS_MIN") = d::RADIUS_MIN;
    m.attr("RADIUS_MAX") = d::RADIUS_MAX;
    m.attr("CANNY_BLUR") = d::CANNY_BLUR;
    m.attr("CANNY_THRESHOLD_1") = d::CANNY_THRESHOLD_1;
    m.attr("CANNY_THRESHOLD_2") = d::CANNY_THRESHOLD_2;
    m.attr("STARBURST_POINTS") = d::STARBURST_POINTS;
    m.attr("PERCENTAGE_INLIERS") = d::PERCENTAGE_INLIERS;
    m.attr("INLIER_ITERATIONS") = d::INLIER_ITERATIONS;
    m.attr("IMAGE_AWARE_SUPPORT") = d::IMAGE_AWARE_SUPPORT;
    m.attr("EARLY_TERMINATION_PERCENTAGE") = d::EARLY_TERMINATION_PERCENTAGE;
    m.attr("EARLY_REJECTION") = d::EARLY_REJECTION;
    m.attr("SEED") = d::SEED;
    m.attr("MAX_INLIERS") = d::MAX_INLIERS;
}
