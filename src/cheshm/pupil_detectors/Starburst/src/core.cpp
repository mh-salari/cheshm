// Starburst pupil detector — Python binding.

#include "cheshm/image/roi.hpp"

#include "Starburst/corneal_reflection.hpp"
#include "Starburst/defaults.hpp"
#include "Starburst/ransac_ellipse.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <opencv2/core.hpp>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace
{

// Returns ``None`` on failure (RANSAC convergence failed, ROI empty)
// or a 2-tuple ``((a, b, cx, cy, theta_rad), edge_points)`` on success.
//   ellipse params: semi-major, semi-minor, centre, rotation (radians);
//                   centre in full-image coords.
//   edge_points: ``(N, 2)`` float64 ndarray of ray-edge hits in
//                full-image coords.
nb::object detect(nb::ndarray<const std::uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> img,
                  int roi_x,
                  int roi_y,
                  int roi_w,
                  int roi_h,
                  double seed_x,
                  double seed_y,
                  int edge_threshold,
                  int rays,
                  int min_feature_candidates,
                  int cr_window_size,
                  int cr_ratio_to_image_height,
                  int max_edge_points)
{
    using namespace cheshm::Starburst; // NOLINT(google-build-using-namespace)

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

    // clone() gives a contiguous owning copy so CR removal can mutate
    // it without touching the caller's numpy buffer.
    cv::Mat working_mat = full(crop).clone();
    const int local_w = working_mat.cols;
    const int local_h = working_mat.rows;
    const double local_seed_x = seed_x - crop.x;
    const double local_seed_y = seed_y - crop.y;

    if (cr_window_size > 0 && cr_ratio_to_image_height > 0)
    {
        int crx = -1, cry = -1, crr = -1;
        const int biggest_crr = local_h / cr_ratio_to_image_height;
        remove_corneal_reflection(working_mat,
                                  static_cast<int>(local_seed_x),
                                  static_cast<int>(local_seed_y),
                                  cr_window_size,
                                  biggest_crr,
                                  crx,
                                  cry,
                                  crr);
    }

    RansacEllipse ransac;
    const int status = ransac.starburst_pupil_contour_detection(working_mat.data,
                                                                cv::Point2d(local_seed_x, local_seed_y),
                                                                local_w,
                                                                local_h,
                                                                edge_threshold,
                                                                rays,
                                                                min_feature_candidates);
    if (status != 0)
    {
        return nb::none();
    }

    int inliers_num = 0;
    int* inliers = ransac.pupil_fitting_inliers(working_mat.data, local_w, local_h, inliers_num);
    if (inliers == nullptr || inliers_num == 0)
    {
        if (inliers != nullptr)
        {
            std::free(inliers);
        }
        return nb::none();
    }
    std::free(inliers);

    const double a = ransac.pupil_param[0];
    const double b = ransac.pupil_param[1];
    const double cx = ransac.pupil_param[2] + crop.x;
    const double cy = ransac.pupil_param[3] + crop.y;
    const double theta_rad = ransac.pupil_param[4];

    const int n = std::min(static_cast<int>(ransac.edge_point.size()), max_edge_points);
    auto edge_owner = std::make_unique<std::vector<double>>(2 * n);
    for (int i = 0; i < n; ++i)
    {
        (*edge_owner)[2 * i] = ransac.edge_point[i]->x + crop.x;
        (*edge_owner)[2 * i + 1] = ransac.edge_point[i]->y + crop.y;
    }
    double* edge_data = edge_owner->data();
    nb::capsule edge_cap(edge_owner.release(), [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
    const std::size_t edge_shape[2] = {static_cast<std::size_t>(n), 2};
    nb::ndarray<nb::numpy, double, nb::ndim<2>> edge_arr(edge_data, 2, edge_shape, edge_cap);

    return nb::make_tuple(nb::make_tuple(a, b, cx, cy, theta_rad), std::move(edge_arr));
}

} // namespace

NB_MODULE(_core, m)
{
    namespace d = cheshm::Starburst::defaults;

    m.def("detect",
          &detect,
          "img"_a,
          "roi_x"_a,
          "roi_y"_a,
          "roi_w"_a,
          "roi_h"_a,
          "seed_x"_a,
          "seed_y"_a,
          "edge_threshold"_a,
          "rays"_a,
          "min_feature_candidates"_a,
          "cr_window_size"_a,
          "cr_ratio_to_image_height"_a,
          "max_edge_points"_a);

    m.attr("EDGE_THRESHOLD") = d::EDGE_THRESHOLD;
    m.attr("RAYS") = d::RAYS;
    m.attr("MIN_FEATURE_CANDIDATES") = d::MIN_FEATURE_CANDIDATES;
    m.attr("CR_WINDOW_SIZE") = d::CR_WINDOW_SIZE;
    m.attr("CR_RATIO_TO_IMAGE_HEIGHT") = d::CR_RATIO_TO_IMAGE_HEIGHT;
    m.attr("MAX_EDGE_POINTS") = d::MAX_EDGE_POINTS;
    m.attr("SEED_THRESHOLD") = d::SEED_THRESHOLD;
}
