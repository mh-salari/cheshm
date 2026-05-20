// Public C surface for the Starburst pupil detector — a single
// extern "C" entry point that cheshm's Python wrapper loads via ctypes.

#include "Starburst/corneal_reflection.hpp"
#include "Starburst/ransac_ellipse.hpp"
#include "cheshm/roi.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <opencv2/core.hpp>

extern "C" {

// Starburst_detect — find the pupil ellipse via Starburst.
//
//  img_data           grayscale uint8 buffer, ``width * height`` bytes,
//                     row-major (numpy default).
//  roi_x, roi_y,      ROI rectangle in full-image coordinates. When
//  roi_w, roi_h       ``roi_w > 0 && roi_h > 0`` the algorithm runs on
//                     the cropped sub-image; otherwise it runs on the
//                     full image. The ROI is clamped to image bounds.
//  seed_x, seed_y     initial guess for the pupil centre in full-image
//                     coordinates (e.g. image centre or previous frame).
//  edge_threshold     intensity-jump threshold for the ray edge test.
//                     The algorithm adapts this downward to find enough
//                     candidates.
//  rays               number of starburst rays (paper notation: N).
//  min_feature_candidates   minimum candidate count before RANSAC.
//  cr_window_size     odd integer, side of the CR search window centred
//                     on (seed_x, seed_y). 0 disables CR removal.
//  cr_ratio_to_image_height   ``image_height / cr_ratio`` is the largest
//                     CR radius accepted; 0 disables CR removal.
//  out_ellipse_params five doubles: ``{a, b, cx, cy, theta_rad}``
//                     (semi-major, semi-minor, centre, rotation).
//                     Centre is in full-image coordinates regardless of
//                     whether an ROI was used.
//  out_n_edge_points  number of edge points written to ``edge_points_xy``.
//  edge_points_xy     caller-allocated buffer of ``2 * max_edge_points``
//                     doubles for ``(x, y)`` pairs in full-image coords.
//  max_edge_points    capacity of ``edge_points_xy`` in points (not doubles).
//
// Returns 1 on success (ellipse found), 0 on failure.
int Starburst_detect(
    const std::uint8_t *img_data,
    int width,
    int height,
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
    double *out_ellipse_params,
    int *out_n_edge_points,
    double *edge_points_xy,
    int max_edge_points)
{
    using namespace cheshm::Starburst;  // NOLINT(google-build-using-namespace)

    if (img_data == nullptr || width <= 0 || height <= 0) {
        return 0;
    }

    const cv::Mat full(height, width, CV_8U, const_cast<std::uint8_t *>(img_data));
    cv::Rect crop(0, 0, width, height);
    if (cheshm::roi_is_active(roi_w, roi_h)) {
        crop = cheshm::clamp_roi(roi_x, roi_y, roi_w, roi_h, width, height);
        if (crop.area() == 0) {
            return 0;
        }
    }

    // clone() gives a contiguous owning copy so CR removal can mutate
    // it without touching the caller's numpy buffer.
    cv::Mat working_mat = full(crop).clone();
    const int local_w = working_mat.cols;
    const int local_h = working_mat.rows;
    const double local_seed_x = seed_x - crop.x;
    const double local_seed_y = seed_y - crop.y;

    if (cr_window_size > 0 && cr_ratio_to_image_height > 0) {
        int crx = -1, cry = -1, crr = -1;
        const int biggest_crr = local_h / cr_ratio_to_image_height;
        remove_corneal_reflection(
            working_mat,
            static_cast<int>(local_seed_x),
            static_cast<int>(local_seed_y),
            cr_window_size,
            biggest_crr,
            crx,
            cry,
            crr);
    }

    RansacEllipse ransac;
    const int status = ransac.starburst_pupil_contour_detection(
        working_mat.data,
        cv::Point2d(local_seed_x, local_seed_y),
        local_w,
        local_h,
        edge_threshold,
        rays,
        min_feature_candidates);
    if (status != 0) {
        if (out_n_edge_points != nullptr) {
            *out_n_edge_points = 0;
        }
        return 0;
    }

    int inliers_num = 0;
    int *inliers = ransac.pupil_fitting_inliers(working_mat.data, local_w, local_h, inliers_num);
    if (inliers == nullptr || inliers_num == 0) {
        if (inliers != nullptr) {
            std::free(inliers);
        }
        if (out_n_edge_points != nullptr) {
            *out_n_edge_points = 0;
        }
        return 0;
    }
    std::free(inliers);

    // Algorithm output is in crop-local coords; shift centre and edge
    // points back to full-image coordinates before handing to caller.
    for (int i = 0; i < 5; i++) {
        out_ellipse_params[i] = ransac.pupil_param[i];
    }
    out_ellipse_params[2] += crop.x;  // cx
    out_ellipse_params[3] += crop.y;  // cy

    const int n = std::min(static_cast<int>(ransac.edge_point.size()), max_edge_points);
    for (int i = 0; i < n; i++) {
        edge_points_xy[2 * i] = ransac.edge_point[i]->x + crop.x;
        edge_points_xy[2 * i + 1] = ransac.edge_point[i]->y + crop.y;
    }
    if (out_n_edge_points != nullptr) {
        *out_n_edge_points = n;
    }

    return 1;
}

}  // extern "C"
