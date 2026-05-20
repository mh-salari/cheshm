// Public C surface for the Starburst pupil detector — a single
// extern "C" entry point that lavan's Python wrapper loads via ctypes.
//
// All in-tree C++ machinery (CR removal, RANSAC + SVD, ray-based contour
// detection) is invoked from one place here. The input image is copied
// into an internal working buffer so the caller's numpy array is never
// mutated.
//
// Algorithm: cvEyeTracker / openEyes ToolKit (2004-2006), GPL.

#include "starburst/corneal_reflection.hpp"
#include "starburst/ransac_ellipse.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <opencv2/core.hpp>
#include <vector>

extern "C" {

// starburst_detect — find the pupil ellipse via Starburst.
//
//  img_data           grayscale uint8 buffer, ``width * height`` bytes,
//                     row-major (numpy default).
//  seed_x, seed_y     initial guess for the pupil centre (e.g. image
//                     centre or the previous detection).
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
//  out_n_edge_points  number of edge points written to ``edge_points_xy``.
//  edge_points_xy     caller-allocated buffer of ``2 * max_edge_points``
//                     doubles for ``(x, y)`` pairs.
//  max_edge_points    capacity of ``edge_points_xy`` in points (not doubles).
//
// Returns 1 on success (ellipse found), 0 on failure.
int starburst_detect(
    const std::uint8_t *img_data,
    int width,
    int height,
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
    using namespace lavan::starburst;  // NOLINT(google-build-using-namespace)

    if (img_data == nullptr || width <= 0 || height <= 0) {
        return 0;
    }

    // Copy into a working buffer so CR removal and any other in-place
    // step never touch the caller's array.
    std::vector<std::uint8_t> working(img_data, img_data + width * height);
    cv::Mat working_mat(height, width, CV_8U, working.data());

    if (cr_window_size > 0 && cr_ratio_to_image_height > 0) {
        int crx = -1, cry = -1, crr = -1;
        const int biggest_crr = height / cr_ratio_to_image_height;
        remove_corneal_reflection(
            working_mat,
            static_cast<int>(seed_x),
            static_cast<int>(seed_y),
            cr_window_size,
            biggest_crr,
            crx,
            cry,
            crr);
    }

    RansacEllipse ransac;
    const int status = ransac.starburst_pupil_contour_detection(
        working.data(),
        cv::Point2d(seed_x, seed_y),
        width,
        height,
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
    int *inliers = ransac.pupil_fitting_inliers(working.data(), width, height, inliers_num);
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

    for (int i = 0; i < 5; i++) {
        out_ellipse_params[i] = ransac.pupil_param[i];
    }

    const int n = std::min(static_cast<int>(ransac.edge_point.size()), max_edge_points);
    for (int i = 0; i < n; i++) {
        edge_points_xy[2 * i] = ransac.edge_point[i]->x;
        edge_points_xy[2 * i + 1] = ransac.edge_point[i]->y;
    }
    if (out_n_edge_points != nullptr) {
        *out_n_edge_points = n;
    }

    return 1;
}

}  // extern "C"
