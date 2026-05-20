// Starburst ellipse fitter — ``starburst_pupil_contour_detection``
// collects candidate pupil-edge points by shooting rays from a seed
// centre and detecting intensity jumps. ``pupil_fitting_inliers`` runs
// RANSAC over those points and leaves the ellipse parameters in
// ``pupil_param = {a, b, cx, cy, theta_rad}``.

#pragma once

#include <cstdint>
#include <opencv2/core/types.hpp>
#include <vector>

namespace lavan::Starburst {

class RansacEllipse {
public:
    RansacEllipse() = default;
    ~RansacEllipse();

    int starburst_pupil_contour_detection(
        std::uint8_t *pupil_image,
        const cv::Point2d &startPoint,
        int width,
        int height,
        int edge_thresh,
        int N,
        int minimum_candidate_features);

    int *pupil_fitting_inliers(
        std::uint8_t *pupil_image,
        int width,
        int height,
        int &return_max_inliers_num);

    std::vector<cv::Point2d *> edge_point;
    double pupil_param[5] = {0.0, 0.0, 0.0, 0.0, 0.0};

private:
    void svd(int m, int n, double **a, double **p, double *d, double **q);
    void destroy_edge_point();
    void locate_edge_points(
        const std::uint8_t *image,
        int width,
        int height,
        double cx,
        double cy,
        int dis,
        double angle_step,
        double angle_normal,
        double angle_spread,
        int edge_thresh);
    cv::Point2d get_edge_mean();
    cv::Point2d *normalize_edge_point(double &dis_scale, cv::Point2d &nor_center, int ep_num);
    bool solve_ellipse(double *conic_param, double *ellipse_param);
    void denormalize_ellipse_param(double *par, double *normalized_par, double dis_scale, cv::Point2d nor_center);
    void get_random_num(int n, int max_num, int *rand_num);

    std::vector<int> edge_intensity_diff;

    static double radius(double u, double v);
};

}  // namespace lavan::Starburst
