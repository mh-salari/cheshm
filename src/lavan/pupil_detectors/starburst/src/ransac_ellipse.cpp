// RANSAC ellipse-fit helpers — random sampling, conic-to-ellipse solve,
// normalisation of edge points, and the main RANSAC loop that picks the
// ellipse with the most inliers.

#include "starburst/ransac_ellipse.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <opencv2/core.hpp>
#include <vector>

namespace lavan::starburst {

void RansacEllipse::get_random_num(int n, int max_num, int *rand_num)
{
    int rand_index = 0;

    if (max_num == n - 1) {
        for (int i = 0; i < n; i++) {
            rand_num[i] = i;
        }
        return;
    }

    while (rand_index < n) {
        bool is_new = true;
        int r = static_cast<int>((std::rand() * 1.0 / RAND_MAX) * max_num);
        for (int i = 0; i < rand_index; i++) {
            if (r == rand_num[i]) {
                is_new = false;
                break;
            }
        }
        if (is_new) {
            rand_num[rand_index] = r;
            rand_index++;
        }
    }
}

bool RansacEllipse::solve_ellipse(double *conic_param, double *ellipse_param)
{
    const double a = conic_param[0];
    const double b = conic_param[1];
    const double c = conic_param[2];
    const double d = conic_param[3];
    const double e = conic_param[4];
    const double f = conic_param[5];

    // Ellipse orientation.
    const double theta = std::atan2(b, a - c) / 2;

    // Scaled major/minor axes.
    const double ct = std::cos(theta);
    const double st = std::sin(theta);
    const double ap = a * ct * ct + b * ct * st + c * st * st;
    const double cp = a * st * st - b * ct * st + c * ct * ct;

    // Translations.
    const double cx = (2 * c * d - b * e) / (b * b - 4 * a * c);
    const double cy = (2 * a * e - b * d) / (b * b - 4 * a * c);

    // Scale factor.
    const double val = a * cx * cx + b * cx * cy + c * cy * cy;
    const double scale_inv = val - f;

    if (scale_inv / ap <= 0 || scale_inv / cp <= 0) {
        std::memset(ellipse_param, 0, sizeof(double) * 5);
        return false;
    }

    ellipse_param[0] = std::sqrt(scale_inv / ap);
    ellipse_param[1] = std::sqrt(scale_inv / cp);
    ellipse_param[2] = cx;
    ellipse_param[3] = cy;
    ellipse_param[4] = theta;
    return true;
}

cv::Point2d *RansacEllipse::normalize_edge_point(double &dis_scale, cv::Point2d &nor_center, int ep_num)
{
    double sumx = 0;
    double sumy = 0;
    double sumdis = 0;
    for (int i = 0; i < ep_num; i++) {
        cv::Point2d *edge = edge_point.at(i);
        sumx += edge->x;
        sumy += edge->y;
        sumdis += std::sqrt(edge->x * edge->x + edge->y * edge->y);
    }

    dis_scale = std::sqrt(2.0) * ep_num / sumdis;
    nor_center.x = sumx / ep_num;
    nor_center.y = sumy / ep_num;

    auto *edge_point_nor = static_cast<cv::Point2d *>(std::malloc(sizeof(cv::Point2d) * ep_num));
    for (int i = 0; i < ep_num; i++) {
        cv::Point2d *edge = edge_point.at(i);
        edge_point_nor[i].x = (edge->x - nor_center.x) * dis_scale;
        edge_point_nor[i].y = (edge->y - nor_center.y) * dis_scale;
    }
    return edge_point_nor;
}

void RansacEllipse::denormalize_ellipse_param(double *par, double *normalized_par, double dis_scale, cv::Point2d nor_center)
{
    par[0] = normalized_par[0] / dis_scale;                  // semi-major
    par[1] = normalized_par[1] / dis_scale;                  // semi-minor
    par[2] = normalized_par[2] / dis_scale + nor_center.x;   // centre x
    par[3] = normalized_par[3] / dis_scale + nor_center.y;   // centre y
}

int *RansacEllipse::pupil_fitting_inliers(
    std::uint8_t * /*pupil_image*/,
    int width,
    int height,
    int &return_max_inliers_num)
{
    const int ep_num = static_cast<int>(edge_point.size());
    const int ellipse_point_num = 5;  // points needed to fit an ellipse
    if (ep_num < ellipse_point_num) {
        std::memset(pupil_param, 0, sizeof(pupil_param));
        return_max_inliers_num = 0;
        return nullptr;
    }
    // Deterministic RANSAC sampling: with the same image + settings the
    // detector returns the same ellipse. (The original cvEyeTracker used
    // an unseeded ``rand()`` which produced different results across
    // runs; surprising during GUI tuning.)
    std::srand(42);

    cv::Point2d nor_center;
    double dis_scale;
    cv::Point2d *edge_point_nor = normalize_edge_point(dis_scale, nor_center, ep_num);

    int *inliers_index = static_cast<int *>(std::malloc(sizeof(int) * ep_num));
    int *max_inliers_index = static_cast<int *>(std::malloc(sizeof(int) * ep_num));
    int ninliers = 0;
    int max_inliers = 0;
    int sample_num = 1000;
    int ransac_count = 0;
    const double dis_threshold = std::sqrt(3.84) * dis_scale / 10;
    double dis_error;

    std::memset(inliers_index, 0, sizeof(int) * ep_num);
    std::memset(max_inliers_index, 0, sizeof(int) * ep_num);
    int rand_index[ellipse_point_num];
    double A[ellipse_point_num + 1][6];
    constexpr int M = ellipse_point_num + 1;
    constexpr int N = 6;
    for (int i = 0; i < M; i++) {
        A[i][5] = 1;
    }
    for (int i = 0; i < N; i++) {
        A[ellipse_point_num][i] = 0;
    }
    double **ppa = static_cast<double **>(std::malloc(sizeof(double *) * M));
    double **ppu = static_cast<double **>(std::malloc(sizeof(double *) * M));
    double **ppv = static_cast<double **>(std::malloc(sizeof(double *) * N));
    for (int i = 0; i < M; i++) {
        ppa[i] = A[i];
        ppu[i] = static_cast<double *>(std::malloc(sizeof(double) * N));
    }
    for (int i = 0; i < N; i++) {
        ppv[i] = static_cast<double *>(std::malloc(sizeof(double) * N));
    }

    double pd[6];
    int min_d_index;
    double conic_par[6] = {0};
    double ellipse_par[5] = {0};
    double best_ellipse_par[5] = {0};

    while (sample_num > ransac_count) {
        get_random_num(ellipse_point_num, ep_num - 1, rand_index);

        for (int i = 0; i < ellipse_point_num; i++) {
            A[i][0] = edge_point_nor[rand_index[i]].x * edge_point_nor[rand_index[i]].x;
            A[i][1] = edge_point_nor[rand_index[i]].x * edge_point_nor[rand_index[i]].y;
            A[i][2] = edge_point_nor[rand_index[i]].y * edge_point_nor[rand_index[i]].y;
            A[i][3] = edge_point_nor[rand_index[i]].x;
            A[i][4] = edge_point_nor[rand_index[i]].y;
        }

        svd(M, N, ppa, ppu, pd, ppv);
        min_d_index = 0;
        for (int i = 1; i < N; i++) {
            if (pd[i] < pd[min_d_index]) {
                min_d_index = i;
            }
        }
        for (int i = 0; i < N; i++) {
            // Column of v corresponding to the smallest singular value:
            // the conic-equation solution.
            conic_par[i] = ppv[i][min_d_index];
        }

        ninliers = 0;
        std::memset(inliers_index, 0, sizeof(int) * ep_num);
        for (int i = 0; i < ep_num; i++) {
            dis_error = conic_par[0] * edge_point_nor[i].x * edge_point_nor[i].x +
                        conic_par[1] * edge_point_nor[i].x * edge_point_nor[i].y +
                        conic_par[2] * edge_point_nor[i].y * edge_point_nor[i].y +
                        conic_par[3] * edge_point_nor[i].x + conic_par[4] * edge_point_nor[i].y +
                        conic_par[5];
            if (std::fabs(dis_error) < dis_threshold) {
                inliers_index[ninliers] = i;
                ninliers++;
            }
        }

        if (ninliers > max_inliers) {
            if (solve_ellipse(conic_par, ellipse_par)) {
                denormalize_ellipse_param(ellipse_par, ellipse_par, dis_scale, nor_center);
                const double ratio = ellipse_par[0] / ellipse_par[1];
                if (ellipse_par[2] > 0 && ellipse_par[2] <= width - 1 &&
                    ellipse_par[3] > 0 && ellipse_par[3] <= height - 1 &&
                    ratio > 0.5 && ratio < 2) {
                    std::memcpy(max_inliers_index, inliers_index, sizeof(int) * ep_num);
                    for (int i = 0; i < 5; i++) {
                        best_ellipse_par[i] = ellipse_par[i];
                    }
                    max_inliers = ninliers;
                    sample_num = static_cast<int>(
                        std::log(1.0 - 0.99) / std::log(1.0 - std::pow(ninliers * 1.0 / ep_num, 5)));
                }
            }
        }
        ransac_count++;
        if (ransac_count > 1500) {
            break;
        }
    }

    if (best_ellipse_par[0] > 0 && best_ellipse_par[1] > 0) {
        for (int i = 0; i < 5; i++) {
            pupil_param[i] = best_ellipse_par[i];
        }
    } else {
        std::memset(pupil_param, 0, sizeof(pupil_param));
        max_inliers = 0;
        std::free(max_inliers_index);
        max_inliers_index = nullptr;
    }

    for (int i = 0; i < M; i++) {
        std::free(ppu[i]);
    }
    for (int i = 0; i < N; i++) {
        std::free(ppv[i]);
    }
    std::free(ppu);
    std::free(ppv);
    std::free(ppa);

    std::free(edge_point_nor);
    std::free(inliers_index);
    return_max_inliers_num = max_inliers;
    return max_inliers_index;
}

}  // namespace lavan::starburst
