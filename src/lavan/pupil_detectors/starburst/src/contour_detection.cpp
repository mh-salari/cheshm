// Starburst ray-based contour detection. ``starburst_pupil_contour_detection``
// shoots rays from a seed, finds the first strong intensity rise on each
// ray (``locate_edge_points``), then re-shoots return rays from those
// edge points. The loop iterates until the centre-of-mass of the edge
// points stabilises.
//
// Algorithm: cvEyeTracker / openEyes ToolKit (2004-2006), GPL.

#include "starburst/ransac_ellipse.hpp"

#include <cmath>
#include <cstdint>
#include <opencv2/core.hpp>

namespace lavan::starburst {

int RansacEllipse::starburst_pupil_contour_detection(
    std::uint8_t *pupil_image,
    const cv::Point2d &startPoint,
    int width,
    int height,
    int edge_thresh,
    int N,
    int minimum_candidate_features)
{
    const int dis = 7;
    const double angle_spread = 100 * CV_PI / 180;
    int loop_count = 0;
    const double angle_step = 2 * CV_PI / N;
    double new_angle_step;
    cv::Point2d *edge;
    cv::Point2d edge_mean;
    double angle_normal;
    double cx = startPoint.x;
    double cy = startPoint.y;
    int first_ep_num;

    while (edge_thresh > 5 && loop_count <= 10) {
        edge_intensity_diff.clear();
        destroy_edge_point();
        while (static_cast<int>(edge_point.size()) < minimum_candidate_features && edge_thresh > 5) {
            edge_intensity_diff.clear();
            destroy_edge_point();
            locate_edge_points(pupil_image, width, height, cx, cy, dis, angle_step, 0, 2 * CV_PI, edge_thresh);
            if (static_cast<int>(edge_point.size()) < minimum_candidate_features) {
                edge_thresh -= 1;
            }
        }
        if (edge_thresh <= 5) {
            break;
        }

        first_ep_num = static_cast<int>(edge_point.size());
        for (int i = 0; i < first_ep_num; i++) {
            edge = edge_point.at(i);
            angle_normal = std::atan2(cy - edge->y, cx - edge->x);
            new_angle_step = angle_step * (edge_thresh * 1.0 / edge_intensity_diff.at(i));
            locate_edge_points(pupil_image, width, height, edge->x, edge->y, dis,
                               new_angle_step, angle_normal, angle_spread, edge_thresh);
        }

        loop_count += 1;
        edge_mean = get_edge_mean();
        if (std::fabs(edge_mean.x - cx) + std::fabs(edge_mean.y - cy) < 10) {
            break;
        }
        cx = edge_mean.x;
        cy = edge_mean.y;
    }

    if (loop_count > 10) {
        destroy_edge_point();
        return 1;  // Edge points did not converge in 10 iterations.
    }
    if (edge_thresh <= 5) {
        destroy_edge_point();
        return 1;  // Adaptive threshold dropped too low.
    }
    return 0;
}

void RansacEllipse::locate_edge_points(
    const std::uint8_t *image,
    int width,
    int height,
    double cx,
    double cy,
    int dis,
    double angle_step,
    double angle_normal,
    double angle_spread,
    int edge_thresh)
{
    cv::Point2d p;
    cv::Point2d *edge;

    for (double angle = angle_normal - angle_spread / 2 + 0.0001;
         angle < angle_normal + angle_spread / 2;
         angle += angle_step) {
        const double dis_cos = dis * std::cos(angle);
        const double dis_sin = dis * std::sin(angle);
        p.x = cx + dis_cos;
        p.y = cy + dis_sin;

        if (p.x < 0 || p.x >= width || p.y < 0 || p.y >= height) {
            continue;
        }

        int pixel_value1 = image[static_cast<int>(p.y) * width + static_cast<int>(p.x)];
        while (true) {
            p.x += dis_cos;
            p.y += dis_sin;
            if (p.x < 0 || p.x >= width || p.y < 0 || p.y >= height) {
                break;
            }
            int pixel_value2 = image[static_cast<int>(p.y) * width + static_cast<int>(p.x)];
            if ((pixel_value2 - pixel_value1) > edge_thresh) {
                edge = new cv::Point2d();
                edge->x = p.x - dis_cos / 2;
                edge->y = p.y - dis_sin / 2;
                edge_point.push_back(edge);
                edge_intensity_diff.push_back(pixel_value2 - pixel_value1);
                break;
            }
            pixel_value1 = pixel_value2;
        }
    }
}

cv::Point2d RansacEllipse::get_edge_mean()
{
    cv::Point2d edge_mean;
    if (edge_point.empty()) {
        edge_mean.x = -1;
        edge_mean.y = -1;
        return edge_mean;
    }
    double sumx = 0;
    double sumy = 0;
    for (auto *edge : edge_point) {
        sumx += edge->x;
        sumy += edge->y;
    }
    edge_mean.x = sumx / edge_point.size();
    edge_mean.y = sumy / edge_point.size();
    return edge_mean;
}

void RansacEllipse::destroy_edge_point()
{
    for (auto *edge : edge_point) {
        delete edge;
    }
    edge_point.clear();
}

RansacEllipse::~RansacEllipse() { destroy_edge_point(); }

}  // namespace lavan::starburst
