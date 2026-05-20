// Corneal-reflection removal.

#include "Starburst/corneal_reflection.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace cheshm::Starburst {
namespace {

void locate_corneal_reflection(
    cv::Mat &image,
    int sx,
    int sy,
    int window_size,
    int biggest_crar,
    int &crx,
    int &cry,
    int &crar)
{
    if (window_size % 2 == 0) {
        // window_size should be odd; we proceed regardless.
    }

    int r = (window_size - 1) / 2;
    int startx = std::max(sx - r, 0);
    startx = std::min(startx, image.size().width - 1);
    int endx = std::min(sx + r, image.size().width - 1);
    int starty = std::max(sy - r, 0);
    starty = std::min(starty, image.size().height - 1);
    int endy = std::min(sy + r, image.size().height - 1);

    cv::Mat roiImage = image(cv::Rect(startx, starty, endx - startx + 1, endy - starty + 1));
    cv::Mat roiThresholdImage;
    roiImage.copyTo(roiThresholdImage);

    double min_value, max_value;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(image, &min_value, &max_value, &min_loc, &max_loc);

    std::vector<double> scores(static_cast<int>(max_value) + 1, 0.0);
    std::vector<std::vector<cv::Point>> contours;
    int area, max_area, sum_area;
    for (int threshold = static_cast<int>(max_value); threshold >= 1; threshold--) {
        cv::threshold(roiImage, roiThresholdImage, threshold, 1, cv::THRESH_BINARY);
        cv::findContours(roiThresholdImage, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
        max_area = 0;
        sum_area = 0;
        std::vector<cv::Point> max_contour;
        for (auto &contour : contours) {
            area = static_cast<int>(contour.size()) + static_cast<int>(std::fabs(cv::contourArea(contour)));
            sum_area += area;
            if (area > max_area) {
                max_area = area;
                max_contour = contour;
            }
        }
        if (sum_area - max_area > 0) {
            scores[threshold - 1] = static_cast<double>(max_area) / (sum_area - max_area);
        } else {
            continue;
        }

        if (scores[threshold - 1] - scores[threshold] < 0) {
            // Found the corneal reflection.
            crar = static_cast<int>(std::sqrt(max_area / CV_PI));
            int sum_x = 0;
            int sum_y = 0;
            for (auto &it : max_contour) {
                sum_x += it.x;
                sum_y += it.y;
            }
            crx = sum_x / static_cast<int>(max_contour.size());
            cry = sum_y / static_cast<int>(max_contour.size());
            break;
        }
    }

    if (crar > biggest_crar) {
        cry = crx = -1;
        crar = -1;
    }

    if (crx != -1 && cry != -1) {
        crx += startx;
        cry += starty;
    }
}

int fit_circle_radius_to_corneal_reflection(
    cv::Mat &image,
    int crx,
    int cry,
    int crar,
    int biggest_crar,
    const double *sin_array,
    const double *cos_array,
    int array_len)
{
    if (crx == -1 || cry == -1 || crar == -1) {
        return -1;
    }

    std::vector<double> ratio(biggest_crar - crar + 1, 0.0);
    const int r_delta = 1;
    int x, y, x2, y2;
    double sum, sum2;
    for (int r = crar; r <= biggest_crar; r++) {
        sum = 0;
        sum2 = 0;
        for (int i = 0; i < array_len; i++) {
            x = static_cast<int>(crx + (r + r_delta) * cos_array[i]);
            y = static_cast<int>(cry + (r + r_delta) * sin_array[i]);
            x2 = static_cast<int>(crx + (r - r_delta) * cos_array[i]);
            y2 = static_cast<int>(cry + (r + r_delta) * sin_array[i]);
            if ((x >= 0 && y >= 0 && x < image.size().width && y < image.size().height) &&
                (x2 >= 0 && y2 >= 0 && x2 < image.size().width && y2 < image.size().height)) {
                sum += *(image.data + y * image.size().width + x);
                sum2 += *(image.data + y2 * image.size().width + x2);
            }
        }
        ratio[r - crar] = sum / sum2;
        if (r - crar >= 2) {
            if (ratio[r - crar - 2] < ratio[r - crar - 1] && ratio[r - crar] < ratio[r - crar - 1]) {
                return r - 1;
            }
        }
    }
    return crar;
}

void interpolate_corneal_reflection(
    cv::Mat &image,
    int crx,
    int cry,
    int crr,
    const double *sin_array,
    const double *cos_array,
    int array_len)
{
    if (crx == -1 || cry == -1 || crr == -1) {
        return;
    }
    if (crx - crr < 0 || crx + crr >= image.size().width ||
        cry - crr < 0 || cry + crr >= image.size().height) {
        return;  // Corneal reflection too near the image border.
    }

    std::vector<std::uint8_t> perimeter_pixel(array_len, 0);
    int sum = 0;
    for (int i = 0; i < array_len; i++) {
        int x = static_cast<int>(crx + crr * cos_array[i]);
        int y = static_cast<int>(cry + crr * sin_array[i]);
        perimeter_pixel[i] = static_cast<std::uint8_t>(*(image.data + y * image.size().width + x));
        sum += perimeter_pixel[i];
    }
    double avg = static_cast<double>(sum) / array_len;

    for (int r = 1; r < crr; r++) {
        int r2 = crr - r;
        for (int i = 0; i < array_len; i++) {
            int x = static_cast<int>(crx + r * cos_array[i]);
            int y = static_cast<int>(cry + r * sin_array[i]);
            *(image.data + y * image.size().width + x) = static_cast<std::uint8_t>(
                (r2 * 1.0 / crr) * avg + (r * 1.0 / crr) * perimeter_pixel[i]);
        }
    }
}

}  // namespace

void remove_corneal_reflection(
    cv::Mat &image,
    int sx,
    int sy,
    int window_size,
    int biggest_crr,
    int &crx,
    int &cry,
    int &crr)
{
    int crar = -1;
    crx = cry = crar = -1;

    const float angle_delta = 1 * CV_PI / 180;
    const int angle_num = static_cast<int>(2 * CV_PI / angle_delta);

    std::vector<double> sin_array(angle_num);
    std::vector<double> cos_array(angle_num);
    for (int i = 0; i < angle_num; i++) {
        const double angle = i * angle_delta;
        sin_array[i] = std::sin(angle);
        cos_array[i] = std::cos(angle);
    }

    locate_corneal_reflection(image, sx, sy, window_size, static_cast<int>(biggest_crr / 2.5), crx, cry, crar);
    crr = fit_circle_radius_to_corneal_reflection(
        image, crx, cry, crar, static_cast<int>(biggest_crr / 2.5),
        sin_array.data(), cos_array.data(), angle_num);
    crr = static_cast<int>(2.5 * crr);
    interpolate_corneal_reflection(image, crx, cry, crr, sin_array.data(), cos_array.data(), angle_num);
}

}  // namespace cheshm::Starburst
