#include "Starburst/auto_seed.hpp"

#include <opencv2/imgproc.hpp>
#include <vector>

namespace cheshm::Starburst
{
namespace
{

bool touches_border(const std::vector<cv::Point>& contour, int width, int height)
{
    const cv::Rect br = cv::boundingRect(contour);
    return br.x == 0 || br.y == 0 || br.x + br.width == width || br.y + br.height == height;
}

} // namespace

cv::Point2d auto_seed(const cv::Mat& image, int seed_threshold)
{
    const cv::Point2d centre{image.cols / 2.0, image.rows / 2.0};

    cv::Mat mask;
    cv::threshold(image, mask, seed_threshold, 255, cv::THRESH_BINARY_INV);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    const std::vector<cv::Point>* best = nullptr;
    double best_area = 0.0;
    for (const auto& c : contours)
    {
        const double area = cv::contourArea(c);
        if (area <= 0.0)
            continue;
        if (touches_border(c, image.cols, image.rows))
            continue;
        if (area > best_area)
        {
            best_area = area;
            best = &c;
        }
    }
    if (best == nullptr)
        return centre;

    const cv::Moments m = cv::moments(*best);
    if (m.m00 <= 0.0)
        return centre;
    return {m.m10 / m.m00, m.m01 / m.m00};
}

} // namespace cheshm::Starburst
