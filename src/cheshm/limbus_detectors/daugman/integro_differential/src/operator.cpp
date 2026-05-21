#include "integro_differential/operator.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace cheshm::Daugman::integro_differential
{
namespace
{

// Symmetric octant of the Bresenham circle, written as ``{|dr|, |dc|}``
// pairs; the eight reflections are produced inline at the call site.
struct CirclePoint
{
    int dr;
    int dc;
};

std::vector<CirclePoint> octant(int r)
{
    std::vector<CirclePoint> pts;
    pts.reserve(static_cast<std::size_t>(r) + 1);
    int dp = 3 - 2 * r;
    int a = 0;
    int b = r;
    while (a <= b)
    {
        pts.push_back({a, b});
        if (dp > 0)
        {
            --b;
            dp += 4 * (a - b) + 10;
        }
        else
        {
            dp += 4 * a + 6;
        }
        ++a;
    }
    return pts;
}

double mean_intensity_on_circle(const cv::Mat& image, int x, int y, int r)
{
    const auto pts = octant(r);
    long long total = 0;
    int count = 0;
    for (const auto& p : pts)
    {
        const int dxs[4] = {p.dr, p.dr, -p.dr, -p.dr};
        const int dys[4] = {p.dc, -p.dc, p.dc, -p.dc};
        for (int k = 0; k < 4; ++k)
        {
            const int rr = x + dxs[k];
            const int cc = y + dys[k];
            if (rr >= 0 && rr < image.rows && cc >= 0 && cc < image.cols)
            {
                total += image.ptr<std::uint8_t>(rr)[cc];
                ++count;
            }
        }
    }
    return count > 0 ? static_cast<double>(total) / static_cast<double>(count) : 0.0;
}

struct Differential
{
    double score;
    int radius;
};

Differential differential_at(const cv::Mat& image, int x, int y, int r_min, int r_max, const std::vector<double>& gk)
{
    const int n_radii = r_max - r_min;
    std::vector<double> values(static_cast<std::size_t>(n_radii));
    for (int ri = 0; ri < n_radii; ++ri)
        values[ri] = mean_intensity_on_circle(image, x, y, r_min + ri);

    const int n_diff = n_radii - 1;
    std::vector<double> diffs(static_cast<std::size_t>(n_diff));
    for (int i = 0; i < n_diff; ++i)
        diffs[i] = values[i + 1] - values[i];

    const int gk_len = static_cast<int>(gk.size());
    const int conv_len = n_diff + gk_len - 1;
    std::vector<double> conv(static_cast<std::size_t>(conv_len), 0.0);
    for (int i = 0; i < n_diff; ++i)
        for (int j = 0; j < gk_len; ++j)
            conv[i + j] += diffs[i] * gk[j];

    double best_val = -std::numeric_limits<double>::infinity();
    int best_idx = 0;
    for (int i = 0; i < conv_len; ++i)
        if (conv[i] > best_val)
        {
            best_val = conv[i];
            best_idx = i;
        }
    return {std::round(best_val), best_idx + r_min};
}

cv::Mat morph_open(const cv::Mat& image)
{
    static const cv::Mat kernel = (cv::Mat_<std::uint8_t>(7, 7) << //
                                       0, 0, 0, 1, 0, 0, 0,        //
                                   0, 1, 1, 1, 1, 1, 0,            //
                                   0, 1, 1, 1, 1, 1, 0,            //
                                   1, 1, 1, 1, 1, 1, 1,            //
                                   0, 1, 1, 1, 1, 1, 0,            //
                                   0, 1, 1, 1, 1, 1, 0,            //
                                   0, 0, 0, 1, 0, 0, 0);
    cv::Mat out;
    cv::morphologyEx(image, out, cv::MORPH_OPEN, kernel);
    return out;
}

std::vector<double> gaussian_kernel_1d()
{
    static const cv::Mat k = cv::getGaussianKernel(3, 1.0, CV_64F);
    return {k.at<double>(0), k.at<double>(1), k.at<double>(2)};
}

} // namespace

std::optional<DetectResult> detect_limbus(
    const cv::Mat& image, cv::Point2d seed_center, int r_min, int r_max, int range_, int step)
{
    cv::Mat gray;
    if (image.channels() == 1)
        gray = image;
    else
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    const cv::Mat opened = morph_open(gray);
    const std::vector<double> gk = gaussian_kernel_1d();

    const int cen_x = static_cast<int>(std::round(seed_center.x));
    const int cen_y = static_cast<int>(std::round(seed_center.y));

    Differential best{-std::numeric_limits<double>::infinity(), 0};
    cv::Point2i best_center{0, 0};
    bool any = false;

    // On tied scores the last-visited grid point wins.
    for (int dx = -range_; dx <= range_; dx += step)
        for (int dy = -range_; dy <= range_; dy += step)
        {
            const int px = cen_x + dx;
            const int py = cen_y + dy;
            const Differential d = differential_at(opened, px, py, r_min, r_max, gk);
            if (!any || d.score >= best.score)
            {
                best = d;
                best_center = {px, py};
                any = true;
            }
        }

    if (!any)
        return std::nullopt;
    return DetectResult{best_center, best.radius};
}

} // namespace cheshm::Daugman::integro_differential
