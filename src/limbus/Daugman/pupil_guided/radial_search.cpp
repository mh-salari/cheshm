#include "cheshm/limbus/Daugman/pupil_guided/radial_search.hpp"

#include <cmath>
#include <limits>
#include <optional>
#include <vector>

namespace cheshm::Daugman::pupil_guided
{
namespace
{

std::optional<float> bilinear_sample(const cv::Mat& image, float x, float y)
{
    if (x < 0.0f || x > static_cast<float>(image.cols - 1) || y < 0.0f || y > static_cast<float>(image.rows - 1))
        return std::nullopt;
    const int x0 = static_cast<int>(x);
    const int y0 = static_cast<int>(y);
    const int x1 = (x0 + 1 < image.cols) ? (x0 + 1) : x0;
    const int y1 = (y0 + 1 < image.rows) ? (y0 + 1) : y0;
    const float fx = x - static_cast<float>(x0);
    const float fy = y - static_cast<float>(y0);
    const float v00 = image.ptr<float>(y0)[x0];
    const float v01 = image.ptr<float>(y0)[x1];
    const float v10 = image.ptr<float>(y1)[x0];
    const float v11 = image.ptr<float>(y1)[x1];
    return (1.0f - fx) * (1.0f - fy) * v00 + fx * (1.0f - fy) * v01 + (1.0f - fx) * fy * v10 + fx * fy * v11;
}

} // namespace

std::vector<double> radial_search(const cv::Mat& gx,
                                  const cv::Mat& gy,
                                  cv::Point2f seed_center,
                                  int n_angles,
                                  std::span<const float> r_min_per_angle,
                                  std::span<const float> r_max_per_angle,
                                  int n_r,
                                  std::span<const double> smoothing_kernel)
{
    std::vector<double> r_theta(static_cast<std::size_t>(n_angles), -1.0);
    if (n_r < 2 || n_angles < 1)
        return r_theta;

    const int k_len = static_cast<int>(smoothing_kernel.size());
    const int half_k = k_len / 2;
    constexpr double kSentinel = -std::numeric_limits<double>::infinity();

    std::vector<double> profile(static_cast<std::size_t>(n_r));
    std::vector<double> smoothed(static_cast<std::size_t>(n_r));

    for (int i = 0; i < n_angles; ++i)
    {
        const float rmin_i = r_min_per_angle[static_cast<std::size_t>(i)];
        const float rmax_i = r_max_per_angle[static_cast<std::size_t>(i)];
        if (!(rmax_i > rmin_i))
            continue;
        const float dr = (rmax_i - rmin_i) / static_cast<float>(n_r - 1);
        const double theta = 2.0 * CV_PI * static_cast<double>(i) / static_cast<double>(n_angles);
        const float cos_t = static_cast<float>(std::cos(theta));
        const float sin_t = static_cast<float>(std::sin(theta));

        bool any_in = false;
        for (int j = 0; j < n_r; ++j)
        {
            const float r = rmin_i + dr * static_cast<float>(j);
            const float x = seed_center.x + r * cos_t;
            const float y = seed_center.y + r * sin_t;
            const auto gxv = bilinear_sample(gx, x, y);
            const auto gyv = bilinear_sample(gy, x, y);
            if (gxv && gyv)
            {
                profile[j] = static_cast<double>(*gxv * cos_t + *gyv * sin_t);
                any_in = true;
            }
            else
            {
                profile[j] = kSentinel;
            }
        }
        if (!any_in)
            continue;

        for (int j = 0; j < n_r; ++j)
        {
            double acc = 0.0;
            bool any_finite = false;
            for (int kk = 0; kk < k_len; ++kk)
            {
                const int idx = j + kk - half_k;
                if (idx < 0 || idx >= n_r)
                    continue;
                const double v = profile[idx];
                if (!std::isfinite(v))
                    continue;
                acc += v * smoothing_kernel[kk];
                any_finite = true;
            }
            smoothed[j] = any_finite ? acc : kSentinel;
        }

        double best_val = kSentinel;
        int best_idx = -1;
        for (int j = 0; j < n_r; ++j)
            if (smoothed[j] > best_val)
            {
                best_val = smoothed[j];
                best_idx = j;
            }
        if (best_idx >= 0 && std::isfinite(best_val))
            r_theta[i] = static_cast<double>(rmin_i + dr * static_cast<float>(best_idx));
    }

    return r_theta;
}

} // namespace cheshm::Daugman::pupil_guided
