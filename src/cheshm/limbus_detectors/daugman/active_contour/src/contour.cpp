#include "active_contour/contour.hpp"
#include "active_contour/radial_search.hpp"
#include "daugman/contour_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <span>
#include <vector>

namespace cheshm::Daugman::active_contour
{
namespace
{

// Upper-eyelid angular wedge (y-down image coords) skipped during the
// radial search and filled by cyclic linear interpolation before the FFT.
constexpr int EYELID_WEDGE_START_DEG = 240;
constexpr int EYELID_WEDGE_END_DEG = 300;

std::vector<std::uint8_t> build_eyelid_mask(int n_angles, bool active)
{
    std::vector<std::uint8_t> mask(static_cast<std::size_t>(n_angles), 0);
    if (!active)
        return mask;
    for (int i = 0; i < n_angles; ++i)
    {
        const int deg = static_cast<int>(std::lround(360.0 * i / n_angles)) % 360;
        if (deg >= EYELID_WEDGE_START_DEG && deg <= EYELID_WEDGE_END_DEG)
            mask[i] = 1;
    }
    return mask;
}

} // namespace

std::optional<DetectResult> detect_limbus(const cv::Mat& image,
                                          cv::Point2d seed_center,
                                          int n_angles,
                                          int m_harmonics,
                                          double gradient_sigma,
                                          double radial_smoothing,
                                          bool skip_eyelid_wedges,
                                          double r_min,
                                          double r_max)
{
    if (n_angles < 8 || m_harmonics < 1 || !(r_max > r_min))
        return std::nullopt;

    cv::Mat gray;
    if (image.channels() == 1)
        gray = image;
    else
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    cv::Mat smoothed;
    cv::GaussianBlur(gray, smoothed, cv::Size(0, 0), gradient_sigma, gradient_sigma);
    cv::Mat gx;
    cv::Mat gy;
    cv::Sobel(smoothed, gx, CV_32F, 1, 0, 3);
    cv::Sobel(smoothed, gy, CV_32F, 0, 1, 3);

    const auto eyelid_mask = build_eyelid_mask(n_angles, skip_eyelid_wedges);
    const auto kernel = cheshm::Daugman::gaussian_smoothing_kernel(radial_smoothing);

    const int n_r =
        std::max(static_cast<int>(std::lround(r_max - r_min)) + 1, 16);

    auto r_theta = radial_search(gx,
                                 gy,
                                 cv::Point2f{static_cast<float>(seed_center.x), static_cast<float>(seed_center.y)},
                                 n_angles,
                                 static_cast<float>(r_min),
                                 static_cast<float>(r_max),
                                 n_r,
                                 std::span<const std::uint8_t>{eyelid_mask},
                                 std::span<const double>{kernel});

    const int n_valid = static_cast<int>(std::count_if(r_theta.begin(), r_theta.end(), [](double v) { return v >= 0.0; }));
    if (n_valid < static_cast<int>(n_angles * 0.3))
        return std::nullopt;

    cheshm::Daugman::cyclic_interpolate(r_theta);
    auto R_theta = cheshm::Daugman::fourier_truncate(r_theta, m_harmonics);

    std::vector<double> thetas(static_cast<std::size_t>(n_angles));
    for (int i = 0; i < n_angles; ++i)
        thetas[static_cast<std::size_t>(i)] = 2.0 * CV_PI * static_cast<double>(i) / static_cast<double>(n_angles);

    return DetectResult{seed_center, std::move(thetas), std::move(R_theta)};
}

} // namespace cheshm::Daugman::active_contour
