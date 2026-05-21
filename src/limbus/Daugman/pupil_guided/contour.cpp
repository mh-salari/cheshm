#include "cheshm/limbus/Daugman/pupil_guided/contour.hpp"

#include "cheshm/limbus/Daugman/_common/contour_ops.hpp"
#include "cheshm/limbus/Daugman/pupil_guided/radial_search.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <span>
#include <vector>

namespace cheshm::Daugman::pupil_guided
{
namespace
{

// Pupil-ellipse radius at each angle. Caller must have set ``a_p`` and
// ``b_p`` to the semi-major and semi-minor axes; ``angle_rad`` is the
// orientation of the major axis (cv::fitEllipse's angle is for the
// width axis, so a 90° rotation is applied when width is the minor).
std::vector<double> pupil_radial_profile(double a_p, double b_p, double angle_rad, std::span<const double> thetas)
{
    std::vector<double> r_p(thetas.size());
    for (std::size_t i = 0; i < thetas.size(); ++i)
    {
        const double phi = thetas[i] - angle_rad;
        const double cphi = std::cos(phi);
        const double sphi = std::sin(phi);
        r_p[i] = (a_p * b_p) / std::hypot(b_p * cphi, a_p * sphi);
    }
    return r_p;
}

} // namespace

std::optional<DetectResult> detect_limbus(const cv::Mat& image,
                                          cv::Point2d seed_center,
                                          const PupilEllipse& pupil,
                                          int n_angles,
                                          int m_harmonics,
                                          double gradient_sigma,
                                          double radial_smoothing,
                                          double k_min,
                                          double k_max)
{
    if (n_angles < 8 || m_harmonics < 1 || !(k_max > k_min) || !(k_min > 0))
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

    const double pw = pupil.size.width;
    const double ph = pupil.size.height;
    const double a_p = std::max(pw, ph) / 2.0;
    const double b_p = std::min(pw, ph) / 2.0;
    const double angle_deg = (pw >= ph) ? pupil.angle_deg : pupil.angle_deg + 90.0;
    const double angle_rad = angle_deg * CV_PI / 180.0;

    std::vector<double> thetas(static_cast<std::size_t>(n_angles));
    for (int i = 0; i < n_angles; ++i)
        thetas[static_cast<std::size_t>(i)] = 2.0 * CV_PI * static_cast<double>(i) / static_cast<double>(n_angles);

    const auto r_p = pupil_radial_profile(a_p, b_p, angle_rad, std::span<const double>{thetas});

    std::vector<float> r_min_arr(static_cast<std::size_t>(n_angles));
    std::vector<float> r_max_arr(static_cast<std::size_t>(n_angles));
    float widest = 0.0f;
    for (int i = 0; i < n_angles; ++i)
    {
        r_min_arr[static_cast<std::size_t>(i)] = static_cast<float>(k_min * r_p[static_cast<std::size_t>(i)]);
        r_max_arr[static_cast<std::size_t>(i)] = static_cast<float>(k_max * r_p[static_cast<std::size_t>(i)]);
        widest = std::max(widest, r_max_arr[i] - r_min_arr[i]);
    }
    const int n_r = std::max(static_cast<int>(std::lround(widest)) + 1, 16);

    const auto kernel = cheshm::Daugman::gaussian_smoothing_kernel(radial_smoothing);
    auto r_theta = radial_search(gx,
                                 gy,
                                 cv::Point2f{static_cast<float>(seed_center.x), static_cast<float>(seed_center.y)},
                                 n_angles,
                                 std::span<const float>{r_min_arr},
                                 std::span<const float>{r_max_arr},
                                 n_r,
                                 std::span<const double>{kernel});

    const int n_valid =
        static_cast<int>(std::count_if(r_theta.begin(), r_theta.end(), [](double v) { return v >= 0.0; }));
    if (n_valid < static_cast<int>(n_angles * 0.3))
        return std::nullopt;

    cheshm::Daugman::cyclic_interpolate(r_theta);
    auto R_theta = cheshm::Daugman::fourier_truncate(r_theta, m_harmonics);

    return DetectResult{seed_center, std::move(thetas), std::move(R_theta)};
}

} // namespace cheshm::Daugman::pupil_guided
