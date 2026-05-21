// Daugman-family helpers shared between the active_contour and
// pupil_guided detectors: 1-D Gaussian kernel builder, cyclic linear
// interpolation of failed-angle radii, and the Daugman 2007
// Fourier-truncation low-pass on the boundary radii.

#pragma once

#include <algorithm>
#include <cmath>
#include <iterator>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <span>
#include <vector>

namespace cheshm::Daugman
{

// Odd-length 1-D Gaussian kernel sized to roughly ±3σ.
inline std::vector<double> gaussian_smoothing_kernel(double sigma)
{
    if (sigma <= 0.0)
        return {1.0};
    const int k_sz = std::max(static_cast<int>(std::lround(sigma * 6.0)) | 1, 3);
    const cv::Mat k = cv::getGaussianKernel(k_sz, sigma, CV_64F);
    std::vector<double> out(static_cast<std::size_t>(k_sz));
    for (int i = 0; i < k_sz; ++i)
        out[static_cast<std::size_t>(i)] = k.at<double>(i);
    return out;
}

// Fill -1 entries of r_theta by cyclic linear interpolation between
// valid samples. Equivalent to ``np.interp`` over an extended index
// range that wraps the valid indices by ±N.
inline void cyclic_interpolate(std::vector<double>& r_theta)
{
    const int n = static_cast<int>(r_theta.size());
    std::vector<int> valid_idx;
    std::vector<double> valid_val;
    valid_idx.reserve(static_cast<std::size_t>(n));
    valid_val.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        if (r_theta[i] >= 0.0)
        {
            valid_idx.push_back(i);
            valid_val.push_back(r_theta[i]);
        }
    if (static_cast<int>(valid_idx.size()) == n || valid_idx.empty())
        return;

    const int v = static_cast<int>(valid_idx.size());
    std::vector<int> ext_idx;
    std::vector<double> ext_val;
    ext_idx.reserve(static_cast<std::size_t>(3 * v));
    ext_val.reserve(static_cast<std::size_t>(3 * v));
    for (int shift : {-n, 0, n})
        for (int i = 0; i < v; ++i)
        {
            ext_idx.push_back(valid_idx[i] + shift);
            ext_val.push_back(valid_val[i]);
        }

    for (int i = 0; i < n; ++i)
    {
        if (r_theta[i] >= 0.0)
            continue;
        const auto it = std::upper_bound(ext_idx.begin(), ext_idx.end(), i);
        const std::ptrdiff_t hi = std::distance(ext_idx.begin(), it);
        const std::ptrdiff_t lo = hi - 1;
        const int x0 = ext_idx[static_cast<std::size_t>(lo)];
        const int x1 = ext_idx[static_cast<std::size_t>(hi)];
        const double y0 = ext_val[static_cast<std::size_t>(lo)];
        const double y1 = ext_val[static_cast<std::size_t>(hi)];
        const double t = static_cast<double>(i - x0) / static_cast<double>(x1 - x0);
        r_theta[i] = y0 + t * (y1 - y0);
    }
}

// Daugman 2007 Fourier low-pass: keep the first ``m_harmonics``
// coefficients (and their negative-frequency conjugates), weight by
// ``1 / (1 + k / m_harmonics)``, inverse transform.
inline std::vector<double> fourier_truncate(std::span<const double> r_theta, int m_harmonics)
{
    const int n = static_cast<int>(r_theta.size());
    cv::Mat input(1, n, CV_64F);
    for (int i = 0; i < n; ++i)
        input.at<double>(0, i) = r_theta[static_cast<std::size_t>(i)];

    cv::Mat coeffs;
    cv::dft(input, coeffs, cv::DFT_COMPLEX_OUTPUT);

    for (int k = 0; k < n; ++k)
    {
        const bool kept = (k < m_harmonics) || (k >= n - m_harmonics + 1);
        if (!kept)
        {
            coeffs.at<cv::Vec2d>(0, k)[0] = 0.0;
            coeffs.at<cv::Vec2d>(0, k)[1] = 0.0;
        }
    }
    for (int k = 0; k < m_harmonics; ++k)
    {
        const double w = 1.0 / (1.0 + static_cast<double>(k) / m_harmonics);
        coeffs.at<cv::Vec2d>(0, k)[0] *= w;
        coeffs.at<cv::Vec2d>(0, k)[1] *= w;
        if (k > 0)
        {
            coeffs.at<cv::Vec2d>(0, n - k)[0] *= w;
            coeffs.at<cv::Vec2d>(0, n - k)[1] *= w;
        }
    }

    cv::Mat inverse;
    cv::idft(coeffs, inverse, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

    std::vector<double> out(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        out[static_cast<std::size_t>(i)] = inverse.at<double>(0, i);
    return out;
}

} // namespace cheshm::Daugman
