// Periodic cubic SMOOTHING spline through a closed sequence of 2-D points.
//
// ``smoothness`` 0 reproduces the interpolating spline (the curve passes
// through every point); larger values trade point-closeness for lower
// curvature (Reinsch smoothing spline) without shrinking the curve inward.
//
// The points are parameterised by cumulative chord length u in [0, 1).
// Each coordinate is smoothed independently: the smoothed knot values g
// solve
//
//   (I + lambda * K) g = y,   K = 6 * D^T * A^{-1} * D
//
// where D is the periodic second-difference operator and A is the cyclic
// tridiagonal Gram matrix of the cubic spline (A = 6R in the Green &
// Silverman formulation). Because u is normalised to [0, 1], lambda is
// scale-invariant. The smoothed values are then interpolated by the
// periodic cubic spline and resampled. Point counts here are small, so the
// matrices are formed densely and solved with cv::solve / cv::invert.

#pragma once

#include "cheshm/helpers/shape/spline.hpp"

#include <cmath>
#include <opencv2/core.hpp>
#include <vector>

namespace cheshm
{

inline std::vector<cv::Point> smoothing_spline(const std::vector<cv::Point>& points, double smoothness, int n_samples)
{
    const int k = static_cast<int>(points.size());
    std::vector<cv::Point> out;
    if (k < 4 || n_samples < 3)
    {
        return out;
    }

    // Cumulative chord-length parameterisation, u in [0, 1).
    std::vector<double> chord(k);
    double total = 0.0;
    for (int i = 0; i < k; ++i)
    {
        const int next = (i + 1) % k;
        const double dx = static_cast<double>(points[next].x) - points[i].x;
        const double dy = static_cast<double>(points[next].y) - points[i].y;
        chord[i] = std::sqrt(dx * dx + dy * dy);
        total += chord[i];
    }
    if (total <= 0.0)
    {
        return out;
    }
    std::vector<double> u(k);
    u[0] = 0.0;
    double acc = 0.0;
    for (int i = 1; i < k; ++i)
    {
        acc += chord[i - 1];
        u[i] = acc / total;
    }
    std::vector<double> h(k);
    for (int i = 0; i < k - 1; ++i)
    {
        h[i] = u[i + 1] - u[i];
    }
    h[k - 1] = 1.0 - u[k - 1];

    // D: periodic second-difference operator; A: cyclic tridiagonal Gram matrix.
    cv::Mat d = cv::Mat::zeros(k, k, CV_64F);
    cv::Mat a = cv::Mat::zeros(k, k, CV_64F);
    for (int i = 0; i < k; ++i)
    {
        const int prev = (i + k - 1) % k;
        const int next = (i + 1) % k;
        const double hp = h[prev];
        const double hc = h[i];
        d.at<double>(i, prev) += 1.0 / hp;
        d.at<double>(i, i) -= 1.0 / hp + 1.0 / hc;
        d.at<double>(i, next) += 1.0 / hc;
        a.at<double>(i, prev) += hp;
        a.at<double>(i, i) += 2.0 * (hp + hc);
        a.at<double>(i, next) += hc;
    }
    cv::Mat a_inv;
    cv::invert(a, a_inv, cv::DECOMP_SVD);
    const cv::Mat penalty = 6.0 * d.t() * a_inv * d;
    const cv::Mat system = cv::Mat::eye(k, k, CV_64F) + std::max(0.0, smoothness) * penalty;

    cv::Mat xs(k, 1, CV_64F);
    cv::Mat ys(k, 1, CV_64F);
    for (int i = 0; i < k; ++i)
    {
        xs.at<double>(i) = points[i].x;
        ys.at<double>(i) = points[i].y;
    }
    cv::Mat gx;
    cv::Mat gy;
    if (!cv::solve(system, xs, gx, cv::DECOMP_LU) || !cv::solve(system, ys, gy, cv::DECOMP_LU))
    {
        return out;
    }

    // Interpolate the smoothed knot values with the periodic cubic spline.
    std::vector<double> gxv(k);
    std::vector<double> gyv(k);
    for (int i = 0; i < k; ++i)
    {
        gxv[i] = gx.at<double>(i);
        gyv[i] = gy.at<double>(i);
    }
    std::vector<double> mx;
    std::vector<double> my;
    detail::periodic_cubic_second_derivs(u, gxv, mx);
    detail::periodic_cubic_second_derivs(u, gyv, my);

    out.resize(n_samples);
    int seg = 0;
    for (int j = 0; j < n_samples; ++j)
    {
        const double t = static_cast<double>(j) / static_cast<double>(n_samples); // [0, 1) closes the loop
        while (seg < k - 1 && t > u[seg + 1])
        {
            ++seg;
        }
        const int next = (seg + 1) % k;
        const double u_seg = u[seg];
        const double u_next = (seg + 1 < k) ? u[seg + 1] : 1.0;
        const double sx = detail::evaluate_segment(t, u_seg, u_next, h[seg], gxv[seg], gxv[next], mx[seg], mx[next]);
        const double sy = detail::evaluate_segment(t, u_seg, u_next, h[seg], gyv[seg], gyv[next], my[seg], my[next]);
        out[j] = cv::Point(static_cast<int>(std::lround(sx)), static_cast<int>(std::lround(sy)));
    }
    return out;
}

} // namespace cheshm
