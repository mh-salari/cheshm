// Robust polar-Fourier "pupil form" fit (Wyatt, Vision Res. 1995).
//
// The human pupil margin is a smooth, near-circular closed curve well
// described by a low-order Fourier series in polar coordinates about the
// pupil centre:
//
//   R(theta) = a_0 + sum_{k=1..K} ( a_k cos(k theta) + b_k sin(k theta) )
//
// Glints and eyelashes cut the threshold contour *inward* (toward the
// centre); they are not pupil shape. The fit therefore uses asymmetric
// iteratively-reweighted least squares: inward residuals are down-weighted
// hard (controlled by `inward_rejection`), outward ones gently, so the
// curve follows the outer pupil envelope and bridges the intrusions.
//
// The least-squares step solves the (2K+1) weighted normal equations with
// cv::solve (Cholesky, SVD fallback). The polar origin is refined once from
// the reconstructed boundary's centroid.

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <vector>

namespace cheshm
{

struct PupilForm
{
    std::vector<cv::Point> boundary; // smoothed margin, crop-local integer points
    double cx = 0.0;                 // centroid of the smoothed margin
    double cy = 0.0;
    bool ok = false;
};

namespace detail
{

inline double polar_radius(const std::vector<double>& coef, int harmonics, double theta)
{
    double r = coef[0];
    for (int k = 1; k <= harmonics; ++k)
    {
        r += coef[2 * k - 1] * std::cos(k * theta) + coef[2 * k] * std::sin(k * theta);
    }
    return r;
}

} // namespace detail

inline PupilForm fit_pupil_form(
    const std::vector<cv::Point>& contour, int harmonics, int samples, int iterations, double inward_rejection)
{
    constexpr double PI = 3.14159265358979323846;
    const int n = static_cast<int>(contour.size());
    const int kk = std::max(1, harmonics);
    const int ncoef = 2 * kk + 1;
    PupilForm out;
    if (n < ncoef + 1 || samples < 8)
    {
        return out;
    }

    double cx = 0.0;
    double cy = 0.0;
    for (const auto& p : contour)
    {
        cx += p.x;
        cy += p.y;
    }
    cx /= n;
    cy /= n;

    const double reject = std::max(0.1, inward_rejection);
    std::vector<double> coef(ncoef, 0.0);
    double bcx = cx;
    double bcy = cy;
    double min_radius = 0.0;
    double max_radius = 0.0;

    for (int pass = 0; pass < 2; ++pass)
    {
        std::vector<double> theta(n);
        std::vector<double> radius(n);
        std::vector<double> w(n, 1.0);
        for (int i = 0; i < n; ++i)
        {
            const double dx = contour[i].x - cx;
            const double dy = contour[i].y - cy;
            theta[i] = std::atan2(dy, dx);
            radius[i] = std::hypot(dx, dy);
        }

        std::vector<double> row(ncoef);
        for (int it = 0; it < std::max(1, iterations); ++it)
        {
            cv::Mat AtA = cv::Mat::zeros(ncoef, ncoef, CV_64F);
            cv::Mat Atb = cv::Mat::zeros(ncoef, 1, CV_64F);
            for (int i = 0; i < n; ++i)
            {
                row[0] = 1.0;
                for (int k = 1; k <= kk; ++k)
                {
                    row[2 * k - 1] = std::cos(k * theta[i]);
                    row[2 * k] = std::sin(k * theta[i]);
                }
                const double wi = w[i];
                for (int a = 0; a < ncoef; ++a)
                {
                    Atb.at<double>(a) += wi * row[a] * radius[i];
                    for (int b = 0; b < ncoef; ++b)
                    {
                        AtA.at<double>(a, b) += wi * row[a] * row[b];
                    }
                }
            }
            cv::Mat sol;
            if (!cv::solve(AtA, Atb, sol, cv::DECOMP_CHOLESKY) && !cv::solve(AtA, Atb, sol, cv::DECOMP_SVD))
            {
                return out;
            }
            for (int a = 0; a < ncoef; ++a)
            {
                coef[a] = sol.at<double>(a);
            }

            std::vector<double> absres(n);
            for (int i = 0; i < n; ++i)
            {
                absres[i] = std::abs(radius[i] - detail::polar_radius(coef, kk, theta[i]));
            }
            std::nth_element(absres.begin(), absres.begin() + n / 2, absres.end());
            const double s = absres[n / 2] * 1.4826 + 1e-6;
            const double c_in = s / reject; // higher rejection -> tighter scale -> drops inward faster
            const double c_out = 3.0 * s;
            for (int i = 0; i < n; ++i)
            {
                const double resid = radius[i] - detail::polar_radius(coef, kk, theta[i]);
                const double cc = (resid < 0.0) ? c_in : c_out;
                const double t = resid / cc;
                w[i] = 1.0 / (1.0 + t * t);
            }
        }

        out.boundary.assign(samples, cv::Point());
        bcx = 0.0;
        bcy = 0.0;
        min_radius = std::numeric_limits<double>::max();
        max_radius = std::numeric_limits<double>::lowest();
        for (int j = 0; j < samples; ++j)
        {
            const double th = -PI + (2.0 * PI * j) / samples;
            const double rr = detail::polar_radius(coef, kk, th);
            min_radius = std::min(min_radius, rr);
            max_radius = std::max(max_radius, rr);
            const double x = cx + rr * std::cos(th);
            const double y = cy + rr * std::sin(th);
            out.boundary[j] = cv::Point(static_cast<int>(std::lround(x)), static_cast<int>(std::lround(y)));
            bcx += x;
            bcy += y;
        }
        bcx /= samples;
        bcy /= samples;
        cx = bcx; // refine the polar origin for the next pass
        cy = bcy;
    }

    // A pupil margin is convex and near-circular: its polar radius stays
    // positive and its max-to-min ratio is small, below ~2.5 even for a
    // strongly foreshortened pupil. A non-positive radius or a larger ratio
    // means the low-order series could not represent the contour and folded the
    // boundary into a self-intersecting shape. Reject it; the caller then keeps
    // the raw contour.
    constexpr double MAX_RADIUS_RATIO = 3.0;
    if (min_radius <= 0.0 || max_radius > MAX_RADIUS_RATIO * min_radius)
    {
        return out;
    }

    out.cx = bcx;
    out.cy = bcy;
    out.ok = true;
    return out;
}

} // namespace cheshm
