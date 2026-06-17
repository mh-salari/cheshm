// Periodic cubic smoothing spline through a closed sequence of 2-D points.
//
// Per coordinate the fitted knot values g minimise
// sum (g_i - y_i)^2 + lambda * integral g''(u)^2 over the closed natural cubic
// spline. ``smoothness`` is lambda: 0 interpolates every point, larger values
// trade closeness for lower curvature. The points are parameterised by
// cumulative chord length u in [0, 1), so lambda is scale-invariant.
//
// In Reinsch form g = y - lambda * D * gamma, where (R + lambda * D^2) gamma =
// D y, D is the cyclic second-difference operator and R the cyclic-tridiagonal
// cubic Gram matrix (Green & Silverman). M = R + lambda * D^2 is symmetric
// cyclic pentadiagonal and is solved in O(k) by an LDL^T factorisation of its
// open band plus a rank-4 Sherman-Morrison-Woodbury correction for the wrap.
// The smoothed knots are interpolated by the periodic cubic spline and
// resampled to ``n_samples`` points.

#pragma once

#include "cheshm/helpers/shape/spline.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>
#include <vector>

namespace cheshm
{

namespace detail
{

// LDL^T factorisation of a symmetric pentadiagonal open (non-cyclic) band:
// M = L D L^T with unit lower-bidiagonal L. ``diag`` is the main diagonal,
// sub1[i] = M[i, i-1] and sub2[i] = M[i, i-2].
struct BandLDLT
{
    std::vector<double> d;
    std::vector<double> l1;
    std::vector<double> l2;
};

inline BandLDLT
band_ldlt(const std::vector<double>& diag, const std::vector<double>& sub1, const std::vector<double>& sub2)
{
    const int k = static_cast<int>(diag.size());
    BandLDLT f{std::vector<double>(k), std::vector<double>(k, 0.0), std::vector<double>(k, 0.0)};
    for (int i = 0; i < k; ++i)
    {
        const double l2 = (i >= 2) ? sub2[i] / f.d[i - 2] : 0.0;
        const double l1 = (i >= 1) ? (sub1[i] - (i >= 2 ? l2 * f.d[i - 2] * f.l1[i - 1] : 0.0)) / f.d[i - 1] : 0.0;
        double piv = diag[i];
        if (i >= 1)
        {
            piv -= l1 * l1 * f.d[i - 1];
        }
        if (i >= 2)
        {
            piv -= l2 * l2 * f.d[i - 2];
        }
        f.d[i] = piv;
        f.l1[i] = l1;
        f.l2[i] = l2;
    }
    return f;
}

// Forward / diagonal / backward substitution against a BandLDLT factor, in place.
inline void band_solve(const BandLDLT& f, std::vector<double>& b)
{
    const int k = static_cast<int>(b.size());
    for (int i = 0; i < k; ++i)
    {
        if (i >= 1)
        {
            b[i] -= f.l1[i] * b[i - 1];
        }
        if (i >= 2)
        {
            b[i] -= f.l2[i] * b[i - 2];
        }
    }
    for (int i = 0; i < k; ++i)
    {
        b[i] /= f.d[i];
    }
    for (int i = k - 1; i >= 0; --i)
    {
        if (i + 1 < k)
        {
            b[i] -= f.l1[i + 1] * b[i + 1];
        }
        if (i + 2 < k)
        {
            b[i] -= f.l2[i + 2] * b[i + 2];
        }
    }
}

} // namespace detail

inline std::vector<cv::Point> smoothing_spline(const std::vector<cv::Point>& points, double smoothness, int n_samples)
{
    std::vector<cv::Point> out;
    if (n_samples < 3)
    {
        return out;
    }

    // Consecutive duplicate points have a zero-length chord and carry no
    // parameterisation information; drop them so 1/h stays finite.
    std::vector<cv::Point> knot;
    knot.reserve(points.size());
    for (const cv::Point& p : points)
    {
        if (knot.empty() || p != knot.back())
        {
            knot.push_back(p);
        }
    }
    if (knot.size() > 1 && knot.front() == knot.back())
    {
        knot.pop_back();
    }
    const int k = static_cast<int>(knot.size());
    if (k < 4)
    {
        return out;
    }

    // Cumulative chord-length parameterisation, u in [0, 1).
    std::vector<double> chord(k);
    double total = 0.0;
    for (int i = 0; i < k; ++i)
    {
        const int next = (i + 1) % k;
        const double dx = static_cast<double>(knot[next].x) - knot[i].x;
        const double dy = static_cast<double>(knot[next].y) - knot[i].y;
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

    const double lambda = std::max(0.0, smoothness);

    // D is the cyclic second-difference operator (symmetric tridiagonal): main
    // diagonal e[i], off-diagonal f[i] = 1/h[i].
    std::vector<double> f(k);
    std::vector<double> e(k);
    for (int i = 0; i < k; ++i)
    {
        f[i] = 1.0 / h[i];
    }
    for (int i = 0; i < k; ++i)
    {
        e[i] = -(f[(i + k - 1) % k] + f[i]);
    }

    // Band entries of M = R + lambda * D^2 from local formulas. M[i,i+1] and
    // M[i,i+2] are cyclic in the index; the three distinct wrap values feed the
    // Woodbury correction below.
    auto m_super1 = [&](int i) { // M[i, i+1]
        const int ni = (i + 1) % k;
        return h[i] / 6.0 + lambda * f[i] * (e[i] + e[ni]);
    };
    auto m_super2 = [&](int i) { // M[i, i+2]
        return lambda * f[i] * f[(i + 1) % k];
    };
    std::vector<double> diag(k);
    std::vector<double> sub1(k, 0.0);
    std::vector<double> sub2(k, 0.0);
    for (int i = 0; i < k; ++i)
    {
        const int pv = (i + k - 1) % k;
        diag[i] = (h[pv] + h[i]) / 3.0 + lambda * (e[i] * e[i] + f[i] * f[i] + f[pv] * f[pv]);
    }
    for (int i = 1; i < k; ++i)
    {
        sub1[i] = m_super1(i - 1);
    }
    for (int i = 2; i < k; ++i)
    {
        sub2[i] = m_super2(i - 2);
    }
    const double wrap1 = m_super1(k - 1); // M[k-1, 0]
    const double wrap2 = m_super2(k - 2); // M[k-2, 0]
    const double wrap3 = m_super2(k - 1); // M[k-1, 1]

    const detail::BandLDLT band = detail::band_ldlt(diag, sub1, sub2);

    // Sherman-Morrison-Woodbury for the wrap entries: M = B + Z E Z^T over the
    // index set S = {0, 1, k-2, k-1}, with B the open band already factored.
    const int sidx[4] = {0, 1, k - 2, k - 1};
    double emat[4][4] = {{0.0}};
    emat[0][3] = emat[3][0] = wrap1;
    emat[0][2] = emat[2][0] = wrap2;
    emat[1][3] = emat[3][1] = wrap3;

    std::vector<double> bz[4]; // columns of B^{-1} Z
    for (int c = 0; c < 4; ++c)
    {
        std::vector<double> col(k, 0.0);
        col[sidx[c]] = 1.0;
        detail::band_solve(band, col);
        bz[c] = std::move(col);
    }
    cv::Matx44d capacitance = cv::Matx44d::eye(); // I + E (Z^T B^{-1} Z)
    for (int a = 0; a < 4; ++a)
    {
        for (int b = 0; b < 4; ++b)
        {
            double eg = 0.0;
            for (int m = 0; m < 4; ++m)
            {
                eg += emat[a][m] * bz[b][sidx[m]];
            }
            capacitance(a, b) += eg;
        }
    }

    auto multiply_d = [&](const std::vector<double>& y) {
        std::vector<double> r(k);
        for (int i = 0; i < k; ++i)
        {
            const int pv = (i + k - 1) % k;
            const int nx = (i + 1) % k;
            r[i] = e[i] * y[i] + f[i] * y[nx] + f[pv] * y[pv];
        }
        return r;
    };
    // Solve M x = rhs in O(k): x = B^{-1} rhs - B^{-1} Z (I + E G)^{-1} E Z^T B^{-1} rhs.
    auto solve_m = [&](std::vector<double> rhs) {
        detail::band_solve(band, rhs);
        cv::Vec4d s(rhs[sidx[0]], rhs[sidx[1]], rhs[sidx[2]], rhs[sidx[3]]);
        cv::Vec4d es;
        for (int a = 0; a < 4; ++a)
        {
            double v = 0.0;
            for (int m = 0; m < 4; ++m)
            {
                v += emat[a][m] * s[m];
            }
            es[a] = v;
        }
        const cv::Vec4d w = capacitance.solve(es, cv::DECOMP_LU);
        for (int i = 0; i < k; ++i)
        {
            double corr = 0.0;
            for (int c = 0; c < 4; ++c)
            {
                corr += bz[c][i] * w[c];
            }
            rhs[i] -= corr;
        }
        return rhs;
    };

    // Smooth each coordinate: g = y - lambda * D ( M^{-1} (D y) ).
    std::vector<double> gx(k);
    std::vector<double> gy(k);
    std::vector<double> yx(k);
    std::vector<double> yy(k);
    for (int i = 0; i < k; ++i)
    {
        yx[i] = knot[i].x;
        yy[i] = knot[i].y;
    }
    if (lambda == 0.0)
    {
        gx = yx;
        gy = yy;
    }
    else
    {
        const std::vector<double> dgx = multiply_d(solve_m(multiply_d(yx)));
        const std::vector<double> dgy = multiply_d(solve_m(multiply_d(yy)));
        for (int i = 0; i < k; ++i)
        {
            gx[i] = yx[i] - lambda * dgx[i];
            gy[i] = yy[i] - lambda * dgy[i];
        }
    }

    // Interpolate the smoothed knots with the periodic cubic spline and resample.
    std::vector<double> mx;
    std::vector<double> my;
    detail::periodic_cubic_second_derivs(u, gx, mx);
    detail::periodic_cubic_second_derivs(u, gy, my);

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
        const double sx = detail::evaluate_segment(t, u_seg, u_next, h[seg], gx[seg], gx[next], mx[seg], mx[next]);
        const double sy = detail::evaluate_segment(t, u_seg, u_next, h[seg], gy[seg], gy[next], my[seg], my[next]);
        out[j] = cv::Point(static_cast<int>(std::lround(sx)), static_cast<int>(std::lround(sy)));
    }
    return out;
}

} // namespace cheshm
