// Periodic interpolating cubic spline through a closed sequence of
// 2-D points, parameterised by cumulative chord length. The spline is
// the unique C^2 cubic that passes through every input point and joins
// back onto itself with matching value, slope, and curvature.
//
// The exported helper samples the spline at ``n_samples`` evenly spaced
// parameter values and returns the centroid of the enclosed polygon
// via Green's theorem.
//
// Second-derivative formulation: for each interval [u_i, u_{i+1}] of
// width h_i, the cubic spline is determined by the second derivatives
// M_i at the knots. C^2 continuity gives the cyclic tridiagonal system
//
//   h_{i-1} M_{i-1} + 2 (h_{i-1} + h_i) M_i + h_i M_{i+1}
//       = 6 ((y_{i+1} - y_i) / h_i - (y_i - y_{i-1}) / h_{i-1})
//
// with all indices wrapped modulo K. The cyclic system is reduced to
// two standard tridiagonal solves via Sherman-Morrison.

#pragma once

#include <cmath>
#include <opencv2/core.hpp>
#include <vector>

namespace cheshm
{

struct SplineCentroid
{
    double cx;
    double cy;
    double area;
};

namespace detail
{

inline void thomas_solve(const std::vector<double>& a,
                         std::vector<double> b,
                         const std::vector<double>& c,
                         const std::vector<double>& r,
                         std::vector<double>& x)
{
    const int n = static_cast<int>(b.size());
    std::vector<double> rr(r);
    for (int i = 1; i < n; ++i)
    {
        const double m = a[i] / b[i - 1];
        b[i] -= m * c[i - 1];
        rr[i] -= m * rr[i - 1];
    }
    x.assign(n, 0.0);
    x[n - 1] = rr[n - 1] / b[n - 1];
    for (int i = n - 2; i >= 0; --i)
    {
        x[i] = (rr[i] - c[i] * x[i + 1]) / b[i];
    }
}

inline void cyclic_tridiag_solve(const std::vector<double>& a,
                                 const std::vector<double>& b,
                                 const std::vector<double>& c,
                                 double alpha,
                                 double beta,
                                 const std::vector<double>& r,
                                 std::vector<double>& x)
{
    const int n = static_cast<int>(b.size());
    const double gamma = -b[0];
    std::vector<double> bb(b);
    bb[0] -= gamma;
    bb[n - 1] -= alpha * beta / gamma;

    std::vector<double> y;
    thomas_solve(a, bb, c, r, y);

    std::vector<double> u(n, 0.0);
    u[0] = gamma;
    u[n - 1] = alpha;
    std::vector<double> z;
    thomas_solve(a, bb, c, u, z);

    const double fact = (y[0] + beta * y[n - 1] / gamma) / (1.0 + z[0] + beta * z[n - 1] / gamma);
    x.assign(n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        x[i] = y[i] - fact * z[i];
    }
}

inline void
periodic_cubic_second_derivs(const std::vector<double>& u, const std::vector<double>& y, std::vector<double>& M)
{
    const int K = static_cast<int>(y.size());
    std::vector<double> h(K);
    for (int i = 0; i < K - 1; ++i)
    {
        h[i] = u[i + 1] - u[i];
    }
    h[K - 1] = 1.0 - u[K - 1];

    std::vector<double> a(K), bb(K), c(K), r(K);
    for (int i = 0; i < K; ++i)
    {
        const int prev = (i + K - 1) % K;
        const double h_prev = h[prev];
        const double h_curr = h[i];
        a[i] = h_prev;
        bb[i] = 2.0 * (h_prev + h_curr);
        c[i] = h_curr;
        const int next = (i + 1) % K;
        r[i] = 6.0 * ((y[next] - y[i]) / h_curr - (y[i] - y[prev]) / h_prev);
    }

    const double alpha = a[0];
    const double beta = c[K - 1];
    cyclic_tridiag_solve(a, bb, c, alpha, beta, r, M);
}

inline double
evaluate_segment(double t, double u_i, double u_ip1, double h, double y_i, double y_ip1, double M_i, double M_ip1)
{
    const double a = u_ip1 - t;
    const double b = t - u_i;
    return (a * a * a * M_i + b * b * b * M_ip1) / (6.0 * h) + (y_i / h - M_i * h / 6.0) * a +
           (y_ip1 / h - M_ip1 * h / 6.0) * b;
}

} // namespace detail

inline SplineCentroid spline_polygon_centroid(const std::vector<cv::Point>& points, int n_samples)
{
    // Periodic cubic spline degree is 3, so the algorithm needs at
    // least four distinct knots.
    const int K = static_cast<int>(points.size());
    if (K < 4 || n_samples < 3)
    {
        return {0.0, 0.0, 0.0};
    }

    std::vector<double> chord(K);
    double total = 0.0;
    for (int i = 0; i < K; ++i)
    {
        const int next = (i + 1) % K;
        const double dx = static_cast<double>(points[next].x) - points[i].x;
        const double dy = static_cast<double>(points[next].y) - points[i].y;
        chord[i] = std::sqrt(dx * dx + dy * dy);
        total += chord[i];
    }
    if (total <= 0.0)
    {
        return {0.0, 0.0, 0.0};
    }

    std::vector<double> u(K);
    u[0] = 0.0;
    double acc = 0.0;
    for (int i = 1; i < K; ++i)
    {
        acc += chord[i - 1];
        u[i] = acc / total;
    }

    std::vector<double> x(K), y(K);
    for (int i = 0; i < K; ++i)
    {
        x[i] = static_cast<double>(points[i].x);
        y[i] = static_cast<double>(points[i].y);
    }

    std::vector<double> Mx, My;
    detail::periodic_cubic_second_derivs(u, x, Mx);
    detail::periodic_cubic_second_derivs(u, y, My);

    std::vector<double> h(K);
    for (int i = 0; i < K - 1; ++i)
    {
        h[i] = u[i + 1] - u[i];
    }
    h[K - 1] = 1.0 - u[K - 1];

    std::vector<double> sx(n_samples), sy(n_samples);
    int seg = 0;
    for (int j = 0; j < n_samples; ++j)
    {
        const double t = (n_samples == 1) ? 0.0 : static_cast<double>(j) / static_cast<double>(n_samples - 1);
        while (seg < K - 1 && t > u[seg + 1])
        {
            ++seg;
        }
        const int next = (seg + 1) % K;
        const double u_seg = u[seg];
        const double u_next = (seg + 1 < K) ? u[seg + 1] : 1.0;
        sx[j] = detail::evaluate_segment(t, u_seg, u_next, h[seg], x[seg], x[next], Mx[seg], Mx[next]);
        sy[j] = detail::evaluate_segment(t, u_seg, u_next, h[seg], y[seg], y[next], My[seg], My[next]);
    }

    double signed_area = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    for (int j = 0; j < n_samples; ++j)
    {
        const int k = (j + 1) % n_samples;
        const double cross = sx[j] * sy[k] - sx[k] * sy[j];
        signed_area += cross;
        cx += (sx[j] + sx[k]) * cross;
        cy += (sy[j] + sy[k]) * cross;
    }
    signed_area *= 0.5;
    if (signed_area == 0.0)
    {
        double mean_x = 0.0;
        double mean_y = 0.0;
        for (int j = 0; j < n_samples; ++j)
        {
            mean_x += sx[j];
            mean_y += sy[j];
        }
        return {mean_x / n_samples, mean_y / n_samples, 0.0};
    }
    cx /= 6.0 * signed_area;
    cy /= 6.0 * signed_area;
    return {cx, cy, std::fabs(signed_area)};
}

} // namespace cheshm
