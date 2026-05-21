// Samples points along an ellipse outline at fixed angular intervals,
// using a precomputed sin/cos lookup table so the inner loop avoids
// transcendental calls.

#pragma once

#include <array>
#include <cmath>
#include <opencv2/core.hpp>
#include <vector>

namespace cheshm {

namespace detail {

// Indexed ``0..720`` so the cos lookup ``sinTable[360 + (90 + angle) % 360]``
// works for ``angle in [0, 360]``.
inline const std::array<float, 721> sinTable = []() {
    std::array<float, 721> t{};
    for (int i = 0; i < 721; ++i) {
        t[i] = std::sin(static_cast<float>((i - 360) * CV_PI / 180.0));
    }
    return t;
}();

inline void sincos_deg(int angle, float &cosval, float &sinval)
{
    angle += (angle < 0 ? 360 : 0);
    sinval = sinTable[360 + angle];
    cosval = sinTable[360 + (90 + angle) % 360];
}

}  // namespace detail

// Returns points sampled along the ellipse outline at every ``delta``
// degrees, in pixel-rounded image coordinates.
inline std::vector<cv::Point> ellipse_to_points(const cv::RotatedRect &ellipse, int delta)
{
    int angle = static_cast<int>(ellipse.angle);
    while (angle < 0) angle += 360;
    while (angle > 360) angle -= 360;

    float alpha, beta;
    detail::sincos_deg(angle, alpha, beta);

    std::vector<cv::Point> points;
    for (int i = 0; i < 360; i += delta) {
        float cosI, sinI;
        detail::sincos_deg(i, cosI, sinI);
        const float x = 0.5f * ellipse.size.width * cosI;
        const float y = 0.5f * ellipse.size.height * sinI;
        points.emplace_back(
            static_cast<int>(std::roundf(ellipse.center.x + x * alpha - y * beta)),
            static_cast<int>(std::roundf(ellipse.center.y + x * beta + y * alpha)));
    }
    return points;
}

}  // namespace cheshm
