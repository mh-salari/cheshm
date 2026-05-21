// Inner-vs-outer intensity-gap vote that scores a candidate ellipse
// against the image it was fit to. Samples the outline at every 10°
// and tallies the fraction of sample points where the intensity on
// the inside differs from the intensity on the outside by at least
// ``bias`` in the expected direction (darker inside).
//
// From Santini, T., Fuhl, W., Kasneci, E. (2018). "PuRe: Robust pupil
// detection for real-time pervasive eye tracking." *Computer Vision
// and Image Understanding*, 170, 40-50.

#pragma once

#include "cheshm/helpers/ellipses/ellipse_sampling.hpp"

#include <cmath>
#include <cstdlib>
#include <opencv2/core.hpp>
#include <optional>
#include <vector>

namespace cheshm
{

// Returns the vote ratio in ``[0, 1]``, or ``std::nullopt`` when no
// outline sample fell inside the image. ``bias`` is the minimum
// intensity gap (uchar units) between the inside and outside means
// required to count a sample as supporting the fit.
inline std::optional<float> outline_contrast_confidence(const cv::Mat& frame, const cv::RotatedRect& outline, int bias)
{
    if (outline.size.width <= 0 || outline.size.height <= 0)
        return std::nullopt;

    const cv::Rect boundaries{0, 0, frame.cols, frame.rows};
    const float minor_axis = std::min(outline.size.width, outline.size.height);
    const int delta = static_cast<int>(0.15f * minor_axis);
    const cv::Point c = outline.center;

    int evaluated = 0;
    int valid_count = 0;

    const std::vector<cv::Point> outline_points = cheshm::ellipse_to_points(outline, 10);
    for (const cv::Point& p : outline_points)
    {
        const int dxp = p.x - c.x;
        const int dyp = p.y - c.y;

        float a = 0.0f;
        if (dxp != 0)
            a = static_cast<float>(dyp) / static_cast<float>(dxp);
        const float b = c.y - a * c.x;

        if (a == 0.0f)
            continue;

        if (std::abs(dxp) > std::abs(dyp))
        {
            const int sx = p.x - delta;
            const int ex = p.x + delta;
            const int sy = static_cast<int>(std::roundf(a * sx + b));
            const int ey = static_cast<int>(std::roundf(a * ex + b));
            ++evaluated;

            if (!boundaries.contains(cv::Point{sx, sy}) || !boundaries.contains(cv::Point{ex, ey}))
                continue;

            float m1 = 0.0f;
            for (int x = sx; x < p.x; ++x)
                m1 += frame.ptr<uchar>(static_cast<int>(std::roundf(a * x + b)))[x];
            m1 = std::roundf(m1 / delta);

            float m2 = 0.0f;
            for (int x = p.x + 1; x <= ex; ++x)
                m2 += frame.ptr<uchar>(static_cast<int>(std::roundf(a * x + b)))[x];
            m2 = std::roundf(m2 / delta);

            if (p.x < c.x)
            {
                if (m1 > m2 + bias)
                    ++valid_count;
            }
            else
            {
                if (m2 > m1 + bias)
                    ++valid_count;
            }
        }
        else
        {
            const int sy = p.y - delta;
            const int ey = p.y + delta;
            const int sx = static_cast<int>(std::roundf((sy - b) / a));
            const int ex = static_cast<int>(std::roundf((ey - b) / a));
            ++evaluated;

            if (!boundaries.contains(cv::Point{sx, sy}) || !boundaries.contains(cv::Point{ex, ey}))
                continue;

            float m1 = 0.0f;
            for (int y = sy; y < p.y; ++y)
                m1 += frame.ptr<uchar>(y)[static_cast<int>(std::roundf((y - b) / a))];
            m1 = std::roundf(m1 / delta);

            float m2 = 0.0f;
            for (int y = p.y + 1; y <= ey; ++y)
                m2 += frame.ptr<uchar>(y)[static_cast<int>(std::roundf((y - b) / a))];
            m2 = std::roundf(m2 / delta);

            if (p.y < c.y)
            {
                if (m1 > m2 + bias)
                    ++valid_count;
            }
            else
            {
                if (m2 > m1 + bias)
                    ++valid_count;
            }
        }
    }

    if (evaluated == 0)
        return std::nullopt;
    return static_cast<float>(valid_count) / static_cast<float>(evaluated);
}

} // namespace cheshm
