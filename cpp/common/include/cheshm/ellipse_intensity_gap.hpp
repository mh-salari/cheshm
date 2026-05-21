// Inner-vs-outer intensity gate for candidate pupil ellipses: returns
// the (outer_ring_mean - inner_box_mean) intensity gap and whether it
// clears the caller-supplied threshold. The outer ring is one
// axis-aligned box with another axis-aligned cutout removed.

#pragma once

#include <opencv2/core.hpp>

namespace cheshm
{

// Half-open integer interval box. Iteration uses
// ``i in [x_start, x_end)`` and ``j in [y_start, y_end)``.
struct PixelBox
{
    int x_start;
    int x_end;
    int y_start;
    int y_end;
};

struct IntensityGapResult
{
    bool passes; // (outer_mean - inner_mean) > threshold
    float diff;  // outer_mean - inner_mean, or 0 when either box is empty
};

// Computes the intensity gap and gates it against ``threshold``. Pixels
// at coordinate ``(i, j)`` are only sampled when ``0 < i < image.cols``
// and ``0 < j < image.rows`` — note the strict bounds, matching the
// original ExCuSe / ElSe checks.
//
// Iteration order is column-major (outer ``i``, inner ``j``) so that
// floating-point sums are accumulated in the same order as the
// original detector implementations.
inline IntensityGapResult check_ellipse_intensity_gap(const cv::Mat& image,
                                                      const PixelBox& inner_box,
                                                      const PixelBox& outer_box,
                                                      const PixelBox& outer_cutout,
                                                      float threshold)
{
    float inner_sum = 0.0f;
    float inner_count = 0.0f;
    for (int i = inner_box.x_start; i < inner_box.x_end; ++i)
    {
        for (int j = inner_box.y_start; j < inner_box.y_end; ++j)
        {
            if (i > 0 && i < image.cols && j > 0 && j < image.rows)
            {
                inner_sum += image.data[(image.cols * j) + i];
                inner_count++;
            }
        }
    }
    if (inner_count <= 0.0f)
        return {false, 0.0f};
    const float inner_mean = inner_sum / inner_count;

    float outer_sum = 0.0f;
    float outer_count = 0.0f;
    for (int i = outer_box.x_start; i < outer_box.x_end; ++i)
    {
        for (int j = outer_box.y_start; j < outer_box.y_end; ++j)
        {
            const bool inside_cutout = i >= outer_cutout.x_start && i < outer_cutout.x_end &&
                                       j >= outer_cutout.y_start && j < outer_cutout.y_end;
            if (inside_cutout)
                continue;
            if (i > 0 && i < image.cols && j > 0 && j < image.rows)
            {
                outer_sum += image.data[(image.cols * j) + i];
                outer_count++;
            }
        }
    }
    if (outer_count <= 0.0f)
        return {false, 0.0f};
    const float outer_mean = outer_sum / outer_count;

    const float diff = outer_mean - inner_mean;
    return {diff > threshold, diff};
}

} // namespace cheshm
