// Sobel-based Canny edge detector with magnitude-histogram threshold
// selection.
//
// From Santini, T., Fuhl, W., Kasneci, E. (2018). "PuRe: Robust pupil
// detection for real-time pervasive eye tracking." *Computer Vision
// and Image Understanding*, 170, 40-50.

#pragma once

#include "cheshm/edges/edge_hysteresis.hpp"

#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace cheshm
{

// Custom Canny edge detector. ``in`` is the (downscaled, normalised)
// grayscale frame. ``dx``, ``dy``, ``magnitude``, ``edgeType`` and
// ``edge`` are caller-supplied work buffers sized like ``in``; the
// returned edge map shares storage with ``edge``.
//
//   ``bins``: histogram bins for threshold selection.
//   ``non_edge_pixels_ratio``: fraction of pixels treated as non-edge.
//   ``low_high_threshold_ratio``: low threshold = ratio * high.
inline cv::Mat canny(const cv::Mat& in,
                     cv::Mat& dx,
                     cv::Mat& dy,
                     cv::Mat& magnitude,
                     cv::Mat& edgeType,
                     cv::Mat& edge,
                     int bins,
                     float non_edge_pixels_ratio,
                     float low_high_threshold_ratio)
{
    cv::Mat blurred;
    cv::GaussianBlur(in, blurred, cv::Size(5, 5), 1.5, 1.5, cv::BORDER_REPLICATE);

    cv::Sobel(blurred, dx, dx.type(), 1, 0, 7, 1, cv::BORDER_REPLICATE);
    cv::Sobel(blurred, dy, dy.type(), 0, 1, 7, 1, cv::BORDER_REPLICATE);

    double minMag = 0;
    double maxMag = 0;
    cv::magnitude(dx, dy, magnitude);
    cv::minMaxLoc(magnitude, &minMag, &maxMag);

    magnitude = magnitude / maxMag;

    cv::Mat res_idx = (bins - 1) * magnitude;
    res_idx.convertTo(res_idx, CV_16U);

    const std::vector<int> channels = {0};
    const std::vector<int> hist_size = {bins};
    const std::vector<float> ranges = {0.0f, static_cast<float>(bins)};
    cv::Mat hist_mat;
    cv::calcHist(std::vector<cv::Mat>{res_idx}, channels, cv::Mat(), hist_mat, hist_size, ranges);

    int sum = 0;
    const int nonEdgePixels = static_cast<int>(non_edge_pixels_ratio * in.rows * in.cols);
    float high_th = 0;
    for (int i = 0; i < bins; ++i)
    {
        sum += static_cast<int>(hist_mat.at<float>(i));
        if (sum > nonEdgePixels)
        {
            high_th = static_cast<float>(i + 1) / bins;
            break;
        }
    }
    const float low_th = low_high_threshold_ratio * high_th;

    // Non-maximum suppression.
    const float tg22_5 = 0.4142135623730950488016887242097f;
    const float tg67_5 = 2.4142135623730950488016887242097f;
    edgeType.setTo(0);
    for (int i = 1; i < magnitude.rows - 1; ++i)
    {
        uchar* _edgeType = edgeType.ptr<uchar>(i);
        const float* p_res = magnitude.ptr<float>(i);
        const float* p_res_t = magnitude.ptr<float>(i - 1);
        const float* p_res_b = magnitude.ptr<float>(i + 1);
        const float* p_x = dx.ptr<float>(i);
        const float* p_y = dy.ptr<float>(i);

        for (int j = 1; j < magnitude.cols - 1; ++j)
        {
            const float m = p_res[j];
            if (m < low_th)
                continue;

            const float iy = p_y[j];
            const float ix = p_x[j];
            const float y = std::abs(iy);
            const float x = std::abs(ix);

            const uchar val = p_res[j] > high_th ? 255 : 128;

            const float tg22_5x = tg22_5 * x;
            if (y < tg22_5x)
            {
                if (m > p_res[j - 1] && m >= p_res[j + 1])
                    _edgeType[j] = val;
            }
            else
            {
                const float tg67_5x = tg67_5 * x;
                if (y > tg67_5x)
                {
                    if (m > p_res_b[j] && m >= p_res_t[j])
                        _edgeType[j] = val;
                }
                else
                {
                    if ((iy <= 0) == (ix <= 0))
                    {
                        if (m > p_res_t[j - 1] && m >= p_res_b[j + 1])
                            _edgeType[j] = val;
                    }
                    else
                    {
                        if (m > p_res_b[j - 1] && m >= p_res_t[j + 1])
                            _edgeType[j] = val;
                    }
                }
            }
        }
    }

    cv::Mat strong;
    cv::Mat weak;
    cv::compare(edgeType, 255, strong, cv::CMP_EQ);
    cv::compare(edgeType, 0, weak, cv::CMP_NE);
    edge = hysteresis_flood_fill(strong, weak);
    return edge;
}


} // namespace cheshm
