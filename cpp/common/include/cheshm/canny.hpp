// Sobel-based Canny edge detector with magnitude-histogram threshold
// selection, plus a three-pass edge-map cleanup pipeline (thin 2×2
// corner clusters, drop over-connected pixels, rewrite short staircase
// / spur patterns).
//
// From Santini, T., Fuhl, W., Kasneci, E. (2018). "PuRe: Robust pupil
// detection for real-time pervasive eye tracking." *Computer Vision
// and Image Understanding*, 170, 40-50.

#pragma once

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

    std::vector<int> histogram(bins, 0);
    cv::Mat res_idx = (bins - 1) * magnitude;
    res_idx.convertTo(res_idx, CV_16U);
    for (int i = 0; i < res_idx.rows; ++i)
    {
        const short* p = res_idx.ptr<short>(i);
        for (int j = 0; j < res_idx.cols; ++j)
            ++histogram[p[j]];
    }

    int sum = 0;
    const int nonEdgePixels = static_cast<int>(non_edge_pixels_ratio * in.rows * in.cols);
    float high_th = 0;
    for (int i = 0; i < bins; ++i)
    {
        sum += histogram[i];
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

    // Hysteresis: every strong-edge pixel seeds a flood fill that pulls
    // in connected weak-edge pixels.
    const int pic_x = edgeType.cols;
    const int pic_y = edgeType.rows;
    const int area = pic_x * pic_y;
    int lines_idx = 0;
    int idx = 0;

    std::vector<int> lines;
    edge.setTo(0);
    for (int i = 1; i < pic_y - 1; ++i)
    {
        for (int j = 1; j < pic_x - 1; ++j)
        {
            if (edgeType.data[idx + j] != 255 || edge.data[idx + j] != 0)
                continue;

            edge.data[idx + j] = 255;
            lines_idx = 1;
            lines.clear();
            lines.push_back(idx + j);
            int akt_idx = 0;

            while (akt_idx < lines_idx)
            {
                const int akt_pos = lines[akt_idx];
                ++akt_idx;

                if (akt_pos - pic_x - 1 < 0 || akt_pos + pic_x + 1 >= area)
                    continue;

                for (int k1 = -1; k1 < 2; ++k1)
                    for (int k2 = -1; k2 < 2; ++k2)
                    {
                        const int neighbour = (akt_pos + (k1 * pic_x)) + k2;
                        if (edge.data[neighbour] != 0 || edgeType.data[neighbour] == 0)
                            continue;
                        edge.data[neighbour] = 255;
                        lines.push_back(neighbour);
                        ++lines_idx;
                    }
            }
        }
        idx += pic_x;
    }

    return edge;
}

// Three-pass edge cleanup:
//   1) Thin 2×2 corner clusters down to a single edge pixel.
//   2) Drop pixels with more than 3 lit neighbours (over-connected).
//   3) Local-pattern rewrites that straighten short staircase / spur
//      segments produced by the Canny step.
//
// ``roi_*`` bound the processed window; values are inset by 5 pixels
// and clamped to the safe ``[5, dim - 5]`` interior. The no-arg form
// processes the entire safe interior.
inline void filter_edges(cv::Mat& edges, int roi_x_start, int roi_x_end, int roi_y_start, int roi_y_end)
{
    int start_x = std::max(roi_x_start + 5, 5);
    int start_y = std::max(roi_y_start + 5, 5);
    int end_x = std::min(roi_x_end - 5, edges.cols - 5);
    int end_y = std::min(roi_y_end - 5, edges.rows - 5);

    for (int j = start_y; j < end_y; ++j)
        for (int i = start_x; i < end_x; ++i)
        {
            uchar box[9];
            box[4] = edges.data[(edges.cols * j) + i];
            if (box[4])
            {
                box[1] = edges.data[(edges.cols * (j - 1)) + i];
                box[3] = edges.data[(edges.cols * j) + (i - 1)];
                box[5] = edges.data[(edges.cols * j) + (i + 1)];
                box[7] = edges.data[(edges.cols * (j + 1)) + i];

                if (box[5] && box[7])
                    edges.data[(edges.cols * j) + i] = 0;
                if (box[5] && box[1])
                    edges.data[(edges.cols * j) + i] = 0;
                if (box[3] && box[7])
                    edges.data[(edges.cols * j) + i] = 0;
                if (box[3] && box[1])
                    edges.data[(edges.cols * j) + i] = 0;
            }
        }

    for (int j = start_y; j < end_y; ++j)
        for (int i = start_x; i < end_x; ++i)
        {
            int neig = 0;
            for (int k1 = -1; k1 < 2; ++k1)
                for (int k2 = -1; k2 < 2; ++k2)
                    if (edges.data[(edges.cols * (j + k1)) + (i + k2)] > 0)
                        ++neig;
            if (neig > 3)
                edges.data[(edges.cols * j) + i] = 0;
        }

    for (int j = start_y; j < end_y; ++j)
        for (int i = start_x; i < end_x; ++i)
        {
            uchar box[17];
            box[4] = edges.data[(edges.cols * j) + i];
            if (box[4])
            {
                box[0] = edges.data[(edges.cols * (j - 1)) + (i - 1)];
                box[1] = edges.data[(edges.cols * (j - 1)) + i];
                box[2] = edges.data[(edges.cols * (j - 1)) + (i + 1)];
                box[3] = edges.data[(edges.cols * j) + (i - 1)];
                box[5] = edges.data[(edges.cols * j) + (i + 1)];
                box[6] = edges.data[(edges.cols * (j + 1)) + (i - 1)];
                box[7] = edges.data[(edges.cols * (j + 1)) + i];
                box[8] = edges.data[(edges.cols * (j + 1)) + (i + 1)];

                box[9] = edges.data[(edges.cols * j) + (i + 2)];
                box[10] = edges.data[(edges.cols * (j + 2)) + i];
                box[11] = edges.data[(edges.cols * j) + (i + 3)];
                box[12] = edges.data[(edges.cols * (j - 1)) + (i + 2)];
                box[13] = edges.data[(edges.cols * (j + 1)) + (i + 2)];
                box[14] = edges.data[(edges.cols * (j + 3)) + i];
                box[15] = edges.data[(edges.cols * (j + 2)) + (i - 1)];
                box[16] = edges.data[(edges.cols * (j + 2)) + (i + 1)];

                if ((box[10] && !box[7]) && (box[8] || box[6]))
                {
                    edges.data[(edges.cols * (j + 1)) + (i - 1)] = 0;
                    edges.data[(edges.cols * (j + 1)) + (i + 1)] = 0;
                    edges.data[(edges.cols * (j + 1)) + i] = 255;
                }
                if ((box[14] && !box[7] && !box[10]) && ((box[8] || box[6]) && (box[16] || box[15])))
                {
                    edges.data[(edges.cols * (j + 1)) + (i + 1)] = 0;
                    edges.data[(edges.cols * (j + 1)) + (i - 1)] = 0;
                    edges.data[(edges.cols * (j + 2)) + (i + 1)] = 0;
                    edges.data[(edges.cols * (j + 2)) + (i - 1)] = 0;
                    edges.data[(edges.cols * (j + 1)) + i] = 255;
                    edges.data[(edges.cols * (j + 2)) + i] = 255;
                }
                if ((box[9] && !box[5]) && (box[8] || box[2]))
                {
                    edges.data[(edges.cols * (j + 1)) + (i + 1)] = 0;
                    edges.data[(edges.cols * (j - 1)) + (i + 1)] = 0;
                    edges.data[(edges.cols * j) + (i + 1)] = 255;
                }
                if ((box[11] && !box[5] && !box[9]) && ((box[8] || box[2]) && (box[13] || box[12])))
                {
                    edges.data[(edges.cols * (j + 1)) + (i + 1)] = 0;
                    edges.data[(edges.cols * (j - 1)) + (i + 1)] = 0;
                    edges.data[(edges.cols * (j + 1)) + (i + 2)] = 0;
                    edges.data[(edges.cols * (j - 1)) + (i + 2)] = 0;
                    edges.data[(edges.cols * j) + (i + 1)] = 255;
                    edges.data[(edges.cols * j) + (i + 2)] = 255;
                }
            }
        }

    for (int j = start_y; j < end_y; ++j)
        for (int i = start_x; i < end_x; ++i)
        {
            uchar box[33];
            box[4] = edges.data[(edges.cols * j) + i];
            if (box[4])
            {
                box[0] = edges.data[(edges.cols * (j - 1)) + (i - 1)];
                box[1] = edges.data[(edges.cols * (j - 1)) + i];
                box[2] = edges.data[(edges.cols * (j - 1)) + (i + 1)];
                box[3] = edges.data[(edges.cols * j) + (i - 1)];
                box[5] = edges.data[(edges.cols * j) + (i + 1)];
                box[6] = edges.data[(edges.cols * (j + 1)) + (i - 1)];
                box[7] = edges.data[(edges.cols * (j + 1)) + i];
                box[8] = edges.data[(edges.cols * (j + 1)) + (i + 1)];

                box[9] = edges.data[(edges.cols * (j - 1)) + (i + 2)];
                box[10] = edges.data[(edges.cols * (j - 1)) + (i - 2)];
                box[11] = edges.data[(edges.cols * (j + 1)) + (i + 2)];
                box[12] = edges.data[(edges.cols * (j + 1)) + (i - 2)];

                box[13] = edges.data[(edges.cols * (j - 2)) + (i - 1)];
                box[14] = edges.data[(edges.cols * (j - 2)) + (i + 1)];
                box[15] = edges.data[(edges.cols * (j + 2)) + (i - 1)];
                box[16] = edges.data[(edges.cols * (j + 2)) + (i + 1)];

                box[17] = edges.data[(edges.cols * (j - 3)) + (i - 1)];
                box[18] = edges.data[(edges.cols * (j - 3)) + (i + 1)];
                box[19] = edges.data[(edges.cols * (j + 3)) + (i - 1)];
                box[20] = edges.data[(edges.cols * (j + 3)) + (i + 1)];

                box[21] = edges.data[(edges.cols * (j + 1)) + (i + 3)];
                box[22] = edges.data[(edges.cols * (j + 1)) + (i - 3)];
                box[23] = edges.data[(edges.cols * (j - 1)) + (i + 3)];
                box[24] = edges.data[(edges.cols * (j - 1)) + (i - 3)];

                box[25] = edges.data[(edges.cols * (j - 2)) + (i - 2)];
                box[26] = edges.data[(edges.cols * (j + 2)) + (i + 2)];
                box[27] = edges.data[(edges.cols * (j - 2)) + (i + 2)];
                box[28] = edges.data[(edges.cols * (j + 2)) + (i - 2)];

                if (box[7] && box[2] && box[9])
                    edges.data[(edges.cols * j) + i] = 0;
                if (box[7] && box[0] && box[10])
                    edges.data[(edges.cols * j) + i] = 0;
                if (box[1] && box[8] && box[11])
                    edges.data[(edges.cols * j) + i] = 0;
                if (box[1] && box[6] && box[12])
                    edges.data[(edges.cols * j) + i] = 0;

                if (box[0] && box[13] && box[17] && box[8] && box[11] && box[21])
                    edges.data[(edges.cols * j) + i] = 0;
                if (box[2] && box[14] && box[18] && box[6] && box[12] && box[22])
                    edges.data[(edges.cols * j) + i] = 0;
                if (box[6] && box[15] && box[19] && box[2] && box[9] && box[23])
                    edges.data[(edges.cols * j) + i] = 0;
                if (box[8] && box[16] && box[20] && box[0] && box[10] && box[24])
                    edges.data[(edges.cols * j) + i] = 0;

                if (box[0] && box[25] && box[2] && box[27])
                    edges.data[(edges.cols * j) + i] = 0;
                if (box[0] && box[25] && box[6] && box[28])
                    edges.data[(edges.cols * j) + i] = 0;
                if (box[8] && box[26] && box[2] && box[27])
                    edges.data[(edges.cols * j) + i] = 0;
                if (box[8] && box[26] && box[6] && box[28])
                    edges.data[(edges.cols * j) + i] = 0;

                uchar box2[18];
                box2[1] = edges.data[(edges.cols * j) + (i - 1)];
                box2[2] = edges.data[(edges.cols * (j - 1)) + (i - 2)];
                box2[3] = edges.data[(edges.cols * (j - 2)) + (i - 3)];
                box2[4] = edges.data[(edges.cols * (j - 1)) + (i + 1)];
                box2[5] = edges.data[(edges.cols * (j - 2)) + (i + 2)];
                box2[6] = edges.data[(edges.cols * (j + 1)) + (i - 2)];
                box2[7] = edges.data[(edges.cols * (j + 2)) + (i - 3)];
                box2[8] = edges.data[(edges.cols * (j + 1)) + (i + 1)];
                box2[9] = edges.data[(edges.cols * (j + 2)) + (i + 2)];
                box2[10] = edges.data[(edges.cols * (j + 1)) + i];
                box2[15] = edges.data[(edges.cols * (j - 1)) + (i - 1)];
                box2[16] = edges.data[(edges.cols * (j - 2)) + (i - 2)];
                box2[11] = edges.data[(edges.cols * (j + 2)) + (i + 1)];
                box2[12] = edges.data[(edges.cols * (j + 3)) + (i + 2)];
                box2[13] = edges.data[(edges.cols * (j + 2)) + (i - 1)];
                box2[14] = edges.data[(edges.cols * (j + 3)) + (i - 2)];

                if (box2[1] && box2[2] && box2[3] && box2[4] && box2[5])
                    edges.data[(edges.cols * j) + i] = 0;
                if (box2[1] && box2[6] && box2[7] && box2[8] && box2[9])
                    edges.data[(edges.cols * j) + i] = 0;
                if (box2[10] && box2[11] && box2[12] && box2[4] && box2[5])
                    edges.data[(edges.cols * j) + i] = 0;
                if (box2[10] && box2[13] && box2[14] && box2[15] && box2[16])
                    edges.data[(edges.cols * j) + i] = 0;
            }
        }
}

inline void filter_edges(cv::Mat& edges)
{
    filter_edges(edges, 0, edges.cols, 0, edges.rows);
}

} // namespace cheshm
