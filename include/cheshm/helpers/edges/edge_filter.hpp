// Three-pass edge-map cleanup for thin binary edge images:
//   1) Thin 2x2 corner clusters down to a single edge pixel.
//   2) Drop pixels with more than 3 lit neighbours (over-connected).
//   3) Local-pattern rewrites that straighten short staircase / spur
//      segments.

#pragma once

#include <algorithm>
#include <opencv2/core.hpp>

namespace cheshm
{

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
