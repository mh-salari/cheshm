// 8-connected hysteresis flood that links strong-edge seeds to
// weak-edge pixels.
//
// From Fuhl, W., Kübler, T., Sippel, K., Rosenstiel, W., Kasneci, E.
// (2015). "ExCuSe: Robust Pupil Detection in Real-World Scenarios."
// *CAIP 2015*, 39-51.

#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace cheshm
{

// Per-seed flood-fill cap. Bounds runtime on pathological inputs where
// the weak-edge mask is densely connected.
inline constexpr int HYSTERESIS_MAX_FILL_PIXELS = 10000;

// Returns a new uint8 single-channel mask the size of ``strong`` /
// ``weak``. A pixel is lit (255) when it is set in ``strong`` or when
// it is reachable from a ``strong`` seed through 8-connected
// neighbours that are set in ``weak``.
inline cv::Mat hysteresis_flood_fill(const cv::Mat& strong, const cv::Mat& weak)
{
    const int pic_x = strong.cols;
    const int pic_y = strong.rows;
    const int area = pic_x * pic_y;

    cv::Mat check = cv::Mat::zeros(pic_y, pic_x, CV_8U);

    std::vector<int> lines(HYSTERESIS_MAX_FILL_PIXELS, 0);
    int lines_idx = 0;
    int idx = 0;

    for (int i = 1; i < pic_y - 1; ++i)
    {
        for (int j = 1; j < pic_x - 1; ++j)
        {
            if (strong.data[idx + j] == 0 || check.data[idx + j] != 0)
                continue;

            check.data[idx + j] = 255;
            lines_idx = 1;
            lines[0] = idx + j;

            int akt_idx = 0;
            while (akt_idx < lines_idx && lines_idx < HYSTERESIS_MAX_FILL_PIXELS)
            {
                const int akt_pos = lines[akt_idx++];
                if (akt_pos - pic_x - 1 < 0 || akt_pos + pic_x + 1 >= area)
                    continue;

                for (int k1 = -1; k1 < 2; ++k1)
                {
                    for (int k2 = -1; k2 < 2; ++k2)
                    {
                        const int neighbour = akt_pos + k1 * pic_x + k2;
                        if (check.data[neighbour] != 0 || weak.data[neighbour] == 0)
                            continue;
                        check.data[neighbour] = 255;
                        if (lines_idx < HYSTERESIS_MAX_FILL_PIXELS)
                        {
                            lines[lines_idx++] = neighbour;
                        }
                    }
                }
            }
        }
        idx += pic_x;
    }

    return check;
}

} // namespace cheshm
