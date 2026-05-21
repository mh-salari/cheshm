// 8-connected hysteresis flood that links strong-edge seeds to
// weak-edge pixels.
//
// From Fuhl, W., Kübler, T., Sippel, K., Rosenstiel, W., Kasneci, E.
// (2015). "ExCuSe: Robust Pupil Detection in Real-World Scenarios."
// *CAIP 2015*, 39-51.

#pragma once

#include <limits>
#include <opencv2/core.hpp>
#include <optional>
#include <vector>

namespace cheshm
{

// Returns a new uint8 single-channel mask the size of ``strong`` /
// ``weak``. A pixel is lit (255) when it is set in ``strong`` or when
// it is reachable from a ``strong`` seed through 8-connected
// neighbours that are set in ``weak``. ``max_fill_pixels`` caps the
// per-seed fill queue; pass ``std::nullopt`` for an uncapped fill.
inline cv::Mat
hysteresis_flood_fill(const cv::Mat& strong, const cv::Mat& weak, std::optional<int> max_fill_pixels = std::nullopt)
{
    const int pic_x = strong.cols;
    const int pic_y = strong.rows;
    const int area = pic_x * pic_y;
    const int cap = max_fill_pixels.value_or(std::numeric_limits<int>::max());

    cv::Mat check = cv::Mat::zeros(pic_y, pic_x, CV_8U);

    std::vector<int> lines;
    if (max_fill_pixels)
        lines.reserve(*max_fill_pixels);

    int idx = 0;
    for (int i = 1; i < pic_y - 1; ++i)
    {
        for (int j = 1; j < pic_x - 1; ++j)
        {
            if (strong.data[idx + j] == 0 || check.data[idx + j] != 0)
                continue;

            check.data[idx + j] = 255;
            lines.clear();
            lines.push_back(idx + j);

            std::size_t akt_idx = 0;
            while (akt_idx < lines.size() && static_cast<int>(lines.size()) < cap)
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
                        if (static_cast<int>(lines.size()) < cap)
                            lines.push_back(neighbour);
                    }
                }
            }
        }
        idx += pic_x;
    }

    return check;
}

} // namespace cheshm
