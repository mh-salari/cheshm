// Sorts cv::connectedComponents labels by column-major seed order
// (smallest x, ties on smallest y) so the iteration matches the
// per-curve tie-break logic in ExCuSe's get_curves.

#pragma once

#include <algorithm>
#include <opencv2/core.hpp>
#include <vector>

namespace cheshm::ExCuSe
{

inline std::vector<int> sort_labels_by_column_major_seed(
    const std::vector<std::vector<cv::Point>>& components, int n_labels)
{
    std::vector<int> order;
    order.reserve(n_labels > 1 ? n_labels - 1 : 0);
    for (int label = 1; label < n_labels; ++label)
        if (!components[label].empty())
            order.push_back(label);

    std::sort(order.begin(), order.end(), [&](int a, int b) {
        auto seed = [](const std::vector<cv::Point>& c) {
            cv::Point s = c[0];
            for (const auto& p : c)
                if (p.x < s.x || (p.x == s.x && p.y < s.y))
                    s = p;
            return s;
        };
        const cv::Point sa = seed(components[a]);
        const cv::Point sb = seed(components[b]);
        return sa.x < sb.x || (sa.x == sb.x && sa.y < sb.y);
    });
    return order;
}

} // namespace cheshm::ExCuSe
