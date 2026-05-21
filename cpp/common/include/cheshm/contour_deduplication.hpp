// Drops contour curves whose first pixel was already claimed by an
// earlier curve in the input vector.

#pragma once

#include <cstdint>
#include <map>
#include <opencv2/core.hpp>
#include <vector>

namespace cheshm {

// Linearises a pixel coordinate into a single int. ``cols`` is the
// stride used to combine row and column into a unique key.
inline int point_hash(cv::Point p, int cols) { return p.y * cols + p.x; }

// Walks ``curves`` from last to first. The first time a curve's
// starting pixel is seen, every pixel on that curve is registered in
// the hash map. Any later curve that starts on an already-registered
// pixel is erased from ``curves``.
inline void deduplicate_contours_by_first_point(
    std::vector<std::vector<cv::Point>> &curves, int cols)
{
    std::map<int, std::uint8_t> contour_map;
    for (std::size_t i = curves.size(); i-- > 0;) {
        if (contour_map.count(point_hash(curves[i][0], cols)) > 0) {
            curves.erase(curves.begin() + i);
        } else {
            for (std::size_t j = 0; j < curves[i].size(); ++j) {
                contour_map[point_hash(curves[i][j], cols)] = 1;
            }
        }
    }
}

}  // namespace cheshm
