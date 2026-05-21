// Starburst pupil detector defaults.

#pragma once

namespace cheshm::Starburst::defaults
{

inline constexpr int EDGE_THRESHOLD = 16;         // intensity jump that counts as a ray edge
inline constexpr int RAYS = 18;                   // number of starburst rays per iteration
inline constexpr int MIN_FEATURE_CANDIDATES = 10; // RANSAC kicks in above this edge-point count
inline constexpr int CR_WINDOW_SIZE = 301;        // corneal-reflection search-window side (px). 0 disables CR removal.
inline constexpr int CR_RATIO_TO_IMAGE_HEIGHT =
    2; // largest accepted CR radius = image_height / this. 0 disables CR removal.
inline constexpr int MAX_EDGE_POINTS = 1024; // cap on edge points kept for RANSAC
inline constexpr int SEED_THRESHOLD = 30;    // auto-seed: centroid of pixels below this intensity

} // namespace cheshm::Starburst::defaults
