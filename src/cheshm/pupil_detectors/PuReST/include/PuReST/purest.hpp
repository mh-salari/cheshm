// PuReST pupil detector + tracker.
//
// Reference: Santini, T., Fuhl, W., Kasneci, E. (2018). "PuReST: Robust
// pupil tracking for real-time pervasive eye tracking." *ETRA 2018*.
//
// Stateful tracker built on PuRe. The first call runs full PuRe
// detection; subsequent calls reuse the previously detected pupil to
// constrain a local greedy + outline search, falling back to full PuRe
// detection if both tracking paths fail.

#pragma once

#include <opencv2/core.hpp>
#include <optional>

namespace cheshm::PuReST {

struct DetectResult {
    cv::RotatedRect ellipse;
    float confidence;
};

class Tracker {
public:
    Tracker(float min_pupil_diameter_mm,
            float max_pupil_diameter_mm,
            float canthi_distance_mm,
            int outline_bias);

    std::optional<DetectResult> detect(const cv::Mat &frame);
    void reset();

private:
    float min_pupil_diameter_mm_;
    float max_pupil_diameter_mm_;
    float canthi_distance_mm_;
    int outline_bias_;

    bool has_previous_;
    cv::RotatedRect previous_pupil_;
    float previous_confidence_;
    bool outline_seed_valid_;
    cv::RotatedRect outline_seed_pupil_;

    cv::Mat open_kernel_;
    cv::Mat dilate_kernel_;
};

}  // namespace cheshm::PuReST