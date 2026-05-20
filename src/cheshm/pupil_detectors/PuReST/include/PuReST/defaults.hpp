// PuReST tracker defaults.

#pragma once

namespace cheshm::PuReST::defaults {

// Maximum scaled tracking-region size (pixels). If the tracking ROI
// (centred on the previous pupil, sized to its major axis) would scale
// up beyond this, the tracker reduces its scaling ratio to keep runtime
// bounded.
inline constexpr int MAX_SCALED_REGION = 100;

// Minimum outline-confidence required to accept an outline-tracker fit.
inline constexpr float MIN_OUTLINE_CONFIDENCE = 0.65f;

// Bias on the high-threshold cutoff in the bright/dark mask computation.
inline constexpr int THRESHOLD_BIAS = 5;

// Maximum allowed growth of the major axis between the last accepted
// pupil and the new outline-tracker fit. Larger growth implies the
// outline drifted off-pupil; the candidate is rejected.
inline constexpr float MAX_MAJOR_AXIS_RATIO = 1.05f;

// Minimum overall confidence for the greedy-search fit to be accepted.
inline constexpr float MIN_GREEDY_CONFIDENCE = 0.66f;

}  // namespace cheshm::PuReST::defaults