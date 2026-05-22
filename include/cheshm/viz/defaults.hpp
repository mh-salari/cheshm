#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace cheshm::viz::defaults
{

inline constexpr int LABEL_HEIGHT = 32;
inline const cv::Scalar LABEL_BG{40.0, 40.0, 40.0};
inline const cv::Scalar LABEL_FG{255.0, 255.0, 255.0};
inline constexpr int LABEL_FONT = cv::FONT_HERSHEY_SIMPLEX;
inline constexpr double LABEL_FONT_SCALE = 0.7;
inline constexpr int LABEL_THICKNESS = 2;

inline constexpr double VMAX_PERCENTILE = 99.0;
inline constexpr double VMAX_FLOOR = 1.0;

inline constexpr double ALIGNMENT_OVERLAY_REF_WEIGHT = 0.5;

inline const cv::Scalar PUPIL_CONTOUR_COLOR{0.0, 255.0, 0.0};
inline const cv::Scalar PUPIL_ELLIPSE_COLOR{0.0, 255.0, 255.0};
inline const cv::Scalar PUPIL_CENTER_COLOR{0.0, 255.0, 0.0};
inline const cv::Scalar PUPIL_MASK_COLOR{0.0, 60.0, 0.0};
inline const cv::Scalar GLINT_CONTOUR_COLOR{0.0, 0.0, 255.0};
inline const cv::Scalar GLINT_ELLIPSE_COLOR{0.0, 165.0, 255.0};
inline const cv::Scalar GLINT_CENTER_COLOR{0.0, 0.0, 255.0};
inline const cv::Scalar LIMBUS_CURVE_COLOR{255.0, 0.0, 255.0};
inline const cv::Scalar LIMBUS_CENTER_COLOR{255.0, 0.0, 255.0};

inline constexpr double MASK_ALPHA = 0.3;
inline constexpr int PUPIL_CENTER_RADIUS = 3;
inline constexpr int GLINT_CENTER_RADIUS = 2;
inline constexpr int LIMBUS_CENTER_RADIUS = 3;

} // namespace cheshm::viz::defaults
