#pragma once

#include "cheshm/viz/defaults.hpp"

#include <opencv2/core.hpp>
#include <optional>
#include <string>
#include <vector>

namespace cheshm::viz
{

struct ElementStyle
{
    bool show;
    cv::Scalar color;
    int thickness;
    double alpha;
};

struct GlintOverlay
{
    std::optional<std::vector<cv::Point>> contour;
    std::optional<cv::RotatedRect> ellipse;
    std::optional<cv::Point> center;
};

struct DetectionOverlayInputs
{
    std::optional<std::vector<cv::Point>> pupil_contour;
    std::optional<cv::RotatedRect> pupil_ellipse;
    std::optional<cv::Point> pupil_center;
    std::optional<cv::Mat> pupil_mask;
    std::vector<GlintOverlay> glints;
    std::optional<std::vector<cv::Point>> limbus_curve;
    std::optional<cv::Point> limbus_center;
};

struct DetectionOverlayStyle
{
    ElementStyle pupil_contour{true, defaults::PUPIL_CONTOUR_COLOR, 1, 1.0};
    ElementStyle pupil_ellipse{true, defaults::PUPIL_ELLIPSE_COLOR, 1, 1.0};
    ElementStyle pupil_center{true, defaults::PUPIL_CENTER_COLOR, defaults::PUPIL_CENTER_RADIUS, 1.0};
    ElementStyle pupil_mask{false, defaults::PUPIL_MASK_COLOR, 0, defaults::MASK_ALPHA};
    ElementStyle glint_contour{true, defaults::GLINT_CONTOUR_COLOR, 1, 1.0};
    ElementStyle glint_ellipse{true, defaults::GLINT_ELLIPSE_COLOR, 1, 1.0};
    ElementStyle glint_center{true, defaults::GLINT_CENTER_COLOR, defaults::GLINT_CENTER_RADIUS, 1.0};
    ElementStyle limbus_curve{true, defaults::LIMBUS_CURVE_COLOR, 1, 1.0};
    ElementStyle limbus_center{true, defaults::LIMBUS_CENTER_COLOR, defaults::LIMBUS_CENTER_RADIUS, 1.0};
};

cv::Mat add_label(const cv::Mat& img, const std::string& text, int height = defaults::LABEL_HEIGHT);

cv::Mat diff_hot(const cv::Mat& a, const cv::Mat& b, double vmax);

double save_diff_heatmap(const std::string& out_path, const cv::Mat& ref, const cv::Mat& aligned, double vmax = -1.0);

void save_alignment_overlay(const std::string& out_path,
                            const cv::Mat& ref_img,
                            const cv::Mat& aligned,
                            double ref_weight = defaults::ALIGNMENT_OVERLAY_REF_WEIGHT,
                            std::optional<std::string> label = std::nullopt);

void save_alignment_comparison(const std::string& out_path,
                               const cv::Mat& ref_img,
                               const cv::Mat& target_img,
                               const cv::Mat& aligned,
                               const std::string& ref_label = "reference",
                               const std::string& target_label = "aligned",
                               double vmax = -1.0);

void save_detection_overlay(const std::string& out_path,
                            const cv::Mat& img,
                            const DetectionOverlayInputs& inputs,
                            const DetectionOverlayStyle& style = {},
                            std::optional<std::string> label = std::nullopt);

} // namespace cheshm::viz
