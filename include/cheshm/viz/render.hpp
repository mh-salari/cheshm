#pragma once

#include "cheshm/viz/defaults.hpp"

#include <opencv2/core.hpp>
#include <optional>
#include <string>
#include <vector>

namespace cheshm::viz
{

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
};

struct DetectionOverlayStyle
{
    cv::Scalar pupil_contour_color = defaults::PUPIL_CONTOUR_COLOR;
    cv::Scalar pupil_ellipse_color = defaults::PUPIL_ELLIPSE_COLOR;
    cv::Scalar pupil_center_color = defaults::PUPIL_CENTER_COLOR;
    cv::Scalar pupil_mask_color = defaults::PUPIL_MASK_COLOR;
    cv::Scalar glint_contour_color = defaults::GLINT_CONTOUR_COLOR;
    cv::Scalar glint_ellipse_color = defaults::GLINT_ELLIPSE_COLOR;
    cv::Scalar glint_center_color = defaults::GLINT_CENTER_COLOR;
    bool show_pupil_contour = defaults::SHOW_PUPIL_CONTOUR;
    bool show_pupil_ellipse = defaults::SHOW_PUPIL_ELLIPSE;
    bool show_pupil_center = defaults::SHOW_PUPIL_CENTER;
    bool show_pupil_mask = defaults::SHOW_PUPIL_MASK;
    bool show_glints = defaults::SHOW_GLINTS;
    double mask_alpha = defaults::MASK_ALPHA;
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
