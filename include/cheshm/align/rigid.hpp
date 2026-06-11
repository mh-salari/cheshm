#pragma once

#include "cheshm/align/defaults.hpp"

#include <opencv2/core.hpp>
#include <optional>
#include <utility>
#include <vector>

namespace cheshm::align
{

enum class Step1Anchor
{
    None,
    Glint,
    Pupil,
};

struct EyeDetection
{
    cv::Point2d pupil_center;
    double pupil_radius;
    std::vector<cv::Point2d> glints;
    cv::Point2d limbus_center;
    double limbus_radius;
};

struct AlignResult
{
    cv::Mat aligned;
    std::optional<cv::Point2d> step1_translation;
    std::optional<cv::Vec3d> step2_transform;
    std::optional<cv::Point2i> rotation_center;
};

cv::Mat make_iris_mask(cv::Size img_size,
                       cv::Point2d limbus_center,
                       double limbus_r,
                       double pupil_r,
                       double exclude_top = defaults::EXCLUDE_TOP,
                       double exclude_bottom = defaults::EXCLUDE_BOTTOM,
                       double inner_margin = defaults::INNER_MARGIN);

cv::Mat make_barrel_mask(cv::Size img_size,
                         cv::Point2d limbus_center,
                         double limbus_r,
                         double pupil_r,
                         double exclude_top = defaults::EXCLUDE_TOP,
                         double exclude_bottom = defaults::EXCLUDE_BOTTOM,
                         double inner_margin = defaults::INNER_MARGIN);

cv::Vec3d align_by_translation(cv::Point2d ref_point, cv::Point2d mov_point);

cv::Mat apply_transform(const cv::Mat& img, cv::Vec3d params, std::optional<cv::Point2d> center = std::nullopt);

std::pair<cv::Vec3d, double> align_by_min_diff(const cv::Mat& img_ref,
                                               const cv::Mat& img_mov,
                                               const cv::Mat& mask,
                                               int dx_lo = defaults::DX_LO,
                                               int dx_hi = defaults::DX_HI,
                                               int dy_lo = defaults::DY_LO,
                                               int dy_hi = defaults::DY_HI,
                                               double rot_start = defaults::ROT_START,
                                               double rot_end = defaults::ROT_END,
                                               double rot_step = defaults::ROT_STEP,
                                               std::optional<cv::Point2d> rotation_center = std::nullopt);

AlignResult align_eye_images(const cv::Mat& ref_img,
                             const cv::Mat& tgt_img,
                             const EyeDetection& ref_det,
                             const EyeDetection& tgt_det,
                             Step1Anchor step1 = Step1Anchor::Glint,
                             bool step2 = true,
                             double exclude_top = defaults::EXCLUDE_TOP,
                             double exclude_bottom = defaults::EXCLUDE_BOTTOM,
                             double inner_margin = defaults::INNER_MARGIN);

} // namespace cheshm::align
