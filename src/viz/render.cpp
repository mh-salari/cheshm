#include "cheshm/viz/render.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace cheshm::viz
{

namespace
{

cv::Mat to_bgr(const cv::Mat& img)
{
    if (img.channels() == 3)
        return img.clone();
    cv::Mat bgr;
    cv::cvtColor(img, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
}

double percentile_99(const cv::Mat& diff)
{
    const std::size_t n = static_cast<std::size_t>(diff.total());
    std::vector<double> values;
    values.reserve(n);
    for (auto it = diff.begin<float>(), end = diff.end<float>(); it != end; ++it)
        values.push_back(static_cast<double>(*it));
    std::sort(values.begin(), values.end());
    if (values.empty())
        return 0.0;
    const double rank = (defaults::VMAX_PERCENTILE / 100.0) * static_cast<double>(values.size() - 1);
    const std::size_t i = static_cast<std::size_t>(std::floor(rank));
    const double frac = rank - static_cast<double>(i);
    if (i + 1 >= values.size())
        return values[i];
    return values[i] + frac * (values[i + 1] - values[i]);
}

double resolve_vmax(double requested, const cv::Mat& diff)
{
    if (requested > 0.0)
        return requested;
    return std::max(percentile_99(diff), defaults::VMAX_FLOOR);
}

cv::Mat absdiff_f32(const cv::Mat& a, const cv::Mat& b)
{
    cv::Mat a_f;
    cv::Mat b_f;
    a.convertTo(a_f, CV_32F);
    b.convertTo(b_f, CV_32F);
    cv::Mat diff;
    cv::absdiff(a_f, b_f, diff);
    return diff;
}

void write_or_throw(const std::string& path, const cv::Mat& img)
{
    if (!cv::imwrite(path, img))
        throw std::runtime_error("failed to write " + path);
}

} // namespace

cv::Mat add_label(const cv::Mat& img, const std::string& text, int height)
{
    cv::Mat label_bar(height, img.cols, CV_8UC3, defaults::LABEL_BG);
    int baseline = 0;
    const cv::Size text_size =
        cv::getTextSize(text, defaults::LABEL_FONT, defaults::LABEL_FONT_SCALE, defaults::LABEL_THICKNESS, &baseline);
    const int x = std::max((img.cols - text_size.width) / 2, 0);
    const int y = (height + text_size.height) / 2;
    cv::putText(label_bar,
                text,
                cv::Point(x, y),
                defaults::LABEL_FONT,
                defaults::LABEL_FONT_SCALE,
                defaults::LABEL_FG,
                defaults::LABEL_THICKNESS);
    cv::Mat result;
    cv::vconcat(label_bar, img, result);
    return result;
}

cv::Mat diff_hot(const cv::Mat& a, const cv::Mat& b, double vmax)
{
    const cv::Mat diff = absdiff_f32(a, b);
    cv::Mat scaled = diff * (255.0 / vmax);
    cv::Mat u8;
    scaled.convertTo(u8, CV_8U);
    cv::Mat hot;
    cv::applyColorMap(u8, hot, cv::COLORMAP_HOT);
    return hot;
}

double save_diff_heatmap(const std::string& out_path, const cv::Mat& ref, const cv::Mat& aligned, double vmax)
{
    if (ref.size() != aligned.size())
        throw std::runtime_error("ref / aligned size mismatch in save_diff_heatmap");
    const cv::Mat diff = absdiff_f32(ref, aligned);
    const double resolved_vmax = resolve_vmax(vmax, diff);
    write_or_throw(out_path, diff_hot(ref, aligned, resolved_vmax));
    return resolved_vmax;
}

void save_alignment_overlay(const std::string& out_path,
                            const cv::Mat& ref_img,
                            const cv::Mat& aligned,
                            double ref_weight,
                            std::optional<std::string> label)
{
    if (ref_img.size() != aligned.size())
        throw std::runtime_error("ref / aligned size mismatch in save_alignment_overlay");
    cv::Mat blend;
    cv::addWeighted(to_bgr(ref_img), ref_weight, to_bgr(aligned), 1.0 - ref_weight, 0.0, blend);
    if (label.has_value())
        blend = add_label(blend, *label);
    write_or_throw(out_path, blend);
}

void save_alignment_comparison(const std::string& out_path,
                               const cv::Mat& ref_img,
                               const cv::Mat& target_img,
                               const cv::Mat& aligned,
                               const std::string& ref_label,
                               const std::string& target_label,
                               double vmax)
{
    if (ref_img.size() != target_img.size() || ref_img.size() != aligned.size())
        throw std::runtime_error("ref / target / aligned size mismatch in save_alignment_comparison");
    const cv::Mat diff_before = absdiff_f32(ref_img, target_img);
    const cv::Mat diff_after = absdiff_f32(ref_img, aligned);
    double resolved_vmax = vmax;
    if (resolved_vmax <= 0.0)
    {
        cv::Mat combined;
        cv::max(diff_before, diff_after, combined);
        resolved_vmax = std::max(percentile_99(combined), defaults::VMAX_FLOOR);
    }
    std::vector<cv::Mat> panels;
    panels.reserve(4);
    panels.push_back(add_label(to_bgr(ref_img), ref_label));
    panels.push_back(add_label(to_bgr(aligned), target_label));
    panels.push_back(add_label(diff_hot(ref_img, target_img, resolved_vmax), "diff (before)"));
    panels.push_back(add_label(diff_hot(ref_img, aligned, resolved_vmax), "diff (after)"));
    cv::Mat composite;
    cv::hconcat(panels, composite);
    write_or_throw(out_path, composite);
}

void save_detection_overlay(const std::string& out_path,
                            const cv::Mat& img,
                            const DetectionOverlayInputs& inputs,
                            const DetectionOverlayStyle& style,
                            std::optional<std::string> label)
{
    cv::Mat canvas = to_bgr(img);

    if (style.show_pupil_mask && inputs.pupil_mask.has_value() && inputs.pupil_mask->dims == 2)
    {
        cv::Mat tint(canvas.size(), canvas.type(), cv::Scalar::all(0.0));
        tint.setTo(style.pupil_mask_color, *inputs.pupil_mask);
        cv::Mat blended;
        cv::addWeighted(canvas, 1.0, tint, style.mask_alpha, 0.0, blended);
        canvas = blended;
    }

    if (style.show_pupil_contour && inputs.pupil_contour.has_value())
    {
        std::vector<std::vector<cv::Point>> contours{*inputs.pupil_contour};
        cv::drawContours(canvas, contours, -1, style.pupil_contour_color, defaults::OUTLINE_THICKNESS);
    }

    if (style.show_pupil_ellipse && inputs.pupil_ellipse.has_value())
    {
        const auto& rr = *inputs.pupil_ellipse;
        const cv::Point center{static_cast<int>(std::lrint(rr.center.x)), static_cast<int>(std::lrint(rr.center.y))};
        const cv::Size axes{std::max(static_cast<int>(std::lrint(rr.size.width / 2.0)), 1),
                            std::max(static_cast<int>(std::lrint(rr.size.height / 2.0)), 1)};
        cv::ellipse(canvas, center, axes, rr.angle, 0, 360, style.pupil_ellipse_color, defaults::OUTLINE_THICKNESS);
    }

    if (style.show_pupil_center && inputs.pupil_center.has_value())
    {
        cv::circle(canvas, *inputs.pupil_center, defaults::PUPIL_CENTER_RADIUS, style.pupil_center_color, -1);
    }

    if (style.show_glints)
    {
        for (const auto& glint : inputs.glints)
        {
            if (glint.contour.has_value())
            {
                std::vector<std::vector<cv::Point>> contours{*glint.contour};
                cv::drawContours(canvas, contours, -1, style.glint_contour_color, defaults::OUTLINE_THICKNESS);
            }
            if (glint.ellipse.has_value())
            {
                const auto& rr = *glint.ellipse;
                const cv::Point center{static_cast<int>(std::lrint(rr.center.x)),
                                       static_cast<int>(std::lrint(rr.center.y))};
                const cv::Size axes{std::max(static_cast<int>(std::lrint(rr.size.width / 2.0)), 1),
                                    std::max(static_cast<int>(std::lrint(rr.size.height / 2.0)), 1)};
                cv::ellipse(
                    canvas, center, axes, rr.angle, 0, 360, style.glint_ellipse_color, defaults::OUTLINE_THICKNESS);
            }
            if (glint.center.has_value())
            {
                cv::circle(canvas, *glint.center, defaults::GLINT_CENTER_RADIUS, style.glint_center_color, -1);
            }
        }
    }

    if (label.has_value())
        canvas = add_label(canvas, *label);

    write_or_throw(out_path, canvas);
}

} // namespace cheshm::viz
