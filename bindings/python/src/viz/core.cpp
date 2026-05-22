#include "cheshm/viz/defaults.hpp"
#include "cheshm/viz/render.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <opencv2/core.hpp>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace
{

using U8Array = nb::ndarray<const std::uint8_t, nb::c_contig, nb::device::cpu>;
using I32Array = nb::ndarray<const std::int32_t, nb::c_contig, nb::device::cpu>;

using EllipseTuple = std::tuple<double, double, double, double, double>;
using CenterTuple = std::tuple<double, double>;
using GlintTuple = std::tuple<std::optional<I32Array>, std::optional<EllipseTuple>, std::optional<CenterTuple>>;
using ScalarTuple = std::tuple<double, double, double>;
using StyleTuple = std::tuple<bool, ScalarTuple, int, double>;

cv::Mat as_gray(U8Array img)
{
    return cv::Mat(
        static_cast<int>(img.shape(0)), static_cast<int>(img.shape(1)), CV_8U, const_cast<std::uint8_t*>(img.data()));
}

cv::Mat as_mask(U8Array img)
{
    return cv::Mat(
        static_cast<int>(img.shape(0)), static_cast<int>(img.shape(1)), CV_8U, const_cast<std::uint8_t*>(img.data()));
}

std::vector<cv::Point> contour_from_ndarray(const I32Array& arr)
{
    const std::size_t n = arr.shape(0);
    std::vector<cv::Point> out;
    out.reserve(n);
    const std::int32_t* data = arr.data();
    const std::size_t stride = arr.ndim() == 3 ? 2 : 2;
    for (std::size_t i = 0; i < n; ++i)
        out.emplace_back(data[i * stride], data[i * stride + 1]);
    return out;
}

cv::RotatedRect rect_from_tuple(const EllipseTuple& t)
{
    return cv::RotatedRect(cv::Point2f(static_cast<float>(std::get<0>(t)), static_cast<float>(std::get<1>(t))),
                           cv::Size2f(static_cast<float>(std::get<2>(t)), static_cast<float>(std::get<3>(t))),
                           static_cast<float>(std::get<4>(t)));
}

cv::Scalar scalar_from_tuple(const ScalarTuple& t)
{
    return cv::Scalar(std::get<0>(t), std::get<1>(t), std::get<2>(t));
}

cheshm::viz::ElementStyle style_from_tuple(const std::optional<StyleTuple>& t, cheshm::viz::ElementStyle fallback)
{
    if (!t.has_value())
        return fallback;
    return {std::get<0>(*t), scalar_from_tuple(std::get<1>(*t)), std::get<2>(*t), std::get<3>(*t)};
}

double save_diff_heatmap(const std::string& out_path, U8Array ref, U8Array aligned, double vmax)
{
    return cheshm::viz::save_diff_heatmap(out_path, as_gray(ref), as_gray(aligned), vmax);
}

void save_alignment_overlay(
    const std::string& out_path, U8Array ref_img, U8Array aligned, double ref_weight, std::optional<std::string> label)
{
    cheshm::viz::save_alignment_overlay(out_path, as_gray(ref_img), as_gray(aligned), ref_weight, label);
}

void save_alignment_comparison(const std::string& out_path,
                               U8Array ref_img,
                               U8Array target_img,
                               U8Array aligned,
                               const std::string& ref_label,
                               const std::string& target_label,
                               double vmax)
{
    cheshm::viz::save_alignment_comparison(
        out_path, as_gray(ref_img), as_gray(target_img), as_gray(aligned), ref_label, target_label, vmax);
}

void save_detection_overlay(const std::string& out_path,
                            U8Array img,
                            std::optional<I32Array> pupil_contour,
                            std::optional<EllipseTuple> pupil_ellipse,
                            std::optional<CenterTuple> pupil_center,
                            std::optional<U8Array> pupil_mask,
                            std::vector<GlintTuple> glints,
                            std::optional<I32Array> limbus_curve,
                            std::optional<CenterTuple> limbus_center,
                            std::optional<StyleTuple> pupil_contour_style,
                            std::optional<StyleTuple> pupil_ellipse_style,
                            std::optional<StyleTuple> pupil_center_style,
                            std::optional<StyleTuple> pupil_mask_style,
                            std::optional<StyleTuple> glint_contour_style,
                            std::optional<StyleTuple> glint_ellipse_style,
                            std::optional<StyleTuple> glint_center_style,
                            std::optional<StyleTuple> limbus_curve_style,
                            std::optional<StyleTuple> limbus_center_style,
                            std::optional<std::string> label)
{
    cheshm::viz::DetectionOverlayInputs inputs;
    if (pupil_contour.has_value())
        inputs.pupil_contour = contour_from_ndarray(*pupil_contour);
    if (pupil_ellipse.has_value())
        inputs.pupil_ellipse = rect_from_tuple(*pupil_ellipse);
    if (pupil_center.has_value())
        inputs.pupil_center = cv::Point(static_cast<int>(std::lrint(std::get<0>(*pupil_center))),
                                        static_cast<int>(std::lrint(std::get<1>(*pupil_center))));
    if (pupil_mask.has_value())
        inputs.pupil_mask = as_mask(*pupil_mask).clone();

    inputs.glints.reserve(glints.size());
    for (auto& [g_contour, g_ellipse, g_center] : glints)
    {
        cheshm::viz::GlintOverlay g;
        if (g_contour.has_value())
            g.contour = contour_from_ndarray(*g_contour);
        if (g_ellipse.has_value())
            g.ellipse = rect_from_tuple(*g_ellipse);
        if (g_center.has_value())
            g.center = cv::Point(static_cast<int>(std::lrint(std::get<0>(*g_center))),
                                 static_cast<int>(std::lrint(std::get<1>(*g_center))));
        inputs.glints.push_back(std::move(g));
    }

    if (limbus_curve.has_value())
        inputs.limbus_curve = contour_from_ndarray(*limbus_curve);
    if (limbus_center.has_value())
        inputs.limbus_center = cv::Point(static_cast<int>(std::lrint(std::get<0>(*limbus_center))),
                                         static_cast<int>(std::lrint(std::get<1>(*limbus_center))));

    cheshm::viz::DetectionOverlayStyle style;
    style.pupil_contour = style_from_tuple(pupil_contour_style, style.pupil_contour);
    style.pupil_ellipse = style_from_tuple(pupil_ellipse_style, style.pupil_ellipse);
    style.pupil_center = style_from_tuple(pupil_center_style, style.pupil_center);
    style.pupil_mask = style_from_tuple(pupil_mask_style, style.pupil_mask);
    style.glint_contour = style_from_tuple(glint_contour_style, style.glint_contour);
    style.glint_ellipse = style_from_tuple(glint_ellipse_style, style.glint_ellipse);
    style.glint_center = style_from_tuple(glint_center_style, style.glint_center);
    style.limbus_curve = style_from_tuple(limbus_curve_style, style.limbus_curve);
    style.limbus_center = style_from_tuple(limbus_center_style, style.limbus_center);

    cheshm::viz::save_detection_overlay(out_path, as_gray(img), inputs, style, label);
}

} // namespace

NB_MODULE(_core, m)
{
    namespace d = cheshm::viz::defaults;

    m.def("save_diff_heatmap", &save_diff_heatmap, "out_path"_a, "ref"_a, "aligned"_a, "vmax"_a);

    m.def("save_alignment_overlay",
          &save_alignment_overlay,
          "out_path"_a,
          "ref_img"_a,
          "aligned"_a,
          "ref_weight"_a,
          "label"_a.none());

    m.def("save_alignment_comparison",
          &save_alignment_comparison,
          "out_path"_a,
          "ref_img"_a,
          "target_img"_a,
          "aligned"_a,
          "ref_label"_a,
          "target_label"_a,
          "vmax"_a);

    m.def("save_detection_overlay",
          &save_detection_overlay,
          "out_path"_a,
          "img"_a,
          "pupil_contour"_a.none(),
          "pupil_ellipse"_a.none(),
          "pupil_center"_a.none(),
          "pupil_mask"_a.none(),
          "glints"_a,
          "limbus_curve"_a.none(),
          "limbus_center"_a.none(),
          "pupil_contour_style"_a.none(),
          "pupil_ellipse_style"_a.none(),
          "pupil_center_style"_a.none(),
          "pupil_mask_style"_a.none(),
          "glint_contour_style"_a.none(),
          "glint_ellipse_style"_a.none(),
          "glint_center_style"_a.none(),
          "limbus_curve_style"_a.none(),
          "limbus_center_style"_a.none(),
          "label"_a.none());

    m.attr("ALIGNMENT_OVERLAY_REF_WEIGHT") = d::ALIGNMENT_OVERLAY_REF_WEIGHT;
}
