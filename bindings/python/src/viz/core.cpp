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
                            ScalarTuple pupil_contour_color,
                            ScalarTuple pupil_ellipse_color,
                            ScalarTuple pupil_center_color,
                            ScalarTuple pupil_mask_color,
                            ScalarTuple glint_contour_color,
                            ScalarTuple glint_ellipse_color,
                            ScalarTuple glint_center_color,
                            bool show_pupil_contour,
                            bool show_pupil_ellipse,
                            bool show_pupil_center,
                            bool show_pupil_mask,
                            bool show_glints,
                            double mask_alpha,
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

    cheshm::viz::DetectionOverlayStyle style;
    style.pupil_contour_color = scalar_from_tuple(pupil_contour_color);
    style.pupil_ellipse_color = scalar_from_tuple(pupil_ellipse_color);
    style.pupil_center_color = scalar_from_tuple(pupil_center_color);
    style.pupil_mask_color = scalar_from_tuple(pupil_mask_color);
    style.glint_contour_color = scalar_from_tuple(glint_contour_color);
    style.glint_ellipse_color = scalar_from_tuple(glint_ellipse_color);
    style.glint_center_color = scalar_from_tuple(glint_center_color);
    style.show_pupil_contour = show_pupil_contour;
    style.show_pupil_ellipse = show_pupil_ellipse;
    style.show_pupil_center = show_pupil_center;
    style.show_pupil_mask = show_pupil_mask;
    style.show_glints = show_glints;
    style.mask_alpha = mask_alpha;

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
          "pupil_contour_color"_a,
          "pupil_ellipse_color"_a,
          "pupil_center_color"_a,
          "pupil_mask_color"_a,
          "glint_contour_color"_a,
          "glint_ellipse_color"_a,
          "glint_center_color"_a,
          "show_pupil_contour"_a,
          "show_pupil_ellipse"_a,
          "show_pupil_center"_a,
          "show_pupil_mask"_a,
          "show_glints"_a,
          "mask_alpha"_a,
          "label"_a.none());

    m.attr("ALIGNMENT_OVERLAY_REF_WEIGHT") = d::ALIGNMENT_OVERLAY_REF_WEIGHT;
    m.attr("PUPIL_CONTOUR_COLOR") =
        std::make_tuple(d::PUPIL_CONTOUR_COLOR[0], d::PUPIL_CONTOUR_COLOR[1], d::PUPIL_CONTOUR_COLOR[2]);
    m.attr("PUPIL_ELLIPSE_COLOR") =
        std::make_tuple(d::PUPIL_ELLIPSE_COLOR[0], d::PUPIL_ELLIPSE_COLOR[1], d::PUPIL_ELLIPSE_COLOR[2]);
    m.attr("PUPIL_CENTER_COLOR") =
        std::make_tuple(d::PUPIL_CENTER_COLOR[0], d::PUPIL_CENTER_COLOR[1], d::PUPIL_CENTER_COLOR[2]);
    m.attr("PUPIL_MASK_COLOR") =
        std::make_tuple(d::PUPIL_MASK_COLOR[0], d::PUPIL_MASK_COLOR[1], d::PUPIL_MASK_COLOR[2]);
    m.attr("GLINT_CONTOUR_COLOR") =
        std::make_tuple(d::GLINT_CONTOUR_COLOR[0], d::GLINT_CONTOUR_COLOR[1], d::GLINT_CONTOUR_COLOR[2]);
    m.attr("GLINT_ELLIPSE_COLOR") =
        std::make_tuple(d::GLINT_ELLIPSE_COLOR[0], d::GLINT_ELLIPSE_COLOR[1], d::GLINT_ELLIPSE_COLOR[2]);
    m.attr("GLINT_CENTER_COLOR") =
        std::make_tuple(d::GLINT_CENTER_COLOR[0], d::GLINT_CENTER_COLOR[1], d::GLINT_CENTER_COLOR[2]);
    m.attr("SHOW_PUPIL_CONTOUR") = d::SHOW_PUPIL_CONTOUR;
    m.attr("SHOW_PUPIL_ELLIPSE") = d::SHOW_PUPIL_ELLIPSE;
    m.attr("SHOW_PUPIL_CENTER") = d::SHOW_PUPIL_CENTER;
    m.attr("SHOW_PUPIL_MASK") = d::SHOW_PUPIL_MASK;
    m.attr("SHOW_GLINTS") = d::SHOW_GLINTS;
    m.attr("MASK_ALPHA") = d::MASK_ALPHA;
}
