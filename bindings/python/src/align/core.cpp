#include "cheshm/align/defaults.hpp"
#include "cheshm/align/rigid.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <opencv2/core.hpp>
#include <optional>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace
{

using U8Array = nb::ndarray<const std::uint8_t, nb::c_contig, nb::device::cpu>;

cv::Mat as_gray(U8Array img)
{
    const int height = static_cast<int>(img.shape(0));
    const int width = static_cast<int>(img.shape(1));
    return cv::Mat(height, width, CV_8U, const_cast<std::uint8_t*>(img.data()));
}

nb::ndarray<nb::numpy, std::uint8_t, nb::ndim<2>> mat_to_numpy(cv::Mat mat)
{
    auto* owner = new cv::Mat(std::move(mat));
    const std::size_t shape[2] = {static_cast<std::size_t>(owner->rows), static_cast<std::size_t>(owner->cols)};
    nb::capsule cap(owner, [](void* p) noexcept { delete static_cast<cv::Mat*>(p); });
    return nb::ndarray<nb::numpy, std::uint8_t, nb::ndim<2>>(owner->data, 2, shape, cap);
}

nb::object make_iris_mask(int height,
                          int width,
                          double lcx,
                          double lcy,
                          double limbus_r,
                          double pupil_r,
                          double exclude_top,
                          double exclude_bottom,
                          double inner_margin)
{
    cv::Mat mask = cheshm::align::make_iris_mask(
        {width, height}, {lcx, lcy}, limbus_r, pupil_r, exclude_top, exclude_bottom, inner_margin);
    return nb::cast(mat_to_numpy(std::move(mask)));
}

nb::object make_barrel_mask(int height,
                            int width,
                            double lcx,
                            double lcy,
                            double limbus_r,
                            double pupil_r,
                            double exclude_top,
                            double exclude_bottom,
                            double inner_margin)
{
    cv::Mat mask = cheshm::align::make_barrel_mask(
        {width, height}, {lcx, lcy}, limbus_r, pupil_r, exclude_top, exclude_bottom, inner_margin);
    return nb::cast(mat_to_numpy(std::move(mask)));
}

nb::object align_by_translation(double ref_x, double ref_y, double mov_x, double mov_y)
{
    const auto v = cheshm::align::align_by_translation({ref_x, ref_y}, {mov_x, mov_y});
    return nb::make_tuple(v[0], v[1], v[2]);
}

nb::object apply_transform(
    U8Array img, double dx, double dy, double theta, std::optional<double> center_x, std::optional<double> center_y)
{
    const cv::Mat src = as_gray(img);
    std::optional<cv::Point2d> center;
    if (center_x.has_value() && center_y.has_value())
        center = cv::Point2d{*center_x, *center_y};
    cv::Mat warped = cheshm::align::apply_transform(src, {dx, dy, theta}, center);
    return nb::cast(mat_to_numpy(std::move(warped)));
}

nb::object align_by_min_diff(U8Array img_ref,
                             U8Array img_mov,
                             U8Array mask,
                             int dx_lo,
                             int dx_hi,
                             int dy_lo,
                             int dy_hi,
                             double rot_start,
                             double rot_end,
                             double rot_step,
                             std::optional<double> center_x,
                             std::optional<double> center_y)
{
    const cv::Mat ref = as_gray(img_ref);
    const cv::Mat mov = as_gray(img_mov);
    const cv::Mat msk = as_gray(mask);
    std::optional<cv::Point2d> center;
    if (center_x.has_value() && center_y.has_value())
        center = cv::Point2d{*center_x, *center_y};
    const auto [params, score] = cheshm::align::align_by_min_diff(
        ref, mov, msk, dx_lo, dx_hi, dy_lo, dy_hi, rot_start, rot_end, rot_step, center);
    return nb::make_tuple(nb::make_tuple(params[0], params[1], params[2]), score);
}

nb::object align_eye_images(U8Array ref_img,
                            U8Array tgt_img,
                            double ref_pupil_cx,
                            double ref_pupil_cy,
                            double ref_pupil_radius,
                            std::vector<std::pair<double, double>> ref_glints,
                            double ref_limbus_cx,
                            double ref_limbus_cy,
                            double ref_limbus_radius,
                            double tgt_pupil_cx,
                            double tgt_pupil_cy,
                            double tgt_pupil_radius,
                            std::vector<std::pair<double, double>> tgt_glints,
                            double tgt_limbus_cx,
                            double tgt_limbus_cy,
                            double tgt_limbus_radius,
                            int step1_code,
                            bool step2,
                            double exclude_top,
                            double exclude_bottom,
                            double inner_margin)
{
    const cv::Mat ref = as_gray(ref_img);
    const cv::Mat tgt = as_gray(tgt_img);

    cheshm::align::EyeDetection ref_det;
    ref_det.pupil_center = {ref_pupil_cx, ref_pupil_cy};
    ref_det.pupil_radius = ref_pupil_radius;
    ref_det.glints.reserve(ref_glints.size());
    for (auto& [x, y] : ref_glints)
        ref_det.glints.emplace_back(x, y);
    ref_det.limbus_center = {ref_limbus_cx, ref_limbus_cy};
    ref_det.limbus_radius = ref_limbus_radius;

    cheshm::align::EyeDetection tgt_det;
    tgt_det.pupil_center = {tgt_pupil_cx, tgt_pupil_cy};
    tgt_det.pupil_radius = tgt_pupil_radius;
    tgt_det.glints.reserve(tgt_glints.size());
    for (auto& [x, y] : tgt_glints)
        tgt_det.glints.emplace_back(x, y);
    tgt_det.limbus_center = {tgt_limbus_cx, tgt_limbus_cy};
    tgt_det.limbus_radius = tgt_limbus_radius;

    cheshm::align::Step1Anchor step1{cheshm::align::Step1Anchor::None};
    if (step1_code == 1)
        step1 = cheshm::align::Step1Anchor::Glint;
    else if (step1_code == 2)
        step1 = cheshm::align::Step1Anchor::Pupil;

    auto result = cheshm::align::align_eye_images(
        ref, tgt, ref_det, tgt_det, step1, step2, exclude_top, exclude_bottom, inner_margin);

    nb::object step1_obj = nb::none();
    if (result.step1_translation)
        step1_obj = nb::make_tuple(result.step1_translation->x, result.step1_translation->y);

    nb::object step2_obj = nb::none();
    if (result.step2_transform)
    {
        const auto& v = *result.step2_transform;
        step2_obj = nb::make_tuple(v[0], v[1], v[2]);
    }

    nb::object center_obj = nb::none();
    if (result.rotation_center)
        center_obj = nb::make_tuple(result.rotation_center->x, result.rotation_center->y);

    return nb::make_tuple(mat_to_numpy(std::move(result.aligned)), step1_obj, step2_obj, center_obj);
}

} // namespace

NB_MODULE(_core, m)
{
    namespace d = cheshm::align::defaults;

    m.def("make_iris_mask",
          &make_iris_mask,
          "height"_a,
          "width"_a,
          "lcx"_a,
          "lcy"_a,
          "limbus_r"_a,
          "pupil_r"_a,
          "exclude_top"_a,
          "exclude_bottom"_a,
          "inner_margin"_a);

    m.def("make_barrel_mask",
          &make_barrel_mask,
          "height"_a,
          "width"_a,
          "lcx"_a,
          "lcy"_a,
          "limbus_r"_a,
          "pupil_r"_a,
          "exclude_top"_a,
          "exclude_bottom"_a,
          "inner_margin"_a);

    m.def("align_by_translation", &align_by_translation, "ref_x"_a, "ref_y"_a, "mov_x"_a, "mov_y"_a);

    m.def("apply_transform",
          &apply_transform,
          "img"_a,
          "dx"_a,
          "dy"_a,
          "theta"_a,
          "center_x"_a.none(),
          "center_y"_a.none());

    m.def("align_by_min_diff",
          &align_by_min_diff,
          "img_ref"_a,
          "img_mov"_a,
          "mask"_a,
          "dx_lo"_a,
          "dx_hi"_a,
          "dy_lo"_a,
          "dy_hi"_a,
          "rot_start"_a,
          "rot_end"_a,
          "rot_step"_a,
          "center_x"_a.none(),
          "center_y"_a.none());

    m.def("align_eye_images",
          &align_eye_images,
          "ref_img"_a,
          "tgt_img"_a,
          "ref_pupil_cx"_a,
          "ref_pupil_cy"_a,
          "ref_pupil_radius"_a,
          "ref_glints"_a,
          "ref_limbus_cx"_a,
          "ref_limbus_cy"_a,
          "ref_limbus_radius"_a,
          "tgt_pupil_cx"_a,
          "tgt_pupil_cy"_a,
          "tgt_pupil_radius"_a,
          "tgt_glints"_a,
          "tgt_limbus_cx"_a,
          "tgt_limbus_cy"_a,
          "tgt_limbus_radius"_a,
          "step1_code"_a,
          "step2"_a,
          "exclude_top"_a = d::EXCLUDE_TOP,
          "exclude_bottom"_a = d::EXCLUDE_BOTTOM,
          "inner_margin"_a = d::INNER_MARGIN);

    m.attr("DX_LO") = d::DX_LO;
    m.attr("DX_HI") = d::DX_HI;
    m.attr("DY_LO") = d::DY_LO;
    m.attr("DY_HI") = d::DY_HI;
    m.attr("ROT_START") = d::ROT_START;
    m.attr("ROT_END") = d::ROT_END;
    m.attr("ROT_STEP") = d::ROT_STEP;
    m.attr("EXCLUDE_TOP") = d::EXCLUDE_TOP;
    m.attr("EXCLUDE_BOTTOM") = d::EXCLUDE_BOTTOM;
    m.attr("INNER_MARGIN") = d::INNER_MARGIN;
}
