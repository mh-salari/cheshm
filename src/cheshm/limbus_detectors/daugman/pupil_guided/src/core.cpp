// Pupil-shape-guided active contour — Python binding.

#include "pupil_guided/contour.hpp"
#include "pupil_guided/defaults.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>

#include <cstdint>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace
{

nb::ndarray<nb::numpy, double, nb::ndim<1>> _to_numpy(std::vector<double> data)
{
    auto owner = std::make_unique<std::vector<double>>(std::move(data));
    double* ptr = owner->data();
    const std::size_t shape[1] = {owner->size()};
    nb::capsule cap(owner.release(), [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
    return nb::ndarray<nb::numpy, double, nb::ndim<1>>(ptr, 1, shape, cap);
}

nb::object detect_limbus(nb::ndarray<const std::uint8_t, nb::c_contig, nb::device::cpu> img,
                         double seed_x,
                         double seed_y,
                         double pupil_cx,
                         double pupil_cy,
                         double pupil_w,
                         double pupil_h,
                         double pupil_angle_deg,
                         int n_angles,
                         int m_harmonics,
                         double gradient_sigma,
                         double radial_smoothing,
                         double k_min,
                         double k_max)
{
    const int height = static_cast<int>(img.shape(0));
    const int width = static_cast<int>(img.shape(1));
    const bool is_color = (img.ndim() == 3 && img.shape(2) == 3);
    const cv::Mat raw(height, width, is_color ? CV_8UC3 : CV_8U, const_cast<std::uint8_t*>(img.data()));
    cv::Mat full;
    if (is_color)
        cv::cvtColor(raw, full, cv::COLOR_BGR2GRAY);
    else
        full = raw;

    const cheshm::Daugman::pupil_guided::PupilEllipse pupil{{pupil_cx, pupil_cy}, {pupil_w, pupil_h}, pupil_angle_deg};

    auto result = cheshm::Daugman::pupil_guided::detect_limbus(full,
                                                               {seed_x, seed_y},
                                                               pupil,
                                                               n_angles,
                                                               m_harmonics,
                                                               gradient_sigma,
                                                               radial_smoothing,
                                                               k_min,
                                                               k_max);
    if (!result)
        return nb::none();

    return nb::make_tuple(
        result->center.x, result->center.y, _to_numpy(std::move(result->thetas)), _to_numpy(std::move(result->R_theta)));
}

} // namespace

NB_MODULE(_core, m)
{
    namespace d = cheshm::Daugman::pupil_guided::defaults;

    m.def("detect_limbus",
          &detect_limbus,
          "img"_a,
          "seed_x"_a,
          "seed_y"_a,
          "pupil_cx"_a,
          "pupil_cy"_a,
          "pupil_w"_a,
          "pupil_h"_a,
          "pupil_angle_deg"_a,
          "n_angles"_a,
          "m_harmonics"_a,
          "gradient_sigma"_a,
          "radial_smoothing"_a,
          "k_min"_a,
          "k_max"_a);

    m.attr("N") = d::N;
    m.attr("M") = d::M;
    m.attr("GRADIENT_SIGMA") = d::GRADIENT_SIGMA;
    m.attr("RADIAL_SMOOTHING") = d::RADIAL_SMOOTHING;
    m.attr("K_MIN") = d::K_MIN;
    m.attr("K_MAX") = d::K_MAX;
}
