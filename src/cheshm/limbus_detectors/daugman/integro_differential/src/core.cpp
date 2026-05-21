// Daugman integro-differential operator — Python binding.

#include "integro_differential/defaults.hpp"
#include "integro_differential/operator.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>

#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace nb = nanobind;
using namespace nb::literals;

namespace
{

nb::object detect_limbus(nb::ndarray<const std::uint8_t, nb::c_contig, nb::device::cpu> img,
                         double seed_x,
                         double seed_y,
                         int r_min,
                         int r_max,
                         int range_,
                         int step)
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

    const auto result =
        cheshm::Daugman::integro_differential::detect_limbus(full, {seed_x, seed_y}, r_min, r_max, range_, step);
    if (!result)
        return nb::none();
    return nb::make_tuple(result->center.x, result->center.y, result->radius);
}

} // namespace

NB_MODULE(_core, m)
{
    namespace d = cheshm::Daugman::integro_differential::defaults;

    m.def("detect_limbus",
          &detect_limbus,
          "img"_a,
          "seed_x"_a,
          "seed_y"_a,
          "r_min"_a,
          "r_max"_a,
          "range_"_a,
          "step"_a);

    m.attr("R_MIN") = d::R_MIN;
    m.attr("R_MAX") = d::R_MAX;
    m.attr("RANGE") = d::RANGE;
    m.attr("STEP") = d::STEP;
}
