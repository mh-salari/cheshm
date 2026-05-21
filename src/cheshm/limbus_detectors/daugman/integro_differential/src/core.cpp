// Daugman integro-differential operator — Python binding.

#include "integro_differential/defaults.hpp"
#include "integro_differential/operator.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>

#include <cstdint>
#include <opencv2/core.hpp>

namespace nb = nanobind;
using namespace nb::literals;

namespace
{

nb::object detect_limbus(nb::ndarray<const std::uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> img,
                         double seed_x,
                         double seed_y,
                         int r_min,
                         int r_max,
                         int range_,
                         int step)
{
    const int height = static_cast<int>(img.shape(0));
    const int width = static_cast<int>(img.shape(1));
    const cv::Mat full(height, width, CV_8U, const_cast<std::uint8_t*>(img.data()));

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
