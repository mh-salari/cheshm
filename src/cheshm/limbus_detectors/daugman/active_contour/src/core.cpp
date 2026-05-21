// Daugman 2007 active contour — Python binding.

#include "active_contour/contour.hpp"
#include "active_contour/defaults.hpp"
#include "daugman/nb_to_numpy.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>

#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace
{

nb::object detect_limbus(nb::ndarray<const std::uint8_t, nb::c_contig, nb::device::cpu> img,
                         double seed_x,
                         double seed_y,
                         int n_angles,
                         int m_harmonics,
                         double gradient_sigma,
                         double radial_smoothing,
                         bool skip_eyelid_wedges,
                         double r_min,
                         double r_max)
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

    auto result = cheshm::Daugman::active_contour::detect_limbus(full,
                                                                 {seed_x, seed_y},
                                                                 n_angles,
                                                                 m_harmonics,
                                                                 gradient_sigma,
                                                                 radial_smoothing,
                                                                 skip_eyelid_wedges,
                                                                 r_min,
                                                                 r_max);
    if (!result)
        return nb::none();

    return nb::make_tuple(
        result->center.x, result->center.y, cheshm::Daugman::nb_to_numpy(std::move(result->thetas)), cheshm::Daugman::nb_to_numpy(std::move(result->R_theta)));
}

} // namespace

NB_MODULE(_core, m)
{
    namespace d = cheshm::Daugman::active_contour::defaults;

    m.def("detect_limbus",
          &detect_limbus,
          "img"_a,
          "seed_x"_a,
          "seed_y"_a,
          "n_angles"_a,
          "m_harmonics"_a,
          "gradient_sigma"_a,
          "radial_smoothing"_a,
          "skip_eyelid_wedges"_a,
          "r_min"_a,
          "r_max"_a);

    m.attr("N") = d::N;
    m.attr("M") = d::M;
    m.attr("GRADIENT_SIGMA") = d::GRADIENT_SIGMA;
    m.attr("RADIAL_SMOOTHING") = d::RADIAL_SMOOTHING;
    m.attr("SKIP_EYELID_WEDGES") = d::SKIP_EYELID_WEDGES;
    m.attr("R_MIN") = d::R_MIN;
    m.attr("R_MAX") = d::R_MAX;
}
