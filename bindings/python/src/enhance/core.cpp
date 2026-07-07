#include "cheshm/enhance/defaults.hpp"
#include "cheshm/enhance/enhance.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstdint>
#include <opencv2/core.hpp>
#include <utility>

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

nb::object clahe(U8Array img, double clip_limit, int tile)
{
    return nb::cast(mat_to_numpy(cheshm::enhance::clahe(as_gray(img), clip_limit, tile)));
}

nb::object percentile_stretch(U8Array img, double lo_pct, double hi_pct)
{
    return nb::cast(mat_to_numpy(cheshm::enhance::percentile_stretch(as_gray(img), lo_pct, hi_pct)));
}

// Named gamma_correction rather than gamma: libm declares a global ::gamma (an obsolete
// lgamma alias) on some platforms, so a bare &gamma is an ambiguous overloaded name.
nb::object gamma_correction(U8Array img, double g)
{
    return nb::cast(mat_to_numpy(cheshm::enhance::gamma(as_gray(img), g)));
}

nb::object bilateral(U8Array img, int d, double sigma_color, double sigma_space)
{
    return nb::cast(mat_to_numpy(cheshm::enhance::bilateral(as_gray(img), d, sigma_color, sigma_space)));
}

nb::object unsharp(U8Array img, double sigma, double amount)
{
    return nb::cast(mat_to_numpy(cheshm::enhance::unsharp(as_gray(img), sigma, amount)));
}

} // namespace

NB_MODULE(_core, m)
{
    namespace d = cheshm::enhance::defaults;

    m.def("clahe", &clahe, "img"_a, "clip_limit"_a = d::CLAHE_CLIP_LIMIT, "tile"_a = d::CLAHE_TILE);
    m.def("percentile_stretch",
          &percentile_stretch,
          "img"_a,
          "lo_pct"_a = d::STRETCH_LO_PCT,
          "hi_pct"_a = d::STRETCH_HI_PCT);
    m.def("gamma", &gamma_correction, "img"_a, "g"_a = d::GAMMA);
    m.def("bilateral",
          &bilateral,
          "img"_a,
          "d"_a = d::BILATERAL_D,
          "sigma_color"_a = d::BILATERAL_SIGMA_COLOR,
          "sigma_space"_a = d::BILATERAL_SIGMA_SPACE);
    m.def("unsharp", &unsharp, "img"_a, "sigma"_a = d::UNSHARP_SIGMA, "amount"_a = d::UNSHARP_AMOUNT);

    m.attr("CLAHE_CLIP_LIMIT") = d::CLAHE_CLIP_LIMIT;
    m.attr("CLAHE_TILE") = d::CLAHE_TILE;
    m.attr("STRETCH_LO_PCT") = d::STRETCH_LO_PCT;
    m.attr("STRETCH_HI_PCT") = d::STRETCH_HI_PCT;
    m.attr("GAMMA") = d::GAMMA;
    m.attr("BILATERAL_D") = d::BILATERAL_D;
    m.attr("BILATERAL_SIGMA_COLOR") = d::BILATERAL_SIGMA_COLOR;
    m.attr("BILATERAL_SIGMA_SPACE") = d::BILATERAL_SIGMA_SPACE;
    m.attr("UNSHARP_SIGMA") = d::UNSHARP_SIGMA;
    m.attr("UNSHARP_AMOUNT") = d::UNSHARP_AMOUNT;
}
