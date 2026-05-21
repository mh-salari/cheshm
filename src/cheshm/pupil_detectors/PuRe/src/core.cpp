// PuRe pupil detector — Python binding.

#include "cheshm/roi.hpp"

#include "PuRe/defaults.hpp"
#include "PuRe/pure.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>

#include <cstdint>
#include <opencv2/core.hpp>

namespace nb = nanobind;
using namespace nb::literals;

namespace
{

nb::object detect(nb::ndarray<const std::uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> img,
                  int roi_x,
                  int roi_y,
                  int roi_w,
                  int roi_h,
                  float min_pupil_diameter_mm,
                  float max_pupil_diameter_mm,
                  float canthi_distance_mm,
                  int outline_bias)
{
    const int height = static_cast<int>(img.shape(0));
    const int width = static_cast<int>(img.shape(1));
    const cv::Mat full(height, width, CV_8U, const_cast<std::uint8_t*>(img.data()));

    cv::Rect crop(0, 0, width, height);
    if (cheshm::roi_is_active(roi_w, roi_h))
    {
        crop = cheshm::clamp_roi(roi_x, roi_y, roi_w, roi_h, width, height);
        if (crop.area() == 0)
            return nb::none();
    }
    const cv::Mat view = full(crop);

    const auto result =
        cheshm::PuRe::detect(view, min_pupil_diameter_mm, max_pupil_diameter_mm, canthi_distance_mm, outline_bias);
    if (!result)
        return nb::none();

    const double cx = static_cast<double>(result->ellipse.center.x) + crop.x;
    const double cy = static_cast<double>(result->ellipse.center.y) + crop.y;
    const double w = result->ellipse.size.width;
    const double h = result->ellipse.size.height;
    const double angle = result->ellipse.angle;
    const double confidence = result->confidence;

    return nb::make_tuple(cx, cy, w, h, angle, confidence);
}

} // namespace

NB_MODULE(_core, m)
{
    m.def("detect",
          &detect,
          "img"_a,
          "roi_x"_a,
          "roi_y"_a,
          "roi_w"_a,
          "roi_h"_a,
          "min_pupil_diameter_mm"_a,
          "max_pupil_diameter_mm"_a,
          "canthi_distance_mm"_a,
          "outline_bias"_a);

    m.attr("MIN_PUPIL_DIAMETER_MM") = cheshm::PuRe::defaults::MIN_PUPIL_DIAMETER_MM;
    m.attr("MAX_PUPIL_DIAMETER_MM") = cheshm::PuRe::defaults::MAX_PUPIL_DIAMETER_MM;
    m.attr("CANTHI_DISTANCE_MM") = cheshm::PuRe::defaults::CANTHI_DISTANCE_MM;
    m.attr("OUTLINE_BIAS") = cheshm::PuRe::defaults::OUTLINE_BIAS;
}