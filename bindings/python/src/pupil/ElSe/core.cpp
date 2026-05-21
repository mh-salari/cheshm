// ElSe pupil detector — nanobind binding.
//
// Marshals a numpy uint8 image into cv::Mat, applies the optional
// ROI crop, calls cheshm::ElSe::detect, and packs the result into a
// method-tagged tuple for the Python wrapper:
//
//   primary ellipse path  -> ("ellipse", cx, cy, w, h, angle_deg)
//   blob-fallback path    -> ("blob_fallback", cx, cy)
//   nothing found / empty -> None
//
// Returned coordinates are in full-image (not crop-local) space.

#include "cheshm/helpers/image/roi.hpp"
#include "cheshm/pupil/ElSe/defaults.hpp"
#include "cheshm/pupil/ElSe/else.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

namespace nb = nanobind;
using namespace nb::literals;

namespace
{

nb::object detect(nb::ndarray<const std::uint8_t, nb::c_contig, nb::device::cpu> img,
                  int roi_x,
                  int roi_y,
                  int roi_w,
                  int roi_h,
                  float min_area_ratio,
                  float max_area_ratio)
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

    cv::Rect crop(0, 0, width, height);
    if (cheshm::roi_is_active(roi_w, roi_h))
    {
        crop = cheshm::clamp_roi(roi_x, roi_y, roi_w, roi_h, width, height);
        if (crop.area() == 0)
        {
            return nb::none();
        }
    }
    const cv::Mat view = full(crop);

    const auto result = cheshm::ElSe::detect(view, min_area_ratio, max_area_ratio);
    if (!result)
    {
        return nb::none();
    }

    const double cx = static_cast<double>(result->center.x) + crop.x;
    const double cy = static_cast<double>(result->center.y) + crop.y;

    if (result->method == cheshm::ElSe::DetectionMethod::Ellipse)
    {
        const cv::RotatedRect& e = *result->ellipse;
        const double w = e.size.width;
        const double h = e.size.height;
        const double angle = e.angle;
        return nb::make_tuple(std::string{"ellipse"}, cx, cy, w, h, angle);
    }

    return nb::make_tuple(std::string{"blob_fallback"}, cx, cy);
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
          "min_area_ratio"_a,
          "max_area_ratio"_a);

    m.attr("MIN_AREA_RATIO") = cheshm::ElSe::defaults::MIN_AREA_RATIO;
    m.attr("MAX_AREA_RATIO") = cheshm::ElSe::defaults::MAX_AREA_RATIO;
}
