// ExCuSe pupil detector — Python binding.

#include "cheshm/roi.hpp"

#include "ExCuSe/excuse.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>

#include <cstdint>
#include <opencv2/core.hpp>

namespace nb = nanobind;
using namespace nb::literals;

namespace
{

// Returns ``(cx, cy, w, h, angle_deg)`` on success, or ``None`` when
// the ROI clamps to zero area. ``cx, cy`` are in full-image coords.
nb::object detect(nb::ndarray<const std::uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> img,
                  int roi_x,
                  int roi_y,
                  int roi_w,
                  int roi_h,
                  int max_ellipse_radi,
                  int good_ellipse_threshold)
{
    const int height = static_cast<int>(img.shape(0));
    const int width = static_cast<int>(img.shape(1));
    const cv::Mat full(height, width, CV_8U, const_cast<std::uint8_t*>(img.data()));

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

    const cv::RotatedRect ellipse = cheshm::ExCuSe::findPupilEllipse(view, max_ellipse_radi, good_ellipse_threshold);

    const double cx = static_cast<double>(ellipse.center.x) + crop.x;
    const double cy = static_cast<double>(ellipse.center.y) + crop.y;
    const double w = ellipse.size.width;
    const double h = ellipse.size.height;
    const double angle = ellipse.angle;

    if (cx == crop.x && cy == crop.y && w == 0.0 && h == 0.0)
    {
        // findPupilEllipse returns a zero RotatedRect when no good
        // ellipse is found; surface that as None to the caller.
        return nb::none();
    }
    return nb::make_tuple(cx, cy, w, h, angle);
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
          "max_ellipse_radi"_a,
          "good_ellipse_threshold"_a);
}
