// PupilLabs2D pupil detector — Python binding.

#include "cheshm/roi.hpp"

#include "PupilLabs2D/defaults.hpp"
#include "PupilLabs2D/pupil_labs_2d.hpp"

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
                  int intensity_range,
                  int blur_size,
                  float canny_threshold,
                  float canny_ratio,
                  int canny_aperture,
                  int pupil_size_max,
                  int pupil_size_min,
                  int contour_size_min,
                  float ellipse_roundness_ratio,
                  float initial_ellipse_fit_threshold,
                  float ellipse_true_support_min_dist,
                  float support_pixel_ratio_exponent,
                  bool coarse_detection,
                  int coarse_filter_min,
                  int coarse_filter_max)
{
    const int height = static_cast<int>(img.shape(0));
    const int width = static_cast<int>(img.shape(1));
    const cv::Mat full(height, width, CV_8U, const_cast<std::uint8_t*>(img.data()));

    cv::Rect roi{0, 0, width, height};
    if (cheshm::roi_is_active(roi_w, roi_h))
    {
        roi = cheshm::clamp_roi(roi_x, roi_y, roi_w, roi_h, width, height);
        if (roi.area() == 0)
            return nb::none();
    }

    auto props = cheshm::PupilLabs2D::default_properties();
    props.intensity_range = intensity_range;
    props.blur_size = blur_size;
    props.canny_threshold = canny_threshold;
    props.canny_ratio = canny_ratio;
    props.canny_aperture = canny_aperture;
    props.pupil_size_max = pupil_size_max;
    props.pupil_size_min = pupil_size_min;
    props.contour_size_min = contour_size_min;
    props.ellipse_roundness_ratio = ellipse_roundness_ratio;
    props.initial_ellipse_fit_threshold = initial_ellipse_fit_threshold;
    props.ellipse_true_support_min_dist = ellipse_true_support_min_dist;
    props.support_pixel_ratio_exponent = support_pixel_ratio_exponent;
    props.coarse_detection = coarse_detection;
    props.coarse_filter_min = coarse_filter_min;
    props.coarse_filter_max = coarse_filter_max;

    const auto result = cheshm::PupilLabs2D::detect(full, roi, props);
    if (!result)
        return nb::none();

    const double cx = result->ellipse.center.x;
    const double cy = result->ellipse.center.y;
    const double w = result->ellipse.size.width;
    const double h = result->ellipse.size.height;
    const double angle = result->ellipse.angle;
    const double confidence = result->confidence;
    return nb::make_tuple(cx, cy, w, h, angle, confidence);
}

} // namespace

NB_MODULE(_core, m)
{
    namespace d = cheshm::PupilLabs2D::defaults;

    m.def("detect",
          &detect,
          "img"_a,
          "roi_x"_a,
          "roi_y"_a,
          "roi_w"_a,
          "roi_h"_a,
          "intensity_range"_a,
          "blur_size"_a,
          "canny_threshold"_a,
          "canny_ratio"_a,
          "canny_aperture"_a,
          "pupil_size_max"_a,
          "pupil_size_min"_a,
          "contour_size_min"_a,
          "ellipse_roundness_ratio"_a,
          "initial_ellipse_fit_threshold"_a,
          "ellipse_true_support_min_dist"_a,
          "support_pixel_ratio_exponent"_a,
          "coarse_detection"_a,
          "coarse_filter_min"_a,
          "coarse_filter_max"_a);

    m.attr("INTENSITY_RANGE") = d::INTENSITY_RANGE;
    m.attr("BLUR_SIZE") = d::BLUR_SIZE;
    m.attr("CANNY_THRESHOLD") = d::CANNY_THRESHOLD;
    m.attr("CANNY_RATIO") = d::CANNY_RATIO;
    m.attr("CANNY_APERTURE") = d::CANNY_APERTURE;
    m.attr("PUPIL_SIZE_MAX") = d::PUPIL_SIZE_MAX;
    m.attr("PUPIL_SIZE_MIN") = d::PUPIL_SIZE_MIN;
    m.attr("CONTOUR_SIZE_MIN") = d::CONTOUR_SIZE_MIN;
    m.attr("ELLIPSE_ROUNDNESS_RATIO") = d::ELLIPSE_ROUNDNESS_RATIO;
    m.attr("INITIAL_ELLIPSE_FIT_THRESHOLD") = d::INITIAL_ELLIPSE_FIT_THRESHOLD;
    m.attr("ELLIPSE_TRUE_SUPPORT_MIN_DIST") = d::ELLIPSE_TRUE_SUPPORT_MIN_DIST;
    m.attr("SUPPORT_PIXEL_RATIO_EXPONENT") = d::SUPPORT_PIXEL_RATIO_EXPONENT;
    m.attr("COARSE_DETECTION") = d::COARSE_DETECTION;
    m.attr("COARSE_FILTER_MIN") = d::COARSE_FILTER_MIN;
    m.attr("COARSE_FILTER_MAX") = d::COARSE_FILTER_MAX;
}
