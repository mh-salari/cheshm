// Standalone pupil-shape bindings: the polar-Fourier form fit and the
// centre estimators, usable on a detector's dense contour or a handful of
// hand-placed boundary points.

#include "cheshm/helpers/shape/pupil_center.hpp"
#include "cheshm/helpers/shape/pupil_form.hpp"
#include "cheshm/helpers/shape/smoothing_spline.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace
{

using PointArray = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

std::vector<cv::Point> to_points(PointArray pts)
{
    const int n = static_cast<int>(pts.shape(0));
    const double* d = pts.data();
    std::vector<cv::Point> out;
    out.reserve(n);
    for (int i = 0; i < n; ++i)
    {
        out.emplace_back(static_cast<int>(std::lround(d[2 * i])), static_cast<int>(std::lround(d[2 * i + 1])));
    }
    return out;
}

// Robust polar-Fourier boundary fit. Returns ``None`` on failure, else a
// tuple ``(boundary (M, 2) float64, (cx, cy))`` in the input coordinates.
nb::object fit_pupil_form(PointArray pts, int harmonics, int samples, int iterations, double inward_rejection)
{
    const std::vector<cv::Point> contour = to_points(pts);
    const cheshm::PupilForm form = cheshm::fit_pupil_form(contour, harmonics, samples, iterations, inward_rejection);
    if (!form.ok)
    {
        return nb::none();
    }

    const int m = static_cast<int>(form.boundary.size());
    auto owner = std::make_unique<std::vector<double>>(2 * m);
    for (int i = 0; i < m; ++i)
    {
        (*owner)[2 * i] = static_cast<double>(form.boundary[i].x);
        (*owner)[2 * i + 1] = static_cast<double>(form.boundary[i].y);
    }
    double* data = owner->data();
    nb::capsule cap(owner.release(), [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
    const std::size_t shape[2] = {static_cast<std::size_t>(m), 2};
    nb::ndarray<nb::numpy, double, nb::ndim<2>> arr(data, 2, shape, cap);
    return nb::make_tuple(std::move(arr), nb::make_tuple(form.cx, form.cy));
}

// Pupil centre from boundary points by ``method``. Computes the convex hull
// and fitted ellipse internally. Returns ``None`` on failure (fewer than 5
// points, or an image-only method invoked without intensity data).
nb::object pupil_center(PointArray pts, int method)
{
    const std::vector<cv::Point> contour = to_points(pts);
    if (contour.size() < 5)
    {
        return nb::none();
    }
    std::vector<cv::Point> hull;
    cv::convexHull(contour, hull);
    const cv::RotatedRect ellipse_fit = cv::fitEllipse(contour);
    const cv::Mat no_image; // empty: image-only methods (center_of_mass) yield None
    const std::optional<cv::Point2d> c = cheshm::pupil_center(method, contour, hull, ellipse_fit, no_image);
    if (!c)
    {
        return nb::none();
    }
    return nb::make_tuple(c->x, c->y);
}

// Periodic cubic smoothing spline through ``points``. ``smoothness`` 0 passes
// through every point; larger values smooth. Returns ``None`` on failure
// (fewer than 4 points), else the sampled boundary as ``(n_samples, 2)``.
nb::object smoothing_spline(PointArray pts, double smoothness, int n_samples)
{
    const std::vector<cv::Point> points = to_points(pts);
    const std::vector<cv::Point> curve = cheshm::smoothing_spline(points, smoothness, n_samples);
    if (curve.empty())
    {
        return nb::none();
    }
    const int m = static_cast<int>(curve.size());
    auto owner = std::make_unique<std::vector<double>>(2 * m);
    for (int i = 0; i < m; ++i)
    {
        (*owner)[2 * i] = static_cast<double>(curve[i].x);
        (*owner)[2 * i + 1] = static_cast<double>(curve[i].y);
    }
    double* data = owner->data();
    nb::capsule cap(owner.release(), [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
    const std::size_t shape[2] = {static_cast<std::size_t>(m), 2};
    nb::ndarray<nb::numpy, double, nb::ndim<2>> arr(data, 2, shape, cap);
    return nb::cast(std::move(arr));
}

// Boundary of an ellipse sampled to ``n_samples`` points, parameterised
// identically to ``cv::ellipse2Poly`` (even steps in the angle parameter, not
// in arc length). ``w`` / ``h`` are the full axis lengths and ``angle`` is in
// degrees, matching ``cv::fitEllipse``. Returns the closed boundary as
// ``(n_samples, 2)`` float64.
nb::object ellipse_boundary(double cx, double cy, double w, double h, double angle, int n_samples)
{
    const int n = std::max(n_samples, 0);
    const double a = 0.5 * w;
    const double b = 0.5 * h;
    const double phi = angle * CV_PI / 180.0;
    const double cos_phi = std::cos(phi);
    const double sin_phi = std::sin(phi);
    auto owner = std::make_unique<std::vector<double>>(2 * static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        const double t = 2.0 * CV_PI * static_cast<double>(i) / static_cast<double>(n);
        const double ex = a * std::cos(t);
        const double ey = b * std::sin(t);
        (*owner)[2 * i] = cx + ex * cos_phi - ey * sin_phi;
        (*owner)[2 * i + 1] = cy + ex * sin_phi + ey * cos_phi;
    }
    double* data = owner->data();
    nb::capsule cap(owner.release(), [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
    const std::size_t shape[2] = {static_cast<std::size_t>(n), 2};
    nb::ndarray<nb::numpy, double, nb::ndim<2>> arr(data, 2, shape, cap);
    return nb::cast(std::move(arr));
}

} // namespace

NB_MODULE(_core, m)
{
    m.def("fit_pupil_form",
          &fit_pupil_form,
          "points"_a,
          "harmonics"_a,
          "samples"_a,
          "iterations"_a,
          "inward_rejection"_a);
    m.def("pupil_center", &pupil_center, "points"_a, "method"_a);
    m.def("smoothing_spline", &smoothing_spline, "points"_a, "smoothness"_a, "n_samples"_a);
    m.def("ellipse_boundary", &ellipse_boundary, "cx"_a, "cy"_a, "w"_a, "h"_a, "angle"_a, "n_samples"_a);

    m.attr("CENTER_CONVEX_HULL_CENTROID") = cheshm::CENTER_CONVEX_HULL_CENTROID;
    m.attr("CENTER_ELLIPSE_FIT") = cheshm::CENTER_ELLIPSE_FIT;
    m.attr("CENTER_MIN_AREA_RECT") = cheshm::CENTER_MIN_AREA_RECT;
    m.attr("CENTER_HULL_MOMENTS") = cheshm::CENTER_HULL_MOMENTS;
}
