// Public C surface for Swirski 2D — single extern "C" entry point
// that the Python wrapper loads via ctypes.

#include "swirski_2d/pupil_tracker.hpp"

#include <algorithm>
#include <cstdint>
#include <opencv2/core.hpp>

extern "C" {

// swirski_2d_detect — find the pupil ellipse via Swirski et al. 2012.
//
//  img_data           grayscale uint8 buffer, ``width * height`` bytes,
//                     row-major (numpy default).
//  radius_min/max     plausible pupil radius range in pixels.
//  canny_blur         Gaussian σ before edge detection.
//  canny_thresh1/2    Canny hysteresis thresholds.
//  starburst_points   N rays shot from each seed centre to gather
//                     candidate edge points.
//  percentage_inliers / inlier_iterations / image_aware_support /
//  early_termination_percentage / early_rejection / seed
//                     pass straight through to ``TrackerParams``.
//  out_ellipse_params five doubles: ``{cx, cy, w, h, angle_deg}``
//                     from ``cv::RotatedRect``.
//  out_n_inliers      count of inlier points written to ``inliers_xy``.
//  inliers_xy         caller-allocated buffer of
//                     ``2 * max_inliers`` doubles for ``(x, y)`` pairs.
//  max_inliers        capacity of ``inliers_xy`` in points.
//
// Returns 1 on success, 0 on failure.
int swirski_2d_detect(
    const std::uint8_t *img_data,
    int width,
    int height,
    int radius_min,
    int radius_max,
    double canny_blur,
    double canny_thresh1,
    double canny_thresh2,
    int starburst_points,
    int percentage_inliers,
    int inlier_iterations,
    int image_aware_support,
    int early_termination_percentage,
    int early_rejection,
    int seed,
    double *out_ellipse_params,
    int *out_n_inliers,
    double *inliers_xy,
    int max_inliers)
{
    using namespace lavan::swirski_2d;  // NOLINT(google-build-using-namespace)

    if (img_data == nullptr || width <= 0 || height <= 0) {
        return 0;
    }

    // Non-owning cv::Mat over the caller's buffer — Swirski2D does not
    // mutate the input.
    const cv::Mat input(height, width, CV_8U, const_cast<std::uint8_t *>(img_data));

    TrackerParams params{};
    params.Radius_Min = radius_min;
    params.Radius_Max = radius_max;
    params.CannyBlur = canny_blur;
    params.CannyThreshold1 = canny_thresh1;
    params.CannyThreshold2 = canny_thresh2;
    params.StarburstPoints = starburst_points;
    params.PercentageInliers = percentage_inliers;
    params.InlierIterations = inlier_iterations;
    params.ImageAwareSupport = image_aware_support != 0;
    params.EarlyTerminationPercentage = early_termination_percentage;
    params.EarlyRejection = early_rejection != 0;
    params.Seed = seed;

    findPupilEllipse_out result;
    const bool ok = findPupilEllipse(params, input, result);
    if (!ok) {
        if (out_n_inliers != nullptr) {
            *out_n_inliers = 0;
        }
        return 0;
    }

    out_ellipse_params[0] = result.elPupil.center.x;
    out_ellipse_params[1] = result.elPupil.center.y;
    out_ellipse_params[2] = result.elPupil.size.width;
    out_ellipse_params[3] = result.elPupil.size.height;
    out_ellipse_params[4] = result.elPupil.angle;  // degrees, cv::RotatedRect convention

    const int n = std::min(static_cast<int>(result.inliers.size()), max_inliers);
    for (int i = 0; i < n; i++) {
        inliers_xy[2 * i] = result.inliers[i].x;
        inliers_xy[2 * i + 1] = result.inliers[i].y;
    }
    if (out_n_inliers != nullptr) {
        *out_n_inliers = n;
    }
    return 1;
}

}  // extern "C"
