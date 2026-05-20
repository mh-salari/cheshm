// Public C surface for the ExCuSe pupil detector — a single extern "C"
// entry point that lavan's Python wrapper loads via ctypes. The crop
// for ``pupil_roi`` is taken as a zero-copy view via the shared
// ``cpp/common`` helpers; the algorithm body itself never mutates the
// caller's buffer.

#include "ExCuSe/excuse.hpp"
#include "lavan/roi.hpp"

#include <cstdint>
#include <opencv2/core.hpp>

extern "C" {

// ExCuSe_detect — find the pupil ellipse via ExCuSe.
//
//  img_data           grayscale uint8 buffer, ``width * height`` bytes,
//                     row-major (numpy default).
//  roi_x, roi_y,      ROI rectangle in full-image coordinates. When
//  roi_w, roi_h       ``roi_w > 0 && roi_h > 0`` the algorithm runs on
//                     the cropped sub-image; otherwise the full image.
//  max_ellipse_radi   Upper bound on accepted ellipse semi-axis length
//                     (paper notation). Drives the quality validation.
//  good_ellipse_threshold   Pixel-count threshold for the goodness
//                     test on the candidate ellipse.
//  out_ellipse_params five doubles: ``{cx, cy, w, h, angle_deg}`` from
//                     ``cv::RotatedRect``. Centre is in full-image
//                     coordinates regardless of whether an ROI was used.
//
// Returns 1 on success, 0 on error.
int ExCuSe_detect(
    const std::uint8_t *img_data,
    int width,
    int height,
    int roi_x,
    int roi_y,
    int roi_w,
    int roi_h,
    int max_ellipse_radi,
    int good_ellipse_threshold,
    double *out_ellipse_params)
{
    if (img_data == nullptr || width <= 0 || height <= 0 || out_ellipse_params == nullptr) {
        return 0;
    }

    const cv::Mat full(height, width, CV_8U, const_cast<std::uint8_t *>(img_data));
    cv::Rect crop(0, 0, width, height);
    if (lavan::roi_is_active(roi_w, roi_h)) {
        crop = lavan::clamp_roi(roi_x, roi_y, roi_w, roi_h, width, height);
        if (crop.area() == 0) {
            return 0;
        }
    }
    const cv::Mat view = full(crop);

    const cv::RotatedRect ellipse =
        lavan::ExCuSe::findPupilEllipse(view, max_ellipse_radi, good_ellipse_threshold);

    out_ellipse_params[0] = ellipse.center.x + crop.x;
    out_ellipse_params[1] = ellipse.center.y + crop.y;
    out_ellipse_params[2] = ellipse.size.width;
    out_ellipse_params[3] = ellipse.size.height;
    out_ellipse_params[4] = ellipse.angle;
    return 1;
}

}  // extern "C"
