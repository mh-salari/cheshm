/*
 * pupil_guided_contour_core.c — radial-gradient search with per-angle
 * search bounds, used by PupilGuidedContour.
 *
 * Algorithmic skeleton is the per-angle radial argmax from Daugman 2007
 * (see active_contour_core.c). The difference is that the search range
 * [r_min, r_max] is supplied as length-N arrays so the caller can shape the
 * annulus to match the pupil ellipse — narrower along its minor axis,
 * wider along its major axis. No eyelid mask: the Fourier low-pass on the
 * Python side handles occluded angles via its smoothing of the full ring.
 *
 * Compile:
 *   cc -O3 -shared -fPIC -o pupil_guided_contour_core.so   pupil_guided_contour_core.c -lm
 *   (macOS: cc -O3 -shared -fPIC -o pupil_guided_contour_core.dylib pupil_guided_contour_core.c -lm)
 */

#define _USE_MATH_DEFINES /* MSVC: enable M_PI in <math.h> */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "defaults.h"

const int pupil_guided_default_n = 360;
const int pupil_guided_default_m = 3;
const double pupil_guided_default_gradient_sigma = 1.0;
const double pupil_guided_default_radial_smoothing = 2.0;
const double pupil_guided_default_k_min = 2.0;
const double pupil_guided_default_k_max = 4.0;

static inline float bilinear_f32(const float* image, int h, int w, float x, float y, int* in_bounds)
{
    if (x < 0.0f || x > (float)(w - 1) || y < 0.0f || y > (float)(h - 1))
    {
        *in_bounds = 0;
        return 0.0f;
    }
    int x0 = (int)x;
    int y0 = (int)y;
    int x1 = (x0 + 1 < w) ? (x0 + 1) : x0;
    int y1 = (y0 + 1 < h) ? (y0 + 1) : y0;
    float fx = x - (float)x0;
    float fy = y - (float)y0;
    float v00 = image[y0 * w + x0];
    float v01 = image[y0 * w + x1];
    float v10 = image[y1 * w + x0];
    float v11 = image[y1 * w + x1];
    *in_bounds = 1;
    return (1.0f - fx) * (1.0f - fy) * v00 + fx * (1.0f - fy) * v01 + (1.0f - fx) * fy * v10 + fx * fy * v11;
}

/*
 * pupil_guided_radial_search:
 *
 *   gx, gy             — row-major float32 Sobel gradients, shape (h, w)
 *   h, w               — image dimensions
 *   cx, cy             — seed centre (column, row), in pixels
 *   n_angles           — number of theta samples
 *   r_min, r_max       — length-n_angles arrays of per-angle radial bounds
 *   n_r                — radial samples per angle (each angle gets its own
 *                         step dr_i = (r_max[i] - r_min[i]) / (n_r - 1))
 *   smoothing_kernel   — 1-D radial smoothing kernel (length k_len, odd)
 *   r_theta_out        — length-n_angles output; argmax radius per angle,
 *                         or -1.0 if every radial sample was OOB.
 *
 * Returns the number of valid (non-(-1)) entries.
 */
int pupil_guided_radial_search(const float* gx,
                               const float* gy,
                               int h,
                               int w,
                               float cx,
                               float cy,
                               int n_angles,
                               const float* r_min,
                               const float* r_max,
                               int n_r,
                               const double* smoothing_kernel,
                               int k_len,
                               double* r_theta_out)
{
    if (n_r < 2 || n_angles < 1)
        return 0;

    double* profile = (double*)malloc(n_r * sizeof(double));
    double* smoothed = (double*)malloc(n_r * sizeof(double));
    int half_k = k_len / 2;

    int n_valid = 0;
    for (int i = 0; i < n_angles; i++)
    {
        float rmin_i = r_min[i];
        float rmax_i = r_max[i];
        if (!(rmax_i > rmin_i))
        {
            r_theta_out[i] = -1.0;
            continue;
        }
        float dr = (rmax_i - rmin_i) / (float)(n_r - 1);

        double theta = 2.0 * M_PI * (double)i / (double)n_angles;
        float cos_t = (float)cos(theta);
        float sin_t = (float)sin(theta);

        int any_in = 0;
        for (int j = 0; j < n_r; j++)
        {
            float r = rmin_i + dr * (float)j;
            float x = cx + r * cos_t;
            float y = cy + r * sin_t;
            int in1 = 0, in2 = 0;
            float gxv = bilinear_f32(gx, h, w, x, y, &in1);
            float gyv = bilinear_f32(gy, h, w, x, y, &in2);
            if (in1 && in2)
            {
                profile[j] = (double)(gxv * cos_t + gyv * sin_t);
                any_in = 1;
            }
            else
            {
                profile[j] = -INFINITY;
            }
        }
        if (!any_in)
        {
            r_theta_out[i] = -1.0;
            continue;
        }

        /* Same-length 1-D convolution; OOB samples (-INFINITY) are dropped
         * from the weighted sum so they cannot win the argmax. */
        for (int j = 0; j < n_r; j++)
        {
            double acc = 0.0;
            int finite_seen = 0;
            for (int kk = 0; kk < k_len; kk++)
            {
                int idx = j + kk - half_k;
                if (idx < 0 || idx >= n_r)
                    continue;
                double v = profile[idx];
                if (!isfinite(v))
                    continue;
                acc += v * smoothing_kernel[kk];
                finite_seen = 1;
            }
            smoothed[j] = finite_seen ? acc : -INFINITY;
        }

        double best_val = -INFINITY;
        int best_idx = -1;
        for (int j = 0; j < n_r; j++)
        {
            if (smoothed[j] > best_val)
            {
                best_val = smoothed[j];
                best_idx = j;
            }
        }
        if (best_idx >= 0 && isfinite(best_val))
        {
            r_theta_out[i] = (double)(rmin_i + dr * (float)best_idx);
            n_valid++;
        }
        else
        {
            r_theta_out[i] = -1.0;
        }
    }

    free(profile);
    free(smoothed);
    return n_valid;
}
