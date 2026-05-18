/*
 * active_contour_core.c — C kernel for Daugman 2007 Fourier-series active
 * contour iris boundary localization.
 *
 *   Daugman, J. (2007). "New Methods in Iris Recognition." IEEE Trans.
 *   Systems, Man, and Cybernetics, Part B, 37(5), 1167-1175, Section II.
 *
 * The Python wrapper (DaugmanActiveContour) supplies precomputed Sobel
 * gradient images Gx, Gy (float32, single channel) and a 1-D Gaussian kernel
 * used to smooth the radial gradient profile per angle. This kernel does the
 * inner per-angle work:
 *
 *   for each angle theta (skipped if eyelid_mask[i] is non-zero):
 *     for r in n_r radii from r_min..r_max:
 *       bilinearly sample Gx, Gy at (cx + r cos theta, cy + r sin theta)
 *       project onto the radial direction: g = Gx cos + Gy sin
 *     smooth g[] with the supplied 1-D Gaussian kernel (full convolution)
 *     write argmax_r of the smoothed profile to r_theta[i]
 *
 * Angles where every radial sample was out-of-bounds, or that were masked
 * out, are returned as -1.0. The Python wrapper interpolates across them.
 *
 * Compile:
 *   cc -O3 -shared -fPIC -o active_contour_core.so active_contour_core.c -lm
 *   (macOS: cc -O3 -shared -fPIC -o active_contour_core.dylib active_contour_core.c -lm)
 */

#define _USE_MATH_DEFINES  /* MSVC: enable M_PI in <math.h> */
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Bilinear-interpolated sample from a float32 single-channel image. Returns
 * 0.0 (and writes 0 to *in_bounds) if (x, y) is outside the safely-sampleable
 * region (need 1 px margin for the bilinear kernel). */
static inline float bilinear_f32(
    const float *image, int h, int w, float x, float y, int *in_bounds)
{
    if (x < 0.0f || x > (float)(w - 1) || y < 0.0f || y > (float)(h - 1)) {
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
    return (1.0f - fx) * (1.0f - fy) * v00
         +        fx  * (1.0f - fy) * v01
         + (1.0f - fx) *        fy  * v10
         +        fx  *        fy  * v11;
}

/*
 * active_contour_radial_search:
 *
 *   gx, gy             — row-major float32 Sobel gradients, shape (h, w)
 *   h, w               — image dimensions
 *   cx, cy             — seed centre, in (column, row) pixel coordinates
 *   n_angles           — number of theta samples (paper "N")
 *   r_min, r_max       — radial search bounds (inclusive)
 *   n_r                — number of radial samples in [r_min, r_max]
 *   eyelid_mask        — uint8 length n_angles; 0 = sample this angle,
 *                         non-zero = skip (eyelid-occluded)
 *   smoothing_kernel   — 1-D radial smoothing kernel (Gaussian), length k_len
 *   k_len              — kernel length (must be odd; centred at (k_len-1)/2)
 *   r_theta_out        — output length n_angles. r_theta_out[i] = radius of
 *                         the smoothed-profile peak, or -1.0 if invalid.
 *
 * Returns: number of valid (non-(-1)) entries written.
 */
int active_contour_radial_search(
    const float *gx, const float *gy, int h, int w,
    float cx, float cy,
    int n_angles, float r_min, float r_max, int n_r,
    const unsigned char *eyelid_mask,
    const double *smoothing_kernel, int k_len,
    double *r_theta_out)
{
    if (n_r < 2 || n_angles < 1) return 0;
    float dr = (r_max - r_min) / (float)(n_r - 1);
    float *radii = (float *)malloc(n_r * sizeof(float));
    for (int j = 0; j < n_r; j++) {
        radii[j] = r_min + dr * (float)j;
    }

    /* Reusable scratch for the per-angle radial-gradient profile. */
    double *profile = (double *)malloc(n_r * sizeof(double));
    double *smoothed = (double *)malloc(n_r * sizeof(double));
    int half_k = k_len / 2;

    int n_valid = 0;
    for (int i = 0; i < n_angles; i++) {
        if (eyelid_mask != NULL && eyelid_mask[i]) {
            r_theta_out[i] = -1.0;
            continue;
        }
        double theta = 2.0 * M_PI * (double)i / (double)n_angles;
        float cos_t = (float)cos(theta);
        float sin_t = (float)sin(theta);

        int any_in = 0;
        for (int j = 0; j < n_r; j++) {
            float x = cx + radii[j] * cos_t;
            float y = cy + radii[j] * sin_t;
            int in1 = 0, in2 = 0;
            float gxv = bilinear_f32(gx, h, w, x, y, &in1);
            float gyv = bilinear_f32(gy, h, w, x, y, &in2);
            if (in1 && in2) {
                profile[j] = (double)(gxv * cos_t + gyv * sin_t);
                any_in = 1;
            } else {
                /* Sentinel: out-of-bounds samples cannot win the argmax. */
                profile[j] = -INFINITY;
            }
        }
        if (!any_in) {
            r_theta_out[i] = -1.0;
            continue;
        }

        /* Same-length 1-D convolution with the smoothing kernel. Masked
         * (-INFINITY) entries are skipped from the sum and from the kept
         * mask so they remain ineligible for the argmax. */
        for (int j = 0; j < n_r; j++) {
            double acc = 0.0;
            int finite_seen = 0;
            for (int kk = 0; kk < k_len; kk++) {
                int idx = j + kk - half_k;
                if (idx < 0 || idx >= n_r) continue;
                double v = profile[idx];
                if (!isfinite(v)) continue;
                acc += v * smoothing_kernel[kk];
                finite_seen = 1;
            }
            smoothed[j] = finite_seen ? acc : -INFINITY;
        }

        double best_val = -INFINITY;
        int best_idx = -1;
        for (int j = 0; j < n_r; j++) {
            if (smoothed[j] > best_val) {
                best_val = smoothed[j];
                best_idx = j;
            }
        }
        if (best_idx >= 0 && isfinite(best_val)) {
            r_theta_out[i] = (double)radii[best_idx];
            n_valid++;
        } else {
            r_theta_out[i] = -1.0;
        }
    }

    free(radii);
    free(profile);
    free(smoothed);
    return n_valid;
}
