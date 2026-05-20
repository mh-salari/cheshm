/*
 * integro_differential_operator_core.c — C implementation of Daugman's Integro-Differential Operator.
 *
 * Compile:
 *   cc -O3 -shared -fPIC -o integro_differential_operator_core.so integro_differential_operator_core.c -lm
 *   (macOS: cc -O3 -shared -fPIC -o integro_differential_operator_core.dylib integro_differential_operator_core.c -lm)
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ── Bresenham half-circle perimeter ──────────────────────────────────────── */

static int circle_perimeter_half(int r, int* out_dr, int* out_dc)
{
    int n = 0;
    int dp = 3 - 2 * r;
    int a = 0, b = r;

    while (a <= b)
    {
        out_dr[n] = a;
        out_dc[n] = b;
        n++;
        out_dr[n] = a;
        out_dc[n] = -b;
        n++;
        out_dr[n] = -a;
        out_dc[n] = b;
        n++;
        out_dr[n] = -a;
        out_dc[n] = -b;
        n++;
        if (dp > 0)
        {
            b--;
            dp += 4 * (a - b) + 10;
        }
        else
        {
            dp += 4 * a + 6;
        }
        a++;
    }
    return n;
}

/* ── Differential operator at one (x, y) ─────────────────────────────────── */

static void differential(const unsigned char* image,
                         int h,
                         int w,
                         int x,
                         int y,
                         int r_min,
                         int r_max,
                         const double* gk,
                         int gk_len,
                         double* out_value,
                         int* out_radius)
{
    int n_radii = r_max - r_min;
    double* values = (double*)malloc(n_radii * sizeof(double));
    /* Worst case: 8 points per Bresenham step, max radius steps = r_max */
    int max_pts = 8 * (r_max + 1);
    int* dr = (int*)malloc(max_pts * sizeof(int));
    int* dc = (int*)malloc(max_pts * sizeof(int));

    /* Compute mean pixel value on each circle */
    for (int ri = 0; ri < n_radii; ri++)
    {
        int r = r_min + ri;
        int n_pts = circle_perimeter_half(r, dr, dc);
        double total = 0.0;
        int count = 0;
        for (int k = 0; k < n_pts; k++)
        {
            int rr = x + dr[k];
            int cc = y + dc[k];
            if (rr >= 0 && rr < h && cc >= 0 && cc < w)
            {
                total += image[rr * w + cc];
                count++;
            }
        }
        values[ri] = (count > 0) ? total / count : 0.0;
    }

    /* Diff */
    int n_diff = n_radii - 1;
    double* diff_values = (double*)malloc(n_diff * sizeof(double));
    for (int i = 0; i < n_diff; i++)
    {
        diff_values[i] = values[i + 1] - values[i];
    }

    /* Convolve with Gaussian (full mode) */
    int conv_len = n_diff + gk_len - 1;
    double* conv = (double*)calloc(conv_len, sizeof(double));
    for (int i = 0; i < n_diff; i++)
    {
        for (int j = 0; j < gk_len; j++)
        {
            conv[i + j] += diff_values[i] * gk[j];
        }
    }

    /* Find max */
    double best_val = -1e30;
    int best_idx = 0;
    for (int i = 0; i < conv_len; i++)
    {
        if (conv[i] > best_val)
        {
            best_val = conv[i];
            best_idx = i;
        }
    }

    *out_value = best_val;
    *out_radius = best_idx + r_min;

    free(values);
    free(dr);
    free(dc);
    free(diff_values);
    free(conv);
}

/* ── Grid search (exported) ───────────────────────────────────────────────── */

/*
 * integro_differential_operator_search:  grid search over (cen_x ± range_, cen_y ± range_).
 *
 * Parameters:
 *   image     — row-major grayscale image (uint8), shape (h, w)
 *   h, w      — image dimensions
 *   cen_x, cen_y — search center (row, col)
 *   range_    — search radius in pixels
 *   step      — search step size
 *   r_min, r_max — iris radius search range
 *   gk        — 1-D Gaussian kernel, length gk_len
 *   gk_len    — length of gk
 *   out       — output array, row-major (n_results, 4): [x, y, score, radius]
 *               caller must allocate at least (2*range_/step + 1)^2 * 4 doubles
 *
 * Returns: number of results written.
 */
int integro_differential_operator_search(const unsigned char* image,
                                         int h,
                                         int w,
                                         int cen_x,
                                         int cen_y,
                                         int range_,
                                         int step,
                                         int r_min,
                                         int r_max,
                                         const double* gk,
                                         int gk_len,
                                         double* out)
{
    int n = 0;
    for (int dx = -range_; dx <= range_; dx += step)
    {
        int px = dx + cen_x;
        for (int dy = -range_; dy <= range_; dy += step)
        {
            int py = dy + cen_y;
            double value;
            int radius;
            differential(image, h, w, px, py, r_min, r_max, gk, gk_len, &value, &radius);
            out[n * 4 + 0] = (double)px;
            out[n * 4 + 1] = (double)py;
            out[n * 4 + 2] = round(value);
            out[n * 4 + 3] = (double)radius;
            n++;
        }
    }
    return n;
}
