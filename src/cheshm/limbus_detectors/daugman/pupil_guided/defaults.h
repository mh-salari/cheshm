/*
 * Pupil-shape-guided active contour for iris boundary — tunable defaults.
 *
 * Radial-gradient framework from:
 *   Daugman, J. (2007). "New Methods in Iris Recognition." IEEE Trans.
 *   Systems, Man, and Cybernetics, Part B, 37(5), 1167-1175.
 */

#ifndef CHESHM_DAUGMAN_PUPIL_GUIDED_DEFAULTS_H
#define CHESHM_DAUGMAN_PUPIL_GUIDED_DEFAULTS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Number of angular samples theta taken around the seed centre. */
extern const int pupil_guided_default_n;

/* Number of Fourier coefficients kept by the boundary low-pass. */
extern const int pupil_guided_default_m;

/* Gaussian sigma (pixels) for the pre-Sobel image blur. */
extern const double pupil_guided_default_gradient_sigma;

/* Gaussian sigma (radial samples) applied to each angle's radial-gradient
 * profile before the argmax. */
extern const double pupil_guided_default_radial_smoothing;

/* Lower scale factor: per-angle search starts at k_min * pupil_radius(theta). */
extern const double pupil_guided_default_k_min;

/* Upper scale factor: per-angle search ends at k_max * pupil_radius(theta). */
extern const double pupil_guided_default_k_max;

#ifdef __cplusplus
}
#endif

#endif /* CHESHM_DAUGMAN_PUPIL_GUIDED_DEFAULTS_H */
