/*
 * Daugman 2007 active-contour radial search — tunable defaults.
 *
 *   Daugman, J. (2007). "New Methods in Iris Recognition." IEEE Trans.
 *   Systems, Man, and Cybernetics, Part B, 37(5), 1167-1175.
 */

#ifndef CHESHM_DAUGMAN_ACTIVE_CONTOUR_DEFAULTS_H
#define CHESHM_DAUGMAN_ACTIVE_CONTOUR_DEFAULTS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Number of angular samples theta taken around the seed centre. */
extern const int active_contour_default_n;

/* Number of Fourier coefficients kept by the boundary low-pass. The 2007
 * paper recommends 5 for the iris outer boundary. */
extern const int active_contour_default_m;

/* Gaussian sigma (pixels) for the pre-Sobel image blur. */
extern const double active_contour_default_gradient_sigma;

/* Gaussian sigma (radial samples) applied to each angle's radial-gradient
 * profile before the argmax. */
extern const double active_contour_default_radial_smoothing;

/* Non-zero masks out the upper-eyelid angular wedge during the radial
 * search; zero searches every angle. */
extern const int active_contour_default_skip_eyelid_wedges;

/* Lower bound (pixels) on candidate iris radius. */
extern const double active_contour_default_r_min;

/* Upper bound (pixels) on candidate iris radius. */
extern const double active_contour_default_r_max;

#ifdef __cplusplus
}
#endif

#endif /* CHESHM_DAUGMAN_ACTIVE_CONTOUR_DEFAULTS_H */
