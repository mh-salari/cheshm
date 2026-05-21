/*
 * Daugman integro-differential operator — tunable defaults.
 *
 *   Daugman, J. (2004). "How Iris Recognition Works." IEEE Trans. Circuits
 *   and Systems for Video Technology, 14(1), 21-30, eq. (1).
 */

#ifndef CHESHM_DAUGMAN_INTEGRO_DIFFERENTIAL_DEFAULTS_H
#define CHESHM_DAUGMAN_INTEGRO_DIFFERENTIAL_DEFAULTS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Lower bound (pixels) on candidate iris radius. */
extern const int integro_differential_default_r_min;

/* Upper bound (pixels) on candidate iris radius. */
extern const int integro_differential_default_r_max;

/* Which circle perimeter feeds the line integral:
 *   "half" — left + right semicircles only (avoids eyelid bias)
 *   "full" — the entire circle. */
extern const char* const integro_differential_default_c_type;

/* Half-width (pixels) of the centre-sweep grid around the seed. */
extern const int integro_differential_default_range;

/* Grid step (pixels) for the centre sweep. */
extern const int integro_differential_default_step;

#ifdef __cplusplus
}
#endif

#endif /* CHESHM_DAUGMAN_INTEGRO_DIFFERENTIAL_DEFAULTS_H */
