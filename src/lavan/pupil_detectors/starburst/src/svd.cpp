// Singular Value Decomposition — Householder reduction + QR iteration.
// Used by ``pupil_fitting_inliers`` to solve the homogeneous conic-fit
// system in normalised coordinates.
//
// Algorithm: cvEyeTracker / openEyes ToolKit (2004-2006), GPL.
// Original sourcing: Numerical Recipes / Forsythe-Malcolm-Moler.

#include "starburst/ransac_ellipse.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace lavan::starburst {

double RansacEllipse::radius(double u, double v)
{
    // Pythagoras without overflow: equivalent to std::hypot(u, v).
    return std::hypot(u, v);
}

void RansacEllipse::svd(int m, int n, double **a, double **p, double *d, double **q)
{
    int flag, i, its, j, jj, k, l, nm;
    const int nm1 = n - 1;
    const int mm1 = m - 1;
    double c, f, h, s, x, y, z;
    double anorm = 0;
    double g = 0;
    double scale = 0;
    std::vector<double> r(n, 0.0);

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            p[i][j] = a[i][j];
        }
    }

    // Householder reduction to bidiagonal form.
    for (i = 0; i < n; i++) {
        l = i + 1;
        r[i] = scale * g;
        g = s = scale = 0.0;
        if (i < m) {
            for (k = i; k < m; k++) {
                scale += std::fabs(p[k][i]);
            }
            if (scale != 0.0) {
                for (k = i; k < m; k++) {
                    p[k][i] /= scale;
                    s += p[k][i] * p[k][i];
                }
                f = p[i][i];
                g = -std::copysign(std::sqrt(s), f);
                h = f * g - s;
                p[i][i] = f - g;
                if (i != nm1) {
                    for (j = l; j < n; j++) {
                        for (s = 0.0, k = i; k < m; k++) {
                            s += p[k][i] * p[k][j];
                        }
                        f = s / h;
                        for (k = i; k < m; k++) {
                            p[k][j] += f * p[k][i];
                        }
                    }
                }
                for (k = i; k < m; k++) {
                    p[k][i] *= scale;
                }
            }
        }
        d[i] = scale * g;
        g = s = scale = 0.0;
        if (i < m && i != nm1) {
            for (k = l; k < n; k++) {
                scale += std::fabs(p[i][k]);
            }
            if (scale != 0.0) {
                for (k = l; k < n; k++) {
                    p[i][k] /= scale;
                    s += p[i][k] * p[i][k];
                }
                f = p[i][l];
                g = -std::copysign(std::sqrt(s), f);
                h = f * g - s;
                p[i][l] = f - g;
                for (k = l; k < n; k++) {
                    r[k] = p[i][k] / h;
                }
                if (i != mm1) {
                    for (j = l; j < m; j++) {
                        for (s = 0.0, k = l; k < n; k++) {
                            s += p[j][k] * p[i][k];
                        }
                        for (k = l; k < n; k++) {
                            p[j][k] += s * r[k];
                        }
                    }
                }
                for (k = l; k < n; k++) {
                    p[i][k] *= scale;
                }
            }
        }
        anorm = std::max(anorm, std::fabs(d[i]) + std::fabs(r[i]));
    }

    // Accumulation of right-hand transformations.
    for (i = n - 1; i >= 0; i--) {
        if (i < nm1) {
            if (g != 0.0) {
                for (j = l; j < n; j++) {
                    q[j][i] = (p[i][j] / p[i][l]) / g;
                }
                for (j = l; j < n; j++) {
                    for (s = 0.0, k = l; k < n; k++) {
                        s += p[i][k] * q[k][j];
                    }
                    for (k = l; k < n; k++) {
                        q[k][j] += s * q[k][i];
                    }
                }
            }
            for (j = l; j < n; j++) {
                q[i][j] = q[j][i] = 0.0;
            }
        }
        q[i][i] = 1.0;
        g = r[i];
        l = i;
    }

    // Accumulation of left-hand transformations.
    for (i = n - 1; i >= 0; i--) {
        l = i + 1;
        g = d[i];
        if (i < nm1) {
            for (j = l; j < n; j++) {
                p[i][j] = 0.0;
            }
        }
        if (g != 0.0) {
            g = 1.0 / g;
            if (i != nm1) {
                for (j = l; j < n; j++) {
                    for (s = 0.0, k = l; k < m; k++) {
                        s += p[k][i] * p[k][j];
                    }
                    f = (s / p[i][i]) * g;
                    for (k = i; k < m; k++) {
                        p[k][j] += f * p[k][i];
                    }
                }
            }
            for (j = i; j < m; j++) {
                p[j][i] *= g;
            }
        } else {
            for (j = i; j < m; j++) {
                p[j][i] = 0.0;
            }
        }
        ++p[i][i];
    }

    // Diagonalisation of the bidiagonal form — QR iteration.
    for (k = n - 1; k >= 0; k--) {
        for (its = 0; its < 30; its++) {
            flag = 1;
            for (l = k; l >= 0; l--) {
                nm = l - 1;
                if (std::fabs(r[l]) + anorm == anorm) {
                    flag = 0;
                    break;
                }
                if (std::fabs(d[nm]) + anorm == anorm) {
                    break;
                }
            }
            if (flag != 0) {
                c = 0.0;
                s = 1.0;
                for (i = l; i <= k; i++) {
                    f = s * r[i];
                    if (std::fabs(f) + anorm != anorm) {
                        g = d[i];
                        h = radius(f, g);
                        d[i] = h;
                        h = 1.0 / h;
                        c = g * h;
                        s = (-f * h);
                        for (j = 0; j < m; j++) {
                            y = p[j][nm];
                            z = p[j][i];
                            p[j][nm] = y * c + z * s;
                            p[j][i] = z * c - y * s;
                        }
                    }
                }
            }
            z = d[k];
            if (l == k) {  // Convergence.
                if (z < 0.0) {
                    d[k] = -z;
                    for (j = 0; j < n; j++) {
                        q[j][k] = -q[j][k];
                    }
                }
                break;
            }
            if (its == 30) {
                return;  // No convergence in 30 SVD iterations.
            }
            x = d[l];  // Shift from bottom 2-by-2 minor.
            nm = k - 1;
            y = d[nm];
            g = r[nm];
            h = r[k];
            f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
            g = radius(f, 1.0);

            // Next QR transformation.
            f = ((x - z) * (x + z) + h * ((y / (f + std::copysign(g, f))) - h)) / x;
            c = s = 1.0;
            for (j = l; j <= nm; j++) {
                i = j + 1;
                g = r[i];
                y = d[i];
                h = s * g;
                g = c * g;
                z = radius(f, h);
                r[j] = z;
                c = f / z;
                s = h / z;
                f = x * c + g * s;
                g = g * c - x * s;
                h = y * s;
                y = y * c;
                for (jj = 0; jj < n; jj++) {
                    x = q[jj][j];
                    z = q[jj][i];
                    q[jj][j] = x * c + z * s;
                    q[jj][i] = z * c - x * s;
                }
                z = radius(f, h);
                d[j] = z;
                if (z != 0.0) {
                    z = 1.0 / z;
                    c = f * z;
                    s = h * z;
                }
                f = (c * g) + (s * y);
                x = (c * y) - (s * g);
                for (jj = 0; jj < m; jj++) {
                    y = p[jj][j];
                    z = p[jj][i];
                    p[jj][j] = y * c + z * s;
                    p[jj][i] = z * c - y * s;
                }
            }
            r[l] = 0.0;
            r[k] = f;
            d[k] = x;
        }
    }
}

}  // namespace lavan::starburst
