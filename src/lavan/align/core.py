"""Iris-based rigid alignment of two eye images.

Given a reference and a target grayscale eye-image plus the pupil + limbus
geometry detected on each, fit a small rigid transform ``(dx, dy, theta)``
that warps the target onto the reference frame. The cost function is the
mean absolute intensity difference inside a barrel-shaped iris mask (top
and bottom eyelid zones excluded), so the alignment is driven by iris
texture rather than by eyelashes or specular glints.

Two equivalent implementations are exposed:

  - :func:`align_by_min_diff` (Numba-accelerated, default).
  - :func:`align_by_min_diff_plain` (pure Python triple loop, ~4x slower).

Both take the same mask, search-range, and rotation-centre arguments and
return ``((dx, dy, theta), best_score)``. A separate :func:`apply_transform`
helper applies the returned parameters to an image, and
:func:`align_by_iris_center` returns the initial translation that places
the target's iris centre on top of the reference's (useful as a
pre-alignment step before the image-diff refinement).
"""

from collections.abc import Iterable

import cv2
import numba
import numpy as np


def make_iris_mask(
    img_shape: tuple[int, ...],
    limbus_center: tuple[float, float],
    limbus_r: float,
    pupil_r: float,
    exclude_top: float = 60,
    exclude_bottom: float = 45,
    inner_margin: float = 15,
) -> np.ndarray:
    """Annular ring mask covering the iris texture, excluding eyelid zones."""
    h, w = img_shape[:2]
    lx, ly = limbus_center
    mask = np.zeros((h, w), dtype=np.uint8)

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - lx) ** 2 + (Y - ly) ** 2)

    inner = pupil_r + inner_margin
    outer = limbus_r + 10
    in_ring = (dist >= inner) & (dist <= outer)

    angle = np.degrees(np.arctan2(Y - ly, X - lx))
    in_top = (angle >= (-90 - exclude_top)) & (angle <= (-90 + exclude_top))
    in_bottom = (angle >= (90 - exclude_bottom)) & (angle <= (90 + exclude_bottom))

    mask[in_ring & ~in_top & ~in_bottom] = 255
    return mask


def make_barrel_mask(
    img_shape: tuple[int, ...],
    limbus_center: tuple[float, float],
    limbus_r: float,
    pupil_r: float,
    exclude_top: float = 60,
    exclude_bottom: float = 45,
    inner_margin: float = 15,
) -> np.ndarray:
    """Filled barrel mask: the iris ring with the gap between left/right crescents filled per row."""
    ring = make_iris_mask(img_shape, limbus_center, limbus_r, pupil_r, exclude_top, exclude_bottom, inner_margin)
    mask = ring.copy()

    for y in range(img_shape[0]):
        row = ring[y, :]
        cols = np.where(row > 0)[0]
        if len(cols) >= 2:
            mask[y, cols[0] : cols[-1] + 1] = 255

    return mask


# =============================================================================
# Plain Python alignment (simple, no dependencies beyond OpenCV + NumPy)
# =============================================================================


def _grid_search_plain(
    ref_f: np.ndarray,
    img_mov: np.ndarray,
    mask_bool: np.ndarray,
    n_pixels: int,
    thetas: np.ndarray,
    dxs: Iterable[float],
    dys: Iterable[float],
    cx: float,
    cy: float,
) -> tuple[tuple[float, float, float], float]:
    """Exhaustive search: one warpAffine per (theta, dx, dy) combo."""
    h, w = ref_f.shape
    best_score = np.inf
    best_params = (0.0, 0.0, 0.0)

    for theta in thetas:
        R = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
        for dx in dxs:
            for dy in dys:
                M = R.copy()
                M[0, 2] += dx
                M[1, 2] += dy
                warped = cv2.warpAffine(img_mov, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                score = np.sum(np.abs(ref_f[mask_bool] - warped[mask_bool].astype(np.float32))) / n_pixels
                if score < best_score:
                    best_score = score
                    best_params = (float(dx), float(dy), float(theta))

    return best_params, best_score


def align_by_min_diff_plain(
    img_ref: np.ndarray,
    img_mov: np.ndarray,
    mask: np.ndarray,
    dx_range: tuple[int, int] = (-10, 11),
    dy_range: tuple[int, int] = (-10, 11),
    rot_range: tuple[float, float, float] = (-2.0, 2.0, 0.05),
    rotation_center: tuple[float, float] | None = None,
) -> tuple[tuple[float, float, float], float]:
    """Plain Python alignment: coarse integer search + fine sub-pixel refinement.

    Straightforward triple loop — easy to read but slow (~4x slower than the Numba version).

    `rotation_center` is the (cx, cy) point around which the candidate rotations are applied.
    Must match whatever rotation center is later used to apply the returned transform, or the
    final alignment will differ from the one that minimized the search cost. Defaults to the
    image center.
    """
    h, w = img_ref.shape
    if rotation_center is None:
        cx, cy = w / 2, h / 2
    else:
        cx, cy = rotation_center
    mask_bool = mask > 0
    n_pixels = mask_bool.sum()
    ref_f = img_ref.astype(np.float32)

    rot_start, rot_end, rot_step = rot_range
    n_rot = round((rot_end - rot_start) / rot_step) + 1
    thetas = np.linspace(rot_start, rot_end, n_rot)

    # Coarse pass (integer px)
    coarse_dxs = range(*dx_range)
    coarse_dys = range(*dy_range)
    best_params, best_score = _grid_search_plain(
        ref_f, img_mov, mask_bool, n_pixels, thetas, coarse_dxs, coarse_dys, cx, cy
    )

    # Fine pass (sub-pixel)
    cdx, cdy, ctheta = best_params
    fine_thetas = np.linspace(ctheta - 0.05, ctheta + 0.05, 11)
    fine_dxs = np.linspace(cdx - 1.0, cdx + 1.0, 21)
    fine_dys = np.linspace(cdy - 1.0, cdy + 1.0, 21)
    best_params, best_score = _grid_search_plain(
        ref_f, img_mov, mask_bool, n_pixels, fine_thetas, fine_dxs, fine_dys, cx, cy
    )

    return best_params, best_score


# =============================================================================
# Numba-accelerated alignment (same logic, ~4x faster)
# One warpAffine per theta, Numba searches all (dx, dy) via bilinear interpolation.
# =============================================================================


@numba.njit(cache=True, parallel=True)
def _search_shifts(
    ref_vals: np.ndarray,
    img: np.ndarray,
    mask_rows: np.ndarray,
    mask_cols: np.ndarray,
    dxs: np.ndarray,
    dys: np.ndarray,
    h: int,
    w: int,
) -> tuple[float, float, float]:
    """Score all (dx, dy) combos on a pre-rotated image. Returns best (dx, dy, score).

    Uses bilinear interpolation to sample rotated image at (mask_row - dy, mask_col - dx),
    matching warpAffine's convention where positive dx shifts the output right.
    Parallelized across all (dx, dy) combos using numba.prange.
    """
    n_dx = len(dxs)
    n_dy = len(dys)
    n_combos = n_dx * n_dy
    scores = np.full(n_combos, 1e30)
    n = len(mask_rows)

    for idx in numba.prange(n_combos):
        di = idx // n_dy
        dj = idx % n_dy
        dx = dxs[di]
        dy = dys[dj]
        total = 0.0
        count = 0
        for k in range(n):
            # Subtract to match warpAffine convention: positive dx = shift output right
            fy = mask_rows[k] - dy
            fx = mask_cols[k] - dx
            x0 = np.int64(np.floor(fx))
            y0 = np.int64(np.floor(fy))
            x1 = x0 + 1
            y1 = y0 + 1
            if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
                continue
            wx = fx - x0
            wy = fy - y0
            val = img[y0, x0] * (1 - wx) * (1 - wy) + img[y0, x1] * wx * (1 - wy)
            val += img[y1, x0] * (1 - wx) * wy + img[y1, x1] * wx * wy
            total += abs(ref_vals[k] - val)
            count += 1
        if count > 0:
            scores[idx] = total / count

    best_idx = np.argmin(scores)
    best_di = best_idx // n_dy
    best_dj = best_idx % n_dy
    return dxs[best_di], dys[best_dj], scores[best_idx]


def _grid_search(
    ref_f: np.ndarray,
    img_mov: np.ndarray,
    mask_bool: np.ndarray,
    thetas: np.ndarray,
    dxs: Iterable[float],
    dys: Iterable[float],
    cx: float,
    cy: float,
) -> tuple[tuple[float, float, float], float]:
    """Fast grid search: one warpAffine per theta, Numba handles all (dx, dy) shifts."""
    h, w = ref_f.shape
    mask_rows, mask_cols = np.where(mask_bool)
    mask_rows_f = mask_rows.astype(np.float64)
    mask_cols_f = mask_cols.astype(np.float64)
    ref_vals = ref_f[mask_bool].astype(np.float64)
    dxs_arr = np.asarray(list(dxs), dtype=np.float64)
    dys_arr = np.asarray(list(dys), dtype=np.float64)

    best_score = np.inf
    best_params = (0.0, 0.0, 0.0)

    for theta in thetas:
        R = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
        rotated = cv2.warpAffine(img_mov, R, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        rotated_f = rotated.astype(np.float64)

        dx, dy, score = _search_shifts(ref_vals, rotated_f, mask_rows_f, mask_cols_f, dxs_arr, dys_arr, h, w)
        if score < best_score:
            best_score = score
            best_params = (float(dx), float(dy), float(theta))

    return best_params, best_score


def align_by_min_diff(
    img_ref: np.ndarray,
    img_mov: np.ndarray,
    mask: np.ndarray,
    dx_range: tuple[int, int] = (-10, 11),
    dy_range: tuple[int, int] = (-10, 11),
    rot_range: tuple[float, float, float] = (-2.0, 2.0, 0.05),
    rotation_center: tuple[float, float] | None = None,
) -> tuple[tuple[float, float, float], float]:
    """Numba-accelerated alignment: coarse integer search + fine sub-pixel refinement.

    For each theta: one warpAffine (rotation only), then Numba exhaustively
    searches all (dx, dy) shifts using bilinear interpolation on the rotated image.

    `rotation_center` is the (cx, cy) point around which the candidate rotations are applied.
    Must match whatever rotation center is later used to apply the returned transform, or the
    final alignment will differ from the one that minimized the search cost. Defaults to the
    image center.
    """
    h, w = img_ref.shape
    if rotation_center is None:
        cx, cy = w / 2, h / 2
    else:
        cx, cy = rotation_center
    mask_bool = mask > 0
    ref_f = img_ref.astype(np.float32)

    rot_start, rot_end, rot_step = rot_range
    n_rot = round((rot_end - rot_start) / rot_step) + 1
    thetas = np.linspace(rot_start, rot_end, n_rot)

    # Coarse pass (integer px)
    coarse_dxs = range(*dx_range)
    coarse_dys = range(*dy_range)
    best_params, best_score = _grid_search(ref_f, img_mov, mask_bool, thetas, coarse_dxs, coarse_dys, cx, cy)

    # Fine pass (sub-pixel)
    cdx, cdy, ctheta = best_params
    fine_thetas = np.linspace(ctheta - 0.05, ctheta + 0.05, 11)
    fine_dxs = np.linspace(cdx - 1.0, cdx + 1.0, 21)
    fine_dys = np.linspace(cdy - 1.0, cdy + 1.0, 21)
    best_params, best_score = _grid_search(ref_f, img_mov, mask_bool, fine_thetas, fine_dxs, fine_dys, cx, cy)

    return best_params, best_score


def align_by_iris_center(
    ref_iris_center: tuple[float, float],
    mov_iris_center: tuple[float, float],
) -> tuple[float, float, float]:
    """Return ``(dx, dy, 0.0)`` translation that places ``mov_iris_center`` on ``ref_iris_center``."""
    dx = ref_iris_center[0] - mov_iris_center[0]
    dy = ref_iris_center[1] - mov_iris_center[1]
    return (dx, dy, 0.0)


def apply_transform(
    img: np.ndarray,
    params: tuple[float, float, float],
    center: tuple[float, float] | None = None,
) -> np.ndarray:
    """Apply (dx, dy, theta) rigid transform to an image."""
    h, w = img.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    dx, dy, theta = params
    M = cv2.getRotationMatrix2D(center, theta, 1.0)
    M[0, 2] += dx
    M[1, 2] += dy
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
