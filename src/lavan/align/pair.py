"""High-level entry point: align two eye images given pre-computed detections.

``align_eye_images(ref_img, tgt_img, ref_det, tgt_det)`` consumes detection
dicts produced by any upstream pupil + glint + limbus detector (e.g.
:mod:`lavan.detect`) and runs the two-step rigid registration. Step 1
translates the target so its mean glint position matches the reference;
step 2 refines ``(dx, dy, theta)`` by minimising image difference inside a
barrel-shaped iris mask. Detection is never run inside this module —
callers pass pre-computed dicts.

Detection dict shape::

    {
        "pupil_center":  [cx, cy],
        "pupil_ellipse": [[cx, cy], [w, h], angle],
        "glints":        [[gx, gy], ...],          # at least one
        "limbus":        {"center": [lx, ly], "radius": r},
    }
"""

import numpy as np

from .core import align_by_iris_center, align_by_min_diff, apply_transform, make_barrel_mask


def _glint_centroid(detection: dict) -> tuple[float, float]:
    """Mean glint centre for the detection. Raises if no glints were found."""
    glints = detection.get("glints") or []
    if not glints:
        raise ValueError("detection has no glints — cannot align without a step-1 reference")
    cx = sum(g[0] for g in glints) / len(glints)
    cy = sum(g[1] for g in glints) / len(glints)
    return float(cx), float(cy)


def _pupil_radius(detection: dict) -> float:
    """Pupil radius (max axis / 2) from the detection's ``pupil_ellipse``."""
    _, (w, h), _ = detection["pupil_ellipse"]
    return max(float(w), float(h)) / 2


def align_eye_images(
    ref_img: np.ndarray,
    tgt_img: np.ndarray,
    ref_det: dict,
    tgt_det: dict,
) -> dict:
    """Rigid-align ``tgt_img`` onto ``ref_img`` using cached detections.

    Returns a dict with the aligned image and the per-step transforms::

        {
            "aligned":            <warped target image, same shape as ref_img>,
            "step1_translation":  (dx, dy),
            "step2_transform":    (dx, dy, theta_deg),
            "rotation_center":    (cx, cy),   # the (rounded) iris centre used
        }

    Both detection dicts must include ``pupil_ellipse``, ``glints``, and a
    non-``None`` ``limbus``. Raises ``ValueError`` if any of those are
    missing.
    """
    if ref_det.get("limbus") is None or tgt_det.get("limbus") is None:
        raise ValueError("both detections must include a non-None limbus")

    ref_glint = _glint_centroid(ref_det)
    tgt_glint = _glint_centroid(tgt_det)

    ref_lim = ref_det["limbus"]
    tgt_lim = tgt_det["limbus"]
    cx = (ref_lim["center"][0] + tgt_lim["center"][0]) / 2
    cy = (ref_lim["center"][1] + tgt_lim["center"][1]) / 2
    center = (round(cx), round(cy))
    iris_r = round((float(ref_lim["radius"]) + float(tgt_lim["radius"])) / 2)
    pupil_r = round(max(_pupil_radius(ref_det), _pupil_radius(tgt_det)))

    # Step 1: rigid translation so glint centroids align.
    p1 = align_by_iris_center(ref_glint, tgt_glint)
    step1 = apply_transform(tgt_img, p1, center=center)

    # Step 2: refine dx/dy/theta inside the iris barrel.
    barrel = make_barrel_mask(ref_img.shape, center, iris_r, pupil_r)
    p2, _ = align_by_min_diff(ref_img, step1, barrel, rotation_center=center)
    aligned = apply_transform(step1, p2, center=center)

    return {
        "aligned": aligned,
        "step1_translation": (float(p1[0]), float(p1[1])),
        "step2_transform": (float(p2[0]), float(p2[1]), float(p2[2])),
        "rotation_center": (int(center[0]), int(center[1])),
    }
