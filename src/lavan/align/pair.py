"""High-level entry point: align two eye images given pre-computed detections.

``align_eye_images(ref_img, tgt_img, ref_det, tgt_det)`` consumes detection
dicts produced by any upstream pupil + glint + limbus detector (e.g.
:mod:`lavan.detect`) and runs a two-step rigid registration. The caller
picks which step 1 anchor (glint centroid or pupil centre) drives the
translation, and whether step 2's iris-texture refinement runs at all.

Detection dict shape::

    {
        "pupil_center":  [cx, cy],
        "pupil_ellipse": [[cx, cy], [w, h], angle],
        "glints":        [[gx, gy], ...],          # at least one
        "limbus":        {"center": [lx, ly], "radius": r},
    }
"""

from typing import Literal

import numpy as np

from .core import align_by_min_diff, align_by_translation, apply_transform, make_barrel_mask

Step1Method = Literal["glint", "pupil"] | None


def _glint_centroid(detection: dict) -> tuple[float, float]:
    """Mean glint centre for the detection. Raises if no glints were found."""
    glints = detection.get("glints") or []
    if not glints:
        raise ValueError("detection has no glints — cannot align without a step-1 reference")
    cx = sum(g[0] for g in glints) / len(glints)
    cy = sum(g[1] for g in glints) / len(glints)
    return float(cx), float(cy)


def _pupil_center(detection: dict) -> tuple[float, float]:
    """Pupil centre from the detection's ``pupil_center`` (or ``pupil_ellipse`` centre)."""
    if "pupil_center" in detection:
        cx, cy = detection["pupil_center"]
        return float(cx), float(cy)
    (cx, cy), _, _ = detection["pupil_ellipse"]
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
    *,
    step1: Step1Method = "glint",
    step2: bool = True,
) -> dict:
    """Rigid-align ``tgt_img`` onto ``ref_img`` using cached detections.

    ``step1`` picks the anchor for the pure-translation pre-alignment:

      - ``"glint"`` (default): translate so the mean glint centroids match.
        Requires ``glints`` in both detections.
      - ``"pupil"``: translate so the pupil centres match. Requires
        ``pupil_center`` or ``pupil_ellipse`` in both detections.
      - ``None``: skip the translation step.

    ``step2`` controls the iris-texture refinement:

      - ``True`` (default): minimise mean absolute intensity difference
        inside an iris-barrel mask, searching over ``(dx, dy, theta)``.
        Requires a non-``None`` ``limbus`` in both detections.
      - ``False``: skip the refinement.

    Returns::

        {
            "aligned":            <warped target image, same shape as ref_img>,
            "step1_translation":  (dx, dy)        or None when step1 is None,
            "step2_transform":    (dx, dy, theta) or None when step2 is False,
            "rotation_center":    (cx, cy)       or None when step2 is False,
        }
    """
    if step1 is None and not step2:
        return {
            "aligned": tgt_img.copy(),
            "step1_translation": None,
            "step2_transform": None,
            "rotation_center": None,
        }

    if step2 and (ref_det.get("limbus") is None or tgt_det.get("limbus") is None):
        raise ValueError("step2=True requires non-None limbus in both detections")

    # Step 1: pure-translation pre-alignment, anchor chosen by ``step1``.
    if step1 is None:
        p1: tuple[float, float, float] = (0.0, 0.0, 0.0)
        step1_out: tuple[float, float] | None = None
        warped = tgt_img.copy()
    else:
        if step1 == "glint":
            ref_point = _glint_centroid(ref_det)
            tgt_point = _glint_centroid(tgt_det)
        elif step1 == "pupil":
            ref_point = _pupil_center(ref_det)
            tgt_point = _pupil_center(tgt_det)
        else:
            raise ValueError(f"step1 must be 'glint', 'pupil', or None — got {step1!r}")
        p1 = align_by_translation(ref_point, tgt_point)
        step1_out = (float(p1[0]), float(p1[1]))

    # Step 2 needs a rotation centre derived from the limbus; compute that
    # only when we actually plan to run the refinement.
    rotation_center: tuple[int, int] | None = None
    if step2:
        ref_lim = ref_det["limbus"]
        tgt_lim = tgt_det["limbus"]
        cx = (ref_lim["center"][0] + tgt_lim["center"][0]) / 2
        cy = (ref_lim["center"][1] + tgt_lim["center"][1]) / 2
        center = (round(cx), round(cy))
        rotation_center = (int(center[0]), int(center[1]))
    else:
        center = None

    # Apply step 1 (or skip if step1 is None) before deciding what to do for step 2.
    if step1 is not None:
        warped = apply_transform(tgt_img, p1, center=center)

    # Step 2: iris-barrel min-diff refinement.
    if step2:
        iris_r = round((float(ref_det["limbus"]["radius"]) + float(tgt_det["limbus"]["radius"])) / 2)
        pupil_r = round(max(_pupil_radius(ref_det), _pupil_radius(tgt_det)))
        barrel = make_barrel_mask(ref_img.shape, center, iris_r, pupil_r)
        p2, _ = align_by_min_diff(ref_img, warped, barrel, rotation_center=center)
        aligned = apply_transform(warped, p2, center=center)
        step2_out: tuple[float, float, float] | None = (float(p2[0]), float(p2[1]), float(p2[2]))
    else:
        aligned = warped
        step2_out = None

    return {
        "aligned": aligned,
        "step1_translation": step1_out,
        "step2_transform": step2_out,
        "rotation_center": rotation_center,
    }
