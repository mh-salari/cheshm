"""Matplotlib QC helper: overlay detected pupil and glints on the eye image."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


def _ellipse_patch(
    ellipse: tuple[tuple[float, float], tuple[float, float], float],
    color: str,
    linewidth: float = 1,
    alpha: float = 0.7,
) -> Ellipse:
    """Create a matplotlib Ellipse patch from cv2.fitEllipse output: ((cx, cy), (w, h), angle)."""
    (cx, cy), (w, h), angle = ellipse
    return plt.matplotlib.patches.Ellipse(
        (cx, cy), w, h, angle=angle, fill=False, edgecolor=color, linewidth=linewidth, alpha=alpha
    )


def plot_detections(img: np.ndarray, detections: dict, title: str = "") -> None:
    """Plot the eye image with detected pupil and glint contours + fitted ellipses."""
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.imshow(img, cmap="gray")

    # Pupil contour (faint)
    pupil_polygon = plt.Polygon(
        detections["pupil_contour"].squeeze(), fill=False, edgecolor="lime", linewidth=1, alpha=0.2
    )
    ax.add_patch(pupil_polygon)

    # Pupil ellipse
    if detections["pupil_ellipse"] is not None:
        ax.add_patch(_ellipse_patch(detections["pupil_ellipse"], color="cyan"))

    # Pupil centers
    px, py = detections["pupil_center"]
    ax.plot(px, py, "o", color="lime", markersize=3, label="centroid")
    if detections["pupil_ellipse"] is not None:
        ex, ey = detections["pupil_ellipse"][0]
        ax.plot(ex, ey, "x", color="cyan", markersize=4, label="ellipse center")

    # Glints
    for g in detections["glints"]:
        glint_polygon = plt.Polygon(g["contour"].squeeze(), fill=False, edgecolor="red", linewidth=1, alpha=0.2)
        ax.add_patch(glint_polygon)

        if g["ellipse"] is not None:
            ax.add_patch(_ellipse_patch(g["ellipse"], color="orange"))
            ex, ey = g["ellipse"][0]
            ax.plot(ex, ey, "x", color="orange", markersize=3)

        ax.plot(g["center"][0], g["center"][1], "o", color="red", markersize=2, alpha=0.5)

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    plt.show()
