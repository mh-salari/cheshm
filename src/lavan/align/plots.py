"""Matplotlib helpers for visualising eye-image alignment results.

  - :func:`plot_diff` — absolute-difference heatmap between two grayscale images.
  - :func:`plot_blend` — averaged blend of two grayscale images.
  - :func:`plot_mask_overlay` — grayscale image with a colour tint on masked pixels.
  - :func:`save_aligned_pair_images` — 4-panel diff QC PNG plus a blend overlay PNG
    for one (reference, aligned target) pair.
  - :func:`save_diff_heatmap` — pixel-clean colour-mapped diff PNG (no axes, no
    titles), sized to the input arrays.

All four plotting helpers accept either ``ax=None`` (the function creates its
own ``matplotlib.figure.Figure``) or a caller-provided axis. The image-saving
:func:`save_aligned_pair_images` uses ``matplotlib.figure.Figure`` directly
instead of pyplot so it can be called from inside another figure's GUI
callback without touching the pyplot figure manager (a macOS-backend
segfault hazard).
"""

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import imsave as _mpl_imsave


def _finalize(
    fig: Figure,
    ax: Axes,
    save: str | Path | None,
    show: bool,
    formats: Iterable[str],
) -> Axes:
    """Handle save/show for standalone plots."""
    fig.tight_layout()
    if save:
        base = str(save).rsplit(".", 1)[0] if "." in str(save) else str(save)
        for fmt in formats:
            kwargs: dict[str, object] = {"bbox_inches": "tight"}
            if fmt == "png":
                kwargs["dpi"] = 300
            fig.savefig(f"{base}.{fmt}", **kwargs)
    if show:
        plt.show()
    return ax


def plot_diff(
    img1: np.ndarray,
    img2: np.ndarray,
    title: str = "",
    vmin: float = 0,
    vmax: float = 80,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (5, 3),
    save: str | Path | None = None,
    show: bool = True,
    formats: Iterable[str] = ("png", "svg"),
) -> Axes:
    """Show absolute difference heatmap between two grayscale images."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    ax.imshow(diff, cmap="hot", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    if standalone:
        return _finalize(fig, ax, save, show, formats)
    return ax


def plot_blend(
    img1: np.ndarray,
    img2: np.ndarray,
    title: str = "",
    ax: Axes | None = None,
    figsize: tuple[float, float] = (5, 3),
    save: str | Path | None = None,
    show: bool = True,
    formats: Iterable[str] = ("png", "svg"),
) -> Axes:
    """Show averaged blend of two grayscale images."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    blend = (img1.astype(np.float32) + img2.astype(np.float32)) / 2
    ax.imshow(blend.astype(np.uint8), cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    if standalone:
        return _finalize(fig, ax, save, show, formats)
    return ax


def plot_mask_overlay(
    img: np.ndarray,
    mask: np.ndarray,
    title: str = "",
    color: tuple[int, int, int] = (0, 80, 0),
    alpha: float = 0.5,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (5, 3),
    save: str | Path | None = None,
    show: bool = True,
    formats: Iterable[str] = ("png", "svg"),
) -> Axes:
    """Show grayscale image with a colour tint on masked pixels."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, cmap="gray")
    overlay = np.zeros((*img.shape[:2], 4), dtype=np.float32)
    r, g, b = color
    overlay[mask > 0] = [r / 255, g / 255, b / 255, alpha]
    ax.imshow(overlay)
    ax.set_title(title)
    ax.axis("off")
    if standalone:
        return _finalize(fig, ax, save, show, formats)
    return ax


def save_aligned_pair_images(
    out_dir: Path,
    prefix: str,
    dil_img: np.ndarray,
    con_img: np.ndarray,
    aligned: np.ndarray,
    *,
    diff_dir: Path | None = None,
    overlay_dir: Path | None = None,
    ref_label: str = "dilated",
    target_label: str = "constricted (aligned)",
) -> None:
    """Write two PNGs that visualise one (reference, aligned target) pair.

    - ``<prefix>_diff.png`` — 4-panel: reference, target (aligned), diff (before
      alignment), diff (after alignment).
    - ``<prefix>_overlay.png`` — blend of reference + aligned target.

    Pass ``diff_dir`` / ``overlay_dir`` to write the two PNGs to separate
    sub-directories; otherwise both land in ``out_dir``. ``ref_label`` /
    ``target_label`` control the panel titles for use cases other than the
    classic dilated/constricted pair.
    """
    diff_dir = diff_dir or out_dir
    overlay_dir = overlay_dir or out_dir

    fig = Figure(figsize=(16, 4))
    axes = fig.subplots(1, 4)
    axes[0].imshow(dil_img, cmap="gray")
    axes[0].set_title(ref_label)
    axes[0].axis("off")
    axes[1].imshow(aligned, cmap="gray")
    axes[1].set_title(target_label)
    axes[1].axis("off")
    plot_diff(dil_img, con_img, title="diff (before)", ax=axes[2])
    plot_diff(dil_img, aligned, title="diff (after)", ax=axes[3])
    fig.suptitle(prefix, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(diff_dir / f"{prefix}_diff.png", dpi=120, bbox_inches="tight")

    fig2 = Figure(figsize=(5, 4))
    ax2 = fig2.subplots()
    plot_blend(dil_img, aligned, title=f"{prefix} overlay", ax=ax2)
    fig2.tight_layout()
    fig2.savefig(overlay_dir / f"{prefix}_overlay.png", dpi=120, bbox_inches="tight")


def save_diff_heatmap(
    out_path: str | Path,
    ref: np.ndarray,
    aligned: np.ndarray,
    *,
    vmax: float | None = None,
    cmap: str = "hot",
) -> float:
    """Write ``|ref - aligned|`` as a colour-mapped PNG sized to the inputs.

    ``vmax=None`` uses the per-image 99th percentile. Returns the value
    actually used so callers can record it.
    """
    if ref.shape != aligned.shape:
        raise ValueError(f"ref shape {ref.shape} differs from aligned shape {aligned.shape}")
    diff = np.abs(ref.astype(np.float32) - aligned.astype(np.float32))
    if vmax is None:
        vmax = max(float(np.percentile(diff, 99)), 1.0)
    _mpl_imsave(str(out_path), diff, cmap=cmap, vmin=0.0, vmax=float(vmax), format="png")
    return float(vmax)
