"""Interactive matplotlib GUI for tuning pupil + glint detection per image.

Given a list of grayscale eye images and an output JSON path, opens a
window with sliders for ``pupil_threshold``, ``glint_threshold``,
``search_radius_factor`` (plus a display-brightness slider that does
not affect detection), navigates Prev / Next through the images, and
persists per-image thresholds to JSON. Saved thresholds are reloaded
on startup so a previous session can be resumed.

The detection pipeline is :func:`detect_pupil` → :func:`detect_glints`;
the glint search disk is centred on the detected pupil and its radius
is ``search_radius_factor * pupil_radius``.

Keys: ``←`` / ``→`` previous / next, ``s`` save.
"""

import json
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import KeyEvent
from matplotlib.widgets import Button, CheckButtons, Slider

from .core import detect_glints, detect_pupil

DEFAULT_PUPIL_THRESHOLD = 25
DEFAULT_GLINT_THRESHOLD = 200
DEFAULT_SEARCH_RADIUS_FACTOR = 2.0


def load_thresholds(thresholds_path: Path) -> dict[str, dict]:
    """Return the previously saved thresholds (empty dict if file is missing/invalid)."""
    if not thresholds_path.exists():
        return {}
    try:
        return json.loads(thresholds_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_thresholds(thresholds_path: Path, thresholds: dict[str, dict]) -> None:
    """Write the thresholds map sorted by key for stable diffs."""
    ordered = {k: thresholds[k] for k in sorted(thresholds)}
    thresholds_path.write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8")


PUPIL_MASK_RGBA = (0, 200, 220, 64)
GLINT_MASK_RGBA = (255, 0, 200, 110)


def _mask_to_rgba(mask: np.ndarray, color: tuple[int, int, int, int]) -> np.ndarray:
    """Turn a uint8 0/255 mask into a transparent RGBA image painted ``color``."""
    h, w = mask.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    inside = mask > 0
    r, g, b, a = color
    rgba[inside, 0] = r
    rgba[inside, 1] = g
    rgba[inside, 2] = b
    rgba[inside, 3] = a
    return rgba


def _draw_detection(
    ax: Axes,
    img: np.ndarray,
    pupil_thr: int,
    glint_thr: int,
    search_radius_factor: float,
    display_max: int,
    *,
    show_overlays: bool,
    show_pupil_mask: bool,
    show_glint_mask: bool,
) -> None:
    """Update ``ax`` with the image + live detection overlay (no clear → keeps zoom/pan).

    Image artists past index 0 (the base eye image) are removed every
    redraw so previous mask overlays do not pile up. Markers and ellipse
    patches go through ``ax.patches`` / ``ax.lines`` / ``ax.texts``,
    which we also clear before re-rendering.
    """
    if ax.images:
        ax.images[0].set_data(img)
        ax.images[0].set_clim(0, display_max)
    else:
        ax.imshow(img, cmap="gray", vmin=0, vmax=display_max)
        ax.axis("off")
        ax.set_aspect("equal")
    while len(ax.images) > 1:
        ax.images[-1].remove()
    for artist in list(ax.patches) + list(ax.lines) + list(ax.texts):
        artist.remove()
    if not show_overlays:
        return

    pupil = detect_pupil(img, pupil_threshold=pupil_thr)
    if pupil is None:
        ax.text(
            0.5,
            0.05,
            "PUPIL DETECTION FAILED",
            color="red",
            transform=ax.transAxes,
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
        return

    if show_pupil_mask:
        ax.imshow(_mask_to_rgba(pupil["mask"], PUPIL_MASK_RGBA), interpolation="nearest")

    (cx, cy), (w, h), angle = pupil["ellipse"]
    loose_dash = (0, (8, 8))
    ax.add_patch(
        mpatches.Ellipse(
            (cx, cy),
            w,
            h,
            angle=angle,
            fill=False,
            edgecolor="red",
            linewidth=1.0,
            alpha=0.85,
            linestyle=loose_dash,
        ),
    )
    ax.plot(cx, cy, "x", color="red", markersize=8, markeredgewidth=2)

    pupil_radius = max(w, h) / 2.0
    glints_result = detect_glints(
        img,
        pupil_center=pupil["center"],
        pupil_radius=pupil_radius,
        glint_threshold=glint_thr,
        search_radius_factor=search_radius_factor,
    )
    if show_glint_mask:
        ax.imshow(_mask_to_rgba(glints_result["search_area"], GLINT_MASK_RGBA), interpolation="nearest")
    for g in glints_result["glints"]:
        gx, gy = g["center"]
        ax.plot(gx, gy, "o", color="lime", markersize=5, markerfacecolor="none", markeredgewidth=1.5)


def tune_thresholds(
    images: list[Path],
    output_path: Path,
    keys: list[str] | None = None,
) -> None:
    """Launch the threshold-tuning GUI for ``images`` and persist to ``output_path``.

    ``images`` is an ordered list of grayscale eye-image PNGs. ``keys`` are
    the keys to use in the output JSON (one per image, same order); if
    ``None``, the image filename is used. Previously-saved thresholds at
    ``output_path`` are reloaded so a session can be resumed; entries in
    the file for images NOT in this session are preserved through the
    next save.
    """
    if not images:
        raise SystemExit("no images to tune.")
    if keys is None:
        keys = [p.name for p in images]
    if len(keys) != len(images):
        raise SystemExit(f"keys length {len(keys)} does not match images length {len(images)}.")

    cached_imgs: dict[str, cv2.typing.MatLike] = {}
    for key, image_path in zip(keys, images, strict=True):
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise SystemExit(f"failed to read {image_path}")
        cached_imgs[key] = img

    saved = load_thresholds(output_path)

    # Seed every image with previously-saved thresholds, falling back to defaults.
    # ``touched`` tracks images that already have real values so navigating to them
    # never overwrites those values via carry-forward.
    thresholds: dict[str, dict] = dict(saved)
    touched: set[str] = set()
    for k in keys:
        prev = saved.get(k)
        if prev:
            thresholds[k] = {
                "pupil_threshold": int(prev["pupil_threshold"]),
                "glint_threshold": int(prev["glint_threshold"]),
                "search_radius_factor": float(prev["search_radius_factor"]),
            }
            touched.add(k)
        else:
            thresholds[k] = {
                "pupil_threshold": DEFAULT_PUPIL_THRESHOLD,
                "glint_threshold": DEFAULT_GLINT_THRESHOLD,
                "search_radius_factor": DEFAULT_SEARCH_RADIUS_FACTOR,
            }

    state = {
        "pos": 0,
        "updating": False,
        "autosave": False,
        "overlays": True,
        "pupil_mask": False,
        "glint_mask": False,
    }

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(
        6,
        6,
        # The bottom row holds the 3-row "view" CheckButtons widget
        # (Overlays / Pupil mask / Glint mask); 1.5 is roughly three
        # times the previous single-row allocation so the rows have
        # readable vertical spacing without overlapping labels.
        height_ratios=[10.0, 0.22, 0.22, 0.22, 0.45, 1.5],
        hspace=0.28,
        wspace=0.18,
        left=0.06,
        right=0.97,
        top=0.96,
        bottom=0.04,
    )
    ax_img = fig.add_subplot(gs[0, :])
    single_row = [0.8, 1.4, 0.8]
    dual_row = [0.3, 0.55, 0.15, 0.3, 0.55, 0.15]
    ax_pupil = fig.add_subplot(gs[1, :].subgridspec(1, 3, width_ratios=single_row, wspace=0.0)[0, 1])
    glint_row = gs[2, :].subgridspec(1, 6, width_ratios=dual_row, wspace=0.0)
    ax_glint = fig.add_subplot(glint_row[0, 1])
    ax_radius = fig.add_subplot(glint_row[0, 4])
    ax_disp = fig.add_subplot(gs[3, :].subgridspec(1, 3, width_ratios=single_row, wspace=0.0)[0, 1])
    ax_status = fig.add_subplot(gs[4, :])
    ax_status.axis("off")
    status_text = ax_status.text(
        0.5,
        0.5,
        "",
        ha="center",
        va="center",
        fontsize=10,
        transform=ax_status.transAxes,
        family="monospace",
    )
    ax_prev = fig.add_subplot(gs[5, 0])
    ax_next = fig.add_subplot(gs[5, 1])
    ax_save = fig.add_subplot(gs[5, 2])
    ax_autosave = fig.add_subplot(gs[5, 3])
    ax_view = fig.add_subplot(gs[5, 4:6])
    # CheckButtons axes carry a default rectangular frame around their
    # label column; strip it so only the tick boxes and labels show.
    for ax in (ax_autosave, ax_view):
        ax.set_frame_on(False)

    init = thresholds[keys[0]]
    s_pupil = Slider(ax_pupil, "pupil_thr", 0, 255, valinit=init["pupil_threshold"], valstep=1)
    s_glint = Slider(ax_glint, "glint_thr", 0, 255, valinit=init["glint_threshold"], valstep=1)
    s_radius = Slider(ax_radius, "search_r x pupil", 0.1, 5.0, valinit=init["search_radius_factor"], valstep=0.1)
    s_disp = Slider(ax_disp, "display_max", 50, 255, valinit=255, valstep=1)
    btn_prev = Button(ax_prev, "< Prev")
    btn_next = Button(ax_next, "Next >")
    btn_save = Button(ax_save, "Save")
    chk_autosave = CheckButtons(ax_autosave, ["Autosave"], [False])
    chk_view = CheckButtons(ax_view, ["Overlays", "Pupil mask", "Glint mask"], [True, False, False])

    def cur_key() -> str:
        return keys[state["pos"]]

    def refresh_status(extra: str = "") -> None:
        k = cur_key()
        thr = thresholds[k]
        head = (
            f"{state['pos'] + 1}/{len(keys)}  {k}\n"
            f"pupil_thr={thr['pupil_threshold']:3d}  glint_thr={thr['glint_threshold']:3d}  "
            f"search_r={thr['search_radius_factor']:.1f}"
        )
        status_text.set_text(head + (("\n" + extra) if extra else ""))

    def redraw(image_changed: bool = False) -> None:
        k = cur_key()
        img = cached_imgs[k]
        thr = thresholds[k]
        if image_changed:
            for artist in list(ax_img.images):
                artist.remove()
        _draw_detection(
            ax_img,
            img,
            thr["pupil_threshold"],
            thr["glint_threshold"],
            thr["search_radius_factor"],
            display_max=int(s_disp.val),
            show_overlays=state["overlays"],
            show_pupil_mask=state["pupil_mask"],
            show_glint_mask=state["glint_mask"],
        )
        fig.suptitle(f"{k}   [{state['pos'] + 1}/{len(keys)}]", fontsize=12, fontweight="bold")
        refresh_status()
        fig.canvas.draw_idle()

    def sync_sliders_to_current() -> None:
        thr = thresholds[cur_key()]
        state["updating"] = True
        s_pupil.set_val(thr["pupil_threshold"])
        s_glint.set_val(thr["glint_threshold"])
        s_radius.set_val(thr["search_radius_factor"])
        state["updating"] = False

    def on_slider(_: object) -> None:
        if state["updating"]:
            return
        k = cur_key()
        thr = thresholds[k]
        thr["pupil_threshold"] = int(s_pupil.val)
        thr["glint_threshold"] = int(s_glint.val)
        thr["search_radius_factor"] = float(s_radius.val)
        touched.add(k)
        if state["autosave"]:
            save_thresholds(output_path, thresholds)
        redraw()

    def on_disp(_: object) -> None:
        redraw()

    def carry_forward(src_key: str, dst_key: str) -> None:
        """Carry-forward only into images that have never been touched.

        Images already loaded from JSON or modified by the user keep their values.
        """
        if dst_key not in touched:
            thresholds[dst_key] = dict(thresholds[src_key])

    def on_prev(_: object = None) -> None:
        if state["pos"] == 0:
            return
        prev_key = cur_key()
        state["pos"] -= 1
        carry_forward(prev_key, cur_key())
        if state["autosave"]:
            save_thresholds(output_path, thresholds)
        sync_sliders_to_current()
        redraw(image_changed=True)

    def on_next(_: object = None) -> None:
        if state["pos"] >= len(keys) - 1:
            return
        prev_key = cur_key()
        state["pos"] += 1
        carry_forward(prev_key, cur_key())
        if state["autosave"]:
            save_thresholds(output_path, thresholds)
        sync_sliders_to_current()
        redraw(image_changed=True)

    def on_save(_: object = None) -> None:
        save_thresholds(output_path, thresholds)
        refresh_status(f"saved -> {output_path.name}")
        fig.canvas.draw_idle()

    def on_autosave(_label: str) -> None:
        state["autosave"] = chk_autosave.get_status()[0]
        if state["autosave"]:
            save_thresholds(output_path, thresholds)
            refresh_status(f"autosave ON   saved -> {output_path.name}")
        else:
            refresh_status("autosave OFF")
        fig.canvas.draw_idle()

    def on_view(_label: str) -> None:
        status = chk_view.get_status()
        state["overlays"] = status[0]
        state["pupil_mask"] = status[1]
        state["glint_mask"] = status[2]
        redraw()

    def on_key(event: KeyEvent) -> None:
        if event.key == "right":
            on_next()
        elif event.key == "left":
            on_prev()
        elif event.key == "s":
            on_save()

    s_pupil.on_changed(on_slider)
    s_glint.on_changed(on_slider)
    s_radius.on_changed(on_slider)
    s_disp.on_changed(on_disp)
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    btn_save.on_clicked(on_save)
    chk_autosave.on_clicked(on_autosave)
    chk_view.on_clicked(on_view)
    fig.canvas.mpl_connect("key_press_event", on_key)

    redraw(image_changed=True)
    plt.show()
