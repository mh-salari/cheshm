"""Dear PyGui workbench for tuning lavan detectors on a folder of eye images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import dearpygui.dearpygui as dpg
import numpy as np

from .registry import Detector, Setting, discover_detectors

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Element-type defaults applied to every (kind, overlay_key).
_TYPE_DEFAULT_ALPHA = {"line": 1.0, "point": 1.0, "fill": 0.5}
_TYPE_DEFAULT_THICKNESS = {"line": 1, "point": 1, "fill": 0}

# cv2 BGR colour per element keyed semantic (overlay_key). Distinct
# hue per kind is set per (kind, key) below.
_DEFAULT_COLOURS_BY_FULL_KEY: dict[str, tuple[int, int, int]] = {
    "pupil_contour": (0, 255, 0),
    "pupil_ellipse": (0, 255, 255),
    "pupil_center": (0, 255, 0),
    "pupil_mask": (0, 200, 0),
    "glint_contour": (0, 0, 255),
    "glint_center": (0, 0, 255),
    "glint_mask": (0, 60, 200),
    "limbus_curve": (255, 0, 255),
    "limbus_center": (255, 0, 255),
    "limbus_mask": (255, 0, 255),
}

# Mask overlays start off; everything else on.
_DEFAULT_SHOW_BY_TYPE = {"line": True, "point": True, "fill": False}


def _initial_overlay_state(detectors: list[Detector]) -> tuple[dict, dict, dict, dict]:
    """Pre-populate per-element state for every overlay any detector could emit."""
    show, colors, alpha, thickness = {}, {}, {}, {}
    for det in detectors:
        for key, elem_type in det.overlays:
            full = f"{det.kind}_{key}"
            show.setdefault(full, _DEFAULT_SHOW_BY_TYPE.get(elem_type, True))
            colors.setdefault(full, _DEFAULT_COLOURS_BY_FULL_KEY.get(full, (255, 255, 255)))
            alpha.setdefault(full, _TYPE_DEFAULT_ALPHA.get(elem_type, 1.0))
            thickness.setdefault(full, _TYPE_DEFAULT_THICKNESS.get(elem_type, 1))
    return show, colors, alpha, thickness


class _State:
    """Application state mutated by widget callbacks."""

    def __init__(self, img_dir: Path) -> None:
        self.img_dir = img_dir
        self.images: list[Path] = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
        if not self.images:
            raise SystemExit(f"no images with extensions {_IMAGE_EXTS} in {img_dir}")
        self.idx = 0
        self.zoom = 1.0
        self.brightness = 0
        self.detectors = discover_detectors()
        self.by_kind: dict[str, list[Detector]] = {
            k: [d for d in self.detectors if d.kind == k] for k in ("pupil", "glint", "limbus")
        }
        # Overlay state is keyed by f"{kind}_{overlay_key}" and seeded
        # from the union of every detector's declared _OVERLAYS.
        (self.show, self.colors, self.alpha,
         self.thickness) = _initial_overlay_state(self.detectors)
        # Detection-result cache. Re-runs detection only when the
        # detection-affecting inputs (image path, active detectors,
        # detector settings) change — overlay show/colour/alpha/thickness
        # tweaks just re-render from the cached result.
        self.last_detection_sig: tuple | None = None
        self.last_detected: tuple[dict | None, dict | None, dict | None] = (None, None, None)
        self.active: dict[str, str | None] = {
            "pupil": self.by_kind["pupil"][0].id if self.by_kind["pupil"] else None,
            "glint": None,
            "limbus": None,
        }
        self.values: dict[str, dict[str, dict[str, Any]]] = {
            kind: {det.id: {s.name: s.default for s in det.settings} for det in detectors}
            for kind, detectors in self.by_kind.items()
        }

    @property
    def current_path(self) -> Path:
        return self.images[self.idx]

    def detector(self, kind: str, det_id: str | None) -> Detector | None:
        if det_id is None:
            return None
        return next((d for d in self.by_kind[kind] if d.id == det_id), None)

    def kwargs_for(self, kind: str, det_id: str | None) -> dict[str, Any]:
        return dict(self.values.get(kind, {}).get(det_id, {})) if det_id else {}


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _apply_brightness(img: np.ndarray, beta: int) -> np.ndarray:
    if beta == 0:
        return img
    return cv2.convertScaleAbs(img, alpha=1.0, beta=float(beta))


def _detection_signature(state: _State) -> tuple:
    """Hashable signature of every input that affects detection output.

    Overlay state (show / colour / alpha / thickness) is intentionally
    excluded — tweaking those should never re-run detection.
    """
    values_sig = tuple(
        (kind, det_id, tuple(sorted(settings.items())))
        for kind, det_dict in state.values.items()
        for det_id, settings in sorted(det_dict.items())
    )
    return (
        str(state.current_path),
        tuple(sorted(state.active.items())),
        values_sig,
    )


def _run_detections(state: _State, img: np.ndarray) -> tuple[dict | None, dict | None, dict | None]:
    pupil_det = state.detector("pupil", state.active["pupil"])
    glint_det = state.detector("glint", state.active["glint"])
    limbus_det = state.detector("limbus", state.active["limbus"])

    pupil = None
    if pupil_det is not None:
        try:
            pupil = pupil_det.function(img, **state.kwargs_for("pupil", pupil_det.id))
        except Exception as exc:
            print(f"[pupil] {exc}")

    glints = None
    if glint_det is not None:
        kwargs = state.kwargs_for("glint", glint_det.id)
        kwargs.pop("pupil_center", None)
        kwargs.pop("pupil_radius", None)
        # Auto-wire pupil info when available; the detector itself
        # handles the no-pupil case (whole-image search).
        if pupil is not None and pupil.get("ellipse") is not None:
            (_, _), (w, h), _ = pupil["ellipse"]
            kwargs["pupil_center"] = pupil["center"]
            kwargs["pupil_radius"] = max(w, h) / 2.0
        try:
            glints = glint_det.function(img, **kwargs)
        except Exception as exc:
            print(f"[glint] {exc}")

    limbus = None
    if limbus_det is not None and pupil is not None:
        kwargs = state.kwargs_for("limbus", limbus_det.id)
        try:
            if "pupil_ellipse" in limbus_det.wired_inputs:
                limbus = limbus_det.function(img, pupil["center"], pupil["ellipse"], **kwargs)
            else:
                limbus = limbus_det.function(img, pupil["center"], **kwargs)
        except Exception as exc:
            print(f"[limbus] {exc}")

    return pupil, glints, limbus


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------


def _alpha_layer(canvas: np.ndarray, alpha: float, draw_fn) -> None:
    """Apply ``draw_fn(canvas)`` blended at ``alpha`` against the pre-draw canvas."""
    if alpha >= 0.999:
        draw_fn(canvas)
        return
    pre = canvas.copy()
    draw_fn(canvas)
    canvas[:] = cv2.addWeighted(pre, 1.0 - alpha, canvas, alpha, 0.0)


def _draw_polyline(canvas: np.ndarray, pts: np.ndarray, color, thickness: int) -> None:
    cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness)


def _draw_ellipse(canvas: np.ndarray, center, axes, angle: float, color, thickness: int) -> None:
    cv2.ellipse(canvas, center, axes, angle, 0, 360, color, thickness)


def _contour_pts(contour: np.ndarray) -> np.ndarray:
    return contour.reshape(-1, 2)


def _enabled_keys(state: _State, kind: str) -> set[str]:
    """Overlay keys declared by the currently-active detector for ``kind``."""
    det = state.detector(kind, state.active[kind])
    return {k for k, _ in (det.overlays if det else ())}


def _draw_pupil(canvas: np.ndarray, pupil: dict | None, state: _State) -> None:
    if pupil is None:
        return
    enabled = _enabled_keys(state, "pupil")

    if "mask" in enabled and state.show.get("pupil_mask") and pupil.get("contour") is not None:
        color = state.colors["pupil_mask"]
        _alpha_layer(canvas, state.alpha["pupil_mask"],
                     lambda c: cv2.drawContours(c, [pupil["contour"]], -1, color, thickness=-1))

    if "contour" in enabled and state.show.get("pupil_contour") and pupil.get("contour") is not None:
        color = state.colors["pupil_contour"]
        th = state.thickness["pupil_contour"]
        pts = _contour_pts(pupil["contour"])
        _alpha_layer(canvas, state.alpha["pupil_contour"],
                     lambda c: _draw_polyline(c, pts, color, th))

    if "ellipse" in enabled and state.show.get("pupil_ellipse") and pupil.get("ellipse") is not None:
        (cx, cy), (w, h), angle = pupil["ellipse"]
        color = state.colors["pupil_ellipse"]
        th = state.thickness["pupil_ellipse"]
        _alpha_layer(canvas, state.alpha["pupil_ellipse"],
                     lambda c: _draw_ellipse(
                         c, (round(cx), round(cy)),
                         (max(round(w / 2), 1), max(round(h / 2), 1)),
                         float(angle), color, th))

    if "center" in enabled and state.show.get("pupil_center") and pupil.get("center") is not None:
        cx, cy = pupil["center"]
        color = state.colors["pupil_center"]
        size = state.thickness["pupil_center"]
        _alpha_layer(canvas, state.alpha["pupil_center"],
                     lambda c: cv2.circle(c, (round(cx), round(cy)), size, color, thickness=-1))


def _draw_glints(canvas: np.ndarray, glints: dict | None, state: _State) -> None:
    if glints is None:
        return
    enabled = _enabled_keys(state, "glint")

    if "mask" in enabled and state.show.get("glint_mask"):
        contours = [g["contour"] for g in (glints.get("glints") or []) if g.get("contour") is not None]
        if contours:
            color = state.colors["glint_mask"]
            _alpha_layer(canvas, state.alpha["glint_mask"],
                         lambda c: cv2.drawContours(c, contours, -1, color, thickness=-1))

    for g in glints.get("glints", []) or []:
        if "contour" in enabled and state.show.get("glint_contour") and g.get("contour") is not None:
            color = state.colors["glint_contour"]
            th = state.thickness["glint_contour"]
            pts = _contour_pts(g["contour"])
            _alpha_layer(canvas, state.alpha["glint_contour"],
                         lambda c, p=pts: _draw_polyline(c, p, color, th))
        if "center" in enabled and state.show.get("glint_center") and g.get("center") is not None:
            gx, gy = g["center"]
            color = state.colors["glint_center"]
            size = state.thickness["glint_center"]
            _alpha_layer(canvas, state.alpha["glint_center"],
                         lambda c, x=gx, y=gy: cv2.circle(c, (round(x), round(y)), size, color, thickness=-1))


def _limbus_polygon(limbus: dict) -> np.ndarray | None:
    cx, cy = limbus["center"]
    if "R_theta" in limbus:
        xs = cx + limbus["R_theta"] * np.cos(limbus["thetas"])
        ys = cy + limbus["R_theta"] * np.sin(limbus["thetas"])
        return np.stack([xs, ys], axis=-1).astype(np.int32)
    return None


def _draw_limbus(canvas: np.ndarray, limbus: dict | None, state: _State) -> None:
    if limbus is None:
        return
    enabled = _enabled_keys(state, "limbus")
    cx, cy = limbus["center"]
    poly = _limbus_polygon(limbus)
    radius = int(limbus["radius"]) if "radius" in limbus else None

    if "mask" in enabled and state.show.get("limbus_mask"):
        color = state.colors["limbus_mask"]
        if poly is not None:
            _alpha_layer(canvas, state.alpha["limbus_mask"],
                         lambda c: cv2.fillPoly(c, [poly], color))
        elif radius is not None:
            _alpha_layer(canvas, state.alpha["limbus_mask"],
                         lambda c: cv2.circle(c, (round(cx), round(cy)), radius, color, thickness=-1))

    if "curve" in enabled and state.show.get("limbus_curve"):
        color = state.colors["limbus_curve"]
        th = state.thickness["limbus_curve"]
        if poly is not None:
            _alpha_layer(canvas, state.alpha["limbus_curve"],
                         lambda c: _draw_polyline(c, poly, color, th))
        elif radius is not None:
            _alpha_layer(canvas, state.alpha["limbus_curve"],
                         lambda c: _draw_ellipse(c, (round(cx), round(cy)),
                                                  (radius, radius), 0.0, color, th))

    if "center" in enabled and state.show.get("limbus_center"):
        color = state.colors["limbus_center"]
        size = state.thickness["limbus_center"]
        _alpha_layer(canvas, state.alpha["limbus_center"],
                     lambda c: cv2.circle(c, (round(cx), round(cy)), size, color, thickness=-1))


def _to_rgba_float(canvas_bgr: np.ndarray) -> np.ndarray:
    rgba = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGBA)
    return (rgba.astype(np.float32) / 255.0).flatten()


# ---------------------------------------------------------------------------
# Setting widgets
# ---------------------------------------------------------------------------


def _slider_int_bounds(setting: Setting, default: int) -> tuple[int, int]:
    lo = int(setting.min) if setting.min is not None else 0
    hi = int(setting.max) if setting.max is not None else max(lo + 1, int(default) * 4 + 1)
    return lo, max(hi, lo + 1)


def _slider_float_bounds(setting: Setting, default: float) -> tuple[float, float]:
    lo = float(setting.min) if setting.min is not None else 0.0
    hi = float(setting.max) if setting.max is not None else max(lo + 1.0, float(default) * 4.0 + 1.0)
    return lo, max(hi, lo + 1e-6)


_LABEL_WRAP = 320  # px; matches the inside of the settings panel


def _add_label(text: str) -> None:
    dpg.add_text(text, wrap=_LABEL_WRAP, color=(220, 220, 220))


def _build_setting_widget(state: _State, kind: str, det_id: str, setting: Setting, on_change) -> None:
    if setting.hidden:
        return
    label = setting.label
    value = state.values[kind][det_id][setting.name]
    user_data = (kind, det_id, setting.name)
    fill_width = -1  # dpg: fill remaining horizontal space

    if setting.type == "int":
        _add_label(label)
        lo, hi = _slider_int_bounds(setting, value if value is not None else 0)
        widget = dpg.add_slider_int(
            default_value=int(value), min_value=lo, max_value=hi, width=fill_width,
            callback=on_change, user_data=user_data,
        )
    elif setting.type == "float":
        _add_label(label)
        lo, hi = _slider_float_bounds(setting, value if value is not None else 0.0)
        widget = dpg.add_slider_float(
            default_value=float(value), min_value=lo, max_value=hi, width=fill_width,
            callback=on_change, user_data=user_data,
        )
    elif setting.type == "bool":
        widget = dpg.add_checkbox(label=label, default_value=bool(value),
                                   callback=on_change, user_data=user_data)
    elif setting.type == "choice":
        _add_label(label)
        widget = dpg.add_combo(
            items=setting.choices, default_value=str(value), width=fill_width,
            callback=on_change, user_data=user_data,
        )
    elif setting.type == "optional_int":
        enabled = value is not None
        slider_tag = dpg.generate_uuid()
        enabled_tag = dpg.add_checkbox(
            label=f"{label} (enable)", default_value=enabled,
            callback=on_change, user_data=("_opt_enable_int", *user_data, slider_tag),
        )
        lo, hi = _slider_int_bounds(setting, value if value is not None else 0)
        widget = dpg.add_slider_int(
            tag=slider_tag, default_value=int(value) if value is not None else lo,
            min_value=lo, max_value=hi, width=fill_width, enabled=enabled,
            callback=on_change, user_data=("_opt_value_int", *user_data, enabled_tag),
        )
    elif setting.type == "optional_float":
        enabled = value is not None
        slider_tag = dpg.generate_uuid()
        enabled_tag = dpg.add_checkbox(
            label=f"{label} (enable)", default_value=enabled,
            callback=on_change, user_data=("_opt_enable_float", *user_data, slider_tag),
        )
        lo, hi = _slider_float_bounds(setting, value if value is not None else 0.0)
        widget = dpg.add_slider_float(
            tag=slider_tag, default_value=float(value) if value is not None else lo,
            min_value=lo, max_value=hi, width=fill_width, enabled=enabled,
            callback=on_change, user_data=("_opt_value_float", *user_data, enabled_tag),
        )
    elif setting.type == "roi":
        _add_label(f"{label} (x, y, w, h)")
        roi = value if value is not None else (0, 0, 0, 0)
        with dpg.group(horizontal=True):
            for i in range(4):
                dpg.add_input_int(
                    default_value=int(roi[i]), width=70,
                    callback=on_change, user_data=("_roi_field", *user_data, i),
                )
        widget = None
    else:
        _add_label(f"{label}  (unsupported type {setting.type!r})")
        widget = None

    if setting.help and widget is not None:
        with dpg.tooltip(widget):
            dpg.add_text(setting.help, wrap=300)


def _on_setting_change(state: _State, redraw_cb):
    def callback(sender, app_data, user_data):
        tag = (
            user_data[0]
            if isinstance(user_data, tuple) and isinstance(user_data[0], str) and user_data[0].startswith("_")
            else None
        )
        if tag is None:
            kind, det_id, name = user_data
            state.values[kind][det_id][name] = app_data
        elif tag in ("_opt_enable_int", "_opt_enable_float"):
            _, kind, det_id, name, slider_tag = user_data
            if app_data:
                # Restore the slider's current value, or 0 if it was None.
                state.values[kind][det_id][name] = dpg.get_value(slider_tag)
            else:
                state.values[kind][det_id][name] = None
            dpg.configure_item(slider_tag, enabled=bool(app_data))
        elif tag in ("_opt_value_int", "_opt_value_float"):
            _, kind, det_id, name, enabled_tag = user_data
            if dpg.get_value(enabled_tag):
                state.values[kind][det_id][name] = app_data
        elif tag == "_roi_field":
            _, kind, det_id, name, idx = user_data
            current = list(state.values[kind][det_id].get(name) or (0, 0, 0, 0))
            current[idx] = int(app_data)
            state.values[kind][det_id][name] = tuple(current) if any(current) else None
        redraw_cb()

    return callback


# ---------------------------------------------------------------------------
# Right panel: overlay rows + detector picker + settings
# ---------------------------------------------------------------------------


def _build_overlay_row(state: _State, kind: str, redraw_cb) -> None:
    """Per-element overlay row, sourced from the active detector's ``_OVERLAYS``."""

    det = state.detector(kind, state.active[kind])
    if det is None or not det.overlays:
        return

    def on_toggle(sender, app_data, user_data) -> None:
        state.show[user_data] = bool(app_data)
        redraw_cb()

    def on_color(sender, app_data, user_data) -> None:
        r, g, b = float(app_data[0]), float(app_data[1]), float(app_data[2])
        # dpg returns colors as floats in [0, 1] in 2.x but as 0..255 in
        # 1.x — detect by magnitude and rescale.
        if max(r, g, b) <= 1.0:
            r, g, b = r * 255.0, g * 255.0, b * 255.0
        state.colors[user_data] = (int(b), int(g), int(r))
        redraw_cb()

    def on_alpha(sender, app_data, user_data) -> None:
        state.alpha[user_data] = float(app_data)
        redraw_cb()

    def on_thickness(sender, app_data, user_data) -> None:
        state.thickness[user_data] = max(1, int(app_data))
        redraw_cb()

    with dpg.collapsing_header(label="overlay", default_open=False):
        for key, elem_type in det.overlays:
            full = f"{kind}_{key}"
            with dpg.group(horizontal=True):
                dpg.add_checkbox(default_value=state.show.get(full, True),
                                  callback=on_toggle, user_data=full)
                dpg.add_text(key)
                b, g, r = state.colors.get(full, (255, 255, 255))
                dpg.add_color_edit(
                    default_value=[r, g, b], no_inputs=True, no_label=True, no_alpha=True,
                    callback=on_color, user_data=full,
                )
                dpg.add_slider_float(default_value=state.alpha.get(full, 1.0),
                                      min_value=0.0, max_value=1.0, width=70, format="α %.2f",
                                      callback=on_alpha, user_data=full)
                if elem_type in ("line", "point"):
                    dpg.add_input_int(default_value=state.thickness.get(full, 1),
                                       min_value=1, max_value=20, step=0, width=50,
                                       callback=on_thickness, user_data=full)


_SECTION_TITLE_COLOR = (160, 220, 255)


def _build_settings_panel(state: _State, on_change, redraw_cb) -> None:
    def on_pick(sender, app_data, user_data) -> None:
        kind = user_data
        state.active[kind] = None if app_data == "-off-" else app_data
        _refresh_settings_group(state, kind, on_change, redraw_cb, on_reset_cb=on_reset)
        redraw_cb()

    def on_reset(sender, app_data, user_data) -> None:
        kind, det_id = user_data
        det = state.detector(kind, det_id)
        if det is None:
            return
        # Detector settings → registry defaults.
        for s in det.settings:
            state.values[kind][det_id][s.name] = s.default
        # Overlay state (show / colour / alpha / thickness) → factory defaults.
        for key, elem_type in det.overlays:
            full = f"{kind}_{key}"
            state.show[full] = _DEFAULT_SHOW_BY_TYPE.get(elem_type, True)
            state.colors[full] = _DEFAULT_COLOURS_BY_FULL_KEY.get(full, (255, 255, 255))
            state.alpha[full] = _TYPE_DEFAULT_ALPHA.get(elem_type, 1.0)
            state.thickness[full] = _TYPE_DEFAULT_THICKNESS.get(elem_type, 1)
        _refresh_settings_group(state, kind, on_change, redraw_cb, on_reset_cb=on_reset)
        redraw_cb()

    card_tags: list[str] = []
    for i, kind in enumerate(("pupil", "glint", "limbus")):
        if i > 0:
            dpg.add_spacer(height=8)
        card_tag = f"section_card_{kind}"
        card_tags.append(card_tag)
        with dpg.child_window(tag=card_tag, border=True, auto_resize_y=True, width=-1):
            dpg.add_text(kind.upper(), color=_SECTION_TITLE_COLOR)
            dpg.add_separator()
            _add_label("detector")
            options = ["-off-"] + [d.id for d in state.by_kind[kind]]
            current = state.active[kind] or "-off-"
            dpg.add_combo(items=options, default_value=current, width=-1,
                          callback=on_pick, user_data=kind)
            dpg.add_group(tag=f"settings_group_{kind}")
            _refresh_settings_group(state, kind, on_change, redraw_cb, on_reset_cb=on_reset)

    with dpg.theme() as card_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, [56, 62, 78])
            dpg.add_theme_color(dpg.mvThemeCol_Border, [110, 130, 170])
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 8.0)
            dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 1.0)
    for t in card_tags:
        dpg.bind_item_theme(t, card_theme)


def _refresh_settings_group(state: _State, kind: str, on_change, redraw_cb, *, on_reset_cb=None) -> None:
    group_tag = f"settings_group_{kind}"
    dpg.delete_item(group_tag, children_only=True)
    det = state.detector(kind, state.active[kind])
    if det is None:
        return
    with dpg.group(parent=group_tag):
        dpg.add_spacer(height=4)
        # Overlay rows live here so they vanish when the detector is -off-.
        _build_overlay_row(state, kind, redraw_cb)
        dpg.add_separator()
        if on_reset_cb is not None:
            dpg.add_button(label="Reset settings to defaults", width=-1,
                           callback=on_reset_cb, user_data=(kind, det.id))
        dpg.add_spacer(height=4)
        for s in det.settings:
            _build_setting_widget(state, kind, det.id, s, on_change)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


_LEFT_W = 240
_RIGHT_W = 400
_SIDE_PAD = 40


def _fit_zoom(canvas_w: int) -> float:
    available = max(dpg.get_viewport_client_width() - _LEFT_W - _RIGHT_W - _SIDE_PAD, 200)
    return min(1.0, available / canvas_w)


def run(img_dir: str | Path) -> None:
    state = _State(Path(img_dir).expanduser().resolve())

    first = cv2.imread(str(state.current_path), cv2.IMREAD_GRAYSCALE)
    if first is None:
        raise SystemExit(f"failed to read {state.current_path}")
    canvas_h, canvas_w = first.shape

    dpg.create_context()
    with dpg.texture_registry():
        dpg.add_dynamic_texture(canvas_w, canvas_h,
                                _to_rgba_float(cv2.cvtColor(first, cv2.COLOR_GRAY2BGR)),
                                tag="image_texture")

    def redraw() -> None:
        img = cv2.imread(str(state.current_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return
        if img.shape != (canvas_h, canvas_w):
            img = cv2.resize(img, (canvas_w, canvas_h), interpolation=cv2.INTER_AREA)
        # Brightness is display-only; detectors see the raw pixels.
        # Only re-run detection when the detection-affecting inputs change.
        sig = _detection_signature(state)
        if sig != state.last_detection_sig:
            state.last_detected = _run_detections(state, img)
            state.last_detection_sig = sig
        pupil, glints, limbus = state.last_detected
        display = _apply_brightness(img, state.brightness)
        canvas = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
        _draw_pupil(canvas, pupil, state)
        _draw_glints(canvas, glints, state)
        _draw_limbus(canvas, limbus, state)
        dpg.set_value("image_texture", _to_rgba_float(canvas))
        disp_w = int(canvas_w * state.zoom)
        disp_h = int(canvas_h * state.zoom)
        # Centre the image inside the panel when it fits; otherwise let
        # the panel's scrollbar handle navigation. On the very first
        # redraw the panel hasn't been laid out yet, so get_item_rect_size
        # returns 0 — fall back to estimating from the viewport.
        panel_w, panel_h = dpg.get_item_rect_size("image_panel")
        if panel_w <= 0 or panel_h <= 0:
            panel_w = max(dpg.get_viewport_client_width() - _LEFT_W - _RIGHT_W - 30, 200)
            panel_h = max(dpg.get_viewport_client_height() - 80, 200)
        pos_x = max((panel_w - disp_w) // 2, 0)
        pos_y = max((panel_h - disp_h) // 2, 0)
        dpg.configure_item("image_widget", width=disp_w, height=disp_h, pos=[pos_x, pos_y])
        dpg.set_value("status_text", f"{state.idx + 1}/{len(state.images)}  "
                                       f"{state.current_path.name}  "
                                       f"({canvas_w}×{canvas_h})  "
                                       f"zoom×{state.zoom:.2f}  β={state.brightness:+d}")

    on_change = _on_setting_change(state, redraw)

    def on_zoom(sender, app_data) -> None:
        state.zoom = float(app_data)
        redraw()

    def on_brightness(sender, app_data) -> None:
        state.brightness = int(app_data)
        redraw()

    def reset_zoom() -> None:
        state.zoom = _fit_zoom(canvas_w)
        dpg.set_value("zoom_slider", state.zoom)
        redraw()

    def reset_brightness() -> None:
        state.brightness = 0
        dpg.set_value("brightness_slider", 0)
        redraw()

    with dpg.window(label="lavan", tag="main_window"):
        dpg.add_text("", tag="status_text")
        with dpg.group(horizontal=True):
            dpg.add_button(label="zoom", small=True, callback=reset_zoom)
            dpg.add_slider_float(tag="zoom_slider", default_value=1.0,
                                  min_value=0.1, max_value=4.0, width=150, callback=on_zoom)
            dpg.add_spacer(width=10)
            dpg.add_button(label="brightness", small=True, callback=reset_brightness)
            dpg.add_slider_int(tag="brightness_slider", default_value=0,
                                min_value=-100, max_value=100, width=150, callback=on_brightness)

        with dpg.group(horizontal=True):
            with dpg.child_window(width=_LEFT_W, autosize_y=True):
                dpg.add_text(str(state.img_dir), wrap=220, color=(180, 180, 180))

                def on_prev() -> None:
                    if state.idx > 0:
                        state.idx -= 1
                        dpg.set_value("image_list", state.current_path.name)
                        redraw()

                def on_next() -> None:
                    if state.idx < len(state.images) - 1:
                        state.idx += 1
                        dpg.set_value("image_list", state.current_path.name)
                        redraw()

                with dpg.group(horizontal=True):
                    dpg.add_button(label="<", callback=on_prev)
                    dpg.add_button(label=">", callback=on_next)

                def on_pick_image(sender, app_data) -> None:
                    for i, p in enumerate(state.images):
                        if p.name == app_data:
                            state.idx = i
                            redraw()
                            break

                dpg.add_listbox(
                    items=[p.name for p in state.images],
                    default_value=state.current_path.name,
                    tag="image_list", num_items=20, width=220, callback=on_pick_image,
                )

            with dpg.child_window(tag="image_panel", width=-_RIGHT_W, autosize_y=True,
                                   horizontal_scrollbar=True):
                dpg.add_image("image_texture", tag="image_widget",
                              width=canvas_w, height=canvas_h)

            with dpg.child_window(tag="settings_panel", width=_RIGHT_W - 20, autosize_y=True,
                                   border=True):
                _build_settings_panel(state, on_change, redraw)

    with dpg.theme() as settings_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, [42, 46, 54])
            dpg.add_theme_color(dpg.mvThemeCol_Border, [120, 140, 170])
    dpg.bind_item_theme("settings_panel", settings_theme)

    dpg.create_viewport(title="lavan", width=1400, height=900)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)
    # Re-centre + refit the image whenever the viewport resizes (e.g.,
    # entering full-screen). Without this, the image stays put until
    # the next redraw is triggered by something else.
    dpg.set_viewport_resize_callback(lambda *_: redraw())
    state.zoom = _fit_zoom(canvas_w)
    dpg.set_value("zoom_slider", state.zoom)
    redraw()
    # After dpg has rendered a couple of frames the panel size resolves
    # to its real value; redraw once more so the centring is exact.
    dpg.set_frame_callback(3, lambda: redraw())
    dpg.start_dearpygui()
    dpg.destroy_context()
