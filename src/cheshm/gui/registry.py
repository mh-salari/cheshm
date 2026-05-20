"""Auto-discover detectors and expose their UI metadata.

Walks ``cheshm.{pupil,glint,limbus}_detectors`` and produces a list of
:class:`Detector` descriptors. Each descriptor combines:

  - data introspected from the function (defaults, type, choices) via
    :func:`inspect.signature` and :func:`typing.get_type_hints`,
  - per-parameter UI metadata read from the module's ``_UI`` dict
    (slider bounds, help text, widget hints, label overrides).

Nothing here is hand-registered — adding a new detector module is the
only step needed to surface it in the GUI.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Union, get_args, get_origin, get_type_hints

# Each top-level category: parent package + the conventional detect
# function name to look for inside each leaf module.
_CATEGORIES = {
    "pupil": ("cheshm.pupil_detectors", "detect_pupil"),
    "glint": ("cheshm.glint_detectors", "detect_glints"),
    "limbus": ("cheshm.limbus_detectors", "detect_limbus"),
    "eyelid": ("cheshm.eyelid_detectors", "detect_eyelid"),
}


@dataclass
class Setting:
    """A single tunable parameter of a detector, ready to bind to a widget."""

    name: str
    type: str
    default: Any
    label: str
    help: str = ""
    choices: list[str] = field(default_factory=list)
    min: float | None = None
    max: float | None = None
    hidden: bool = False  # set via _UI[name]["hidden"]; the GUI skips hidden settings


@dataclass
class Detector:
    """A discovered detector: its identity, wiring requirements, and settings."""

    kind: str
    family: str
    id: str
    name: str
    description: str
    module: str
    function: Callable
    wired_inputs: list[str]
    settings: list[Setting]
    overlays: tuple[tuple[str, str], ...] = ()  # (overlay_key, element_type) declared by _OVERLAYS


def _auto_label(name: str) -> str:
    """``snake_case`` → ``"Snake case"``. Override via ``_UI[name]["label"]``.

    Only the first character is upper-cased; the rest is preserved verbatim
    so proper-noun detector names like ``Swirski2D`` survive intact.
    """
    label = name.replace("_", " ").strip()
    return label[:1].upper() + label[1:] if label else label


def _is_optional(annotation: Any) -> tuple[bool, Any]:
    """Detect ``X | None`` / ``Optional[X]`` and return ``(True, X)``; else ``(False, None)``."""
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        args = get_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1 and len(args) == len(non_none) + 1:
            return True, non_none[0]
    return False, None


def _infer_type_and_choices(annotation: Any, ui_meta: dict) -> tuple[str, list[str]]:
    """Map a parameter's annotation to a UI type tag (+ choices list for ``choice``)."""
    if ui_meta.get("widget") == "roi":
        return "roi", []

    if get_origin(annotation) is Literal:
        return "choice", [str(c) for c in get_args(annotation)]

    is_opt, base = _is_optional(annotation)
    if is_opt:
        if base is int:
            return "optional_int", []
        if base is float:
            return "optional_float", []
        if get_origin(base) is tuple:
            # `tuple[int, int, int, int] | None` with no roi widget hint —
            # fall through to "any".
            pass

    if annotation is int:
        return "int", []
    if annotation is float:
        return "float", []
    if annotation is bool:
        return "bool", []

    return "any", []


def _build_setting(name: str, param: inspect.Parameter, annotation: Any, ui_meta: dict) -> Setting:
    ttype, choices = _infer_type_and_choices(annotation, ui_meta)
    return Setting(
        name=name,
        type=ttype,
        default=param.default,
        label=ui_meta.get("label", _auto_label(name)),
        help=ui_meta.get("help", ""),
        choices=choices,
        min=ui_meta.get("min"),
        max=ui_meta.get("max"),
        hidden=bool(ui_meta.get("hidden", False)),
    )


def _describe_function(fn: Callable, ui_table: dict) -> tuple[list[str], list[Setting]]:
    """Split a detector function's parameters into wired inputs vs. settings.

    A *wired input* is a positional parameter with no default — the GUI
    fills it from upstream (the loaded image, the pupil centre, etc.).
    Everything else is a setting bound to a widget.
    """
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn, include_extras=True)
    except Exception:
        hints = {}

    wired: list[str] = []
    settings: list[Setting] = []
    positional_kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY)
    for pname, param in sig.parameters.items():
        is_wired = param.default is inspect.Parameter.empty and param.kind in positional_kinds
        if is_wired:
            wired.append(pname)
            continue
        annot = hints.get(pname, param.annotation)
        settings.append(_build_setting(pname, param, annot, ui_table.get(pname, {})))
    return wired, settings


def discover_detectors() -> list[Detector]:
    """Walk every detector category and return one :class:`Detector` per leaf package.

    Only packages are considered (``ispkg=True``), matching cheshm's
    convention that every detector is a subpackage with ``__init__.py``
    re-exporting from ``core.py``. ``_UI`` metadata is read from the
    ``core`` submodule, where it lives next to the function definition.
    """
    detectors: list[Detector] = []
    for kind, (parent_pkg, fn_name) in _CATEGORIES.items():
        parent = importlib.import_module(parent_pkg)
        for module_info in pkgutil.walk_packages(parent.__path__, prefix=f"{parent_pkg}."):
            if not module_info.ispkg:
                continue
            try:
                mod = importlib.import_module(module_info.name)
            except ImportError:
                continue
            fn = getattr(mod, fn_name, None)
            if not callable(fn):
                continue

            try:
                core_mod = importlib.import_module(f"{module_info.name}.core")
                ui_table = getattr(core_mod, "_UI", {})
                overlays = tuple(getattr(core_mod, "_OVERLAYS", ()))
            except ImportError:
                ui_table = {}
                overlays = ()

            parts = module_info.name.removeprefix(f"{parent_pkg}.").split(".")
            detector_id = parts[-1]
            family = parts[-2] if len(parts) > 1 else parts[-1]

            wired, settings = _describe_function(fn, ui_table)
            description = (fn.__doc__ or "").strip().split("\n", 1)[0]

            detectors.append(
                Detector(
                    kind=kind,
                    family=family,
                    id=detector_id,
                    name=_auto_label(detector_id),
                    description=description,
                    module=module_info.name,
                    function=fn,
                    wired_inputs=wired,
                    settings=settings,
                    overlays=overlays,
                ),
            )
    return detectors
