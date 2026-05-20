"""Dear PyGui visual workbench for tuning lavan detectors on eye images.

Three-panel layout: left = image browser, centre = image canvas with
detection overlays, right = per-detector settings panels built from the
:mod:`lavan.gui.registry` output.

Launch via ``lavan-gui`` (CLI) or :func:`run`.
"""

from .app import run

__all__ = ["run"]
