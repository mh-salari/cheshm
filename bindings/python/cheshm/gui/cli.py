"""``cheshm-gui`` CLI entry point: launch the Dear PyGui tuning workbench."""

import argparse
from pathlib import Path

from .app import run


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``cheshm-gui`` script."""
    parser = argparse.ArgumentParser(
        prog="cheshm-gui",
        description="Interactive workbench for tuning cheshm detectors on eye images.",
    )
    parser.add_argument(
        "paths",
        type=Path,
        nargs="*",
        help="Either a directory of eye images, a single image, or multiple "
        "image paths. If omitted, open a folder or add files from the GUI.",
    )
    args = parser.parse_args(argv)
    paths: list[Path] = args.paths

    if not paths:
        run(None)
    elif len(paths) == 1:
        run(paths[0])
    else:
        run(paths)
