"""``cheshm-gui`` CLI entry point: launch the Dear PyGui tuning workbench."""

import argparse
from pathlib import Path

from .app import run


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``cheshm-gui`` script."""
    parser = argparse.ArgumentParser(
        prog="cheshm-gui",
        description="Interactive workbench for tuning cheshm detectors on a directory of eye images.",
    )
    parser.add_argument(
        "dir",
        type=Path,
        nargs="?",
        default=None,
        help="Optional directory of eye images (PNG / JPG / TIFF / BMP). "
        "If omitted, open a folder or add files from the GUI.",
    )
    args = parser.parse_args(argv)
    run(args.dir)
