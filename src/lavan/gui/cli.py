"""``lavan-gui`` CLI entry point: launch the Dear PyGui tuning workbench."""

import argparse
from pathlib import Path

from .app import run


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``lavan-gui`` script."""
    parser = argparse.ArgumentParser(
        prog="lavan-gui",
        description="Interactive workbench for tuning lavan detectors on a directory of eye images.",
    )
    parser.add_argument("dir", type=Path, help="Directory of eye images (PNG / JPG / TIFF / BMP).")
    args = parser.parse_args(argv)
    run(args.dir)
