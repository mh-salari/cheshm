"""Compile the 3 C kernels into the platform-specific shared library.

Run from the repo root:

    python scripts/build_c.py

Produces, next to each ``*_core.c`` file:

  - macOS  → ``*_core.dylib`` (via ``clang``)
  - Linux  → ``*_core.so``    (via ``cc`` / ``gcc``)
  - Windows → ``*_core.dll``  (via ``gcc`` on MinGW)

The compile commands are the same flags the GitHub Actions release workflow
uses, so a wheel built locally matches the one CI produces.
"""

import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = REPO_ROOT / "src" / "daugman_derived_boundary_detectors"

KERNELS = (
    "active_contour_core",
    "integro_differential_operator_core",
    "pupil_guided_contour_core",
)


def lib_extension() -> str:
    """Return the shared-library extension for the current OS."""
    system = platform.system()
    if system == "Darwin":
        return ".dylib"
    if system == "Linux":
        return ".so"
    if system == "Windows":
        return ".dll"
    raise SystemExit(f"unsupported platform: {system!r}")


def compile_command(c_file: Path, out_file: Path) -> list[str]:
    """Return the compiler argv for this OS."""
    system = platform.system()
    if system == "Darwin":
        return ["clang", "-O3", "-shared", "-fPIC", "-o", str(out_file), str(c_file), "-lm"]
    if system == "Linux":
        compiler = "cc" if shutil.which("cc") else "gcc"
        return [compiler, "-O3", "-shared", "-fPIC", "-o", str(out_file), str(c_file), "-lm"]
    if system == "Windows":
        # MinGW gcc; tested on the windows-latest GitHub Actions runner.
        return ["gcc", "-O3", "-shared", "-o", str(out_file), str(c_file), "-lm"]
    raise SystemExit(f"unsupported platform: {system!r}")


def main() -> int:
    """Compile every kernel for the current OS into ``PKG_DIR``."""
    ext = lib_extension()
    for kernel in KERNELS:
        c_file = PKG_DIR / f"{kernel}.c"
        out_file = PKG_DIR / f"{kernel}{ext}"
        if not c_file.exists():
            print(f"ERROR: missing C source: {c_file}", file=sys.stderr)
            return 1
        cmd = compile_command(c_file, out_file)
        print(f"building {kernel}{ext}")
        print("  $", " ".join(cmd))
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            return result.returncode
    print(f"\nBuilt 3 kernels into {PKG_DIR.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
