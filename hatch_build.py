"""Hatch custom build hook: compile the 3 C kernels and include them in the wheel.

When ``python -m build --wheel`` runs (locally or under cibuildwheel on
CI), this hook:

1. Detects the host OS, picks the right shared-library extension
   (``.dylib`` / ``.so`` / ``.dll``).
2. Invokes ``scripts/build_c.py`` if the compiled binary isn't already
   present (avoids re-compiling during editable installs where the user
   has run the script manually).
3. Adds the compiled binary to ``build_data["force_include"]`` so it
   lands inside the wheel next to the Python sources.
4. Sets ``pure_python = False`` so hatchling tags the wheel with the
   correct platform (``macosx_*_arm64``, ``linux_x86_64``,
   ``win_amd64`` etc.) rather than the misleading ``py3-none-any``.

The 3 ``*_core.c`` files have no third-party C dependencies (just
``libm``), so the build is self-contained and doesn't need a CMake setup.
"""

import platform
import subprocess
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

KERNELS = (
    "active_contour_core",
    "integro_differential_operator_core",
    "pupil_guided_contour_core",
)


def _lib_extension() -> str:
    system = platform.system()
    if system == "Darwin":
        return ".dylib"
    if system == "Linux":
        return ".so"
    if system == "Windows":
        return ".dll"
    raise RuntimeError(f"unsupported platform: {system!r}")


class CKernelBuildHook(BuildHookInterface):
    """Compile the 3 C kernels into the platform-specific shared library and bundle it."""

    PLUGIN_NAME = "custom"

    def initialize(self, _version: str, build_data: dict) -> None:
        """Compile any missing kernel binaries and register them with the wheel."""
        ext = _lib_extension()
        pkg_dir = Path(self.root) / "src" / "lavan" / "boundary"

        missing = [k for k in KERNELS if not (pkg_dir / f"{k}{ext}").exists()]
        if missing:
            self.app.display_info(
                f"compiling {len(missing)} C kernel(s) for {platform.system()}: {missing}",
            )
            result = subprocess.run(
                [sys.executable, str(Path(self.root) / "scripts" / "build_c.py")],
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(f"scripts/build_c.py failed with exit code {result.returncode}")

        # Tell hatchling to ship the freshly-compiled binary in the wheel.
        build_data.setdefault("force_include", {}).update({
            str(pkg_dir / f"{kernel}{ext}"): f"lavan/boundary/{kernel}{ext}" for kernel in KERNELS
        })

        # The wheel contains a compiled binary, so it is NOT pure Python.
        # This makes hatchling tag it with the correct platform tag.
        build_data["pure_python"] = False
        build_data["infer_tag"] = True
