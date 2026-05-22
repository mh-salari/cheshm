#!/usr/bin/env bash
# Regenerate Python type stubs (_core.pyi) from compiled binding modules.
# Run after changing any binding's signature and commit the updated .pyi
# alongside the .cpp.
set -euo pipefail

cd "$(dirname "$0")/.."

# uv pip needs a venv to operate. Create one if none is active.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  uv venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

uv pip install --editable . --reinstall \
  --config-settings=cmake.define.CHESHM_BUILD_STUBS=ON
