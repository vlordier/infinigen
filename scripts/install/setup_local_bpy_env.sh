#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-.venv}"
BPY_VERSION="${BPY_VERSION:-5.0.1}"
BPY_INDEX_URL="${BPY_INDEX_URL:-https://download.blender.org/pypi/}"
INSTALL_EDITABLE="${INSTALL_EDITABLE:-1}"
INSTALL_EXTRAS="${INSTALL_EXTRAS:-dev}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "error: required interpreter not found: $PYTHON_BIN" >&2
  echo "hint: install Python 3.11 or run with PYTHON_BIN=/path/to/python3.11" >&2
  exit 2
fi

echo "Creating virtual environment at $VENV_DIR using $PYTHON_BIN"
"$PYTHON_BIN" -m venv "$VENV_DIR"

PY="$VENV_DIR/bin/python"

echo "Upgrading packaging tools"
"$PY" -m pip install --upgrade pip setuptools wheel

echo "Installing bpy==$BPY_VERSION from $BPY_INDEX_URL"
"$PY" -m pip install "bpy==$BPY_VERSION" --extra-index-url "$BPY_INDEX_URL"

if [ "$INSTALL_EDITABLE" = "1" ]; then
  if [ -n "$INSTALL_EXTRAS" ]; then
    echo "Installing editable repo with extras: [$INSTALL_EXTRAS]"
    "$PY" -m pip install -e ".[${INSTALL_EXTRAS}]" --extra-index-url "$BPY_INDEX_URL"
  else
    echo "Installing editable repo without extras"
    "$PY" -m pip install -e . --extra-index-url "$BPY_INDEX_URL"
  fi
fi

echo "Verifying runtime"
"$PY" -c "import bpy, sys; print('python', sys.version.split()[0]); print('blender', bpy.app.version_string); print('version_tuple', bpy.app.version)"
