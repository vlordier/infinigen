#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${1:-.venv/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "error: python executable not found: $PYTHON_BIN" >&2
  echo "hint: create a Python 3.11 venv first, e.g. uv venv --python 3.11" >&2
  exit 2
fi

"$PYTHON_BIN" - <<'PY'
import sys

print(f"python: {sys.version.split()[0]}")
try:
    import bpy
except Exception as exc:
    print(f"bpy: not importable ({exc})")
    raise SystemExit(1)

print(f"bpy.app.version: {bpy.app.version}")
print(f"bpy.app.version_string: {bpy.app.version_string}")
if bpy.app.version[:2] != (5, 0):
    print("warning: expected Blender 5.0.x runtime for this repo")
PY
