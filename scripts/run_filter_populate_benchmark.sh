#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/run_filter_populate_benchmark.sh /path/to/scene.blend [extra benchmark args]
#
# Example:
#   scripts/run_filter_populate_benchmark.sh scene.blend --repeat 5 --output /tmp/filter_populate_targets_bench.json

if [ $# -lt 1 ]; then
  echo "Usage: $0 /path/to/scene.blend [extra benchmark args]" >&2
  exit 2
fi

SCENE_PATH="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_SITE_PACKAGES="$(ls -d "$REPO_ROOT"/.venv/lib/python*/site-packages 2>/dev/null | head -n 1 || true)"

if [ ! -f "$SCENE_PATH" ]; then
  echo "Scene file not found: $SCENE_PATH" >&2
  exit 2
fi

find_blender() {
  if command -v blender >/dev/null 2>&1; then
    command -v blender
    return 0
  fi

  if [ -x "/Applications/Blender.app/Contents/MacOS/Blender" ]; then
    echo "/Applications/Blender.app/Contents/MacOS/Blender"
    return 0
  fi

  if [ -x "$HOME/Applications/Blender.app/Contents/MacOS/Blender" ]; then
    echo "$HOME/Applications/Blender.app/Contents/MacOS/Blender"
    return 0
  fi

  return 1
}

BLENDER_BIN="$(find_blender || true)"
if [ -z "$BLENDER_BIN" ]; then
  echo "Could not find Blender binary. Install Blender or add it to PATH." >&2
  exit 3
fi

OUTPUT="/tmp/filter_populate_targets_bench.json"
EXTRA_ARGS=("$@")

for ((i=0; i<${#EXTRA_ARGS[@]}; i++)); do
  if [ "${EXTRA_ARGS[$i]}" = "--output" ] && [ $((i+1)) -lt ${#EXTRA_ARGS[@]} ]; then
    OUTPUT="${EXTRA_ARGS[$((i+1))]}"
  fi
done

if [ -n "$VENV_SITE_PACKAGES" ]; then
  export PYTHONPATH="$REPO_ROOT:$VENV_SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
else
  export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
fi

"$BLENDER_BIN" -b "$SCENE_PATH" --python-use-system-env \
  --python "$REPO_ROOT/scripts/benchmark_filter_populate_targets.py" -- \
  "${EXTRA_ARGS[@]}"

if [ ! -f "$OUTPUT" ]; then
  echo "Benchmark output file was not created: $OUTPUT" >&2
  exit 4
fi

python3 - <<'PY' "$OUTPUT"
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

speedup = data.get("speedup_batched_over_legacy")
legacy = data.get("legacy_seconds_median")
batched = data.get("batched_seconds_median")
print(f"output={path}")
print(f"legacy_seconds_median={legacy}")
print(f"batched_seconds_median={batched}")
print(f"speedup_batched_over_legacy={speedup}")
PY
