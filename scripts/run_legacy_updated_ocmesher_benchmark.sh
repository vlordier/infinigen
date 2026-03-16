#!/usr/bin/env bash
set -euo pipefail

# Runs benchmark twice:
# 1) legacy: current local OcMesher as-is
# 2) updated: swaps in vlordier/OcMesher:develop, requires full infinigen mode
# Restores original OcMesher tree afterwards.
#
# Usage:
#   scripts/run_legacy_updated_ocmesher_benchmark.sh /path/to/scene.blend [benchmark args]

if [ $# -lt 1 ]; then
  echo "Usage: $0 /path/to/scene.blend [benchmark args]" >&2
  exit 2
fi

SCENE_PATH="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNNER="$REPO_ROOT/scripts/run_filter_populate_benchmark.sh"
OCPATH="$REPO_ROOT/infinigen/OcMesher"
TMPROOT="/tmp/infinigen_ocmesher_bench"
LEGACY_JSON="$TMPROOT/legacy.json"
UPDATED_JSON="$TMPROOT/updated.json"
COMBINED_JSON="$TMPROOT/legacy_vs_updated.json"

mkdir -p "$TMPROOT"

if [ ! -x "$RUNNER" ]; then
  echo "Runner not executable: $RUNNER" >&2
  exit 3
fi
if [ ! -d "$OCPATH" ]; then
  echo "OcMesher path not found: $OCPATH" >&2
  exit 3
fi

EXTRA_ARGS=("$@")

echo "[1/4] Running legacy benchmark (current OcMesher)"
"$RUNNER" "$SCENE_PATH" --run-label legacy --output "$LEGACY_JSON" "${EXTRA_ARGS[@]}"

BACKUP="$TMPROOT/OcMesher.backup"
CLONE="$TMPROOT/OcMesher.develop"
rm -rf "$BACKUP" "$CLONE"
cp -R "$OCPATH" "$BACKUP"

cleanup() {
  if [ -d "$BACKUP" ]; then
    rm -rf "$OCPATH"
    mv "$BACKUP" "$OCPATH"
  fi
}
trap cleanup EXIT

echo "[2/4] Cloning vlordier/OcMesher:develop"
git clone --depth 1 --branch develop https://github.com/vlordier/OcMesher.git "$CLONE"

if [ ! -f "$CLONE/ocmesher/__init__.py" ]; then
  echo "Cloned OcMesher does not look valid: missing ocmesher/__init__.py" >&2
  exit 4
fi

# Infinigen currently expects OcMesher version string "2.0".
# If the develop branch advertises a different version, apply a benchmark-only
# compatibility shim so imports proceed and we can compare runtime behavior.
if ! rg -n "__version__\s*=\s*\"2\.0\"" "$CLONE/ocmesher/__init__.py" >/dev/null 2>&1; then
  python3 - <<'PY' "$CLONE/ocmesher/__init__.py"
from pathlib import Path
import re
import sys

p = Path(sys.argv[1])
txt = p.read_text(encoding="utf-8")
txt2 = re.sub(r'__version__\s*=\s*"[^"]+"', '__version__ = "2.0"', txt)
if txt == txt2:
    txt2 = txt + '\n__version__ = "2.0"\n'
p.write_text(txt2, encoding="utf-8")
print("Applied benchmark version shim to", p)
PY
fi

echo "[3/4] Swapping in updated OcMesher and running updated benchmark"
rm -rf "$OCPATH"
mkdir -p "$OCPATH"
cp -R "$CLONE"/. "$OCPATH"/

"$RUNNER" "$SCENE_PATH" --run-label updated --require-full-infinigen --output "$UPDATED_JSON" "${EXTRA_ARGS[@]}"

cleanup
trap - EXIT

echo "[4/4] Combining reports"
python3 - <<'PY' "$LEGACY_JSON" "$UPDATED_JSON" "$COMBINED_JSON"
import json
import sys

legacy_path, updated_path, out_path = sys.argv[1:4]
with open(legacy_path, "r", encoding="utf-8") as f:
    legacy = json.load(f)
with open(updated_path, "r", encoding="utf-8") as f:
    updated = json.load(f)

combined = {
    "legacy": legacy,
    "updated": updated,
    "updated_vs_legacy_runtime_ratio": (
        updated["batched_seconds_median"] / legacy["legacy_seconds_median"]
        if legacy.get("legacy_seconds_median", 0) else None
    ),
    "updated_batched_vs_legacy_batched_speedup": (
        legacy["batched_seconds_median"] / updated["batched_seconds_median"]
        if updated.get("batched_seconds_median", 0) else None
    ),
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(combined, f, indent=2)

print(f"legacy_output={legacy_path}")
print(f"updated_output={updated_path}")
print(f"combined_output={out_path}")
print(f"legacy_mode={legacy.get('mode')}, legacy_torch_backend={legacy.get('torch_backend')}")
print(f"updated_mode={updated.get('mode')}, updated_torch_backend={updated.get('torch_backend')}")
print(f"legacy_speedup_batched_over_legacy={legacy.get('speedup_batched_over_legacy')}")
print(f"updated_speedup_batched_over_legacy={updated.get('speedup_batched_over_legacy')}")
PY
