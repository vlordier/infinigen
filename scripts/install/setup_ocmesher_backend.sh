#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

OCPATH="${INFINIGEN_OCMESHER_PATH:-$REPO_ROOT/infinigen/OcMesher}"
OCMESHER_REPO_URL="${INFINIGEN_OCMESHER_REPO_URL:-https://github.com/vlordier/OcMesher.git}"
OCMESHER_BRANCH="${INFINIGEN_OCMESHER_BRANCH:-develop}"
OCMESHER_MODE="${INFINIGEN_OCMESHER_MODE:-torch}"
OCMESHER_INSTALL_RUST="${INFINIGEN_OCMESHER_INSTALL_RUST:-1}"
OCMESHER_PIP_EDITABLE="${INFINIGEN_OCMESHER_PIP_EDITABLE:-1}"

if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
  PYTHON_BIN="${INFINIGEN_OCMESHER_PYTHON:-$VIRTUAL_ENV/bin/python}"
elif [ -x "$REPO_ROOT/.venv/bin/python" ]; then
  PYTHON_BIN="${INFINIGEN_OCMESHER_PYTHON:-$REPO_ROOT/.venv/bin/python}"
else
  PYTHON_BIN="${INFINIGEN_OCMESHER_PYTHON:-python3}"
fi

if ! command -v git >/dev/null 2>&1; then
  echo "error: git is required" >&2
  exit 2
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "error: python not found: $PYTHON_BIN" >&2
  exit 2
fi

echo "[ocmesher] target path: $OCPATH"
echo "[ocmesher] source: ${OCMESHER_REPO_URL} (${OCMESHER_BRANCH})"

if [ ! -d "$OCPATH/.git" ]; then
  rm -rf "$OCPATH"
  git clone --depth 1 --branch "$OCMESHER_BRANCH" "$OCMESHER_REPO_URL" "$OCPATH"
else
  git -C "$OCPATH" fetch origin "$OCMESHER_BRANCH" --depth 1
  git -C "$OCPATH" checkout -B "$OCMESHER_BRANCH" "origin/$OCMESHER_BRANCH"
fi

if [ "$OCMESHER_PIP_EDITABLE" = "1" ]; then
  echo "[ocmesher] installing Python package (editable)"
  "$PYTHON_BIN" -m pip install -e "$OCPATH"
else
  echo "[ocmesher] installing Python package"
  "$PYTHON_BIN" -m pip install "$OCPATH"
fi

if [ "$OCMESHER_MODE" = "rust" ] && [ "$OCMESHER_INSTALL_RUST" = "1" ]; then
  if ! command -v cargo >/dev/null 2>&1; then
    echo "error: rust mode requested but cargo not found; install Rust toolchain first" >&2
    exit 2
  fi

  echo "[ocmesher] building Rust extension via maturin"
  "$PYTHON_BIN" -m pip install maturin
  (
    cd "$OCPATH/ocmesher-rust"
    PYO3_PYTHON="$PYTHON_BIN" maturin develop --release
  )
fi

echo "[ocmesher] done"
echo "[ocmesher] suggested runtime class:"
case "$OCMESHER_MODE" in
  rust)
    echo "  export INFINIGEN_OCMESHER_CLASS=ocmesher.RustOcMesher"
    ;;
  torch)
    echo "  export INFINIGEN_OCMESHER_CLASS=ocmesher.TorchOcMesher"
    ;;
  *)
    echo "  export INFINIGEN_OCMESHER_CLASS=ocmesher.OcMesher"
    ;;
esac
echo "[ocmesher] device hint (Apple Silicon): export INFINIGEN_OCMESHER_DEVICE=mps"
