#!/bin/bash
# Build marching cubes Cython extension for the current platform
set -e
cd "$(dirname "$0")/infinigen/terrain/marching_cubes"

PYTHON_BIN=$(dirname $(which blender 2>/dev/null || which python3))/python3
[ ! -f "$PYTHON_BIN" ] && PYTHON_BIN=$(find /opt/blender -name "python3.*" -path "*/bin/*" | head -1)
[ ! -f "$PYTHON_BIN" ] && { echo "No Python found"; exit 1; }

echo "Python: $PYTHON_BIN"
$PYTHON_BIN -m pip install -q Cython numpy 2>/dev/null

PY_INC=$($PYTHON_BIN -c "import sysconfig; print(sysconfig.get_config_var('INCLUDEPY'))")
NP_INC=$($PYTHON_BIN -c "import numpy; print(numpy.get_include())")

# Try full headers first, fall back to any python3.13 headers
if [ ! -f "$PY_INC/Python.h" ]; then
    PY_INC=$(find / -name "Python.h" -path "*/python3.13/*" 2>/dev/null | head -1 | xargs dirname)
    [ ! -d "$PY_INC" ] && { echo "No Python.h found"; exit 1; }
fi

echo "Python include: $PY_INC"
echo "Numpy include: $NP_INC"

# Remove old builds
rm -f _marching_cubes_lewiner_cy.cpython-*.so

# Cythonize
$PYTHON_BIN -m cython _marching_cubes_lewiner_cy.pyx -3

# Compile
gcc -shared -fPIC \
  -I"$PY_INC" \
  -I"$NP_INC" \
  -O2 \
  -undefined dynamic_lookup \
  -o _marching_cubes_lewiner_cy.cpython-313-darwin.so \
  _marching_cubes_lewiner_cy.c 2>/dev/null \
|| gcc -shared -fPIC \
  -I"$PY_INC" \
  -I"$NP_INC" \
  -O2 \
  -o _marching_cubes_lewiner_cy.cpython-313-x86_64-linux-gnu.so \
  _marching_cubes_lewiner_cy.c

ls -lh _marching_cubes_lewiner_cy*.so && echo "BUILD OK"
