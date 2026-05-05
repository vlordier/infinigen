#!/bin/bash
# VAST AI deployment — builds marching cubes, runs infinigen, uploads to S3
set -e
exec > /workspace/deploy.log 2>&1
echo "=== INFINIGEN DEPLOY $(date) ==="

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git wget libxrender1 libxi6 libgl1 libxkbcommon0 libglib2.0-0

echo "[1] Installing Blender 5.1..."
wget -q https://download.blender.org/release/Blender5.1/blender-5.1.1-linux-x64.tar.xz
tar -xf blender-5.1.1-linux-x64.tar.xz
mv blender-5.1.1-linux-x64 /opt/blender
ln -sf /opt/blender/blender /usr/local/bin/blender

echo "[2] Installing Python deps..."
BPY=/opt/blender/5.1/python/bin/python3.13
$BPY -m pip install -q Cython gin-config numpy psutil OpenEXR trimesh shapely scipy scikit-image landlab

echo "[3] Cloning infinigen..."
cd /workspace
git clone --depth 1 https://github.com/vlordier/infinigen.git
cd infinigen

echo "[4] Building marching cubes (.so)..."
cd infinigen/terrain/marching_cubes
rm -f _marching_cubes_lewiner_cy.cpython-*.so
$BPY -m cython _marching_cubes_lewiner_cy.pyx -3
PY_INC=$(find / -name "Python.h" -path "*/python3.13/*" 2>/dev/null | head -1 | xargs dirname)
NP_INC=$($BPY -c "import numpy; print(numpy.get_include())")
gcc -shared -fPIC -I"$PY_INC" -I"$NP_INC" -O2 -o _marching_cubes_lewiner_cy.cpython-313-x86_64-linux-gnu.so _marching_cubes_lewiner_cy.c
ls -lh _marching_cubes_lewiner_cy*.so || echo "WARNING: .so build may have failed"
cd /workspace/infinigen

echo "[5] Smoke tests..."
blender --background --python tests/smoke_test_vispos.py 2>&1 | tail -5

echo "[6] Running coarse pipeline (forest)..."
INFINIGEN_OCMESHER_CLASS=infinigen.OcMesher.ocmesher.OcMesher blender --background --python infinigen_examples/generate_nature.py -- --seed 0 --task coarse -g infinigen_examples/configs_nature/scene_types/forest.gin -g infinigen_examples/configs_nature/base_nature.gin --output_folder /workspace/output 2>&1 | tail -10

echo "[7] Running our demo..."
blender --background --python tests/full_demo.py 2>&1 | tail -5
ls -la output/vispos_demo/

echo "[8] Uploading to S3..."
aws s3 sync output/ s3://infinigen-gnss/vispos_$(date +%Y%m%d_%H%M)/ --no-progress --region us-east-1 || echo "S3 upload skipped"

echo "=== DONE $(date) ==="
sleep 3600
