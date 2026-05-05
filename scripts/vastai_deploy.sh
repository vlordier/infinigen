#!/bin/bash
# VAST AI deployment setup script — runs on the GPU instance
# Installs Blender, clones infinigen, installs deps, runs demo, uploads to S3

set -e

echo "=== INFINIGEN VISPOS DEPLOYMENT ==="
echo "Started at $(date)"

# === Configuration ===
S3_BUCKET="s3://infinigen-gnss"
AWS_REGION="us-east-1"
BLENDER_VERSION="5.1.1"
REPO_URL="https://github.com/vlordier/infinigen.git"
BRANCH="main"

# === Install Blender ===
echo "[1/6] Installing Blender..."
if ! command -v blender &> /dev/null; then
    apt-get update -qq
    apt-get install -y -qq wget libxrender1 libxi6 libgl1 libxkbcommon0
    wget -q "https://download.blender.org/release/Blender5.1/blender-${BLENDER_VERSION}-linux-x64.tar.xz"
    tar -xf "blender-${BLENDER_VERSION}-linux-x64.tar.xz"
    mv "blender-${BLENDER_VERSION}-linux-x64" /opt/blender
    ln -sf /opt/blender/blender /usr/local/bin/blender
fi
echo "  Blender: $(blender --version 2>/dev/null | head -1 || echo 'OK')"

# === Install Blender Python dependencies ===
echo "[2/6] Installing Python dependencies..."
BLENDER_PY=$(find /opt/blender -name "python3.*" -path "*/bin/*" | head -1)
$BLENDER_PY -m pip install -q gin-config numpy trimesh psutil OpenEXR

# === Clone infinigen ===
echo "[3/6] Cloning infinigen..."
cd /workspace
if [ ! -d infinigen ]; then
    git clone $REPO_URL -b $BRANCH
fi
cd infinigen
git pull origin $BRANCH

# === Run demo generation ===
echo "[4/6] Running demo generation..."
mkdir -p output/demo
blender --background --python tests/demo_generation.py

# === Run smoke tests ===
echo "[5/6] Running smoke tests..."
blender --background --python tests/smoke_test_vispos.py

# === Upload to S3 ===
echo "[6/6] Uploading to S3..."
aws s3 sync output/ ${S3_BUCKET}/vispos_demo_$(date +%Y%m%d_%H%M%S)/ --region $AWS_REGION --no-progress

echo "=== DEPLOYMENT COMPLETE ==="
echo "Output: ${S3_BUCKET}/vispos_demo_$(date +%Y%m%d_%H%M%S)/"
