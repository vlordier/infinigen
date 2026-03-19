#!/bin/bash
# Generate and export 5 different Infinigen scenes for Genesis rendering
# Each scene will be generated with coarse+populate+fine_terrain+render, then exported to USD
# Uses EEVEE rendering

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

# Use venv python directly (not launch_blender - that's for Blender Python script install)
PYTHON_CMD=".venv/bin/python"

# Scene configurations: (scene_type, output_folder, seed)
declare -a SCENES=(
    "desert:outputs/genesis_desert:0"
    "forest:outputs/genesis_forest:1"
    "arctic:outputs/genesis_arctic:2"
    "canyon:outputs/genesis_canyon:3"
    "coral_reef:outputs/genesis_coral_reef:4"
)

for scene_config in "${SCENES[@]}"; do
    IFS=':' read -r scene_type output_folder seed <<< "$scene_config"
    
    echo ""
    echo "=========================================="
    echo "Processing: $scene_type -> $output_folder (seed=$seed)"
    echo "=========================================="
    
    # Step 1: Coarse terrain generation
    echo "[1/5] Generating coarse terrain for $scene_type..."
    $PYTHON_CMD -m infinigen_examples.generate_nature --seed "$seed" --task coarse -g "${scene_type}.gin" base.gin --output_folder "${output_folder}/coarse"
    
    # Step 2: Populate assets
    echo "[2/5] Populating assets for $scene_type..."
    $PYTHON_CMD -m infinigen_examples.generate_nature --seed "$seed" --task populate -g "${scene_type}.gin" base.gin --input_folder "${output_folder}/coarse" --output_folder "${output_folder}/fine"
    
    # Step 3: Fine terrain
    echo "[3/5] Fine terrain processing for $scene_type..."
    $PYTHON_CMD -m infinigen_examples.generate_nature --seed "$seed" --task fine_terrain -g "${scene_type}.gin" base.gin --input_folder "${output_folder}/coarse" --output_folder "${output_folder}/fine"
    
    # Step 4: Render with EEVEE
    echo "[4/5] Rendering with EEVEE for $scene_type..."
    $PYTHON_CMD -m infinigen_examples.generate_nature --seed "$seed" --task render -g "${scene_type}.gin" base.gin eevee_annotations.gin --input_folder "${output_folder}/fine" --output_folder "${output_folder}/frames"
    
    # Step 5: Export to USD for Genesis
    echo "[5/5] Exporting to USD for $scene_type..."
    $PYTHON_CMD -m infinigen_examples.generate_nature --seed "$seed" --task export -g "${scene_type}.gin" base.gin --input_folder "${output_folder}/fine" --output_folder "${output_folder}/export"
    
    echo "✓ Completed: $scene_type"
    echo "  Frames: ${output_folder}/frames/"
    echo "  Export: ${output_folder}/export/"
    ls -la "${output_folder}/frames/" 2>/dev/null | head -5 || echo "  (checking frames folder...)"
    
done

echo ""
echo "=========================================="
echo "All 5 scenes exported successfully!"
echo "=========================================="
echo "Output folders:"
for scene_config in "${SCENES[@]}"; do
    IFS=':' read -r scene_type output_folder seed <<< "$scene_config"
    echo "  - $scene_type: $output_folder/ (frames + export)"
done
