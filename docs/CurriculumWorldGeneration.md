# Progressive 3D World Generation for Curriculum Learning

This guide explains how to use `infinigen.core.syndata` to generate 3D environments of increasing complexity — from trivial flight corridors to dense photorealistic worlds — for curriculum-based RL training.

## Quick Start

```python
from infinigen.core.syndata import WorldConfig, generate_world, world_gin_overrides

# Generate the simplest possible environment
cfg = WorldConfig.flappy(seed=42)
boxes = generate_world(cfg)          # list[BBox3D] — pure 3D geometry
overrides = world_gin_overrides(cfg) # Infinigen pipeline Gin parameters
```

## Architecture

This package controls **what Infinigen generates** — 3D geometry, textures, lighting, and materials. It does **not** run Blender, physics simulation, or RL training. Those responsibilities belong to other systems:

```
infinigen.core.syndata          Infinigen (Blender)         Genesis / DroneEnv
┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────┐
│ WorldConfig          │     │ Procedural 3D scene  │     │ Physics sim      │
│ generate_world()     │ ──► │ generation pipeline   │ ──► │ RL gym env       │
│ world_gin_overrides()│     │ (meshes, materials,   │     │ (obs, reward,    │
│ overlay_hints        │     │  textures, lighting)  │     │  done, actions)  │
└──────────────────────┘     └──────────────────────┘     └──────────────────┘
      Configuration               Asset production             Simulation
```

## Complexity Ladder: From Simplest to Most Complex

The entire progression is driven by a single `complexity` parameter in `[0, 1]`. Start at `0` for basic training and increase toward `1` for photorealistic environments.

### Stage 0 — Pre-training: "3D Flappy Bird" (before `WorldConfig`)

For the very first training steps, use `FlappyColumnConfig` to create an ultra-simple corridor with column-gap obstacles. This is even simpler than `WorldConfig(complexity=0)` — just columns with gaps.

```python
from infinigen.core.syndata import (
    FlappyColumnConfig, generate_flappy_obstacles, flappy_frame_metadata,
)

# Use preset difficulty levels:
easy_cfg = FlappyColumnConfig.easy()     # wide gaps, 3 columns, wide corridor
medium_cfg = FlappyColumnConfig.medium() # moderate gaps, 5 columns
hard_cfg = FlappyColumnConfig.hard()     # narrow gaps, 8 columns, tight corridor

# Generate obstacles from any preset
obstacles = generate_flappy_obstacles(easy_cfg, seed=42)
# Returns list[FlappyObstacle] with .center, .half_extents, .label

# Get metadata for validation / curriculum tracking
metadata = flappy_frame_metadata(easy_cfg, seed=42)
# Returns FrameMetadata-compatible dict with obstacles, depth_stats, etc.

# Or customise directly:
cfg = FlappyColumnConfig(
    corridor_length=8.0,
    corridor_width=2.0,
    corridor_height=2.0,
    num_columns=5,
    gap_height=0.6,
)
obstacles = generate_flappy_obstacles(cfg, seed=42)
```

**Preset progression** (use these to gradually increase flappy difficulty):

| Preset | Columns | Gap height | Corridor width | Difficulty |
|---|---|---|---|---|
| `FlappyColumnConfig.easy()` | 3 | 1.2 m | 3.0 m | Trivial |
| `FlappyColumnConfig.medium()` | 5 | 0.8 m | 2.0 m | Moderate |
| `FlappyColumnConfig.hard()` | 8 | 0.5 m | 1.5 m | Challenging |

**What the agent learns:** basic flight control — fly through gaps, avoid columns.

### Stage 1 — Flappy Preset (`complexity ≈ 0.05`)

Straight corridor with a few column-gap obstacles. Flat shading, no assets.

```python
from infinigen.core.syndata import WorldConfig, generate_world

cfg = WorldConfig.flappy(seed=42)
boxes = generate_world(cfg)
# ~10 BBox3D boxes: floor, ceiling, 2 walls + column obstacles
```

| Property | Value |
|---|---|
| Geometry | Straight corridor, ~10 boxes |
| Textures | 256px, flat shading |
| Lighting | Uniform |
| Assets | None — pure collision geometry |

**What the agent learns:** up/down, left/right, forward/back avoidance.

### Stage 2 — Textured Corridor (`complexity ≈ 0.25`)

Same topology, but with visual variety — coloured walls, varied floor textures, subtle fog.

```python
cfg = WorldConfig.corridor(seed=42)
boxes = generate_world(cfg)
# ~20 BBox3D boxes: more columns, tighter gaps
```

| Property | Value |
|---|---|
| Geometry | Corridor, ~20 boxes |
| Textures | 512px, basic PBR |
| Lighting | Single sun |
| Assets | Coloured PBR surfaces |

**What the agent learns:** vision robustness to surface appearance changes.

### Stage 3 — Indoor Rooms (`complexity ≈ 0.45`)

Connected rooms with doors and furniture-like clutter. Right-angle turns, doorway transitions.

```python
cfg = WorldConfig.rooms(seed=42)
boxes = generate_world(cfg)
# ~40–60 BBox3D boxes: rooms + furniture
```

| Property | Value |
|---|---|
| Geometry | Connected rooms, ~40–60 boxes |
| Textures | 1024px, basic PBR |
| Lighting | Multi-light setup |
| Assets | Furniture, doors, shelves |

**What the agent learns:** room-to-room navigation, tight spaces, planning turns.

### Stage 4 — Branching Corridors (`complexity ≈ 0.65`)

T-junctions, dead-ends, mixed indoor/outdoor. Vegetation, dynamic objects, fog.

```python
cfg = WorldConfig.branches(seed=42)
boxes = generate_world(cfg)
# ~80–120 BBox3D boxes
```

| Property | Value |
|---|---|
| Geometry | Branching paths, ~80–120 boxes |
| Textures | 1024px, full PBR |
| Lighting | Multi-light |
| Assets | Vegetation, dynamic objects (branches, doors), fog |

**What the agent learns:** path planning, exploration, backtracking, handling dynamic obstacles.

### Stage 5 — Multi-Level Maze (`complexity ≈ 0.85`)

Vertical shafts, upper corridors, outdoor streets/forest complexity.

```python
cfg = WorldConfig.maze(seed=42)
boxes = generate_world(cfg)
# ~150–200 BBox3D boxes
```

| Property | Value |
|---|---|
| Geometry | Multi-level, ~150–200 boxes |
| Textures | 2048px, full PBR |
| Lighting | HDR environment |
| Assets | Vehicles, pedestrians, full PBR, vegetation |

**What the agent learns:** 3D navigation in all directions, altitude changes.

### Stage 6 — Full Photorealism (`complexity ≈ 0.98`)

Dense Doom-like 3D maze with all Infinigen assets at maximum quality.

```python
cfg = WorldConfig.doom(seed=42)
boxes = generate_world(cfg)
# ~250–350 BBox3D boxes
```

| Property | Value |
|---|---|
| Geometry | Dense 3D maze, ~250–350 boxes |
| Textures | 4096px |
| Lighting | HDR environment |
| Assets | All Infinigen assets, subsurface materials, weather, moving branches |

**What the agent learns:** complex multi-axis navigation in fully realistic environments.

## Automatic Curriculum Progression

Instead of manually choosing presets, use `from_curriculum_progress()` to smoothly scale complexity based on a training progress value `[0, 1]`:

```python
from infinigen.core.syndata import WorldConfig, generate_world

# progress=0.0 → trivial corridor
# progress=0.5 → branching corridors with fog
# progress=1.0 → full photorealism
for progress in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
    cfg = WorldConfig.from_curriculum_progress(progress, seed=42)
    boxes = generate_world(cfg)
    print(f"progress={progress:.1f} → complexity={cfg.complexity:.2f}, "
          f"{len(boxes)} boxes, overlay: {cfg.overlay_hints.stage_description}")
```

The progression uses a sqrt curve so early stages ramp slowly (more time on simple environments):

| Progress | Complexity | Environment | Approx. Boxes |
|---|---|---|---|
| 0.00 | 0.00 | Trivial corridor | ~10 |
| 0.05 | 0.22 | Textured corridor | ~16 |
| 0.20 | 0.45 | Rooms with furniture | ~50 |
| 0.50 | 0.71 | Branching + fog | ~100 |
| 0.80 | 0.89 | Multi-level maze | ~180 |
| 1.00 | 1.00 | Full photorealism | ~300 |

## Getting Infinigen Pipeline Parameters

Each world config produces Gin overrides that control the Infinigen rendering pipeline:

```python
from infinigen.core.syndata import (
    WorldConfig, world_gin_overrides, world_summary, to_gin_bindings,
)

cfg = WorldConfig.rooms(seed=42)

# Get gin-compatible parameter dict
overrides = world_gin_overrides(cfg)
# {
#     "grid_coarsen": 3,
#     "object_count": 14,
#     "scatter_density_multiplier": 0.505,
#     "fog_density": 0.0,
#     "material.color_saturation": 0.675,
#     "texture_resolution": 1024,
#     "enable_furniture": True,
#     ...
# }

# Convert to gin binding strings
bindings = to_gin_bindings(overrides)
# ["configure_render_cycles.exposure = 0.3625", "enable_furniture = True", ...]

# Human-readable summary for logging
summary = world_summary(cfg)
```

## Understanding Overlay Hints

`InfinigenOverlayHints` describes which Infinigen asset categories to activate at each complexity level. These are informational — they tell the pipeline (or a human) what to configure:

```python
from infinigen.core.syndata import WorldConfig

cfg = WorldConfig(complexity=0.6, seed=42)
hints = cfg.overlay_hints

print(hints.stage_description)      # "Mixed indoor/outdoor — vegetation, ..."
print(hints.environment_type)       # "mixed"
print(hints.texture_resolution)     # 1024
print(hints.material_complexity)    # "full_pbr"
print(hints.enabled_vegetation)     # True
print(hints.enabled_furniture)      # True
print(hints.enabled_dynamic_objects)# True
```

Or derive hints directly from a complexity value (without a `WorldConfig`):

```python
from infinigen.core.syndata import overlay_hints_for_complexity

# Convenience function — no WorldConfig needed
hints = overlay_hints_for_complexity(0.8)
gin_hints = hints.to_gin_hints()  # dict for Infinigen pipeline
```

## Render Quality Presets

Control render quality independently from world complexity:

```python
from infinigen.core.syndata import drone_preset, to_gin_bindings

# Available presets: "preview", "fast", "medium", "high"
overrides = drone_preset("fast")
# Resolution: 256×256, 64 samples, denoised

overrides = drone_preset("high")
# Resolution: 1024×1024, 512 samples, motion blur, volume scatter

# Custom resolution with a preset's quality settings
overrides = drone_preset("medium", resolution_override=(640, 480))

# Convert to gin bindings
bindings = to_gin_bindings(overrides)
```

| Preset | Resolution | Samples | Denoise | Motion Blur | Use Case |
|---|---|---|---|---|---|
| `preview` | 128×128 | 16 | No | No | Fast iteration, early curriculum |
| `fast` | 256×256 | 64 | Yes | No | RL training at scale |
| `medium` | 512×512 | 128 | Yes | Yes | Mid-curriculum, balanced quality |
| `high` | 1024×1024 | 512 | Yes | Yes | Final curriculum stages |

## Domain Randomisation

Scale randomisation with difficulty — tight ranges at early stages, wide ranges at later stages:

```python
from infinigen.core.syndata import DomainRandomiser

# Easy: minimal randomisation (early curriculum)
rand_easy = DomainRandomiser(difficulty=0.1, seed=42)

# Hard: wide randomisation (late curriculum)
rand_hard = DomainRandomiser(difficulty=0.9, seed=42)

# View all ranges at this difficulty
print(rand_hard.ranges())
# {"sun_elevation": (8.5, 82.5), "fog_density": (0.0, 0.45), ...}

# Get a concrete random sample
sample = rand_hard.sample()
# {"sun_elevation": 47.3, "fog_density": 0.12, ...}

# Get gin overrides
overrides = rand_hard.gin_overrides()

# Or derive from curriculum progress automatically
rand = DomainRandomiser.from_curriculum_progress(0.5, seed=42)
```

## Camera Configuration

Configure drone cameras for the Infinigen rendering pipeline:

```python
from infinigen.core.syndata import DroneCamera, CameraRigConfig

# Single monocular camera
camera = DroneCamera(fov_deg=90)
rig = CameraRigConfig.monocular(n_drones=1)

# Stereo pair for depth estimation
rig_stereo = CameraRigConfig.stereo(baseline_m=0.065, n_drones=4)

# Get gin overrides
overrides = rig_stereo.gin_overrides()
```

## Observation Space Configuration

Define what render passes Infinigen should produce:

```python
from infinigen.core.syndata import (
    ObservationConfig, SensorNoiseModel,
    PASSES_MINIMAL, PASSES_NAVIGATION, PASSES_FULL,
)

# Minimal: depth + object segmentation (fast to render)
obs = ObservationConfig(passes=PASSES_MINIMAL)

# Navigation: depth + normals + segmentation (good for RL)
obs = ObservationConfig(passes=PASSES_NAVIGATION)

# With sensor noise for sim-to-real transfer
noise = SensorNoiseModel.drone_default(difficulty=0.5)
obs = ObservationConfig(passes=PASSES_NAVIGATION, noise=noise)
```

## Episode Configuration

Configure temporal rendering for the Infinigen pipeline:

```python
from infinigen.core.syndata import EpisodeConfig

# Single frame (object detection / static observation)
ep = EpisodeConfig.single_frame()

# Short trajectory (early curriculum)
ep = EpisodeConfig.short_trajectory(num_frames=30, fps=10)

# Full navigation episode
ep = EpisodeConfig.navigation_episode(num_frames=120, fps=24)
```

## Curriculum Config (Stage-Based Progression)

For fine-grained stage-based control of scene parameters:

```python
from infinigen.core.syndata import CurriculumConfig, curriculum_overrides

# Stage 3 of 10
cfg = CurriculumConfig(stage=3, total_stages=10)

print(cfg.progress)            # 0.09 (exponential ease-in)
print(cfg.subdiv_level)        # 0
print(cfg.texture_resolution)  # 128
print(cfg.object_count)        # 10
print(cfg.scatter_density)     # 0.09

# Get pipeline overrides
overrides = curriculum_overrides(cfg)
```

## Scene Budget Estimation

Check render feasibility before starting expensive Blender jobs:

```python
from infinigen.core.syndata import SceneBudget

budget = SceneBudget(
    num_objects=50,
    num_lights=4,
    num_samples=128,
    resolution=(512, 512),
)

print(budget.estimated_vram_gb)     # VRAM estimate with 1.5× safety factor
print(budget.estimated_render_time) # Render time estimate in seconds

# Check if a batch of frames fits in a VRAM budget
fits = budget.batch_fits(num_frames=10, vram_limit_gb=8.0)
```

## Complete Example: Progressive Curriculum Pipeline

```python
"""Generate worlds at increasing complexity for curriculum learning."""
from infinigen.core.syndata import (
    WorldConfig,
    generate_world,
    world_gin_overrides,
    world_summary,
    drone_preset,
    DomainRandomiser,
    ObservationConfig,
    EpisodeConfig,
    to_gin_bindings,
    PASSES_NAVIGATION,
)

# Simulate curriculum progress from 0% to 100%
for progress in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    # 1. Generate world geometry
    world_cfg = WorldConfig.from_curriculum_progress(progress, seed=42)
    boxes = generate_world(world_cfg)

    # 2. Get Infinigen pipeline overrides
    world_overrides = world_gin_overrides(world_cfg)

    # 3. Choose render quality (scale with progress)
    quality = "preview" if progress < 0.3 else "fast" if progress < 0.6 else "medium"
    quality_overrides = drone_preset(quality)

    # 4. Domain randomisation (scales with curriculum)
    randomiser = DomainRandomiser.from_curriculum_progress(progress, seed=42)
    rand_overrides = randomiser.gin_overrides()

    # 5. Camera and observation config
    obs = ObservationConfig(passes=PASSES_NAVIGATION)
    obs_overrides = obs.gin_overrides()

    # 6. Episode timing
    episode = EpisodeConfig.short_trajectory(num_frames=30, fps=10)
    ep_overrides = episode.gin_overrides()

    # 7. Merge all overrides into a single gin config
    all_overrides = {
        **world_overrides,
        **quality_overrides,
        **rand_overrides,
        **obs_overrides,
        **ep_overrides,
    }
    gin_bindings = to_gin_bindings(all_overrides)

    # 8. Log summary
    summary = world_summary(world_cfg)
    print(f"\n--- Progress {progress:.0%} ---")
    print(f"  Complexity: {world_cfg.complexity:.2f}")
    print(f"  Boxes: {len(boxes)}")
    print(f"  Overlay: {world_cfg.overlay_hints.stage_description}")
    print(f"  Quality: {quality}")
    print(f"  Gin bindings: {len(gin_bindings)} entries")
```

## Custom Worlds

Override individual parameters while keeping complexity-driven defaults:

```python
from infinigen.core.syndata import WorldConfig, VisualStyle, generate_world

# Custom: medium complexity but extra-wide corridor
cfg = WorldConfig(
    complexity=0.5,
    corridor_width=4.0,      # wider than default (2m)
    corridor_height=3.0,     # taller
    num_columns=8,           # override auto-derived column count
    seed=42,
)
boxes = generate_world(cfg)

# Custom visual style (override auto-derived style)
style = VisualStyle(
    wall_color_saturation=0.8,
    floor_roughness=0.7,
    fog_density=0.3,
    cloud_density=0.5,
    point_light_count=4,
)
cfg = WorldConfig(complexity=0.5, style=style, seed=42)
boxes = generate_world(cfg)
```

## Output Format

All generation functions return standard data types:

- **`generate_world()`** → `list[BBox3D]` — each box has `.center`, `.extent`, `.label`
- **`generate_flappy_obstacles()`** → `list[FlappyObstacle]` — each has `.center`, `.half_extents`, `.label`
- **`flappy_frame_metadata()`** → `dict[str, Any]` — FrameMetadata-compatible dict with obstacles, depth, traversability
- **`world_gin_overrides()`** → `dict[str, object]` — Infinigen gin parameter names → values
- **`world_summary()`** → `dict[str, Any]` — human-readable summary for logging
- **`overlay_hints_for_complexity()`** → `InfinigenOverlayHints` — asset category hints without building a `WorldConfig`
- **`to_gin_bindings()`** → `list[str]` — gin-parseable binding strings

## Complexity Stage Constants

The named constants define the curriculum stage boundaries:

```python
from infinigen.core.syndata import (
    COMPLEXITY_CORRIDOR,   # 0.15 — below this: flat corridor
    COMPLEXITY_ROOMS,      # 0.35 — below this: textured corridor
    COMPLEXITY_BRANCHES,   # 0.55 — below this: indoor rooms
    COMPLEXITY_MAZE,       # 0.75 — below this: branching corridors
    COMPLEXITY_DOOM,       # 0.90 — below this: multi-level maze
    #                        ≥0.90: full photorealism
)
```

These outputs are consumed by the Infinigen rendering pipeline or by external bridge projects (Genesis World, GenesisDroneEnv) that run the physics simulation and RL training.
