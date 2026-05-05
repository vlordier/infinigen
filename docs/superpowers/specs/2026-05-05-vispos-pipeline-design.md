# Spec: Visual Positioning Dataset Pipeline

**Feature branch**: `feature/vispos-pipeline`
**Date**: 2026-05-05
**Status**: Draft

## Summary

Orchestration layer that composes all visual positioning subsystems into coherent training datasets. Defines dataset specifications, manages paired scene generation across conditions (season × TOD × damage × perspective × modality), and outputs standardized data for geolocalization, visual place recognition, and visual odometry/SLAM training.

## Motivation

The individual subsystems (#1-4) generate capabilities; this pipeline ties them into datasets usable by downstream ML training. Without a unified pipeline, each research team manually scripts their own data generation, leading to inconsistency, missing ground truth, and incompatible formats. A standardized dataset specification system ensures reproducibility and enables systematic ablation studies (e.g., "does adding winter data improve summer localization?").

## Design

### Architecture

```
infinigen/datagen/vispos/
├── __init__.py
├── dataset_spec.py           # Dataset specification schema
├── scene_plan.py             # Which scenes × conditions to generate
├── paired_executor.py        # Paired scene generation (intact→damaged, season→season)
├── ground_truth.py           # Standardized ground truth format
├── output_layout.py          # Dataset directory structure
├── curriculum.py             # Curriculum-based difficulty progression
├── validation.py             # Dataset validity checks
├── export/
│   ├── geoloc.py             # Geolocalization format
│   ├── vpr.py                # Visual Place Recognition format
│   └── vo_slam.py            # VO/SLAM format
└── configs/                  # GIN configs for dataset profiles
```

### Component 1: Dataset Specification (`dataset_spec.py`)

A declarative schema for what data to generate:

```python
@gin.configurable
@dataclass
class VisPosDatasetSpec:
    name: str                              # e.g. "vispos_isr_summer_v1"
    description: str
    
    # Scene parameters
    scene_types: list[str]                 # ["nature_forest", "nature_desert", "indoor_urban", "urban_dense_city", "urban_suburban", "urban_industrial"]
    num_scenes: int                        # Number of unique scenes
    
    # Variation axes
    seasons: list[str]                     # ["spring", "summer", etc.]
    times_of_day: list[str]                # ["dawn", "morning", etc.]
    platforms: list[str]                   # ["isr_orbit", "fpv_racing", "satellite"]
    
    # Condition pairing
    paired_damage: bool                    # Generate intact+damaged pairs
    damage_types: list[str]                # ["earthquake", "war"]
    damage_severities: list[str]           # ["mild", "moderate", "severe"]
    damage_progression_stages: int         # 2 (intact+damaged), 3 (intact+mild+severe), 5 (full progression)
    
    # Weather degradation
    weather_types: list[str]               # ["clear", "fog", "rain", "snow", "dust"]
    weather_levels: list[int]              # [0, 1, 2] for light/medium/heavy degradation
    
    # Modality
    modalities: list[str]                  # ["eo", "lwir", "mwir", "swir"]
    ir_fidelity: str                       # "physics" or "heuristic"
    
    # Invariance axes (which pairs to generate for invariance training)
    invariance_axes: list[str]             # ["altitude", "rotation", "time_of_day", "season", "weather", "modality", "damage", "platform"]
    
    # Task-specific configuration
    tasks: list[str]                       # ["geoloc", "vpr", "vo_slam"]
    
    # Camera
    poses_per_scene: int                   # Unique camera poses per scene
    frames_per_trajectory: int             # For VO/SLAM sequences
    frame_rate: float                      # fps for video sequences
    
    # Output
    resolution: tuple[int, int]            # Render resolution
    render_engine: str                     # "cycles" or "eevee"
```

This spec is resolved into a concrete `ScenePlan` — a list of `(scene_id, season, tod, platform, damage_type, damage_severity, weather_type, weather_level)` tuples to generate.

### Invariance Design

The core advantage of synthetic data for visual positioning: generate the **exact same scene** under different conditions with **perfect ground truth**. This enables training models that are invariant to nuisance factors while retaining place identity.

**Invariance axes** — each represents a condition that changes visual appearance without changing location:

| Axis | What changes | How it's generated | Training signal |
|------|-------------|-------------------|-----------------|
| **Altitude** | Camera height (0.5m → 500km) | Same scene, multiple camera rigs at different altitudes. Each frame pair = same location, different altitude. | Model learns that altitude changes scale/perspective but not place identity. |
| **Rotation / viewpoint** | Camera orientation | Same scene, multiple camera poses. Positive pairs = nearby poses (same place, different angle). Negative pairs = far apart poses. | Model learns viewpoint invariance within a place. |
| **Time of day** | Sun position, sky color, shadows, ambient light | Same scene, rendered at dawn/noon/dusk/night. Identical camera pose. | Model learns to ignore lighting, shadow changes. Critical for day/night relocalization. |
| **Season** | Vegetation, snow cover, ground color, water state | Same scene, seasonal material overrides. Identical camera pose. | Model learns seasonal invariance. Winter snow can completely change visual appearance. |
| **Weather** | Visibility, atmospheric scattering, precipitation | Same scene, different fog/rain/snow/dust levels. Identical camera pose. | Model learns to localize through degraded visibility, precipitation artifacts. |
| **Modality** | EO ↔ LWIR ↔ MWIR ↔ SWIR | Same scene rendered in all bands via multi-sensor rig. Identical camera pose. | Cross-modal localization: query with IR, localize against EO reference (or vice versa). |
| **Damage** | Structural integrity, rubble, craters, scorch | Same scene, progressive damage stages. Identical camera pose. | Model learns that a rubble pile is the same place as the intact building. |
| **Platform** | ISR ↔ FPV ↔ UGV ↔ Satellite | Same location viewed from different platforms. Different motion blur, FOV, perspective. | Cross-platform localization. |

**Pairing strategy**:

For each invariance axis, generate positive pairs (same place, different axis value) and negative pairs (different place):

```
Scene A, summer, noon, clear  ──── positive ──── Scene A, winter, noon, clear
     │
     ├── positive ──── Scene A, summer, dusk, clear
     │
     ├── positive ──── Scene A, summer, noon, fog_heavy
     │
     └── negative ──── Scene B, summer, noon, clear
```

The `invariance_axes` field in the dataset spec controls which axes are varied. When `invariance_axes = ["altitude", "season", "damage"]`, the pipeline generates all combinations of those axes for each scene, producing a matrix of paired data.

**Weather degradation levels** (structured visibility degradation):

| Level | Fog/Haze | Rain | Snow | Dust |
|-------|----------|------|------|------|
| 0 (clear) | No fog, visibility >5km | No rain | No snow | No dust |
| 1 (light) | Light haze, visibility 2-5km | Light drizzle, 5mm/h | Light flurries | Light dust haze |
| 2 (medium) | Moderate fog, visibility 500m-2km | Moderate rain, 10-20mm/h | Moderate snowfall | Moderate dust, reduced contrast |
| 3 (heavy) | Dense fog, visibility <500m | Heavy rain, >20mm/h, water on lens | Heavy snow, whiteout conditions | Dense dust storm, near-zero visibility |

Each level is a gin-configurable preset that sets atmosphere density, particle emission rates, and lens effects (water droplets/condensation on lens). The same scene can be rendered at all 4 levels with identical camera poses.

**Why this matters**: A dataset with 100 scenes × 4 seasons × 4 TOD × 4 weather levels × 2 damage states × 2 modalities ... produces ~25,600 condition variants from just 100 locations. With matched camera poses across all variants, this provides millions of positive/negative pairs for contrastive training. This is the superpower synthetic data has over real imagery — you can't take the same photo at noon and midnight, but you can render both.

### Component 2: Scene Plan Generation (`scene_plan.py`)

Converts a `VisPosDatasetSpec` into executable generation jobs:

```python
def plan_generation(spec: VisPosDatasetSpec) -> GenerationPlan:
    """
    1. Assign scenes to season×TOD combinations
    2. If paired_damage: each scene gets an intact and damaged variant
    3. Distribute platform/camera configurations
    4. Calculate total render count
    5. Output job list for manage_jobs.py
    """
```

**Coverage strategies**:
- **Exhaustive**: Every scene × every condition combination (small datasets, maximum coverage)
- **Sampled**: Randomly sample condition combinations (large datasets, manage compute)
- **Stratified**: Ensure each condition appears at least N times across scenes

**Dataset composition modes** (first-class presets):

Localization research shows that *more unique places* often beats *more conditions per place* for geolocalization and VPR. The pipeline provides two modes:

| Mode | Scenes | Conditions per scene | Total frames (est.) | Best for |
|------|--------|---------------------|---------------------|----------|
| **Diversity** (default) | 5,000-50,000 | 1-2 | 250K-5M | Geolocalization, VPR — learning what makes places different |
| **Invariance** | 100-500 | Full matrix (up to 256) | 25K-640K | Cross-condition robustness — learning what doesn't change per place |

**Diversity mode** maximizes unique place count at the cost of per-place condition coverage. Each scene gets one randomly assigned (season, TOD, weather, damage) combination. IR rendering is heuristic-only. Render engine is EEVEE for speed. This is the *default* because most localization training benefits more from place diversity.

**Invariance mode** generates the full condition matrix for fewer scenes. Used when the research question is specifically about condition invariance (day/night relocalization, cross-season VPR, damage-robust localization). Typically run as a secondary dataset after the diversity-mode base.

**Hybrid mode**: 5,000 diversity scenes + 100 invariance scenes. The diversity scenes provide place coverage; the invariance scenes provide explicit cross-condition training signal. This is the recommended configuration for production training.

**Example plan** for `vispos_isr_summer_v1` (100 scenes, summer only, ISR orbit + FPV, EO + LWIR, paired earthquake):
```
100 scenes × 1 season × 1 TOD × 2 platforms × 2 modalities × 2 damage_states
= 800 scene-condition combinations
× 50 poses each = 40,000 frames
× 2 (EO render + IR render) = 80,000 renders
```

### Component 3: Paired Execution (`paired_executor.py`)

Manages the multi-pass generation of a single scene under different conditions:

```
For each scene in ScenePlan:
  1. Generate base scene (intact, first season/TOD combo)
  2. Place cameras on base scene
  3. Snapshot camera trajectories
  4. For each (season, TOD, damage, modality) variant:
     a. Reapply season/TOD environment
     b. Apply damage (if variant is damaged)
     c. Validate/adjust cameras
     d. Render all passes (EO + IR bands as specified)
     e. Save per-variant outputs
```

**State machine**:
```
BASE_SCENE → [CAMERAS] → [ENV_VARIANTS] → [DAMAGE_VARIANTS] → [RENDER]
                ↑                                              |
                └────────── reuse cameras ─────────────────────┘
```

Each variant gets its own output subdirectory with its own `metadata.json`. Shared data (scene geometry, base cameras) is stored once at the scene level.

### Component 4: Ground Truth Format (`ground_truth.py`)

Standardized per-frame metadata:

```python
@dataclass
class FrameMetadata:
    # Frame identity
    scene_id: str
    frame_index: int
    timestamp: float                     # seconds from sequence start
    
    # Camera parameters
    camera_intrinsics: CameraIntrinsics  # fx, fy, cx, cy, distortion
    camera_extrinsics: CameraExtrinsics  # 4×4 world→camera transform
    
    # Georeferencing
    lat: float                           # WGS84 latitude (mapped from Blender coords)
    lon: float                           # WGS84 longitude
    alt: float                           # Altitude above WGS84 ellipsoid
    camera_pose_ecef: np.ndarray         # 4×4 ECEF transform
    gsd_meters: float                    # Ground sample distance
    
    # Environment
    season: str
    time_of_day: str
    sun_azimuth: float                   # degrees
    sun_elevation: float                 # degrees
    weather: dict                        # rain, snow, fog parameters
    
    # Scene state
    is_damaged: bool
    damage_type: str                     # "none", "earthquake", "war"
    damage_severity: str                 # "none", "mild", "moderate", "severe"
    
    # Sensor
    modality: str                        # "eo", "lwir", "mwir", "swir"
    sensor_spec: dict                    # sensor parameters
    
    # Task labels
    place_id: str                        # For VPR: unique location identifier
    sequence_id: str                     # For VO/SLAM: trajectory identifier

@dataclass
class PerPixelPasses:
    """Optional per-pixel ground truth passes saved alongside frames."""
    permanence: bool                     # 3-class: permanent / semi-permanent / transient
    cross_condition_correspondence: bool # Pixel correspondences to same pose in ref condition
    covisibility: bool                   # Which frames see which 3D points
    valid_depth: bool                    # Sensor-realistic depth validity mask
```

**Feature permanence labels** (`permanence` pass, new):

A per-pixel 3-class label that only synthetic data can provide perfectly:

| Class | Value | Examples | Behavior across conditions |
|-------|-------|----------|---------------------------|
| **Permanent** | 2 | Terrain, bedrock, building structure, roads, bridges, permanent infrastructure | Always present at same 3D location. May change appearance (winter snow on road) but geometry persists. |
| **Semi-permanent** | 1 | Trees, streetlights, signs, landmarks, power poles | Present for months/years. Seasonal appearance change. Damageable. May disappear in severe damage stages. |
| **Transient** | 0 | Vehicles, pedestrians, construction equipment, debris, seasonal snow cover, vegetation undergrowth, parked objects | May not exist days later. Free to use or ignore depending on task. |

This is ground truth that no real-world dataset can provide — real data only captures one moment in time with no knowledge of what will persist. The permanence mask is computed from the scene graph: objects tagged with their permanence class at generation time, rendered as a flat-shaded segmentation pass alongside RGB.

**Cross-condition segmentation consistency**: Object and instance IDs are guaranteed consistent across all condition variants of the same scene. Building_0042 has ID 37 in summer/intact, summer/damaged, winter/intact, and winter/damaged. The segmentation is stored once per scene and referenced by per-variant metadata — not regenerated per variant.

**Cross-condition pixel correspondences**: When the same 3D surface point is visible from the same camera pose in two condition variants (e.g., summer/noon and winter/dusk), the pipeline provides a 2-channel float32 EXR mapping `(u, v)` from each pixel in variant A to its corresponding pixel in variant B. Pixels with no correspondence (occluded, out of frame) are `(-1, -1)`. This is the ground-truth supervision signal for cross-condition descriptor learning.

**Covisibility graphs**: Per scene, a sparse matrix of which frame pairs share >N visible 3D points. Computed from depth+pose projection. Enables efficient positive/negative pair mining for VPR training.

**GPS noise model** (in `FrameMetadata`):

GPS coordinates are provided at two fidelity levels:
- **Clean**: Exact Blender→WGS84 transform (for validation, debugging)
- **Noisy**: Realistic GPS error applied based on scene context:
  - Open terrain: 2-5m horizontal, 3-8m vertical (low PDOP)
  - Urban canyon (buildings nearby): 10-30m horizontal, 15-50m vertical (multipath)
  - Under bridge/overpass: GPS dropout with configurable probability
  - Noise characteristics: non-zero-mean bias from multipath + Gaussian jitter
  - Configurable via `gps_noise_preset`: "clean", "urban", "harsh", "custom"

This enables training models that learn to fuse noisy GPS with visual positioning rather than overtrusting perfect coordinates.

### Component 5: Output Layout (`output_layout.py`)

Standardized directory structure:

```
dataset/
├── scenes/
│   ├── scene_0001/
│   │   ├── geometry/                    # Base scene mesh (optional export)
│   │   ├── intact/
│   │   │   ├── summer_noon_isr_orbit/
│   │   │   │   ├── eo/
│   │   │   │   │   ├── frame_0000.exr
│   │   │   │   │   ├── frame_0000_depth.exr
│   │   │   │   │   ├── frame_0000_normal.exr
│   │   │   │   │   ├── frame_0000_segmentation.exr
│   │   │   │   │   └── ...
│   │   │   │   ├── lwir/
│   │   │   │   │   └── ...
│   │   │   │   └── metadata.json
│   │   │   └── winter_dusk_fpv_scout/
│   │   │       └── ...
│   │   └── damaged/
│   │       └── earthquake_moderate/
│   │           └── summer_noon_isr_orbit/
│   │               └── ...  (same camera poses as intact counterpart)
├── splits/
│   ├── train_scenes.txt
│   ├── val_scenes.txt
│   └── test_scenes.txt
└── dataset_spec.json                    # Original spec that produced this dataset
```

### Component 6: Task-Specific Export (`export/`)

**Geolocalization** (`geoloc.py`):
- Single-frame: image + GPS (lat/lon) per frame
- Triplet format: (anchor, positive, negative) for contrastive learning
- Positive = same location, different condition (season/TOD/damage)
- Negative = different location

**Visual Place Recognition** (`vpr.py`):
- Query-reference pairs: query from one condition, reference from another (same location)
- Multiple reference conditions per query
- Format compatible with standard VPR benchmarks (MSLS, Pitts250k conventions)

**Visual Odometry / SLAM** (`vo_slam.py`):
- Temporal sequences: consecutive frames with incremental pose
- Ground truth: 6-DoF pose per frame, relative transforms between frames
- IMU simulation (optional): synthetic accelerometer + gyroscope data from trajectory derivatives
- Format: TUM RGB-D, EuRoC, or KITTI conventions

### Component 7: Curriculum Generation (`curriculum.py`)

Difficulty progression for training robust models:

```
Stage 1 (Easy):   Summer, noon, no damage, single season, clear weather
Stage 2 (Medium): Season variation (same scene in 2 seasons), mild TOD variation
Stage 3 (Hard):   Full season×TOD matrix, moderate damage, multiple platforms
Stage 4 (Expert): Severe damage, all IR bands, night conditions, adverse weather
```

Each stage config is a `VisPosDatasetSpec` with different parameters. Models can be progressively trained through the curriculum.

### Component 8: Evaluation Methodology

A held-out test protocol for measuring whether models trained on this data actually work. The pipeline is a tool — this section defines what success looks like and how to measure it.

**Held-out test methodology**:

| Split | What | Purpose |
|-------|------|---------|
| **Training** | 70% of scenes, baseline conditions | Standard training data |
| **Validation** | 15% of scenes, same conditions as training | Hyperparameter tuning, early stopping |
| **Test (in-domain)** | 15% of scenes, same conditions as training | Measure baseline localization performance |
| **Test (cross-condition)** | Same test scenes, different conditions | Measure robustness: how much does accuracy drop in winter vs summer? |
| **Test (cross-region)** | New scenes from held-out regional style | Generalization: can the model localize in a region it never saw during training? |
| **Test (cross-damage)** | Same test scenes, held-out damage stage | Damage robustness: trained on mild+moderate, tested on severe |

**Held-out conditions for generalization testing** (configurable):

```python
@gin.configurable
class EvaluationSplit:
    held_out_styles: list[str]       # Regional styles excluded from training, e.g. ["soviet"]
    held_out_damage: list[int]       # Damage stages excluded, e.g. [4] (total destruction)
    held_out_weather: list[int]      # Weather levels excluded, e.g. [3] (heavy)
    held_out_platforms: list[str]    # Platforms excluded, e.g. ["satellite"]
    held_out_seasons: list[str]      # Seasons excluded, e.g. ["winter"]
```

**Target metrics** (for baseline validation):

| Task | Metric | Target threshold |
|------|--------|-----------------|
| Geolocalization | Mean haversine error (km) | <1km at country scale, <100m at city scale |
| VPR | Recall@1, Recall@5 | >80% R@1 on in-domain, >60% R@1 on cross-condition |
| VO/SLAM | ATE RMSE (m), RPE (%) | <5% drift per 100m traveled |

**Real-data validation** (recommended, not in v1):
- A small set of real ISR/UAS imagery (200-500 frames) with GPS ground truth, collected in conditions matching the synthetic data (same season, similar environment)
- Measure domain gap: compute feature distributions or train a domain classifier between synthetic and real
- If available, fine-tune on real data and measure improvement — quantifies the pretraining value of synthetic data

**Ablation framework**:
The invariance axes enable systematic ablation. Train N models, each with one axis removed from the training data, test all models on the held-out condition for that axis:

```
Baseline: trained on all 8 axes → test on cross-condition winter
- TOD:      train without TOD variation  → test on winter
- Season:   train without season variation → test on winter  
- Weather:  train without weather variation → test on winter
```

The delta between baseline and each ablation measures the marginal training value of each axis. This answers the research question: "Does adding winter training data actually improve winter localization?"

### Component 9: Dataset Validation (`validation.py`)

Automated checks run after generation:

1. **Completeness**: All expected frames exist, no missing files
2. **Pairing**: Damaged scenes have matching camera poses in intact counterpart (within tolerance)
3. **Metadata integrity**: All metadata.json files parse, all required fields present
4. **Image validity**: No all-black/all-white frames, no NaN pixels
5. **Coordinate sanity**: Lat/lon in valid range, camera poses are SE(3)
6. **Segmentation**: Object IDs consistent across condition variants for same scene
7. **Disk efficiency**: No redundant large files (shared scene data not copied per variant)

### Integration

- Plugs into `manage_jobs.py` as a new job type (`vispos_dataset`)
- Each scene-condition combination becomes a `Job` in the job manager
- `VisPosDatasetSpec` resolved to gin configs that drive scene generation
- Standard pipeline: `compose_nature()` / `compose_indoors()` → `populate_scene()` → `render_image()` → `post_render.py` for EXR processing
- Ground truth extraction adds per-frame metadata writing alongside existing `save_camera_parameters()`

### Testing Strategy

1. **Spec resolution**: Test that a `VisPosDatasetSpec` resolves to the correct number of jobs
2. **Paired generation**: Generate 3 scenes with damage, verify intact+damaged frames exist and cameras match
3. **Export formats**: Verify geoloc, VPR, and VO/SLAM exports are loadable by standard dataloaders
4. **Curriculum**: Verify each stage produces scenes within its difficulty bounds
5. **Metadata round-trip**: Write metadata → read back → verify all values preserved
6. **Scale test**: Plan a 1000-scene dataset, verify plan generation completes and total render count is correct
7. **Validation pass**: Run `validate_dataset()` on a small generated dataset, confirm all checks pass

### Dependencies

- Integrates with all other features (#1-6) but works with stubs when subsystems aren't available
- When a subsystem is missing, the spec falls back gracefully: no urban scenes → nature/indoor only, no IR → EO only, no damage → intact only
- Depends on `manage_jobs.py`, `render.py`, `post_render.py`, `camera.py` (all existing)
- Urban scenes (#6) provide the primary scene diversity for visual positioning in built environments

### Open Questions

1. Should the pipeline support incremental dataset building (add more scenes later, merge with existing)? Initial: no, each dataset is generated as a unit.
2. How to handle the Blender world → WGS84 transform for curved-Earth satellite views? Initial: planar approximation (scene is small enough that Earth curvature is negligible at 500m. For satellite: use orthographic projection.)
3. Should we produce pre-computed image embeddings (e.g., from a pretrained model) alongside raw images? No — this is training data for other models. Let consumers compute their own features.
4. Coordinate convention: OpenCV (Y-down, Z-forward) or OpenGL/Blender (Y-up, Z-backward)? Store in OpenCV convention to match standard vision libraries. Convert from Blender's convention using existing transform in `save_camera_parameters()`.
