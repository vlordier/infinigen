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
    scene_types: list[str]                 # ["nature_forest", "nature_desert", "nature_desert_dunes", "nature_ocean", "nature_snowfield", "nature_barren_steppe", "indoor_room", "urban_dense_city", "urban_suburban", "urban_industrial", "urban_harbour", "urban_coastal"]
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

**Long-range traverses**: Visual odometry drift matters most on long traversals. Scene tiles can be stitched for extended trajectories:

| Traverse type | Distance | Tiles | Use case |
|--------------|----------|-------|----------|
| Short (default) | 100-500m | 1 | FPV scouting, UGV local patrol |
| Medium | 1-5 km | 2-4 | ISR orbit over urban area, UGV cross-town |
| Long | 5-20 km | 4-16 | ISR plane transit, long-endurance survey |
| Extended | 20-100 km | 16-64 | Satellite ground track, continental transit |

Tiling: Adjacent terrain tiles are generated with matching edge geometry (shared terrain SDF, matching road endpoints at tile boundaries). A trajectory is planned across tiles; rendering stitches tiles at the camera frustum boundary. For tiled trajectories, `trajectory_total_distance_m` is recorded in metadata for drift analysis.

**Feature-sparse environments**: Navigation must work when there's nothing to see. Explicit scene types:
- `nature_desert_dunes`: Repeating sand dune patterns, zero vertical features — pure texture-based odometry challenge
- `nature_ocean`: Open water with wave patterns only — no static features at all (optical flow from waves is all you get)
- `nature_snowfield`: White terrain, overcast lighting, zero contrast — extreme feature poverty
- `nature_barren_steppe`: Flat grassland, horizon is the only feature — tests long-range feature matching
- Per-frame feature density metadata (`feature_count`, `harris_corner_count`) for dataset filtering by difficulty

**Contested environment presets**: GNSS denial often comes with cascading effects. Bundled gin presets combine multiple degradation axes with a causal timeline:

| Preset | Timeline | Effects |
|--------|----------|---------|
| `"clear_sky_ops"` | Nominal throughout | GPS nominal, clear weather, no damage — baseline |
| `"jammed_approach"` | GPS nominal 0-200, degraded 200-300, denied 300+, fog level 1 throughout | Flying into jamming radius with marginal weather |
| `"post_strike_blackout"` | GPS denied throughout, war damage stage 3, smoke/fog level 2, fire particles active | Operating after an attack — GPS denied, visual degraded by smoke and destruction |
| `"spoofed_ambush"` | GPS nominal 0-100, spoofed 100-300 (dragged 5km east), denied 300+ | Platform deceived then jammed |
| `"urban_canyon_degraded"` | Intermittent GPS (alternating 20-frame windows), urban multipath noise, no weather | Natural urban GPS denial without adversarial action |
| `"winter_whiteout"` | GPS nominal, snowfield scene, snow weather level 3, overcast TOD | Extreme feature poverty with heavy snowfall — visual navigation near-impossible |

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
    loop_closure: list[tuple[int, float]]# List of (frame_index, relative_distance) for frames within loop closure radius
    
    # Navigation-specific ground truth
    gravity_vector_camera: list[float]   # [gx, gy, gz] in camera frame, for VIO gravity alignment
    metric_scale: float                  # World units → meters scale factor (1.0 for metric scenes)
    stereo_baseline: float               # Stereo baseline in meters (0.0 for monocular)
    relocalization: bool                 # True if this frame triggered relocalization (drift correction > threshold)
    relocalization_correction: list[float] # [dx, dy, dz, droll, dpitch, dyaw] correction applied

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

**GNSS receiver model** — replaces simple GPS noise with a stateful denial-capable model:

GPS is not just noisy; GNSS-denied navigation means GPS can disappear, degrade, or lie. The receiver model has a state machine:

```
            ┌─────────┐
            │ NOMINAL │ ← normal operation, noise per preset ("clean"/"urban"/"harsh")
            └────┬────┘
                 │ signal degraded
            ┌────▼────┐
            │DEGRADED │ ← increasing error over N frames, SNR dropping
            └────┬────┘
           ┌─────┴─────┐
           │            │
      ┌────▼───┐   ┌───▼──────┐
      │ DENIED │   │ SPOOFED  │ ← wrong position with high DOP confidence
      │(silent)│   │(deceived)│
      └────┬───┘   └───┬──────┘
           │            │
           └─────┬──────┘
                 │ reacquisition
            ┌────▼────┐
            │REACQUIRE│ ← stale position, then convergence to true position over N frames
            └────┬────┘
                 │ lock reestablished
            ┌────▼────┐
            │ NOMINAL │
            └─────────┘
```

**GPS availability timeline** (per trajectory):

```python
@gin.configurable
@dataclass
class GPSAvailabilitySchedule:
    # List of (start_frame, end_frame, mode) tuples. 
    # mode: "nominal" | "degraded" | "denied" | "spoofed" | "reacquire"
    windows: list[tuple[int, int, str]]
    spoofed_offset_m: tuple[float, float, float]  # ECEF offset when spoofed (e.g., 5000, 0, 0)
    spoofed_drift_mps: float                      # Drift rate during spoofing (simulates dragging)
    degraded_snr_decay_db_per_frame: float        # SNR decline rate during degradation
    reacquisition_frames: int                     # How many frames to reacquire
    denial_transition_frames: int                 # Frames of boundary degradation when entering/leaving denial
```

**Standard scenarios** (gin presets):

| Scenario | GPS timeline | Training signal |
|----------|-------------|-----------------|
| `"always_on"` | Nominal throughout | Baseline — GPS always available |
| `"last_known_position"` | Nominal frames 0-100, then denied | Standard GNSS-denied: start from fix, navigate without |
| `"intermittent"` | Nominal [0-100, 350-400, 700-750], denied elsewhere | GPS returns sporadically (clearing, open terrain) |
| `"spoofed_midflight"` | Nominal 0-200, spoofed 200-500, denied 500+ | GPS lies for a while, then disappears |
| `"gradual_jamming"` | Nominal 0-50, degraded 50-150, denied 150+ | Progressive jamming as platform approaches emitter |
| `"urban_canyon_dropout"` | Alternating nominal/denied with 10-30 frame periods | GPS cuts in and out between buildings |

**FrameMetadata GPS fields** (per frame, from the receiver model):
- `gps_lat`, `gps_lon`, `gps_alt`: Position reported by receiver (may be wrong during spoof)
- `gps_horizontal_error_m`, `gps_vertical_error_m`: DOP-based error estimate reported by receiver (may be overconfident during spoof)
- `gps_mode`: "nominal", "degraded", "denied", "spoofed", "reacquire"
- `gps_true_lat`, `gps_true_lon`, `gps_true_alt`: Always correct ground truth (for validation)
- `gps_snr_db`: Signal-to-noise ratio (NaN when denied)

### Component 4b: IMU & VIO Sensor Data (`imu_sensor.py`)

GNSS-denied navigation uses visual-inertial odometry (VIO) as the primary navigation source when GPS is unavailable. The pipeline must generate realistic synthetic IMU data rigidly attached to the camera.

```python
@gin.configurable
@dataclass
class IMUSpec:
    # IMU parameters (realistic MEMS-grade defaults)
    accel_noise_density: float          # μg/√Hz, e.g., 150 for consumer MEMS, 10 for tactical
    accel_random_walk: float            # m/s²/√hr
    gyro_noise_density: float           # °/√hr, e.g., 0.5 for tactical, 5 for consumer
    gyro_random_walk: float             # °/√hr³/²
    accel_bias_instability_ug: float    # μg, Allan variance bias instability
    gyro_bias_instability_deg_per_hr: float # °/hr
    accel_scale_factor_error_ppm: float # Scale factor error in parts per million
    gyro_scale_factor_error_ppm: float
    axis_misalignment_deg: float        # Cross-axis sensitivity
    sample_rate_hz: int                 # 100-1000 Hz (IMU runs faster than camera)
    temperature_drift_enabled: bool     # Simulate warm-up drift over first N seconds
    long_term_bias_walk_mps2_per_hr: float  # Slow bias drift during multi-hour missions
    bias_shock_sigma_mps2: float        # Shock-induced bias shift (landings, rough terrain)
    
    # Magnetometer (for heading when GPS lost)
    mag_enabled: bool
    mag_noise_uT: float                 # μT noise density
    mag_hard_iron: list[float]          # 3-vector hard iron offset
    mag_soft_iron: list[float]          # 3×3 soft iron matrix (flattened)
    mag_declination_deg: float          # Local magnetic declination
    
    # Barometric altimeter
    baro_enabled: bool
    baro_noise_hpa: float               # hPa noise density (1 hPa ≈ 8.4m at sea level)
    baro_drift_hpa_per_hr: float        # Pressure drift from weather fronts
    baro_rotor_wash_sigma_hpa: float    # Venturi effect from rotor/propeller wash
    baro_temperature_coefficient_hpa_per_c: float  # Temperature dependence
    baro_quantization_hpa: float        # ADC quantization step
    
    # IMU-camera calibration
    T_cam_imu: list[float]              # 4×4 rigid transform camera→IMU (flattened)
    T_cam_imu_noise_mm: float           # Per-axis calibration uncertainty
    time_offset_cam_imu_us: int         # Microseconds, camera vs IMU clock offset
```

**IMU data generation**: From the camera trajectory (position keyframes), compute analytical velocity and acceleration via spline derivatives. Sample at the IMU rate (200-1000 Hz). Apply the IMU error model:
1. Add bias instability (static offset + random walk)
2. Add scale factor error (proportional to true signal)
3. Add axis misalignment (cross-axis leakage)
4. Add white noise (noise density × √bandwidth)
5. Transform to IMU frame via `T_cam_imu`
6. Add gravity vector in IMU frame (from camera orientation)

**Output format**: IMU data stored as CSV per trajectory:
```
timestamp_ns, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z
```
Rates: IMU at sample_rate_hz, magnetometer at 50-100 Hz (configurable). Timestamps in nanoseconds since sequence start, shared clock with camera frames.

**IMU-camera synchronization**: Each camera frame timestamp is the exposure center. IMU samples are time-stamped at their sample instant. The `time_offset_cam_imu_us` models real clock drift between sensor clocks. Multi-rate interpolation: IMU data is provided at native rate; consumers must interpolate to camera frame times for VIO training (realistic sensor fusion challenge).

**VIO ground truth**: In addition to raw IMU, provide:
- `vio_gravity_aligned_pose`: Camera pose in gravity-aligned world frame (Z=up, gravity=[0,0,-9.81]), for VIO initialization supervision
- Per frame: gravity vector in camera frame (from `FrameMetadata.gravity_vector_camera`)
- Integration ground truth: pre-integrated IMU deltas between camera frames (ΔR, Δv, Δp with covariance), using standard IMU preintegration formulation (Forster et al., 2015)

### Component 4c: Loop Closure Annotations

Loop closure is the critical failure point for visual SLAM. Provide explicit ground truth:

```python
@dataclass
class LoopClosureAnnotations:
    # Sparse N×N matrix: pairs (i,j) where ||pose_i - pose_j|| < radius
    pairs: list[tuple[int, int]]
    relative_transform: list[np.ndarray]  # T_ij for each pair
    distance_m: list[float]               # Euclidean distance between poses
    viewpoint_angle_deg: list[float]      # Angle between view directions (0=same, 180=opposite)
    covisible_points: list[int]           # Number of 3D points visible in both frames
    difficulty: list[str]                 # "easy" (same TOD/weather) / "medium" (different TOD) / "hard" (different season + weather)
```

Stored as NPZ per trajectory alongside frames. Enables training of loop closure detection networks with difficulty-stratified supervision.

**Terrain-relative navigation ground truth**: For high-altitude ISR and satellite, horizon/ridgeline matching is a primary long-range navigation method. Optional per-frame outputs:

- **Horizon profile**: For ISR loiter/plane frames (look angle >30° from nadir), extract the horizon line as an elevation-angle-vs-azimuth array (360 values at 1° resolution). Where the horizon is occluded by buildings/terrain, record the occlusion distance.
- **Skyline distinctiveness**: Per-frame metric (0-1) quantifying horizon recognizability. Flat ocean = 0.0, mountain range = 1.0. Enables filtering datasets by skyline quality.
- **DEM raster**: Per scene, export terrain height as a georeferenced raster at configurable resolution (5-30m/pixel). Reference for DEM-matching localization algorithms.
- **Ridgeline peak annotations**: Persistent 3D point IDs for prominent terrain peaks visible above the horizon, with their 2D projections per frame. Stable across seasons/TOD (same peak IDs in summer and winter).

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

### Map-Based Localization Reference Data

Beyond query imagery (what the platform sees), generate reference data (what it matches against). Many GNSS-denied approaches match current view against pre-existing references:

| Reference type | Generated by | Format | Use case |
|---------------|-------------|--------|----------|
| **Satellite orthophoto** | Satellite camera at scene center, nadir, rendered at configurable GSD (0.3-5m/pixel) | Single large GeoTIFF per scene | Match ISR oblique view against satellite reference — standard geolocalization |
| **3D reference map** | Pre-rendered textured 3D model export (point cloud + descriptors from one condition, e.g., summer/noon) | PLY with per-point SIFT/SuperPoint descriptors | Match current view against prior mapping pass — standard VPR |
| **Reference trajectory** | A previous overflight (different platform, different season, different TOD) rendered alongside query trajectory | Full frame sequence with metadata | Cross-platform, cross-time relocalization |

**Query-vs-reference pairing**: When `invariance_axes` includes `"reference"`, the pipeline generates both query and reference data for each scene. The reference is rendered once (e.g., satellite orthophoto at summer/noon); queries span all condition variants. This enables training cross-modal geolocalization: match an ISR LWIR query at winter/dusk against an EO satellite reference at summer/noon.

**Satellite reference rendering**: Satellite camera at scene center, altitude = configurable (250-500km), nadir pointing, FOV covering the full scene. Rendered at configurable GSD (0.3m for high-res, 5m for Landsat-like). Atmospheric correction applied (Rayleigh scattering removal). Output: single GeoTIFF with world file + per-pixel lat/lon if needed.

### Storage Budgets

Per-configuration storage estimates for infrastructure planning. Assumes EXR float16 for RGB/IR (4 channels), float32 for depth/normal/flow (1-4 channels), PNG for visualization:

| Configuration | Frames | Resolution | RGB (EXR) | GT passes | IMU | Total |
|-------------|--------|-----------|-----------|-----------|-----|-------|
| Diversity 5K, EO only, 1 cond | 250K | 1920×1080 | 2.0 TB | 1.5 TB | 0.05 TB | ~3.5 TB |
| Diversity 10K, EO+LWIR, 1 cond | 500K | 1920×1080 | 4.0 TB | 3.0 TB | 0.10 TB | ~7.1 TB |
| Invariance 100, EO+LWIR, 128 cond | 640K | 1920×1080 | 5.1 TB | 3.8 TB | 0.13 TB | ~9.0 TB |
| Hybrid 5K+100, EO+LWIR | 890K | 1920×1080 | 7.1 TB | 5.3 TB | 0.18 TB | ~12.6 TB |
| Satellite ref (per scene) | 1 | 4096×4096 | 0.067 GB | 0.067 GB | — | ~0.13 GB |

Storage can be reduced 4-8× with lossless compression (EXR PIZ). PNG proxies add ~10%. For large datasets, consider lazy generation (render on demand for specific frames) rather than pre-rendering everything.

### Data Augmentation vs Generated Variation

Not all visual variation should be baked into rendered data. Guide:

| Variation | Render or Augment? | Rationale |
|-----------|-------------------|-----------|
| Season, TOD, weather, damage | **Render** | Cannot be simulated in RGB space. Requires physics. |
| Modality (EO↔IR) | **Render** | Requires separate render passes. |
| Camera pose, altitude, platform | **Render** | Requires 3D scene access. |
| Color jitter, brightness, contrast | **Augment** | Photometric augmentations are cheap and effective at training time. |
| Random crops, flips, small rotations | **Augment** | Standard data augmentation, free during training. |
| Gaussian noise, JPEG compression | **Either** | Render path has better sensor realism (h264), but augmentation path is cheaper for simple noise. |
| Motion blur, rolling shutter, lens flare | **Render** | Physically accurate simulation requires 3D scene + camera model. |

**Recommendation**: Render the structural axes (season, TOD, weather, damage, modality, pose). Apply photometric augmentations at training time. This reduces render cost ~10× vs rendering all 256 condition combinations.

### Shadow Dynamics Warning

At dawn/dusk TOD presets, the low sun angle causes cast shadows that move perceptibly between consecutive VO/SLAM frames (1-5 pixels/frame at ISR altitudes near terminator lighting). This creates apparent motion in static scenes — a well-known VO failure mode. Per-frame metadata includes `shadow_velocity_px_per_frame` to flag high-risk frames. VO/SLAM training should include dawn/dusk sequences specifically to learn shadow-robust feature matching.

### Fog Altitude Stratification

Fog is not uniform — it has strong vertical stratification (thickest near ground, thinning with altitude). Weather level definitions updated:

| Level | Ground visibility | Visibility at 100m AGL | Visibility at 500m AGL | Fog ceiling |
|-------|------------------|----------------------|----------------------|-------------|
| 1 (light) | 2-5 km | 3-8 km | 5-15 km | 200-300m |
| 2 (medium) | 500m-2km | 800m-3km | 2-5 km | 150-250m |
| 3 (heavy) | <500m | 500m-1km | 1-3 km | 100-200m |

This matters for multi-platform datasets: a UGV at 2m AGL sees nearly zero visibility through heavy fog while an ISR plane at 500m sees the ground clearly.

### Pre-Training → Fine-Tuning Curriculum

The curriculum (Component 7) should be structured as a transfer learning schedule, not just difficulty progression:

1. **Pre-train on Diversity mode**: 10K+ unique places, 1-2 conditions each, all platform types. Model learns place distinctiveness — what makes locations different. This is the largest dataset by frame count.
2. **Fine-tune on Invariance mode**: 100-500 places, full condition matrix. Model learns condition invariance — what stays the same when everything else changes. Lower learning rate, fewer epochs.
3. **Evaluate on held-out splits**: Cross-region (untrained regional style), cross-damage (untrained severity), cross-platform (untrained platform). Measure generalization.

The dataset splits should partition scenes such that pre-training and fine-tuning use disjoint scene sets (no place overlap) to prevent memorization. The invariance scenes are new places the model has never seen — it must learn generalizable condition invariance, not memorize that "scene_42 in winter looks like this."

### Training Baseline Recommendations

Recommended architectures and training protocols per task:

| Task | Model | Loss | Key hyperparameters | Expected baseline |
|------|-------|------|---------------------|-------------------|
| Geolocalization (EO) | CosPlace, MixVPR | InfoNCE, τ=0.07 | Batch 256, AdamW lr=1e-4, 50 epochs | R@1 > 60% on in-domain, > 40% cross-condition |
| Geolocalization (cross-modal) | CVM-Net, Cross-Modal VPR | Triplet, margin=0.3 | Batch 128, lr=5e-5, 100 epochs | R@1 > 30% EO→LWIR, > 20% LWIR→EO |
| VPR (EO) | NetVLAD, CosPlace, SALAD | Triplet + ClusterLoss | Batch 128, clusters=64, lr=1e-4 | R@1 > 80% in-domain, > 60% cross-season |
| VPR (cross-condition) | MixVPR, AnyLoc | Multi-similarity loss | Batch 256, lr=5e-5 | R@1 > 50% cross-weather, > 40% cross-damage |
| VO (monocular+IMU) | DROID-SLAM, DPVO | Photometric + pose graph | Seq len 7, lr=1e-4, IMU preintegration | ATE < 2% of distance traveled |
| VO (stereo) | RAFT-Stereo + VIO | Disparity + pose | Seq len 5, lr=2e-4 | ATE < 1% of distance traveled |
| Loop closure detection | NetVLAD + SeqSLAM | Triplet, margin=0.5 | Seq len 10, lr=1e-5 | Precision > 90% at 100% recall |

Evaluation protocol: For each task, train on the recommended training split, evaluate on in-domain test, cross-condition test, and cross-region test. Report Recall@K (geoloc/VPR), ATE RMSE (VO/SLAM), and precision-recall AUC (loop closure).

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
