# Spec: Platform Camera Models

**Feature branch**: `feature/flight-cameras`
**Date**: 2026-05-05
**Status**: Draft

## Summary

Add specialized camera placement, motion models, and sensor characteristics for aerial and ground platforms: ISR drones, fixed-wing ISR planes, FPV drones, UGV ground vehicles, and satellite push-broom. Support multi-sensor EO/IR rigs and generate the diverse viewpoints needed for visual positioning training across altitudes from 0.5m (ground) to 500km (orbital).

## Motivation

Visual positioning systems are deployed on diverse platforms with fundamentally different viewpoint characteristics:
- **ISR drones** (50-500m): Stable, orbital, systematic coverage, oblique views
- **FPV drones** (5-100m): Agile, forward-facing, high speed, aggressive banking
- **Satellites** (200km+): Nadir, push-broom scanning, near-orthographic

Infinigen's current camera system supports pedestrian-level viewpoints (1.5-2.5m altitude, random walks). These don't match aerial platform behavior. Training on viewpoint-appropriate data is critical — a network trained on ground-level views won't generalize to drone perspectives. Urban scenes (#6) add building-scale obstacles (urban canyons) that drive richer camera planning.

## Design

### Architecture

```
infinigen/core/placement/
├── flight_camera.py         # Flight camera rig + sensor models
├── flight_trajectories.py   # Platform-specific motion models
├── flight_sensor.py         # Sensor characteristics, degradation
└── flight_configs/          # GIN configs per platform type

Extensions to existing:
├── camera.py                # Add flight modes to existing camera system
├── animation_policy.py      # Add flight animation policies
└── camera_trajectories.py   # Add flight trajectory generators
```

### Component 1: Platform Types

```python
class FlightPlatform(Enum):
    ISR_ORBIT    = "isr_orbit"     # Orbiting ISR drone, gimbaled sensor
    ISR_RASTER   = "isr_raster"    # Raster-scan survey drone
    ISR_LOITER   = "isr_loiter"    # Loitering/stationary ISR
    ISR_PLANE    = "isr_plane"     # Fixed-wing ISR aircraft (high alt, fast, long endurance)
    FPV_RACING   = "fpv_racing"    # High-speed aggressive FPV
    FPV_SCOUT    = "fpv_scout"     # Slower reconnaissance FPV
    UGV_WHEELED  = "ugv_wheeled"   # Wheeled ground vehicle (0.5-2m AGL)
    UGV_TRACKED  = "ugv_tracked"   # Tracked ground vehicle (0.5-2m AGL)
    SATELLITE    = "satellite"     # Orbital push-broom
```

### Component 2: Camera Rig Configurations

Each platform defines a `FlightRigSpec`:

```python
@gin.configurable
@dataclass
class FlightRigSpec:
    platform: FlightPlatform
    altitude_range: tuple[float, float]       # meters AGL
    speed_range: tuple[float, float]          # m/s
    sensor_fov_range: tuple[float, float]     # degrees HFOV
    look_angle_range: tuple[float, float]     # degrees from nadir (0=nadir, 90=horizon)
    gimbal_stabilization: bool                # Sensor stabilized vs body-fixed
    multi_sensor: bool                        # EO + IR co-boresighted rig
    sensor_baseline: float                    # Meters, for stereo/multi-sensor
```

**Platform-specific defaults**:

| Platform | Altitude | Speed | FOV | Look angle | Gimbal |
|----------|---------|-------|-----|-----------|--------|
| ISR Orbit | 100-500m | 15-40 m/s | 30-60° | 30-60° from nadir | Yes |
| ISR Raster | 80-300m | 20-50 m/s | 40-80° | 0-10° from nadir | Yes |
| ISR Loiter | 50-500m | 0-5 m/s | 20-60° | 45-90° from nadir | Yes |
| ISR Plane | 500-5000m | 50-150 m/s | 5-30° | 30-60° from nadir | Yes |
| FPV Racing | 5-50m | 30-80 m/s | 80-120° | 0-30° from forward | No (body-fixed) |
| FPV Scout | 10-100m | 10-30 m/s | 60-100° | 0-20° from forward | Partial |
| UGV Wheeled | 0.5-2m | 2-15 m/s | 60-100° | 0-20° from forward | Partial (pitch only) |
| UGV Tracked | 0.5-2m | 1-10 m/s | 60-100° | 0-20° from forward | Partial (pitch only) |
| Satellite | 200-500km | 7000 m/s | 1-5° | <1° from nadir | Yes |

### Component 3: Flight Motion Models (`flight_trajectories.py`)

**ISR Orbit** (`ISROrbitPolicy`):
- Circular orbit around a center point at fixed radius
- Radius = horizontal distance from center to maintain given look angle at altitude
- Altitude varies slowly with sinusoidal perturbation (±10%)
- Camera gimbaled to point at orbit center (or offset point)
- Generates smooth circular trajectories with configurable number of orbits

**ISR Raster** (`ISRRasterPolicy`):
- Back-and-forth raster scan pattern (lawnmower)
- Strips parallel to X or Y axis with configurable overlap (30-60%)
- Camera points nadir (top-down mapping)
- Returns at end of each strip for next pass
- Configurable strip length, spacing, overlap

**ISR Loiter** (`ISRLoiterPolicy`):
- Near-stationary with slow drift
- Orbit around a POI at very tight radius
- High-res stare at single location from multiple angles
- Altitude may vary slowly

**FPV Racing** (`FPVRacingPolicy`):
- Aggressive forward flight with banking in turns
- Trajectory: series of waypoints with spline interpolation
- Bank angle proportional to turn curvature (up to 60°)
- Camera body-fixed: tilts with vehicle
- High velocity variation (acceleration through open areas, deceleration near obstacles)
- Terrain-following at low altitude with configurable AGL clearance

**FPV Scout** (`FPVScoutPolicy`):
- Slower, more stable forward flight
- Waypoint-based path with gentle turns
- Camera partially stabilized (limited gimbal range)
- Can pause and orbit POIs briefly

**ISR Plane** (`ISRPlanePolicy`):
- Fixed-wing aircraft dynamics: constant forward velocity, coordinated turns (banked, constant radius)
- High altitude (500-5000m AGL) with wide-area coverage
- Flight patterns: race-track (parallel strips with 180° turns at ends), expanding spiral, or loiter circle around POI
- Camera in gimbaled turret (pan/tilt stabilized, independent of aircraft attitude)
- Turn radius constraint: min_radius = v² / (g × tan(max_bank)), typical bank angle 15-30°
- Multiple sensor payloads: wide-area search camera + narrow-FOV spotter camera on same platform
- Straight-and-level segments between turns (stable imaging periods, configurable duration)

**UGV Wheeled** (`UGVWheeledPolicy`):
- Ground vehicle following roads, paths, or cross-country terrain
- Altitude: fixed 0.5-2m above ground (terrain-following with suspension model)
- Motion: steering-based path following (Ackermann geometry), forward velocity with speed-dependent turn rate
- Suspension bounce: vertical oscillation with configurable frequency/amplitude (1-3 Hz, ±5cm)
- Camera: forward-facing or roof-mounted, limited pitch gimbal (±20°), no roll gimbal
- Road following: prefer OSM road paths when available, cross-country when off-road
- Speed variation: slow through urban areas, faster on open terrain
- Dust kick-up: particle emitter at wheel contact points when on unpaved surfaces (configurable density)

**UGV Tracked** (`UGVTrackedPolicy`):
- Similar to wheeled but with skid-steering motion model (differential track speed)
- Slower, more vibration (track clatter, 5-15 Hz)
- Better off-road capability (steeper slopes, rougher terrain)
- Camera stabilization less effective (more high-frequency jitter)
- Dust/debris: more aggressive ground disturbance than wheeled

**Satellite** (`SatellitePolicy`):
- Straight-line push-broom trajectory
- Nadir-pointing camera
- TDI (time delay integration) simulation: frames rendered at regular along-track intervals
- Orthographic projection approximation (long focal length → low perspective distortion)
- Ground sample distance (GSD) computed from altitude + pixel pitch: GSD = altitude × pixel_pitch / focal_length

### Component 4: Motion Model Implementation

All flight policies extend `AnimPolicy` (from `animation_policy.py`):

```python
class AnimPolicyFlight(AnimPolicy):
    def __init__(self, rig_spec: FlightRigSpec, terrain_bvh, ...):
        self.rig_spec = rig_spec
        self.terrain_bvh = terrain_bvh
    
    def propose_pose(self, frame: int) -> Pose:
        """Return rig pose for given frame."""
    
    def validate_pose(self, pose: Pose) -> bool:
        """Check pose validity (terrain clearance, not inside geometry)."""
```

**Terrain following**: All low-altitude platforms use terrain height queries via BVH raycasting to maintain minimum AGL clearance. The existing BVH infrastructure in `infinigen/core/placement/camera.py` is reused.

**Obstacle avoidance**: For FPV/ISR platforms, check line-of-sight to trajectory ahead. If blocked, generate RRT* shortcut (reuses existing `rrt.py` path planner).

**Keyframe generation**: Trajectories are output as Blender keyframes on the camera rig empty, exactly as existing animation policies do. This ensures seamless integration with the existing rendering pipeline.

### Component 5: Sensor Characteristics (`flight_sensor.py`)

Simulate real sensor artifacts and limitations:

```python
@gin.configurable
@dataclass
class SensorCharacteristics:
    # Optical
    focal_length_mm: float
    aperture: float                     # f-number
    hfov_deg: float
    
    # Detector
    resolution: tuple[int, int]         # pixels
    pixel_pitch_um: float               # microns
    
    # Degradations
    motion_blur: bool                   # Simulate integration-time blur
    rolling_shutter: bool               # Simulate rolling shutter (FPV)
    rolling_shutter_mode: str           # "subframe" (accurate, expensive) or "compositor" (fast, approximate) or "none"
    lens_distortion: bool               # Apply radial distortion
    vibration_profile: str              # "none", "light", "heavy"
    noise_model: str                    # "none", "gaussian", "photon"
    lens_flare: bool                    # Simulate lens flare and veiling glare
    saturation_behavior: str            # "clip" or "bloom" (charge bleeding to adjacent pixels)
    compression: str                    # "none", "jpeg_85", "h264_medium" (post-render degradation)
```

**Motion blur**: Applied via Cycles with configurable shutter time. Fast FPV platforms get more blur. Shutter time is constrained to be physically plausible for the platform: ISR raster at 30 fps gets max 33ms shutter; FPV at 80 m/s with 100° HFOV at 5m gets <1ms to avoid total frame smear.

**Rolling shutter**: Two modes:
- **Subframe mode** (accurate): Render N sub-frames at evenly spaced times within the frame interval, then assemble using per-scanline subsampling. This correctly handles occlusion changes between scanlines — a fast-moving building that enters frame mid-exposure appears correctly sheared. Compute cost: N × single frame render. Use for high-fidelity datasets.
- **Compositor mode** (fast, approximate): Per-row time offset applied in post-processing. No occlusion handling — a building that enters frame between scanlines always appears or doesn't, rather than being partially occluded. Acceptable for slow platforms (UGV, ISR loiter). Inadequate for FPV racing (>50 m/s) where rolling shutter is the dominant artifact.

**Vibration**: Apply high-frequency low-amplitude jitter to camera transform keyframes. Configurable amplitude vs frequency profile.

**Lens distortion**: Apply radial distortion model (k1, k2, k3) in post-processing on rendered frames. Distortion parameters stored in metadata for undistortion.

**Lens flare and veiling glare**: When the sun is within ±30° of the camera FOV (common for dawn/dusk ISR orbits, satellite with low sun angle):
- Veiling glare: global contrast reduction proportional to sun→FOV angle (configurable attenuation curve)
- Ghost reflections: 2-4 aperture ghost images at positions symmetric around frame center relative to sun position
- Configurable per platform: ISR gimbaled sensors typically have lens hoods (less flare); FPV wide-angle lenses have more

**Sensor saturation bloom** (IR only, spec #1): When pixel irradiance exceeds well capacity:
- `clip` mode: hard clip at saturation (unrealistic but fast)
- `bloom` mode: excess charge bleeds to adjacent pixels in column-parallel direction (vertical streaks for CMOS). Configurable bloom factor.

**Compression artifacts**: Optional post-render degradation simulating real video pipelines:
- `jpeg_85`: JPEG compression at quality 85 — DCT blocking artifacts at high-contrast edges
- `h264_medium`: H.264 compression at medium bitrate — inter-frame compression ringing, motion-adaptive quantization
- Applied after all other passes, before metadata write. Metadata records compression parameters.

### Component 6: Multi-Sensor Rig

For EO + IR co-boresighted platforms (ISR, satellite):

```python
@dataclass
class MultiSensorRig:
    eo_camera: bpy.types.Object       # Main EO camera
    ir_cameras: dict[str, bpy.types.Object]  # "lwir", "mwir", "swir" cameras
    baseline_mm: float                 # Physical separation between EO and IR
```

- All cameras parented to same rig empty → share trajectory
- Small physical offset between EO and IR sensors (configurable)
- IR cameras use IR-specific sensor specs from `IRSensorSpec` (#1)
- Per-frame extrinsics saved for each sensor
- Integrates with feature #1 (IR rendering) — multi-sensor rig triggers multi-pass rendering

### Integration

- Flight rigs are spawned by `spawn_camera_rigs()` with a `rig_type` parameter
- Flight trajectory policies replace standard `AnimPolicy*` classes when flight mode is selected
- All flight modes are gin-configurable; existing indoor/nature configs work unchanged
- Multi-sensor rig output handled by `render_image()` — render multiple passes (EO + IR bands)

### Testing Strategy

1. **Trajectory validity**: All generated trajectories maintain terrain clearance and avoid geometry penetration
2. **Coverage**: ISR raster covers the specified area with configured overlap
3. **Platform realism**: FPV banking matches turn curvature within ±5° tolerance
4. **Satellite**: GSD matches computed value, push-broom frames show correct along-track overlap
5. **Multi-sensor**: EO and IR frames share trajectory, offsets match configured baseline
6. **Rendering**: Flight trajectories render correctly in both Cycles and EEVEE
7. **Existing tests**: Non-flight camera modes unaffected

### Dependencies

- Extends existing `camera.py`, `animation_policy.py`, `camera_trajectories.py`, `rrt.py`
- Integrates with #1 (IR rendering) for multi-sensor EO/IR rigs
- Integrates with #5 (vispos pipeline) for platform-specific dataset organization
- No dependency on #2 (damage) or #3 (seasons)

### Open Questions

1. Should satellite mode actually simulate orbital mechanics (Keplerian elements) or just do straight-line push-broom? Start with straight-line for simplicity.
2. FPV rolling shutter: is the per-scanline time offset worth the rendering overhead? Initial: approximate with motion blur + metadata flag.
3. Does the satellite mode need atmospheric correction (Rayleigh scattering, aerosols at nadir)? Initial: use existing volume rendering with tuned density.
