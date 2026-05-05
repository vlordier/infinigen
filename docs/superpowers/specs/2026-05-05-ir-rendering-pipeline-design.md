# Spec: IR Rendering Pipeline

**Feature branch**: `feature/ir-rendering`
**Date**: 2026-05-05
**Status**: Draft

## Summary

Extend infinigen's rendering pipeline to produce LWIR (8-14µm), MWIR (3-5µm), and SWIR (1-2.5µm) imagery alongside the existing EO (RGB) output. Support two fidelity levels: full thermal physics simulation (steady-state heat equation + Planck radiance) and a fast heuristic approximation. Deliver radiometrically calibrated multi-band EXR passes.

## Motivation

Visual positioning in GNSS-denied environments relies on sensors that work across day/night and through obscurants. IR bands are critical for:
- **LWIR/MWIR**: Night operations, thermal contrast between warm objects and cold background
- **SWIR**: Penetrating haze/smoke, nightglow imaging, material discrimination

Training IR-based vision models requires large volumes of labeled IR imagery paired with ground-truth pose — exactly what infinigen's procedural pipeline can provide. No existing open-source synthetic data generator produces calibrated multi-band IR with per-pixel ground truth.

## Design

### Architecture

```
infinigen/assets/materials/thermal/
├── properties.py          # Thermal property definitions, defaults per material class
├── solver.py              # Steady-state thermal solver
├── render.py              # IR sensor pass + radiometric calibration
├── heuristic.py           # Fast approximation pathway
└── configs/               # GIN configs for sensor bands, atmospherics

infinigen/core/rendering/
├── ir_pass.py             # IR render pass integration into existing pipeline
└── ir_sensor.py           # Sensor response model (MTF, NETD, spectral)
```

### Component 1: Thermal Material Properties

Assign every material in `infinigen/assets/materials/` thermal properties via a `ThermalProperties` dataclass registered alongside the existing shader:

```python
@dataclass
class ThermalProperties:
    emissivity: float              # 0-1, per-band optional
    solar_absorptivity: float      # 0-1, for solar loading
    thermal_conductivity: float    # W/(m·K)
    heat_capacity: float           # J/(kg·K)
    density: float                 # kg/m³
```

**Strategy**: Add `thermal_properties` as an optional attribute on existing material classes. For materials that don't specify properties, infer from material category defaults (metal→high conductivity/reflectivity, wood→low conductivity, vegetation→high emissivity).

**Per-band emissivity**: Support band-specific emissivity arrays `(emiss_lwir, emiss_mwir, emiss_swir)`. Most materials are gray-body (same emissivity across bands), but metals and coatings show spectral variation in SWIR/MWIR.

### Component 2: Thermal Solver

Steady-state heat equation solver operating on the scene's surface mesh:

```
For each face:
  Q_in  = Q_solar + Q_sky + Q_atmosphere + Q_internal
  Q_out = εσT⁴ + h(T - T_ambient)
  Solve Q_in = Q_out for T
```

**Inputs**:
- Sun position (azimuth, elevation) — already available from sky lighting
- Solar irradiance (configurable, default ~1000 W/m² direct + ~100 W/m² diffuse)
- Sky temperature (configurable, default ~250K for clear sky via Swinbank model)
- Ambient air temperature (configurable, default ~293K)
- Wind convection coefficient `h` (configurable, 5-25 W/m²·K)
- Per-face material thermal properties from Component 1
- Shadow map from sun position (face in sun vs shade → different Q_solar)

**Output**: Per-face equilibrium temperature `T`.

**Shadow map integration**: Use Blender's existing shadow raycasting or a low-res Cycles bake to determine which faces are sunlit vs shaded. Faces in shadow receive only diffuse solar (no direct beam).

**Optimization**: The solver is O(n_faces). For large terrain meshes (>1M faces), use numpy vectorized operations. Cache results per (scene, sun_pos, ambient_temp) tuple.

### Component 3: IR Sensor Render Pass

Three rendering modes:

**A. Full physics (Cycles)**:
- Replace all materials with blackbody emission shaders at the solved temperature T
- Apply per-band emissivity as emission strength multiplier
- Render with Cycles with atmospheric path radiance disabled (atmosphere added in post)
- Apply sensor spectral response curve as a wavelength-weighted integration

**B. Heuristic approximation (Cycles or EEVEE)**:
- Map material type → emissivity → plausible temperature range
- Add Perlin-noise temperature variation for texture (hot spots on vehicles, cool spots under shade)
- No full heat equation solve — just plausible temperature distributions

**C. Per-pixel radiometric calibration**:
- Convert Cycles radiance output to at-sensor radiance (W/m²·sr)
- Apply atmospheric transmission model (MODTRAN-style lookup table or simple exponential)
- Apply sensor model: MTF convolution, NETD noise, quantization

**Output passes** (per rendered frame):
| Pass | Content | Format |
|------|---------|--------|
| `IR_LWIR` | At-sensor radiance, 8-14µm integrated | 1ch EXR, float32 |
| `IR_MWIR` | At-sensor radiance, 3-5µm integrated | 1ch EXR, float32 |
| `IR_SWIR` | At-sensor radiance, 1-2.5µm integrated | 1ch EXR, float32 |
| `Temperature` | Per-pixel surface temperature (K) | 1ch EXR, float32 |
| `Emissivity` | Per-pixel emissivity | 1ch EXR, float32 |

### Component 4: Sensor Model

Configurable sensor parameters per band:

```python
@gin.configurable
class IRSensorSpec:
    band: str                        # "LWIR", "MWIR", "SWIR"
    spectral_range: tuple[float, float]  # µm
    f_number: float                  # e.g. 1.0
    focal_length: float              # mm
    pixel_pitch: float               # µm, e.g. 12.0
    netd: float                      # mK, noise equivalent temp diff
    well_capacity: int               # electrons
    integration_time: float          # ms
```

Applied in post-processing:
- MTF convolution (Gaussian approximation, configurable sigma)
- NETD → noise floor in radiance units
- Well capacity → saturation limit
- Optional: bad pixel simulation, non-uniformity correction residual

### Component 5: Environment Controls

Thermal environment is gin-configurable:

```python
# in infinigen/assets/materials/thermal/configs/environment.gin
IRThermalEnvironment.solar_direct_irradiance = 900.0      # W/m²
IRThermalEnvironment.solar_diffuse_irradiance = 100.0     # W/m²
IRThermalEnvironment.ambient_air_temperature = 293.15     # K
IRThermalEnvironment.wind_speed = 3.0                     # m/s
IRThermalEnvironment.sky_emissivity = 0.8
IRThermalEnvironment.ground_temperature = 288.15          # K
```

## Integration with existing pipeline

Hooks into the render stage of `infinigen/core/execute_tasks.py`:

1. After material assignment but before rendering: run thermal solver (`ThermalSolver.solve(scene, sun_pos, env)`)
2. Store per-face temperatures as vertex/face attributes on meshes
3. Before Cycles render: inject IR material overrides (`IRRenderer.apply_thermal_materials(scene, band)`)
4. After render: apply sensor calibration (`IRSensor.calibrate(render_output, sensor_spec)`)

The existing `render_image()` in `infinigen/core/rendering/render.py` supports multiple render passes — IR passes are added alongside existing RGB/depth/normal passes.

## Performance considerations

| Mode | Render cost vs RGB | Thermal solve cost |
|------|-------------------|-------------------|
| Physics IR (Cycles) | ~1.2x (emission-only, no path tracing) | O(n_faces), seconds |
| Heuristic IR (EEVEE) | ~0.1x (rasterization) | O(1), negligible |

For bulk dataset generation, the heuristic path with EEVEE is the default. Full physics is reserved for high-fidelity subsets.

### Testing Strategy

1. **Unit tests**: Thermal solver against analytical solutions (flat plate, Stefan-Boltzmann)
2. **Material tests**: Verify all existing material classes have valid thermal properties
3. **Integration tests**: Render a test scene (material balls) in all 3 IR bands, verify radiance values are physically plausible (no NaN, no negative, temperature in 200-400K range for outdoor scenes)
4. **Regression**: Existing RGB pipeline unaffected when IR not enabled
5. **Validation**: Compare against reference scenes (known materials at known temperatures)

### Known Limitations & Risk Assessment

The steady-state thermal solver has systematic simplifications. These are documented so dataset consumers understand what their models are (and aren't) learning:

| Limitation | Effect on IR imagery | Severity | Mitigation |
|-----------|---------------------|----------|------------|
| **No thermal inertia** | Temperature changes instantly with sun position. Shadow boundaries have sharp thermal edges rather than gradual gradients. | High | Add thermal time-constant per material in v2. For v1, apply Gaussian blur to temperature map at shadow boundaries as crude approximation. |
| **No sky reflections** | Low-emissivity surfaces (metal roofs, glass, water) appear as pure emitters. Misses reflected cold-sky radiance that dominates metal-roof IR signatures. | High | Document that metal/glass IR appearance is physically wrong. In v2, add specular reflection pass to IR render. |
| **No internal heat sources** | Buildings at night are at ambient (invisible in LWIR). Vehicles are cold. Streetlights emit no thermal radiation. Industrial areas flat. | Critical for night | Night IR data from this pipeline should not be used for thermal target detection training. For localization, night IR is still useful (terrain + sky contrast exists). |
| **Exponential-only atmosphere** | Path radiance (atmospheric emission) is not modeled. Significant for satellite altitudes. | Medium for ISR, high for satellite | For ISR altitudes (<5km), path radiance is modest. Satellite IR should use heuristic mode or be validated against real satellite IR. |
| **No solar reflection in MWIR/SWIR** | MWIR and SWIR have significant solar reflection components that the emission-only render misses. | High for MWIR/SWIR | MWIR/SWIR output from physics path is incomplete. Use heuristic path for MWIR/SWIR that includes configurable solar reflection ratio. |

**Recommendation for v1**: Focus on LWIR with physics path for daytime scenes (solar loading dominates, simplifications less severe). Use heuristic path for MWIR/SWIR and all night scenes. Invest in real IR data collection (200-500 frames of real ISR/drone IR with GPS) for validation before committing to the full physics implementation — validate that the simplified model transfers, then decide whether the remaining physics gaps are worth engineering.

**Thermal inertia approximation (v1)**: As a minimal fix for the sharp shadow boundary problem, apply a Gaussian blur (σ = 5-15 pixels at render resolution) to the temperature map at shadow boundaries. This crudely approximates the thermal smearing that real materials exhibit. Not physically correct, but it visually reduces the most obvious domain gap artifact.

### Dependencies

- None on other new features. Purely additive to rendering pipeline.
- Uses existing Blender Cycles GPU rendering, material system, and surface attribute infrastructure.

### Open Questions

1. Should internal heat sources (engines, electronics, bodies) be modeled? Initial scope: no, external solar-only. Night IR from v1 should be used for localization (terrain/sky contrast) but not for thermal target detection.
2. Atmospheric path radiance: full MODTRAN integration or simple empirical model? Start with simple exponential attenuation, upgrade path for MODTRAN. Path radiance is negligible at ISR altitudes (<5km) but significant for satellite.
3. Should foliage/vegetation include evapotranspiration cooling? Initial scope: no, treat vegetation as passive thermal mass.
