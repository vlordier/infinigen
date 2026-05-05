# Spec: Seasonal & Temporal Enhancement

**Feature branch**: `feature/seasons-and-time`
**Date**: 2026-05-05
**Status**: Draft

## Summary

Extend infinigen's season system from plant-only to full-scene seasonal variation. Add structured time-of-day presets with appropriate atmosphere and lighting. Enable systematic combined variation across the season×TOD matrix for diverse domain randomization.

## Motivation

Infinigen already has plant-centric season selection (leaf color changes per season) and basic sun elevation control. For visual positioning, these need to be:
- **Comprehensive**: Seasons must affect terrain, lighting, water, and atmosphere — not just trees
- **Systematic**: Generate the same location across multiple seasons and times of day with matched cameras
- **Temporally consistent**: Within a video sequence, the environment state must remain consistent (snow doesn't appear mid-sequence, sun doesn't jump)

Operations in GNSS-denied environments happen at any hour, in any season. Training must cover this variance.

## Design

### Architecture

```
infinigen/assets/weather/
├── season_system.py         # Central season state + scene-wide effects
├── seasonal_terrain.py      # Terrain material variation by season
├── seasonal_water.py        # Water state (liquid/frozen) by season
├── seasonal_lighting.py     # Sky color, sun path, ambient by season
├── seasonal_atmosphere.py   # Fog/haze/turbidity by season
├── time_of_day.py           # TOD presets and sun path computation
├── combined_variation.py    # Season×TOD matrix generator
└── configs/                 # GIN configs per season, TOD, combination
```

### Component 1: Season State

Central `SeasonState` object that carries season information throughout the pipeline:

```python
@gin.configurable
@dataclass
class SeasonState:
    season: str                      # "spring", "summer", "autumn", "winter"
    temperature: float               # °C, drives snow/ice/fog behavior
    precipitation: float             # 0-1 intensity
    snow_cover: float                # 0-1 fraction (winter = 0.5-1.0)
    vegetation_phase: float          # 0-1 (0=dormant, 1=peak growth)
    day_length_hours: float          # Hours of daylight
    ground_is_frozen: bool           # Affects terrain material
    water_is_frozen: bool            # Rivers/lakes frozen in winter
```

This replaces the current `random_season()` return value and flows through all generation stages instead of being consumed only by tree generation.

### Component 2: Seasonal Terrain (`seasonal_terrain.py`)

**Snow cover**: When `snow_cover > 0`:
- Apply snow material to terrain faces with upward normals (existing `snow_layer.py` mechanism, extended)
- Snow line elevation: snow appears above a threshold altitude that descends as `snow_cover` increases
- Transition zone: mix between snow and ground material based on altitude + slope

**Vegetation color**: When `vegetation_phase < 1.0`:
- Desaturate and brown-shift grass, ground cover, and undergrowth materials
- Expose soil through thinning vegetation in winter (reduce small scatter density)
- Autumn: warm color shift (reds, oranges, yellows) on deciduous vegetation

**Terrain wetness**: Affects soil material roughness and albedo:
- Spring: wet, dark soil, high specular (mud/puddles)
- Summer: dry, lighter soil, diffuse
- Autumn: moderate
- Winter: frozen or snow-covered

### Component 3: Seasonal Water (`seasonal_water.py`)

**State transition**:
- `water_is_frozen = True` (winter): Replace water surfaces with ice material. Rivers become solid ice paths. Lakes become flat ice fields.
- Partial freeze: Transition zones with ice patches on water surfaces

**River flow variation**:
- Spring: High flow (snowmelt), wider rivers, more turbulence
- Summer: Medium flow
- Autumn: Medium-low flow
- Winter: Frozen or very low flow

### Component 4: Seasonal Lighting (`seasonal_lighting.py`)

**Sun path variation** (affects `nishita_lighting()` in `sky_lighting.py`):
| Season | Max elevation | Dominant color temp | Daylight hours |
|--------|--------------|---------------------|---------------|
| Spring | 50-60° | 5500K | 12-14h |
| Summer | 60-75° | 5500-6500K | 14-16h |
| Autumn | 30-45° | 4500-5000K | 10-12h |
| Winter | 15-25° | 5500-6500K (clear), 4000-5000K (overcast) | 8-10h |

**Sky appearance**:
- Seasonal turbidity (atmospheric scattering): spring (medium), summer (low, clear), autumn (medium-low), winter (variable, often overcast)
- Ground albedo feedback (snow cover → brighter ambient from ground bounce)

### Component 5: Time-of-Day System (`time_of_day.py`)

Structured TOD presets replacing raw sun_elevation:

```python
class TimeOfDay(Enum):
    DAWN         = "dawn"         # sun at -6° to 0°, civil/nautical twilight
    MORNING      = "morning"      # sun at 5° to 25°
    NOON         = "noon"         # sun at max elevation (season-dependent)
    AFTERNOON    = "afternoon"    # sun at 25° to 60°
    DUSK         = "dusk"         # sun at 0° to -6°
    NIGHT        = "night"        # no direct sun, moon + stars
```

Each preset defines:
- Sun elevation range (randomized within range)
- Sun azimuth (can be fixed or randomized)
- Sky model parameters (haze, air, dust density for Nishita)
- Ambient light level
- Moon lighting for night scenes (optional)
- Artificial lighting state for indoor scenes (on/off, intensity)

**Night-specific**:
- Moon as secondary light source (position = sun elevation + 180° azimuth)
- Starfield background
- Reduced visibility (fog/darkness gradient)
- Indoor scenes: artificial lights ON, emissive materials active

### Component 6: Combined Variation (`combined_variation.py`)

Systematic generation across the season×TOD matrix:

```python
@gin.configurable
def generate_variation_matrix(
    seasons: list[str] = ["spring", "summer", "autumn", "winter"],
    times_of_day: list[str] = ["dawn", "morning", "noon", "afternoon", "dusk", "night"],
    scene_count: int = 10,
) -> list[tuple[str, str]]:
    """Return list of (season, tod) pairs to generate.
    If scene_count < len(seasons)*len(times_of_day), sample randomly.
    If scene_count >= full matrix, cycle through all combinations."""
```

For paired generation (same scene, different conditions):
1. Generate scene with one (season, TOD) combination
2. Snapshot camera trajectories
3. Re-apply environment for each target (season, TOD) combination
4. Render all variants

This is critical for visual place recognition: the same location viewed at dawn in winter vs noon in summer.

### Component 7: Temporal Consistency

For video sequences within one (season, TOD) combination:
- Time progression locked to frame index: TOD advances smoothly across frames (sunset/sunrise sequences)
- Weather state persistent: cloud positions, snow accumulation, water state don't change mid-sequence
- Existing `dynamic` mode in `nishita_lighting()` handles smooth sun animation

### Integration

- `SeasonState` is created at scene composition start, passed to all stages
- Replaces direct calls to `random_season()` with `SeasonState` parameter
- Existing tree season code continues to work — `SeasonState` feeds into `random_season()`
- `TimeOfDay` is resolved to sun_elevation/sun_azimuth before passing to `nishita_lighting()`
- Combined variation is a `job_config` level concern — managed by `manage_jobs.py`

**Backward compatibility**: When not using the enhanced system, existing behavior is preserved. `SeasonState` defaults produce same output as current `random_season()`.

### Testing Strategy

1. **Visual regression**: Same scene rendered in all 4 seasons, verify each looks distinct and seasonally appropriate
2. **TOD consistency**: Same scene at dawn/noon/dusk/night, verify sun position, sky color, and ambient light match expectations
3. **Frozen water**: Verify rivers/lakes correctly transition to ice in winter
4. **Snow coverage**: Verify snow appears on upward-facing surfaces above snow line
5. **Determinism**: Same seed + (season, TOD) → identical output
6. **Existing tests pass**: Season system enhancement doesn't break existing generation

### Dependencies

- Extends existing `sky_lighting.py`, `snow_layer.py`, tree season system
- No dependency on other new features (#1-5)
- But feeds combined variation into #5 (vispos pipeline)

### Open Questions

1. Should seasonal variation include different tree species composition (e.g., deciduous vs evergreen ratio)? Initial scope: no, keep species fixed per scene.
2. Night lighting: how to handle scenes that have no indoor structures (pure nature)? Moon + starfield only. Low light = grainy rendering (like real night vision). Should we boost Cycles samples for night scenes?
3. Should seasonal fog/haze be tied to time-of-day? Morning fog that burns off by noon? Add as a secondary stage — initial release ties fog to season only.

**Feature-sparse seasonal variants**: Beyond the standard seasons, certain environments become effectively featureless under specific seasonal conditions:
- **Winter overcast on snowfield** (`nature_snowfield` + winter + overcast): White terrain, white sky, zero contrast. Extreme feature poverty scenario for visual navigation.
- **Summer haze on open ocean** (`nature_ocean` + summer + haze): Horizon blending into sky, wave-only features.
- **Winter on barren steppe** (`nature_barren_steppe` + winter): Snow-covered flat terrain with no vertical features beyond the horizon.

These are controlled by combined season×scene_type parameters in the dataset spec rather than being separate season presets.
