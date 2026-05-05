# Spec: Disaster Damage System

**Feature branch**: `feature/disaster-damage`
**Date**: 2026-05-05
**Status**: Draft

## Summary

Add procedural damage to infinigen scenes simulating earthquake and war destruction. Generate paired intact/damaged versions of the same scene with identical camera poses, enabling training of visual positioning models robust to catastrophic structural change.

## Motivation

Visual positioning must work after disasters when pre-event reference imagery (maps, satellite) no longer matches current conditions. Training requires paired data: the same location viewed from the same position, with and without damage. No existing synthetic dataset provides this at scale with full ground truth.

Two damage regimes:
- **Earthquake**: Widespread structural damage — collapsed buildings, cracked infrastructure, landslides, toppled objects, rubble fields
- **War**: Localized destruction — blast craters, building facade destruction, fire damage, smoke, debris, destroyed vehicles

## Design

### Architecture

```
infinigen/assets/damage/
├── __init__.py
├── damage_system.py         # DamageStageExecutor, damage application pipeline
├── state_snapshot.py        # Scene state serialization for paired generation
├── earthquake/
│   ├── building_collapse.py # Structural failure physics
│   ├── ground_fracture.py   # Terrain cracks, landslides
│   ├── topple.py            # Object toppling, furniture displacement
│   └── rubble.py            # Rubble generation from destroyed meshes
├── war/
│   ├── craters.py           # Blast craters (terrain + buildings)
│   ├── facade_damage.py     # Building facade destruction (holes, scorch)
│   ├── fire_effects.py      # Fire damage, scorch marks, smoke
│   └── debris.py            # Debris field generation
├── shared/
│   ├── fracture.py          # Mesh fracture utilities (Voronoi shatter)
│   ├── displacement.py      # Mesh displacement/deformation
│   └── constraints.py       # Physics constraint helpers
└── configs/                 # GIN configs per damage type and severity
```

### Component 1: Paired Scene Generation Protocol

The core requirement is generating two versions of the same scene (intact + damaged) with matching camera trajectories.

**Protocol**:
1. Generate intact scene normally (existing `compose_nature()` / `compose_indoors()` pipeline)
2. Place camera rigs and compute camera trajectories on intact scene
3. **Snapshot** scene state: serialize all object transforms, mesh data, material assignments, and camera keyframes
4. **Apply damage**: Run damage operators on the scene, modifying geometry and materials
5. **Validate cameras**: For each camera pose, verify it's not inside new geometry (SDF check, BVH raycast). If occluded, apply minimal rerouting via RRT* along original trajectory
6. Render both versions with identical camera parameters

The key insight: **damage is applied AFTER camera placement on the intact scene**. Cameras are selected based on the intact geometry. The damaged scene reuses the same camera data but may need minor trajectory adjustments if the original camera path is now inside rubble.

**Scene snapshot format**:
```python
@dataclass
class SceneSnapshot:
    objects: dict[str, SnapshottedObject]   # name → transform + mesh hash
    cameras: dict[str, SnapshottedCamera]   # name → intrinsic + keyframes
    seed: int                               # RNG seed for reproducibility
    damage_config: dict                     # What damage was applied
```

### Component 2: Earthquake Damage

**Building Collapse** (`earthquake/building_collapse.py`):

For indoor scenes (rooms, structures):
- Identify load-bearing walls via constraint graph (walls with `StableAgainst` relations)
- Apply horizontal displacement to wall segments (simulated seismic shear)
- Fracture walls that exceed stress thresholds using Voronoi shatter
- Let fractured pieces fall under gravity (rigid body simulation, Blender physics)
- Generate rubble field from fragments that hit the floor
- Remove or damage ceiling panels

For outdoor structures: same principle applied to procedurally-placed structures.

Severity levels:
- **Mild**: Cracks in walls, minor object displacement, some toppled items
- **Moderate**: Partial wall collapse, significant rubble, most furniture displaced
- **Severe**: Total structural collapse, building reduced to rubble pile

**Ground Fracture** (`earthquake/ground_fracture.py`):
- 2D crack network on terrain surface (Voronoi-based fault lines)
- Vertical displacement along crack edges
- Landslide simulation on steep slopes (material creep downslope)
- Road damage: surface cracks, buckling, subsidence

**Object Toppling** (`earthquake/topple.py`):
- Identify tall/narrow objects (furniture, poles, towers, trees)
- Apply random rotation + translation with gravity settlement
- Constraint-based: objects "fall" in the direction of seismic acceleration
- Inter-object collision avoidance

**Rubble Generation** (`earthquake/rubble.py`):
- From destroyed structures: fragment original mesh, scatter fragments in radius
- Rubble material: mix of concrete, wood, metal fragments matching source materials
- Dust particle system triggered by collapse events

### Component 3: War Damage

**Blast Craters** (`war/craters.py`):

Two modes:
- **Terrain craters**: Apply to terrain mesh via SDF-based subtraction (spherical deformation with ejecta lip). Configurable diameter, depth, ejecta radius.
- **Building craters**: Cut spherical holes through building meshes. Apply scorch material to crater interior.

Crater placement: random within scene bounds, weighted toward structures (buildings, roads). Multiple craters per scene.

**Facade Damage** (`war/facade_damage.py`):
- Procedural holes in building walls (Boolean cut with randomized shapes)
- Exposed rebar geometry in larger holes
- Scorch/soot material application around holes (darkened, charred)
- Partial collapse of damaged sections
- Broken windows with glass shard scattering

**Fire Effects** (`war/fire_effects.py`):
- Reuses existing `fluid.py` fire simulation (`set_fire_to_assets()`) but adds persistent scorch marks
- Post-fire surface darkening (char material blend on affected surfaces)
- Smoke column particle system (reuses existing `FallingParticles` with buoyant smoke params)
- Burnt-out vehicle hulls as static debris assets

**Debris Generation** (`war/debris.py`):
- Rubble similar to earthquake but more localized (near craters and damaged facades)
- Added: military debris assets (destroyed vehicles, equipment fragments)
- Debris scattered with physically-based distribution (farther from crater = smaller fragments)

### Component 4: Damage Application Pipeline

`DamageStageExecutor` extends the existing `RandomStageExecutor` pattern:

```python
class DamageStageExecutor(RandomStageExecutor):
    def __init__(self, damage_types, severity_range):
        """
        damage_types: list of DamageType enums (EARTHQUAKE, WAR, BOTH)
        severity_range: (min, max) for damage severity
        """
```

Pipeline stages (runs after standard composition, before population):
1. `snapshot_intact_state()` — save transforms + cameras
2. `apply_terrain_damage()` — craters, ground fracture, landslides
3. `apply_structure_damage()` — building collapse, facade damage
4. `apply_object_damage()` — toppling, displacement
5. `generate_rubble()` — rubble fields from destroyed geometry
6. `apply_surface_damage()` — scorch marks, charring, soot
7. `apply_particle_damage()` — dust, smoke emitters
8. `validate_camera_trajectories()` — check and repair camera paths
9. `finalize_damaged_scene()`

### Component 5: Camera Trajectory Repair

After damage, some camera poses may be invalid (inside collapsed geometry, occluded by rubble). Repair strategy:

1. **SDF check**: If camera position is inside any mesh SDF → invalid
2. **Raycast check**: Cast rays to scene center from camera. If all rays hit within 1m → camera is buried/trapped → invalid
3. **Repair**: For invalid poses, trace backward along original trajectory to last valid pose. From there, plan short RRT* path to reconnect with original trajectory further ahead. Preserve as much of the original trajectory as possible.
4. **Minimal change**: Camera repair is a last resort. Track which frames were repaired in metadata.

### Integration

- Hooks into `compose_nature()` after standard stages but before `populate_scene()` (damage operates on coarse geometry)
- Paired generation runs the scene pipeline twice: once without damage, once with damage
- `DamageStageExecutor` is gin-configurable and can be inserted into any scene generation config
- When IR rendering is present (#1), damage includes thermal signature changes (exposed hot interior surfaces in crater walls, warm rubble)

### Testing Strategy

1. **Determinism**: Same seed + config → identical damage output
2. **Camera validity**: Post-damage, all camera poses pass SDF and visibility checks
3. **Mesh integrity**: Damaged meshes are manifold (no non-manifold edges introduced by fracture)
4. **Paired consistency**: Non-damaged elements are identical between paired scenes (same tree positions, same terrain outside damage zone)
5. **Visual quality**: Rendered damaged scenes pass manual spot-check (no floating geometry, no Z-fighting)
6. **Scale test**: Generate 100 paired scenes, verify no pipeline crashes

### Dependencies

- Requires indoor constraint solver to identify structural elements (wall relationships) — already exist in `infinigen/core/constraints/`
- Uses Blender's rigid body physics for collapse simulation (bpy built-in)
- Uses Blender mesh Boolean operations for crater/facade cutting (bpy built-in)
- Fire simulation reuses existing `infinigen/assets/fluid/fluid.py`

### Open Questions

1. Should damaged buildings retain their Room/object constraint structure for the indoor solver? No — damage is applied post-composition; the solver is not rerun.
2. How to handle damage to procedurally-generated trees? Apply trunk fracture + crown fall for earthquake; scorch + partial burn for war.
3. Should damage be applied to populated (high-detail) or coarse geometry? Initial design: coarse geometry, for performance. Damage detail is inherent in the damage operators (rubble fragments are detailed enough).
