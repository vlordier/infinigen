# Spec: Urban Scene Generation

**Feature branch**: `feature/urban-scenes`
**Date**: 2026-05-05
**Status**: Draft

## Summary

Generate urban and suburban landscapes using a hybrid asset sourcing strategy: OSM data for city layouts, procedural generation for structural elements, AI image-to-3D and static libraries for detail objects. Provides the structural assets that damage, flight camera, and visual positioning features depend on.

## Asset Sourcing Strategy

Where assets come from, by category:

| Category | Source | Rationale |
|----------|--------|-----------|
| **Road networks** | OSM data (primary) + procedural infill (gap filling) | Real city layouts provide authentic road topology. Procedural fills gaps where OSM is sparse or for purely synthetic cities. |
| **Building footprints** | OSM data (primary) + procedural lot subdivision (fill) | OSM provides real building footprints where available. Procedural generates plausible infill for un-mapped areas. |
| **Building shells, facades, roofs** | Procedural (infinigen-style) | Extruded from footprints. Too many variations, must integrate with damage system, needs structural metadata. |
| **Bridges, power lines, streetlights, signs** | Procedural | Simple repeating geometry, placement logic is the hard part. |
| **Vehicles (cars, trucks, buses)** | Static asset library + AI image-to-3D | Complex curved surfaces impractical for procedural. Curated CC0 library for common types. AI-generated for variety. |
| **Street furniture (benches, bins, hydrants, planters)** | Static library + some AI-generated | Bespoke designs, small meshes. Procedural where simple (bollards, planters). |
| **Terrain** | Existing infinigen terrain system | Reuses what works. |
| **Vegetation** | Existing infinigen tree/plant factories | Reuses what works. |

**OSM integration**: A pre-processing step downloads OSM data for a given bounding box, parses road graphs and building footprints into the internal representation. OSM data is optional — the system falls back to fully procedural generation when OSM is unavailable or undesired.

**AI image-to-3D pipeline**: For generating variety in detail objects, use the latest image-to-3D models (e.g., TRELLIS, InstantMesh) to convert reference images into textured 3D meshes. A curation step filters outputs for quality. This produces meshes that feed into the static asset import pipeline. Generated assets are cached and reused across scenes.

**Static asset pipeline**: Extends infinigen's existing static asset support (`StaticAssets.md`). A `StaticAssetFactory` wraps imported meshes with the standard placeholder/populate pattern for LOD and camera-culling.

## Design

### Architecture

```
infinigen/assets/urban/
├── __init__.py
├── urban_scene.py            # Urban scene composer (top-level)
├── osm_loader.py             # OSM download, parse, project to local coords
├── road_network.py           # Road graph (OSM primary, procedural fallback)
├── road_mesher.py            # Road surface geometry from graph
├── intersection.py           # Intersection geometry
├── buildings/
│   ├── building_generator.py # Building placement + type selection
│   ├── exterior.py           # Building shell geometry (from footprints)
│   ├── facade.py             # Facade variation (windows, materials, style)
│   ├── roof.py               # Roof geometry (flat, pitched, domed, etc.)
│   ├── interior_bridge.py    # Integration with existing indoor solver
│   └── templates/            # Gin configs per building type
├── infrastructure/
│   ├── bridges.py            # Bridge generation over terrain gaps/rivers
│   ├── streetlights.py       # Street light placement along roads
│   ├── power_lines.py        # Power/utility poles and lines
│   ├── signage.py            # Traffic signs, billboards, signals
│   └── barriers.py           # Guardrails, fences, walls, jersey barriers
├── urban_scatter/
│   ├── vehicles.py           # Vehicle placement (from static + AI assets)
│   ├── street_furniture.py   # Benches, trash cans, mailboxes, planters
│   ├── parking.py            # Parking lot generation (lines, layout)
│   └── construction.py       # Construction sites, scaffolding, barriers
├── asset_pipeline/
│   ├── static_importer.py    # Static mesh import + Factory wrapper
│   ├── ai_asset_gen.py       # Image-to-3D pipeline
│   └── asset_cache.py        # Cached asset management
├── urban_surface.py          # Asphalt, concrete, sidewalk materials
├── urban_terrain.py          # Terrain flattening + modification for urban use
└── configs/                  # GIN configs for city types, building styles

infinigen_examples/
├── generate_urban.py         # Top-level urban scene driver
└── configs_urban/            # Urban scene gin configs
```

### Component 1: Road Network (`road_network.py`)

**Primary source: OpenStreetMap**. A pre-processing step (`osm_loader.py`) downloads OSM data for a bounding box via the Overpass API, parses ways tagged `highway=*` into the internal `RoadGraph`, and projects lat/lon to local coordinates.

**Fallback: procedural generation**. When OSM data is insufficient or disabled, generate synthetically:

1. **Seed points**: Place N seed nodes across scene bounds, weighted by terrain flatness
2. **Population centers**: Identify clusters of flat terrain → "city centers" with higher road density
3. **Highway backbone**: Connect population centers with arterial roads using A* on a cost grid (cost = slope × curvature)
4. **Grid infill**: Within each population center, generate a semi-regular grid of local roads with noise
5. **Suburban sprawl**: Radiating local roads from grid edges, more organic, more cul-de-sacs
6. **Terrain adaptation**: Roads follow terrain at configurable max grade (default 8%)

**OSM road tag mapping**:

| OSM tag | Internal type | Lanes (default) |
|---------|---------------|-----------------|
| `motorway`, `trunk` | Highway | 2-3 per side |
| `primary`, `secondary` | Arterial | 1-2 per side |
| `tertiary`, `residential`, `unclassified` | Local | 1 per side |
| `service`, `alley` | Alley | 1 total |

OSM data is not projected onto terrain — roads are snapped to the generated terrain with smoothing. This may introduce slight deviations from real road alignments, which is acceptable (we're using OSM for topology, not centimeter-accurate georegistration).

**Road types and parameters**:

| Type | Lanes | Width | Sidewalk | Min bend radius | Max grade |
|------|-------|-------|----------|-----------------|-----------|
| Highway | 2-3 per side | 24-36m | No | 200m | 6% |
| Arterial | 1-2 per side | 16-24m | Yes | 100m | 8% |
| Local | 1 per side | 10-14m | Yes | 30m | 10% |
| Alley | 1 total | 4-6m | No | 15m | 12% |

### Component 2: Road Mesher (`road_mesher.py`)

Convert the road graph into 3D geometry:

1. **Road surface**: Extrude each road edge into a ribbon mesh. Width = road type width. Follow terrain elevation with smoothing (vertical curves for grade changes).
2. **Crown**: Road cross-section has a slight crown (1-2% slope from center to edge) for drainage
3. **Sidewalks**: Raised 15cm above road surface, 1.5-3m wide, with curb geometry
4. **Intersections**: Boolean union of crossing road ribbons → single intersection mesh. Traffic circle option for arterial crossings.
5. **Lane markings**: Apply as separate thin meshes or use decal-like geometry on road surface (dashed center line, solid edge lines, crosswalks at intersections)
6. **Curb cuts**: At intersections, lower sidewalk to road level (accessibility ramps)

### Component 3: Building Generation (`buildings/`)

**Building types** (parameterized templates, not constraint-solved):

| Type | Floors | Floor height | Footprint | Roof style | Features |
|------|--------|-------------|-----------|------------|----------|
| Residential small | 1-2 | 2.8m | 80-200m² | Pitched, hip | Porch, garage, chimney |
| Residential medium | 3-6 | 2.8m | 150-500m² | Flat, pitched | Balconies, fire escapes |
| Residential tower | 8-20+ | 3.0m | 200-800m² | Flat | Elevator shaft, rooftop AC |
| Commercial retail | 1-2 | 4.0m | 200-1000m² | Flat | Storefront windows, awning |
| Commercial office | 3-15 | 3.5m | 300-2000m² | Flat | Glass curtain wall, lobby |
| Industrial warehouse | 1 | 6-10m | 500-5000m² | Flat/sawtooth | Roll-up doors, loading dock |
| Institutional | 2-6 | 3.5-4.5m | 500-5000m² | Varied | Columns, large windows |

**Building placement**:

1. **OSM footprints** (primary): When OSM data is available, parse `building=*` ways as building footprint polygons. OSM tags provide height/floors/type hints where available. Footprints are snapped to terrain elevation.
2. **Procedural lot subdivision** (fallback/infill): For blocks with no OSM coverage, subdivide blocks enclosed by roads into buildable lots. Lots are rectangular with random aspect ratio variation. Infill also fills gaps between OSM buildings within the same block.
3. **Setbacks**: Buildings are offset from lot edges by configurable setback distance (front/side/back). For OSM footprints, setbacks are already encoded in the footprint shape.
4. **Type assignment**: Building type assigned by:
   - OSM tags when available (`building=apartments` → residential tower, `building=commercial` → commercial, etc.)
   - Zonal inference when no OSM tags: commercial along arterials, residential in interior blocks, industrial at edges
5. **Height**: From OSM `height`/`building:levels` tags when available. Otherwise inferred from building type with ±20% variation.

**Building exterior geometry** (`exterior.py`):

Generated as a single watertight mesh per building:
1. Base extrusion: lot polygon extruded to building height
2. Roof addition: roof type applied to top face
3. Window/door cutouts: facade pattern subtracted via Boolean or applied as geometry
4. Details: window frames, sills, lintels, cornices, balconies added as separate geometry pieces

**Facade variation** (`facade.py`):

Each facade wall gets a procedural pattern:
- **Window grid**: Regular grid of windows with configurable spacing, size, and offset
- **Material**: Brick, concrete, stucco, glass curtain, or metal panel — selected by building type and zone
- **Ground floor differentiation**: Commercial ground floors have larger windows, different material, awnings
- **Procedural variation**: Window style (casement, sliding, fixed), sill depth, frame color randomized per building
- **Entrances**: 1-3 doorways per street-facing facade, with steps/ramp

**Roof generation** (`roof.py`):

| Style | Geometry | Buildings |
|-------|----------|-----------|
| Flat | Horizontal plane with parapet wall | Commercial, industrial, towers |
| Pitched (gable) | Two slopes meeting at ridge | Residential |
| Hip | Four slopes meeting at ridge and hips | Residential |
| Mansard | Steep lower + shallow upper slope | Institutional, European-style |
| Sawtooth | Asymmetric repeated ridges | Industrial |
| Domed/barrel | Curved surface | Institutional, special |

Roof materials: shingles (residential), membrane/gravel (commercial flat), metal (industrial), tile (regional variation).

**Interior bridge** (`interior_bridge.py`):

Connect building exteriors to the existing indoor constraint solver:
- For a subset of buildings (configurable fraction, e.g. 10-20% for performance), generate interior rooms using the existing `compose_indoors()` pipeline
- Building exterior shell defines the "room" bounds
- Existing furniture/object placement via constraint solver fills rooms
- Windows align between interior and exterior
- This is optional — most buildings are exterior-only. Interiors are only needed when camera trajectories go inside or when damage reveals interior structure.

### Component 4: Infrastructure (`infrastructure/`)

**Bridges** (`bridges.py`):
- Triggered when a road edge crosses a river or terrain depression
- Bridge types: beam bridge (short spans), truss bridge (medium spans), arch bridge (valleys)
- Deck width = road width + sidewalks
- Piers/supports generated at regular intervals
- Bridge deck follows road grade, supports extend to terrain below

**Streetlights** (`streetlights.py`):
- Placed along road edges at regular intervals (25-50m)
- Pole types vary by road type (highway→tall mast, local→short decorative)
- Light source: point or spot lights with configurable intensity, color temp
- Night rendering: lights activate for night TOD scenes

**Power lines** (`power_lines.py`):
- Distribution lines along roads, transmission lines cross-country (straight between towers)
- Utility poles (wood for local, steel lattice for transmission)
- Catenary wire curves between poles (gravity sag)
- Transformers, insulators as small detail assets

**Signage** (`signage.py`):
- Traffic signs at intersections (stop, yield, directional)
- Street name signs
- Highway signs (overhead gantry, roadside)
- Billboards along highways and arterials

### Component 5: Urban Scatter (`urban_scatter/`)

Detail objects are sourced from multiple pipelines:

| Source | What it provides | How it works |
|--------|-----------------|--------------|
| **Static asset library** | Vehicles (sedan, SUV, truck, van, bus, motorcycle), common street furniture | Curated CC0 meshes from Poly Haven, BlendSwap. Imported via `StaticAssetFactory` with Blender import. |
| **AI image-to-3D** | Vehicle variants, uncommon furniture, regional-specific objects | Reference images → image-to-3D model (TRELLIS, InstantMesh, or equivalent) → mesh decimation + UV repair → static asset import. Cached per asset type. |
| **Procedural** | Simple objects (bollards, barriers, planters, trash piles), parking lines | Generated via Blender Python. |

**AI asset pipeline** (`ai_asset_gen.py`):
1. Curate a library of ~50 reference images spanning vehicle types, furniture, urban objects
2. For each reference image, run image-to-3D model → outputs textured OBJ/GLB
3. Post-process: decimate to ~5K faces, repair UVs, scale to real-world size, validate watertight
4. Cache generated meshes in `~/.cache/infinigen/ai_assets/`
5. Feed into `StaticAssetFactory` with same interface as manually curated assets
6. Re-run periodically to refresh asset variety as models improve

**Vehicles** (`vehicles.py`):
- Placed along curbs (street parking) with configurable density, in parking lots, and in driveways
- Vehicle types drawn from asset library: sedan, SUV, truck, van, bus, motorcycle
- Type proportions configurable per zone (more trucks in industrial, more buses on arterials)
- Static only (no traffic simulation for v1)

**Street furniture** (`street_furniture.py`):
- Items: benches, trash bins, mailboxes, fire hydrants, planters, bicycle racks, bus shelters
- Source: static library for detailed items, procedural for simple (bollards, planters)
- Placement: density higher in commercial areas, along sidewalks, at bus stops

### Component 6: Urban Terrain & Surfaces (`urban_terrain.py`, `urban_surface.py`)

**Terrain preparation**:
1. **Flattening**: Areas designated for urban development are flattened to acceptable grade (<2% for building pads, <8% for roads). Terrain is modified using SDF-based deformation (smooth leveling with transition zones).
2. **Cut and fill**: Where flattening requires cutting into hills, retaining walls are generated. Where fill is needed, the slope is blended into natural terrain.
3. **Drainage**: Subtle grading directs water flow toward road edges and storm drains.

**Surface materials**:
- **Asphalt**: Dark gray, rough, with procedural crack/wear patterns
- **Concrete**: Light gray, smoother, with expansion joints
- **Sidewalk**: Concrete with scored grid pattern
- **Curb**: Concrete, slightly lighter than sidewalk
- **Building pads**: Gravel or concrete foundation

These are applied via the existing surface material system (`infinigen/core/surface.py`).

### Component 7: Urban Scene Composer (`urban_scene.py`)

Top-level composition orchestrating all urban systems:

```python
def compose_urban(output_blend, scene_seed, **kwargs):
    """
    Urban scene composition pipeline:
    1. Generate terrain (reuse existing Terrain system)
    2. Generate road network
    3. Flatten terrain for urban areas
    4. Mesh roads + intersections
    5. Place and generate buildings
    6. Place infrastructure (bridges, streetlights, power, signs)
    7. Place urban scatter (vehicles, furniture)
    8. Place vegetation (street trees, parks, yards)
    9. Place cameras (flight or ground)
    10. Setup lighting (sky, streetlights for night)
    11. Run weather/particle systems
    """
```

This is a `RandomStageExecutor`-compatible pipeline, meaning each stage can be randomized, skipped, or configured via gin.

**Scene types** (gin config presets):

| Config | Description | Building density | Road density |
|--------|-------------|-----------------|--------------|
| `dense_city` | High-rise downtown | High | Grid, dense |
| `suburban` | Residential neighborhoods | Medium | Curvilinear streets |
| `industrial` | Warehouses, factories | Low-medium | Arterial + access roads |
| `rural_town` | Small town with surrounding farms | Low | Single main road + branches |
| `coastal_city` | City on waterfront | High | Adapted to coast |
| `infrastructure_corridor` | Highway + bridges + power lines | Minimal | Single highway + access |

### Component 8: Urban Scatter via Existing Infrastructure

Reuses existing infinigen scatter systems for urban-appropriate small elements:

- **Street trees**: Trees placed along sidewalks at regular intervals (reuses tree factories)
- **Park vegetation**: Grass, bushes, flower beds in parks and yards (reuses scatter system)
- **Weeds in cracks**: Small vegetation between sidewalk cracks, in abandoned lots
- **Trash/debris**: Scattered small objects using existing scatter density system
- **Grime/weathering**: Existing `wear_tear/` materials applied to urban surfaces

### Integration

- New scene type alongside `generate_nature.py` and `generate_indoors.py`
- `generate_urban.py` as the top-level driver
- Works with existing `manage_jobs.py` for job management
- Road graph → camera trajectory integration: flight cameras (#4) can follow roads as navigation corridors
- Building structure → damage integration (#2): buildings are the primary damage targets
- Building interiors → indoor solver: optional per-building interior generation

### Performance Considerations

| Element | Geometry cost | Optimization |
|---------|--------------|-------------|
| Road mesh | Moderate (large continuous surface) | Adaptive tessellation, LOD by distance |
| Buildings | High (many buildings × facade detail) | Exterior-only for most; interior only for camera-proximal buildings |
| Infrastructure | Low (few, simple shapes) | Instanced where repeated (streetlights) |
| Urban scatter | Very high if dense | View-dependent LOD, placeholder/populate pattern reused |
| Terrain modification | One-time cost | SDF perturbation, not full remesh |

**View-dependent detail** (reuses existing populate pattern): Buildings within camera frustum get full facade detail. Distant buildings are simple boxes with material-only detail (normal-mapped windows). This is identical to how trees use placeholders → populated assets.

### Testing Strategy

1. **Road graph validity**: No disconnected components, no self-intersections, all edges have proper lane counts
2. **Road mesh**: Watertight, no Z-fighting at intersections, proper UVs for lane markings
3. **Building coverage**: Target building count achieved, building types distributed by zone
4. **Building mesh**: All buildings watertight, no inverted normals, facades properly UV-mapped
5. **Terrain adaption**: No roads with grade > max_grade, buildings on <2% slope
6. **Bridge generation**: Bridges appear where roads cross water or deep terrain
7. **Determinism**: Same seed → identical city layout
8. **Visual spot-check**: Rendered frames look like plausible cities (not identical boxes)
9. **Scale**: Generate a 1km² suburban scene within 30 minutes on reasonable hardware
10. **Integration test**: `compose_urban()` → `populate_scene()` → `render_image()` pipeline works end-to-end

### Dependencies

- Depends on existing terrain system for base landscape
- Reuses scatter infrastructure, material system, BVH queries
- Building interior integration optionally uses indoor constraint solver
- No dependency on other new features (#1-5)
- But is a dependency of #2 (damage needs buildings) and #5 (vispos needs urban scenes for urban-ops training data)

### Open Questions

1. **Building architectural styles**: Should buildings vary by geographic region (European, Asian, Middle Eastern, American)? Initial scope: two styles (Western modern + one alternate). Add more as content.
2. **Dynamic elements**: Should we simulate traffic (moving vehicles) or just static parked cars? Initial: static only. Dynamic traffic adds complexity with limited training benefit.
3. **Building interior fraction**: What percentage of buildings get interiors? 10% as default, gin-configurable. Interiors are expensive (full constraint solve per building).
4. **Historical/cultural landmarks**: Should special buildings (church, mosque, temple, stadium) be generated? Initial: no, too complex for procedural generation. Alternative: place pre-made landmark assets.
5. **Underground**: Subways, tunnels, underground parking? Initial: no, subsurface is out of scope.
6. **Road network scale**: How large an area? 1-4 km² seems reasonable for drone operations. Larger areas needed for satellite views? Satellite can use tiling: repeat urban pattern with variation.
