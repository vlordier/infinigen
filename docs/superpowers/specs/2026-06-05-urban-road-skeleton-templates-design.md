# Urban Road Skeleton + District Templates Design

## Problem

The current `GraphGenerator` uses random planar subdivision (jittered centroid fan triangulation), which produces Voronoi-like patterns that do not resemble real road networks. It creates star patterns, degenerate edges, and lacks road hierarchy.

## Solution

A **Skeleton + District Template** architecture that separates city-level road topology from block-level fill patterns. The skeleton defines major roads and superblocks; district templates fill each superblock with internal streets and building lots in a single pass.

## Architecture

```
CityPreset
  ├── skeleton_type + params (city-wide)
  └── zone_templates: dict[zone_id → DistrictTemplateConfig]

SkeletonGenerator.generate(preset, seed)
  └── CitySkeleton { road_segments[], blocks[{boundary, zone_id}] }

for each block:
  DistrictTemplate.fill(block.boundary, template_config, rng)
    └── DistrictFill { road_segments[], building_lots[] }

Combine all segments → RoadToDCEL.build() → DCEL → RoadMesher + IntersectionMesher
Combine all lots → existing building pipeline (block_subdivision, building_generator)
```

### Key Design Decisions

1. **Skeleton and fillers work with polygons and segments, not DCEL.** Clean data types, testable in isolation. The DCEL is built as a final step from the combined road segments.

2. **Templates produce both roads and lots together.** Street spacing and lot sizing are coupled — dense medieval blocks have narrow streets and tiny lots; Soviet blocks have wide arterials and massive footprints.

3. **Two orthogonal styling systems.** `DistrictTemplate` controls street topology + lot geometry. `RegionalStyle` (existing `regional_styles.py`) controls building appearance (materials, colors, heights). Presets compose both.

4. **Connection stubs ensure continuity.** The skeleton marks connection points on each block boundary. The filler's internal roads snap to these stubs. This guarantees roads connect at block boundaries.

## Components

### 1. CitySkeleton (dataclass)

```
CitySkeleton:
  road_segments: list[RoadSegment]
  blocks: list[BlockFace]

BlockFace:
  boundary: list[(x, y)]
  zone_id: str           # key into zone_templates
  connection_nodes: list[(x, y)]  # pre-computed road stubs to connect to
```

### 2. SkeletonGenerator

Creates the city backbone: major roads and superblocks.

**Types and their generators:**

| Type | Method | Output Blocks |
|------|--------|---------------|
| Radial | Center + N radials + M rings | Wedge blocks between radials and rings |
| Grid | N×M rectangular grid | Rectangular blocks |
| Radial+Grid | Radial core, grid transitions outward | Radial inner, rectangular outer |
| OrganicSpine | One curving main street + random branching | Irregular blocks along spine |
| SingleSpine | One straight/long road + perpendicular lanes | Rectangular side blocks |

Each skeleton generator:
- Places nodes at street intersections
- Connects nodes with RoadSegment objects
- Extracts block boundaries as closed polygons
- Assigns zone_ids based on distance from center / other rules
- Computes connection_nodes at block boundary midpoints

### 3. DistrictTemplateConfig

```
DistrictTemplateConfig:
  internal_road_width: float         # meters
  internal_sidewalk: bool
  lot_depth_range: (float, float)
  lot_min_area: float
  irregularity: float                # 0.0 (perfect) to 1.0 (chaotic)
  dead_end_chance: float
  density: float                     # 0.0 (sparse) to 1.0 (dense)
```

### 4. DistrictTemplate.fill()

Pure function:

```
fill(boundary: list[(x,y)], config: DistrictTemplateConfig, rng: random.Random)
  → DistrictFill { road_segments: list[RoadSegment], building_lots: list[BuildingLot] }
```

**Template implementations:**

| Template | Road Pattern | Lot Pattern | Algorithm |
|----------|-------------|-------------|-----------|
| OrganicGrid | Irregular grid, slight angle variation | Mixed 100-400m² | Grid with per-edge noise + dead-end probability |
| MedievalOrganic | Narrow winding alleys, dead ends, tiny squares | Tiny 20-80m², irregular | Random internal nodes → Voronoi → filter by area |
| SuburbanCulDeSac | Winding collector + branching dead ends | Large 500-2000m² | Spine + lateral dead-end roads, large lots on curves |
| RectangularGrid | Regular grid, right angles | Uniform 200-500m² | Even subdivision with minor perturbation |
| SovietBlock | Wide arterials, internal pedestrian paths | Huge 2000-5000m² | Sparse grid, combine cells into mega-lots |
| GardenPlots | Single spine + perpendicular narrow strips | Long narrow 500-1000m² | Subdivide perpendicular to spine |
| SparseOrganic | Minimal internal roads | Very large 1000-3000m² | Only add roads if needed, large lots |

### 5. RoadToDCEL

Takes a list of `RoadSegment` objects and builds a DCEL.

```
RoadToDCEL.build(segments: list[RoadSegment]) → DCEL
```

- Collects all unique node positions (deduplicated by position tolerance)
- For each node, sorts incident segments by angle
- Creates half-edges with correct twin/next/prev wiring
- Extracts faces from the half-edge cycles
- Handles boundary face detection

This is the **inverse of GraphParser** — instead of reading from DCEL, it writes to DCEL.

## Presets

A preset bundles skeleton type, skeleton params, zone template mapping, and regional style:

```python
CITY_PRESETS = {
    "european_old": {
        "skeleton_type": "radial",
        "skeleton_params": {"n_radials": 10, "n_rings": 5, "irregularity": 0.3},
        "zone_templates": {
            "core":       {"template": "organic_grid", "config": {...}},
            "inner_ring": {"template": "organic_grid", "config": {...}},
            "outer_ring": {"template": "suburban_cul_de_sac", "config": {...}},
        },
        "regional_style": "mediterranean",
    },
    "medieval_village": {
        "skeleton_type": "organic_spine",
        ...
        "regional_style": "mediterranean",
    },
    "suburban_estonia": {
        ...
        "regional_style": "baltic",
    },
    "ukrainian_city": {
        ...
        "regional_style": "soviet",
    },
    "ukrainian_village": {
        "skeleton_type": "single_spine",
        ...
        "regional_style": "soviet",
    },
    "soviet_microdistrict": {
        "skeleton_type": "radial_grid",
        ...
        "regional_style": "soviet",
    },
}
```

Zone mapping rules (how blocks get zone_ids):
- **Radial**: inner rings → "core", middle → "inner_ring", outer → "outer_ring"
- **Grid**: center area → "core", edges → "outer"
- **Spine**: blocks touching the spine → "core", others → "outer"

## Integration with Existing Pipeline

The new generators feed into the existing pipeline:

```
RoadToDCEL → DCEL → GraphParser → RoadMesher + IntersectionMesher
           → BlockSubdivision (only for lots not already produced by templates)
           → BuildingGenerator
           → Streetlights + Landmarks
```

If a `DistrictTemplate.fill()` already produced lots for a block, those lots bypass `BlockSubdivision` and go directly to `BuildingGenerator`. Blocks without template lots fall back to the existing `subdivide_lots()`.

## Files

| File | Purpose |
|------|---------|
| `infinigen/assets/urban/skeleton.py` | CitySkeleton, BlockFace, SkeletonGenerator base + all skeleton types |
| `infinigen/assets/urban/templates.py` | DistrictTemplateConfig, DistrictFill, all template implementations |
| `infinigen/assets/urban/road_to_dcel.py` | RoadToDCEL: list[RoadSegment] → DCEL |
| `infinigen/assets/urban/city_presets.py` | CITY_PRESETS dict, load_preset() |
| `infinigen/assets/urban/compose_urban.py` | Updated to use SkeletonGenerator + templates |

## Testing Strategy

- **Unit tests** for each skeleton type (deterministic, correct segment count, closed blocks)
- **Unit tests** for each template type (lots inside boundary, segments connect at stubs)
- **Unit tests** for RoadToDCEL (round-trip: DCEL → Parser → RoadToDCEL → DCEL, same topology)
- **Render tests**: Blender renders for each preset to verify visual quality
- **No bpy dependency** in skeleton/templates/road_to_dcel — testable without Blender
