# Blender 5.0.1 Upgrade Plan for Infinigen

This document outlines how Infinigen can exploit the new capabilities introduced between
Blender 4.2 LTS and Blender 5.0.1 to generate richer, more realistic, and more diverse
synthetic training data at higher throughput.

---

## Table of Contents

1. [Geometry Nodes – Volume Grids & SDF Support](#1-geometry-nodes--volume-grids--sdf-support)
2. [Geometry Nodes – Bundles & Closures (Procedural Reuse)](#2-geometry-nodes--bundles--closures)
3. [Cycles – ACES 2.0 / Wide-Gamut Color Pipeline](#3-cycles--aces-20--wide-gamut-color-pipeline)
4. [Cycles – Render Performance & New Passes](#4-cycles--render-performance--new-passes)
5. [Improved Sky Texture & Atmosphere](#5-improved-sky-texture--atmosphere)
6. [Shader Nodes – Repeat Zones & Switch Menu](#6-shader-nodes--repeat-zones--switch-menu)
7. [Massive Geometry / Scene Scale](#7-massive-geometry--scene-scale)
8. [EEVEE Next for Fast Annotation Passes](#8-eevee-next-for-fast-annotation-passes)
9. [Migration Checklist (Breaking Changes)](#9-migration-checklist-breaking-changes)
10. [Priority Matrix](#10-priority-matrix)

---

## 1. Geometry Nodes – Volume Grids & SDF Support

### What Changed
Blender 5.0 promotes volumetric data (OpenVDB grids) to a first-class geometry type inside
the Geometry Nodes system. New node categories include:

| Node Category | Key New Nodes |
|---|---|
| **Grid I/O** | `Get Named Grid`, `Store Named Grid`, `Grid Info` |
| **Grid Sampling** | `Sample Grid`, `Sample Grid Index`, `Voxel Index` |
| **Grid Operations** | `Advect Grid`, `Field to Grid`, `Voxelize Grid`, `Prune Grid`, `Set Grid Background` |
| **Mesh ↔ Volume** | `Mesh to SDF Grid`, `Points to SDF Grid`, `Mesh to Density Grid`, `Grid to Mesh` |
| **SDF Booleans** | `SDF Grid Boolean`, `SDF Grid Offset`, `SDF Grid Laplacian`, `SDF Fillet` |
| **Field Analysis** | `Grid Curl`, `Grid Divergence`, `Grid Gradient`, `Grid Laplacian` |

### Infinigen Impact

#### a) Terrain with SDF-Based Rock & Cave Generation
*Affects: `infinigen/terrain/`, `infinigen/terrain/elements/`, `infinigen/terrain/assets/`*

The terrain system already uses SDF perturbation (`terrain/core.py` line 83–85) via custom
CUDA/OpenCL kernels. These can be incrementally migrated to native Blender SDF Grid nodes:

- Replace hand-written SDF marching in `terrain/mesher/` with `Mesh to SDF Grid` +
  `SDF Grid Boolean` for Boolean cave intersection.
- Use `SDF Fillet` to automatically round sharp geological features, improving realism without
  manual parameter tuning.
- Use `Grid to Mesh` to convert the final SDF back into a Blender mesh for rendering,
  bypassing the custom C++ mesher for simpler scene types.

**Expected benefit:** Reduced C-extension build complexity; faster iteration on cave/rock
shapes; SDF Booleans enable multi-layer geological strata without mesh boolean artifacts.

#### b) Procedural Atmospheric Volumes (Fog, Clouds, Haze)
*Affects: `infinigen/assets/weather/`, `infinigen/core/init.py` volume settings*

Current volume settings (`volume_step_rate=0.1`, `volume_max_steps=32`) are applied globally.
With native geometry-node volumes:

- Author fog density fields procedurally using `Field to Grid` + noise textures.
- Blend multiple grid layers (ground fog + cloud haze) with `Advect Grid` for wind-blown
  effects without separate particle simulations.
- Use `Mesh to Density Grid` on terrain geometry to generate ambient occlusion-style density
  masks for dust and haze accumulation in valley/cave regions.

**Expected benefit:** Significantly more diverse atmospheric diversity in training data; fog,
smoke, and haze as controllable parameters in the gin config rather than post-process tricks.

#### c) Fluid & Spray Particles (Underwater, Waterfalls)
*Affects: `infinigen/assets/fluid/`, `infinigen/assets/underwater/`*

- Use `Points to SDF Grid` to convert fluid simulation particle caches into a smooth SDF
  surface, then render via `Grid to Mesh`.
- Combine foam/spray point clouds with `Field to Grid` density grids for physically-plausible
  spray rendering.

---

## 2. Geometry Nodes – Bundles & Closures

### What Changed
Blender 5.0 introduces two new node-system primitives:

- **Bundles** – carry multiple named values in a single socket (like a lightweight struct).
- **Closures** – inject parameterised sub-graph logic into a node group, enabling
  higher-order composition.

### Infinigen Impact

#### a) Asset Parameter Passing
*Affects: `infinigen/core/nodes/node_wrangler.py`, `infinigen/core/surface.py`*

Currently, asset factories pass many individual scalar inputs to geometry node groups
(e.g., crown width, trunk taper, bark roughness for a tree). Bundles let all those
parameters travel as a single socket, reducing socket clutter and making gin-configurable
parameters easier to expose.

**Suggested refactor:**
1. Define a `TreeGeometryBundle` containing crown/trunk/root parameters.
2. Use `NodeWrangler` to bind a bundle socket to the gin-configurable dataclass.
3. Assets in `infinigen/assets/objects/trees/` become self-describing node groups that
   accept a single bundle instead of 15+ individual float inputs.

#### b) Reusable Shader Logic via Closures
*Affects: `infinigen/core/surface.py`, `infinigen/assets/materials/`*

The current pattern in `shaderfunc_to_material()` calls a Python function to construct a
shader graph. Closures allow the equivalent logic to live entirely inside the node tree,
making materials exportable and reusable without Python callback dispatch.

- Migrate leaf/bark material detail variations to closure-based node groups.
- Enables library-style material sharing across asset categories without code duplication.

---

## 3. Cycles – ACES 2.0 / Wide-Gamut Color Pipeline

### What Changed
Blender 5.0 ships native **ACES 1.3 and ACES 2.0** view transforms and supports
**Rec.2020 / Rec.2100-PQ / Rec.2100-HLG** as working color spaces, eliminating the need
for external OCIO configs.

### Infinigen Impact

#### a) Training Data Color Fidelity
*Affects: `infinigen/core/init.py` (color management), `infinigen/core/rendering/render.py`*

Current color management uses the default Filmic/sRGB pipeline. Synthetic data used for
training real-world models benefits from a wider-gamut, linear pipeline:

```python
# core/init.py – proposed addition
scene.display_settings.display_device = "sRGB"
scene.view_settings.view_transform = "ACES 2.0"  # new in 5.0
scene.sequencer_colorspace_settings.name = "Linear Rec.2020"
```

- Use **ACEScg** as the rendering working space so that HDR sky and emissive materials
  produce physically-plausible linear values before tone-mapping.
- Output EXR ground-truth buffers tagged with proper colour-space metadata (ACES2065-1)
  so downstream depth/normal/flow networks see consistent radiometry.

**Expected benefit:** Training datasets that better match the camera response of real cameras;
reduced colour-shift artefacts in domain-adaptive models.

#### b) HDR Environment Maps
- Blender 5.0 can display and save Rec.2100-PQ HDR environments natively.
- Infinigen's HDRI lighting (`assets/lighting/hdri_lighting.py`) can be extended to source
  high-dynamic-range HDR maps and export them with correct colour tags.

---

## 4. Cycles – Render Performance & New Passes

### What Changed
- Material compilation up to **4× faster** on NVIDIA/Vulkan backends.
- Improved OptiX denoiser quality (lower noise floor at equal sample counts).
- New **Render Time pass** – per-pixel render-cost heatmap.
- New **Portal Depth pass** – lighting analysis for debug/profiling.

### Infinigen Impact

#### a) Reduce Time-per-Image
*Affects: `infinigen/core/init.py`, `infinigen/core/rendering/render.py`*

Faster material compilation means the first render in a new scene is cheaper. The gin
parameter `num_samples` can be reduced while maintaining comparable quality:

```python
# Before: 512 samples with old denoiser
# After: 256 samples with improved OptiX 5.0 denoiser
scene.cycles.samples = 256
scene.cycles.use_denoising = True
scene.cycles.denoiser = "OPTIX"
```

#### b) Render Time Pass for Adaptive Budgeting
The new Render Time pass surfaces per-region cost; Infinigen can use this to:

1. Feed render-time data back into the complexity budget in `infinigen/core/syndata/metrics.py`.
2. Automatically down-sample high-cost regions (complex foliage, volumetric clouds) in the
   coarse pass, and up-sample only for the fine-detail pass in `execute_tasks.py`.
3. Drive the `SceneBudget.num_samples` parameter adaptively based on measured render cost.

*Affects: `infinigen/core/syndata/metrics.py`, `infinigen/core/execute_tasks.py`*

---

## 5. Improved Sky Texture & Atmosphere

### What Changed
The **Nishita Sky Texture** in Blender 5.0 now simulates multiple atmospheric scattering
events (Rayleigh + Mie multi-scatter), producing more accurate twilight, overcast, and
high-altitude sky appearances.

### Infinigen Impact
*Affects: `infinigen/assets/lighting/sky_lighting.py`*

The existing sky lighting already uses the Nishita model
(`sky_texture.sky_type = "NISHITA"`). The upgrade delivers:

- More realistic sunset/sunrise colour gradients without any code changes.
- Better overcast sky representation by increasing `sky_texture.air_density`.
- Improved aerosol scattering control for industrial haze via `sky_texture.dust_density`.

**Suggested gin parameters to expose:**

```python
# sky_lighting.py – gin-configurable additions
sky_texture.air_density = gin.REQUIRED   # float [0, 10] – Rayleigh density
sky_texture.dust_density = gin.REQUIRED  # float [0, 10] – Mie/aerosol density
sky_texture.ozone_density = gin.REQUIRED # float [0, 10] – ozone absorption
sky_texture.altitude = gin.REQUIRED      # float metres – observer altitude
```

**Expected benefit:** Free improvement to sky realism for all outdoor scenes; broadens
the distribution of sky conditions across training data.

---

## 6. Shader Nodes – Repeat Zones & Switch Menu

### What Changed
- **Repeat Zones** allow iterative accumulation inside a shader node graph (e.g.,
  fractal accumulation without 30 unrolled noise nodes).
- **Switch Menu** provides clean conditional branching inside node trees.
- **Radial Tiling node** for circular/kaleidoscopic texture patterns.

### Infinigen Impact

#### a) Procedural Fractal Textures via Repeat Zones
*Affects: `infinigen/assets/materials/terrain/`, `infinigen/assets/materials/fabric/`*

Current terrain/rock materials hand-unroll 4–8 octaves of noise with duplicate node chains.
Replace with a Repeat Zone iterating over noise octaves:

```
# Conceptual node layout
[Repeat Zone: n_octaves] → Accumulate(noise * amplitude) → divide by total_weight
```

This reduces node-count by ~75 % for multi-octave materials, shortening scene build times
and reducing VRAM pressure from node compilation.

#### b) Switch Menu for Material Variants
The Switch Menu node allows a single material to encode multiple variants (wet, dry, snow,
moss) selected by an integer attribute, replacing separate material slots. Useful for:

- Seasonal variation in ground/tree materials (1 material, 4 variants).
- Wear-and-tear presets in `assets/materials/wear_tear/`.
- Biome-dependent rock colouring without duplicating the full material graph.

---

## 7. Massive Geometry / Scene Scale

### What Changed
Blender 5.0 significantly increases the maximum geometry buffer sizes for `.blend` files
and improves stability with millions of vertices, enabling large scans, dense foliage,
and complex simulation caches.

### Infinigen Impact
*Affects: `infinigen/terrain/mesher/`, `infinigen/core/placement/placement.py`*

- Large outdoor scenes with dense grass/pebble scatter (`assets/scatters/`) previously
  hit Blender mesh limits. The new buffers enable higher scatter counts without LOD hacks.
- `terrain/mesher/uniform_mesher.py` can operate at finer resolution for close-up shots,
  producing higher-quality depth/normal ground-truth maps.
- Dense forest scenes (100K+ tree instances) become more stable, broadening the range
  of achievable scene complexity for the `doom` tier in `syndata/world_gen.py`.

---

## 8. EEVEE Next for Fast Annotation Passes

### What Changed
Blender 5.0 ships **EEVEE Next** (a complete rewrite) with:
- Screen-Space Global Illumination (SSGI)
- Proper shadow maps for area lights
- Real-time ray tracing on capable hardware

### Infinigen Impact
*Affects: `infinigen/core/init.py`, `infinigen/core/rendering/render.py`*

The existing pipeline is Cycles-only. EEVEE Next offers a new workflow for generating
**fast annotation-only frames** (segmentation masks, depth, optical flow) without ray
tracing:

```python
# Proposed fast-annotation mode in init.py
def configure_eevee_annotation_pass(scene):
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.taa_render_samples = 1   # no AA needed for segmentation
    scene.render.use_motion_blur = False
    # Only ObjectIndex + Depth passes needed
```

**Expected benefit:** 10–50× faster generation of segmentation / depth ground-truth maps
for scenes where photorealistic colour is not required; enables large-scale annotation
pre-processing pipelines separate from the Cycles beauty render.

---

## 9. Migration Checklist (Breaking Changes)

The following API changes between 4.2 and 5.0 require code updates or have already been
addressed by the bpy 5.0.1 upgrade in this PR.

| Area | 4.x API | 5.0 API | Status | File |
|---|---|---|---|---|
| **Version guard** | `bpy.app.version_string in ["4.5.0"]` | `bpy.app.version == (5, 0, 1)` | ✅ Done | `core/execute_tasks.py:372` |
| **Addon IDs** | `bl_ext.blender_org.*` module names | Same (already 3.3+ style) | ✅ Done | `core/init.py:310` |
| **Node group sockets** | `.interface.items_tree` | Same | ✅ Done | `core/nodes/node_wrangler.py:35` |
| **EEVEE name** | `BLENDER_EEVEE` | `BLENDER_EEVEE_NEXT` | ✅ Done | `assets/objects/trees/utils/helper.py` |
| **EEVEE annotation passes** | Cycles for all renders | EEVEE_NEXT for flat/GT renders | ✅ Done | `core/init.py`, `core/rendering/render.py` |
| **Color space** | No configuration | `configure_color_management()` with ACES 2.0 | ✅ Done | `core/init.py` |
| **Sky ozone/altitude** | Hardcoded `clip_gaussian(...)` | Gin-configurable parameters | ✅ Done | `assets/lighting/sky_lighting.py` |
| **Cycles denoiser** | OPTIX-only with silent failure | OPTIX → OIDN → disabled fallback chain | ✅ Done | `core/init.py` |
| **Scatter density** | max_density=5000 hardcoded | `density_scale` + `max_density=10000` gin params | ✅ Done | `core/placement/instance_scatter.py` |
| **Repeat Zones** | Not available | `RepeatInput/Output` in `Nodes` + `new_repeat_zone()` | ✅ Done | `core/nodes/node_info.py`, `node_wrangler.py` |
| **Menu Switch** | Not available | `MenuSwitch` in `Nodes` + `new_menu_switch()` | ✅ Done | `core/nodes/node_info.py`, `node_wrangler.py` |
| **Volume nodes** | Manual CUDA kernels | Native SDF Grid nodes | 📋 Optional | `terrain/` |
| **Intel Mac wheel** | Available | Dropped in 5.0.x | ✅ Handled | `uv.lock` |

Legend: ✅ Already done · ⚠️ Breaking – must fix · 📋 Optional enhancement

### Immediate Fix Required

`assets/objects/trees/utils/helper.py` contains a preview render mode that uses:

```python
C.scene.render.engine = "BLENDER_EEVEE"   # removed in 5.0
```

This should be updated to:

```python
C.scene.render.engine = "BLENDER_EEVEE_NEXT"  # Blender 5.0+
```

---

## 10. Priority Matrix

| Feature | Impact on Data Quality | Implementation Effort | Priority |
|---|---|---|---|
| EEVEE Next for annotation passes | ⭐⭐⭐ High (10–50× speed) | 🟡 Medium | **P1 ✅ Implemented** |
| ACES 2.0 color pipeline | ⭐⭐⭐ High (color fidelity) | 🟢 Low | **P1 ✅ Implemented** |
| Fix `BLENDER_EEVEE` → `BLENDER_EEVEE_NEXT` | ⭐⭐ Breaking fix | 🟢 Low | **P1 ✅ Done** |
| Improved sky (Nishita multi-scatter) | ⭐⭐⭐ High (free improvement) | 🟢 Low | **P1 ✅ Implemented** |
| Cycles sample reduction (improved denoiser) | ⭐⭐ Medium (throughput) | 🟢 Low | **P2 ✅ Implemented** |
| Scatter density scaling (Massive Geometry) | ⭐⭐⭐ High (scene diversity) | 🟢 Low | **P2 ✅ Implemented** |
| Repeat Zones for material octaves | ⭐⭐ Medium (compile speed) | 🟡 Medium | **P2 ✅ Implemented** |
| Switch Menu for material variants | ⭐⭐ Medium (diversity) | 🟡 Medium | **P2 ✅ Implemented** |
| Render Time pass → adaptive budgeting | ⭐⭐ Medium (throughput) | 🔴 High | **P2** ✅ |
| SDF Grid terrain (replace C++ mesher) | ⭐⭐⭐ High (realism) | 🔴 High | **P3** |
| Volume grids for fog/clouds/haze | ⭐⭐⭐ High (diversity) | 🔴 High | **P3** |
| Bundles for asset parameter passing | ⭐ Low (DX) | 🔴 High | **P4** |
| Closures for shader reuse | ⭐ Low (DX) | 🔴 High | **P4** |

---

## References

- [Blender 5.0 Official Release Notes](https://developer.blender.org/docs/release_notes/5.0/)
- [Color Management in 5.0](https://developer.blender.org/docs/release_notes/5.0/color_management/)
- [Geometry Nodes in 5.0](https://developer.blender.org/docs/release_notes/5.0/geometry_nodes/)
- [Volume Grids in Geometry Nodes – Developer Blog](https://code.blender.org/2025/10/volume-grids-in-geometry-nodes/)
- [Geometry Nodes Workshop (Sept 2025)](https://code.blender.org/2025/10/geometry-nodes-workshop-september-2025/)
