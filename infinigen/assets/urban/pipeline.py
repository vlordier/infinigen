from infinigen.assets.urban.city_presets import load_preset
from infinigen.assets.urban.block_subdivision import subdivide_block_fill, BuildingLot
from infinigen.assets.urban.graph_parser import GraphParser, RoadSegment
from infinigen.assets.urban.templates import BaseTemplate, DistrictTemplateConfig, get_template


class UrbanPipeline:
    """Step-by-step urban city generator.

    Each step has a standalone `run()` method that takes the output of the
    previous step and returns its own result.  Steps can also be tested
    independently by providing inputs directly.
    """

    def __init__(self, seed=42, city_size=200, preset_name="european_old",
                 terrain_mode="flat", terrain_noise_amplitude=3.0):
        import random
        self.seed = seed
        self.city_size = city_size
        self.preset_name = preset_name
        self.preset = load_preset(preset_name)
        self.rng = random.Random(seed)
        self.terrain_mode = terrain_mode
        self.terrain_noise_amp = terrain_noise_amplitude
        self.terrain_modifier = None

    # ------------------------------------------------------------------
    # Step 0 – terrain
    # ------------------------------------------------------------------
    def step0_terrain(self):
        """Create a TerrainProvider + TerrainModifier for this city."""
        from infinigen.assets.urban.terrain import (
            TerrainProvider, TerrainModifier, terrain_from_preset, terrain_from_osm,
        )
        sk_type = self.preset["skeleton_type"]
        if self.terrain_mode == "flat":
            tp = TerrainProvider.flat()
        elif sk_type == "osmnx":
            from infinigen.assets.urban.osmnx_skeleton import OsmnxSkeleton
            sk = self._cached_skeleton if hasattr(self, '_cached_skeleton') else None
            if sk is None:
                sk = OsmnxSkeleton.generate(**self.preset["skeleton_params"])
                self._cached_skeleton = sk
            tp = terrain_from_osm(sk.road_segments, seed=self.seed)
        else:
            tp = terrain_from_preset(
                self.preset_name, city_size=self.city_size,
                noise_amplitude=self.terrain_noise_amp, seed=self.seed,
            )
        self.terrain_modifier = TerrainModifier(tp, mode=self.terrain_mode)
        return tp

    # ------------------------------------------------------------------
    # Step 1 – city skeleton
    # ------------------------------------------------------------------
    def step1_skeleton(self):
        """Generate the road skeleton + block faces."""
        from infinigen.assets.urban.skeleton import (
            RadialGenerator, GridGenerator, OrganicSpineGenerator, SingleSpineGenerator,
        )
        from infinigen.assets.urban.osmnx_skeleton import OsmnxSkeleton

        sk_type = self.preset["skeleton_type"]
        if sk_type == "osmnx":
            sk = OsmnxSkeleton.generate(**self.preset["skeleton_params"])
        else:
            mapping = {
                "radial": RadialGenerator,
                "grid": GridGenerator,
                "organic_spine": OrganicSpineGenerator,
                "single_spine": SingleSpineGenerator,
            }
            cls = mapping.get(sk_type)
            sk = cls.generate(size=self.city_size, seed=self.rng.randint(0, 2**31),
                              **self.preset["skeleton_params"])
        return sk

    # ------------------------------------------------------------------
    # Step 2 – fill blocks with templates and building lots
    # ------------------------------------------------------------------
    def step2_fill_blocks(self, skeleton):
        """Template fill for each block → extra road segments + building lots."""
        all_segs = list(skeleton.road_segments)
        all_lots = []
        zone_templates = self.preset["zone_templates"]
        for block in skeleton.blocks:
            entry = zone_templates.get(block.zone_id)
            if entry is None:
                continue
            t_cls = get_template(entry["template"])
            if t_cls is None:
                continue
            fill = t_cls.fill(block.boundary, entry["config"], self.rng)
            all_segs.extend(fill.road_segments)
            if fill.building_lots:
                all_lots.extend(fill.building_lots)
            else:
                all_lots.extend(subdivide_block_fill(block.boundary, rng=self.rng))
        return all_segs, all_lots

    # ------------------------------------------------------------------
    # Step 3 – build DCEL + parser
    # ------------------------------------------------------------------
    def step3_dcel(self, road_segments):
        """Convert road segments → DCEL → parsed road network + city areas."""
        from infinigen.assets.urban.road_to_dcel import RoadToDCEL
        dcel = RoadToDCEL.build(road_segments)
        parser = GraphParser(dcel)
        return dcel, parser

    # ------------------------------------------------------------------
    # Step 4 – mesh roads and sidewalks
    # ------------------------------------------------------------------
    def step4_mesh_roads(self, parser):
        """Blender meshes for roads and sidewalks (with terrain Z)."""
        from infinigen.assets.urban.road_mesher import RoadMesher
        m = RoadMesher()
        z_func = None
        if self.terrain_modifier:
            z_func = lambda x, y: self.terrain_modifier.road_vertex_z(x, y)
        return (m.mesh_roads(parser.road_segments, z_func=z_func),
                m.mesh_sidewalks(parser.road_segments, z_func=z_func))

    # ------------------------------------------------------------------
    # Step 5 – mesh intersections
    # ------------------------------------------------------------------
    def step5_mesh_intersections(self, dcel, parser):
        """Blender meshes for road junctions."""
        from infinigen.assets.urban.intersection import IntersectionMesher
        m = IntersectionMesher()
        return m.mesh_intersections(dcel, parser.road_segments)

    # ------------------------------------------------------------------
    # Step 6 – road markings and crosswalks
    # ------------------------------------------------------------------
    def step6_markings(self, dcel, parser):
        """Lane lines, edge lines, dashed centerlines, crosswalk stripes."""
        from infinigen.assets.urban.road_markings import RoadMarkingMesher
        m = RoadMarkingMesher()
        return m.mesh_markings(parser.road_segments), m.mesh_crosswalks(dcel, parser.road_segments)

    # ------------------------------------------------------------------
    # Step 7 – buildings
    # ------------------------------------------------------------------
    def step7_buildings(self, lots):
        """Extrude building shells from lot boundaries (with terrain Z)."""
        import bpy
        from infinigen.assets.urban.buildings.building_generator import generate_building_shell
        objs = []
        for i, lot in enumerate(lots):
            h = max(6.0, min(40.0, lot.area ** 0.5 * 0.5))
            z_base = 0.0
            if self.terrain_modifier:
                z_base = self.terrain_modifier.building_z(lot.boundary)
            obj = generate_building_shell(lot.boundary, h, name_suffix=str(i), z_base=z_base)
            bpy.context.scene.collection.objects.link(obj)
            objs.append(obj)
        return objs

    # ------------------------------------------------------------------
    # Step 8 – streetlights
    # ------------------------------------------------------------------
    def step8_streetlights(self, parser):
        """Streetlight poles with point light emitters."""
        from infinigen.assets.urban.infrastructure.streetlights import place_streetlights
        positions = [
            ((s.source[0] + s.target[0]) * 0.5, (s.source[1] + s.target[1]) * 0.5)
            for s in parser.road_segments if s.sidewalk
        ]
        if not positions:
            return [], []
        return place_streetlights(positions, spacing=30, seed=self.seed + 4)

    # ------------------------------------------------------------------
    # Step 9 – landmarks
    # ------------------------------------------------------------------
    def step9_landmarks(self):
        """Scatter landmark objects (church, stadium, silo, …)."""
        from infinigen.assets.urban.buildings.landmarks import place_landmarks
        from infinigen.assets.urban.regional_styles import get_regional_style
        style = get_regional_style(self.preset.get("regional_style", "generic"))
        return place_landmarks(
            (self.city_size, self.city_size), style,
            count=self.preset.get("landmark_count", 5), seed=self.seed + 5,
        )

    # ------------------------------------------------------------------
    # Step 10 – OpenDRIVE export
    # ------------------------------------------------------------------
    def step10_opendrive(self, parser, path="/tmp/pipeline.xodr"):
        """Export road network in OpenDRIVE 1.4 XML format."""
        from infinigen.assets.urban.opendrive_exporter import export_opendrive
        return export_opendrive(parser.road_segments, path)

    # ------------------------------------------------------------------
    # Run all steps in sequence
    # ------------------------------------------------------------------
    def run_all(self):
        sk = self.step1_skeleton()
        segs, lots = self.step2_fill_blocks(sk)
        dcel, parser = self.step3_dcel(segs)
        road_objs, sw_objs = self.step4_mesh_roads(parser)
        inter_objs = self.step5_mesh_intersections(dcel, parser)
        mark_objs, cross_objs = self.step6_markings(dcel, parser)
        bldg_objs = self.step7_buildings(lots)
        sl_objs, lt_objs = self.step8_streetlights(parser)
        lm_objs = self.step9_landmarks()
        xodr = self.step10_opendrive(parser)
        return {
            "skeleton": sk, "road_segments": segs, "lots": lots, "dcel": dcel,
            "parser": parser, "roads": road_objs, "sidewalks": sw_objs,
            "intersections": inter_objs, "markings": mark_objs,
            "crosswalks": cross_objs, "buildings": bldg_objs,
            "streetlights": sl_objs, "landmarks": lm_objs, "opendrive": xodr,
        }