import logging
import gin
from infinigen.core.util.pipeline import RandomStageExecutor
from infinigen.assets.urban.road_mesher import RoadMesher
from infinigen.assets.urban.intersection import IntersectionMesher
from infinigen.assets.urban.block_subdivision import subdivide_lots
from infinigen.assets.urban.buildings.building_generator import generate_buildings_from_lots
from infinigen.assets.urban.infrastructure.streetlights import place_streetlights
from infinigen.assets.urban.buildings.landmarks import place_landmarks
from infinigen.assets.urban.regional_styles import get_regional_style
from infinigen.assets.urban.city_presets import load_preset
from infinigen.assets.urban.road_to_dcel import RoadToDCEL
from infinigen.assets.urban.graph_parser import GraphParser
from infinigen.assets.urban.templates import get_template

logger = logging.getLogger(__name__)

_SKELETON_GENERATORS = {}


def _get_skeleton_generator(skeleton_type):
    if skeleton_type == "osmnx":
        from infinigen.assets.urban.osmnx_skeleton import OsmnxSkeleton
        return OsmnxSkeleton
    from infinigen.assets.urban.skeleton import (
        RadialGenerator, GridGenerator, OrganicSpineGenerator, SingleSpineGenerator,
    )
    mapping = {
        "radial": RadialGenerator,
        "grid": GridGenerator,
        "organic_spine": OrganicSpineGenerator,
        "single_spine": SingleSpineGenerator,
    }
    return mapping.get(skeleton_type)


@gin.configurable
def compose_urban(output_folder, scene_seed, preset_name="european_old", **params):
    p = RandomStageExecutor(scene_seed, output_folder, params)
    regional_style = get_regional_style()

    def add_base_plane():
        import bpy
        bpy.ops.mesh.primitive_grid_add(
            size=params.get("city_size", 200),
            location=(0, 0, 0),
        )
        plane = bpy.context.active_object
        plane.name = "ground"
        return plane

    ground = p.run_stage("ground", add_base_plane, use_chance=False)

    def add_road_network():
        import random as rng_mod
        city_size = params.get("city_size", 200)
        preset = load_preset(preset_name)
        skeleton_cls = _get_skeleton_generator(preset["skeleton_type"])
        if skeleton_cls is None:
            raise ValueError(f"Unknown skeleton type: {preset['skeleton_type']}")
        rng = rng_mod.Random(scene_seed + 1)
        skeleton = skeleton_cls.generate(
            size=city_size, seed=rng.randint(0, 2**31),
            **preset["skeleton_params"],
        )
        all_segments = list(skeleton.road_segments)
        all_lots = []
        zone_templates = preset["zone_templates"]
        for block in skeleton.blocks:
            zone_entry = zone_templates.get(block.zone_id)
            if zone_entry is None:
                continue
            template_cls = get_template(zone_entry["template"])
            if template_cls is None:
                continue
            config = zone_entry["config"]
            fill = template_cls.fill(block.boundary, config, rng)
            all_segments.extend(fill.road_segments)
            all_lots.extend(fill.building_lots)
        dcel = RoadToDCEL.build(all_segments)
        parser = GraphParser(dcel)
        mesher = RoadMesher()
        road_objs = mesher.mesh_roads(parser.road_segments)
        sidewalk_objs = mesher.mesh_sidewalks(parser.road_segments)
        inter_mesher = IntersectionMesher()
        inter_objs = inter_mesher.mesh_intersections(dcel, parser.road_segments)
        return parser, all_lots, road_objs, sidewalk_objs, inter_objs

    result = p.run_stage("road_network", add_road_network, use_chance=False)
    if result is None:
        return
    parser, all_lots, road_objs, sidewalk_objs, inter_objs = result

    def add_buildings():
        lots = all_lots if all_lots else subdivide_lots(parser.city_areas, seed=scene_seed + 2)
        buildings = generate_buildings_from_lots(lots, regional_style, seed=scene_seed + 3)
        logger.info(f"Generated {len(buildings)} buildings")
        return buildings

    p.run_stage("buildings", add_buildings, use_chance=False)

    def add_streetlights():
        light_positions = [
            ((seg.source[0] + seg.target[0]) * 0.5, (seg.source[1] + seg.target[1]) * 0.5)
            for seg in parser.road_segments
            if seg.sidewalk
        ]
        if not light_positions:
            return []
        lights = place_streetlights(
            light_positions,
            spacing=params.get("streetlight_spacing", 30),
            seed=scene_seed + 4,
        )
        logger.info(f"Placed {len(lights)} streetlights")
        return lights

    p.run_stage("streetlights", add_streetlights, use_chance=False)

    def add_landmarks():
        city_size = params.get("city_size", 200)
        bounds = (city_size, city_size)
        objs = place_landmarks(
            bounds, regional_style,
            count=params.get("landmark_count", 5),
            seed=scene_seed + 5,
        )
        logger.info(f"Placed {len(objs)} landmarks")
        return objs

    p.run_stage("landmarks", add_landmarks, use_chance=False)
