import logging
import gin
from infinigen.core.util.pipeline import RandomStageExecutor
from infinigen.assets.urban.graph_generator import GraphGenerator
from infinigen.assets.urban.graph_parser import GraphParser
from infinigen.assets.urban.road_mesher import RoadMesher
from infinigen.assets.urban.intersection import IntersectionMesher
from infinigen.assets.urban.block_subdivision import subdivide_lots
from infinigen.assets.urban.buildings.building_generator import generate_buildings_from_lots
from infinigen.assets.urban.infrastructure.streetlights import place_streetlights
from infinigen.assets.urban.buildings.landmarks import place_landmarks
from infinigen.assets.urban.regional_styles import get_regional_style

logger = logging.getLogger(__name__)


@gin.configurable
def compose_urban(output_folder, scene_seed, **params):
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
        city_size = params.get("city_size", 200)
        seed = scene_seed + 1
        dcel = GraphGenerator.generate(city_size, city_size, seed)
        parser = GraphParser(dcel)
        mesher = RoadMesher()
        road_objs = mesher.mesh_roads(parser.road_segments)
        sidewalk_objs = mesher.mesh_sidewalks(parser.road_segments)
        inter_mesher = IntersectionMesher()
        inter_objs = inter_mesher.mesh_intersections(dcel, parser.road_segments)
        return parser, road_objs, sidewalk_objs, inter_objs

    result = p.run_stage("road_network", add_road_network, use_chance=False)
    if result is None:
        return
    parser, road_objs, sidewalk_objs, inter_objs = result

    def add_buildings():
        regional_style = get_regional_style()
        lots = subdivide_lots(parser.city_areas, seed=scene_seed + 2)
        buildings = generate_buildings_from_lots(lots, regional_style, seed=scene_seed + 3)
        logger.info(f"Generated {len(buildings)} buildings")
        return buildings

    p.run_stage("buildings", add_buildings, use_chance=False)

    def add_streetlights():
        light_positions = [
            [(seg.source, seg.target)]
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
