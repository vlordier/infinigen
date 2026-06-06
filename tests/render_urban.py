#!/usr/bin/env blender --python
"""Generate and render urban scenes with Eevee.

Usage:  blender -b -P tests/render_urban.py
        blender -b -P tests/render_urban.py -- --seed 42 --output /tmp/urban_render.png
        blender -b -P tests/render_urban.py -- --batch 3 --output_dir /tmp/urban_gallery
        blender -b -P tests/render_urban.py -- --preset medieval_village
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="/tmp/urban_render.png")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--city-size", type=float, default=200)
    parser.add_argument("--building-height", type=float, default=20)
    parser.add_argument("--render-samples", type=int, default=32)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--top-down", action="store_true")
    parser.add_argument("--preset", type=str, default="european_old")
    parser.add_argument("--tree-spacing", type=float, default=8.0)
    parser.add_argument("--car-density", type=float, default=0.3)
    parser.add_argument("--street-view", action="store_true")
    parser.add_argument("--export-xodr", type=str, default=None)
    try:
        sep = sys.argv.index("--")
        script_argv = [sys.argv[0]] + sys.argv[sep + 1:]
    except ValueError:
        script_argv = sys.argv
    args, remaining = parser.parse_known_args(script_argv)
    sys.argv = [sys.argv[0]] + remaining
    return args


def setup_scene():
    import bpy
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.ops.outliner.orphans_purge()
    return bpy.context.scene


def create_material(name, color, roughness=0.8, metallic=0.0):
    import bpy
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return mat


def add_ground(city_size):
    import bpy
    bpy.ops.mesh.primitive_plane_add(
        size=city_size * 1.8, location=(city_size * 0.5, city_size * 0.5, -0.05),
    )
    ground = bpy.context.active_object
    ground.name = "ground"
    bpy.ops.object.modifier_add(type="SUBSURF")
    bpy.ops.object.shade_smooth()
    mat = create_material("ground_mat", (0.25, 0.28, 0.2, 1), roughness=1.0)
    ground.data.materials.append(mat)
    return ground


def add_camera(scene, city_size, top_down=False, street_view=False):
    import bpy, mathutils
    if top_down:
        cam_data = bpy.data.cameras.new("Camera")
        cam = bpy.data.objects.new("Camera", cam_data)
        cam.location = mathutils.Vector((city_size * 0.5, city_size * 0.5, city_size * 1.2))
        cam.rotation_euler = mathutils.Euler((math.radians(90), 0, 0))
        cam_data.ortho_scale = city_size * 1.1
        cam_data.type = "ORTHO"
        scene.collection.objects.link(cam)
        scene.camera = cam
        return cam
    offset_x = city_size * 0.5
    offset_y = city_size * 0.5
    cx, cy = offset_x, offset_y
    cam_data = bpy.data.cameras.new("Camera")
    if street_view:
        cam_data.lens = 24
        cam_pos = mathutils.Vector((cx - city_size * 0.25, cy - city_size * 0.3, 6))
        target = mathutils.Vector((cx + city_size * 0.1, cy + city_size * 0.2, 3))
    else:
        cam_data.lens = 32
        cam_pos = mathutils.Vector((cx + city_size * 0.45, cy - city_size * 0.45, city_size * 0.28))
        target = mathutils.Vector((cx, cy, 0))
    cam = bpy.data.objects.new("Camera", cam_data)
    cam.location = cam_pos
    direction = (target - cam_pos).normalized()
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    scene.collection.objects.link(cam)
    scene.camera = cam
    return cam
    cam.location = cam_pos
    direction = (target - cam_pos).normalized()
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()
    scene.collection.objects.link(cam)
    scene.camera = cam
    return cam


def add_sun(scene, city_size):
    import bpy, mathutils
    sun_data = bpy.data.lights.new("Sun", type="SUN")
    sun = bpy.data.objects.new("Sun", sun_data)
    sun.rotation_euler = mathutils.Euler((math.radians(50), 0, math.radians(35)))
    sun_data.energy = 3.5
    sun_data.color = (1.0, 0.95, 0.85)
    scene.collection.objects.link(sun)
    return sun


def add_hdri(scene):
    import bpy
    world = scene.world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Strength"].default_value = 0.4
    bg.inputs["Color"].default_value = (0.7, 0.78, 0.88, 1)


def generate_city(args):
    import bpy, random as rng_mod, math
    from infinigen.assets.urban.city_presets import load_preset
    from infinigen.assets.urban.block_subdivision import BuildingLot, subdivide_block_fill
    from infinigen.assets.urban.skeleton import (
        RadialGenerator, GridGenerator, OrganicSpineGenerator, SingleSpineGenerator,
    )
    from infinigen.assets.urban.templates import get_template
    from infinigen.assets.urban.road_to_dcel import RoadToDCEL
    from infinigen.assets.urban.graph_parser import GraphParser
    from infinigen.assets.urban.road_mesher import RoadMesher
    from infinigen.assets.urban.intersection import IntersectionMesher
    from infinigen.assets.urban.road_markings import RoadMarkingMesher
    from infinigen.assets.urban.trees import place_trees_along_roads
    from infinigen.assets.urban.cars import place_parked_cars
    from infinigen.assets.urban.infrastructure.streetlights import place_streetlights
    from infinigen.assets.urban.buildings.building_generator import generate_building_shell

    preset = load_preset(args.preset)
    skeleton_map = {
        "radial": RadialGenerator,
        "grid": GridGenerator,
        "organic_spine": OrganicSpineGenerator,
        "single_spine": SingleSpineGenerator,
    }
    rng = rng_mod.Random(args.seed)
    sk_type = preset["skeleton_type"]

    if sk_type == "osmnx":
        from infinigen.assets.urban.osmnx_skeleton import OsmnxSkeleton
        skeleton = OsmnxSkeleton.generate(
            **preset["skeleton_params"],
        )
        all_xs = [p[0] for seg in skeleton.road_segments for p in (seg.source, seg.target)]
        all_ys = [p[1] for seg in skeleton.road_segments for p in (seg.source, seg.target)]
        min_x, max_x = min(all_xs), max(all_xs)
        min_y, max_y = min(all_ys), max(all_ys)
        city_bounds = max(max_x - min_x, max_y - min_y) * 1.3
        offset_x = -min_x + city_bounds * 0.05
        offset_y = -min_y + city_bounds * 0.05
        all_segments = list(skeleton.road_segments)
        all_lots = []
        for block in skeleton.blocks:
            if len(block.boundary) < 3:
                continue
            lots = subdivide_block_fill(block.boundary, rng=rng)
            all_lots.extend(lots)
    else:
        skeleton_cls = skeleton_map.get(sk_type)
        skeleton = skeleton_cls.generate(
            size=args.city_size, seed=rng.randint(0, 2**31),
            **preset["skeleton_params"],
        )
        city_bounds = args.city_size
        offset_x = city_bounds * 0.5
        offset_y = city_bounds * 0.5
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
            if fill.building_lots:
                all_lots.extend(fill.building_lots)
            else:
                lots = subdivide_block_fill(block.boundary, rng=rng)
                all_lots.extend(lots)

    def _shift(obj, dx, dy):
        if obj is None:
            return
        v = obj.location
        obj.location = (v.x + dx, v.y + dy, v.z)

    dcel = RoadToDCEL.build(all_segments)
    parser = GraphParser(dcel)
    road_mesher = RoadMesher()

    road_objs = road_mesher.mesh_roads(parser.road_segments)
    sidewalk_objs = road_mesher.mesh_sidewalks(parser.road_segments)
    for obj in road_objs:
        _shift(obj, offset_x, offset_y)
    for obj in sidewalk_objs:
        _shift(obj, offset_x, offset_y)

    inter_mesher = IntersectionMesher()
    inter_objs = inter_mesher.mesh_intersections(dcel, parser.road_segments)
    for obj in inter_objs:
        _shift(obj, offset_x, offset_y)

    marking_mesher = RoadMarkingMesher()
    marking_objs = marking_mesher.mesh_markings(parser.road_segments)
    crosswalk_objs = marking_mesher.mesh_crosswalks(dcel, parser.road_segments)
    for obj in marking_objs + crosswalk_objs:
        _shift(obj, offset_x, offset_y)

    road_mat = create_material("road_mat", (0.12, 0.12, 0.12, 1), roughness=0.95)
    sidewalk_mat = create_material("sidewalk_mat", (0.7, 0.68, 0.62, 1), roughness=0.85)
    inter_mat = create_material("inter_mat", (0.12, 0.12, 0.12, 1), roughness=0.95)
    for obj in road_objs:
        obj.data.materials.append(road_mat)
    for obj in sidewalk_objs:
        obj.data.materials.append(sidewalk_mat)
    for obj in inter_objs:
        obj.data.materials.append(inter_mat)

    building_mats = [
        create_material("bldg_concrete", (0.55, 0.50, 0.45, 1), roughness=0.7),
        create_material("bldg_brick", (0.65, 0.30, 0.15, 1), roughness=0.8),
        create_material("bldg_glass", (0.3, 0.5, 0.7, 1), roughness=0.15, metallic=0.2),
        create_material("bldg_stucco", (0.8, 0.75, 0.7, 1), roughness=0.9),
    ]

    bldg_count = 0
    for lot in all_lots:
        h = max(5.0, lot.area ** 0.5 * 0.35)
        h = min(h, args.building_height)
        h += (hash(str(lot.boundary)) % 20) * 0.3
        if lot.building_type == "industrial":
            h *= 1.5
        elif lot.building_type == "residential":
            h *= 0.6
        obj = generate_building_shell(lot.boundary, h)
        bpy.context.scene.collection.objects.link(obj)
        bldg_count += 1

    trees = place_trees_along_roads(
        parser.road_segments,
        spacing=args.tree_spacing,
        seed=args.seed + 100,
        city_bounds=(min_x, max_x, min_y, max_y) if sk_type == "osmnx" else None,
    )
    for t in trees:
        _shift(t, offset_x, offset_y)

    cars = place_parked_cars(
        parser.road_segments,
        density=args.car_density,
        seed=args.seed + 200,
        city_bounds=(min_x, max_x, min_y, max_y) if sk_type == "osmnx" else None,
    )
    for c in cars:
        _shift(c, offset_x, offset_y)

    light_positions = [
        ((seg.source[0] + seg.target[0]) * 0.5, (seg.source[1] + seg.target[1]) * 0.5)
        for seg in parser.road_segments if seg.sidewalk
    ]
    streetlights = []
    light_objs = []
    if light_positions:
        streetlights, light_objs = place_streetlights(
            light_positions,
            spacing=30,
            seed=args.seed + 300,
        )
        for s in streetlights + light_objs:
            _shift(s, offset_x, offset_y)

    print(f"Roads:{len(road_objs)}  Sidewalks:{len(sidewalk_objs)}  Intersections:{len(inter_objs)}  "
          f"Buildings:{bldg_count}  Trees:{len(trees)}  Cars:{len(cars)}  Streetlights:{len(streetlights)}  "
          f"Markings:{len(marking_objs)}  Crosswalks:{len(crosswalk_objs)}")
    args.city_bounds = city_bounds
    return parser


def render(scene, args):
    import bpy
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.filepath = args.output
    scene.render.image_settings.file_format = "PNG"
    scene.eevee.taa_samples = args.render_samples
    print(f"Rendering {args.output}...")
    bpy.ops.render.render(write_still=True)
    print(f"Saved {args.output}")


def main():
    args = parse_args()
    seeds = [args.seed + i for i in range(args.batch)]

    for i, seed in enumerate(seeds):
        args.seed = seed
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            args.output = os.path.join(args.output_dir, f"urban_seed{seed:03d}.png")

        scene = setup_scene()
        parser = generate_city(args)
        if args.export_xodr:
            from infinigen.assets.urban.opendrive_exporter import export_opendrive
            export_opendrive(parser.road_segments, args.export_xodr)
            print(f"Exported OpenDRIVE to {args.export_xodr}")
        cb = getattr(args, 'city_bounds', args.city_size)
        add_ground(cb)
        add_camera(scene, cb, top_down=args.top_down, street_view=args.street_view)
        add_sun(scene, cb)
        add_hdri(scene)
        render(scene, args)


if __name__ == "__main__":
    main()
