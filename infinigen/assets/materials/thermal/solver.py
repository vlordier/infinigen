import gin
import numpy as np
from mathutils import Vector
from .properties import ThermalProperties, get_thermal_properties

_STEFAN_BOLTZMANN = 5.670374419e-8

@gin.configurable
class ThermalEnvironment:
    solar_direct_irradiance: float = 900.0
    solar_diffuse_irradiance: float = 100.0
    ambient_air_temperature: float = 293.15
    wind_speed: float = 3.0
    sky_emissivity: float = 0.8

@gin.configurable
class ThermalSolver:
    env = ThermalEnvironment()
    
    @staticmethod
    def solve_temperatures(mesh_obj, sun_direction, face_material_categories):
        import bmesh
        bm = bmesh.new()
        bm.from_mesh(mesh_obj.data)
        bm.faces.ensure_lookup_table()
        n_faces = len(bm.faces)
        temperatures = np.full(n_faces, ThermalSolver.env.ambient_air_temperature)
        sun_dir = Vector(sun_direction).normalized()
        h = 15.0  # convection coefficient
        for i, face in enumerate(bm.faces):
            normal = face.normal.normalized()
            cos_theta = max(0, normal.dot(sun_dir))
            cat = face_material_categories[i] if i < len(face_material_categories) else "soil"
            props = get_thermal_properties(cat)
            q_solar = ThermalSolver.env.solar_direct_irradiance * cos_theta * props.solar_absorptivity
            q_solar += ThermalSolver.env.solar_diffuse_irradiance * props.solar_absorptivity
            T = ThermalSolver.env.ambient_air_temperature
            for _ in range(30):
                q_rad_out = props.emissivity * _STEFAN_BOLTZMANN * T**4
                q_conv = h * (T - ThermalSolver.env.ambient_air_temperature)
                residual = q_solar - q_rad_out - q_conv
                T += residual * 0.005
                T = max(200.0, min(400.0, T))
            temperatures[i] = T
        bm.free()
        return temperatures

def solve_scene_temperatures(scene_objects, sun_direction):
    results = {}
    for obj in scene_objects:
        if obj is not None and hasattr(obj, 'type') and obj.type == 'MESH' and obj.data and len(obj.data.polygons) > 0:
            cats = ["soil"] * len(obj.data.polygons)
            results[obj.name] = ThermalSolver.solve_temperatures(obj, sun_direction, cats)
    return results
