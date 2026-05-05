import gin
import bpy
from mathutils import Color


@gin.configurable
def create_urban_materials():
    mats = {}
    mat = bpy.data.materials.new("Urban_Asphalt")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.15, 0.15, 0.16, 1)
        bsdf.inputs["Roughness"].default_value = 0.9
    mats["asphalt"] = mat

    mat = bpy.data.materials.new("Urban_Concrete")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.7, 0.7, 0.72, 1)
        bsdf.inputs["Roughness"].default_value = 0.7
    mats["concrete"] = mat

    mat = bpy.data.materials.new("Urban_Curb")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.6, 0.6, 0.62, 1)
        bsdf.inputs["Roughness"].default_value = 0.8
    mats["curb"] = mat

    return mats


@gin.configurable
def apply_surface_material(obj, material_name, materials_cache=None):
    if materials_cache is None:
        materials_cache = create_urban_materials()
    mat = materials_cache.get(material_name)
    if mat and obj and hasattr(obj, 'data') and obj.data:
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
