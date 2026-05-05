# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Apply seasonal adjustments to terrain meshes: snow cover, color shifts, wetness."""

import logging

import bpy
import gin
import numpy as np

from infinigen.core import surface
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler

from .season_system import SeasonState

logger = logging.getLogger(__name__)


def _winter_snow_shader(nw: NodeWrangler):
    geometry = nw.new_node(Nodes.NewGeometry)
    noise_texture = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": geometry.outputs["Position"],
            "Scale": 12.0,
            "Detail": 6.0,
        },
    )
    ramp = nw.new_node(
        Nodes.ColorRamp, input_kwargs={"Fac": noise_texture.outputs["Fac"]}
    )
    ramp.color_ramp.elements[0].position = 0.4
    ramp.color_ramp.elements[0].color = (0.82, 0.82, 0.88, 1.0)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = (0.72, 0.76, 0.85, 1.0)
    principled = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": ramp.outputs["Color"],
            "Roughness": 0.30,
            "Subsurface Weight": 0.15,
        },
    )
    nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": principled},
        attrs={"is_active_output": True},
    )
    return principled


def _autumn_shader(nw: NodeWrangler):
    principled = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": (0.22, 0.12, 0.06, 1.0),
            "Roughness": 0.85,
        },
    )
    nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": principled},
        attrs={"is_active_output": True},
    )
    return principled


def _spring_shader(nw: NodeWrangler):
    principled = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": (0.06, 0.04, 0.03, 1.0),
            "Roughness": 0.25,
            "Specular IOR Level": 0.5,
        },
    )
    nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": principled},
        attrs={"is_active_output": True},
    )
    return principled


@gin.configurable
def get_snow_line(snow_cover, max_alt=100, min_alt=-10):
    return max_alt - (snow_cover * (max_alt - min_alt))


@gin.configurable
def apply_season_to_terrain(terrain_mesh, season_state, snow_line=None):
    """Adjust terrain material based on season.

    When season_state is None or terrain_mesh is None, no adjustments are made.
    """
    if season_state is None or terrain_mesh is None:
        return

    if not isinstance(terrain_mesh, bpy.types.Object) or terrain_mesh.type != "MESH":
        return

    season = season_state.season

    if season == "winter" and season_state.snow_cover > 0:
        if snow_line is None:
            snow_line = get_snow_line(season_state.snow_cover)
        _apply_snow_layer(terrain_mesh, snow_line)
    elif season == "autumn":
        _set_season_attribute(terrain_mesh, "season_autumn", 1.0)
        surface.add_material(terrain_mesh, _autumn_shader, selection="season_autumn")
    elif season == "spring":
        _set_season_attribute(terrain_mesh, "season_spring", 1.0)
        surface.add_material(terrain_mesh, _spring_shader, selection="season_spring")


def _set_season_attribute(mesh_obj, attr_name, value):
    mesh = mesh_obj.data
    n = len(mesh.vertices)
    vals = np.full(n, value, dtype=np.float32)
    if attr_name not in mesh.attributes:
        mesh.attributes.new(name=attr_name, type="FLOAT", domain="POINT")
    mesh.attributes[attr_name].data.foreach_set("value", vals)


def _apply_snow_layer(terrain_mesh, snow_line):
    """Tag upward-facing faces above snow line and apply snow material."""
    mesh = terrain_mesh.data
    verts = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", verts)
    co = verts.reshape(-1, 3)

    normals = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("normal", normals)
    normals = normals.reshape(-1, 3)

    upward = normals[:, 2] > 0.2
    altitude_mask = co[:, 2] > snow_line
    snow_verts = (upward & altitude_mask).astype(np.float32)

    attr_name = "season_snow"
    if attr_name not in mesh.attributes:
        mesh.attributes.new(name=attr_name, type="FLOAT", domain="POINT")
    mesh.attributes[attr_name].data.foreach_set("value", snow_verts)

    if snow_verts.max() > 0:
        surface.add_material(terrain_mesh, _winter_snow_shader, selection=attr_name)
