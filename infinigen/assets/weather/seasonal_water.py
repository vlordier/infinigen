# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Apply seasonal water state: freeze water surfaces, adjust river flow."""

import logging

import bpy
import gin

from infinigen.assets.materials.terrain.ice import Ice
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler

from .season_system import SeasonState

logger = logging.getLogger(__name__)


_RIVER_FLOW_SPEEDS = {
    "spring": 1.5,
    "summer": 1.0,
    "autumn": 0.6,
    "winter": 0.1,
}


@gin.configurable
def apply_season_to_water(water_objects, season_state, river_name=None):
    """Apply seasonal water state.

    When season_state is None, no adjustments are made.
    water_objects can be a list of bpy.types.Object or None.
    """
    if season_state is None or water_objects is None:
        return

    if not isinstance(water_objects, list):
        water_objects = [water_objects]

    season = season_state.season

    if season_state.water_is_frozen:
        _apply_ice_to_water(water_objects)

    _adjust_river_flow(river_name or "simulated_river", season)


def _apply_ice_to_water(water_objects):
    """Replace water material with ice material on water surfaces."""
    ice = Ice()
    for obj in water_objects:
        if obj is None or obj.type != "MESH":
            continue
        try:
            ice.apply(obj)
            logger.debug(f"Applied ice material to {obj.name}")
        except Exception as e:
            logger.warning(f"Could not apply ice to {obj.name}: {e}")


def _adjust_river_flow(river_name, season):
    """Adjust river flow speed parameter if the river object exists."""
    river_obj = bpy.data.objects.get(river_name)
    if river_obj is None:
        return

    speed = _RIVER_FLOW_SPEEDS.get(season, 1.0)

    if river_obj.modifiers:
        for mod in river_obj.modifiers:
            if mod.type == "NODES" and mod.node_group is not None:
                _set_flow_speed(mod.node_group, speed)


def _set_flow_speed(node_group, speed):
    """Try to find and set a flow speed value node in the node group."""
    for node in node_group.nodes:
        if node.type == "VALUE" and "flow" in (node.label or "").lower():
            node.outputs[0].default_value = speed
            break
