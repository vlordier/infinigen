import gin
from dataclasses import dataclass, field
from typing import Optional

@gin.configurable
@dataclass
class RegionalStyle:
    name: str = "generic"
    building_material_weights: dict = field(default_factory=lambda: {"concrete": 0.4, "brick": 0.3, "stucco": 0.2, "glass": 0.1})
    roof_style_weights: dict = field(default_factory=lambda: {"flat": 0.5, "pitched": 0.3, "hip": 0.2})
    window_style_weights: dict = field(default_factory=lambda: {"symmetric_grid": 0.8, "irregular": 0.2})
    building_height_range: tuple = (1, 6)
    building_color_palette: list = field(default_factory=lambda: ["#c0c0c0", "#d4c5b9", "#a8a8a8", "#e8dcc8", "#b0a090"])
    tree_species_weights: dict = field(default_factory=lambda: {"oak": 0.3, "pine": 0.3, "birch": 0.2, "maple": 0.2})
    undergrowth_density: float = 0.3
    road_material: str = "asphalt_dark"
    sidewalk_material: str = "concrete_tile"
    streetlight_style: str = "modern_metal"
    landmark_type_weights: dict = field(default_factory=lambda: {"water_tower": 0.3, "cell_tower": 0.3, "church": 0.2, "silo": 0.2})
    preferred_elevation_range: tuple = (0, 500)
    terrain_flattening_aggressiveness: float = 0.5

REGIONAL_STYLES = {
    "soviet": RegionalStyle(
        name="soviet",
        building_material_weights={"concrete_panel": 0.6, "brick": 0.3, "stucco": 0.1},
        roof_style_weights={"flat": 0.8, "pitched_shallow": 0.2},
        window_style_weights={"symmetric_grid": 0.9, "irregular": 0.1},
        building_height_range=(3, 9),
        building_color_palette=["#b0b0b0", "#d4c5b9", "#c8b898", "#a89880"],
        tree_species_weights={"birch": 0.4, "pine": 0.3, "poplar": 0.2, "oak": 0.1},
        undergrowth_density=0.3,
        road_material="asphalt_gray",
        sidewalk_material="concrete_gray",
        streetlight_style="soviet_concrete",
        landmark_type_weights={"water_tower": 0.3, "silo": 0.3, "cell_tower": 0.2, "church_orthodox": 0.2},
    ),
    "baltic": RegionalStyle(
        name="baltic",
        building_material_weights={"red_brick": 0.5, "timber_frame": 0.2, "concrete": 0.2, "stucco": 0.1},
        roof_style_weights={"pitched_steep": 0.7, "hip": 0.2, "flat": 0.1},
        window_style_weights={"dormer": 0.4, "symmetric_grid": 0.4, "irregular": 0.2},
        building_height_range=(2, 5),
        building_color_palette=["#8b4513", "#a0522d", "#cd853f", "#f5deb3", "#d2b48c"],
        tree_species_weights={"pine": 0.35, "spruce": 0.3, "birch": 0.25, "oak": 0.1},
        undergrowth_density=0.6,
        road_material="cobblestone_old_town",
        sidewalk_material="stone_tile",
        streetlight_style="scandinavian_wood",
        landmark_type_weights={"lighthouse": 0.3, "church": 0.3, "water_tower": 0.2, "cell_tower": 0.2},
    ),
    "mediterranean": RegionalStyle(
        name="mediterranean",
        building_material_weights={"white_stucco": 0.5, "stone": 0.3, "concrete": 0.2},
        roof_style_weights={"flat_terrace": 0.6, "shallow_pitched": 0.3, "dome": 0.1},
        window_style_weights={"arched_shutter": 0.5, "rectangular_shutter": 0.3, "irregular": 0.2},
        building_height_range=(1, 4),
        building_color_palette=["#ffffff", "#faf0e6", "#f5deb3", "#deb887", "#4169e1"],
        tree_species_weights={"cypress": 0.3, "olive": 0.25, "pine": 0.25, "palm": 0.2},
        undergrowth_density=0.2,
        road_material="stone_light",
        sidewalk_material="stone_tile",
        streetlight_style="mediterranean_wrought_iron",
        landmark_type_weights={"church": 0.3, "lighthouse": 0.25, "water_tower": 0.25, "stadium": 0.2},
    ),
    "generic": RegionalStyle(
        name="generic",
        building_material_weights={"concrete": 0.4, "brick": 0.3, "stucco": 0.2, "glass": 0.1},
        roof_style_weights={"flat": 0.5, "pitched": 0.3, "hip": 0.2},
        building_height_range=(1, 6),
        building_color_palette=["#c0c0c0", "#d4c5b9", "#a8a8a8", "#e8dcc8", "#b0a090"],
    ),
}

@gin.configurable
def get_regional_style(style_name: str = "generic") -> RegionalStyle:
    return REGIONAL_STYLES.get(style_name, REGIONAL_STYLES["generic"])

def list_regional_styles():
    return list(REGIONAL_STYLES.keys())
