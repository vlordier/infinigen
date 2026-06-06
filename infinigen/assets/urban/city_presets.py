from infinigen.assets.urban.templates import DistrictTemplateConfig


CITY_PRESETS = {
    "european_old": {
        "skeleton_type": "radial",
        "skeleton_params": {"n_radials": 10, "n_rings": 5, "irregularity": 0.2},
        "zone_templates": {
            "core":  {"template": "organic_grid",  "config": DistrictTemplateConfig(lot_width=15, lot_depth=20, irregularity=0.15, internal_road_width=6.0)},
            "inner": {"template": "organic_grid",  "config": DistrictTemplateConfig(lot_width=20, lot_depth=25, irregularity=0.1, internal_road_width=8.0)},
            "outer": {"template": "rectangular_grid", "config": DistrictTemplateConfig(lot_width=25, lot_depth=30, irregularity=0.05, internal_road_width=10.0)},
        },
        "regional_style": "mediterranean",
    },
    "medieval_village": {
        "skeleton_type": "organic_spine",
        "skeleton_params": {"n_branches": 6, "irregularity": 0.5},
        "zone_templates": {
            "inner": {"template": "medieval_organic", "config": DistrictTemplateConfig(lot_width=8, lot_depth=10, lot_min_area=20, dead_end_chance=0.3, density=0.9, internal_road_width=4.0)},
        },
        "regional_style": "mediterranean",
    },
    "suburban_estonia": {
        "skeleton_type": "grid",
        "skeleton_params": {"rows": 4, "cols": 4, "irregularity": 0.1},
        "zone_templates": {
            "inner": {"template": "suburban_cul_de_sac", "config": DistrictTemplateConfig(lot_width=30, lot_depth=40, lot_min_area=500, internal_road_width=8.0, density=0.3)},
        },
        "regional_style": "baltic",
    },
    "ukrainian_city": {
        "skeleton_type": "grid",
        "skeleton_params": {"rows": 6, "cols": 6, "irregularity": 0.05},
        "zone_templates": {
            "inner": {"template": "rectangular_grid", "config": DistrictTemplateConfig(lot_width=40, lot_depth=50, lot_min_area=500, internal_road_width=16.0)},
        },
        "regional_style": "soviet",
    },
    "ukrainian_village": {
        "skeleton_type": "single_spine",
        "skeleton_params": {"n_lanes": 8},
        "zone_templates": {
            "outer": {"template": "garden_plots", "config": DistrictTemplateConfig(lot_width=15, lot_depth=40, lot_min_area=300, internal_road_width=5.0)},
        },
        "regional_style": "soviet",
    },
    "soviet_microdistrict": {
        "skeleton_type": "radial",
        "skeleton_params": {"n_radials": 6, "n_rings": 3, "irregularity": 0.05},
        "zone_templates": {
            "core":  {"template": "soviet_block", "config": DistrictTemplateConfig(lot_width=80, lot_depth=100, internal_road_width=24.0, density=0.3)},
            "inner": {"template": "soviet_block", "config": DistrictTemplateConfig(lot_width=60, lot_depth=80, internal_road_width=18.0, density=0.4)},
            "outer": {"template": "sparse_organic", "config": DistrictTemplateConfig(lot_width=50, lot_depth=50, lot_min_area=1000)},
        },
        "regional_style": "soviet",
    },
    "vatican": {
        "skeleton_type": "osmnx",
        "skeleton_params": {"place": "Vatican City", "network_type": "drive"},
        "zone_templates": {
            "inner": {"template": "medieval_organic", "config": DistrictTemplateConfig(lot_width=10, lot_depth=12, lot_min_area=30, dead_end_chance=0.2, density=0.8, internal_road_width=4.0)},
        },
        "regional_style": "mediterranean",
    },
    "paris_louvre": {
        "skeleton_type": "osmnx",
        "skeleton_params": {"place": "Louvre, Paris, France", "network_type": "drive"},
        "zone_templates": {
            "inner": {"template": "organic_grid", "config": DistrictTemplateConfig(lot_width=15, lot_depth=20, lot_min_area=100, irregularity=0.15, internal_road_width=6.0)},
        },
        "regional_style": "mediterranean",
    },
    "kyiv_center": {
        "skeleton_type": "osmnx",
        "skeleton_params": {"place": "Kyiv, Ukraine", "network_type": "drive"},
        "zone_templates": {
            "inner": {"template": "rectangular_grid", "config": DistrictTemplateConfig(lot_width=30, lot_depth=40, lot_min_area=300, internal_road_width=12.0)},
        },
        "regional_style": "soviet",
    },
    "berlin_mitte": {
        "skeleton_type": "osmnx",
        "skeleton_params": {"place": "Berlin, Germany", "network_type": "drive"},
        "zone_templates": {
            "inner": {"template": "organic_grid", "config": DistrictTemplateConfig(lot_width=20, lot_depth=25, lot_min_area=150, irregularity=0.1, internal_road_width=8.0)},
        },
        "regional_style": "mediterranean",
    },
    "rostov_on_don": {
        "skeleton_type": "osmnx",
        "skeleton_params": {"place": "Rostov-on-Don, Russia", "network_type": "drive"},
        "zone_templates": {
            "inner": {"template": "rectangular_grid", "config": DistrictTemplateConfig(lot_width=25, lot_depth=35, lot_min_area=250, internal_road_width=12.0)},
        },
        "regional_style": "soviet",
    },
}


def load_preset(name: str) -> dict:
    if name not in CITY_PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(CITY_PRESETS.keys())}")
    return dict(CITY_PRESETS[name])
