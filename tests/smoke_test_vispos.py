#!/usr/bin/env python3
"""Smoke test for infinigen vispos feature set. Run inside Blender."""
import sys
import os
import bpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("INFINIGEN VISPOS SMOKE TEST")
print("=" * 60)

# 1. Season System
print("\n[1/6] SeasonState...")
from infinigen.assets.weather.season_system import SeasonState, create_season_state
winter = create_season_state("winter")
assert winter.season == "winter"
assert winter.snow_cover > 0.5
print(f"  OK: {winter}")

# 2. Time of Day
print("[2/6] TimeOfDay...")
from infinigen.assets.weather.time_of_day import TimeOfDay, get_tod_sun_params
# Returns dict with sun_elevation: ("uniform", range_min, range_max)
params = get_tod_sun_params(TimeOfDay.NOON, "summer")
assert "sun_elevation" in params
print(f"  OK: noon sun params keys = {list(params.keys())}")

# 3. Thermal properties
print("[3/6] ThermalProperties...")
from infinigen.assets.materials.thermal.properties import ThermalProperties, get_thermal_properties
props = get_thermal_properties("metal")
assert props.emissivity < 0.5, "Metal should have low emissivity"
print(f"  OK: metal emissivity = {props.emissivity}")

# 4. Flight cameras
print("[4/6] FlightPlatform...")
from infinigen.core.placement.flight_camera import FlightPlatform, get_platform_rig_spec
spec = get_platform_rig_spec("isr_orbit")
assert spec.platform == "isr_orbit"
print(f"  OK: ISR orbit altitude = {spec.altitude_range}")

# 5. Damage system
print("[5/6] DamageSystem...")
from infinigen.assets.damage.damage_system import DamageSeverity, DamageConfig
config = DamageConfig(damage_type="earthquake", severity=2, progression_stages=3)
assert config.severity == 2
print(f"  OK: damage config = {config}")

# 6. Urban + VisPos
print("[6/6] Urban + VisPos...")
from infinigen.assets.urban.regional_styles import get_regional_style
style = get_regional_style("soviet")
assert style.name == "soviet"

from infinigen.datagen.vispos.dataset_spec import VisPosDatasetSpec, get_preset
spec = get_preset("diversity_5k_eo")
assert spec.num_scenes == 5000

from infinigen.datagen.vispos.ground_truth import FrameMetadata, IMUReading
meta = FrameMetadata(scene_id="test_001", season="winter", modality="lwir")
meta.to_dict()

print(f"  OK: soviet style + diversity preset + FrameMetadata")

print("\n" + "=" * 60)
print("ALL SMOKE TESTS PASSED")
print("=" * 60)
