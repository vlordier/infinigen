import gin
from dataclasses import dataclass
from typing import Optional

@gin.configurable
@dataclass
class ThermalProperties:
    emissivity: float = 0.9
    solar_absorptivity: float = 0.7
    thermal_conductivity: float = 0.5
    heat_capacity: float = 1000.0
    density: float = 1500.0
    band_emissivities: Optional[dict] = None

MATERIAL_THERMAL_DEFAULTS = {
    "metal": ThermalProperties(emissivity=0.3, solar_absorptivity=0.5, thermal_conductivity=50.0, heat_capacity=500.0, density=7800.0),
    "wood": ThermalProperties(emissivity=0.9, solar_absorptivity=0.6, thermal_conductivity=0.15, heat_capacity=1700.0, density=600.0),
    "concrete": ThermalProperties(emissivity=0.9, solar_absorptivity=0.7, thermal_conductivity=1.5, heat_capacity=880.0, density=2400.0),
    "vegetation": ThermalProperties(emissivity=0.95, solar_absorptivity=0.8, thermal_conductivity=0.3, heat_capacity=2500.0, density=800.0),
    "water": ThermalProperties(emissivity=0.95, solar_absorptivity=0.9, thermal_conductivity=0.6, heat_capacity=4180.0, density=1000.0),
    "soil": ThermalProperties(emissivity=0.92, solar_absorptivity=0.75, thermal_conductivity=1.0, heat_capacity=800.0, density=1600.0),
    "glass": ThermalProperties(emissivity=0.85, solar_absorptivity=0.3, thermal_conductivity=1.0, heat_capacity=840.0, density=2500.0),
    "plastic": ThermalProperties(emissivity=0.9, solar_absorptivity=0.6, thermal_conductivity=0.2, heat_capacity=1500.0, density=1200.0),
    "asphalt": ThermalProperties(emissivity=0.93, solar_absorptivity=0.85, thermal_conductivity=0.75, heat_capacity=920.0, density=2300.0),
}

def get_thermal_properties(material_category: str) -> ThermalProperties:
    return MATERIAL_THERMAL_DEFAULTS.get(material_category, ThermalProperties())
