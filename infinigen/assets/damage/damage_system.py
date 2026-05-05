import gin
import random
import bpy
from mathutils import Vector
from dataclasses import dataclass, field
from enum import Enum

class DamageSeverity(Enum):
    INTACT = 0
    MILD = 1
    MODERATE = 2
    SEVERE = 3
    TOTAL = 4

class DamageType(Enum):
    EARTHQUAKE = "earthquake"
    WAR = "war"
    BOTH = "both"

@gin.configurable
@dataclass
class DamageConfig:
    damage_type: str = "earthquake"
    severity: int = 2
    progression_stages: int = 2
    seed: int = 42

@gin.configurable
class DamageStageExecutor:
    def __init__(self, config: DamageConfig = None):
        self.config = config or DamageConfig()
    
    def apply_terrain_damage(self, terrain_objects, severity):
        """Apply craters, ground fractures to terrain."""
        from .shared.displacement import apply_crater
        for obj in terrain_objects:
            if obj and obj.type == 'MESH':
                n_craters = severity * 3
                for _ in range(n_craters):
                    cx = random.uniform(-50, 50)
                    cy = random.uniform(-50, 50)
                    radius = random.uniform(2, 10) * (severity / 2)
                    depth = random.uniform(0.5, 3) * (severity / 2)
                    apply_crater(obj, (cx, cy), radius, depth)
    
    def apply_structure_damage(self, building_objects, severity):
        """Apply structural damage to buildings."""
        from .shared.fracture import fracture_object
        for obj in building_objects:
            if obj and obj.type == 'MESH' and severity >= 2:
                fracture_intensity = min(1.0, severity / 4.0)
                if random.random() < fracture_intensity * 0.5:
                    fracture_object(obj, fracture_intensity)
    
    def apply_object_damage(self, objects, severity):
        """Topple and displace objects."""
        for obj in objects:
            if obj and random.random() < severity * 0.15:
                obj.location.x += random.uniform(-0.5, 0.5) * severity
                obj.location.y += random.uniform(-0.5, 0.5) * severity
                obj.rotation_euler.x += random.uniform(-0.3, 0.3) * severity
                obj.rotation_euler.y += random.uniform(-0.3, 0.3) * severity
    
    def execute_stage(self, terrain_objects, building_objects, other_objects, severity):
        self.apply_terrain_damage(terrain_objects, severity)
        self.apply_structure_damage(building_objects, severity)
        self.apply_object_damage(other_objects, severity)
