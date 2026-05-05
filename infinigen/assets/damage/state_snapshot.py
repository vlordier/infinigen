import gin
from dataclasses import dataclass, field
import json
import bpy

@gin.configurable
@dataclass
class SnapshottedObject:
    name: str
    location: list = field(default_factory=lambda: [0, 0, 0])
    rotation: list = field(default_factory=lambda: [0, 0, 0])
    scale: list = field(default_factory=lambda: [1, 1, 1])

@dataclass
class SnapshottedCamera:
    name: str
    location: list = field(default_factory=lambda: [0, 0, 0])
    rotation: list = field(default_factory=lambda: [0, 0, 0])
    focal_length: float = 35.0

@gin.configurable
@dataclass
class SceneSnapshot:
    objects: dict = field(default_factory=dict)
    cameras: dict = field(default_factory=dict)
    seed: int = 0
    damage_config: dict = field(default_factory=dict)

@gin.configurable
def snapshot_intact_state(blend_path=None):
    """Save the current Blender scene state as a .blend file snapshot."""
    if blend_path is None:
        blend_path = bpy.path.abspath("//snapshot_intact.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path, copy=True)
    return blend_path

@gin.configurable
def restore_intact_state(blend_path):
    """Restore scene from snapshot blend file."""
    bpy.ops.wm.open_mainfile(filepath=blend_path)
