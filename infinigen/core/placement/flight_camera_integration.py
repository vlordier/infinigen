"""
Integration layer connecting flight camera policies to the existing infinigen camera system.
"""
import gin
import random
import bpy
from mathutils import Vector, Euler
from .flight_camera import (
    FlightPlatform, FlightRigSpec, LoopClosureConfig, GPSAvailabilitySchedule,
    get_platform_rig_spec, sample_platform_param
)
from .flight_trajectories import get_flight_policy


@gin.configurable
def create_flight_rig(platform_name="isr_orbit", location=(0, 0, 0)):
    """Create a Blender camera rig for a flight platform."""
    rig_spec = get_platform_rig_spec(platform_name)
    if rig_spec is None:
        return None
    
    # Create rig empty
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=location)
    rig = bpy.context.active_object
    rig.name = f"FlightRig_{platform_name}"
    rig.empty_display_size = 2.0
    
    # Create child camera
    bpy.ops.object.camera_add(location=(0, 0, 0.5))
    cam = bpy.context.active_object
    cam.name = f"Cam_{platform_name}"
    cam.parent = rig
    
    # Set camera parameters
    fov = sample_platform_param(rig_spec.sensor_fov_range)
    cam.data.angle = fov * (3.14159 / 180.0)
    cam.data.sensor_width = 36.0
    
    # Store rig spec as custom property
    rig["flight_platform"] = platform_name
    rig["rig_spec"] = {
        "platform": rig_spec.platform,
        "altitude_range": rig_spec.altitude_range,
        "speed_range": rig_spec.speed_range,
        "trajectory_distance_m": rig_spec.trajectory_total_distance_m,
    }
    
    return rig, cam, rig_spec


@gin.configurable
def spawn_flight_rigs(n_rigs=1, platform_types=None, scene_center=(0, 0, 0)):
    """Spawn multiple flight camera rigs."""
    if platform_types is None:
        platform_types = ["isr_orbit"]
    
    rigs = []
    for i in range(n_rigs):
        plat = random.choice(platform_types)
        offset = (random.uniform(-50, 50), random.uniform(-50, 50), 100)
        loc = (scene_center[0] + offset[0], scene_center[1] + offset[1], offset[2])
        result = create_flight_rig(plat, loc)
        if result:
            rigs.append(result)
    return rigs


@gin.configurable
def animate_flight_rig(rig, cam, rig_spec, num_frames=300, terrain_bvh=None):
    """Animate a flight camera rig using its platform-specific policy."""
    policy = get_flight_policy(rig_spec.platform, rig_spec, terrain_bvh)
    if policy is None:
        return
    
    start_frame = bpy.context.scene.frame_start
    for frame in range(num_frames):
        bpy.context.scene.frame_set(start_frame + frame)
        pos, rot = policy(None, rig, frame)
        rig.location = pos
        rig.rotation_euler = rot
        rig.keyframe_insert(data_path="location", frame=start_frame + frame)
        rig.keyframe_insert(data_path="rotation_euler", frame=start_frame + frame)


@gin.configurable
def setup_flight_scene(scene, platform_types=None, n_rigs=3, num_frames=300):
    """Complete setup: spawn rigs and animate them."""
    rigs = spawn_flight_rigs(n_rigs, platform_types)
    results = []
    for rig, cam, rig_spec in rigs:
        animate_flight_rig(rig, cam, rig_spec, num_frames)
        results.append((rig, cam))
    
    # Link rigs to scene
    for rig, cam in results:
        if rig.name not in scene.collection.objects:
            scene.collection.objects.link(rig)
        if cam.name not in scene.collection.objects:
            scene.collection.objects.link(cam)
    
    return results
