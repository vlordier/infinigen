import gin
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FlightPlatform(Enum):
    ISR_ORBIT = "isr_orbit"
    ISR_RASTER = "isr_raster"
    ISR_LOITER = "isr_loiter"
    ISR_PLANE = "isr_plane"
    FPV_RACING = "fpv_racing"
    FPV_SCOUT = "fpv_scout"
    UGV_WHEELED = "ugv_wheeled"
    UGV_TRACKED = "ugv_tracked"
    SATELLITE = "satellite"


@gin.configurable
@dataclass
class LoopClosureConfig:
    enabled: bool = False
    revisit_interval_frames: int = 300
    revisit_count: int = 3
    min_approach_angle_deg: float = 30.0
    loop_difficulty_distribution: dict = field(default_factory=lambda: {"easy": 0.4, "medium": 0.4, "hard": 0.2})


@gin.configurable
@dataclass
class GPSAvailabilitySchedule:
    windows: list = field(default_factory=lambda: [(0, -1, "nominal")])


@gin.configurable
@dataclass
class FlightRigSpec:
    platform: str = "isr_orbit"
    altitude_range: tuple = (100.0, 500.0)
    speed_range: tuple = (15.0, 40.0)
    sensor_fov_range: tuple = (30.0, 60.0)
    look_angle_range: tuple = (30.0, 60.0)
    gimbal_stabilization: bool = True
    multi_sensor: bool = False
    sensor_baseline: float = 0.0
    trajectory_total_distance_m: float = 5000.0
    trajectory_duration_s: float = 300.0
    loop_closure: Optional[LoopClosureConfig] = None
    gps_schedule: Optional[GPSAvailabilitySchedule] = None


_PLATFORM_DEFAULTS = {
    FlightPlatform.ISR_ORBIT: dict(
        altitude_range=(100, 500), speed_range=(15, 40),
        sensor_fov_range=(30, 60), look_angle_range=(30, 60),
        gimbal_stabilization=True, trajectory_total_distance_m=5000, trajectory_duration_s=300),
    FlightPlatform.ISR_RASTER: dict(
        altitude_range=(80, 300), speed_range=(20, 50),
        sensor_fov_range=(40, 80), look_angle_range=(0, 10),
        gimbal_stabilization=True, trajectory_total_distance_m=3000, trajectory_duration_s=180),
    FlightPlatform.ISR_LOITER: dict(
        altitude_range=(50, 500), speed_range=(0, 5),
        sensor_fov_range=(20, 60), look_angle_range=(45, 90),
        gimbal_stabilization=True, trajectory_total_distance_m=1000, trajectory_duration_s=300),
    FlightPlatform.ISR_PLANE: dict(
        altitude_range=(500, 5000), speed_range=(50, 150),
        sensor_fov_range=(5, 30), look_angle_range=(30, 60),
        gimbal_stabilization=True, trajectory_total_distance_m=50000, trajectory_duration_s=600),
    FlightPlatform.FPV_RACING: dict(
        altitude_range=(5, 50), speed_range=(30, 80),
        sensor_fov_range=(80, 120), look_angle_range=(0, 30),
        gimbal_stabilization=False, trajectory_total_distance_m=2000, trajectory_duration_s=60),
    FlightPlatform.FPV_SCOUT: dict(
        altitude_range=(10, 100), speed_range=(10, 30),
        sensor_fov_range=(60, 100), look_angle_range=(0, 20),
        gimbal_stabilization=False, trajectory_total_distance_m=1500, trajectory_duration_s=120),
    FlightPlatform.UGV_WHEELED: dict(
        altitude_range=(0.5, 2), speed_range=(2, 15),
        sensor_fov_range=(60, 100), look_angle_range=(0, 20),
        gimbal_stabilization=False, trajectory_total_distance_m=3000, trajectory_duration_s=300),
    FlightPlatform.UGV_TRACKED: dict(
        altitude_range=(0.5, 2), speed_range=(1, 10),
        sensor_fov_range=(60, 100), look_angle_range=(0, 20),
        gimbal_stabilization=False, trajectory_total_distance_m=3000, trajectory_duration_s=600),
    FlightPlatform.SATELLITE: dict(
        altitude_range=(200000, 500000), speed_range=(7000, 7000),
        sensor_fov_range=(1, 5), look_angle_range=(0, 1),
        gimbal_stabilization=True, trajectory_total_distance_m=200000, trajectory_duration_s=30),
}


def get_platform_rig_spec(platform_name):
    plat = FlightPlatform(platform_name) if isinstance(platform_name, str) else platform_name
    defaults = _PLATFORM_DEFAULTS.get(plat, {})
    return FlightRigSpec(platform=plat.value, **defaults)


def sample_platform_param(param_range):
    import random
    lo, hi = param_range
    return random.uniform(lo, hi)
