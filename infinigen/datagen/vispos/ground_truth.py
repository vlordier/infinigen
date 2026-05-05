import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional

@dataclass
class CameraIntrinsics:
    fx: float = 1000.0
    fy: float = 1000.0
    cx: float = 960.0
    cy: float = 540.0
    distortion: list = field(default_factory=lambda: [0, 0, 0, 0, 0])

@dataclass
class FrameMetadata:
    scene_id: str = ""
    frame_index: int = 0
    timestamp: float = 0.0
    camera_intrinsics: Optional[CameraIntrinsics] = None
    camera_extrinsics: list = field(default_factory=lambda: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    lat: float = 0.0
    lon: float = 0.0
    alt: float = 0.0
    gsd_meters: float = 0.1
    season: str = "summer"
    time_of_day: str = "noon"
    sun_azimuth: float = 180.0
    sun_elevation: float = 45.0
    weather: dict = field(default_factory=dict)
    is_damaged: bool = False
    damage_type: str = "none"
    damage_severity: str = "none"
    modality: str = "eo"
    place_id: str = ""
    sequence_id: str = ""
    gravity_vector_camera: list = field(default_factory=lambda: [0, 0, -9.81])
    metric_scale: float = 1.0
    stereo_baseline: float = 0.0
    relocalization: bool = False
    loop_closure: list = field(default_factory=list)
    gps_lat: float = 0.0
    gps_lon: float = 0.0
    gps_alt: float = 0.0
    gps_mode: str = "nominal"
    gps_true_lat: float = 0.0
    gps_true_lon: float = 0.0
    gps_true_alt: float = 0.0
    shadow_velocity_px_per_frame: float = 0.0
    thermal_contrast_std: float = 0.0
    exposure_time_s: float = 0.0
    gain_iso: float = 100.0
    feature_count: int = 0

    def to_dict(self):
        d = asdict(self)
        return d
    
    def to_json(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod 
    def from_json(cls, filepath):
        with open(filepath) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

@dataclass
class IMUReading:
    timestamp_ns: int = 0
    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = 0.0
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    mag_x: float = 0.0
    mag_y: float = 0.0
    mag_z: float = 0.0
    baro_pressure_hpa: float = 1013.25

@dataclass
class LoopClosureAnnotations:
    pairs: list = field(default_factory=list)
    relative_transform: list = field(default_factory=list)
    distance_m: list = field(default_factory=list)
    viewpoint_angle_deg: list = field(default_factory=list)
    difficulty: list = field(default_factory=list)
