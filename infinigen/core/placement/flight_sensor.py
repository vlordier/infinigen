import gin
from dataclasses import dataclass, field

@gin.configurable
@dataclass
class SensorCharacteristics:
    focal_length_mm: float = 35.0
    aperture: float = 2.8
    hfov_deg: float = 60.0
    resolution: tuple = (1920, 1080)
    pixel_pitch_um: float = 5.5
    render_pyramid_levels: int = 1
    auto_exposure: bool = False
    exposure_target_mean: float = 0.18
    exposure_slew_rate_stops_per_frame: float = 0.5
    exposure_min_stops: float = -3.0
    exposure_max_stops: float = 3.0
    exposure_metering_mode: str = "center_weighted"
    motion_blur: bool = False
    rolling_shutter: bool = False
    rolling_shutter_mode: str = "compositor"
    lens_distortion: bool = False
    vibration_profile: str = "none"
    noise_model: str = "none"
    lens_flare: bool = False
    saturation_behavior: str = "clip"
    compression: str = "none"
    water_specular: bool = False

@gin.configurable
@dataclass
class MultiSensorTimingSpec:
    exposure_stagger_us: list = field(default_factory=lambda: [0, 0, 0, 0])
    clock_drift_ppm: float = 0.0
    timestamp_convention: str = "exposure_center"
    frame_trigger_mode: str = "simultaneous"

@dataclass
class MultiSensorRig:
    eo_camera = None
    ir_cameras: dict = field(default_factory=dict)
    baseline_mm: float = 50.0
    timing: MultiSensorTimingSpec = field(default_factory=MultiSensorTimingSpec)

@gin.configurable
@dataclass
class StereoRigSpec:
    camera_count: int = 2
    baseline_m: float = 0.5
    baseline_orientation: str = "horizontal"
    camera_layout: list = field(default_factory=lambda: [[-0.25, 0, 0], [0.25, 0, 0]])
    synchronization: str = "hardware_sync"
    depth_ground_truth: bool = False
