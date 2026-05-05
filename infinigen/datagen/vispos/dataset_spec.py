import gin
from dataclasses import dataclass, field


@gin.configurable
@dataclass
class VisPosDatasetSpec:
    name: str = "vispos_default"
    description: str = ""
    scene_types: list = field(default_factory=lambda: ["nature_forest", "urban_suburban"])
    num_scenes: int = 100
    seasons: list = field(default_factory=lambda: ["summer"])
    times_of_day: list = field(default_factory=lambda: ["noon"])
    platforms: list = field(default_factory=lambda: ["isr_orbit"])
    paired_damage: bool = False
    damage_types: list = field(default_factory=lambda: ["earthquake"])
    damage_severities: list = field(default_factory=lambda: ["moderate"])
    damage_progression_stages: int = 2
    weather_types: list = field(default_factory=lambda: ["clear"])
    weather_levels: list = field(default_factory=lambda: [0])
    modalities: list = field(default_factory=lambda: ["eo"])
    ir_fidelity: str = "heuristic"
    tasks: list = field(default_factory=lambda: ["geoloc", "vpr", "vo_slam"])
    invariance_axes: list = field(default_factory=lambda: ["altitude", "rotation"])
    poses_per_scene: int = 50
    frames_per_trajectory: int = 100
    frame_rate: float = 30.0
    resolution: tuple = (1920, 1080)
    render_engine: str = "cycles"
    dataset_mode: str = "diversity"


DATASET_PRESETS = {
    "diversity_5k_eo": VisPosDatasetSpec(
        name="diversity_5k_eo", num_scenes=5000,
        scene_types=["nature_forest", "nature_desert", "urban_suburban"],
        seasons=["summer"], times_of_day=["noon"],
        platforms=["isr_orbit"], modalities=["eo"],
        dataset_mode="diversity"),
    "invariance_100_winter": VisPosDatasetSpec(
        name="invariance_100_winter", num_scenes=100,
        scene_types=["urban_dense_city"],
        seasons=["spring", "summer", "autumn", "winter"],
        times_of_day=["dawn", "morning", "noon", "afternoon", "dusk"],
        platforms=["isr_orbit", "fpv_scout"],
        modalities=["eo", "lwir"],
        paired_damage=True, dataset_mode="invariance"),
    "hybrid_ref": VisPosDatasetSpec(
        name="hybrid_ref", num_scenes=5100,
        scene_types=["nature_forest", "nature_desert", "urban_dense_city", "urban_suburban"],
        dataset_mode="hybrid"),
}


def get_preset(name):
    return DATASET_PRESETS.get(name)
