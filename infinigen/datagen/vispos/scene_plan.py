import gin
import random
import itertools
from collections import defaultdict
from .dataset_spec import VisPosDatasetSpec


@gin.configurable
class ScenePlanGenerator:
    def __init__(self, spec: VisPosDatasetSpec = None, seed: int = 42):
        self.spec = spec or VisPosDatasetSpec()
        self.seed = seed
        random.seed(seed)

    def generate_plan(self):
        spec = self.spec
        plan = []
        scene_ids = list(range(spec.num_scenes))

        if spec.dataset_mode == "diversity":
            plan = self._plan_diversity(scene_ids)
        elif spec.dataset_mode == "invariance":
            plan = self._plan_invariance(scene_ids)
        elif spec.dataset_mode == "hybrid":
            n_div = int(spec.num_scenes * 0.95)
            n_inv = spec.num_scenes - n_div
            plan = self._plan_diversity(list(range(n_div))) + self._plan_invariance(list(range(n_div, n_div + n_inv)))

        return plan

    def _plan_diversity(self, scene_ids):
        plan = []
        for sid in scene_ids:
            scene_type = random.choice(self.spec.scene_types)
            season = random.choice(self.spec.seasons)
            tod = random.choice(self.spec.times_of_day)
            platform = random.choice(self.spec.platforms)
            weather = random.choice(self.spec.weather_types)
            weather_level = random.choice(self.spec.weather_levels)
            damage = "none"
            damage_sev = "none"
            if self.spec.paired_damage:
                dt = random.choice(self.spec.damage_types)
                ds = random.choice(self.spec.damage_severities)
                plan.append((sid, scene_type, season, tod, platform, weather, weather_level, "intact", "none"))
                plan.append((sid, scene_type, season, tod, platform, weather, weather_level, dt, ds))
                continue
            plan.append((sid, scene_type, season, tod, platform, weather, weather_level, damage, damage_sev))
        return plan

    def _plan_invariance(self, scene_ids):
        plan = []
        axes = [self.spec.seasons, self.spec.times_of_day, self.spec.platforms,
                self.spec.weather_types, self.spec.weather_levels]
        if self.spec.paired_damage:
            axes.append([("intact", "none")] + [(dt, ds) for dt in self.spec.damage_types for ds in self.spec.damage_severities])
        else:
            axes.append([("none", "none")])

        for sid in scene_ids:
            scene_type = random.choice(self.spec.scene_types)
            combos = list(itertools.product(*axes))
            random.shuffle(combos)
            for combo in combos[:min(256, len(combos))]:
                season, tod, platform, wt, wl, (dam_type, dam_sev) = combo
                plan.append((sid, scene_type, season, tod, platform, wt, wl, dam_type, dam_sev))
        return plan

    def get_total_frames(self, plan):
        return len(plan) * self.spec.poses_per_scene

    def get_storage_estimate_gb(self, plan):
        total_frames = self.get_total_frames(plan)
        n_channels = len(self.spec.modalities) + 3
        bytes_per_frame = self.spec.resolution[0] * self.spec.resolution[1] * n_channels * 2
        return total_frames * bytes_per_frame / (1024**3)


@gin.configurable
def plan_dataset(spec: VisPosDatasetSpec):
    gen = ScenePlanGenerator(spec)
    plan = gen.generate_plan()
    return {
        "plan": plan,
        "total_scene_variants": len(plan),
        "total_frames_est": gen.get_total_frames(plan),
        "storage_est_gb": gen.get_storage_estimate_gb(plan),
    }
