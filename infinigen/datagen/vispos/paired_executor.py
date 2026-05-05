import gin
import os
import json
from .ground_truth import FrameMetadata
from .output_layout import create_dataset_structure, get_scene_dir, get_variant_dir, get_modality_dir


@gin.configurable
class PairedExecutor:
    def __init__(self, base_output_path, spec):
        self.base_path = base_output_path
        self.spec = spec
        create_dataset_structure(base_output_path)

    def execute_plan(self, plan):
        results = []
        for entry in plan:
            scene_id, scene_type, season, tod, platform, weather, weather_level, dam_type, dam_sev = entry
            variant_name = f"{season}_{tod}_{platform}_{weather}_{weather_level}_{dam_type}_{dam_sev}"
            variant_dir = get_variant_dir(self.base_path, scene_id, variant_name)
            os.makedirs(variant_dir, exist_ok=True)

            metadata = FrameMetadata(
                scene_id=f"scene_{scene_id:04d}",
                season=season,
                time_of_day=tod,
                is_damaged=(dam_type != "none"),
                damage_type=dam_type,
                damage_severity=dam_sev,
            )
            metadata_path = os.path.join(variant_dir, "metadata.json")
            metadata.to_json(metadata_path)

            results.append({
                "scene_id": scene_id,
                "variant": variant_name,
                "variant_dir": variant_dir,
            })
        return results

    def run_multi_stage(self, scene_id, scene_type, season, tod, platform, damage_type, stages=5):
        stage_results = []
        for stage in range(min(stages, self.spec.damage_progression_stages)):
            severity_map = {0: "none", 1: "mild", 2: "moderate", 3: "severe", 4: "total"}
            variant_name = f"{season}_{tod}_{platform}_{damage_type}_{severity_map[stage]}"
            variant_dir = get_variant_dir(self.base_path, scene_id, variant_name)
            os.makedirs(variant_dir, exist_ok=True)

            metadata = FrameMetadata(
                scene_id=f"scene_{scene_id:04d}",
                season=season,
                time_of_day=tod,
                is_damaged=(stage > 0),
                damage_type=damage_type,
                damage_severity=severity_map[stage],
            )
            metadata.to_json(os.path.join(variant_dir, "metadata.json"))
            stage_results.append({"stage": stage, "variant_dir": variant_dir})
        return stage_results
