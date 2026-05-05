import os
import json


def create_dataset_structure(base_path):
    dirs = [
        "scenes",
        "splits",
    ]
    for d in dirs:
        os.makedirs(os.path.join(base_path, d), exist_ok=True)

    splits = {
        "train_scenes.txt": "",
        "val_scenes.txt": "",
        "test_scenes.txt": "",
    }
    for fname, content in splits.items():
        path = os.path.join(base_path, "splits", fname)
        if not os.path.exists(path):
            with open(path, 'w') as f:
                f.write(content)


def get_scene_dir(base_path, scene_id):
    return os.path.join(base_path, "scenes", f"scene_{scene_id:04d}")


def get_variant_dir(base_path, scene_id, variant_name):
    return os.path.join(get_scene_dir(base_path, scene_id), variant_name)


def get_modality_dir(base_path, scene_id, variant_name, modality):
    return os.path.join(get_variant_dir(base_path, scene_id, variant_name), modality)


def get_frame_path(base_path, scene_id, variant_name, modality, frame_idx):
    d = get_modality_dir(base_path, scene_id, variant_name, modality)
    return os.path.join(d, f"frame_{frame_idx:04d}.exr")


def write_dataset_spec(base_path, spec):
    path = os.path.join(base_path, "dataset_spec.json")
    with open(path, 'w') as f:
        json.dump(spec if isinstance(spec, dict) else spec.__dict__, f, indent=2, default=str)
