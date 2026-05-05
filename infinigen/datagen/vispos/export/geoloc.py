import json
import os


def export_geolocalization(scene_variants, output_dir):
    pairs = []
    for variant in scene_variants:
        metadata_path = os.path.join(variant["variant_dir"], "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                meta = json.load(f)
            pairs.append({
                "scene_id": meta["scene_id"],
                "variant": variant["variant"],
                "lat": meta.get("gps_lat", meta.get("lat", 0)),
                "lon": meta.get("gps_lon", meta.get("lon", 0)),
                "alt": meta.get("gps_alt", meta.get("alt", 0)),
            })
    out_path = os.path.join(output_dir, "geoloc_pairs.json")
    with open(out_path, 'w') as f:
        json.dump(pairs, f, indent=2)
    return out_path


def export_vpr_pairs(scene_variants, same_place_threshold_m=25):
    queries = []
    references = []
    for i, v1 in enumerate(scene_variants):
        for j, v2 in enumerate(scene_variants):
            if i >= j:
                continue
            m1_path = os.path.join(v1["variant_dir"], "metadata.json")
            m2_path = os.path.join(v2["variant_dir"], "metadata.json")
            if os.path.exists(m1_path) and os.path.exists(m2_path):
                with open(m1_path) as f:
                    m1 = json.load(f)
                with open(m2_path) as f:
                    m2 = json.load(f)
                is_same = (m1["scene_id"] == m2["scene_id"])
                pair = {"query_scene": m1["scene_id"], "ref_scene": m2["scene_id"],
                        "query_variant": v1["variant"], "ref_variant": v2["variant"],
                        "is_same_place": is_same}
                if is_same:
                    queries.append(pair)
                references.append(pair)
    return {"query_reference_pairs": queries, "all_pairs": references}
