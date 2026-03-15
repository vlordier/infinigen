# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Tests for infinigen.core.syndata — bpy-free curriculum-learning utilities.

Every test imports only from ``infinigen.core.syndata`` (which is pure
Python / NumPy) so that the suite runs in CI without Blender.
"""


import math

import numpy as np
import pytest

# ── Module imports (all bpy-free) ──────────────────────────────────────────
from infinigen.core.syndata.camera_config import (
    ASPECT_4_3,
    CameraRigConfig,
    DroneCamera,
)
from infinigen.core.syndata.complexity import CurriculumConfig, curriculum_overrides
from infinigen.core.syndata.density_scaling import DensityScaler
from infinigen.core.syndata.drone_env_bridge import (
    DroneEnvConfig,
    TrainingOutcome,
    apply_curriculum_adjustment,
    outcome_to_curriculum_params,
    scene_to_drone_entities,
    syndata_to_drone_env_config,
)
from infinigen.core.syndata.episode import EpisodeConfig
from infinigen.core.syndata.genesis_export import (
    GenesisCamera,
    GenesisEntityConfig,
    GenesisEpisodeConfig,
    GenesisLight,
    GenesisObservationConfig,
    GenesisSceneConfig,
    GenesisSceneManifest,
    build_genesis_config,
    camera_from_syndata,
    episode_to_genesis,
    manifest_to_entities,
    metadata_to_entities,
    observation_to_genesis,
    observation_to_render_kwargs,
    randomisation_to_genesis_lights,
    scene_manifest_from_dir,
    to_genesis_script,
)
from infinigen.core.syndata.metadata import BBox3D, DepthStats, FrameMetadata
from infinigen.core.syndata.metrics import SceneBudget
from infinigen.core.syndata.observation import (
    PASSES_MINIMAL,
    PASSES_NAVIGATION,
    ObservationConfig,
    SensorNoiseModel,
)
from infinigen.core.syndata.parallel_stages import Stage, StageGraph
from infinigen.core.syndata.pretraining import (
    FlappyColumnConfig,
    flappy_drone_env_config,
    flappy_genesis_entities,
    generate_flappy_obstacles,
)
from infinigen.core.syndata.quality_presets import (
    VALID_PRESETS,
    drone_preset,
    to_gin_bindings,
)
from infinigen.core.syndata.randomisation import DomainRandomiser
from infinigen.core.syndata.resolution import resolution_for_stage
from infinigen.core.syndata.validation import SceneValidator

# ═══════════════════════════════════════════════════════════════════════════
#  1. CurriculumConfig / complexity
# ═══════════════════════════════════════════════════════════════════════════


class TestCurriculumConfig:
    def test_stage_zero_is_easiest(self):
        cfg = CurriculumConfig(stage=0, total_stages=10)
        assert cfg.progress == 0.0
        assert cfg.subdiv_level == cfg.min_subdiv
        assert cfg.texture_resolution == cfg.min_texture_res
        assert cfg.object_count == cfg.min_objects

    def test_last_stage_is_hardest(self):
        cfg = CurriculumConfig(stage=9, total_stages=10)
        assert cfg.progress == 1.0
        assert cfg.subdiv_level == cfg.max_subdiv
        assert cfg.texture_resolution == cfg.max_texture_res
        assert cfg.object_count == cfg.max_objects

    def test_monotonic_difficulty(self):
        cfgs = [CurriculumConfig(stage=i, total_stages=10) for i in range(10)]
        progresses = [c.progress for c in cfgs]
        assert progresses == sorted(progresses)

    def test_single_stage(self):
        cfg = CurriculumConfig(stage=0, total_stages=1)
        assert cfg.progress == 0.0

    def test_invalid_stage_raises(self):
        with pytest.raises(ValueError, match="stage must be"):
            CurriculumConfig(stage=10, total_stages=10)
        with pytest.raises(ValueError, match="stage must be"):
            CurriculumConfig(stage=-1, total_stages=5)

    def test_invalid_total_stages(self):
        with pytest.raises(ValueError, match="total_stages must be"):
            CurriculumConfig(stage=0, total_stages=0)

    def test_texture_resolution_power_of_two(self):
        for i in range(10):
            cfg = CurriculumConfig(stage=i, total_stages=10)
            res = cfg.texture_resolution
            assert res > 0
            assert (res & (res - 1)) == 0, f"{res} is not a power of 2"

    def test_scatter_density_range(self):
        for i in range(10):
            cfg = CurriculumConfig(stage=i, total_stages=10)
            assert 0.0 < cfg.scatter_density <= 1.0

    def test_inverted_subdiv_raises(self):
        with pytest.raises(ValueError, match="min_subdiv"):
            CurriculumConfig(stage=0, total_stages=5, min_subdiv=4, max_subdiv=2)

    def test_inverted_texture_res_raises(self):
        with pytest.raises(ValueError, match="min_texture_res"):
            CurriculumConfig(stage=0, total_stages=5, min_texture_res=2048, max_texture_res=64)

    def test_inverted_objects_raises(self):
        with pytest.raises(ValueError, match="min_objects"):
            CurriculumConfig(stage=0, total_stages=5, min_objects=100, max_objects=10)

    def test_negative_exponent_raises(self):
        with pytest.raises(ValueError, match="exponent must be positive"):
            CurriculumConfig(stage=0, total_stages=5, exponent=-1.0)

    def test_custom_scatter_density_floor(self):
        cfg = CurriculumConfig(stage=0, total_stages=5, min_scatter_density=0.2)
        assert cfg.scatter_density >= 0.2


class TestCurriculumOverrides:
    def test_returns_dict(self):
        cfg = CurriculumConfig(stage=5, total_stages=10)
        overrides = curriculum_overrides(cfg)
        assert isinstance(overrides, dict)
        assert "grid_coarsen" in overrides
        assert "execute_tasks.generate_resolution" in overrides

    def test_resolution_tuple(self):
        cfg = CurriculumConfig(stage=0, total_stages=10)
        overrides = curriculum_overrides(cfg)
        res = overrides["execute_tasks.generate_resolution"]
        assert isinstance(res, tuple)
        assert len(res) == 2


# ═══════════════════════════════════════════════════════════════════════════
#  2. Quality presets
# ═══════════════════════════════════════════════════════════════════════════


class TestQualityPresets:
    @pytest.mark.parametrize("name", sorted(VALID_PRESETS))
    def test_valid_preset(self, name):
        overrides = drone_preset(name)
        assert isinstance(overrides, dict)
        assert "configure_render_cycles.num_samples" in overrides

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            drone_preset("ultra_mega")

    def test_resolution_override(self):
        overrides = drone_preset("fast", resolution_override=(320, 240))
        assert overrides["execute_tasks.generate_resolution"] == (320, 240)

    def test_preview_fastest(self):
        preview = drone_preset("preview")
        high = drone_preset("high")
        assert preview["configure_render_cycles.num_samples"] < high["configure_render_cycles.num_samples"]

    def test_exposure_present(self):
        for name in VALID_PRESETS:
            overrides = drone_preset(name)
            assert "configure_render_cycles.exposure" in overrides

    def test_resolution_override_too_small(self):
        with pytest.raises(ValueError, match="resolution too small"):
            drone_preset("fast", resolution_override=(16, 16))

    def test_resolution_override_too_large(self):
        with pytest.raises(ValueError, match="resolution too large"):
            drone_preset("fast", resolution_override=(16384, 16384))


# ═══════════════════════════════════════════════════════════════════════════
#  3. Scene budget / metrics
# ═══════════════════════════════════════════════════════════════════════════


class TestSceneBudget:
    def test_empty_scene_fits(self):
        b = SceneBudget()
        assert b.fits(max_vram_mb=4096, max_seconds=60)

    def test_huge_scene_exceeds_budget(self):
        b = SceneBudget(
            poly_count=50_000_000,
            texture_pixels=4096 * 4096 * 20,
            num_samples=1024,
            resolution=(4096, 4096),
        )
        assert not b.fits(max_vram_mb=1024, max_seconds=5)

    def test_vram_positive(self):
        b = SceneBudget(poly_count=100_000)
        assert b.estimated_vram_mb > 0

    def test_render_seconds_positive(self):
        b = SceneBudget(num_samples=128, resolution=(512, 512))
        assert b.estimated_render_seconds > 0

    def test_summary_keys(self):
        s = SceneBudget().summary()
        assert "poly_count" in s
        assert "estimated_vram_mb" in s
        assert "estimated_render_seconds" in s

    def test_batch_fits(self):
        b = SceneBudget(num_samples=16, resolution=(128, 128))
        assert b.batch_fits(100, max_total_seconds=3600)

    def test_batch_exceeds(self):
        b = SceneBudget(num_samples=512, resolution=(2048, 2048))
        assert not b.batch_fits(10000, max_total_seconds=1)

    def test_vram_includes_safety_factor(self):
        b = SceneBudget(poly_count=100_000)
        # With 1.5× safety factor, should be higher than raw calculation
        raw_geo = 100_000 * 120 / (1024 * 1024)
        assert b.estimated_vram_mb > raw_geo


# ═══════════════════════════════════════════════════════════════════════════
#  4. Resolution ladder
# ═══════════════════════════════════════════════════════════════════════════


class TestResolution:
    def test_first_stage_min_res(self):
        w, h = resolution_for_stage(0, 10, min_res=64, max_res=2048)
        assert w == 64
        assert h == 64

    def test_last_stage_max_res(self):
        w, h = resolution_for_stage(9, 10, min_res=64, max_res=2048)
        assert w == 2048
        assert h == 2048

    def test_aspect_ratio_landscape(self):
        w, h = resolution_for_stage(5, 10, aspect_ratio=2.0)
        assert w > h

    def test_aspect_ratio_portrait(self):
        w, h = resolution_for_stage(5, 10, aspect_ratio=0.5)
        assert w < h

    def test_monotonic_resolution(self):
        resolutions = [resolution_for_stage(i, 10)[0] for i in range(10)]
        assert resolutions == sorted(resolutions)

    def test_invalid_stage(self):
        with pytest.raises(ValueError):
            resolution_for_stage(-1, 10)
        with pytest.raises(ValueError):
            resolution_for_stage(10, 10)

    def test_power_of_two(self):
        for i in range(10):
            w, h = resolution_for_stage(i, 10)
            assert (w & (w - 1)) == 0, f"width {w} not power of 2"
            assert (h & (h - 1)) == 0, f"height {h} not power of 2"

    def test_min_greater_than_max_raises(self):
        with pytest.raises(ValueError, match="min_res"):
            resolution_for_stage(0, 5, min_res=2048, max_res=64)

    def test_extreme_aspect_ratio_raises(self):
        with pytest.raises(ValueError, match="aspect_ratio"):
            resolution_for_stage(0, 5, aspect_ratio=10.0)
        with pytest.raises(ValueError, match="aspect_ratio"):
            resolution_for_stage(0, 5, aspect_ratio=0.1)


# ═══════════════════════════════════════════════════════════════════════════
#  5. Domain randomisation
# ═══════════════════════════════════════════════════════════════════════════


class TestDomainRandomiser:
    def test_easy_tight_ranges(self):
        r = DomainRandomiser(difficulty=0.0)
        ranges = r.ranges()
        for name, (lo, hi) in ranges.items():
            assert lo <= hi, f"{name}: lo={lo} > hi={hi}"

    def test_hard_wider_ranges(self):
        easy = DomainRandomiser(difficulty=0.0).ranges()
        hard = DomainRandomiser(difficulty=1.0).ranges()
        for name in easy:
            easy_spread = easy[name][1] - easy[name][0]
            hard_spread = hard[name][1] - hard[name][0]
            assert hard_spread >= easy_spread, f"{name} did not widen"

    def test_invalid_difficulty(self):
        with pytest.raises(ValueError, match="difficulty"):
            DomainRandomiser(difficulty=-0.1)
        with pytest.raises(ValueError, match="difficulty"):
            DomainRandomiser(difficulty=1.1)

    def test_gin_overrides(self):
        r = DomainRandomiser(difficulty=0.5)
        overrides = r.gin_overrides()
        assert "camera.rotation_jitter" in overrides
        # All ranges should be exported
        assert "lighting.sun_elevation_range" in overrides
        assert "weather.cloud_density" in overrides
        assert "material.roughness_variance" in overrides
        assert "configure_render_cycles.exposure" in overrides

    def test_from_curriculum_progress(self):
        r = DomainRandomiser.from_curriculum_progress(0.0)
        assert r.difficulty == 0.0
        r = DomainRandomiser.from_curriculum_progress(1.0)
        assert r.difficulty == 1.0
        # Mid-point uses sqrt curve
        r = DomainRandomiser.from_curriculum_progress(0.25)
        assert abs(r.difficulty - 0.5) < 0.01


# ═══════════════════════════════════════════════════════════════════════════
#  6. Density scaling
# ═══════════════════════════════════════════════════════════════════════════


class TestDensityScaler:
    def test_zero_difficulty(self):
        ds = DensityScaler(difficulty=0.0)
        assert ds.scatter_multiplier == pytest.approx(0.1)
        assert ds.obstacle_count == 2

    def test_full_difficulty(self):
        ds = DensityScaler(difficulty=1.0)
        assert ds.scatter_multiplier == pytest.approx(1.0)
        assert ds.obstacle_count == 50

    def test_quadratic_curve(self):
        ds_lin = DensityScaler(difficulty=0.5, curve="linear")
        ds_quad = DensityScaler(difficulty=0.5, curve="quadratic")
        assert ds_quad.scatter_multiplier < ds_lin.scatter_multiplier

    def test_sqrt_curve(self):
        ds_lin = DensityScaler(difficulty=0.5, curve="linear")
        ds_sqrt = DensityScaler(difficulty=0.5, curve="sqrt")
        assert ds_sqrt.scatter_multiplier > ds_lin.scatter_multiplier

    def test_invalid_curve(self):
        with pytest.raises(ValueError, match="curve"):
            DensityScaler(curve="cubic")

    def test_gin_overrides(self):
        overrides = DensityScaler(difficulty=0.5).gin_overrides()
        assert "scatter_density_multiplier" in overrides
        assert "obstacle_count" in overrides

    def test_inverted_multiplier_raises(self):
        with pytest.raises(ValueError, match="min_multiplier"):
            DensityScaler(min_multiplier=1.0, max_multiplier=0.1)

    def test_negative_multiplier_raises(self):
        with pytest.raises(ValueError, match="min_multiplier must be non-negative"):
            DensityScaler(min_multiplier=-0.5)

    def test_inverted_obstacle_raises(self):
        with pytest.raises(ValueError, match="obstacle_min"):
            DensityScaler(obstacle_min=100, obstacle_max=5)


# ═══════════════════════════════════════════════════════════════════════════
#  7. Parallel stages
# ═══════════════════════════════════════════════════════════════════════════


class TestStageGraph:
    def test_default_waves(self):
        g = StageGraph()
        waves = g.parallel_groups()
        assert len(waves) >= 2
        assert "coarse" in waves[0]

    def test_topological_order(self):
        g = StageGraph()
        order = g.topological_order()
        idx = {name: i for i, name in enumerate(order)}
        for s in g.stages:
            for dep in s.depends_on:
                assert idx[dep] < idx[s.name], f"{dep} should come before {s.name}"

    def test_gpu_stages(self):
        g = StageGraph()
        gpu = g.gpu_stages()
        assert "render" in gpu
        cpu = g.cpu_only_stages()
        assert "coarse" in cpu

    def test_custom_graph(self):
        stages = (
            Stage(name="a"),
            Stage(name="b", depends_on=frozenset({"a"})),
            Stage(name="c", depends_on=frozenset({"a"})),
            Stage(name="d", depends_on=frozenset({"b", "c"})),
        )
        g = StageGraph(stages=stages)
        waves = g.parallel_groups()
        assert waves == [["a"], ["b", "c"], ["d"]]

    def test_cycle_detection(self):
        stages = (
            Stage(name="a", depends_on=frozenset({"b"})),
            Stage(name="b", depends_on=frozenset({"a"})),
        )
        g = StageGraph(stages=stages)
        with pytest.raises(RuntimeError, match="Cycle"):
            g.parallel_groups()


# ═══════════════════════════════════════════════════════════════════════════
#  8. Metadata
# ═══════════════════════════════════════════════════════════════════════════


class TestMetadata:
    def test_round_trip_json(self, tmp_path):
        meta = FrameMetadata(
            frame_id=42,
            scene_seed=12345,
            camera_position=(1.0, 2.0, 3.0),
            camera_rotation_euler=(0.1, 0.2, 0.3),
            obstacles=[BBox3D(center=(0, 0, 0), extent=(1, 1, 1), label="tree")],
            depth_stats=DepthStats(min_m=0.5, max_m=80.0, mean_m=20.0, median_m=18.0, std_m=15.0),
            traversability_ratio=0.6,
            curriculum_stage=3,
        )
        path = tmp_path / "frame_42.json"
        meta.save_json(path)
        loaded = FrameMetadata.load_json(path)
        assert loaded.frame_id == 42
        assert loaded.scene_seed == 12345
        assert len(loaded.obstacles) == 1
        assert loaded.obstacles[0].label == "tree"
        assert loaded.depth_stats is not None
        assert loaded.depth_stats.min_m == pytest.approx(0.5)

    def test_to_dict(self):
        meta = FrameMetadata(frame_id=1)
        d = meta.to_dict()
        assert isinstance(d, dict)
        assert d["frame_id"] == 1

    def test_depth_stats_from_array(self):
        depth = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = DepthStats.from_depth_array(depth)
        assert stats.min_m == pytest.approx(1.0)
        assert stats.max_m == pytest.approx(5.0)
        assert stats.mean_m == pytest.approx(3.0)

    def test_depth_stats_with_inf(self):
        depth = np.array([1.0, np.inf, 3.0, -np.inf])
        stats = DepthStats.from_depth_array(depth)
        assert stats.min_m == pytest.approx(1.0)
        assert stats.max_m == pytest.approx(3.0)

    def test_depth_stats_all_invalid(self):
        depth = np.array([np.inf, np.nan])
        stats = DepthStats.from_depth_array(depth)
        # Should return defaults without crashing
        assert stats.min_m == 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  9. Validation
# ═══════════════════════════════════════════════════════════════════════════


class TestSceneValidator:
    def test_valid_scene(self):
        v = SceneValidator(min_obstacles=1, max_obstacles=10)
        meta = {
            "obstacles": [{"center": (0, 0, 0), "extent": (1, 1, 1)}] * 5,
            "depth_stats": {"min_m": 0.5, "max_m": 50.0},
            "traversability_ratio": 0.4,
            "poly_count": 50_000,
        }
        assert v.is_valid(meta)

    def test_too_few_obstacles(self):
        v = SceneValidator(min_obstacles=3)
        meta = {"obstacles": []}
        results = v.validate(meta)
        obstacle_check = next(r for r in results if r.name == "obstacle_count")
        assert not obstacle_check.passed

    def test_too_many_obstacles(self):
        v = SceneValidator(max_obstacles=5)
        meta = {"obstacles": [{}] * 10}
        assert not v.is_valid(meta)

    def test_depth_too_shallow(self):
        v = SceneValidator(min_depth_range_m=10.0)
        meta = {"depth_stats": {"min_m": 1.0, "max_m": 5.0}}
        results = v.validate(meta)
        depth_check = next(r for r in results if r.name == "depth_range")
        assert not depth_check.passed

    def test_traversability_bounds(self):
        v = SceneValidator(min_traversability=0.1, max_traversability=0.9)
        assert not v.is_valid({"traversability_ratio": 0.05})
        assert not v.is_valid({"traversability_ratio": 0.95})
        assert v.is_valid({"traversability_ratio": 0.5})

    def test_poly_count_bounds(self):
        v = SceneValidator(min_poly_count=1000, max_poly_count=100_000)
        assert v.is_valid({"poly_count": 50_000})
        assert not v.is_valid({"poly_count": 500})
        assert not v.is_valid({"poly_count": 200_000})

    def test_custom_check(self):
        def check_has_sky(meta):
            return meta.get("has_sky", False), "sky check"

        v = SceneValidator(custom_checks=[("sky", check_has_sky)])
        assert not v.is_valid({"has_sky": False})
        assert v.is_valid({"has_sky": True})

    def test_missing_keys_skip(self):
        """Missing metadata keys should not cause failures — they're just skipped."""
        v = SceneValidator()
        results = v.validate({})
        assert len(results) == 0
        assert v.is_valid({})


# ═══════════════════════════════════════════════════════════════════════════
#  10. Integration: end-to-end curriculum pipeline
# ═══════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """Ensure the modules compose correctly for a full curriculum run."""

    def test_full_curriculum_pipeline(self):
        """Walk through all 10 stages and verify monotonic scaling."""
        total = 10
        prev_polys = 0
        prev_res = 0

        for i in range(total):
            # Build config
            cfg = CurriculumConfig(stage=i, total_stages=total)

            # Quality preset based on progress
            if cfg.progress < 0.25:
                preset_name = "preview"
            elif cfg.progress < 0.5:
                preset_name = "fast"
            elif cfg.progress < 0.75:
                preset_name = "medium"
            else:
                preset_name = "high"
            preset = drone_preset(preset_name)

            # Resolution
            res = resolution_for_stage(i, total)
            assert res[0] >= prev_res
            prev_res = res[0]

            # Density
            ds = DensityScaler(difficulty=cfg.progress)
            assert ds.scatter_multiplier >= 0

            # Randomisation
            rand = DomainRandomiser.from_curriculum_progress(cfg.progress)
            assert 0.0 <= rand.difficulty <= 1.0

            # Budget check
            budget = SceneBudget(
                poly_count=cfg.object_count * 1000,
                num_samples=preset["configure_render_cycles.num_samples"],
                resolution=res,
            )
            # Early stages must fit tight budgets
            if i < 3:
                assert budget.fits(max_vram_mb=2048, max_seconds=30)

    def test_stage_graph_covers_all_tasks(self):
        g = StageGraph()
        names = {s.name for s in g.stages}
        assert "coarse" in names
        assert "render" in names
        assert "ground_truth" in names

    def test_metadata_validation_roundtrip(self, tmp_path):
        """Generate metadata, validate it, then round-trip through JSON."""
        meta = FrameMetadata(
            frame_id=0,
            scene_seed=999,
            obstacles=[BBox3D(label="drone_target")],
            depth_stats=DepthStats.from_depth_array(np.random.uniform(1, 100, size=(64, 64))),
            traversability_ratio=0.5,
            curriculum_stage=5,
        )

        # Validate
        v = SceneValidator()
        assert v.is_valid(meta.to_dict())

        # Round-trip
        path = tmp_path / "meta.json"
        meta.save_json(path)
        loaded = FrameMetadata.load_json(path)
        assert loaded.curriculum_stage == 5
        assert loaded.obstacles[0].label == "drone_target"


# ═══════════════════════════════════════════════════════════════════════════
#  11. Camera intrinsics sync (pipeline-critical fix)
# ═══════════════════════════════════════════════════════════════════════════


class TestCameraIntrinsicsSync:
    """Verify that quality presets always sync get_sensor_coords.H/W
    with execute_tasks.generate_resolution — without this, depth maps
    and 3D projections are silently wrong."""

    @pytest.mark.parametrize("name", sorted(VALID_PRESETS))
    def test_preset_syncs_intrinsics(self, name):
        overrides = drone_preset(name)
        w, h = overrides["execute_tasks.generate_resolution"]
        assert overrides["get_sensor_coords.H"] == h
        assert overrides["get_sensor_coords.W"] == w

    def test_resolution_override_syncs_intrinsics(self):
        overrides = drone_preset("fast", resolution_override=(640, 480))
        assert overrides["get_sensor_coords.H"] == 480
        assert overrides["get_sensor_coords.W"] == 640

    def test_motion_blur_key_uses_correct_gin_name(self):
        overrides = drone_preset("medium")
        assert "configure_blender.motion_blur" in overrides


# ═══════════════════════════════════════════════════════════════════════════
#  12. Gin binding generation
# ═══════════════════════════════════════════════════════════════════════════


class TestGinBindings:
    def test_basic_types(self):
        lines = to_gin_bindings({"foo.bar": 42, "baz.qux": True})
        assert "baz.qux = True" in lines
        assert "foo.bar = 42" in lines

    def test_tuple_formatting(self):
        lines = to_gin_bindings({"res": (256, 256)})
        assert "res = (256, 256)" in lines

    def test_string_quoting(self):
        lines = to_gin_bindings({"mode": "fast"})
        assert "mode = 'fast'" in lines

    def test_false_boolean(self):
        lines = to_gin_bindings({"flag": False})
        assert "flag = False" in lines

    def test_sorted_output(self):
        lines = to_gin_bindings({"z": 1, "a": 2, "m": 3})
        keys = [l.split(" = ")[0] for l in lines]
        assert keys == sorted(keys)

    def test_roundtrip_preset_to_gin(self):
        """All preset values should produce valid gin strings."""
        for name in VALID_PRESETS:
            overrides = drone_preset(name)
            lines = to_gin_bindings(overrides)
            assert len(lines) > 0
            for line in lines:
                assert " = " in line


# ═══════════════════════════════════════════════════════════════════════════
#  13. Observation space
# ═══════════════════════════════════════════════════════════════════════════


class TestObservation:
    def test_default_channels(self):
        obs = ObservationConfig()
        names = obs.channel_names
        assert "R" in names  # RGB included by default
        assert obs.num_channels >= 4  # R, G, B + at least depth

    def test_no_rgb(self):
        obs = ObservationConfig(include_rgb=False)
        assert "R" not in obs.channel_names
        assert obs.num_channels >= 1  # at least the passes

    def test_minimal_passes(self):
        obs = ObservationConfig(passes=PASSES_MINIMAL)
        overrides = obs.gin_overrides()
        # Should export flat passes
        assert isinstance(overrides, dict)

    def test_navigation_passes(self):
        obs = ObservationConfig(passes=PASSES_NAVIGATION)
        names = obs.channel_names
        assert "z" in names
        assert "normal" in names
        assert "object_index" in names

    def test_invalid_depth_clip(self):
        with pytest.raises(ValueError, match="depth_clip_m"):
            ObservationConfig(depth_clip_m=-10)


class TestSensorNoise:
    def test_default_is_clean(self):
        noise = SensorNoiseModel()
        assert noise.gaussian_std == 0.0
        assert noise.salt_pepper_prob == 0.0

    def test_drone_default_scales_with_difficulty(self):
        easy = SensorNoiseModel.drone_default(0.0)
        hard = SensorNoiseModel.drone_default(1.0)
        assert easy.gaussian_std < hard.gaussian_std
        assert easy.motion_blur_px < hard.motion_blur_px

    def test_invalid_noise_params(self):
        with pytest.raises(ValueError, match="gaussian_std"):
            SensorNoiseModel(gaussian_std=-1)
        with pytest.raises(ValueError, match="salt_pepper_prob"):
            SensorNoiseModel(salt_pepper_prob=2.0)
        with pytest.raises(ValueError, match="motion_blur_px"):
            SensorNoiseModel(motion_blur_px=-5)


# ═══════════════════════════════════════════════════════════════════════════
#  14. Camera configuration
# ═══════════════════════════════════════════════════════════════════════════


class TestDroneCamera:
    def test_focal_length_90fov(self):
        cam = DroneCamera(fov_deg=90.0, aspect_ratio=ASPECT_4_3)
        # focal length should be positive and finite
        assert 0 < cam.focal_length_mm < 100

    def test_invalid_fov(self):
        with pytest.raises(ValueError, match="fov_deg"):
            DroneCamera(fov_deg=5)  # too narrow
        with pytest.raises(ValueError, match="fov_deg"):
            DroneCamera(fov_deg=200)  # too wide

    def test_wide_angle(self):
        narrow = DroneCamera(fov_deg=60)
        wide = DroneCamera(fov_deg=150)
        assert wide.focal_length_mm < narrow.focal_length_mm


class TestCameraRig:
    def test_monocular(self):
        rig = CameraRigConfig.monocular(n_drones=3)
        assert rig.n_rigs == 3
        assert len(rig.effective_cameras) == 1

    def test_stereo(self):
        rig = CameraRigConfig.stereo(baseline_m=0.065)
        assert len(rig.effective_cameras) == 2
        # Second camera should be offset by baseline
        left = rig.effective_cameras[0]["loc"]
        right = rig.effective_cameras[1]["loc"]
        assert right[0] - left[0] == pytest.approx(0.065)

    def test_gin_overrides(self):
        rig = CameraRigConfig.stereo(baseline_m=0.1, n_drones=2)
        overrides = rig.gin_overrides()
        assert overrides["camera.spawn_camera_rigs.n_camera_rigs"] == 2
        assert len(overrides["camera.spawn_camera_rigs.camera_rig_config"]) == 2

    def test_invalid_n_rigs(self):
        with pytest.raises(ValueError, match="n_rigs"):
            CameraRigConfig(n_rigs=0)

    def test_no_stereo_baseline(self):
        rig = CameraRigConfig.monocular()
        assert len(rig.effective_cameras) == 1


# ═══════════════════════════════════════════════════════════════════════════
#  15. Episode configuration
# ═══════════════════════════════════════════════════════════════════════════


class TestEpisode:
    def test_single_frame(self):
        ep = EpisodeConfig.single_frame()
        assert ep.num_frames == 1
        assert ep.frame_range == (1, 1)
        assert ep.duration_seconds == pytest.approx(1 / 24)

    def test_short_trajectory(self):
        ep = EpisodeConfig.short_trajectory(num_frames=30, fps=10)
        assert ep.frame_range == (1, 30)
        assert ep.duration_seconds == pytest.approx(3.0)

    def test_navigation_episode(self):
        ep = EpisodeConfig.navigation_episode(num_frames=120)
        assert ep.trajectory == "rrt"
        assert ep.end_frame == 120

    def test_gin_overrides(self):
        ep = EpisodeConfig(num_frames=60, fps=30)
        overrides = ep.gin_overrides()
        assert overrides["execute_tasks.frame_range"] == [1, 60]
        assert overrides["execute_tasks.fps"] == 30

    def test_invalid_frames(self):
        with pytest.raises(ValueError, match="num_frames"):
            EpisodeConfig(num_frames=0)

    def test_invalid_fps(self):
        with pytest.raises(ValueError, match="fps"):
            EpisodeConfig(fps=0)
        with pytest.raises(ValueError, match="fps"):
            EpisodeConfig(fps=200)

    def test_invalid_trajectory(self):
        with pytest.raises(ValueError, match="trajectory"):
            EpisodeConfig(trajectory="teleport")


# ═══════════════════════════════════════════════════════════════════════════
#  16. Enriched metadata (velocity, obstacles, swarm)
# ═══════════════════════════════════════════════════════════════════════════


class TestEnrichedMetadata:
    def test_velocity_default(self):
        meta = FrameMetadata()
        assert meta.velocity == (0.0, 0.0, 0.0)

    def test_nearest_obstacle_default_inf(self):
        meta = FrameMetadata()
        assert math.isinf(meta.nearest_obstacle_m)

    def test_swarm_positions(self):
        meta = FrameMetadata(
            swarm_positions=[(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
        )
        assert len(meta.swarm_positions) == 2

    def test_json_roundtrip_with_new_fields(self, tmp_path):
        meta = FrameMetadata(
            frame_id=7,
            velocity=(1.5, -0.3, 0.0),
            nearest_obstacle_m=2.5,
            swarm_positions=[(10, 20, 30), (40, 50, 60)],
        )
        path = tmp_path / "enriched.json"
        meta.save_json(path)
        loaded = FrameMetadata.load_json(path)
        assert loaded.velocity == (1.5, -0.3, 0.0)
        assert loaded.nearest_obstacle_m == pytest.approx(2.5)
        assert len(loaded.swarm_positions) == 2
        assert loaded.swarm_positions[0] == (10, 20, 30)

    def test_json_roundtrip_infinity_obstacle(self, tmp_path):
        """nearest_obstacle_m=inf must survive JSON serialisation."""
        meta = FrameMetadata(nearest_obstacle_m=float("inf"))
        path = tmp_path / "inf_test.json"
        meta.save_json(path)
        loaded = FrameMetadata.load_json(path)
        assert math.isinf(loaded.nearest_obstacle_m)


# ═══════════════════════════════════════════════════════════════════════════
#  17. Full UAV swarm integration
# ═══════════════════════════════════════════════════════════════════════════


class TestUAVSwarmIntegration:
    """End-to-end test: configure a full UAV swarm training pipeline."""

    def test_full_swarm_config(self):
        """Build a complete config for a 4-drone swarm curriculum."""
        total_stages = 8
        for stage_idx in range(total_stages):
            cfg = CurriculumConfig(stage=stage_idx, total_stages=total_stages)

            # Preset selection based on difficulty
            preset_name = "preview" if cfg.progress < 0.5 else "fast"
            preset = drone_preset(preset_name)

            # Camera: stereo rig for 4 drones
            rig = CameraRigConfig.stereo(baseline_m=0.065, n_drones=4)

            # Episode: longer episodes at higher difficulty
            ep_frames = max(1, int(30 * (1 + cfg.progress)))
            episode = EpisodeConfig(num_frames=ep_frames, fps=10, trajectory="random_walk")

            # Observation: add more passes as training progresses
            if cfg.progress < 0.3:
                obs = ObservationConfig(passes=PASSES_MINIMAL)
            else:
                obs = ObservationConfig(passes=PASSES_NAVIGATION)

            # Sensor noise scales with difficulty
            noise = SensorNoiseModel.drone_default(cfg.progress)

            # All overrides should be valid
            assert preset["get_sensor_coords.H"] == preset["execute_tasks.generate_resolution"][1]
            assert rig.n_rigs == 4
            assert episode.num_frames >= 1
            assert obs.num_channels >= 1
            assert noise.gaussian_std >= 0

    def test_gin_bindings_composable(self):
        """All gin_overrides methods should produce to_gin_bindings-able dicts."""
        preset = drone_preset("fast")
        rig = CameraRigConfig.monocular()
        episode = EpisodeConfig.short_trajectory()

        # Merge all overrides
        all_overrides: dict = {}
        all_overrides.update(preset)
        all_overrides.update(rig.gin_overrides())
        all_overrides.update(episode.gin_overrides())

        # Should produce valid gin bindings
        lines = to_gin_bindings(all_overrides)
        assert len(lines) > 5
        for line in lines:
            assert " = " in line


# ═══════════════════════════════════════════════════════════════════════════
#  18. Fixes for review comments
# ═══════════════════════════════════════════════════════════════════════════


class TestDomainRandomiserSample:
    """Tests for DomainRandomiser.sample() and seed reproducibility."""

    def test_sample_returns_all_params(self):
        r = DomainRandomiser(difficulty=0.5, seed=42)
        s = r.sample()
        assert len(s) == 10
        for name in r.ranges():
            assert name in s

    def test_sample_within_ranges(self):
        r = DomainRandomiser(difficulty=0.5, seed=42)
        ranges = r.ranges()
        s = r.sample()
        for name, val in s.items():
            lo, hi = ranges[name]
            assert lo <= val <= hi, f"{name}: {val} not in [{lo}, {hi}]"

    def test_sample_reproducible_with_seed(self):
        s1 = DomainRandomiser(difficulty=0.5, seed=42).sample()
        s2 = DomainRandomiser(difficulty=0.5, seed=42).sample()
        assert s1 == s2

    def test_sample_different_seeds_differ(self):
        s1 = DomainRandomiser(difficulty=0.5, seed=42).sample()
        s2 = DomainRandomiser(difficulty=0.5, seed=99).sample()
        assert s1 != s2

    def test_sample_none_seed_works(self):
        r = DomainRandomiser(difficulty=0.5, seed=None)
        s = r.sample()
        assert len(s) == 10


class TestSceneBudgetValidation:
    """Tests for SceneBudget.__post_init__ input validation."""

    def test_negative_poly_count_raises(self):
        with pytest.raises(ValueError, match="poly_count"):
            SceneBudget(poly_count=-1)

    def test_negative_texture_pixels_raises(self):
        with pytest.raises(ValueError, match="texture_pixels"):
            SceneBudget(texture_pixels=-100)

    def test_zero_num_lights_raises(self):
        with pytest.raises(ValueError, match="num_lights"):
            SceneBudget(num_lights=0)

    def test_zero_num_samples_raises(self):
        with pytest.raises(ValueError, match="num_samples"):
            SceneBudget(num_samples=0)

    def test_zero_resolution_raises(self):
        with pytest.raises(ValueError, match="resolution"):
            SceneBudget(resolution=(0, 256))
        with pytest.raises(ValueError, match="resolution"):
            SceneBudget(resolution=(256, -1))


class TestValidationDepthStatsKeys:
    """Depth check should require both min_m and max_m."""

    def test_missing_min_m_fails(self):
        v = SceneValidator(min_depth_range_m=1.0)
        results = v.validate({"depth_stats": {"max_m": 50.0}})
        depth_check = next(r for r in results if r.name == "depth_range")
        assert not depth_check.passed
        assert "min_m" in depth_check.message

    def test_missing_max_m_fails(self):
        v = SceneValidator(min_depth_range_m=1.0)
        results = v.validate({"depth_stats": {"min_m": 0.5}})
        depth_check = next(r for r in results if r.name == "depth_range")
        assert not depth_check.passed
        assert "max_m" in depth_check.message

    def test_both_present_ok(self):
        v = SceneValidator(min_depth_range_m=1.0)
        results = v.validate({"depth_stats": {"min_m": 0.5, "max_m": 50.0}})
        depth_check = next(r for r in results if r.name == "depth_range")
        assert depth_check.passed

    def test_empty_depth_stats_fails(self):
        v = SceneValidator(min_depth_range_m=1.0)
        results = v.validate({"depth_stats": {}})
        depth_check = next(r for r in results if r.name == "depth_range")
        assert not depth_check.passed
        assert "min_m" in depth_check.message
        assert "max_m" in depth_check.message


class TestBBox3DTupleRoundTrip:
    """BBox3D center/extent must come back as tuples after JSON round-trip."""

    def test_bbox_tuples_preserved(self, tmp_path):
        meta = FrameMetadata(
            frame_id=1,
            obstacles=[BBox3D(center=(1.0, 2.0, 3.0), extent=(4.0, 5.0, 6.0), label="wall")],
        )
        path = tmp_path / "bbox_test.json"
        meta.save_json(path)
        loaded = FrameMetadata.load_json(path)
        obs = loaded.obstacles[0]
        assert isinstance(obs.center, tuple), f"center is {type(obs.center)}"
        assert isinstance(obs.extent, tuple), f"extent is {type(obs.extent)}"
        assert obs.center == (1.0, 2.0, 3.0)
        assert obs.extent == (4.0, 5.0, 6.0)
        assert obs.label == "wall"


class TestTextureResolutionClamping:
    """texture_resolution must never exceed max_texture_res."""

    def test_non_po2_max_clamped(self):
        cfg = CurriculumConfig(
            stage=9, total_stages=10,
            min_texture_res=64, max_texture_res=3000,
        )
        assert cfg.texture_resolution <= 3000

    def test_non_po2_min_respected(self):
        cfg = CurriculumConfig(
            stage=0, total_stages=10,
            min_texture_res=100, max_texture_res=2048,
        )
        assert cfg.texture_resolution >= 100


class TestScatterDensityValidation:
    """min_scatter_density must be in (0, 1]."""

    def test_negative_scatter_density_raises(self):
        with pytest.raises(ValueError, match="min_scatter_density"):
            CurriculumConfig(stage=0, total_stages=5, min_scatter_density=-0.1)

    def test_zero_scatter_density_raises(self):
        with pytest.raises(ValueError, match="min_scatter_density"):
            CurriculumConfig(stage=0, total_stages=5, min_scatter_density=0.0)

    def test_scatter_density_over_one_raises(self):
        with pytest.raises(ValueError, match="min_scatter_density"):
            CurriculumConfig(stage=0, total_stages=5, min_scatter_density=1.5)


class TestResolutionClamping:
    """resolution_for_stage must respect [min_res, max_res]."""

    def test_non_po2_max_clamped(self):
        w, h = resolution_for_stage(9, 10, min_res=64, max_res=1500)
        assert min(w, h) <= 1500

    def test_non_po2_min_respected(self):
        w, h = resolution_for_stage(0, 10, min_res=100, max_res=2048)
        assert w >= 100
        assert h >= 100


class TestStageGraphValidation:
    """StageGraph should distinguish missing deps from cycles."""

    def test_duplicate_stage_names_raises(self):
        stages = (
            Stage(name="a"),
            Stage(name="a"),
        )
        g = StageGraph(stages=stages)
        with pytest.raises(ValueError, match="Duplicate"):
            g.parallel_groups()

    def test_unknown_dependency_raises(self):
        stages = (
            Stage(name="a", depends_on=frozenset({"nonexistent"})),
        )
        g = StageGraph(stages=stages)
        with pytest.raises(ValueError, match="unknown"):
            g.parallel_groups()

    def test_cycle_still_detected(self):
        stages = (
            Stage(name="a", depends_on=frozenset({"b"})),
            Stage(name="b", depends_on=frozenset({"a"})),
        )
        g = StageGraph(stages=stages)
        with pytest.raises(RuntimeError, match="Cycle"):
            g.parallel_groups()

    def test_docstring_example_matches(self):
        g = StageGraph()
        waves = g.parallel_groups()
        assert waves == [
            ["coarse"],
            ["fine_terrain", "populate"],
            ["export", "mesh_save", "render"],
            ["ground_truth"],
        ]


# ═══════════════════════════════════════════════════════════════════════════
#  19. Genesis World integration
# ═══════════════════════════════════════════════════════════════════════════


class TestGenesisEntityConfig:
    def test_auto_detect_obj(self):
        ent = GenesisEntityConfig.from_file("/scene/mesh.obj")
        assert ent.morph_type == "Mesh"
        assert ent.name == "mesh"

    def test_auto_detect_urdf(self):
        ent = GenesisEntityConfig.from_file("/scene/robot.urdf")
        assert ent.morph_type == "URDF"

    def test_auto_detect_mjcf(self):
        ent = GenesisEntityConfig.from_file("/scene/model.xml")
        assert ent.morph_type == "MJCF"

    def test_auto_detect_stl(self):
        ent = GenesisEntityConfig.from_file("/scene/part.stl")
        assert ent.morph_type == "Mesh"

    def test_auto_detect_ply(self):
        ent = GenesisEntityConfig.from_file("/scene/cloud.ply")
        assert ent.morph_type == "Mesh"

    def test_auto_detect_gltf(self):
        ent = GenesisEntityConfig.from_file("/scene/model.glb")
        assert ent.morph_type == "Mesh"

    def test_auto_detect_usd(self):
        ent = GenesisEntityConfig.from_file("/scene/world.usda")
        assert ent.morph_type == "USD"

    def test_unknown_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            GenesisEntityConfig.from_file("/scene/data.npz")

    def test_invalid_morph_type_raises(self):
        with pytest.raises(ValueError, match="morph_type"):
            GenesisEntityConfig(morph_type="InvalidType")

    def test_box_entity(self):
        ent = GenesisEntityConfig(
            name="box1", morph_type="Box", pos=(1, 2, 3),
            extra={"size": (2, 2, 2)},
        )
        assert ent.morph_type == "Box"
        assert ent.extra["size"] == (2, 2, 2)

    def test_plane_entity(self):
        ent = GenesisEntityConfig(morph_type="Plane", is_fixed=True)
        assert ent.is_fixed

    def test_to_dict_round_trip(self):
        ent = GenesisEntityConfig(
            name="test", file_path="/a.obj", morph_type="Mesh",
            pos=(1, 2, 3), surface="Rough",
        )
        d = ent.to_dict()
        restored = GenesisEntityConfig(**d)
        assert restored.name == "test"
        assert restored.pos == (1, 2, 3)

    def test_custom_name(self):
        ent = GenesisEntityConfig.from_file("/scene/mesh.obj", name="custom")
        assert ent.name == "custom"


class TestGenesisCamera:
    def test_default(self):
        cam = GenesisCamera()
        assert cam.res == (640, 480)
        assert cam.fov == 60.0

    def test_invalid_resolution_raises(self):
        with pytest.raises(ValueError, match="Resolution"):
            GenesisCamera(res=(0, 480))

    def test_invalid_fov_raises(self):
        with pytest.raises(ValueError, match="fov"):
            GenesisCamera(fov=0.5)

    def test_invalid_clip_raises(self):
        with pytest.raises(ValueError, match="clip"):
            GenesisCamera(near=-1.0)

    def test_to_dict(self):
        cam = GenesisCamera(name="test", res=(320, 240), fov=90.0)
        d = cam.to_dict()
        assert d["name"] == "test"
        assert d["res"] == (320, 240)


class TestGenesisLight:
    def test_default(self):
        light = GenesisLight()
        assert light.pos == (0.0, 0.0, 10.0)
        assert light.radius == 3.0

    def test_to_dict(self):
        light = GenesisLight(pos=(5, 5, 20), color=(15, 15, 15))
        d = light.to_dict()
        assert d["pos"] == (5, 5, 20)
        assert d["color"] == (15, 15, 15)
        assert "radius" in d


class TestGenesisSceneConfig:
    def test_default(self):
        cfg = GenesisSceneConfig()
        assert cfg.renderer == "RayTracer"
        assert cfg.backend == "cuda"

    def test_invalid_renderer_raises(self):
        with pytest.raises(ValueError, match="renderer"):
            GenesisSceneConfig(renderer="OpenGL")

    def test_invalid_dt_raises(self):
        with pytest.raises(ValueError, match="dt"):
            GenesisSceneConfig(dt=0)

    def test_json_round_trip(self, tmp_path):
        cfg = GenesisSceneConfig(
            entities=[
                GenesisEntityConfig(name="box", morph_type="Box", pos=(1, 2, 3)),
            ],
            cameras=[GenesisCamera(name="cam", res=(320, 240))],
            lights=[GenesisLight(pos=(0, 0, 5))],
            renderer="Rasterizer",
            backend="cpu",
        )
        path = tmp_path / "scene.json"
        cfg.save_json(path)
        loaded = GenesisSceneConfig.load_json(path)
        assert loaded.renderer == "Rasterizer"
        assert loaded.backend == "cpu"
        assert len(loaded.entities) == 1
        assert loaded.entities[0].name == "box"
        assert len(loaded.cameras) == 1
        assert len(loaded.lights) == 1

    def test_to_dict(self):
        cfg = GenesisSceneConfig(entities=[], cameras=[], lights=[])
        d = cfg.to_dict()
        assert isinstance(d["entities"], list)
        assert isinstance(d["cameras"], list)


class TestGenesisSceneManifest:
    def test_manifest_from_dir(self, tmp_path):
        # Create mock export directory
        (tmp_path / "mesh.obj").write_text("v 0 0 0")
        (tmp_path / "robot.urdf").write_text("<robot/>")
        (tmp_path / "scene.xml").write_text("<mujoco/>")
        (tmp_path / "world.usda").write_text("")
        (tmp_path / "meta.json").write_text("{}")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "part.stl").write_bytes(b"\x00" * 10)

        manifest = scene_manifest_from_dir(tmp_path)
        assert manifest.root_dir == str(tmp_path.resolve())
        assert "mesh.obj" in manifest.mesh_files
        assert any("part.stl" in f for f in manifest.mesh_files)
        assert "robot.urdf" in manifest.urdf_files
        assert "scene.xml" in manifest.mjcf_files
        assert "world.usda" in manifest.usd_files
        assert "meta.json" in manifest.metadata_files
        assert manifest.total_assets >= 5

    def test_manifest_nonexistent_dir_raises(self):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            scene_manifest_from_dir("/nonexistent/path")

    def test_manifest_json_round_trip(self, tmp_path):
        manifest = GenesisSceneManifest(
            root_dir="/test",
            mesh_files=["a.obj", "b.ply"],
            mjcf_files=["robot.xml"],
        )
        path = tmp_path / "manifest.json"
        manifest.save_json(path)
        loaded = GenesisSceneManifest.load_json(path)
        assert loaded.root_dir == "/test"
        assert loaded.mesh_files == ["a.obj", "b.ply"]
        assert loaded.total_assets == 3

    def test_empty_dir(self, tmp_path):
        manifest = scene_manifest_from_dir(tmp_path)
        assert manifest.total_assets == 0


class TestCameraFromSyndata:
    def test_monocular_single_drone(self):
        cam = DroneCamera(fov_deg=90.0, aspect_ratio=ASPECT_4_3)
        rig = CameraRigConfig.monocular(n_drones=1)
        genesis_cams = camera_from_syndata(cam, rig, resolution=(320, 240))
        assert len(genesis_cams) == 1
        assert genesis_cams[0].fov == 90.0
        assert genesis_cams[0].res == (320, 240)

    def test_stereo_multi_drone(self):
        cam = DroneCamera(fov_deg=120.0)
        rig = CameraRigConfig.stereo(baseline_m=0.065, n_drones=3)
        genesis_cams = camera_from_syndata(cam, rig)
        # 3 rigs × 2 cameras (stereo) = 6
        assert len(genesis_cams) == 6
        for gc in genesis_cams:
            assert gc.fov == 120.0


class TestObservationToRenderKwargs:
    def test_navigation_passes(self):
        obs = ObservationConfig(passes=PASSES_NAVIGATION, include_rgb=True)
        kwargs = observation_to_render_kwargs(obs)
        assert kwargs["rgb"] is True
        assert kwargs["depth"] is True
        assert kwargs["segmentation"] is True
        assert kwargs["normal"] is True

    def test_minimal_passes(self):
        obs = ObservationConfig(passes=PASSES_MINIMAL, include_rgb=False)
        kwargs = observation_to_render_kwargs(obs)
        assert kwargs["rgb"] is False
        assert kwargs["depth"] is True
        assert kwargs["segmentation"] is True
        assert kwargs["normal"] is False


class TestRandomisationToLights:
    def test_produces_lights(self):
        r = DomainRandomiser(difficulty=0.5, seed=42)
        lights = randomisation_to_genesis_lights(r)
        assert len(lights) >= 1
        assert all(isinstance(l, GenesisLight) for l in lights)

    def test_reproducible(self):
        r = DomainRandomiser(difficulty=0.5, seed=42)
        l1 = randomisation_to_genesis_lights(r)
        l2 = randomisation_to_genesis_lights(r)
        assert l1[0].pos == l2[0].pos
        assert l1[0].color == l2[0].color


class TestMetadataToEntities:
    def test_obstacle_boxes(self):
        meta = FrameMetadata(
            obstacles=[
                BBox3D(center=(1, 2, 3), extent=(0.5, 0.5, 0.5), label="tree"),
                BBox3D(center=(4, 5, 6), extent=(1, 1, 2), label="wall"),
            ],
        )
        ents = metadata_to_entities(meta)
        assert len(ents) == 2
        assert ents[0].morph_type == "Box"
        assert ents[0].pos == (1, 2, 3)
        assert ents[0].extra["size"] == (1.0, 1.0, 1.0)  # 2 × half-extents (0.5, 0.5, 0.5)
        assert "tree" in ents[0].name
        assert ents[1].extra["size"] == (2, 2, 4)

    def test_no_obstacles(self):
        meta = FrameMetadata(obstacles=[])
        ents = metadata_to_entities(meta)
        assert ents == []


class TestManifestToEntities:
    def test_converts_manifest(self):
        manifest = GenesisSceneManifest(
            root_dir="/export",
            mesh_files=["mesh.obj", "terrain.ply"],
            mjcf_files=["robot.xml"],
            urdf_files=["arm.urdf"],
        )
        ents = manifest_to_entities(manifest)
        assert len(ents) == 4
        types = {e.morph_type for e in ents}
        assert types == {"Mesh", "MJCF", "URDF"}

    def test_filter_types(self):
        manifest = GenesisSceneManifest(
            root_dir="/export",
            mesh_files=["mesh.obj"],
            mjcf_files=["robot.xml"],
            urdf_files=["arm.urdf"],
        )
        ents = manifest_to_entities(manifest, include_mjcf=False, include_urdf=False)
        assert len(ents) == 1
        assert ents[0].morph_type == "Mesh"


class TestToGenesisScript:
    def test_generates_valid_python(self):
        cfg = GenesisSceneConfig(
            entities=[
                GenesisEntityConfig(name="ground", morph_type="Plane", is_fixed=True),
                GenesisEntityConfig(
                    name="obstacle", morph_type="Box", pos=(1, 0, 0.5),
                    extra={"size": (2, 2, 1)},
                ),
            ],
            cameras=[GenesisCamera(name="cam_0", res=(320, 240), fov=90.0)],
            lights=[GenesisLight(pos=(0, 0, 10))],
            backend="cpu",
        )
        script = to_genesis_script(cfg)
        assert "import genesis as gs" in script
        assert "gs.init(backend=gs.cpu)" in script
        assert "scene = gs.Scene(" in script
        assert "gs.morphs.Plane" in script
        assert "gs.morphs.Box" in script
        assert "scene.add_camera(" in script
        assert "scene.build()" in script
        assert "scene.step(" in script
        assert "cam_0.render(" in script
        # The script should be parseable Python
        compile(script, "<genesis_script>", "exec")

    def test_empty_scene_compiles(self):
        cfg = GenesisSceneConfig()
        script = to_genesis_script(cfg)
        compile(script, "<genesis_script>", "exec")

    def test_mesh_entity_in_script(self):
        cfg = GenesisSceneConfig(
            entities=[
                GenesisEntityConfig.from_file("/scene/mesh.obj", pos=(0, 0, 1)),
            ],
        )
        script = to_genesis_script(cfg)
        assert "gs.morphs.Mesh(" in script
        assert '"/scene/mesh.obj"' in script

    def test_surface_in_script(self):
        cfg = GenesisSceneConfig(
            entities=[
                GenesisEntityConfig(
                    name="wall", morph_type="Box",
                    surface="Rough", surface_color=(0.8, 0.2, 0.1),
                ),
            ],
        )
        script = to_genesis_script(cfg)
        assert "gs.surfaces.Rough(" in script


class TestBuildGenesisConfig:
    def test_from_metadata_only(self):
        meta = FrameMetadata(
            obstacles=[BBox3D(center=(1, 0, 1), extent=(0.5, 0.5, 0.5))],
        )
        cfg = build_genesis_config(frame_metadata=meta, backend="cpu")
        assert cfg.backend == "cpu"
        # Should have ground plane + 1 obstacle = at least 2 entities
        assert any(e.morph_type == "Plane" for e in cfg.entities)
        assert any(e.morph_type == "Box" for e in cfg.entities)
        assert len(cfg.entities) >= 2
        assert len(cfg.cameras) >= 1
        assert len(cfg.lights) >= 1

    def test_from_camera_config(self):
        cam = DroneCamera(fov_deg=90.0)
        rig = CameraRigConfig.monocular(n_drones=2)
        cfg = build_genesis_config(
            drone_camera=cam, camera_rig=rig,
            resolution=(320, 240), backend="cpu",
        )
        assert len(cfg.cameras) == 2

    def test_from_randomiser(self):
        r = DomainRandomiser(difficulty=0.7, seed=42)
        cfg = build_genesis_config(randomiser=r, backend="cpu")
        assert len(cfg.lights) >= 1

    def test_from_export_dir(self, tmp_path):
        (tmp_path / "terrain.obj").write_text("v 0 0 0")
        (tmp_path / "robot.urdf").write_text("<robot/>")
        cfg = build_genesis_config(export_dir=tmp_path, backend="cpu")
        # Should have ground + terrain.obj + robot.urdf
        mesh_ents = [e for e in cfg.entities if e.morph_type == "Mesh"]
        urdf_ents = [e for e in cfg.entities if e.morph_type == "URDF"]
        assert len(mesh_ents) >= 1
        assert len(urdf_ents) >= 1

    def test_full_pipeline(self, tmp_path):
        """End-to-end: syndata configs → Genesis scene config → script."""
        # Export dir with assets
        (tmp_path / "terrain.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3")
        (tmp_path / "drone.xml").write_text("<mujoco><worldbody/></mujoco>")

        meta = FrameMetadata(
            frame_id=0,
            camera_position=(3, 0, 2),
            obstacles=[
                BBox3D(center=(1, 0, 1), extent=(0.5, 0.5, 0.5), label="rock"),
            ],
        )
        cam = DroneCamera(fov_deg=90.0, aspect_ratio=ASPECT_4_3)
        rig = CameraRigConfig.stereo(baseline_m=0.065, n_drones=2)
        ep = EpisodeConfig.short_trajectory(num_frames=30, fps=10)
        obs = ObservationConfig(passes=PASSES_NAVIGATION)
        randomiser = DomainRandomiser(difficulty=0.5, seed=42)

        cfg = build_genesis_config(
            export_dir=tmp_path,
            frame_metadata=meta,
            drone_camera=cam,
            camera_rig=rig,
            episode=ep,
            observation=obs,
            randomiser=randomiser,
            resolution=(320, 240),
            backend="cpu",
        )

        # Verify structure
        assert cfg.backend == "cpu"
        assert len(cfg.entities) >= 4  # ground + terrain.obj + drone.xml + obstacle
        assert len(cfg.cameras) == 4  # 2 drones × 2 (stereo)
        assert len(cfg.lights) >= 1

        # Verify episode and observation mapping
        assert cfg.episode is not None
        assert cfg.episode.num_steps == 30
        assert cfg.episode.fps == 10
        assert cfg.observation is not None
        assert cfg.observation.depth is True
        assert cfg.observation.normal is True

        # Generate script
        script = to_genesis_script(cfg)
        assert "import genesis as gs" in script
        assert "start_recording" in script  # video recording for multi-frame
        compile(script, "<genesis_full_pipeline>", "exec")

        # Check observation mapping (legacy helper still works)
        render_kwargs = observation_to_render_kwargs(obs)
        assert render_kwargs["depth"] is True
        assert render_kwargs["normal"] is True

        # Save/load config
        cfg_path = tmp_path / "genesis_scene.json"
        cfg.save_json(cfg_path)
        loaded = GenesisSceneConfig.load_json(cfg_path)
        assert len(loaded.entities) == len(cfg.entities)
        assert loaded.episode is not None
        assert loaded.episode.num_steps == 30
        assert loaded.observation is not None
        assert loaded.observation.depth is True


# ═══════════════════════════════════════════════════════════════════════════
#  20. Genesis Episode + Observation bridge (Genesis handles these natively)
# ═══════════════════════════════════════════════════════════════════════════


class TestGenesisEpisodeConfig:
    def test_defaults(self):
        ep = GenesisEpisodeConfig()
        assert ep.num_steps == 1000
        assert ep.dt == 0.01
        assert ep.fps == 24

    def test_invalid_num_steps(self):
        with pytest.raises(ValueError, match="num_steps"):
            GenesisEpisodeConfig(num_steps=0)

    def test_invalid_dt(self):
        with pytest.raises(ValueError, match="dt"):
            GenesisEpisodeConfig(dt=-0.01)

    def test_invalid_fps(self):
        with pytest.raises(ValueError, match="fps"):
            GenesisEpisodeConfig(fps=0)

    def test_to_dict(self):
        ep = GenesisEpisodeConfig(num_steps=100, dt=0.005)
        d = ep.to_dict()
        assert d["num_steps"] == 100
        assert d["dt"] == 0.005


class TestGenesisObservationConfig:
    def test_defaults(self):
        obs = GenesisObservationConfig()
        assert obs.rgb is True
        assert obs.depth is True
        assert obs.segmentation is False

    def test_render_kwargs(self):
        obs = GenesisObservationConfig(
            rgb=True, depth=True, segmentation=True, normal=True,
        )
        kw = obs.render_kwargs()
        assert kw == {"rgb": True, "depth": True, "segmentation": True, "normal": True}

    def test_render_kwargs_minimal(self):
        obs = GenesisObservationConfig(rgb=True, depth=False, segmentation=False, normal=False)
        kw = obs.render_kwargs()
        assert kw["depth"] is False

    def test_invalid_depth_clip(self):
        with pytest.raises(ValueError, match="depth_clip"):
            GenesisObservationConfig(depth_clip_m=0)

    def test_invalid_noise(self):
        with pytest.raises(ValueError, match="gaussian_noise"):
            GenesisObservationConfig(gaussian_noise_std=-0.1)

    def test_to_dict(self):
        obs = GenesisObservationConfig(depth=True, normal=True, depth_clip_m=50.0)
        d = obs.to_dict()
        assert d["depth"] is True
        assert d["normal"] is True
        assert d["depth_clip_m"] == 50.0


class TestEpisodeToGenesis:
    def test_static_episode(self):
        ep = EpisodeConfig.single_frame()
        gep = episode_to_genesis(ep)
        assert gep.num_steps == 1
        assert gep.fps == 24
        assert gep.record_video is False  # single frame → no video

    def test_trajectory_episode(self):
        ep = EpisodeConfig.short_trajectory(num_frames=30, fps=10)
        gep = episode_to_genesis(ep, dt=0.005)
        assert gep.num_steps == 30
        assert gep.fps == 10
        assert gep.dt == 0.005
        assert gep.max_episode_length == 30
        assert gep.record_video is True  # multi-frame → video

    def test_navigation_episode(self):
        ep = EpisodeConfig.navigation_episode(num_frames=120, fps=24)
        gep = episode_to_genesis(ep)
        assert gep.num_steps == 120
        assert gep.fps == 24
        assert gep.record_video is True

    def test_custom_dt(self):
        ep = EpisodeConfig(num_frames=50, fps=30)
        gep = episode_to_genesis(ep, dt=0.002)
        assert gep.dt == 0.002


class TestObservationToGenesis:
    def test_navigation_passes(self):
        obs = ObservationConfig(passes=PASSES_NAVIGATION, include_rgb=True)
        gobs = observation_to_genesis(obs)
        assert gobs.rgb is True
        assert gobs.depth is True
        assert gobs.segmentation is True  # PASS_OBJECT_INDEX in PASSES_NAVIGATION
        assert gobs.normal is True
        assert gobs.depth_clip_m == obs.depth_clip_m

    def test_minimal_passes(self):
        obs = ObservationConfig(passes=PASSES_MINIMAL, include_rgb=False)
        gobs = observation_to_genesis(obs)
        assert gobs.rgb is False
        assert gobs.depth is True
        assert gobs.segmentation is True
        assert gobs.normal is False

    def test_noise_passthrough(self):
        from infinigen.core.syndata.observation import SensorNoiseModel
        noise = SensorNoiseModel(gaussian_std=0.02)
        obs = ObservationConfig(noise=noise)
        gobs = observation_to_genesis(obs)
        assert gobs.gaussian_noise_std == 0.02

    def test_depth_clip_passthrough(self):
        obs = ObservationConfig(depth_clip_m=50.0)
        gobs = observation_to_genesis(obs)
        assert gobs.depth_clip_m == 50.0


class TestGenesisScriptWithEpisodeObservation:
    def test_script_with_episode(self):
        cfg = GenesisSceneConfig(
            cameras=[GenesisCamera(name="cam_0")],
            episode=GenesisEpisodeConfig(num_steps=50, dt=0.005, fps=10, record_video=True),
            backend="cpu",
        )
        script = to_genesis_script(cfg)
        assert "start_recording" in script
        assert "stop_recording" in script
        assert "range(50)" in script
        assert "dt=0.005" in script
        assert "fps=10" in script
        compile(script, "<genesis_episode_script>", "exec")

    def test_script_with_observation(self):
        cfg = GenesisSceneConfig(
            cameras=[GenesisCamera(name="cam_0")],
            observation=GenesisObservationConfig(
                rgb=True, depth=True, segmentation=False, normal=True,
            ),
            backend="cpu",
        )
        script = to_genesis_script(cfg)
        assert "rgb=True" in script
        assert "depth=True" in script
        assert "normal=True" in script
        assert "segmentation=True" not in script
        compile(script, "<genesis_obs_script>", "exec")

    def test_script_depth_only(self):
        cfg = GenesisSceneConfig(
            cameras=[GenesisCamera(name="cam_0")],
            observation=GenesisObservationConfig(
                rgb=False, depth=True, segmentation=False, normal=False,
            ),
            backend="cpu",
        )
        script = to_genesis_script(cfg)
        assert "depth = cam_0.render(" in script
        compile(script, "<genesis_depth_only>", "exec")

    def test_script_no_video_when_not_recording(self):
        cfg = GenesisSceneConfig(
            cameras=[GenesisCamera(name="cam_0")],
            episode=GenesisEpisodeConfig(num_steps=10, record_video=False),
            backend="cpu",
        )
        script = to_genesis_script(cfg)
        assert "start_recording" not in script
        assert "stop_recording" not in script

    def test_json_round_trip_with_episode_observation(self, tmp_path):
        cfg = GenesisSceneConfig(
            cameras=[GenesisCamera(name="cam_0")],
            episode=GenesisEpisodeConfig(num_steps=50, dt=0.005, fps=10),
            observation=GenesisObservationConfig(
                depth=True, normal=True, depth_clip_m=50.0,
            ),
            backend="cpu",
        )
        path = tmp_path / "scene.json"
        cfg.save_json(path)
        loaded = GenesisSceneConfig.load_json(path)
        assert loaded.episode is not None
        assert loaded.episode.num_steps == 50
        assert loaded.episode.dt == 0.005
        assert loaded.observation is not None
        assert loaded.observation.depth is True
        assert loaded.observation.depth_clip_m == 50.0


class TestBuildGenesisConfigWithEpisodeObservation:
    def test_build_with_episode(self):
        ep = EpisodeConfig.short_trajectory(num_frames=30, fps=10)
        cfg = build_genesis_config(episode=ep, backend="cpu")
        assert cfg.episode is not None
        assert cfg.episode.num_steps == 30
        assert cfg.episode.fps == 10

    def test_build_with_observation(self):
        obs = ObservationConfig(passes=PASSES_NAVIGATION)
        cfg = build_genesis_config(observation=obs, backend="cpu")
        assert cfg.observation is not None
        assert cfg.observation.depth is True
        assert cfg.observation.normal is True

    def test_build_without_episode_observation(self):
        cfg = build_genesis_config(backend="cpu")
        assert cfg.episode is None
        assert cfg.observation is None


# ═══════════════════════════════════════════════════════════════════════════
#  DroneEnvConfig
# ═══════════════════════════════════════════════════════════════════════════


class TestDroneEnvConfig:
    """Tests for DroneEnvConfig — GenesisDroneEnv YAML config equivalent."""

    def test_defaults(self):
        cfg = DroneEnvConfig()
        assert cfg.num_envs == 4096
        assert cfg.dt == 0.01
        assert cfg.cam_res == (256, 256)
        assert cfg.num_actions == 4
        assert cfg.num_obs == 23

    def test_invalid_num_envs(self):
        with pytest.raises(ValueError, match="num_envs"):
            DroneEnvConfig(num_envs=0)

    def test_invalid_dt(self):
        with pytest.raises(ValueError, match="dt"):
            DroneEnvConfig(dt=-0.01)

    def test_invalid_cam_res(self):
        with pytest.raises(ValueError, match="cam_res"):
            DroneEnvConfig(cam_res=(0, 256))

    def test_to_genesis_env_yaml(self):
        cfg = DroneEnvConfig(num_envs=1024, dt=0.005, cam_res=(320, 240))
        y = cfg.to_genesis_env_yaml()
        assert y["num_envs"] == 1024
        assert y["dt"] == 0.005
        assert y["cam_res"] == [320, 240]
        assert y["controller"] == "angle"
        assert y["drone_num"] == 1
        assert "init_x_range" in y
        assert "init_y_range" in y
        assert "init_z_range" in y

    def test_to_rl_env_yaml(self):
        cfg = DroneEnvConfig(
            reward_scales={"target": 5.0, "crash": -20.0},
            max_episode_length=2000,
        )
        y = cfg.to_rl_env_yaml()
        assert "task" in y
        task = y["task"]
        assert task["reward_scales"]["target"] == 5.0
        assert task["reward_scales"]["crash"] == -20.0
        assert task["max_episode_length"] == 2000
        assert task["num_actions"] == 4
        assert "command_cfg" in task
        assert "pos_x_range" in task["command_cfg"]

    def test_to_genesis_env_yaml_has_all_keys(self):
        """Ensure generated YAML has the keys GenesisDroneEnv expects."""
        y = DroneEnvConfig().to_genesis_env_yaml()
        expected_keys = {
            "num_envs", "drone_num", "dt", "max_vis_FPS",
            "use_FPV_camera", "cam_quat", "cam_pos", "cam_res",
            "drone_init_pos", "vis_waypoints", "viewer_follow_drone",
            "show_cam_GUI", "fixed_init_pos", "znear", "zfar",
            "load_map", "use_rc", "show_viewer", "render_cam",
            "controller", "min_dis", "map_width", "map_length",
            "init_x_range", "init_y_range", "init_z_range", "target_thr",
        }
        assert expected_keys.issubset(set(y.keys()))

    def test_json_round_trip(self, tmp_path):
        cfg = DroneEnvConfig(
            num_envs=512,
            cam_res=(320, 240),
            drone_init_pos=(1.0, 2.0, 0.5),
            map_size=(5.0, 5.0),
        )
        p = tmp_path / "drone_env.json"
        cfg.save_json(p)
        loaded = DroneEnvConfig.load_json(p)
        assert loaded.num_envs == 512
        assert loaded.cam_res == (320, 240)
        assert loaded.drone_init_pos == (1.0, 2.0, 0.5)
        assert loaded.map_size == (5.0, 5.0)

    def test_json_round_trip_tuple_normalisation(self, tmp_path):
        """Ensure tuples survive JSON round-trip (JSON only has arrays)."""
        cfg = DroneEnvConfig(
            init_pos_range={"x": (-1.0, 1.0), "y": (-0.5, 0.5), "z": (0.3, 0.7)},
            command_ranges={"x": (-2.0, 2.0), "y": (-2.0, 2.0), "z": (0.5, 1.5)},
        )
        p = tmp_path / "cfg.json"
        cfg.save_json(p)
        loaded = DroneEnvConfig.load_json(p)
        assert isinstance(loaded.cam_res, tuple)
        assert isinstance(loaded.init_pos_range["x"], tuple)
        assert isinstance(loaded.command_ranges["x"], tuple)

    def test_custom_termination(self):
        cfg = DroneEnvConfig(
            termination_conditions={"roll_deg": 90.0, "pitch_deg": 90.0, "ground_m": 0.2}
        )
        y = cfg.to_rl_env_yaml()
        assert y["task"]["termination_if_roll_greater_than"] == 90.0
        assert y["task"]["termination_if_pitch_greater_than"] == 90.0
        assert y["task"]["termination_if_close_to_ground"] == 0.2


# ═══════════════════════════════════════════════════════════════════════════
#  TrainingOutcome
# ═══════════════════════════════════════════════════════════════════════════


class TestTrainingOutcome:
    """Tests for TrainingOutcome — structured training feedback."""

    def test_defaults(self):
        outcome = TrainingOutcome()
        assert outcome.success_rate == 0.0
        assert outcome.crash_rate == 0.0
        assert outcome.difficulty == 0.0

    def test_valid_outcome(self):
        outcome = TrainingOutcome(
            success_rate=0.8,
            mean_reward=150.0,
            crash_rate=0.1,
            timeout_rate=0.1,
            difficulty=0.5,
            num_episodes=1000,
        )
        assert outcome.success_rate == 0.8
        assert outcome.num_episodes == 1000

    def test_invalid_success_rate(self):
        with pytest.raises(ValueError, match="success_rate"):
            TrainingOutcome(success_rate=1.5)

    def test_invalid_crash_rate(self):
        with pytest.raises(ValueError, match="crash_rate"):
            TrainingOutcome(crash_rate=-0.1)

    def test_invalid_difficulty(self):
        with pytest.raises(ValueError, match="difficulty"):
            TrainingOutcome(difficulty=1.1)

    def test_json_round_trip(self, tmp_path):
        outcome = TrainingOutcome(
            success_rate=0.75,
            mean_reward=120.5,
            crash_rate=0.15,
            failure_modes={"roll": 0.05, "ground_collision": 0.1},
            difficulty=0.3,
            metadata={"scene_id": "scene_001"},
        )
        p = tmp_path / "outcome.json"
        outcome.save_json(p)
        loaded = TrainingOutcome.load_json(p)
        assert loaded.success_rate == 0.75
        assert loaded.crash_rate == 0.15
        assert loaded.failure_modes["roll"] == 0.05
        assert loaded.metadata["scene_id"] == "scene_001"

    def test_reward_breakdown(self):
        outcome = TrainingOutcome(
            reward_breakdown={"target": 8.5, "crash": -2.0, "smooth": -0.01}
        )
        assert outcome.reward_breakdown["target"] == 8.5
        assert outcome.reward_breakdown["crash"] == -2.0


# ═══════════════════════════════════════════════════════════════════════════
#  outcome_to_curriculum_params
# ═══════════════════════════════════════════════════════════════════════════


class TestOutcomeToCurriculumParams:
    """Tests for curriculum progression from training outcomes."""

    def test_advance_on_high_success(self):
        outcome = TrainingOutcome(success_rate=0.85, crash_rate=0.05)
        result = outcome_to_curriculum_params(outcome, current_stage=3, total_stages=10)
        assert result["action"] == "advance"
        assert result["recommended_stage"] == 4

    def test_regress_on_high_crash(self):
        outcome = TrainingOutcome(success_rate=0.2, crash_rate=0.6)
        result = outcome_to_curriculum_params(outcome, current_stage=5, total_stages=10)
        assert result["action"] == "regress"
        assert result["recommended_stage"] == 4

    def test_hold_on_moderate_performance(self):
        outcome = TrainingOutcome(success_rate=0.5, crash_rate=0.3)
        result = outcome_to_curriculum_params(outcome, current_stage=3, total_stages=10)
        assert result["action"] == "hold"
        assert result["recommended_stage"] == 3

    def test_advance_capped_at_max_stage(self):
        outcome = TrainingOutcome(success_rate=0.95, crash_rate=0.01)
        result = outcome_to_curriculum_params(outcome, current_stage=9, total_stages=10)
        assert result["recommended_stage"] == 9  # can't go beyond last stage
        assert result["action"] == "advance"

    def test_regress_capped_at_zero(self):
        outcome = TrainingOutcome(success_rate=0.1, crash_rate=0.8)
        result = outcome_to_curriculum_params(outcome, current_stage=0, total_stages=10)
        assert result["recommended_stage"] == 0  # can't go below 0
        assert result["action"] == "regress"

    def test_difficulty_scales_with_stage(self):
        outcome = TrainingOutcome(success_rate=0.9, crash_rate=0.02)
        result = outcome_to_curriculum_params(outcome, current_stage=4, total_stages=10)
        # Stage 5 → difficulty = 5/9 ≈ 0.556
        assert 0.5 < result["difficulty"] < 0.6

    def test_scene_adjustments_on_ground_collision(self):
        outcome = TrainingOutcome(
            success_rate=0.5,
            crash_rate=0.4,
            failure_modes={"ground_collision": 0.35},
        )
        result = outcome_to_curriculum_params(outcome, current_stage=3, total_stages=10)
        assert result["scene_adjustments"]["widen_corridors"] is True

    def test_scene_adjustments_increase_obstacles(self):
        outcome = TrainingOutcome(success_rate=0.95, crash_rate=0.02)
        result = outcome_to_curriculum_params(outcome, current_stage=3, total_stages=10)
        assert result["scene_adjustments"]["increase_obstacles"] is True

    def test_scene_adjustments_edge_cases(self):
        outcome = TrainingOutcome(
            success_rate=0.5,
            crash_rate=0.3,
            failure_modes={"roll": 0.2, "pitch": 0.15},
        )
        result = outcome_to_curriculum_params(outcome, current_stage=3, total_stages=10)
        assert result["scene_adjustments"]["add_edge_cases"] is True

    def test_invalid_total_stages(self):
        outcome = TrainingOutcome()
        with pytest.raises(ValueError, match="total_stages"):
            outcome_to_curriculum_params(outcome, current_stage=0, total_stages=0)

    def test_invalid_current_stage(self):
        outcome = TrainingOutcome()
        with pytest.raises(ValueError, match="current_stage"):
            outcome_to_curriculum_params(outcome, current_stage=10, total_stages=10)

    def test_custom_thresholds(self):
        outcome = TrainingOutcome(success_rate=0.6, crash_rate=0.2)
        result = outcome_to_curriculum_params(
            outcome,
            current_stage=3,
            total_stages=10,
            advance_threshold=0.5,
            regress_crash_threshold=0.3,
        )
        assert result["action"] == "advance"

    def test_single_stage_curriculum(self):
        outcome = TrainingOutcome(success_rate=0.9, crash_rate=0.05)
        result = outcome_to_curriculum_params(outcome, current_stage=0, total_stages=1)
        assert result["recommended_stage"] == 0
        assert result["difficulty"] == 0.0  # 0/max(1,0)


# ═══════════════════════════════════════════════════════════════════════════
#  syndata_to_drone_env_config
# ═══════════════════════════════════════════════════════════════════════════


class TestSyndataToDroneEnvConfig:
    """Tests for Infinigen → DroneEnv config conversion."""

    def test_defaults(self):
        cfg = syndata_to_drone_env_config()
        assert isinstance(cfg, DroneEnvConfig)
        assert cfg.num_envs == 4096

    def test_with_camera(self):
        cam = DroneCamera(fov_deg=90, aspect_ratio=16 / 9)
        cfg = syndata_to_drone_env_config(drone_camera=cam)
        w, h = cfg.cam_res
        # Should scale width to match 16:9 aspect
        assert w > h

    def test_with_observation_enables_camera(self):
        obs = ObservationConfig(passes=PASSES_NAVIGATION)
        cfg = syndata_to_drone_env_config(observation=obs)
        assert cfg.render_cam is True
        assert cfg.use_fpv_camera is True

    def test_with_metadata_obstacles(self):
        meta = FrameMetadata(
            frame_id=0,
            obstacles=[
                BBox3D(center=(1.0, 2.0, 0.5), extent=(0.5, 0.5, 0.5), label="tree"),
                BBox3D(center=(-1.0, -1.0, 0.5), extent=(0.3, 0.3, 1.0), label="building"),
            ],
        )
        cfg = syndata_to_drone_env_config(frame_metadata=meta)
        assert len(cfg.obstacle_entities) == 2
        assert cfg.obstacle_entities[0]["name"] == "obstacle_0_tree"
        # Init range should be derived from obstacle positions
        assert cfg.init_pos_range["x"][0] < 0  # covers both obstacles

    def test_with_curriculum_scales_crash_penalty(self):
        curriculum = CurriculumConfig(stage=5, total_stages=10)
        cfg = syndata_to_drone_env_config(curriculum=curriculum)
        # Higher stage → harsher crash penalty
        assert cfg.reward_scales["crash"] < -10.0

    def test_with_randomiser(self):
        rand = DomainRandomiser(difficulty=0.8, seed=42)
        cfg = syndata_to_drone_env_config(randomiser=rand)
        # Higher difficulty → wider init ranges
        _x_lo, x_hi = cfg.init_pos_range["x"]
        assert x_hi > 0.2  # Much wider than default 0.05

    def test_with_scene_config(self):
        scene = GenesisSceneConfig(
            entities=[
                GenesisEntityConfig(name="ground", morph_type="Plane"),
                GenesisEntityConfig(name="wall_1", morph_type="Box", pos=(1, 0, 0.5)),
                GenesisEntityConfig(name="rock", morph_type="Mesh", file_path="rock.obj"),
            ],
            backend="cpu",
        )
        cfg = syndata_to_drone_env_config(scene_config=scene)
        # Ground plane should be excluded; only wall_1 and rock included
        names = [e["name"] for e in cfg.obstacle_entities]
        assert "ground" not in names
        assert "wall_1" in names
        assert "rock" in names

    def test_custom_num_envs(self):
        cfg = syndata_to_drone_env_config(num_envs=512)
        assert cfg.num_envs == 512


# ═══════════════════════════════════════════════════════════════════════════
#  apply_curriculum_adjustment
# ═══════════════════════════════════════════════════════════════════════════


class TestApplyCurriculumAdjustment:
    """Tests for applying curriculum feedback to DroneEnvConfig."""

    def test_widen_corridors(self):
        base = DroneEnvConfig(map_size=(3.5, 3.5))
        adj = {"scene_adjustments": {"widen_corridors": True}, "difficulty": 0.5}
        result = apply_curriculum_adjustment(base, adj)
        assert result.map_size[0] > 3.5
        assert result.map_size[1] > 3.5

    def test_reduce_clutter(self):
        base = DroneEnvConfig(command_ranges={"x": (-2.0, 2.0), "y": (-2.0, 2.0), "z": (0.5, 1.0)})
        adj = {"scene_adjustments": {"reduce_clutter": True}, "difficulty": 0.5}
        result = apply_curriculum_adjustment(base, adj)
        # Command range should shrink
        x_lo, x_hi = result.command_ranges["x"]
        assert abs(x_hi - x_lo) < 4.0

    def test_increase_obstacles_penalty(self):
        base = DroneEnvConfig(reward_scales={"crash": -10.0})
        adj = {"scene_adjustments": {"increase_obstacles": True}, "difficulty": 0.5}
        result = apply_curriculum_adjustment(base, adj)
        assert result.reward_scales["crash"] < -10.0

    def test_difficulty_scales_init_spread(self):
        base = DroneEnvConfig()
        adj_easy = {"scene_adjustments": {}, "difficulty": 0.1}
        adj_hard = {"scene_adjustments": {}, "difficulty": 0.9}
        easy = apply_curriculum_adjustment(base, adj_easy)
        hard = apply_curriculum_adjustment(base, adj_hard)
        easy_spread = easy.init_pos_range["x"][1] - easy.init_pos_range["x"][0]
        hard_spread = hard.init_pos_range["x"][1] - hard.init_pos_range["x"][0]
        assert hard_spread > easy_spread


# ═══════════════════════════════════════════════════════════════════════════
#  scene_to_drone_entities
# ═══════════════════════════════════════════════════════════════════════════


class TestSceneToDroneEntities:
    """Tests for scene_to_drone_entities code generation."""

    def test_basic_entity(self):
        scene = GenesisSceneConfig(
            entities=[
                GenesisEntityConfig(name="wall", morph_type="Box", pos=(1, 0, 0.5), is_fixed=True),
            ],
            backend="cpu",
        )
        lines = scene_to_drone_entities(scene)
        assert len(lines) == 1
        assert "gs.morphs.Box" in lines[0]
        assert "fixed=True" in lines[0]
        assert "self.wall" in lines[0]

    def test_skips_ground_plane(self):
        scene = GenesisSceneConfig(
            entities=[
                GenesisEntityConfig(name="ground", morph_type="Plane"),
                GenesisEntityConfig(name="rock", morph_type="Mesh", file_path="rock.obj"),
            ],
            backend="cpu",
        )
        lines = scene_to_drone_entities(scene)
        assert len(lines) == 1
        assert "rock" in lines[0]
        assert "gs.morphs.Mesh" in lines[0]

    def test_file_path_in_output(self):
        scene = GenesisSceneConfig(
            entities=[
                GenesisEntityConfig(name="terrain", morph_type="Mesh", file_path="/data/terrain.obj"),
            ],
            backend="cpu",
        )
        lines = scene_to_drone_entities(scene)
        assert "/data/terrain.obj" in lines[0]

    def test_extra_kwargs(self):
        scene = GenesisSceneConfig(
            entities=[
                GenesisEntityConfig(
                    name="box_1",
                    morph_type="Box",
                    extra={"size": (2.0, 1.0, 0.5)},
                ),
            ],
            backend="cpu",
        )
        lines = scene_to_drone_entities(scene)
        assert "size=" in lines[0]

    def test_empty_scene(self):
        scene = GenesisSceneConfig(backend="cpu")
        lines = scene_to_drone_entities(scene)
        assert lines == []


# ═══════════════════════════════════════════════════════════════════════════
#  End-to-end: Infinigen → DroneEnv → Training → Curriculum feedback
# ═══════════════════════════════════════════════════════════════════════════


class TestEndToEndCurriculumLoop:
    """End-to-end test for the bidirectional Infinigen ↔ DroneEnv bridge."""

    def test_full_curriculum_loop(self):
        """Simulate a complete curriculum training loop:
        1. Generate Infinigen scene config for stage 0
        2. Convert to DroneEnvConfig
        3. Simulate training outcome (success)
        4. Get curriculum adjustment → advance
        5. Apply adjustment to next stage config
        """
        # Stage 0: easy curriculum
        curriculum_0 = CurriculumConfig(stage=0, total_stages=5)
        drone_cam = DroneCamera(fov_deg=90)
        rig = CameraRigConfig.stereo(baseline_m=0.065, n_drones=1)
        meta = FrameMetadata(
            frame_id=0,
            obstacles=[
                BBox3D(center=(2.0, 0.0, 0.5), extent=(0.5, 0.5, 0.5), label="tree"),
            ],
        )

        # Step 1: Infinigen → DroneEnv config
        env_cfg = syndata_to_drone_env_config(
            curriculum=curriculum_0,
            drone_camera=drone_cam,
            frame_metadata=meta,
            num_envs=1024,
        )
        assert isinstance(env_cfg, DroneEnvConfig)
        assert len(env_cfg.obstacle_entities) == 1

        # Step 2: Verify YAML output is valid
        genesis_yaml = env_cfg.to_genesis_env_yaml()
        assert genesis_yaml["num_envs"] == 1024
        rl_yaml = env_cfg.to_rl_env_yaml()
        assert "task" in rl_yaml

        # Step 3: Simulate a good training outcome
        outcome = TrainingOutcome(
            success_rate=0.85,
            mean_reward=150.0,
            crash_rate=0.05,
            timeout_rate=0.10,
            difficulty=0.0,
            num_episodes=10000,
        )

        # Step 4: Get curriculum recommendation
        adj = outcome_to_curriculum_params(outcome, current_stage=0, total_stages=5)
        assert adj["action"] == "advance"
        assert adj["recommended_stage"] == 1

        # Step 5: Apply adjustment
        next_cfg = apply_curriculum_adjustment(env_cfg, adj)
        assert isinstance(next_cfg, DroneEnvConfig)

    def test_curriculum_regression_on_failure(self):
        """When the agent fails badly, curriculum should regress."""
        outcome = TrainingOutcome(
            success_rate=0.1,
            crash_rate=0.7,
            failure_modes={"ground_collision": 0.4, "roll": 0.2, "pitch": 0.1},
            difficulty=0.5,
        )
        adj = outcome_to_curriculum_params(outcome, current_stage=5, total_stages=10)
        assert adj["action"] == "regress"
        assert adj["recommended_stage"] == 4
        assert adj["scene_adjustments"]["widen_corridors"] is True

    def test_genesis_scene_to_drone_env_round_trip(self, tmp_path):
        """Build Genesis scene → convert to DroneEnv → verify entities preserved."""
        # Build Genesis scene from Infinigen outputs
        genesis_cfg = build_genesis_config(
            drone_camera=DroneCamera(fov_deg=90),
            camera_rig=CameraRigConfig.stereo(n_drones=2),
            randomiser=DomainRandomiser(difficulty=0.5, seed=42),
            backend="cpu",
        )

        # Convert to DroneEnv config
        env_cfg = syndata_to_drone_env_config(scene_config=genesis_cfg, num_envs=256)
        assert isinstance(env_cfg, DroneEnvConfig)

        # Verify JSON persistence
        p = tmp_path / "drone_env.json"
        env_cfg.save_json(p)
        loaded = DroneEnvConfig.load_json(p)
        assert loaded.num_envs == 256


# ===========================================================================
# Pre-training (flappy-bird corridor)
# ===========================================================================


class TestFlappyColumnConfig:
    def test_defaults(self):
        cfg = FlappyColumnConfig()
        assert cfg.corridor_length == 8.0
        assert cfg.num_columns == 5
        assert cfg.gap_height >= 0.2

    def test_custom(self):
        cfg = FlappyColumnConfig(num_columns=10, gap_height=0.8)
        assert cfg.num_columns == 10
        assert cfg.gap_height == 0.8

    def test_zero_columns(self):
        cfg = FlappyColumnConfig(num_columns=0)
        assert cfg.num_columns == 0

    def test_negative_columns_raises(self):
        with pytest.raises(ValueError, match="num_columns"):
            FlappyColumnConfig(num_columns=-1)

    def test_small_gap_raises(self):
        with pytest.raises(ValueError, match="gap_height"):
            FlappyColumnConfig(gap_height=0.1)

    def test_negative_corridor_length_raises(self):
        with pytest.raises(ValueError, match="corridor_length"):
            FlappyColumnConfig(corridor_length=-1)

    def test_negative_gap_variation_raises(self):
        with pytest.raises(ValueError, match="gap_height_variation"):
            FlappyColumnConfig(gap_height_variation=-0.1)

    def test_inverted_gap_z_raises(self):
        with pytest.raises(ValueError, match="max_gap_z"):
            FlappyColumnConfig(min_gap_z=1.5, max_gap_z=0.5)


class TestGenerateFlappyObstacles:
    def test_default_produces_obstacles(self):
        cfg = FlappyColumnConfig()
        obs = generate_flappy_obstacles(cfg, seed=42)
        # 5 columns × 2 (upper+lower) + floor + ceiling = 12
        assert len(obs) >= 7  # at least floor + ceiling + some columns

    def test_zero_columns_has_floor_ceiling(self):
        cfg = FlappyColumnConfig(num_columns=0)
        obs = generate_flappy_obstacles(cfg, seed=0)
        labels = [o.label for o in obs]
        assert "floor" in labels
        assert "ceiling" in labels

    def test_reproducible_with_seed(self):
        cfg = FlappyColumnConfig(num_columns=3)
        obs1 = generate_flappy_obstacles(cfg, seed=123)
        obs2 = generate_flappy_obstacles(cfg, seed=123)
        assert len(obs1) == len(obs2)
        for a, b in zip(obs1, obs2):
            assert a.center == b.center
            assert a.half_extents == b.half_extents

    def test_different_seeds_differ(self):
        cfg = FlappyColumnConfig(num_columns=5)
        obs1 = generate_flappy_obstacles(cfg, seed=1)
        obs2 = generate_flappy_obstacles(cfg, seed=2)
        # At least one obstacle should differ in z position
        any_differ = any(
            a.center[2] != b.center[2]
            for a, b in zip(obs1, obs2)
            if a.label == b.label and "column" in a.label
        )
        assert any_differ

    def test_obstacle_labels(self):
        cfg = FlappyColumnConfig(num_columns=2)
        obs = generate_flappy_obstacles(cfg, seed=42)
        labels = {o.label for o in obs}
        assert "floor" in labels
        assert "ceiling" in labels
        # At least one column obstacle
        assert any("column" in lbl for lbl in labels)

    def test_invalid_config_type_raises(self):
        with pytest.raises(TypeError, match="FlappyColumnConfig"):
            generate_flappy_obstacles("not a config")

    def test_obstacles_are_dataclass(self):
        cfg = FlappyColumnConfig(num_columns=1)
        obs = generate_flappy_obstacles(cfg, seed=0)
        for o in obs:
            assert hasattr(o, "center")
            assert hasattr(o, "half_extents")
            assert hasattr(o, "label")
            d = o.to_dict()
            assert "center" in d

    def test_half_extents_positive(self):
        cfg = FlappyColumnConfig(num_columns=5, corridor_height=3.0)
        obs = generate_flappy_obstacles(cfg, seed=42)
        for o in obs:
            for v in o.half_extents:
                assert v > 0, f"Non-positive half_extent in {o.label}"


class TestFlappyDroneEnvConfig:
    def test_defaults(self):
        result = flappy_drone_env_config()
        assert result["num_envs"] == 2048
        assert result["dt"] == 0.01
        assert len(result["obstacle_entities"]) > 0

    def test_custom_num_envs(self):
        result = flappy_drone_env_config(num_envs=512)
        assert result["num_envs"] == 512

    def test_custom_config(self):
        cfg = FlappyColumnConfig(num_columns=3, corridor_length=5.0)
        result = flappy_drone_env_config(cfg, seed=42)
        assert result["map_size"][0] == pytest.approx(6.0)

    def test_invalid_num_envs_raises(self):
        with pytest.raises(ValueError, match="num_envs"):
            flappy_drone_env_config(num_envs=0)

    def test_invalid_dt_raises(self):
        with pytest.raises(ValueError, match="dt"):
            flappy_drone_env_config(dt=0)

    def test_reward_scales(self):
        result = flappy_drone_env_config()
        assert "crash" in result["reward_scales"]
        assert result["reward_scales"]["crash"] < 0

    def test_termination_conditions(self):
        result = flappy_drone_env_config()
        assert "roll_deg" in result["termination_conditions"]
        # Strict termination for pre-training
        assert result["termination_conditions"]["roll_deg"] <= 90.0

    def test_command_ranges_forward_biased(self):
        cfg = FlappyColumnConfig(corridor_length=8.0)
        result = flappy_drone_env_config(cfg)
        x_lo, x_hi = result["command_ranges"]["x"]
        # Target at far end of corridor
        assert x_lo > 2.0
        assert x_hi > x_lo

    def test_can_construct_drone_env_config(self):
        """Verify the output dict is compatible with DroneEnvConfig."""
        result = flappy_drone_env_config(seed=42)
        env = DroneEnvConfig(**result)
        assert env.num_envs == 2048


class TestFlappyGenesisEntities:
    def test_produces_entities(self):
        entities = flappy_genesis_entities(seed=42)
        assert len(entities) > 0

    def test_entity_structure(self):
        entities = flappy_genesis_entities(FlappyColumnConfig(num_columns=2), seed=0)
        for ent in entities:
            assert "name" in ent
            assert "morph_type" in ent
            assert ent["morph_type"] == "Box"
            assert "pos" in ent
            assert "is_fixed" in ent

    def test_custom_config(self):
        cfg = FlappyColumnConfig(num_columns=0)
        entities = flappy_genesis_entities(cfg, seed=0)
        # Floor + ceiling only
        names = {e["name"] for e in entities}
        assert "floor" in names
        assert "ceiling" in names


# ===========================================================================
# Input sanitisation for converter functions
# ===========================================================================


class TestConverterInputSanitisation:
    """Verify converter functions reject invalid input types early."""

    def test_camera_from_syndata_bad_cam(self):
        from infinigen.core.syndata.genesis_export import camera_from_syndata

        rig = CameraRigConfig.monocular()
        with pytest.raises(TypeError, match="fov_deg"):
            camera_from_syndata("not a cam", rig)

    def test_camera_from_syndata_bad_rig(self):
        from infinigen.core.syndata.genesis_export import camera_from_syndata

        cam = DroneCamera(fov_deg=90)
        with pytest.raises(TypeError, match="effective_cameras"):
            camera_from_syndata(cam, "not a rig")

    def test_camera_from_syndata_bad_resolution(self):
        from infinigen.core.syndata.genesis_export import camera_from_syndata

        cam = DroneCamera(fov_deg=90)
        rig = CameraRigConfig.monocular()
        with pytest.raises(ValueError, match="resolution"):
            camera_from_syndata(cam, rig, resolution=(0, 480))

    def test_episode_to_genesis_bad_episode(self):
        from infinigen.core.syndata.genesis_export import episode_to_genesis

        with pytest.raises(TypeError, match="num_frames"):
            episode_to_genesis("not an episode")

    def test_episode_to_genesis_bad_dt(self):
        from infinigen.core.syndata.genesis_export import episode_to_genesis

        ep = EpisodeConfig.single_frame()
        with pytest.raises(ValueError, match="dt"):
            episode_to_genesis(ep, dt=0)

    def test_observation_to_genesis_bad_obs(self):
        from infinigen.core.syndata.genesis_export import observation_to_genesis

        with pytest.raises(TypeError, match="passes"):
            observation_to_genesis("not an obs")

    def test_observation_to_render_kwargs_bad_obs(self):
        from infinigen.core.syndata.genesis_export import observation_to_render_kwargs

        with pytest.raises(TypeError, match="include_rgb"):
            observation_to_render_kwargs("not an obs")

    def test_randomisation_to_genesis_lights_bad_randomiser(self):
        from infinigen.core.syndata.genesis_export import (
            randomisation_to_genesis_lights,
        )

        with pytest.raises(TypeError, match="sample"):
            randomisation_to_genesis_lights("not a randomiser")

    def test_randomisation_to_genesis_lights_bad_height(self):
        from infinigen.core.syndata.genesis_export import (
            randomisation_to_genesis_lights,
        )

        rand = DomainRandomiser(difficulty=0.5, seed=42)
        with pytest.raises(ValueError, match="base_height"):
            randomisation_to_genesis_lights(rand, base_height=-1)

    def test_metadata_to_entities_bad_metadata(self):
        from infinigen.core.syndata.genesis_export import metadata_to_entities

        with pytest.raises(TypeError, match="obstacles"):
            metadata_to_entities("not metadata")

    def test_scene_to_drone_entities_bad_config(self):
        with pytest.raises(TypeError, match="entities"):
            scene_to_drone_entities("not a config")

    def test_syndata_to_drone_env_invalid_num_envs(self):
        with pytest.raises(ValueError, match="num_envs"):
            syndata_to_drone_env_config(num_envs=0)

    def test_syndata_to_drone_env_invalid_dt(self):
        with pytest.raises(ValueError, match="dt"):
            syndata_to_drone_env_config(dt=-1)

    def test_outcome_to_curriculum_bad_outcome(self):
        with pytest.raises(TypeError, match="success_rate"):
            outcome_to_curriculum_params("not an outcome", 0, 5)

    def test_outcome_to_curriculum_bad_advance_threshold(self):
        outcome = TrainingOutcome(success_rate=0.5)
        with pytest.raises(ValueError, match="advance_threshold"):
            outcome_to_curriculum_params(outcome, 0, 5, advance_threshold=1.5)

    def test_outcome_to_curriculum_bad_regress_threshold(self):
        outcome = TrainingOutcome(success_rate=0.5)
        with pytest.raises(ValueError, match="regress_crash_threshold"):
            outcome_to_curriculum_params(outcome, 0, 5, regress_crash_threshold=-0.1)

    def test_apply_curriculum_adjustment_bad_config(self):
        with pytest.raises(TypeError, match="DroneEnvConfig"):
            apply_curriculum_adjustment("not a config", {})

    def test_apply_curriculum_adjustment_bad_adjustment(self):
        with pytest.raises(TypeError, match="dict"):
            apply_curriculum_adjustment(DroneEnvConfig(), "not a dict")


# ===========================================================================
# Separation of concerns — verify clean import groups
# ===========================================================================


class TestImportSeparation:
    """Verify that the public API is properly grouped by concern."""

    def test_infinigen_exports_no_genesis(self):
        """Core Infinigen types should not depend on Genesis types."""
        import importlib

        for mod in [
            "infinigen.core.syndata.camera_config",
            "infinigen.core.syndata.complexity",
            "infinigen.core.syndata.episode",
            "infinigen.core.syndata.observation",
            "infinigen.core.syndata.randomisation",
        ]:
            m = importlib.import_module(mod)
            assert m is not None

    def test_genesis_bridge_imports(self):
        """Genesis bridge types should all be importable."""
        import importlib

        m = importlib.import_module("infinigen.core.syndata.genesis_export")
        for name in [
            "GenesisCamera",
            "GenesisEntityConfig",
            "GenesisEpisodeConfig",
            "GenesisObservationConfig",
            "GenesisSceneConfig",
        ]:
            assert hasattr(m, name)

    def test_drone_env_bridge_imports(self):
        """DroneEnv bridge types should all be importable."""
        import importlib

        m = importlib.import_module("infinigen.core.syndata.drone_env_bridge")
        for name in [
            "DroneEnvConfig",
            "TrainingOutcome",
            "apply_curriculum_adjustment",
            "outcome_to_curriculum_params",
        ]:
            assert hasattr(m, name)

    def test_pretraining_imports(self):
        """Pre-training types should all be importable."""
        import importlib

        m = importlib.import_module("infinigen.core.syndata.pretraining")
        for name in [
            "FlappyColumnConfig",
            "FlappyObstacle",
            "flappy_drone_env_config",
            "flappy_genesis_entities",
            "generate_flappy_obstacles",
        ]:
            assert hasattr(m, name)

    def test_all_exports_present(self):
        """All __all__ entries should be importable."""
        import infinigen.core.syndata as syndata

        for name in syndata.__all__:
            assert hasattr(syndata, name), f"Missing export: {name}"


class TestEndToEndFlappyPretraining:
    """End-to-end: flappy corridor → DroneEnvConfig → curriculum feedback."""

    def test_flappy_to_drone_env_to_outcome(self):
        # Stage 0: simplest possible environment
        flappy_cfg = FlappyColumnConfig(num_columns=3, gap_height=0.8)
        env_dict = flappy_drone_env_config(flappy_cfg, seed=0, num_envs=128)
        env = DroneEnvConfig(**env_dict)

        assert env.num_envs == 128
        assert len(env.obstacle_entities) > 0
        # Strict termination for pre-training
        assert env.termination_conditions["roll_deg"] <= 90.0

        # Simulate training feedback — agent learned basic control
        outcome = TrainingOutcome(
            success_rate=0.85,
            crash_rate=0.1,
            mean_reward=50.0,
            difficulty=0.0,
        )

        # Curriculum says: advance
        params = outcome_to_curriculum_params(outcome, current_stage=0, total_stages=10)
        assert params["action"] == "advance"
        assert params["recommended_stage"] == 1

        # Now use full Infinigen scene for stage 1
        cfg = CurriculumConfig(stage=1, total_stages=10)
        assert cfg.object_count > 0

    def test_flappy_genesis_entities_to_scene_config(self):
        """Flappy entities → GenesisSceneConfig → script."""
        entities = flappy_genesis_entities(
            FlappyColumnConfig(num_columns=2), seed=42
        )
        from infinigen.core.syndata.genesis_export import (
            GenesisEntityConfig,
            GenesisSceneConfig,
            to_genesis_script,
        )

        genesis_ents = [GenesisEntityConfig(**e) for e in entities]
        scene = GenesisSceneConfig(entities=genesis_ents, backend="cpu")
        script = to_genesis_script(scene)
        assert "genesis" in script
        # Should compile cleanly
        compile(script, "<flappy>", "exec")
