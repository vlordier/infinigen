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
from infinigen.core.syndata.episode import EpisodeConfig
from infinigen.core.syndata.metadata import BBox3D, DepthStats, FrameMetadata
from infinigen.core.syndata.metrics import SceneBudget
from infinigen.core.syndata.observation import (
    PASSES_MINIMAL,
    PASSES_NAVIGATION,
    ObservationConfig,
    SensorNoiseModel,
)
from infinigen.core.syndata.parallel_stages import Stage, StageGraph
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
