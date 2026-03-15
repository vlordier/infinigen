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
from infinigen.core.syndata.pretraining import (
    FlappyColumnConfig,
    flappy_frame_metadata,
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
from infinigen.core.syndata.world_gen import (
    InfinigenOverlayHints,
    VisualStyle,
    WorldConfig,
    generate_world,
    overlay_hints_for_complexity,
    world_gin_overrides,
    world_summary,
    world_to_frame_metadata,
)

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
            assert a.extent == b.extent

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
            assert hasattr(o, "extent")
            assert hasattr(o, "label")

    def test_extent_positive(self):
        cfg = FlappyColumnConfig(num_columns=5, corridor_height=3.0)
        obs = generate_flappy_obstacles(cfg, seed=42)
        for o in obs:
            for v in o.extent:
                assert v > 0, f"Non-positive extent in {o.label}"


class TestFlappyPresets:
    """Tests for FlappyColumnConfig preset factory methods."""

    def test_easy_preset(self):
        cfg = FlappyColumnConfig.easy()
        assert cfg.num_columns == 3
        assert cfg.gap_height >= 0.5
        assert cfg.corridor_width >= 2.5

    def test_medium_preset(self):
        cfg = FlappyColumnConfig.medium()
        assert cfg.num_columns == 5
        assert cfg.gap_height >= 0.5

    def test_hard_preset(self):
        cfg = FlappyColumnConfig.hard()
        assert cfg.num_columns == 8
        assert cfg.gap_height >= 0.2
        assert cfg.corridor_width < 2.0

    def test_presets_increasing_difficulty(self):
        easy = FlappyColumnConfig.easy()
        medium = FlappyColumnConfig.medium()
        hard = FlappyColumnConfig.hard()
        # More columns as difficulty increases
        assert easy.num_columns < medium.num_columns < hard.num_columns
        # Smaller gaps as difficulty increases
        assert easy.gap_height > medium.gap_height > hard.gap_height

    def test_easy_generates_obstacles(self):
        obs = generate_flappy_obstacles(FlappyColumnConfig.easy(), seed=42)
        assert len(obs) >= 5

    def test_hard_generates_obstacles(self):
        obs = generate_flappy_obstacles(FlappyColumnConfig.hard(), seed=42)
        assert len(obs) >= 10


class TestFlappyFrameMetadata:
    """Tests for flappy_frame_metadata helper."""

    def test_returns_dict(self):
        cfg = FlappyColumnConfig(num_columns=3)
        meta = flappy_frame_metadata(cfg, seed=42)
        assert isinstance(meta, dict)

    def test_has_required_keys(self):
        cfg = FlappyColumnConfig(num_columns=3)
        meta = flappy_frame_metadata(cfg, seed=42)
        for key in ("frame_id", "camera_position", "obstacles",
                     "depth_stats", "traversability_ratio"):
            assert key in meta, f"Missing key: {key}"

    def test_obstacles_are_dicts(self):
        cfg = FlappyColumnConfig(num_columns=2)
        meta = flappy_frame_metadata(cfg, seed=42)
        assert isinstance(meta["obstacles"], list)
        for obs in meta["obstacles"]:
            assert isinstance(obs, dict)
            assert "center" in obs

    def test_depth_stats_valid(self):
        cfg = FlappyColumnConfig(num_columns=3)
        meta = flappy_frame_metadata(cfg, seed=42)
        ds = meta["depth_stats"]
        assert ds["min_m"] > 0
        assert ds["max_m"] > ds["min_m"]

    def test_traversability_in_range(self):
        cfg = FlappyColumnConfig(num_columns=3)
        meta = flappy_frame_metadata(cfg, seed=42)
        assert 0 < meta["traversability_ratio"] <= 1.0

    def test_invalid_config_type_raises(self):
        with pytest.raises(TypeError, match="FlappyColumnConfig"):
            flappy_frame_metadata("not a config")

    def test_zero_columns(self):
        cfg = FlappyColumnConfig(num_columns=0)
        meta = flappy_frame_metadata(cfg, seed=42)
        assert isinstance(meta["obstacles"], list)


class TestVisualStyle:
    """Tests for VisualStyle dataclass."""

    def test_defaults(self):
        s = VisualStyle()
        assert s.wall_color_hue == 0.0
        assert s.wall_color_saturation == 0.0
        assert s.fog_density == 0.0
        assert s.point_light_count == 0

    def test_saturation_bounds(self):
        with pytest.raises(ValueError, match="wall_color_saturation"):
            VisualStyle(wall_color_saturation=-0.1)
        with pytest.raises(ValueError, match="wall_color_saturation"):
            VisualStyle(wall_color_saturation=1.1)

    def test_roughness_bounds(self):
        with pytest.raises(ValueError, match="floor_roughness"):
            VisualStyle(floor_roughness=-0.1)
        with pytest.raises(ValueError, match="floor_roughness"):
            VisualStyle(floor_roughness=1.1)

    def test_fog_bounds(self):
        with pytest.raises(ValueError, match="fog_density"):
            VisualStyle(fog_density=-0.1)
        with pytest.raises(ValueError, match="fog_density"):
            VisualStyle(fog_density=1.5)

    def test_cloud_bounds(self):
        with pytest.raises(ValueError, match="cloud_density"):
            VisualStyle(cloud_density=-0.1)
        with pytest.raises(ValueError, match="cloud_density"):
            VisualStyle(cloud_density=1.1)

    def test_negative_ambient(self):
        with pytest.raises(ValueError, match="ambient_intensity"):
            VisualStyle(ambient_intensity=-1.0)

    def test_negative_point_lights(self):
        with pytest.raises(ValueError, match="point_light_count"):
            VisualStyle(point_light_count=-1)


class TestWorldConfig:
    """Tests for WorldConfig dataclass and presets."""

    def test_defaults(self):
        cfg = WorldConfig()
        assert cfg.complexity == 0.0
        assert cfg.effective_num_columns == 2  # minimum
        assert cfg.effective_gap_height == 2.0  # wide at c=0
        assert cfg.effective_num_rooms == 0
        assert cfg.effective_num_branches == 0
        assert cfg.effective_num_levels == 1

    def test_complexity_bounds(self):
        with pytest.raises(ValueError, match="complexity"):
            WorldConfig(complexity=-0.1)
        with pytest.raises(ValueError, match="complexity"):
            WorldConfig(complexity=1.1)

    def test_positive_dimensions(self):
        with pytest.raises(ValueError, match="corridor_length"):
            WorldConfig(corridor_length=-1.0)
        with pytest.raises(ValueError, match="corridor_width"):
            WorldConfig(corridor_width=0.0)
        with pytest.raises(ValueError, match="corridor_height"):
            WorldConfig(corridor_height=-1.0)

    def test_gap_height_validation(self):
        with pytest.raises(ValueError, match="gap_height"):
            WorldConfig(gap_height=0.1)  # < 0.3

    def test_debris_density_validation(self):
        with pytest.raises(ValueError, match="debris_density"):
            WorldConfig(debris_density=-0.1)
        with pytest.raises(ValueError, match="debris_density"):
            WorldConfig(debris_density=1.5)

    def test_room_size_range_validation(self):
        with pytest.raises(ValueError, match="room_size_range"):
            WorldConfig(room_size_range=(6.0, 3.0))

    def test_presets_are_valid(self):
        """All presets produce valid configs."""
        for name in ["flappy", "corridor", "rooms", "branches", "maze", "doom"]:
            cfg = getattr(WorldConfig, name)(seed=42)
            assert 0.0 <= cfg.complexity <= 1.0
            assert cfg.seed == 42

    def test_from_curriculum_progress(self):
        cfg = WorldConfig.from_curriculum_progress(0.25, seed=10)
        assert 0.0 <= cfg.complexity <= 1.0
        assert cfg.seed == 10
        # sqrt(0.25) = 0.5
        assert abs(cfg.complexity - 0.5) < 1e-6

    def test_complexity_monotonicity(self):
        """Higher complexity → more obstacles and features."""
        prev_cols = 0
        prev_rooms = 0
        for c in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
            cfg = WorldConfig(complexity=c)
            assert cfg.effective_num_columns >= prev_cols
            assert cfg.effective_num_rooms >= prev_rooms
            prev_cols = cfg.effective_num_columns
            prev_rooms = cfg.effective_num_rooms

    def test_gap_height_decreases_with_complexity(self):
        """Harder = narrower gaps."""
        low = WorldConfig(complexity=0.0)
        high = WorldConfig(complexity=1.0)
        assert low.effective_gap_height > high.effective_gap_height

    def test_style_derived_from_complexity(self):
        low = WorldConfig(complexity=0.0)
        high = WorldConfig(complexity=1.0)
        assert low.effective_style.fog_density < high.effective_style.fog_density
        assert low.effective_style.wall_color_saturation < high.effective_style.wall_color_saturation

    def test_custom_style_override(self):
        style = VisualStyle(fog_density=0.9, wall_color_saturation=0.5)
        cfg = WorldConfig(complexity=0.0, style=style)
        assert cfg.effective_style.fog_density == 0.9

    def test_manual_overrides(self):
        cfg = WorldConfig(
            complexity=0.5,
            num_columns=3,
            gap_height=1.0,
            num_rooms=2,
            num_branches=1,
            num_levels=3,
            debris_density=0.8,
        )
        assert cfg.effective_num_columns == 3
        assert cfg.effective_gap_height == 1.0
        assert cfg.effective_num_rooms == 2
        assert cfg.effective_num_branches == 1
        assert cfg.effective_num_levels == 3
        assert cfg.effective_debris_density == 0.8


class TestGenerateWorld:
    """Tests for generate_world() procedural generation."""

    def test_basic_generation(self):
        cfg = WorldConfig(complexity=0.1, seed=42)
        boxes = generate_world(cfg)
        assert len(boxes) > 0
        assert all(isinstance(b, BBox3D) for b in boxes)

    def test_type_check(self):
        with pytest.raises(TypeError, match="WorldConfig"):
            generate_world("not a config")

    def test_seed_reproducibility(self):
        cfg = WorldConfig(complexity=0.5, seed=42)
        boxes1 = generate_world(cfg)
        boxes2 = generate_world(cfg)
        assert len(boxes1) == len(boxes2)
        for b1, b2 in zip(boxes1, boxes2):
            assert b1.center == b2.center
            assert b1.extent == b2.extent
            assert b1.label == b2.label

    def test_different_seeds_differ(self):
        boxes1 = generate_world(WorldConfig(complexity=0.5, seed=1))
        boxes2 = generate_world(WorldConfig(complexity=0.5, seed=2))
        # Different seeds should produce different obstacle positions
        if len(boxes1) == len(boxes2) and len(boxes1) > 4:
            any_diff = any(
                b1.center != b2.center
                for b1, b2 in zip(boxes1, boxes2)
                if "floor" not in b1.label and "ceiling" not in b1.label
            )
            assert any_diff

    def test_complexity_scaling(self):
        """Higher complexity produces more boxes."""
        low = generate_world(WorldConfig(complexity=0.1, seed=42))
        high = generate_world(WorldConfig(complexity=0.9, seed=42))
        assert len(high) > len(low)

    def test_all_presets_generate(self):
        """All presets produce valid worlds without errors."""
        for name in ["flappy", "corridor", "rooms", "branches", "maze", "doom"]:
            cfg = getattr(WorldConfig, name)(seed=42)
            boxes = generate_world(cfg)
            assert len(boxes) > 0, f"{name} produced no boxes"

    def test_labels_unique_enough(self):
        """Most labels should be distinct (some repeats OK for floor/ceiling)."""
        cfg = WorldConfig(complexity=0.5, seed=42)
        boxes = generate_world(cfg)
        labels = [b.label for b in boxes]
        # At minimum floor/ceiling labels exist
        assert any("floor" in l for l in labels)
        assert any("ceiling" in l for l in labels)
        # Most labels should be unique (allow ~30% duplicates for floor/ceiling/wall)
        assert len(set(labels)) >= len(labels) * 0.7

    def test_rooms_at_moderate_complexity(self):
        cfg = WorldConfig(complexity=0.5, seed=42)
        boxes = generate_world(cfg)
        labels = [b.label for b in boxes]
        assert any("room_" in l for l in labels)

    def test_branches_at_high_complexity(self):
        cfg = WorldConfig(complexity=0.7, seed=42)
        boxes = generate_world(cfg)
        labels = [b.label for b in boxes]
        assert any("branch_" in l for l in labels)

    def test_levels_at_very_high_complexity(self):
        cfg = WorldConfig(complexity=0.9, seed=42)
        boxes = generate_world(cfg)
        labels = [b.label for b in boxes]
        assert any("level_" in l for l in labels)

    def test_debris_at_moderate_complexity(self):
        cfg = WorldConfig(complexity=0.5, seed=42)
        boxes = generate_world(cfg)
        labels = [b.label for b in boxes]
        assert any("debris" in l for l in labels)


class TestWorldSummary:
    """Tests for world_summary()."""

    def test_returns_dict(self):
        cfg = WorldConfig(complexity=0.5, seed=42)
        s = world_summary(cfg)
        assert isinstance(s, dict)
        assert "complexity" in s
        assert "num_columns" in s
        assert "num_rooms" in s
        assert "debris_density" in s
        assert "fog_density" in s

    def test_values_match_config(self):
        cfg = WorldConfig(complexity=0.5, seed=42)
        s = world_summary(cfg)
        assert s["complexity"] == 0.5
        assert s["num_columns"] == cfg.effective_num_columns


class TestWorldToFrameMetadata:
    """Tests for world_to_frame_metadata()."""

    def test_basic(self):
        cfg = WorldConfig(complexity=0.3, seed=42)
        boxes = generate_world(cfg)
        meta = world_to_frame_metadata(cfg, boxes)
        assert meta["frame_id"] == 0
        assert "obstacles" in meta
        assert "depth_stats" in meta
        assert "traversability_ratio" in meta
        assert 0.0 <= meta["traversability_ratio"] <= 1.0

    def test_depth_stats_present(self):
        cfg = WorldConfig(complexity=0.5, seed=42)
        boxes = generate_world(cfg)
        meta = world_to_frame_metadata(cfg, boxes)
        ds = meta["depth_stats"]
        assert "min_m" in ds
        assert "max_m" in ds
        assert ds["min_m"] > 0
        assert ds["max_m"] > ds["min_m"]


class TestWorldGinOverrides:
    """Tests for world_gin_overrides()."""

    def test_basic(self):
        cfg = WorldConfig(complexity=0.5, seed=42)
        overrides = world_gin_overrides(cfg)
        assert "grid_coarsen" in overrides
        assert "object_count" in overrides
        assert "scatter_density_multiplier" in overrides
        assert "fog_density" in overrides
        assert "configure_render_cycles.exposure" in overrides

    def test_scatter_increases_with_complexity(self):
        low = world_gin_overrides(WorldConfig(complexity=0.0))
        high = world_gin_overrides(WorldConfig(complexity=1.0))
        assert high["scatter_density_multiplier"] > low["scatter_density_multiplier"]

    def test_grid_coarsen_decreases_with_complexity(self):
        low = world_gin_overrides(WorldConfig(complexity=0.0))
        high = world_gin_overrides(WorldConfig(complexity=1.0))
        assert high["grid_coarsen"] <= low["grid_coarsen"]


class TestEndToEndWorldGenPipeline:
    """End-to-end: WorldConfig → generate → gin overrides → metadata."""

    def test_progressive_difficulty_pipeline(self):
        """Simulate a full difficulty progression."""
        for c in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            cfg = WorldConfig(complexity=c, seed=42)
            boxes = generate_world(cfg)
            assert len(boxes) > 0

            meta = world_to_frame_metadata(cfg, boxes)
            assert meta["curriculum_stage"] == round(c * 10)

            gin = world_gin_overrides(cfg)
            assert "grid_coarsen" in gin

    def test_separation_of_concerns(self):
        """World gen produces pure BBox3D — no external types."""
        cfg = WorldConfig(complexity=0.5, seed=42)
        boxes = generate_world(cfg)
        assert all(isinstance(b, BBox3D) for b in boxes)

        # Gin overrides are Infinigen-specific
        gin = world_gin_overrides(cfg)
        assert "grid_coarsen" in gin  # Infinigen parameter

    def test_complexity_increases_box_count(self):
        """Higher complexity produces more geometry."""
        boxes_low = generate_world(WorldConfig(complexity=0.1, seed=42))
        boxes_high = generate_world(WorldConfig(complexity=0.9, seed=42))
        assert len(boxes_high) > len(boxes_low)

    def test_world_to_frame_metadata(self):
        """World gen → frame metadata with obstacle info."""
        cfg = WorldConfig(complexity=0.5, seed=42)
        boxes = generate_world(cfg)
        meta = world_to_frame_metadata(cfg, boxes)
        assert "obstacles" in meta
        assert "depth_stats" in meta
        assert "curriculum_stage" in meta


# ═══════════════════════════════════════════════════════════════════════════
#  InfinigenOverlayHints — asset category & render quality per stage
# ═══════════════════════════════════════════════════════════════════════════


class TestInfinigenOverlayHints:
    """Tests for InfinigenOverlayHints — what Infinigen renders at each stage."""

    def test_from_complexity_low(self):
        """Very low complexity → flat corridor, no assets."""
        h = InfinigenOverlayHints.from_complexity(0.0)
        assert h.environment_type == "corridor"
        assert h.material_complexity == "flat"
        assert not h.enabled_vegetation
        assert not h.enabled_furniture
        assert not h.enabled_vehicles
        assert not h.enabled_dynamic_objects
        assert h.texture_resolution == 256

    def test_from_complexity_mid_low(self):
        """Low-mid complexity → basic PBR, still corridor."""
        h = InfinigenOverlayHints.from_complexity(0.25)
        assert h.environment_type == "corridor"
        assert h.material_complexity == "basic_pbr"
        assert h.lighting_complexity == "single_sun"
        assert h.texture_resolution == 512

    def test_from_complexity_mid(self):
        """Mid complexity → indoor with furniture."""
        h = InfinigenOverlayHints.from_complexity(0.45)
        assert h.environment_type == "indoor"
        assert h.enabled_furniture
        assert h.lighting_complexity == "multi_light"
        assert h.texture_resolution == 1024

    def test_from_complexity_mid_high(self):
        """Mid-high → mixed, vegetation, dynamic objects, weather."""
        h = InfinigenOverlayHints.from_complexity(0.65)
        assert h.environment_type == "mixed"
        assert h.enabled_vegetation
        assert h.enabled_dynamic_objects
        assert h.enabled_weather
        assert h.material_complexity == "full_pbr"

    def test_from_complexity_high(self):
        """High → outdoor, vehicles, pedestrians, HDR."""
        h = InfinigenOverlayHints.from_complexity(0.85)
        assert h.environment_type == "outdoor_street"
        assert h.enabled_vehicles
        assert h.enabled_pedestrians
        assert h.lighting_complexity == "hdr_environment"
        assert h.texture_resolution == 2048

    def test_from_complexity_max(self):
        """Maximum → full photorealism, subsurface, 4K textures."""
        h = InfinigenOverlayHints.from_complexity(1.0)
        assert h.material_complexity == "subsurface"
        assert h.texture_resolution == 4096
        assert h.subdiv_level == 3
        assert h.enabled_vegetation
        assert h.enabled_vehicles
        assert h.enabled_dynamic_objects
        assert h.enabled_weather
        assert h.enabled_pedestrians

    def test_progressive_texture_resolution(self):
        """Texture resolution increases monotonically with complexity."""
        prev_resolution = 0
        for c in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            h = InfinigenOverlayHints.from_complexity(c)
            assert h.texture_resolution >= prev_resolution
            prev_resolution = h.texture_resolution

    def test_progressive_subdiv_level(self):
        """Subdiv level increases monotonically with complexity."""
        prev_subdiv = 0
        for c in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            h = InfinigenOverlayHints.from_complexity(c)
            assert h.subdiv_level >= prev_subdiv
            prev_subdiv = h.subdiv_level

    def test_to_gin_hints(self):
        """to_gin_hints() returns a dict with all overlay keys."""
        h = InfinigenOverlayHints.from_complexity(0.5)
        gin = h.to_gin_hints()
        assert "environment_type" in gin
        assert "texture_resolution" in gin
        assert "enable_vegetation" in gin
        assert "enable_furniture" in gin
        assert "material_complexity" in gin

    def test_worldconfig_overlay_hints_property(self):
        """WorldConfig.overlay_hints matches from_complexity(c)."""
        cfg = WorldConfig(complexity=0.7, seed=42)
        h = cfg.overlay_hints
        expected = InfinigenOverlayHints.from_complexity(0.7)
        assert h.environment_type == expected.environment_type
        assert h.texture_resolution == expected.texture_resolution
        assert h.enabled_vegetation == expected.enabled_vegetation

    def test_gin_overrides_include_overlay(self):
        """world_gin_overrides() includes overlay hints."""
        cfg = WorldConfig(complexity=0.8, seed=42)
        overrides = world_gin_overrides(cfg)
        assert "environment_type" in overrides
        assert "texture_resolution" in overrides
        assert "enable_vegetation" in overrides
        assert overrides["enable_vegetation"] is True

    def test_world_summary_includes_overlay(self):
        """world_summary() includes overlay section."""
        cfg = WorldConfig.doom(seed=42)
        s = world_summary(cfg)
        assert "overlay" in s
        assert "stage_description" in s["overlay"]
        assert "environment_type" in s["overlay"]

    def test_validation_invalid_env_type(self):
        """Invalid environment_type raises ValueError."""
        with pytest.raises(ValueError, match="environment_type"):
            InfinigenOverlayHints(environment_type="spaceship")

    def test_validation_invalid_material(self):
        """Invalid material_complexity raises ValueError."""
        with pytest.raises(ValueError, match="material_complexity"):
            InfinigenOverlayHints(material_complexity="raytraced")

    def test_validation_invalid_lighting(self):
        """Invalid lighting_complexity raises ValueError."""
        with pytest.raises(ValueError, match="lighting_complexity"):
            InfinigenOverlayHints(lighting_complexity="neon")

    def test_validation_negative_texture_res(self):
        """Negative texture_resolution raises ValueError."""
        with pytest.raises(ValueError, match="texture_resolution"):
            InfinigenOverlayHints(texture_resolution=0)

    def test_validation_negative_subdiv(self):
        """Negative subdiv_level raises ValueError."""
        with pytest.raises(ValueError, match="subdiv_level"):
            InfinigenOverlayHints(subdiv_level=-1)

    def test_clamp_complexity_bounds(self):
        """from_complexity clamps values outside [0, 1]."""
        h_low = InfinigenOverlayHints.from_complexity(-0.5)
        assert h_low.environment_type == "corridor"
        h_high = InfinigenOverlayHints.from_complexity(1.5)
        assert h_high.material_complexity == "subsurface"


class TestOverlayHintsForComplexity:
    """Tests for the overlay_hints_for_complexity convenience function."""

    def test_returns_hints(self):
        h = overlay_hints_for_complexity(0.5)
        assert isinstance(h, InfinigenOverlayHints)

    def test_matches_from_complexity(self):
        for c in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            h1 = overlay_hints_for_complexity(c)
            h2 = InfinigenOverlayHints.from_complexity(c)
            assert h1 == h2

    def test_low_complexity(self):
        h = overlay_hints_for_complexity(0.05)
        assert h.environment_type == "corridor"
        assert h.material_complexity == "flat"

    def test_high_complexity(self):
        h = overlay_hints_for_complexity(0.95)
        assert h.material_complexity == "subsurface"
        assert h.texture_resolution == 4096
