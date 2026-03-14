# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Tests for infinigen.core.syndata — bpy-free curriculum-learning utilities.

Every test imports only from ``infinigen.core.syndata`` (which is pure
Python / NumPy) so that the suite runs in CI without Blender.
"""


import numpy as np
import pytest

# ── Module imports (all bpy-free) ──────────────────────────────────────────
from infinigen.core.syndata.complexity import CurriculumConfig, curriculum_overrides
from infinigen.core.syndata.density_scaling import DensityScaler
from infinigen.core.syndata.metadata import BBox3D, DepthStats, FrameMetadata
from infinigen.core.syndata.metrics import SceneBudget
from infinigen.core.syndata.parallel_stages import Stage, StageGraph
from infinigen.core.syndata.quality_presets import VALID_PRESETS, drone_preset
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


class TestCurriculumOverrides:
    def test_returns_dict(self):
        cfg = CurriculumConfig(stage=5, total_stages=10)
        overrides = curriculum_overrides(cfg)
        assert isinstance(overrides, dict)
        assert "compose_scene.grid_coarsen" in overrides
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
        assert "scatter.density_multiplier" in overrides
        assert "compose_scene.obstacle_count" in overrides


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
