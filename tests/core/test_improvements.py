# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Tests for the 10 logic improvements: complexity, metrics, quality presets,
reproducibility, caching, budget, metadata, parallel stages, density scaling,
and validation.

All modules are pure Python and do not require bpy.
"""

from __future__ import annotations

import time

import pytest

from infinigen.core.syndata.budget import SceneBudget
from infinigen.core.syndata.caching import StageCache
from infinigen.core.syndata.complexity import (
    ComplexityLevel,
    ComplexityParams,
    ComplexityScheduler,
    get_complexity_params,
    get_gin_overrides,
    interpolate_params,
)
from infinigen.core.syndata.density_scaling import (
    density_ramp,
    resolution_for_level,
    scaled_density,
    species_count_for_level,
)
from infinigen.core.syndata.metadata import (
    MetadataCollector,
    ObjectRecord,
    SceneMetadata,
)
from infinigen.core.syndata.metrics import DatasetDiversityStats, SceneMetrics
from infinigen.core.syndata.parallel_stages import (
    ParallelStageExecutor,
    StageSpec,
)
from infinigen.core.syndata.quality_presets import (
    QualityConfig,
    QualityPreset,
    get_quality_config,
    quality_gin_overrides,
)
from infinigen.core.syndata.reproducibility import (
    SeedRegistry,
    component_seed_context,
)
from infinigen.core.syndata.validation import (
    Severity,
    ValidationReport,
    ValidationResult,
    check_camera_position,
    check_object_count,
    check_polygon_budget,
    check_resolution,
    check_seed,
    validate_scene_config,
)

# ── 1. Complexity Controller ─────────────────────────────────────────────────


class TestComplexityController:
    def test_all_levels_defined(self):
        for level in range(1, 6):
            params = get_complexity_params(level)
            assert isinstance(params, ComplexityParams)

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError, match="Complexity level"):
            get_complexity_params(0)
        with pytest.raises(ValueError, match="Complexity level"):
            get_complexity_params(6)

    def test_monotonic_density(self):
        """Higher complexity levels should have >= density."""
        densities = [get_complexity_params(lv).tree_density for lv in range(1, 6)]
        for i in range(len(densities) - 1):
            assert densities[i] <= densities[i + 1]

    def test_monotonic_samples(self):
        samples = [get_complexity_params(lv).render_num_samples for lv in range(1, 6)]
        for i in range(len(samples) - 1):
            assert samples[i] <= samples[i + 1]

    def test_gin_overrides_are_strings(self):
        overrides = get_gin_overrides(3)
        assert isinstance(overrides, list)
        assert all(isinstance(o, str) for o in overrides)
        assert len(overrides) > 0

    def test_enum_values(self):
        assert ComplexityLevel.MINIMAL == 1
        assert ComplexityLevel.FULL == 5

    def test_interpolate_endpoints(self):
        low_params = get_complexity_params(1)
        result_0 = interpolate_params(1, 2, 0.0)
        assert result_0["tree_density"] == low_params.tree_density

        high_params = get_complexity_params(2)
        result_1 = interpolate_params(1, 2, 1.0)
        assert result_1["tree_density"] == high_params.tree_density

    def test_interpolate_midpoint(self):
        result = interpolate_params(1, 5, 0.5)
        low = get_complexity_params(1)
        high = get_complexity_params(5)
        expected_density = (low.tree_density + high.tree_density) / 2
        assert abs(result["tree_density"] - expected_density) < 1e-9

    def test_interpolate_invalid_t(self):
        with pytest.raises(ValueError, match="Interpolation factor"):
            interpolate_params(1, 2, -0.1)
        with pytest.raises(ValueError, match="Interpolation factor"):
            interpolate_params(1, 2, 1.1)

    def test_interpolate_invalid_levels(self):
        with pytest.raises(ValueError, match="level_low"):
            interpolate_params(3, 2, 0.5)


class TestComplexityScheduler:
    def test_initial_state(self):
        s = ComplexityScheduler(start_level=1, max_level=5, episodes_per_level=10)
        assert s.current_level == 1

    def test_step_advances(self):
        s = ComplexityScheduler(start_level=1, max_level=5, episodes_per_level=2)
        s.step()  # episode 1
        assert s.current_level == 1
        s.step()  # episode 2 => advance
        assert s.current_level == 2

    def test_does_not_exceed_max(self):
        s = ComplexityScheduler(start_level=4, max_level=5, episodes_per_level=1)
        s.step()  # => 5
        assert s.current_level == 5
        s.step()  # should stay at 5
        assert s.current_level == 5

    def test_reset(self):
        s = ComplexityScheduler(start_level=1, max_level=5, episodes_per_level=1)
        s.step()
        s.step()
        assert s.current_level > 1
        s.reset()
        assert s.current_level == 1

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            ComplexityScheduler(start_level=0)
        with pytest.raises(ValueError):
            ComplexityScheduler(start_level=3, max_level=2)
        with pytest.raises(ValueError):
            ComplexityScheduler(episodes_per_level=0)

    def test_current_params(self):
        s = ComplexityScheduler(start_level=3, max_level=5, episodes_per_level=10)
        params = s.current_params
        assert isinstance(params, ComplexityParams)
        assert params == get_complexity_params(3)


# ── 2. Scene Metrics ─────────────────────────────────────────────────────────


class TestSceneMetrics:
    def test_default_score_zero(self):
        m = SceneMetrics()
        assert m.complexity_score == 0.0

    def test_score_increases_with_objects(self):
        m1 = SceneMetrics(object_count=10)
        m2 = SceneMetrics(object_count=100)
        assert m2.complexity_score > m1.complexity_score

    def test_score_capped_at_one(self):
        m = SceneMetrics(
            object_count=100_000,
            unique_material_count=1_000,
            total_polygon_count=100_000_000,
            total_vertex_count=50_000_000,
            scatter_instance_count=1_000_000,
            light_count=100,
            max_object_depth=100,
        )
        assert m.complexity_score <= 1.0

    def test_to_dict(self):
        m = SceneMetrics(object_count=5)
        d = m.to_dict()
        assert d["object_count"] == 5
        assert "complexity_score" in d
        assert "_complexity_score" not in d


class TestDatasetDiversityStats:
    def test_empty_list(self):
        stats = DatasetDiversityStats.from_metrics_list([])
        assert stats.scene_count == 0

    def test_single_scene(self):
        metrics = [SceneMetrics(object_count=10)]
        stats = DatasetDiversityStats.from_metrics_list(metrics)
        assert stats.scene_count == 1
        assert stats.mean_object_count == 10.0
        assert stats.std_object_count == 0.0

    def test_multiple_scenes(self):
        metrics = [
            SceneMetrics(object_count=10, total_polygon_count=1000),
            SceneMetrics(object_count=20, total_polygon_count=2000),
        ]
        stats = DatasetDiversityStats.from_metrics_list(metrics)
        assert stats.scene_count == 2
        assert stats.mean_object_count == 15.0
        assert stats.std_object_count > 0


# ── 3. Quality Presets ────────────────────────────────────────────────────────


class TestQualityPresets:
    def test_all_presets_exist(self):
        for preset in QualityPreset:
            cfg = get_quality_config(preset)
            assert isinstance(cfg, QualityConfig)

    def test_string_lookup(self):
        cfg = get_quality_config("preview")
        assert cfg.num_samples == 16

    def test_invalid_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown quality preset"):
            get_quality_config("nonexistent")

    def test_quality_order(self):
        """Higher quality should mean more samples."""
        levels = [
            QualityPreset.PREVIEW, QualityPreset.FAST, QualityPreset.MEDIUM,
            QualityPreset.HIGH, QualityPreset.ULTRA,
        ]
        samples = [get_quality_config(p).num_samples for p in levels]
        for i in range(len(samples) - 1):
            assert samples[i] <= samples[i + 1]

    def test_gin_overrides(self):
        overrides = quality_gin_overrides("fast")
        assert isinstance(overrides, list)
        assert any("num_samples" in o for o in overrides)


# ── 4. Reproducibility ───────────────────────────────────────────────────────


class TestSeedRegistry:
    def test_deterministic(self):
        r1 = SeedRegistry(base_seed=42)
        r2 = SeedRegistry(base_seed=42)
        assert r1.get("terrain") == r2.get("terrain")

    def test_different_components_differ(self):
        r = SeedRegistry(base_seed=42)
        assert r.get("terrain") != r.get("vegetation")

    def test_different_base_seeds_differ(self):
        r1 = SeedRegistry(base_seed=42)
        r2 = SeedRegistry(base_seed=99)
        assert r1.get("terrain") != r2.get("terrain")

    def test_pin_overrides(self):
        r = SeedRegistry(base_seed=42)
        original = r.get("terrain")
        r.pin("terrain", 12345)
        assert r.get("terrain") == 12345
        r.unpin("terrain")
        assert r.get("terrain") == original

    def test_access_log(self):
        r = SeedRegistry(base_seed=42)
        r.get("a")
        r.get("b")
        r.get("a")
        assert r.access_log == ["a", "b", "a"]
        r.clear_log()
        assert r.access_log == []

    def test_empty_component_raises(self):
        r = SeedRegistry(base_seed=42)
        with pytest.raises(ValueError, match="non-empty"):
            r.get("")

    def test_invalid_base_seed_type(self):
        with pytest.raises(TypeError):
            SeedRegistry(base_seed="hello")

    def test_context_manager(self):
        r = SeedRegistry(base_seed=42)
        with component_seed_context(r, "terrain") as seed:
            assert isinstance(seed, int)


# ── 5. Stage Caching ─────────────────────────────────────────────────────────


class TestStageCache:
    def test_put_and_get(self, tmp_path):
        cache = StageCache(tmp_path / "cache")
        cache.put("terrain", {"seed": 42}, {"heightmap": [1, 2, 3]})
        result = cache.get("terrain", {"seed": 42})
        assert result == {"heightmap": [1, 2, 3]}
        assert cache.hit_count == 1

    def test_miss_on_different_params(self, tmp_path):
        cache = StageCache(tmp_path / "cache")
        cache.put("terrain", {"seed": 42}, "result_42")
        result = cache.get("terrain", {"seed": 99})
        assert cache.is_miss(result)
        assert cache.miss_count == 1

    def test_disabled_cache(self, tmp_path):
        cache = StageCache(tmp_path / "cache", enabled=False)
        cache.put("terrain", {}, "data")
        result = cache.get("terrain", {})
        assert cache.is_miss(result)

    def test_invalidate_all(self, tmp_path):
        cache = StageCache(tmp_path / "cache")
        cache.put("a", {}, "data_a")
        cache.put("b", {}, "data_b")
        removed = cache.invalidate()
        assert removed == 2
        assert cache.is_miss(cache.get("a", {}))

    def test_invalidate_specific(self, tmp_path):
        cache = StageCache(tmp_path / "cache")
        cache.put("a", {}, "data_a")
        cache.put("b", {}, "data_b")
        removed = cache.invalidate("a")
        assert removed == 1
        assert not cache.is_miss(cache.get("b", {}))


# ── 6. Scene Budget ──────────────────────────────────────────────────────────


class TestSceneBudget:
    def test_basic_allocation(self):
        b = SceneBudget(max_polygons=1000, max_objects=10)
        assert b.try_allocate(polygons=500, objects=5)
        assert b.used_polygons == 500
        assert b.used_objects == 5

    def test_over_budget_rejected(self):
        b = SceneBudget(max_polygons=1000, max_objects=10)
        assert b.try_allocate(polygons=500)
        assert not b.try_allocate(polygons=600)  # 500 + 600 > 1000
        assert b.used_polygons == 500  # unchanged

    def test_rejection_count(self):
        b = SceneBudget(max_polygons=100)
        b.try_allocate(polygons=200)
        assert b.rejection_count == 1

    def test_remaining(self):
        b = SceneBudget(max_polygons=1000, max_objects=10)
        b.try_allocate(polygons=300, objects=3)
        assert b.remaining_polygons() == 700
        assert b.remaining_objects() == 7

    def test_release(self):
        b = SceneBudget(max_polygons=1000)
        b.try_allocate(polygons=600)
        b.release(polygons=200)
        assert b.used_polygons == 400

    def test_utilisation(self):
        b = SceneBudget(max_polygons=1000, max_objects=100)
        b.try_allocate(polygons=500, objects=25)
        u = b.utilisation()
        assert abs(u["polygons"] - 0.5) < 1e-9
        assert abs(u["objects"] - 0.25) < 1e-9

    def test_reset(self):
        b = SceneBudget(max_polygons=1000)
        b.try_allocate(polygons=500)
        b.reset()
        assert b.used_polygons == 0
        assert b.rejection_count == 0

    def test_can_allocate_no_side_effect(self):
        b = SceneBudget(max_polygons=1000)
        assert b.can_allocate(polygons=500)
        assert b.used_polygons == 0  # should not modify state

    def test_force_allocate_exceeds_limit(self):
        b = SceneBudget(max_polygons=100)
        b.force_allocate(polygons=200)
        assert b.used_polygons == 200  # allowed to exceed


# ── 7. Metadata ──────────────────────────────────────────────────────────────


class TestSceneMetadataModule:
    def test_add_object(self):
        m = SceneMetadata(scene_seed=42)
        m.add_object(ObjectRecord(name="Tree_0", category="vegetation", polygon_count=500))
        assert m.object_count == 1
        assert m.total_polygon_count == 500

    def test_categories(self):
        m = SceneMetadata()
        m.add_object(ObjectRecord(category="vegetation"))
        m.add_object(ObjectRecord(category="furniture"))
        m.add_object(ObjectRecord(category="vegetation"))
        assert m.categories == {"vegetation", "furniture"}

    def test_json_round_trip(self, tmp_path):
        m = SceneMetadata(scene_seed=42, complexity_level=3)
        m.add_object(ObjectRecord(name="Rock_0", category="rock", polygon_count=100))
        path = tmp_path / "meta.json"
        m.save_json(path)
        loaded = SceneMetadata.load_json(path)
        assert loaded.scene_seed == 42
        assert loaded.object_count == 1

    def test_to_dict(self):
        m = SceneMetadata(scene_seed=1)
        d = m.to_dict()
        assert d["scene_seed"] == 1
        assert isinstance(d["objects"], list)


class TestMetadataCollector:
    def test_finalise_sets_time(self):
        c = MetadataCollector(scene_seed=42)
        time.sleep(0.01)
        meta = c.finalise()
        assert meta.generation_time_s > 0

    def test_record_objects_and_stages(self):
        c = MetadataCollector()
        c.record_object(ObjectRecord(name="T", polygon_count=100))
        c.record_stage_timing("terrain", 1.5)
        c.set_extra("note", "test")
        meta = c.finalise()
        assert meta.object_count == 1
        assert meta.stage_timings["terrain"] == 1.5
        assert meta.extra["note"] == "test"


# ── 8. Parallel Stages ───────────────────────────────────────────────────────


class TestParallelStageExecutor:
    def test_independent_stages(self):
        def add(a, b):
            return a + b

        stages = [
            StageSpec(name="add_1", fn=add, args=(1, 2)),
            StageSpec(name="add_2", fn=add, args=(3, 4)),
        ]
        exe = ParallelStageExecutor(max_workers=2)
        outcomes = exe.run(stages)
        results = {o.name: o.result for o in outcomes}
        assert results["add_1"] == 3
        assert results["add_2"] == 7

    def test_dependency_ordering(self):
        """Stage B depends on Stage A completing first."""
        order = []

        def stage_a():
            order.append("a")
            return 10

        def stage_b():
            order.append("b")
            return 20

        stages = [
            StageSpec(name="a", fn=stage_a),
            StageSpec(name="b", fn=stage_b, depends_on=["a"]),
        ]
        exe = ParallelStageExecutor(max_workers=1)
        outcomes = exe.run(stages)
        assert order == ["a", "b"]

    def test_error_captured(self):
        def fail():
            raise RuntimeError("boom")

        stages = [StageSpec(name="fail", fn=fail)]
        exe = ParallelStageExecutor(max_workers=1)
        outcomes = exe.run(stages)
        assert not outcomes[0].success
        assert "boom" in str(outcomes[0].error)

    def test_unknown_dependency_raises(self):
        stages = [StageSpec(name="a", fn=lambda: 1, depends_on=["nonexistent"])]
        exe = ParallelStageExecutor(max_workers=1)
        with pytest.raises(ValueError, match="unknown stage"):
            exe.run(stages)

    def test_empty_stages(self):
        exe = ParallelStageExecutor(max_workers=1)
        outcomes = exe.run([])
        assert outcomes == []

    def test_invalid_max_workers(self):
        with pytest.raises(ValueError, match="max_workers"):
            ParallelStageExecutor(max_workers=0)


# ── 9. Density Scaling ───────────────────────────────────────────────────────


class TestDensityScaling:
    def test_full_level_no_scaling(self):
        assert scaled_density(0.12, 5) == pytest.approx(0.12)

    def test_minimal_level_zero(self):
        assert scaled_density(0.12, 1) == 0.0

    def test_density_ramp_clamp(self):
        val = density_ramp(0.12, 5, floor=0.05, ceiling=0.10)
        assert val == 0.10

    def test_species_count(self):
        assert species_count_for_level(6, 1) == 0
        assert species_count_for_level(6, 5) == 6

    def test_resolution_even(self):
        for level in range(1, 6):
            x, y = resolution_for_level(1920, 1080, level)
            assert x % 2 == 0
            assert y % 2 == 0
            assert x >= 2 and y >= 2


# ── 10. Validation ───────────────────────────────────────────────────────────


class TestValidation:
    def test_empty_scene_warning(self):
        r = check_object_count({"object_count": 0})
        assert r.severity == Severity.WARNING

    def test_normal_object_count(self):
        r = check_object_count({"object_count": 50})
        assert r.severity == Severity.PASS

    def test_excessive_polygons(self):
        r = check_polygon_budget({"total_polygon_count": 20_000_000})
        assert r.severity == Severity.ERROR

    def test_good_resolution(self):
        r = check_resolution({"resolution": (1920, 1080)})
        assert r.severity == Severity.PASS

    def test_tiny_resolution(self):
        r = check_resolution({"resolution": (10, 10)})
        assert r.severity == Severity.ERROR

    def test_missing_seed_warning(self):
        r = check_seed({})
        assert r.severity == Severity.WARNING

    def test_valid_seed(self):
        r = check_seed({"scene_seed": 42})
        assert r.severity == Severity.PASS

    def test_camera_in_bounds(self):
        r = check_camera_position({"camera_position": (0, 0, 10)})
        assert r.severity == Severity.PASS

    def test_camera_out_of_bounds(self):
        r = check_camera_position({"camera_position": (0, 0, 99999)})
        assert r.severity == Severity.WARNING

    def test_aggregate_validator_passes(self):
        config = {
            "object_count": 50,
            "total_polygon_count": 100_000,
            "resolution": (1280, 720),
            "scene_seed": 42,
        }
        report = validate_scene_config(config)
        assert report.passed

    def test_aggregate_validator_with_errors(self):
        config = {
            "object_count": 50,
            "total_polygon_count": 100_000_000,
            "resolution": (1, 1),
        }
        report = validate_scene_config(config)
        assert not report.passed
        assert report.error_count >= 1

    def test_custom_check(self):
        def custom_check(config):
            return ValidationResult("custom", Severity.WARNING, "custom warning")

        report = validate_scene_config({}, extra_checks=[custom_check])
        assert report.warning_count >= 1

    def test_report_summary(self):
        report = ValidationReport()
        report.add(ValidationResult("test", Severity.PASS, "ok"))
        report.add(ValidationResult("bad", Severity.ERROR, "fail"))
        summary = report.summary()
        assert "1 error" in summary
        assert "bad" in summary


# ── Edge Case Tests ──────────────────────────────────────────────────────────


class TestEdgeCases:
    """Additional edge case tests identified during code review."""

    def test_corrupt_cache_handled(self, tmp_path):
        """Verify corrupt pickle files are handled gracefully."""
        cache = StageCache(tmp_path / "cache")
        # Manually create a corrupt pickle
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(exist_ok=True)
        corrupt_file = cache_dir / "test_corrupt_bad.pkl"
        corrupt_file.write_text("this is not valid pickle data")
        # Access via internal path should tolerate corrupt data
        result = cache.get("test_corrupt", {"bad": True})
        assert cache.is_miss(result)

    def test_budget_negative_release(self):
        """Verify releasing more than allocated is safe."""
        b = SceneBudget(max_polygons=1000)
        b.try_allocate(polygons=100)
        b.release(polygons=200)  # over-release
        assert b.used_polygons == 0

    def test_seed_registry_unicode(self):
        """Verify non-ASCII component names work."""
        r = SeedRegistry(base_seed=42)
        seed = r.get("terrain_\u6a39")  # 樹 = tree in Chinese
        assert isinstance(seed, int)

    def test_metadata_load_unknown_fields(self, tmp_path):
        """Verify load_json tolerates extra fields from future schema."""
        import json

        path = tmp_path / "meta.json"
        data = {
            "scene_seed": 42,
            "complexity_level": 3,
            "quality_preset": "medium",
            "generation_time_s": 0.0,
            "objects": [],
            "gin_overrides": [],
            "stage_timings": {},
            "extra": {},
            "unknown_future_field": "should be ignored",
        }
        with open(path, "w") as f:
            json.dump(data, f)
        loaded = SceneMetadata.load_json(path)
        assert loaded.scene_seed == 42

    def test_gin_overrides_complete(self):
        """Verify all ComplexityParams fields are reflected in gin overrides."""
        overrides = get_gin_overrides(3)
        override_text = " ".join(overrides)
        assert "max_tree_species" in override_text
        assert "max_bush_species" in override_text
        assert "tree_density" in override_text
        assert "num_samples" in override_text
        assert "terrain_erosion_enabled" in override_text
        assert "clouds_enabled" in override_text
        assert "weather_enabled" in override_text
        assert "creatures_enabled" in override_text
        assert "scatter_density_multiplier" in override_text
        assert "resolution_scale" in override_text

    def test_validation_summary_grammar(self):
        """Verify singular/plural grammar in validation summary."""
        report = ValidationReport()
        report.add(ValidationResult("a", Severity.ERROR, "fail"))
        summary = report.summary()
        assert "1 error," in summary
        assert "0 warnings" in summary

        report2 = ValidationReport()
        report2.add(ValidationResult("a", Severity.ERROR, "f1"))
        report2.add(ValidationResult("b", Severity.ERROR, "f2"))
        report2.add(ValidationResult("c", Severity.WARNING, "w1"))
        summary2 = report2.summary()
        assert "2 errors," in summary2
        assert "1 warning" in summary2

    def test_parallel_executor_all_fail(self):
        """Verify executor handles all stages failing gracefully."""
        def fail_1():
            raise ValueError("fail_1")

        def fail_2():
            raise ValueError("fail_2")

        stages = [
            StageSpec(name="a", fn=fail_1),
            StageSpec(name="b", fn=fail_2),
        ]
        exe = ParallelStageExecutor(max_workers=2)
        outcomes = exe.run(stages)
        assert all(not o.success for o in outcomes)
        assert len(outcomes) == 2

    def test_quality_preset_case_insensitive(self):
        """Verify quality presets are case-insensitive."""
        cfg1 = get_quality_config("PREVIEW")
        cfg2 = get_quality_config("preview")
        assert cfg1 == cfg2
