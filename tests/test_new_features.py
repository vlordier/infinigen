# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""
Tests for newly added utility features.

These tests are designed to run **without bpy** using ``--noconftest``.

Run with::

    pytest tests/test_new_features.py --noconftest -v
"""

import importlib
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers – load modules without triggering bpy
# ---------------------------------------------------------------------------

_UTIL_DIR = Path(__file__).resolve().parent.parent / "infinigen" / "core" / "util"
_TOOLS_DIR = Path(__file__).resolve().parent.parent / "infinigen" / "tools"


def _load_module(name: str, filepath: Path):
    """Load a single Python file as a module, bypassing package __init__."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _get_retry():
    return _load_module("retry", _UTIL_DIR / "retry.py")


def _get_health_check():
    return _load_module("health_check", _UTIL_DIR / "health_check.py")


def _get_dataset_integrity():
    return _load_module("dataset_integrity", _TOOLS_DIR / "dataset_integrity.py")


def _get_config_validation():
    return _load_module("config_validation", _UTIL_DIR / "config_validation.py")


# ═══════════════════════════════════════════════════════════════════════════
# 1.  retry.py — Exponential backoff
# ═══════════════════════════════════════════════════════════════════════════


class TestRetry:
    """Tests for the retry decorator and helpers."""

    def test_succeeds_first_try(self):
        mod = _get_retry()
        call_count = 0

        @mod.retry(max_attempts=3, base_delay=0, jitter=False)
        def succeeds():
            nonlocal call_count
            call_count += 1
            return 42

        assert succeeds() == 42
        assert call_count == 1

    def test_retries_on_failure(self):
        mod = _get_retry()
        call_count = 0

        @mod.retry(max_attempts=3, base_delay=0, jitter=False)
        def fails_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "ok"

        assert fails_twice() == "ok"
        assert call_count == 3

    def test_raises_max_retries_exceeded(self):
        mod = _get_retry()

        @mod.retry(max_attempts=2, base_delay=0, jitter=False)
        def always_fails():
            raise RuntimeError("boom")

        with pytest.raises(mod.MaxRetriesExceeded) as exc_info:
            always_fails()
        assert exc_info.value.attempts == 2
        assert isinstance(exc_info.value.last_exception, RuntimeError)

    def test_respects_retryable_filter(self):
        mod = _get_retry()

        @mod.retry(
            max_attempts=5,
            base_delay=0,
            jitter=False,
            retryable=(ValueError,),
        )
        def raises_type_error():
            raise TypeError("non-retryable")

        # TypeError is not in retryable, so it should propagate immediately
        with pytest.raises(TypeError, match="non-retryable"):
            raises_type_error()

    def test_on_retry_callback(self):
        mod = _get_retry()
        callbacks = []

        @mod.retry(
            max_attempts=3,
            base_delay=0,
            jitter=False,
            on_retry=lambda attempt, exc, delay: callbacks.append(attempt),
        )
        def fails_then_succeeds():
            if len(callbacks) < 2:
                raise ValueError("fail")
            return "done"

        result = fails_then_succeeds()
        assert result == "done"
        assert callbacks == [0, 1]

    def test_compute_delay_no_jitter(self):
        mod = _get_retry()
        assert mod.compute_delay(0, 1.0, 60.0, jitter=False) == 1.0
        assert mod.compute_delay(1, 1.0, 60.0, jitter=False) == 2.0
        assert mod.compute_delay(2, 1.0, 60.0, jitter=False) == 4.0
        # Capped at max_delay
        assert mod.compute_delay(10, 1.0, 60.0, jitter=False) == 60.0

    def test_compute_delay_with_jitter(self):
        mod = _get_retry()
        delays = [mod.compute_delay(2, 1.0, 60.0, jitter=True) for _ in range(100)]
        # All delays should be in [0, 4.0]
        assert all(0 <= d <= 4.0 for d in delays)
        # With 100 samples, we should see some variation
        assert len(set(round(d, 3) for d in delays)) > 1

    def test_invalid_max_attempts(self):
        mod = _get_retry()
        with pytest.raises(ValueError, match="max_attempts"):
            mod.retry(max_attempts=0)

    def test_invalid_base_delay(self):
        mod = _get_retry()
        with pytest.raises(ValueError, match="base_delay"):
            mod.retry(base_delay=-1)

    def test_invalid_max_delay(self):
        mod = _get_retry()
        with pytest.raises(ValueError, match="max_delay"):
            mod.retry(max_delay=0.5, base_delay=1.0)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  health_check.py — System prerequisites
# ═══════════════════════════════════════════════════════════════════════════


class TestHealthCheck:
    """Tests for system health check utilities."""

    def test_python_version_passes(self):
        mod = _get_health_check()
        result = mod.check_python_version(
            required_major=sys.version_info.major,
            required_minor=sys.version_info.minor,
        )
        assert result.status == mod.HealthStatus.PASS

    def test_python_version_fails_for_wrong_version(self):
        mod = _get_health_check()
        result = mod.check_python_version(required_major=2, required_minor=7)
        assert result.status == mod.HealthStatus.FAIL

    def test_disk_space_passes(self, tmp_path):
        mod = _get_health_check()
        result = mod.check_disk_space(tmp_path, min_gb=0.001)
        assert result.status in (mod.HealthStatus.PASS, mod.HealthStatus.WARN)

    def test_disk_space_fails_for_huge_requirement(self, tmp_path):
        mod = _get_health_check()
        result = mod.check_disk_space(tmp_path, min_gb=999999)
        assert result.status == mod.HealthStatus.FAIL

    def test_required_packages_pass(self):
        mod = _get_health_check()
        result = mod.check_required_packages(["sys", "os", "json"])
        assert result.status == mod.HealthStatus.PASS

    def test_required_packages_fail(self):
        mod = _get_health_check()
        result = mod.check_required_packages(["nonexistent_package_xyz123"])
        assert result.status == mod.HealthStatus.FAIL
        assert "nonexistent_package_xyz123" in result.message

    def test_gpu_check_returns_valid_status(self):
        mod = _get_health_check()
        result = mod.check_gpu_availability()
        # On CI without GPU, expect SKIP (no torch) or WARN (no GPU)
        assert result.status in (
            mod.HealthStatus.PASS,
            mod.HealthStatus.WARN,
            mod.HealthStatus.SKIP,
        )

    def test_output_directory_skip_when_none(self):
        mod = _get_health_check()
        result = mod.check_output_directory(None)
        assert result.status == mod.HealthStatus.SKIP

    def test_output_directory_creates_missing(self, tmp_path):
        mod = _get_health_check()
        new_dir = tmp_path / "new_output"
        result = mod.check_output_directory(new_dir)
        assert result.status == mod.HealthStatus.PASS
        assert new_dir.is_dir()

    def test_output_directory_writable(self, tmp_path):
        mod = _get_health_check()
        result = mod.check_output_directory(tmp_path)
        assert result.status == mod.HealthStatus.PASS

    def test_run_health_checks_returns_list(self):
        mod = _get_health_check()
        results = mod.run_health_checks()
        assert isinstance(results, list)
        assert len(results) >= 6  # python, disk, packages, gpu, blender, output

    def test_format_report(self):
        mod = _get_health_check()
        results = mod.run_health_checks()
        report = mod.format_report(results)
        assert "System Health Check Report" in report


# ═══════════════════════════════════════════════════════════════════════════
# 3.  dataset_integrity.py — Dataset validation
# ═══════════════════════════════════════════════════════════════════════════


class TestDatasetIntegrity:
    """Tests for dataset integrity checking."""

    def _make_scene(self, tmp_path, frames=None):
        """Create a minimal synthetic scene folder structure."""
        scene = tmp_path / "test_scene"
        frames_dir = scene / "frames"

        if frames is None:
            frames = [1, 2, 3, 4]

        # Create Image frames
        img_dir = frames_dir / "Image" / "camera_0"
        img_dir.mkdir(parents=True)
        for f in frames:
            p = img_dir / f"Image_0_0_{f:04d}_0.png"
            # Write a minimal valid PNG (just big enough)
            p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 60)

        return scene

    def test_frame_continuity_ok(self, tmp_path):
        mod = _get_dataset_integrity()
        scene = self._make_scene(tmp_path, frames=[1, 2, 3, 4])
        issues = mod.check_frame_continuity(scene)
        assert issues == []

    def test_frame_continuity_gap(self, tmp_path):
        mod = _get_dataset_integrity()
        scene = self._make_scene(tmp_path, frames=[1, 2, 5, 6])
        issues = mod.check_frame_continuity(scene)
        assert len(issues) >= 1
        assert "gap" in issues[0].lower() or "step" in issues[0].lower()

    def test_frame_continuity_missing_dir(self, tmp_path):
        mod = _get_dataset_integrity()
        issues = mod.check_frame_continuity(tmp_path / "nonexistent")
        assert len(issues) == 1
        assert "Missing" in issues[0]

    def test_file_sizes_ok(self, tmp_path):
        mod = _get_dataset_integrity()
        scene = self._make_scene(tmp_path)
        issues = mod.check_file_sizes(scene)
        assert issues == []

    def test_file_sizes_detects_empty(self, tmp_path):
        mod = _get_dataset_integrity()
        scene = self._make_scene(tmp_path)
        # Create an empty PNG file
        empty_png = scene / "frames" / "Image" / "camera_0" / "Image_0_0_9999_0.png"
        empty_png.write_bytes(b"")
        issues = mod.check_file_sizes(scene)
        assert len(issues) >= 1
        assert "too small" in issues[0].lower()

    def test_checksum_roundtrip(self, tmp_path):
        mod = _get_dataset_integrity()
        scene = self._make_scene(tmp_path)
        manifest = mod.generate_checksum_manifest(scene)
        assert len(manifest) > 0

        # Verify passes
        issues = mod.verify_checksum_manifest(scene)
        assert issues == []

    def test_checksum_detects_corruption(self, tmp_path):
        mod = _get_dataset_integrity()
        scene = self._make_scene(tmp_path)
        mod.generate_checksum_manifest(scene)

        # Corrupt a file
        img_dir = scene / "frames" / "Image" / "camera_0"
        first_file = next(img_dir.iterdir())
        first_file.write_bytes(b"CORRUPTED DATA" + b"\x00" * 60)

        issues = mod.verify_checksum_manifest(scene)
        assert len(issues) >= 1
        assert "mismatch" in issues[0].lower()

    def test_checksum_detects_missing_file(self, tmp_path):
        mod = _get_dataset_integrity()
        scene = self._make_scene(tmp_path)
        mod.generate_checksum_manifest(scene)

        # Remove a file
        img_dir = scene / "frames" / "Image" / "camera_0"
        first_file = next(img_dir.iterdir())
        first_file.unlink()

        issues = mod.verify_checksum_manifest(scene)
        assert len(issues) >= 1
        assert "missing" in issues[0].lower()

    def test_metadata_empty_scene(self, tmp_path):
        mod = _get_dataset_integrity()
        # No frames dir at all
        issues = mod.check_metadata_completeness(tmp_path)
        assert issues == []  # No metadata dirs = nothing to check

    def test_metadata_invalid_json(self, tmp_path):
        mod = _get_dataset_integrity()
        scene = tmp_path / "scene"
        obj_dir = scene / "frames" / "Objects" / "camera_0"
        obj_dir.mkdir(parents=True)
        (obj_dir / "Objects_0_0_0001_0.json").write_text("{invalid json")

        issues = mod.check_metadata_completeness(scene)
        assert len(issues) >= 1
        assert "Invalid JSON" in issues[0]


# ═══════════════════════════════════════════════════════════════════════════
# 4.  config_validation.py — Gin config validation
# ═══════════════════════════════════════════════════════════════════════════


class TestConfigValidation:
    """Tests for gin configuration validation."""

    @pytest.fixture(autouse=True)
    def _skip_without_gin(self):
        pytest.importorskip("gin")

    def _setup_gin(self):
        import gin

        gin.clear_config()
        return gin

    def test_no_rules_no_issues(self):
        mod = _get_config_validation()
        gin = self._setup_gin()
        issues = mod.validate_gin_config()
        assert issues == []

    def test_required_key_missing(self):
        mod = _get_config_validation()
        gin = self._setup_gin()
        issues = mod.validate_gin_config(
            required_keys=["nonexistent_function.param"]
        )
        assert len(issues) == 1
        assert "not bound" in issues[0]

    def test_required_key_present(self):
        mod = _get_config_validation()
        gin = self._setup_gin()

        @gin.configurable
        def dummy_fn(fps=24):
            pass

        gin.parse_config("dummy_fn.fps = 30")
        issues = mod.validate_gin_config(required_keys=["dummy_fn.fps"])
        assert issues == []

    def test_type_check_passes(self):
        mod = _get_config_validation()
        gin = self._setup_gin()

        @gin.configurable
        def dummy_fn2(fps=24):
            pass

        gin.parse_config("dummy_fn2.fps = 30")
        issues = mod.validate_gin_config(
            type_rules={"dummy_fn2.fps": int}
        )
        assert issues == []

    def test_type_check_fails(self):
        mod = _get_config_validation()
        gin = self._setup_gin()

        @gin.configurable
        def dummy_fn3(fps=24):
            pass

        gin.parse_config('dummy_fn3.fps = "not_a_number"')
        issues = mod.validate_gin_config(
            type_rules={"dummy_fn3.fps": int}
        )
        assert len(issues) == 1
        assert "type" in issues[0].lower()

    def test_range_check_passes(self):
        mod = _get_config_validation()
        gin = self._setup_gin()

        @gin.configurable
        def dummy_fn4(fps=24):
            pass

        gin.parse_config("dummy_fn4.fps = 30")
        issues = mod.validate_gin_config(
            range_rules={"dummy_fn4.fps": (1, 240)}
        )
        assert issues == []

    def test_range_check_below_min(self):
        mod = _get_config_validation()
        gin = self._setup_gin()

        @gin.configurable
        def dummy_fn5(fps=24):
            pass

        gin.parse_config("dummy_fn5.fps = 0")
        issues = mod.validate_gin_config(
            range_rules={"dummy_fn5.fps": (1, 240)}
        )
        assert len(issues) == 1
        assert "below" in issues[0].lower()

    def test_range_check_above_max(self):
        mod = _get_config_validation()
        gin = self._setup_gin()

        @gin.configurable
        def dummy_fn6(fps=24):
            pass

        gin.parse_config("dummy_fn6.fps = 999")
        issues = mod.validate_gin_config(
            range_rules={"dummy_fn6.fps": (1, 240)}
        )
        assert len(issues) == 1
        assert "exceeds" in issues[0].lower()

    def test_custom_rule(self):
        mod = _get_config_validation()
        gin = self._setup_gin()

        @gin.configurable
        def dummy_fn7(fps=24):
            pass

        gin.parse_config("dummy_fn7.fps = 25")
        issues = mod.validate_gin_config(
            custom_rules={"dummy_fn7.fps": lambda v: v % 2 == 0}
        )
        assert len(issues) == 1
        assert "custom" in issues[0].lower()

    def test_gin_validation_error(self):
        mod = _get_config_validation()
        with pytest.raises(mod.GinValidationError) as exc_info:
            raise mod.GinValidationError(["issue1", "issue2"])
        assert exc_info.value.issues == ["issue1", "issue2"]
        assert "2 issue(s)" in str(exc_info.value)
