# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""
Non-regression tests for core utility modules.

These tests verify that key functions produce correct, stable results and do
not regress in correctness or behaviour.  They are designed to run in CI
without ``bpy`` by using ``importlib.util.spec_from_file_location`` to load
individual modules directly.

Run with:
    pytest tests/test_nonregression.py --noconftest
"""

import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers – load modules without triggering bpy via __init__.py
# ---------------------------------------------------------------------------

_UTIL_DIR = Path(__file__).resolve().parent.parent / "infinigen" / "core" / "util"


def _load_module(name: str, filepath: Path):
    """Load a single Python file as a module, bypassing package __init__."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _get_array_ops():
    return _load_module("array_ops", _UTIL_DIR / "array_ops.py")


def _get_batch_ops():
    return _load_module("batch_ops", _UTIL_DIR / "batch_ops.py")


# ═══════════════════════════════════════════════════════════════════════════
# 1.  array_ops.unique_rows – deterministic correctness
# ═══════════════════════════════════════════════════════════════════════════


class TestUniqueRowsNonRegression:
    """Ensure unique_rows produces bit-exact results on known inputs."""

    def test_known_int_input(self):
        """Fixed input → fixed output (sorted unique rows)."""
        mod = _get_array_ops()
        arr = np.array(
            [[3, 1], [1, 2], [3, 1], [5, 0], [1, 2], [5, 0]],
            dtype=np.int32,
        )
        result = mod.unique_rows(arr)
        expected = np.array([[1, 2], [3, 1], [5, 0]], dtype=np.int32)
        # unique_rows may sort differently, so compare as sets
        result_set = {tuple(r) for r in result}
        expected_set = {tuple(r) for r in expected}
        assert result_set == expected_set

    def test_known_float_input(self):
        mod = _get_array_ops()
        arr = np.array(
            [[1.5, 2.5], [3.5, 4.5], [1.5, 2.5]],
            dtype=np.float64,
        )
        result = mod.unique_rows(arr)
        assert result.shape == (2, 2)

    def test_inverse_reconstructs_original(self):
        """unique[inverse] must exactly reconstruct the original array."""
        mod = _get_array_ops()
        np.random.seed(2024)
        arr = np.random.randint(0, 5, (200, 3), dtype=np.int32)
        uniq, inv = mod.unique_rows(arr, return_inverse=True)
        np.testing.assert_array_equal(uniq[inv], arr)

    def test_counts_sum_to_total(self):
        mod = _get_array_ops()
        np.random.seed(42)
        arr = np.random.randint(0, 3, (500, 2), dtype=np.int32)
        _, counts = mod.unique_rows(arr, return_counts=True)
        assert counts.sum() == 500

    def test_single_row(self):
        mod = _get_array_ops()
        arr = np.array([[7, 8, 9]], dtype=np.int32)
        result = mod.unique_rows(arr)
        np.testing.assert_array_equal(result, arr)

    def test_empty_like(self):
        """All-identical rows → single unique row."""
        mod = _get_array_ops()
        arr = np.ones((50, 4), dtype=np.float32)
        result = mod.unique_rows(arr)
        assert result.shape == (1, 4)

    def test_rejects_1d(self):
        mod = _get_array_ops()
        with pytest.raises(ValueError, match="2-D"):
            mod.unique_rows(np.array([1, 2, 3]))

    def test_large_deterministic(self):
        """A large fixed-seed array must always produce the same count."""
        mod = _get_array_ops()
        np.random.seed(12345)
        arr = np.random.randint(0, 20, (10_000, 5), dtype=np.int32)
        result = mod.unique_rows(arr)
        # Verify against np.unique for correctness
        expected_count = np.unique(arr, axis=0).shape[0]
        assert result.shape[0] == expected_count

    def test_matches_numpy_unique(self):
        """Must produce the same set of rows as np.unique(..., axis=0)."""
        mod = _get_array_ops()
        np.random.seed(9999)
        arr = np.random.randint(0, 10, (2000, 3), dtype=np.int32)
        ref = np.unique(arr, axis=0)
        fast = mod.unique_rows(arr)
        assert {tuple(r) for r in ref} == {tuple(r) for r in fast}


# ═══════════════════════════════════════════════════════════════════════════
# 2.  batch_ops – parallel_map, chunked_concat, batched_apply
# ═══════════════════════════════════════════════════════════════════════════


class TestBatchOpsNonRegression:
    """Verify batch_ops helpers produce correct results."""

    def test_parallel_map_identity(self):
        mod = _get_batch_ops()
        items = list(range(20))
        result = mod.parallel_map(lambda x: x * 2, items, num_workers=2)
        assert result == [x * 2 for x in items]

    def test_parallel_map_single_worker(self):
        """Fallback to sequential map with 1 worker."""
        mod = _get_batch_ops()
        items = list(range(10))
        result = mod.parallel_map(lambda x: x + 1, items, num_workers=1)
        assert result == [x + 1 for x in items]

    def test_parallel_map_preserves_order(self):
        mod = _get_batch_ops()
        items = list(range(100))
        result = mod.parallel_map(lambda x: -x, items, num_workers=4)
        assert result == [-x for x in items]

    def test_chunked_concat_basic(self):
        mod = _get_batch_ops()
        arrays = [np.arange(10 * i, 10 * (i + 1)) for i in range(5)]
        result = mod.chunked_concat(arrays)
        expected = np.arange(50)
        np.testing.assert_array_equal(result, expected)

    def test_chunked_concat_many_arrays(self):
        """With > 16 arrays, parallel path should be taken."""
        mod = _get_batch_ops()
        arrays = [np.array([i]) for i in range(100)]
        result = mod.chunked_concat(arrays, num_workers=4)
        expected = np.arange(100)
        np.testing.assert_array_equal(result, expected)

    def test_chunked_concat_empty_raises(self):
        mod = _get_batch_ops()
        with pytest.raises(ValueError, match="at least one array"):
            mod.chunked_concat([])

    def test_chunked_concat_2d(self):
        mod = _get_batch_ops()
        arrays = [np.ones((3, 2)) * i for i in range(20)]
        result = mod.chunked_concat(arrays, axis=0, num_workers=2)
        assert result.shape == (60, 2)
        # First 3 rows should be all zeros
        np.testing.assert_array_equal(result[:3], np.zeros((3, 2)))

    def test_batched_apply_identity(self):
        mod = _get_batch_ops()
        data = np.arange(100).reshape(10, 10)
        result = mod.batched_apply(lambda chunk: chunk * 2, data, batch_size=3)
        np.testing.assert_array_equal(result, data * 2)

    def test_batched_apply_exact_batches(self):
        """When data length is a multiple of batch_size."""
        mod = _get_batch_ops()
        data = np.ones((20, 5))
        result = mod.batched_apply(lambda c: c + 1, data, batch_size=5)
        np.testing.assert_array_equal(result, np.full((20, 5), 2.0))

    def test_batched_apply_single_batch(self):
        """batch_size >= data length → single call."""
        mod = _get_batch_ops()
        data = np.arange(10, dtype=np.float64).reshape(2, 5)
        result = mod.batched_apply(lambda c: c ** 2, data, batch_size=100)
        np.testing.assert_array_equal(result, data ** 2)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  device.py – device selection and capabilities
# ═══════════════════════════════════════════════════════════════════════════


class TestDeviceNonRegression:
    """Test device utilities return valid values on CI (CPU-only)."""

    @pytest.fixture(autouse=True)
    def _skip_without_torch(self):
        pytest.importorskip("torch")

    def _get_device_mod(self):
        return _load_module("device", _UTIL_DIR / "device.py")

    def test_get_torch_device_returns_device(self):
        import torch

        mod = self._get_device_mod()
        dev = mod.get_torch_device()
        assert isinstance(dev, torch.device)

    def test_get_torch_device_cpu_fallback(self):
        import os

        mod = self._get_device_mod()
        old = os.environ.get("INFINIGEN_TORCH_DEVICE")
        try:
            os.environ["INFINIGEN_TORCH_DEVICE"] = "cpu"
            dev = mod.get_torch_device()
            assert dev.type == "cpu"
        finally:
            if old is None:
                os.environ.pop("INFINIGEN_TORCH_DEVICE", None)
            else:
                os.environ["INFINIGEN_TORCH_DEVICE"] = old

    def test_optimal_dtype_returns_torch_dtype(self):
        import torch

        mod = self._get_device_mod()
        dtype = mod.optimal_dtype(torch.device("cpu"))
        assert dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)

    def test_optimal_batch_size_positive(self):
        import torch

        mod = self._get_device_mod()
        bs = mod.optimal_batch_size(torch.device("cpu"))
        assert isinstance(bs, int)
        assert bs > 0

    def test_optimal_num_threads_positive(self):
        import torch

        mod = self._get_device_mod()
        nt = mod.optimal_num_threads(torch.device("cpu"))
        assert isinstance(nt, int)
        assert nt >= 1

    def test_device_capabilities_dict(self):
        import torch

        mod = self._get_device_mod()
        caps = mod.device_capabilities(torch.device("cpu"))
        assert isinstance(caps, dict)
        assert "backend" in caps
        assert "optimal_dtype" in caps
        assert "batch_size" in caps
        assert "num_threads" in caps
        assert caps["backend"] == "cpu"

    def test_is_apple_silicon_returns_bool(self):
        mod = self._get_device_mod()
        result = mod.is_apple_silicon()
        assert isinstance(result, bool)

    def test_env_override_dtype(self):
        import os

        import torch

        mod = self._get_device_mod()
        old = os.environ.get("INFINIGEN_TORCH_DTYPE")
        try:
            os.environ["INFINIGEN_TORCH_DTYPE"] = "float64"
            result = mod.optimal_dtype(torch.device("cpu"))
            assert result == torch.float64
        finally:
            if old is None:
                os.environ.pop("INFINIGEN_TORCH_DTYPE", None)
            else:
                os.environ["INFINIGEN_TORCH_DTYPE"] = old

    def test_env_override_num_threads(self):
        import os

        import torch

        mod = self._get_device_mod()
        old = os.environ.get("INFINIGEN_NUM_THREADS")
        try:
            os.environ["INFINIGEN_NUM_THREADS"] = "3"
            result = mod.optimal_num_threads(torch.device("cpu"))
            assert result == 3
        finally:
            if old is None:
                os.environ.pop("INFINIGEN_NUM_THREADS", None)
            else:
                os.environ["INFINIGEN_NUM_THREADS"] = old


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Performance non-regression – ensure optimisations stay fast
# ═══════════════════════════════════════════════════════════════════════════


class TestPerformanceNonRegression:
    """Guard-rail tests: optimised paths must remain faster than naive ones."""

    def test_unique_rows_faster_than_np_unique(self):
        """unique_rows (void-view) must beat np.unique(..., axis=0)."""
        mod = _get_array_ops()
        np.random.seed(42)
        arr = np.random.randint(0, 100, (100_000, 4), dtype=np.int32)

        # Warm up
        mod.unique_rows(arr)
        np.unique(arr, axis=0)

        t0 = time.perf_counter()
        for _ in range(3):
            np.unique(arr, axis=0)
        t_ref = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(3):
            mod.unique_rows(arr)
        t_opt = time.perf_counter() - t0

        speedup = t_ref / t_opt if t_opt > 0 else float("inf")
        assert speedup >= 1.2, (
            f"unique_rows regression: only {speedup:.2f}× vs np.unique "
            f"(ref={t_ref*1000:.1f}ms, opt={t_opt*1000:.1f}ms)"
        )

    def test_chunked_concat_not_slower(self):
        """Parallel chunked_concat must not be drastically slower than serial."""
        mod = _get_batch_ops()
        # Use larger arrays to amortise thread-pool overhead
        arrays = [np.random.rand(10_000) for _ in range(200)]

        # Serial baseline
        t0 = time.perf_counter()
        for _ in range(3):
            np.concatenate(arrays)
        t_serial = time.perf_counter() - t0

        # Parallel chunked
        t0 = time.perf_counter()
        for _ in range(3):
            mod.chunked_concat(arrays, num_workers=4)
        t_parallel = time.perf_counter() - t0

        # Allow up to 10× slower to account for thread-pool startup on CI
        assert t_parallel < t_serial * 10, (
            f"chunked_concat too slow: {t_parallel*1000:.1f}ms vs {t_serial*1000:.1f}ms serial"
        )
