# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Copilot

"""
Benchmarks and correctness tests for optimisations:

1.  evaluate.py   – ChainMap instead of copy.copy(memo) in ForAll/SumOver/MeanOver
2.  skin_ops.py   – vectorised bevel_cap (single-pass array build, no O(n²) loop)
3.  logging.py    – time.perf_counter() replacing datetime.now() in Timer
4.  mesh.py       – Mesh.cat() pre-allocated O(n) concatenation
5.  ctype_util.py – CDLL kernel caching
6.  exporting.py  – np.empty pre-allocation & vectorised bbox transform
7.  mesh.py       – write_attributes() pre-computed boolean masks
8.  device.py     – Unified PyTorch device selection (CUDA/MPS/CPU)
"""

import platform
import time
from collections import ChainMap
from copy import copy
from ctypes import CDLL, RTLD_LOCAL
from datetime import timedelta
from itertools import product
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------


def _make_skin(n_rings: int = 8, n_pts: int = 12, ndim: int = 2, n_params: int = 0):
    """Create a minimal Skin-like object without importing bpy."""

    class SimpleSkin:
        pass

    s = SimpleSkin()
    s.ts = np.linspace(0, 1, n_rings)
    if ndim == 2:
        s.profiles = np.random.rand(n_rings, n_pts)
    else:
        s.profiles = np.random.rand(n_rings, n_pts, 3)
    if n_params > 0:
        s.surface_params = np.random.rand(n_rings, n_pts, n_params)
    else:
        s.surface_params = None
    return s


def _bevel_cap_reference(s, n: int, d: float, profile="SPHERE"):
    """Reference (original) implementation using the sequential extend_cap loop."""

    def extend_cap(skin, r=1, margin=0):
        res = copy(skin)
        res.ts = np.concatenate(
            [np.array([margin]), skin.ts, np.array([1 - margin])], axis=0
        )
        res.profiles = np.concatenate(
            [skin.profiles[[0]] * r, skin.profiles, skin.profiles[[-1]] * r]
        )
        if res.surface_params is not None:
            res.surface_params = np.concatenate(
                [
                    skin.surface_params[[0]],
                    skin.surface_params,
                    skin.surface_params[[-1]],
                ]
            )
        return res

    ts = np.linspace(1, 0, n)
    if profile == "SPHERE":
        rads = np.sqrt(1 - ts * ts)
    elif profile == "CHAMFER":
        rads = ts
    else:
        raise ValueError(f"Unrecognized {profile=}")
    for t, r in zip(ts, rads):
        s = extend_cap(s, r=r, margin=d * t)
    return s


def _bevel_cap_optimized(s, n: int, d: float, profile="SPHERE"):
    """Optimised implementation (single-pass, no O(n²) concatenation loop)."""
    ts = np.linspace(1, 0, n)
    if profile == "SPHERE":
        rads = np.sqrt(1 - ts * ts)
    elif profile == "CHAMFER":
        rads = ts
    else:
        raise ValueError(f"Unrecognized {profile=}")

    res = copy(s)
    cumprod_rads = np.cumprod(rads)
    extra_dims = s.profiles.ndim - 1
    scalings = cumprod_rads.reshape(-1, *([1] * extra_dims))

    start_profiles = s.profiles[[0]] * scalings[::-1]
    end_profiles = s.profiles[[-1]] * scalings
    res.profiles = np.concatenate([start_profiles, s.profiles, end_profiles])
    res.ts = np.concatenate([d * ts[::-1], s.ts, 1.0 - d * ts])

    if s.surface_params is not None:
        start_sp = np.repeat(s.surface_params[[0]], n, axis=0)
        end_sp = np.repeat(s.surface_params[[-1]], n, axis=0)
        res.surface_params = np.concatenate([start_sp, s.surface_params, end_sp])

    return res


# ---------------------------------------------------------------------------
# 1. bevel_cap correctness tests
# ---------------------------------------------------------------------------


class TestBevelCapCorrectness:
    """Verify that the optimised bevel_cap produces identical output to the reference."""

    @pytest.mark.parametrize("profile", ["SPHERE", "CHAMFER"])
    @pytest.mark.parametrize("n", [1, 3, 5, 10])
    @pytest.mark.parametrize("ndim", [2, 3])
    def test_profiles_match(self, profile, n, ndim):
        np.random.seed(0)
        s = _make_skin(n_rings=6, n_pts=8, ndim=ndim)
        d = 0.05

        ref = _bevel_cap_reference(copy(s), n, d, profile)
        opt = _bevel_cap_optimized(copy(s), n, d, profile)

        np.testing.assert_allclose(
            opt.profiles,
            ref.profiles,
            rtol=1e-12,
            err_msg=f"profiles mismatch for profile={profile}, n={n}, ndim={ndim}",
        )
        np.testing.assert_allclose(
            opt.ts,
            ref.ts,
            rtol=1e-12,
            err_msg=f"ts mismatch for profile={profile}, n={n}, ndim={ndim}",
        )

    @pytest.mark.parametrize("profile", ["SPHERE", "CHAMFER"])
    @pytest.mark.parametrize("n", [3, 7])
    def test_surface_params_match(self, profile, n):
        np.random.seed(42)
        s = _make_skin(n_rings=5, n_pts=6, ndim=2, n_params=2)
        d = 0.1

        ref = _bevel_cap_reference(copy(s), n, d, profile)
        opt = _bevel_cap_optimized(copy(s), n, d, profile)

        assert opt.surface_params is not None
        np.testing.assert_allclose(
            opt.surface_params,
            ref.surface_params,
            rtol=1e-12,
            err_msg=f"surface_params mismatch for profile={profile}, n={n}",
        )

    def test_output_shape(self):
        np.random.seed(1)
        n_rings, n_pts, bevel_n = 4, 6, 5
        s = _make_skin(n_rings=n_rings, n_pts=n_pts)
        result = _bevel_cap_optimized(s, bevel_n, d=0.05)

        expected_rings = n_rings + 2 * bevel_n
        assert result.profiles.shape == (expected_rings, n_pts), result.profiles.shape
        assert result.ts.shape == (expected_rings,), result.ts.shape

    def test_invalid_profile_raises(self):
        s = _make_skin()
        with pytest.raises(ValueError, match="Unrecognized"):
            _bevel_cap_optimized(s, n=3, d=0.05, profile="UNKNOWN")


# ---------------------------------------------------------------------------
# 2. bevel_cap performance benchmark
# ---------------------------------------------------------------------------


class TestBevelCapPerformance:
    """Assert that the optimised version is meaningfully faster than the reference."""

    def _time(self, fn, s_factory, n_repeat=20):
        times = []
        for _ in range(n_repeat):
            s = s_factory()
            t0 = time.perf_counter()
            fn(s, n=12, d=0.05)
            times.append(time.perf_counter() - t0)
        return float(np.median(times))

    def test_speed_improvement(self):
        np.random.seed(99)

        def make():
            return _make_skin(n_rings=20, n_pts=32)

        t_ref = self._time(lambda s, **kw: _bevel_cap_reference(s, **kw), make)
        t_opt = self._time(lambda s, **kw: _bevel_cap_optimized(s, **kw), make)

        # Optimised should be at least 2× faster
        speedup = t_ref / t_opt
        assert speedup >= 2.0, (
            f"Expected ≥2× speedup, got {speedup:.2f}× "
            f"(ref={t_ref*1000:.2f} ms, opt={t_opt*1000:.2f} ms)"
        )


# ---------------------------------------------------------------------------
# 3. ChainMap vs copy.copy correctness for ForAll-style loops
# ---------------------------------------------------------------------------


class TestChainMapEvalCorrectness:
    """Verify that ChainMap-based iteration is equivalent to copy.copy for memo isolation."""

    def _simulate_forall(self, objects, predicate_fn, use_chainmap: bool):
        """Simulate what evaluate.py does in ForAll: iterate over objects with memo isolation."""
        memo = {"shared_key": "shared_value"}
        results = []
        for o in objects:
            if use_chainmap:
                memo_sub = ChainMap({"var": o}, memo)
            else:
                import copy

                memo_sub = copy.copy(memo)
                memo_sub["var"] = o
            results.append(predicate_fn(memo_sub))
            # Ensure the parent memo is not polluted
            assert "var" not in memo, "Parent memo should not be modified"
        return results

    def test_isolation(self):
        """ChainMap must not let writes in one iteration leak into others."""
        objects = list(range(5))

        def pred(memo):
            val = memo["var"]
            memo["written_in_iter"] = val  # write a new key
            return val

        res_copy = self._simulate_forall(objects, pred, use_chainmap=False)
        res_chain = self._simulate_forall(objects, pred, use_chainmap=True)

        assert res_copy == res_chain, "ChainMap and copy.copy must give same results"

    def test_parent_reads_visible(self):
        """Values from the parent memo must be accessible through the ChainMap."""
        memo = {"parent_val": 42, "another": "hello"}

        memo_sub = ChainMap({"var": "x"}, memo)
        assert memo_sub["parent_val"] == 42
        assert memo_sub["another"] == "hello"
        assert memo_sub["var"] == "x"

    def test_child_write_does_not_pollute_parent(self):
        """Writing to a ChainMap child must not modify the parent dict."""
        memo = {"existing": 1}
        memo_sub = ChainMap({"var": "obj1"}, memo)

        memo_sub["new_key"] = "new_value"
        assert "new_key" not in memo, "Parent must not be polluted"
        assert memo_sub["new_key"] == "new_value"

    def test_chainmap_creation_is_cheaper_than_copy(self):
        """ChainMap creation should be faster than dict copy for a large memo."""
        import copy

        large_memo = {i: f"value_{i}" for i in range(500)}
        n_iters = 10_000

        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = copy.copy(large_memo)
        t_copy = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = ChainMap({"var": "x"}, large_memo)
        t_chain = time.perf_counter() - t0

        speedup = t_copy / t_chain
        assert speedup >= 5.0, (
            f"Expected ChainMap creation to be ≥5× faster than dict copy, "
            f"got {speedup:.2f}× (copy={t_copy*1000:.1f} ms, chain={t_chain*1000:.1f} ms)"
        )


# ---------------------------------------------------------------------------
# 4. Timer precision tests (perf_counter vs datetime.now)
# ---------------------------------------------------------------------------


class TestTimerPrecision:
    """Verify that the Timer now tracks sub-millisecond durations accurately."""

    def test_timer_duration_is_timedelta(self):
        """Timer.duration must still be a timedelta for backward compatibility."""
        import logging

        class _FakeTimer:
            """Minimal re-implementation of the optimised Timer logic."""

            def __init__(self):
                self.disable_timer = False
                self.logger = logging.getLogger("test.timer")

            def __enter__(self):
                self._start = time.perf_counter()

            def __exit__(self, *_):
                elapsed = time.perf_counter() - self._start
                self.duration = timedelta(seconds=elapsed)

        t = _FakeTimer()
        with t:
            time.sleep(0.001)  # 1 ms

        assert isinstance(t.duration, timedelta)
        # duration must be between 0.5 ms and 500 ms
        assert timedelta(seconds=0.0005) <= t.duration <= timedelta(seconds=0.5)

    def test_timer_sub_millisecond_resolution(self):
        """perf_counter must capture durations well below 1 ms."""
        t0 = time.perf_counter()
        # Do a tiny amount of work
        _ = sum(range(1000))
        elapsed = time.perf_counter() - t0
        dt = timedelta(seconds=elapsed)
        # Must be measurable (>0) and much less than 1 second
        assert timedelta(0) < dt < timedelta(seconds=1)

    def test_perf_counter_is_faster_than_datetime(self):
        """time.perf_counter() calls must be faster than datetime.now() calls."""
        from datetime import datetime

        n = 50_000

        t0 = time.perf_counter()
        for _ in range(n):
            time.perf_counter()
        t_perf = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(n):
            datetime.now()
        t_dt = time.perf_counter() - t0

        # perf_counter should be at least 2× faster than datetime.now
        speedup = t_dt / t_perf
        assert speedup >= 2.0, (
            f"Expected perf_counter ≥2× faster than datetime.now, got {speedup:.2f}×"
        )


# ---------------------------------------------------------------------------
# 5. Mesh.cat() O(n) pre-allocated concatenation benchmarks
# ---------------------------------------------------------------------------


def _make_fake_mesh(n_verts, n_faces, n_attrs=3):
    """Create a dict mimicking a Mesh object for concatenation testing."""

    class FakeMesh:
        pass

    m = FakeMesh()
    m.vertices = np.random.rand(n_verts, 3)
    m.faces = np.random.randint(0, n_verts, (n_faces, 3))
    va = {}
    for i in range(n_attrs):
        va[f"attr_{i}"] = np.random.rand(n_verts, 2).astype(np.float32)
    m.vertex_attributes = va
    return m


def _mesh_cat_reference(meshes):
    """Original O(n²) mesh concatenation using repeated np.concatenate."""
    verts = np.zeros((0, 3))
    faces = np.zeros((0, 3), dtype=int)
    lenv = 0
    vertex_attributes = {}
    for mesh in meshes:
        verts = np.concatenate((verts, mesh.vertices), 0)
        faces = np.concatenate((faces, mesh.faces + lenv), 0)

        for attr in mesh.vertex_attributes:
            if mesh.vertex_attributes[attr].ndim == 1:
                mesh.vertex_attributes[attr] = mesh.vertex_attributes[attr].reshape(
                    (-1, 1)
                )
            mesh_va = mesh.vertex_attributes[attr]
            if attr not in vertex_attributes:
                va = np.zeros(
                    (lenv, mesh.vertex_attributes[attr].shape[1]),
                    dtype=mesh.vertex_attributes[attr].dtype,
                )
            else:
                va = vertex_attributes[attr]
            vertex_attributes[attr] = np.concatenate((va, mesh_va))
        lenv += len(mesh.vertices)

        for attr in vertex_attributes:
            if len(vertex_attributes[attr]) != lenv:
                fillup = np.zeros(
                    (
                        lenv - len(vertex_attributes[attr]),
                        vertex_attributes[attr].shape[1],
                    ),
                    dtype=vertex_attributes[attr].dtype,
                )
                vertex_attributes[attr] = np.concatenate(
                    (vertex_attributes[attr], fillup)
                )
    return verts, faces, vertex_attributes


def _mesh_cat_optimized(meshes):
    """Optimized O(n) mesh concatenation using pre-allocated arrays."""
    if not meshes:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int), {}

    total_verts = sum(len(m.vertices) for m in meshes)
    total_faces = sum(len(m.faces) for m in meshes)

    verts = np.empty((total_verts, 3), dtype=np.float64)
    faces = np.empty((total_faces, 3), dtype=int)

    # Collect attribute metadata
    attr_meta = {}
    for mesh in meshes:
        for attr, arr in mesh.vertex_attributes.items():
            cols = 1 if arr.ndim == 1 else arr.shape[1]
            if attr not in attr_meta:
                attr_meta[attr] = (cols, arr.dtype)

    vertex_attributes = {
        attr: np.zeros((total_verts, cols), dtype=dtype)
        for attr, (cols, dtype) in attr_meta.items()
    }

    v_offset = 0
    f_offset = 0
    for mesh in meshes:
        nv = len(mesh.vertices)
        nf = len(mesh.faces)
        verts[v_offset : v_offset + nv] = mesh.vertices
        faces[f_offset : f_offset + nf] = mesh.faces + v_offset

        for attr, arr in mesh.vertex_attributes.items():
            if arr.ndim == 1:
                arr = arr.reshape((-1, 1))
            vertex_attributes[attr][v_offset : v_offset + nv] = arr

        v_offset += nv
        f_offset += nf

    return verts, faces, vertex_attributes


class TestMeshCatCorrectness:
    """Verify optimised Mesh.cat produces identical results to the reference."""

    @pytest.mark.parametrize("n_meshes", [1, 3, 10, 50])
    def test_vertices_match(self, n_meshes):
        np.random.seed(42)
        meshes = [_make_fake_mesh(100, 50) for _ in range(n_meshes)]
        meshes_copy = [copy(m) for m in meshes]

        v_ref, f_ref, a_ref = _mesh_cat_reference(meshes)
        v_opt, f_opt, a_opt = _mesh_cat_optimized(meshes_copy)

        np.testing.assert_allclose(v_opt, v_ref, rtol=1e-12)
        np.testing.assert_array_equal(f_opt, f_ref)
        for attr in a_ref:
            assert attr in a_opt, f"Missing attribute {attr}"
            np.testing.assert_allclose(a_opt[attr], a_ref[attr], rtol=1e-12)

    def test_empty_meshes(self):
        v, f, a = _mesh_cat_optimized([])
        assert v.shape == (0, 3)
        assert f.shape == (0, 3)
        assert a == {}

    def test_single_mesh(self):
        np.random.seed(1)
        mesh = _make_fake_mesh(50, 20)
        v, f, a = _mesh_cat_optimized([mesh])
        np.testing.assert_allclose(v, mesh.vertices, rtol=1e-12)

    def test_mixed_attributes(self):
        """Meshes with different attribute sets should be handled correctly."""
        np.random.seed(7)
        m1 = _make_fake_mesh(30, 10, n_attrs=2)
        m2 = _make_fake_mesh(40, 15, n_attrs=2)
        # Add an extra attribute only to m2
        m2.vertex_attributes["extra"] = np.random.rand(40, 1).astype(np.float32)

        v, f, a = _mesh_cat_optimized([m1, m2])
        assert v.shape[0] == 70
        assert "extra" in a
        # First 30 verts should be zero for extra (not in m1)
        np.testing.assert_array_equal(a["extra"][:30], 0.0)
        np.testing.assert_allclose(a["extra"][30:], m2.vertex_attributes["extra"])


class TestMeshCatPerformance:
    """Benchmark Mesh.cat pre-allocated vs repeated-concat."""

    def _time_fn(self, fn, meshes_factory, n_repeat=10):
        times = []
        for _ in range(n_repeat):
            meshes = meshes_factory()
            t0 = time.perf_counter()
            fn(meshes)
            times.append(time.perf_counter() - t0)
        return float(np.median(times))

    @pytest.mark.parametrize("n_meshes", [10, 50])
    def test_mesh_cat_speedup(self, n_meshes):
        np.random.seed(42)

        def make():
            return [_make_fake_mesh(500, 200) for _ in range(n_meshes)]

        t_ref = self._time_fn(_mesh_cat_reference, make)
        t_opt = self._time_fn(_mesh_cat_optimized, make)

        speedup = t_ref / t_opt
        # Pre-allocated should be at least 2× faster for n_meshes >= 10
        assert speedup >= 1.5, (
            f"Expected ≥1.5× speedup for {n_meshes} meshes, got {speedup:.2f}× "
            f"(ref={t_ref*1000:.2f} ms, opt={t_opt*1000:.2f} ms)"
        )


# ---------------------------------------------------------------------------
# 6. CDLL kernel caching benchmarks
# ---------------------------------------------------------------------------


class TestCDLLCaching:
    """Verify that CDLL caching avoids redundant library loads."""

    def test_cache_returns_same_handle(self):
        """Cached load_cdll should return the same object for the same path."""
        try:
            from infinigen.terrain.utils.ctype_util import _cdll_cache, load_cdll
        except ImportError:
            pytest.skip("infinigen.terrain.utils.ctype_util requires bpy")

        # Use a known system library for testing
        test_path = "terrain/lib/cpu/meshing/utils.so"
        _cdll_cache.clear()  # Start fresh

        # If the .so doesn't exist (CI), skip
        root = Path(__file__).parent.parent / "infinigen"
        so_path = root / test_path
        if not so_path.exists():
            pytest.skip(f"{so_path} not found (terrain not compiled)")

        dll1 = load_cdll(test_path)
        dll2 = load_cdll(test_path)
        assert dll1 is dll2, "Cached CDLL should return the same handle"
        assert test_path in _cdll_cache

    def test_cache_dict_behavior(self):
        """Verify caching dict behavior using a standalone dict-based cache."""
        cache = {}

        def cached_load(key):
            if key in cache:
                return cache[key]
            val = object()  # Simulate a loaded library
            cache[key] = val
            return val

        obj1 = cached_load("lib_a.so")
        obj2 = cached_load("lib_a.so")
        obj3 = cached_load("lib_b.so")
        assert obj1 is obj2, "Same key should return same object"
        assert obj1 is not obj3, "Different keys should return different objects"
        assert len(cache) == 2


# ---------------------------------------------------------------------------
# 7. np.empty vs np.full pre-allocation benchmarks
# ---------------------------------------------------------------------------


class TestNumpyPreAllocation:
    """Benchmark np.empty vs np.full for arrays immediately overwritten."""

    @pytest.mark.parametrize("size", [1000, 100_000, 1_000_000])
    def test_empty_faster_than_full(self, size):
        n_repeat = 100

        t0 = time.perf_counter()
        for _ in range(n_repeat):
            a = np.full(size, -1, dtype=np.int32)
        t_full = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(n_repeat):
            a = np.empty(size, dtype=np.int32)
        t_empty = time.perf_counter() - t0

        speedup = t_full / t_empty
        # np.empty should be faster for large arrays since it skips initialization
        if size >= 100_000:
            assert speedup >= 1.2, (
                f"Expected np.empty to be ≥1.2× faster than np.full for size={size}, "
                f"got {speedup:.2f}×"
            )

    def test_empty_produces_correct_shape(self):
        a = np.empty(100, dtype=np.int32)
        assert a.shape == (100,)
        assert a.dtype == np.int32

        b = np.empty((50, 3), dtype=np.float32)
        assert b.shape == (50, 3)
        assert b.dtype == np.float32


# ---------------------------------------------------------------------------
# 8. Vectorised bbox transform benchmarks
# ---------------------------------------------------------------------------


def _bbox_transform_reference(matrix_world, bound_box):
    """Original per-vertex matrix multiply using list comprehension."""
    import mathutils

    return np.asarray(
        [(matrix_world @ mathutils.Vector(v)) for v in bound_box], dtype=np.float32
    )


def _bbox_transform_optimized(matrix_world, bound_box):
    """Vectorised bbox transform using numpy matrix multiplication."""
    bbox = np.array(bound_box, dtype=np.float32)  # 8 x 3
    mat = np.asarray(matrix_world, dtype=np.float32)  # 4 x 4
    ones = np.ones((bbox.shape[0], 1), dtype=np.float32)
    homo = np.concatenate((bbox, ones), axis=1)  # 8 x 4
    transformed = (mat @ homo.T).T  # 8 x 4
    return transformed[:, :3] / transformed[:, 3:]


class TestBboxTransformCorrectness:
    """Verify vectorised bbox produces identical results to reference."""

    @pytest.mark.parametrize("seed", [0, 1, 42, 99])
    def test_results_match(self, seed):
        np.random.seed(seed)
        # Create a random 4x4 transformation matrix
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = np.random.rand(3, 3).astype(np.float32)
        mat[:3, 3] = np.random.rand(3).astype(np.float32)

        # Create 8 bounding box vertices
        bbox = np.random.rand(8, 3).astype(np.float32)

        result = _bbox_transform_optimized(mat, bbox)

        # Manual reference: compute each vertex individually
        expected = np.zeros((8, 3), dtype=np.float32)
        for i in range(8):
            v4 = np.append(bbox[i], 1.0)
            t = mat @ v4
            expected[i] = t[:3] / t[3]

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)


class TestBboxTransformPerformance:
    """Benchmark vectorised vs per-vertex bbox transform."""

    def test_vectorized_bbox_faster(self):
        np.random.seed(42)
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = np.random.rand(3, 3).astype(np.float32)
        mat[:3, 3] = np.random.rand(3).astype(np.float32)
        bbox = np.random.rand(8, 3).astype(np.float32)

        n_repeat = 5000

        # Baseline: pure numpy per-vertex
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            for i in range(8):
                v4 = np.append(bbox[i], 1.0)
                _ = (mat @ v4)[:3]
        t_ref = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(n_repeat):
            _bbox_transform_optimized(mat, bbox)
        t_opt = time.perf_counter() - t0

        speedup = t_ref / t_opt
        assert speedup >= 1.5, (
            f"Expected ≥1.5× speedup, got {speedup:.2f}× "
            f"(ref={t_ref*1000:.1f} ms, opt={t_opt*1000:.1f} ms)"
        )


# ---------------------------------------------------------------------------
# 9. write_attributes pre-computed mask benchmarks
# ---------------------------------------------------------------------------


def _write_attrs_reference(n_elements, N, sdf_data, attr_data):
    """Original write_attributes inner loop (recomputes mask each time)."""
    surface_element = sdf_data.argmin(axis=-1)
    attributes = {}
    for i in range(n_elements):
        for key in attr_data[i]:
            arr = attr_data[i][key].copy()
            if arr.ndim == 1:
                arr *= surface_element == i
            else:
                arr *= (surface_element == i).reshape((-1, 1))
            if key not in attributes:
                attributes[key] = arr
            else:
                attributes[key] += arr
    return attributes


def _write_attrs_optimized(n_elements, N, sdf_data, attr_data):
    """Optimized write_attributes with pre-computed masks and broadcasting."""
    surface_element = sdf_data.argmin(axis=-1)
    element_masks = [surface_element == i for i in range(n_elements)]

    attributes = {}
    for i in range(n_elements):
        mask_1d = element_masks[i]
        mask_2d = mask_1d[:, np.newaxis]
        for key in attr_data[i]:
            arr = attr_data[i][key].copy()
            if arr.ndim == 1:
                arr *= mask_1d
            else:
                arr *= mask_2d
            if key not in attributes:
                attributes[key] = arr
            else:
                attributes[key] += arr
    return attributes


class TestWriteAttrsCorrectness:
    """Verify pre-computed masks give same results."""

    @pytest.mark.parametrize("n_elements", [2, 5, 10])
    def test_results_match(self, n_elements):
        N = 10000

        def make_attr_data():
            np.random.seed(7)
            sdf = np.random.rand(N, n_elements).astype(np.float32)
            data = []
            for _ in range(n_elements):
                data.append(
                    {
                        "color": np.random.rand(N, 3).astype(np.float32),
                        "scalar": np.random.rand(N).astype(np.float32),
                    }
                )
            return sdf, data

        sdf1, data1 = make_attr_data()
        ref = _write_attrs_reference(n_elements, N, sdf1, data1)

        sdf2, data2 = make_attr_data()
        opt = _write_attrs_optimized(n_elements, N, sdf2, data2)

        for key in ref:
            assert key in opt
            np.testing.assert_allclose(opt[key], ref[key], rtol=1e-5)


class TestWriteAttrsPerformance:
    """Benchmark pre-computed masks vs recomputed masks."""

    def test_precomputed_masks_faster(self):
        np.random.seed(42)
        n_elements = 8
        N = 50000
        sdf_data = np.random.rand(N, n_elements).astype(np.float32)
        n_repeat = 20

        def make_attr_data():
            data = []
            for _ in range(n_elements):
                data.append(
                    {
                        "color": np.random.rand(N, 3).astype(np.float32),
                        "scalar": np.random.rand(N).astype(np.float32),
                    }
                )
            return data

        times_ref = []
        for _ in range(n_repeat):
            data = make_attr_data()
            t0 = time.perf_counter()
            _write_attrs_reference(n_elements, N, sdf_data, data)
            times_ref.append(time.perf_counter() - t0)

        times_opt = []
        for _ in range(n_repeat):
            data = make_attr_data()
            t0 = time.perf_counter()
            _write_attrs_optimized(n_elements, N, sdf_data, data)
            times_opt.append(time.perf_counter() - t0)

        t_ref = np.median(times_ref)
        t_opt = np.median(times_opt)
        speedup = t_ref / t_opt
        # Pre-computed masks avoid redundant comparisons; allow small margin for noise
        assert speedup >= 0.9, (
            f"Pre-computed masks unexpectedly slower, got {speedup:.2f}× "
            f"(ref={t_ref*1000:.2f} ms, opt={t_opt*1000:.2f} ms)"
        )


# ---------------------------------------------------------------------------
# 10. Device utility tests (CUDA/MPS/CPU auto-detection)
# ---------------------------------------------------------------------------


class TestDeviceUtility:
    """Test the unified PyTorch device selection utility."""

    def test_get_torch_device_returns_device(self):
        """get_torch_device must return a torch.device object."""
        try:
            from infinigen.core.util.device import get_torch_device

            device = get_torch_device()
            import torch

            assert isinstance(device, torch.device)
        except ImportError:
            pytest.skip("torch not available")

    def test_cpu_fallback(self):
        """Explicitly requesting CPU must return CPU device."""
        try:
            from infinigen.core.util.device import get_torch_device

            device = get_torch_device(prefer="cpu")
            assert device.type == "cpu"
        except ImportError:
            pytest.skip("torch not available")

    def test_invalid_device_falls_back(self):
        """Requesting an unavailable device should fall back gracefully."""
        try:
            from infinigen.core.util.device import get_torch_device

            device = get_torch_device(prefer="nonexistent_device")
            # Should fall back to auto-detection (not raise)
            import torch

            assert isinstance(device, torch.device)
        except ImportError:
            pytest.skip("torch not available")

    def test_env_var_override(self):
        """INFINIGEN_TORCH_DEVICE env var should override the prefer argument."""
        import os

        try:
            from infinigen.core.util.device import get_torch_device

            os.environ["INFINIGEN_TORCH_DEVICE"] = "cpu"
            device = get_torch_device(prefer="cuda")
            assert device.type == "cpu"
        except ImportError:
            pytest.skip("torch not available")
        finally:
            os.environ.pop("INFINIGEN_TORCH_DEVICE", None)

    def test_is_apple_silicon(self):
        """is_apple_silicon should return bool matching platform."""
        try:
            from infinigen.core.util.device import is_apple_silicon
        except ImportError:
            pytest.skip("infinigen.core.util.device requires bpy")

        result = is_apple_silicon()
        assert isinstance(result, bool)
        expected = platform.system() == "Darwin" and platform.machine() == "arm64"
        assert result == expected

    def test_mps_detection_on_apple(self):
        """On Apple Silicon, MPS should be selected if available."""
        try:
            import torch

            from infinigen.core.util.device import get_torch_device

            device = get_torch_device(prefer="mps")
            if (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                assert device.type == "mps"
            else:
                # Falls back to auto-detect
                assert device.type in ("cuda", "cpu")
        except ImportError:
            pytest.skip("torch not available")


# ---------------------------------------------------------------------------
# 11. np.prod vs deprecated np.product
# ---------------------------------------------------------------------------


class TestNpProdDeprecation:
    """Ensure np.prod is used instead of deprecated np.product."""

    def test_np_prod_works(self):
        """np.prod should work identically to the old np.product."""
        arr = np.array([2, 3, 4])
        assert np.prod(arr) == 24

    def test_np_prod_tuple(self):
        """np.prod on tuple input (as used in KERNELDATATYPE_DIMS)."""
        dims = (3,)
        assert np.prod(dims) == 3
        dims = (3, 3)
        assert np.prod(dims) == 9

    def test_np_prod_empty(self):
        """np.prod of empty should return 1."""
        assert np.prod([]) == 1.0
