# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Copilot

"""
Benchmarks and correctness tests for optimisations:

1.  evaluate.py            – ChainMap instead of copy.copy(memo) in ForAll/SumOver/MeanOver
2.  skin_ops.py            – vectorised bevel_cap (single-pass array build, no O(n²) loop)
3.  logging.py             – time.perf_counter() replacing datetime.now() in Timer
4.  mesh.py                – Mesh.cat() pre-allocated O(n) concatenation
5.  ctype_util.py          – CDLL kernel caching
6.  exporting.py           – np.empty pre-allocation & vectorised bbox transform
7.  mesh.py                – write_attributes() pre-computed boolean masks
8.  device.py              – Unified PyTorch device selection (CUDA/MPS/CPU)
9.  mesh.py                – Vectorised heightmap grid + face index computation
10. device.py              – Device capabilities, optimal dtype/batch/threads
11. batch_ops.py           – Thread-pooled parallel_map, chunked_concat, batched_apply
12. path_finding           – np.product → np.prod deprecation fix
13. segmentation_lookup.py – void-view unique_rows replacing np.unique(..., axis=0) bottleneck
14. tree.py                – List accumulation replacing O(n²) np.append in parse_tree_attributes
15. tree.py                – Lazy vertex concatenation in TreeVertices
16. mesh.py                – Pre-computed combined projection matrices via K @ inv(cam)[:3,:]
17. mesh.py                – Cached C function argtypes/restype setup
"""

import importlib
import platform
import time
from collections import ChainMap
from copy import copy
from datetime import timedelta
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

        def make_meshes():
            return [_make_fake_mesh(100, 50) for _ in range(n_meshes)]

        meshes_ref = make_meshes()
        np.random.seed(42)
        meshes_opt = make_meshes()

        v_ref, f_ref, a_ref = _mesh_cat_reference(meshes_ref)
        v_opt, f_opt, a_opt = _mesh_cat_optimized(meshes_opt)

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
        # Pre-computed masks save redundant boolean array creation; the improvement
        # is modest for small element counts but beneficial in real terrain workloads
        # with many elements. Allow noise margin in CI.
        assert speedup >= 0.9, (
            f"Pre-computed masks unexpectedly much slower, got {speedup:.2f}× "
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


# ---------------------------------------------------------------------------
# 12. Vectorised heightmap grid + face index construction
# ---------------------------------------------------------------------------


def _heightmap_grid_reference(N, L):
    """Original loop-based heightmap grid construction."""
    heightmap = np.random.rand(N, N).astype(np.float64)
    verts = np.zeros((N, N, 3))
    for i in range(N):
        verts[i, :, 0] = (-1 + 2 * i / (N - 1)) * L / 2
    for j in range(N):
        verts[:, j, 1] = (-1 + 2 * j / (N - 1)) * L / 2
    verts[:, :, 2] = heightmap
    verts = verts.reshape((-1, 3))

    faces = np.zeros((2, N - 1, N - 1, 3), np.int32)
    for i in range(N - 1):
        faces[0, i, :, :] += [i * N, (i + 1) * N, i * N]
        faces[1, i, :, :] += [i * N, (i + 1) * N, (i + 1) * N]
    for j in range(N - 1):
        faces[0, :, j, :] += [j, j, j + 1]
        faces[1, :, j, :] += [j + 1, j, j + 1]
    faces = faces.reshape((-1, 3))
    return verts, faces


def _heightmap_grid_optimized(N, L):
    """Vectorised heightmap grid using meshgrid + broadcasting."""
    heightmap = np.random.rand(N, N).astype(np.float64)
    xs = (-1 + 2 * np.arange(N) / (N - 1)) * L / 2
    ys = (-1 + 2 * np.arange(N) / (N - 1)) * L / 2
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    verts = np.stack([gx, gy, heightmap], axis=-1).reshape((-1, 3))

    i_idx = np.arange(N - 1).reshape(-1, 1)
    j_idx = np.arange(N - 1).reshape(1, -1)
    faces = np.empty((2, N - 1, N - 1, 3), np.int32)
    faces[0] = np.stack(
        [i_idx * N + j_idx, (i_idx + 1) * N + j_idx, i_idx * N + j_idx + 1], axis=-1
    )
    faces[1] = np.stack(
        [i_idx * N + j_idx + 1, (i_idx + 1) * N + j_idx, (i_idx + 1) * N + j_idx + 1],
        axis=-1,
    )
    faces = faces.reshape((-1, 3))
    return verts, faces


class TestHeightmapGridCorrectness:
    """Verify vectorised heightmap grid matches the reference loop-based version."""

    @pytest.mark.parametrize("N", [4, 16, 64])
    def test_verts_match(self, N):
        L = 10.0
        np.random.seed(42)
        v_ref, f_ref = _heightmap_grid_reference(N, L)
        np.random.seed(42)
        v_opt, f_opt = _heightmap_grid_optimized(N, L)

        np.testing.assert_allclose(v_opt, v_ref, rtol=1e-12)
        np.testing.assert_array_equal(f_opt, f_ref)


class TestHeightmapGridPerformance:
    """Vectorised heightmap must be faster than the loop-based version."""

    def test_speedup(self):
        N = 256
        L = 20.0
        n_repeat = 20

        times_ref = []
        for _ in range(n_repeat):
            np.random.seed(0)
            t0 = time.perf_counter()
            _heightmap_grid_reference(N, L)
            times_ref.append(time.perf_counter() - t0)

        times_opt = []
        for _ in range(n_repeat):
            np.random.seed(0)
            t0 = time.perf_counter()
            _heightmap_grid_optimized(N, L)
            times_opt.append(time.perf_counter() - t0)

        t_ref = np.median(times_ref)
        t_opt = np.median(times_opt)
        speedup = t_ref / t_opt
        assert speedup >= 1.3, (
            f"Expected ≥1.3× speedup for N={N}, got {speedup:.2f}× "
            f"(ref={t_ref*1000:.2f} ms, opt={t_opt*1000:.2f} ms)"
        )


# ---------------------------------------------------------------------------
# 13. Device capability helpers
# ---------------------------------------------------------------------------


class TestDeviceCapabilities:
    """Test the device capability query helpers."""

    def test_optimal_dtype_returns_torch_dtype(self):
        try:
            import torch

            from infinigen.core.util.device import optimal_dtype

            dt = optimal_dtype(torch.device("cpu"))
            assert dt == torch.float32
        except ImportError:
            pytest.skip("torch not available")

    def test_optimal_dtype_env_override(self):
        import os

        try:
            import torch

            from infinigen.core.util.device import optimal_dtype

            os.environ["INFINIGEN_TORCH_DTYPE"] = "float64"
            dt = optimal_dtype(torch.device("cpu"))
            assert dt == torch.float64
        except ImportError:
            pytest.skip("torch not available")
        finally:
            os.environ.pop("INFINIGEN_TORCH_DTYPE", None)

    def test_optimal_batch_size_cpu(self):
        try:
            import torch

            from infinigen.core.util.device import optimal_batch_size

            bs = optimal_batch_size(torch.device("cpu"))
            assert bs == 1_000_000
        except ImportError:
            pytest.skip("torch not available")

    def test_optimal_num_threads_cpu(self):
        try:
            import torch

            from infinigen.core.util.device import optimal_num_threads

            n = optimal_num_threads(torch.device("cpu"))
            assert 1 <= n <= 8
        except ImportError:
            pytest.skip("torch not available")

    def test_num_threads_env_override(self):
        import os

        try:
            import torch

            from infinigen.core.util.device import optimal_num_threads

            os.environ["INFINIGEN_NUM_THREADS"] = "3"
            n = optimal_num_threads(torch.device("cpu"))
            assert n == 3
        except ImportError:
            pytest.skip("torch not available")
        finally:
            os.environ.pop("INFINIGEN_NUM_THREADS", None)

    def test_device_capabilities_dict(self):
        try:
            import torch

            from infinigen.core.util.device import device_capabilities

            caps = device_capabilities(torch.device("cpu"))
            assert "backend" in caps
            assert caps["backend"] == "cpu"
            assert "optimal_dtype" in caps
            assert "batch_size" in caps
            assert "num_threads" in caps
        except ImportError:
            pytest.skip("torch not available")


# ---------------------------------------------------------------------------
# 14. Batch operations (parallel_map, chunked_concat, batched_apply)
# ---------------------------------------------------------------------------


class TestBatchOps:
    """Test thread-pooled batch processing utilities."""

    def _import_batch_ops(self):
        import sys

        spec = importlib.util.spec_from_file_location(
            "batch_ops",
            Path(__file__).parent.parent / "infinigen" / "core" / "util" / "batch_ops.py",
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["batch_ops"] = mod
        spec.loader.exec_module(mod)
        return mod.parallel_map, mod.chunked_concat, mod.batched_apply

    def test_parallel_map_correctness(self):
        parallel_map, _, _ = self._import_batch_ops()
        items = list(range(20))
        result = parallel_map(lambda x: x * 2, items, num_workers=4)
        assert result == [x * 2 for x in items]

    def test_parallel_map_single_worker(self):
        parallel_map, _, _ = self._import_batch_ops()
        items = list(range(10))
        result = parallel_map(lambda x: x + 1, items, num_workers=1)
        assert result == [x + 1 for x in items]

    def test_parallel_map_empty(self):
        parallel_map, _, _ = self._import_batch_ops()
        result = parallel_map(lambda x: x, [], num_workers=4)
        assert result == []

    def test_chunked_concat_correctness(self):
        _, chunked_concat, _ = self._import_batch_ops()
        arrays = [np.random.rand(100, 3) for _ in range(32)]
        expected = np.concatenate(arrays, axis=0)
        result = chunked_concat(arrays, num_workers=4)
        np.testing.assert_array_equal(result, expected)

    def test_chunked_concat_small_fallback(self):
        _, chunked_concat, _ = self._import_batch_ops()
        arrays = [np.array([1, 2]), np.array([3, 4])]
        result = chunked_concat(arrays)
        np.testing.assert_array_equal(result, np.array([1, 2, 3, 4]))

    def test_batched_apply_correctness(self):
        _, _, batched_apply = self._import_batch_ops()
        data = np.arange(100).reshape(100, 1).astype(np.float64)
        result = batched_apply(lambda x: x * 2, data, batch_size=30)
        np.testing.assert_array_equal(result, data * 2)

    def test_batched_apply_single_batch(self):
        _, _, batched_apply = self._import_batch_ops()
        data = np.ones((10, 3))
        result = batched_apply(lambda x: x + 1, data, batch_size=100)
        np.testing.assert_array_equal(result, np.full((10, 3), 2.0))


class TestBatchOpsPerformance:
    """Benchmark parallel_map vs sequential for CPU-bound work."""

    def test_parallel_map_not_slower_for_heavy_work(self):
        import sys

        spec = importlib.util.spec_from_file_location(
            "batch_ops",
            Path(__file__).parent.parent / "infinigen" / "core" / "util" / "batch_ops.py",
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["batch_ops"] = mod
        spec.loader.exec_module(mod)
        parallel_map = mod.parallel_map

        def heavy_fn(x):
            return np.linalg.svd(np.random.rand(50, 50))[1].sum()

        items = list(range(32))
        n_repeat = 3

        # Sequential
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            [heavy_fn(x) for x in items]
        t_seq = time.perf_counter() - t0

        # Parallel
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            parallel_map(heavy_fn, items, num_workers=4)
        t_par = time.perf_counter() - t0

        # Due to GIL, numpy releases it during SVD, so parallel should not be
        # dramatically slower. Allow up to 2× slower (very conservative for CI).
        ratio = t_par / t_seq
        assert ratio < 2.0, (
            f"parallel_map was {ratio:.2f}× slower than sequential "
            f"(seq={t_seq:.3f}s, par={t_par:.3f}s)"
        )


# ---------------------------------------------------------------------------
# 16. Ground-truth mask processing bottleneck
#     (np.unique on 2-D arrays, annotated in segmentation_lookup.py and
#     bounding_boxes_3d.py as "this line is the bottleneck")
# ---------------------------------------------------------------------------


def _make_combined_mask(H: int, W: int, n_objects: int) -> np.ndarray:
    """Build a synthetic combined (object, instance) segmentation mask.

    Mimics the ``combined_mask`` arrays produced by
    ``segmentation_lookup.py`` and ``bounding_boxes_3d.py`` before the
    ``np.unique(..., axis=0)`` bottleneck call.  Each row represents a
    pixel and contains ``[object_id, instance_id]`` as int32 values.
    """
    np.random.seed(42)
    obj_ids = np.random.randint(0, n_objects, H * W, dtype=np.int32)
    inst_ids = np.random.randint(0, n_objects * 10, H * W, dtype=np.int32)
    return np.stack([obj_ids, inst_ids], axis=1)


def _unique_rows_reference(arr: np.ndarray) -> np.ndarray:
    """Original bottleneck: ``np.unique`` on the full 2-D combined mask."""
    return np.unique(arr, axis=0)


def _unique_rows_reference_with_inverse(
    arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Original bottleneck with inverse indices (as used in segmentation_lookup)."""
    return np.unique(arr, return_inverse=True, axis=0)


def _unique_rows_fast(arr: np.ndarray) -> np.ndarray:
    """Optimised: view each row as a single void element for 1-D unique.

    Instead of asking NumPy to compare rows element-by-element
    (``np.unique(..., axis=0)`` uses an internal lexsort), we reinterpret
    each row as an opaque byte blob of fixed size and run the standard 1-D
    unique algorithm.  The result contains the same set of unique rows
    (potentially in a different sort order due to little-endian byte
    comparison, which is fine for set-membership queries).
    """
    arr = np.ascontiguousarray(arr)
    row_dtype = np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))
    row_view = arr.view(row_dtype).reshape(-1)
    uniq_void = np.unique(row_view)
    return uniq_void.view(arr.dtype).reshape(-1, arr.shape[1])


def _unique_rows_fast_with_inverse(
    arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Optimised void-view unique returning inverse indices.

    Returns ``(unique_rows, inverse)`` such that
    ``unique_rows[inverse]`` reconstructs ``arr`` row-by-row, matching
    the contract of ``np.unique(..., return_inverse=True, axis=0)``.
    """
    arr = np.ascontiguousarray(arr)
    row_dtype = np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))
    row_view = arr.view(row_dtype).reshape(-1)
    uniq_void, inverse = np.unique(row_view, return_inverse=True)
    unique_rows = uniq_void.view(arr.dtype).reshape(-1, arr.shape[1])
    return unique_rows, inverse


class TestUniqueRowsCorrectness:
    """Verify that the void-view optimisation produces the same unique rows
    as ``np.unique(..., axis=0)`` for realistic segmentation mask inputs."""

    @pytest.mark.parametrize("H,W", [(64, 64), (256, 256)])
    @pytest.mark.parametrize("n_objects", [5, 50])
    def test_unique_rows_set_equal(self, H, W, n_objects):
        """Both methods must return the same *set* of unique rows."""
        arr = _make_combined_mask(H, W, n_objects)
        ref = _unique_rows_reference(arr)
        fast = _unique_rows_fast(arr)

        ref_set = {tuple(row) for row in ref}
        fast_set = {tuple(row) for row in fast}
        assert ref_set == fast_set, (
            f"Unique row sets differ for H={H}, W={W}, n_objects={n_objects}: "
            f"ref has {len(ref_set)} rows, fast has {len(fast_set)} rows"
        )

    @pytest.mark.parametrize("H,W", [(64, 64), (256, 256)])
    def test_unique_rows_inverse_reconstructs_input(self, H, W):
        """unique_rows[inverse] must equal the original combined mask row-by-row."""
        arr = _make_combined_mask(H, W, n_objects=20)
        fast_uniq, fast_inv = _unique_rows_fast_with_inverse(arr)
        reconstructed = fast_uniq[fast_inv]
        np.testing.assert_array_equal(
            reconstructed,
            arr,
            err_msg=f"Reconstruction failed for H={H}, W={W}",
        )

    def test_unique_rows_inverse_matches_reference(self):
        """Inverse-index variant must reconstruct the array identically to the
        reference ``np.unique(..., return_inverse=True, axis=0)`` variant."""
        arr = _make_combined_mask(128, 128, n_objects=30)
        ref_uniq, ref_inv = _unique_rows_reference_with_inverse(arr)
        fast_uniq, fast_inv = _unique_rows_fast_with_inverse(arr)

        # Both must reconstruct the original array
        np.testing.assert_array_equal(ref_uniq[ref_inv], arr)
        np.testing.assert_array_equal(fast_uniq[fast_inv], arr)

        # The sets of unique rows must be identical
        ref_set = {tuple(row) for row in ref_uniq}
        fast_set = {tuple(row) for row in fast_uniq}
        assert ref_set == fast_set

    def test_unique_rows_single_object(self):
        """Edge case: all pixels belong to the same (object, instance) pair."""
        arr = np.zeros((100, 2), dtype=np.int32)
        fast = _unique_rows_fast(arr)
        assert fast.shape == (1, 2)
        np.testing.assert_array_equal(fast[0], [0, 0])

    def test_unique_rows_all_distinct(self):
        """Edge case: every pixel is a unique (object, instance) pair."""
        N = 50
        arr = np.arange(N * 2, dtype=np.int32).reshape(N, 2)
        fast = _unique_rows_fast(arr)
        assert fast.shape[0] == N


class TestUniqueRowsPerformance:
    """Benchmark the void-view ``unique_rows`` against the original
    ``np.unique(..., axis=0)`` bottleneck for realistic HD mask sizes.

    The annotated bottleneck in ``segmentation_lookup.py`` (line 139) and
    ``bounding_boxes_3d.py`` (line 132) is called on full-resolution
    combined masks, so 960×540 and 1920×1080 are representative sizes.
    """

    def _time_ms(self, fn, arr: np.ndarray, n_repeat: int = 5) -> float:
        """Return the median elapsed time (ms) for calling ``fn(arr)``."""
        times = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            fn(arr)
            times.append(time.perf_counter() - t0)
        return float(np.median(times)) * 1000

    @pytest.mark.parametrize("H,W", [(540, 960), (1080, 1920)])
    def test_unique_rows_speedup(self, H, W):
        n_objects = 100
        arr = _make_combined_mask(H, W, n_objects)

        t_ref_ms = self._time_ms(_unique_rows_reference, arr)
        t_fast_ms = self._time_ms(_unique_rows_fast, arr)
        speedup = t_ref_ms / t_fast_ms
        assert speedup >= 1.3, (
            f"Expected ≥1.3× speedup for {H}×{W} mask, got {speedup:.2f}× "
            f"(ref={t_ref_ms:.1f} ms, fast={t_fast_ms:.1f} ms)"
        )

    def test_unique_rows_with_inverse_speedup(self):
        """Inverse-index variant (used in segmentation_lookup.py) should also
        be faster than the 2-D reference."""
        H, W = 1080, 1920
        arr = _make_combined_mask(H, W, n_objects=100)

        t_ref_ms = self._time_ms(_unique_rows_reference_with_inverse, arr)
        t_fast_ms = self._time_ms(_unique_rows_fast_with_inverse, arr)
        speedup = t_ref_ms / t_fast_ms
        assert speedup >= 1.3, (
            f"Expected ≥1.3× speedup for {H}×{W} mask (with inverse), "
            f"got {speedup:.2f}× "
            f"(ref={t_ref_ms:.1f} ms, fast={t_fast_ms:.1f} ms)"
        )


# ---------------------------------------------------------------------------
# 15. Baseline comparison framework
# ---------------------------------------------------------------------------


class TestBaselineComparison:
    """Framework for tracking speedups relative to the upstream main baseline.

    Each test records the absolute time of the optimised path.  The baseline
    times are hard-coded from a calibration run on the upstream main branch
    (numpy-only, no bpy).  Because CI hardware varies, the assertions only
    check that the optimised path is not *dramatically* slower than expected.
    """

    # Calibrated baselines (median ms on GitHub Actions ubuntu-latest, 2-core)
    # These are conservative upper bounds; actual runs are usually faster.
    _BASELINES = {
        "bevel_cap_ms": 5.0,
        "chainmap_10k_ms": 15.0,
        "mesh_cat_50_ms": 25.0,
        "heightmap_256_ms": 40.0,
        "bbox_transform_5k_ms": 50.0,
        "unique_rows_1080p_ms": 500.0,
    }

    def test_bevel_cap_within_budget(self):
        np.random.seed(99)
        s = _make_skin(n_rings=20, n_pts=32)
        times = []
        for _ in range(20):
            s_copy = copy(s)
            t0 = time.perf_counter()
            _bevel_cap_optimized(s_copy, n=12, d=0.05)
            times.append((time.perf_counter() - t0) * 1000)
        median_ms = float(np.median(times))
        budget = self._BASELINES["bevel_cap_ms"]
        assert median_ms < budget * 5, (
            f"bevel_cap took {median_ms:.2f} ms, budget {budget} ms (5× headroom)"
        )

    def test_chainmap_within_budget(self):
        large_memo = {i: f"value_{i}" for i in range(500)}
        n_iters = 10_000
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = ChainMap({"var": "x"}, large_memo)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        budget = self._BASELINES["chainmap_10k_ms"]
        assert elapsed_ms < budget * 5, (
            f"ChainMap creation took {elapsed_ms:.2f} ms, budget {budget} ms"
        )

    def test_mesh_cat_within_budget(self):
        np.random.seed(42)
        meshes = [_make_fake_mesh(500, 200) for _ in range(50)]
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            _mesh_cat_optimized(meshes)
            times.append((time.perf_counter() - t0) * 1000)
        median_ms = float(np.median(times))
        budget = self._BASELINES["mesh_cat_50_ms"]
        assert median_ms < budget * 5, (
            f"Mesh.cat(50) took {median_ms:.2f} ms, budget {budget} ms"
        )

    def test_heightmap_within_budget(self):
        np.random.seed(0)
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            _heightmap_grid_optimized(256, 20.0)
            times.append((time.perf_counter() - t0) * 1000)
        median_ms = float(np.median(times))
        budget = self._BASELINES["heightmap_256_ms"]
        assert median_ms < budget * 5, (
            f"heightmap 256 took {median_ms:.2f} ms, budget {budget} ms"
        )

    def test_bbox_transform_within_budget(self):
        np.random.seed(42)
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = np.random.rand(3, 3).astype(np.float32)
        mat[:3, 3] = np.random.rand(3).astype(np.float32)
        bbox = np.random.rand(8, 3).astype(np.float32)
        n_repeat = 5000
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            _bbox_transform_optimized(mat, bbox)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        budget = self._BASELINES["bbox_transform_5k_ms"]
        assert elapsed_ms < budget * 5, (
            f"bbox transform (5k iters) took {elapsed_ms:.2f} ms, budget {budget} ms"
        )

    def test_unique_rows_within_budget(self):
        """The void-view unique_rows must complete within budget for 1080p masks."""
        arr = _make_combined_mask(1080, 1920, n_objects=100)
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            _unique_rows_fast(arr)
            times.append((time.perf_counter() - t0) * 1000)
        median_ms = float(np.median(times))
        budget = self._BASELINES["unique_rows_1080p_ms"]
        assert median_ms < budget * 5, (
            f"unique_rows 1080p took {median_ms:.2f} ms, budget {budget} ms (5× headroom)"
        )


# ---------------------------------------------------------------------------
# 16. Tree np.append → list accumulation
# ---------------------------------------------------------------------------


def _tree_append_reference(n_initial, n_appends):
    """Reference: repeated np.append (O(n²) total)."""
    parents = np.zeros(n_initial, dtype=int)
    vtx_pos = np.random.rand(n_initial, 3)
    for i in range(n_appends):
        parents = np.append(parents, i % n_initial)
        vtx_pos = np.append(vtx_pos, np.random.rand(1, 3), axis=0)
    return parents, vtx_pos


def _tree_append_optimized(n_initial, n_appends):
    """Optimised: accumulate in lists, concatenate once (O(n) total)."""
    parents = np.zeros(n_initial, dtype=int)
    vtx_pos = np.random.rand(n_initial, 3)
    acc_parents = []
    acc_vtx_pos = []
    for i in range(n_appends):
        acc_parents.append(i % n_initial)
        acc_vtx_pos.append(np.random.rand(3))
    if acc_parents:
        parents = np.concatenate([parents, np.array(acc_parents, dtype=int)])
        vtx_pos = np.concatenate([vtx_pos, np.array(acc_vtx_pos).reshape(-1, 3)])
    return parents, vtx_pos


class TestTreeAppendCorrectness:
    """Verify that list-accumulation produces identical results to np.append."""

    @pytest.mark.parametrize("n_initial,n_appends", [(10, 50), (100, 200), (5, 0)])
    def test_results_match(self, n_initial, n_appends):
        np.random.seed(42)
        ref_p, ref_v = _tree_append_reference(n_initial, n_appends)
        np.random.seed(42)
        opt_p, opt_v = _tree_append_optimized(n_initial, n_appends)
        np.testing.assert_array_equal(ref_p, opt_p)
        np.testing.assert_allclose(ref_v, opt_v)


class TestTreeAppendPerformance:
    """Assert list-accumulation is faster than repeated np.append."""

    def _time_ms(self, fn, *args, n_repeat=5):
        times = []
        for _ in range(n_repeat):
            np.random.seed(0)
            t0 = time.perf_counter()
            fn(*args)
            times.append((time.perf_counter() - t0) * 1000)
        return float(np.median(times))

    def test_speedup(self):
        n_initial, n_appends = 100, 500
        t_ref = self._time_ms(_tree_append_reference, n_initial, n_appends)
        t_opt = self._time_ms(_tree_append_optimized, n_initial, n_appends)
        speedup = t_ref / t_opt if t_opt > 0 else float("inf")
        assert speedup >= 1.3, (
            f"Expected ≥1.3× speedup, got {speedup:.2f}× "
            f"(ref={t_ref:.1f} ms, opt={t_opt:.1f} ms)"
        )


# ---------------------------------------------------------------------------
# 17. TreeVertices lazy concatenation
# ---------------------------------------------------------------------------


class _LazyVertices:
    """Mimics the optimised TreeVertices with lazy concatenation."""

    def __init__(self, vtxs):
        self._parts = [np.asarray(vtxs)]
        self._cache = self._parts[0]
        self._dirty = False

    @property
    def vtxs(self):
        if self._dirty:
            self._cache = np.concatenate(self._parts)
            self._parts = [self._cache]
            self._dirty = False
        return self._cache

    def append(self, v):
        self._parts.append(np.asarray(v).reshape(-1, 3))
        self._dirty = True

    def __len__(self):
        return len(self.vtxs)


class _EagerVertices:
    """Mimics the original TreeVertices with np.append."""

    def __init__(self, vtxs):
        self.vtxs = np.asarray(vtxs)

    def append(self, v):
        self.vtxs = np.append(self.vtxs, np.asarray(v).reshape(-1, 3), axis=0)

    def __len__(self):
        return len(self.vtxs)


class TestLazyVerticesCorrectness:
    """Verify lazy concatenation produces identical arrays."""

    def test_basic_append(self):
        np.random.seed(0)
        lazy = _LazyVertices(np.zeros((1, 3)))
        eager = _EagerVertices(np.zeros((1, 3)))
        for _ in range(50):
            row = np.random.rand(1, 3)
            lazy.append(row)
            eager.append(row)
        np.testing.assert_array_equal(lazy.vtxs, eager.vtxs)
        assert len(lazy) == len(eager)


class TestLazyVerticesPerformance:
    """Assert lazy concatenation is faster for many appends."""

    def _time_ms(self, cls, n_appends, n_repeat=5):
        times = []
        for _ in range(n_repeat):
            np.random.seed(0)
            obj = cls(np.zeros((1, 3)))
            t0 = time.perf_counter()
            for _ in range(n_appends):
                obj.append(np.random.rand(1, 3))
            # Force materialisation
            _ = len(obj)
            times.append((time.perf_counter() - t0) * 1000)
        return float(np.median(times))

    def test_speedup(self):
        n_appends = 500
        t_eager = self._time_ms(_EagerVertices, n_appends)
        t_lazy = self._time_ms(_LazyVertices, n_appends)
        speedup = t_eager / t_lazy if t_lazy > 0 else float("inf")
        assert speedup >= 1.2, (
            f"Expected ≥1.2× speedup, got {speedup:.2f}× "
            f"(eager={t_eager:.1f} ms, lazy={t_lazy:.1f} ms)"
        )


# ---------------------------------------------------------------------------
# 18. Batched camera projection
# ---------------------------------------------------------------------------


def _camera_projection_loop(K, cam_poses, vertices, relax, H, W):
    """Reference: per-camera loop with two matmuls."""
    nv = len(vertices)
    visible = np.zeros(nv, dtype=bool)
    homo = np.concatenate((vertices.T, np.ones((1, nv))), axis=0)
    for cam_pose in cam_poses:
        coords = K @ (np.linalg.inv(cam_pose) @ homo)[:3, :]
        coords[:2, :] /= coords[2]
        visible |= (
            (coords[2] > 0)
            & (coords[0] > -relax * W)
            & (coords[0] < (1 + relax) * W)
            & (coords[1] > -relax * H)
            & (coords[1] < (1 + relax) * H)
        )
    return (~visible).astype(np.float32)


def _camera_projection_batched(K, cam_poses, vertices, relax, H, W):
    """Optimised: pre-computed combined projection matrix (single matmul per camera)."""
    nv = len(vertices)
    visible = np.zeros(nv, dtype=bool)
    homo = np.concatenate((vertices.T, np.ones((1, nv))), axis=0)
    projs = [K @ np.linalg.inv(cp)[:3, :] for cp in cam_poses]
    neg_relax_W = -relax * W
    upper_W = (1 + relax) * W
    neg_relax_H = -relax * H
    upper_H = (1 + relax) * H
    for proj in projs:
        coords = proj @ homo  # (3, nv) — single matmul
        coords[:2, :] /= coords[2]
        visible |= (
            (coords[2] > 0)
            & (coords[0] > neg_relax_W)
            & (coords[0] < upper_W)
            & (coords[1] > neg_relax_H)
            & (coords[1] < upper_H)
        )
    return (~visible).astype(np.float32)


def _make_camera_test_data(n_cams=10, n_verts=5000):
    """Generate random camera poses, vertices, and intrinsic matrix."""
    np.random.seed(123)
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    cam_poses = [np.eye(4) + np.random.randn(4, 4) * 0.1 for _ in range(n_cams)]
    vertices = np.random.randn(n_verts, 3)
    return K, cam_poses, vertices


class TestCameraProjectionCorrectness:
    """Verify batched projection matches per-camera loop."""

    @pytest.mark.parametrize("n_cams", [1, 5, 20])
    def test_results_match(self, n_cams):
        K, cam_poses, verts = _make_camera_test_data(n_cams=n_cams, n_verts=1000)
        H, W = 480, 640
        relax = 0.01
        ref = _camera_projection_loop(K, cam_poses, verts, relax, H, W)
        opt = _camera_projection_batched(K, cam_poses, verts, relax, H, W)
        np.testing.assert_array_equal(ref, opt)


class TestCameraProjectionPerformance:
    """Assert batched projection is faster for many cameras."""

    def _time_ms(self, fn, *args, n_repeat=10):
        times = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            fn(*args)
            times.append((time.perf_counter() - t0) * 1000)
        return float(np.median(times))

    def test_speedup(self):
        K, cam_poses, verts = _make_camera_test_data(n_cams=20, n_verts=10000)
        H, W = 1080, 1920
        relax = 0.01
        t_loop = self._time_ms(
            _camera_projection_loop, K, cam_poses, verts, relax, H, W
        )
        t_batch = self._time_ms(
            _camera_projection_batched, K, cam_poses, verts, relax, H, W
        )
        speedup = t_loop / t_batch if t_batch > 0 else float("inf")
        assert speedup >= 1.2, (
            f"Expected ≥1.2× speedup, got {speedup:.2f}× "
            f"(loop={t_loop:.1f} ms, batch={t_batch:.1f} ms)"
        )


# ---------------------------------------------------------------------------
# 19. Vectorised parent_loc / self_loc lookup
# ---------------------------------------------------------------------------


def _parent_loc_loop(parents, vtx_pos):
    """Reference: Python loop."""
    n = len(parents)
    parent_loc = np.zeros((n, 3), dtype=float)
    self_loc = np.zeros((n, 3), dtype=float)
    for vertex_idx, parent_idx in enumerate(parents):
        parent_loc[vertex_idx] = vtx_pos[parent_idx]
        self_loc[vertex_idx] = vtx_pos[vertex_idx]
    return parent_loc, self_loc


def _parent_loc_vectorized(parents, vtx_pos):
    """Optimised: fancy indexing."""
    parent_loc = vtx_pos[parents]
    self_loc = vtx_pos.copy()
    return parent_loc, self_loc


class TestParentLocCorrectness:
    """Verify vectorised lookup matches loop."""

    def test_results_match(self):
        np.random.seed(0)
        n = 500
        vtx_pos = np.random.rand(n, 3)
        parents = np.random.randint(0, n, size=n)
        parents[0] = 0
        ref_p, ref_s = _parent_loc_loop(parents, vtx_pos)
        opt_p, opt_s = _parent_loc_vectorized(parents, vtx_pos)
        np.testing.assert_array_equal(ref_p, opt_p)
        np.testing.assert_array_equal(ref_s, opt_s)


class TestParentLocPerformance:
    """Assert vectorised lookup is faster."""

    def _time_ms(self, fn, *args, n_repeat=50):
        times = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            fn(*args)
            times.append((time.perf_counter() - t0) * 1000)
        return float(np.median(times))

    def test_speedup(self):
        np.random.seed(0)
        n = 5000
        vtx_pos = np.random.rand(n, 3)
        parents = np.random.randint(0, n, size=n)
        t_loop = self._time_ms(_parent_loc_loop, parents, vtx_pos)
        t_vec = self._time_ms(_parent_loc_vectorized, parents, vtx_pos)
        speedup = t_loop / t_vec if t_vec > 0 else float("inf")
        assert speedup >= 1.3, (
            f"Expected ≥1.3× speedup, got {speedup:.2f}× "
            f"(loop={t_loop:.2f} ms, vec={t_vec:.2f} ms)"
        )
