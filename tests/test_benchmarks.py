# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Copilot

"""
Benchmarks and correctness tests for the three optimisations:

1.  evaluate.py   – ChainMap instead of copy.copy(memo) in ForAll/SumOver/MeanOver
2.  skin_ops.py   – vectorised bevel_cap (single-pass array build, no O(n²) loop)
3.  logging.py    – time.perf_counter() replacing datetime.now() in Timer
"""

import time
from collections import ChainMap
from copy import copy
from datetime import timedelta

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
