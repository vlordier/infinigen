#!/usr/bin/env python3
# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""
Verify benchmark output equivalence (optimized vs upstream baseline paths).

This script compares *outputs* of each benchmark implementation rather than
timings. It monkeypatches ``tests/run_benchmarks.py::_median_time`` so each
benchmark executes once and returns the computed result object, then checks
optimized and baseline outputs for equality (or near-equality for floats).

Usage:

    python tests/verify_benchmark_outputs.py

    python tests/verify_benchmark_outputs.py \
      --rtol 1e-6 --atol 1e-8 \
      --output /tmp/benchmark_output_check.json

    python tests/verify_benchmark_outputs.py \
      --benchmarks unique_rows,heightmap_grid,mesh_cat
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _load_runner_module():
    """Load tests/run_benchmarks.py as a module."""
    runner_path = Path(__file__).resolve().parent / "run_benchmarks.py"
    spec = importlib.util.spec_from_file_location("run_benchmarks", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark runner from {runner_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _as_numpy(x: Any) -> np.ndarray | None:
    """Convert array-like / torch tensors to NumPy arrays when possible."""
    if isinstance(x, np.ndarray):
        return x

    # Handle torch tensors without importing torch eagerly.
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        try:
            return x.detach().cpu().numpy()
        except Exception:
            return None

    return None


@dataclass
class CompareResult:
    ok: bool
    reason: str
    max_abs_diff: float | None = None


def _compare_values(a: Any, b: Any, rtol: float, atol: float, path: str = "root") -> CompareResult:
    """Recursively compare two values with numeric tolerance."""
    # Fast path for identity / exact equality where safe.
    if a is b:
        return CompareResult(True, "identical")

    # Dicts
    if isinstance(a, dict) and isinstance(b, dict):
        ka, kb = set(a.keys()), set(b.keys())
        if ka != kb:
            return CompareResult(False, f"{path}: key sets differ")
        max_diff = 0.0
        for k in sorted(ka):
            res = _compare_values(a[k], b[k], rtol, atol, f"{path}.{k}")
            if not res.ok:
                return res
            if res.max_abs_diff is not None:
                max_diff = max(max_diff, res.max_abs_diff)
        return CompareResult(True, "dict match", max_abs_diff=max_diff)

    # Sequences
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return CompareResult(False, f"{path}: sequence lengths differ")
        max_diff = 0.0
        for i, (av, bv) in enumerate(zip(a, b)):
            res = _compare_values(av, bv, rtol, atol, f"{path}[{i}]")
            if not res.ok:
                return res
            if res.max_abs_diff is not None:
                max_diff = max(max_diff, res.max_abs_diff)
        return CompareResult(True, "sequence match", max_abs_diff=max_diff)

    # NumPy / tensor arrays
    a_np = _as_numpy(a)
    b_np = _as_numpy(b)
    if a_np is not None and b_np is not None:
        if a_np.shape != b_np.shape:
            return CompareResult(False, f"{path}: shape mismatch {a_np.shape} != {b_np.shape}")

        # Numeric comparison
        if np.issubdtype(a_np.dtype, np.number) and np.issubdtype(b_np.dtype, np.number):
            if a_np.size == 0:
                return CompareResult(True, "empty numeric arrays")
            abs_diff = np.abs(a_np.astype(np.float64) - b_np.astype(np.float64))
            max_abs = float(np.nanmax(abs_diff))
            if np.allclose(a_np, b_np, rtol=rtol, atol=atol, equal_nan=True):
                return CompareResult(True, "numeric arrays close", max_abs_diff=max_abs)
            return CompareResult(False, f"{path}: numeric arrays differ", max_abs_diff=max_abs)

        # Non-numeric array comparison
        if np.array_equal(a_np, b_np, equal_nan=True):
            return CompareResult(True, "arrays equal")
        return CompareResult(False, f"{path}: arrays differ")

    # Scalar numbers
    if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
        av = float(a)
        bv = float(b)
        if math.isclose(av, bv, rel_tol=rtol, abs_tol=atol):
            return CompareResult(True, "scalars close", max_abs_diff=abs(av - bv))
        return CompareResult(False, f"{path}: scalar mismatch {av} != {bv}", max_abs_diff=abs(av - bv))

    # Fallback exact match for opaque objects; guard against array-like `==` behavior.
    try:
        eq = a == b
    except Exception:
        eq = False

    if isinstance(eq, np.ndarray):
        if eq.shape == () and bool(eq):
            return CompareResult(True, "exact match")
    elif isinstance(eq, (bool, np.bool_)) and bool(eq):
        return CompareResult(True, "exact match")

    return CompareResult(False, f"{path}: values differ ({type(a).__name__} vs {type(b).__name__})")


def _capture_benchmark_output(runner_mod, bench_fn, upstream: bool):
    """Run one benchmark function and capture the underlying output object."""
    captured: list[Any] = []
    orig_median_time = runner_mod._median_time

    def _capture_once(fn, n_repeat=1):
        out = fn()
        captured.append(out)
        return 0.0

    runner_mod._median_time = _capture_once
    try:
        key, _ = bench_fn(upstream=upstream)
        if captured:
            return key, captured[0], None
        return key, None, "no output captured (likely skipped)"
    except Exception as e:  # pragma: no cover - diagnostic path
        key = getattr(bench_fn, "__name__", "unknown").replace("bench_", "")
        return key, None, str(e)
    finally:
        runner_mod._median_time = orig_median_time


def _force_torch_cpu_if_available(runner_mod):
    """Force CPU torch device for deterministic numeric parity on Apple Silicon."""
    orig_torch_device = runner_mod._torch_device

    def _cpu_device():
        torch = runner_mod._get_torch()
        if torch is None:
            return None
        return torch.device("cpu")

    runner_mod._torch_device = _cpu_device
    return orig_torch_device


def main():
    parser = argparse.ArgumentParser(description="Verify benchmark output equivalence")
    parser.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-8, help="Absolute tolerance")
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="",
        help="Comma-separated benchmark keys to run (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional JSON report output path",
    )
    args = parser.parse_args()

    runner = _load_runner_module()
    selected = {b.strip() for b in args.benchmarks.split(",") if b.strip()}

    orig_torch_device = _force_torch_cpu_if_available(runner)
    try:
        rows = []
        pass_count = 0
        fail_count = 0
        skip_count = 0

        print("Verifying benchmark outputs (optimized vs upstream baseline)\n")
        for bench_fn in runner.ALL_BENCHMARKS:
            probe_key = bench_fn.__name__.replace("bench_", "")
            if selected and probe_key not in selected:
                continue

            key_opt, out_opt, err_opt = _capture_benchmark_output(runner, bench_fn, upstream=False)
            key_up, out_up, err_up = _capture_benchmark_output(runner, bench_fn, upstream=True)
            key = key_opt or key_up

            if err_opt or err_up or out_opt is None or out_up is None:
                skip_count += 1
                reason = f"opt_err={err_opt!r}, up_err={err_up!r}"
                rows.append({"benchmark": key, "status": "SKIP", "reason": reason})
                print(f"  {key:<24s} SKIP  {reason}")
                continue

            res = _compare_values(out_opt, out_up, rtol=args.rtol, atol=args.atol)
            if res.ok:
                pass_count += 1
                rows.append(
                    {
                        "benchmark": key,
                        "status": "PASS",
                        "reason": res.reason,
                        "max_abs_diff": res.max_abs_diff,
                    }
                )
                diff_str = f" max_abs_diff={res.max_abs_diff:.3e}" if res.max_abs_diff is not None else ""
                print(f"  {key:<24s} PASS  {res.reason}{diff_str}")
            else:
                fail_count += 1
                rows.append(
                    {
                        "benchmark": key,
                        "status": "FAIL",
                        "reason": res.reason,
                        "max_abs_diff": res.max_abs_diff,
                    }
                )
                diff_str = f" max_abs_diff={res.max_abs_diff:.3e}" if res.max_abs_diff is not None else ""
                print(f"  {key:<24s} FAIL  {res.reason}{diff_str}")

        summary = {
            "pass": pass_count,
            "fail": fail_count,
            "skip": skip_count,
            "rtol": args.rtol,
            "atol": args.atol,
            "rows": rows,
        }

        print("\nSummary:")
        print(f"  pass={pass_count} fail={fail_count} skip={skip_count}")

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(summary, indent=2))
            print(f"  wrote report: {out_path}")

        # Non-zero exit if any functional mismatch.
        raise SystemExit(1 if fail_count > 0 else 0)
    finally:
        runner._torch_device = orig_torch_device


if __name__ == "__main__":
    main()
