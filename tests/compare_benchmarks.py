#!/usr/bin/env python3
# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""
Compare benchmark results from PR branch vs upstream and produce a Markdown
report showing cross-branch speedups with ASCII bar charts.

Each benchmark key exists in *both* JSON files (PR uses the optimised
implementation, upstream uses the baseline).  The speedup is::

    speedup = upstream_time / pr_time

Usage::

    python tests/compare_benchmarks.py \\
        --pr /tmp/pr_results.json \\
        --upstream /tmp/upstream_results.json \\
        --output /tmp/benchmark_report.md
"""

import argparse
import json
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Benchmark keys and human-friendly labels for the speedup chart.
# Each entry: (json_key, display_label)
# ---------------------------------------------------------------------------

BENCHMARKS = [
    ("unique_rows", "unique_rows (void-view vs np.unique)"),
    ("heightmap_grid", "heightmap grid (vectorised vs loop)"),
    ("mesh_cat", "mesh cat (pre-alloc vs vstack)"),
    ("tree_vertices", "tree vertices (lazy vs eager)"),
    ("projection", "projection (combined K vs separate)"),
    ("distance_transform", "distance transform (EDT vs loop)"),
    ("boundary_detection", "boundary (vectorised vs loop)"),
    ("surface_normals", "normals (vectorised vs loop)"),
    ("full_pipeline", "⭐ full pipeline (composite)"),
]

BAR_WIDTH = 40


def _bar(value: float, max_value: float, width: int = BAR_WIDTH) -> str:
    """Render a horizontal bar using Unicode block characters."""
    if max_value <= 0 or math.isnan(value) or value <= 0:
        return "N/A"
    ratio = min(value / max_value, 1.0)
    full_blocks = int(ratio * width)
    remainder = (ratio * width) - full_blocks
    bar_str = "█" * full_blocks
    if remainder >= 0.5 and full_blocks < width:
        bar_str += "▌"
    return bar_str


def _bar_log(value: float, max_value: float, width: int = BAR_WIDTH) -> str:
    """Render a bar using log₁₀ scale for wide speedup ranges."""
    if max_value <= 1 or value <= 0 or math.isnan(value):
        return "N/A"
    log_val = math.log10(max(value, 1.0))
    log_max = math.log10(max_value)
    if log_max <= 0:
        return "N/A"
    ratio = min(log_val / log_max, 1.0)
    full_blocks = int(ratio * width)
    remainder = (ratio * width) - full_blocks
    bar_str = "█" * full_blocks
    if remainder >= 0.5 and full_blocks < width:
        bar_str += "▌"
    return bar_str


def _speedup_emoji(speedup: float) -> str:
    # Thresholds: 10× = exceptional, 3× = significant, 1.5× = meaningful
    if speedup >= 10:
        return "🚀"
    if speedup >= 3:
        return "⚡"
    if speedup >= 1.5:
        return "✅"
    if speedup >= 0.8:
        return "➖"
    return "⚠️"


def _geometric_mean(values):
    """Geometric mean of positive floats."""
    positive = [v for v in values if v > 0]
    if not positive:
        return float("nan")
    log_sum = sum(math.log(v) for v in positive)
    return math.exp(log_sum / len(positive))


def build_report(pr: dict, upstream: dict) -> str:
    """Build the full Markdown benchmark report."""
    lines = []
    lines.append("## 📊 Benchmark Results — PR vs Upstream (Princeton main)")
    lines.append("")
    lines.append(
        "> Cross-branch comparison: same benchmark name runs optimised code on the PR "
        "and baseline code on upstream.  Speedup = upstream_time / pr_time."
    )
    lines.append("")

    # ── Section 1: Cross-branch speedup chart ────────────────────────────
    lines.append("### Speedups (PR optimised vs Upstream baseline)")
    lines.append("")
    lines.append("```")
    lines.append(
        f"{'Benchmark':<40s} {'Speedup':>8s}  "
        f"{'Bar':40s}  {'PR (ms)':>10s}  {'Upstream (ms)':>14s}"
    )
    lines.append("─" * 120)

    # Pre-compute speedups for consistent bar scaling
    speedups = []
    row_data = []  # (label, speedup, pr_ms, up_ms) or None
    for key, label in BENCHMARKS:
        pr_t = pr.get(key)
        up_t = upstream.get(key)

        if pr_t is None or up_t is None or pr_t <= 0:
            row_data.append(None)
            continue

        speedup = up_t / pr_t
        speedups.append(speedup)
        row_data.append((label, speedup, pr_t * 1000, up_t * 1000))

    bar_max = max(speedups + [10]) if speedups else 10
    min_sp = min((s for s in speedups if s > 0), default=1)
    use_log = bar_max / max(min_sp, 1e-9) > 100

    for idx, entry in enumerate(row_data):
        if entry is None:
            label = BENCHMARKS[idx][1]
            lines.append(
                f"{label:<40s} {'N/A':>8s}  {'(benchmark skipped)':40s}"
            )
            continue
        label, speedup, pr_ms, up_ms = entry
        if use_log:
            bar = _bar_log(speedup, bar_max, width=BAR_WIDTH)
        else:
            bar = _bar(speedup, bar_max, width=BAR_WIDTH)
        emoji = _speedup_emoji(speedup)
        lines.append(
            f"{label:<40s} {speedup:>7.1f}×  "
            f"{bar:40s}  {pr_ms:>9.2f}  {up_ms:>13.2f}  {emoji}"
        )

    lines.append("```")
    lines.append("")

    # ── Overall geometric-mean speedup ───────────────────────────────────
    if speedups:
        geo_mean = _geometric_mean(speedups)
        scale_note = ", log₁₀ scale" if use_log else ""
        lines.append(
            f"**Overall geometric-mean speedup: {geo_mean:.1f}×** "
            f"(across {len(speedups)} benchmarks{scale_note})"
        )
        lines.append("")

    # ── Section 2: Raw timings table ─────────────────────────────────────
    lines.append("### Raw Timings")
    lines.append("")
    lines.append("| Benchmark | PR (ms) | Upstream (ms) | Speedup | Δ |")
    lines.append("|:----------|--------:|--------------:|--------:|:-:|")

    all_keys = sorted(set(list(pr.keys()) + list(upstream.keys())))
    for key in all_keys:
        pr_val = pr.get(key)
        up_val = upstream.get(key)
        pr_str = f"{pr_val * 1000:.2f}" if pr_val is not None else "—"
        up_str = f"{up_val * 1000:.2f}" if up_val is not None else "—"
        if (
            pr_val is not None
            and up_val is not None
            and pr_val > 0
            and up_val > 0
        ):
            sp = up_val / pr_val
            sp_str = f"{sp:.1f}×"
            if sp >= 1.5:
                delta = "🟢 faster"
            elif sp <= 0.67:
                delta = "🔴 slower"
            else:
                delta = "⚪ ~same"
        else:
            sp_str = "—"
            delta = "—"
        lines.append(
            f"| {key} | {pr_str} | {up_str} | {sp_str} | {delta} |"
        )

    lines.append("")
    lines.append(
        "*Generated automatically by CI. "
        "Measurements on GitHub Actions runners may have noise; "
        "relative comparisons are more meaningful than absolute times.*"
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results")
    parser.add_argument("--pr", required=True, help="PR benchmark results JSON")
    parser.add_argument(
        "--upstream", required=True, help="Upstream benchmark results JSON"
    )
    parser.add_argument("--output", required=True, help="Output Markdown file")
    args = parser.parse_args()

    with open(args.pr) as f:
        pr = json.load(f)
    with open(args.upstream) as f:
        upstream = json.load(f)

    report = build_report(pr, upstream)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    print(report)
    print(f"\nReport written to {output_path}")


if __name__ == "__main__":
    main()
