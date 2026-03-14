#!/usr/bin/env python3
# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""
Compare benchmark results from PR branch vs upstream and produce a Markdown
report with detailed ASCII horizontal bar charts.

Usage:
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
# Paired benchmarks: (optimised_key, baseline_key, friendly_label)
# These pairs let us compute speedup ratios.
# NOTE: chunked_concat is intentionally excluded — np.concatenate holds the
# GIL so threading cannot outperform a single call.  Both timings still
# appear in the Raw Timings table.
# ---------------------------------------------------------------------------

PAIRS = [
    ("unique_rows (optimised)", "np.unique axis=0 baseline", "unique_rows (void-view)"),
    ("meshgrid vectorised 256x256", "meshgrid loop 256x256", "meshgrid vectorised"),
    (
        "tree vertices lazy concat",
        "tree vertices eager append",
        "tree vertices (lazy)",
    ),
    ("projection precomputed", "projection separate steps", "projection (combined K)"),
    (
        "distance_transform_edt 64x64",
        "grid_distance loop 64x64",
        "distance_transform (EDT)",
    ),
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
    """Render a bar using log scale — better when speedups span many orders of magnitude."""
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
    if speedup >= 5:
        return "🚀"
    if speedup >= 2:
        return "⚡"
    if speedup >= 1.2:
        return "✅"
    if speedup >= 0.8:
        return "➖"
    return "⚠️"


def build_report(pr: dict, upstream: dict) -> str:
    """Build the full Markdown benchmark report."""
    lines = []
    lines.append("## 📊 Benchmark Results — PR vs Upstream (Princeton main)")
    lines.append("")
    lines.append(
        "> Automated micro-benchmark comparison.  "
        "Lower times are better; speedup > 1× means the PR is faster."
    )
    lines.append("")

    # ── Section 1: Paired speedup chart ──────────────────────────────────
    lines.append("### Optimisation Speedups")
    lines.append("")
    lines.append("```")
    lines.append(
        f"{'Benchmark':<30s} {'Speedup':>8s}  {'Bar (vs baseline)':40s}  {'PR (ms)':>10s}  {'Base (ms)':>10s}"
    )
    lines.append("─" * 105)

    # Pre-compute all speedups so bars use a consistent max across rows
    speedups = []
    pair_data = []  # (label, speedup, pr_opt, pr_base) or None for skipped
    for opt_key, base_key, label in PAIRS:
        pr_opt = pr.get(opt_key)
        pr_base = pr.get(base_key)

        if pr_opt is None or pr_base is None:
            pair_data.append(None)
            continue

        if pr_opt <= 0:
            speedup = float("inf")
        else:
            speedup = pr_base / pr_opt

        speedups.append(speedup)
        pair_data.append((label, speedup, pr_opt, pr_base))

    bar_max = max(speedups + [10]) if speedups else 10
    min_sp = min((s for s in speedups if s > 0), default=1)
    # Use log scale when the range spans more than 2 orders of magnitude
    use_log_s1 = bar_max / max(min_sp, 1e-9) > 100

    for entry in pair_data:
        if entry is None:
            label = PAIRS[pair_data.index(entry)][2]
            lines.append(f"{label:<30s} {'N/A':>8s}  {'(benchmark skipped)':40s}")
            continue
        label, speedup, pr_opt, pr_base = entry
        if use_log_s1:
            bar = _bar_log(speedup, bar_max, width=BAR_WIDTH)
        else:
            bar = _bar(speedup, bar_max, width=BAR_WIDTH)
        emoji = _speedup_emoji(speedup)
        lines.append(
            f"{label:<30s} {speedup:>7.1f}×  {bar:40s}  {pr_opt*1000:>9.2f}  {pr_base*1000:>9.2f}  {emoji}"
        )

    lines.append("```")
    lines.append("")

    # Re-render with consistent max for better visual comparison
    if speedups:
        max_sp = max(speedups)
        min_sp = min(s for s in speedups if s > 0)
        # Use log scale when the range spans more than 2 orders of magnitude
        use_log = max_sp / max(min_sp, 1e-9) > 100
        scale_label = "log₁₀ scale" if use_log else "linear"
        lines_chart = []
        lines_chart.append("### Detailed Speedup Chart")
        lines_chart.append("")
        lines_chart.append("```")
        lines_chart.append(
            f"{'Benchmark':<30s}  |  Speedup bar (max = {max_sp:.0f}×, {scale_label})"
        )
        lines_chart.append("─" * 80)

        idx = 0
        for opt_key, base_key, label in PAIRS:
            pr_opt = pr.get(opt_key)
            pr_base = pr.get(base_key)
            if pr_opt is None or pr_base is None:
                lines_chart.append(f"{label:<30s}  | (skipped)")
                continue
            sp = speedups[idx]
            idx += 1
            if use_log:
                bar = _bar_log(sp, max_sp, width=BAR_WIDTH)
            else:
                bar = _bar(sp, max_sp, width=BAR_WIDTH)
            lines_chart.append(f"{label:<30s}  |{bar} {sp:.1f}×")

        lines_chart.append("```")
        lines_chart.append("")
        lines.extend(lines_chart)

    # ── Section 2: Raw timings table ─────────────────────────────────────
    lines.append("### Raw Timings")
    lines.append("")
    lines.append("| Benchmark | PR (ms) | Upstream (ms) | Δ |")
    lines.append("|:----------|--------:|--------------:|:-:|")

    all_keys = sorted(set(list(pr.keys()) + list(upstream.keys())))
    for key in all_keys:
        pr_val = pr.get(key)
        up_val = upstream.get(key)
        pr_str = f"{pr_val*1000:.2f}" if pr_val is not None else "—"
        up_str = f"{up_val*1000:.2f}" if up_val is not None else "—"
        if pr_val is not None and up_val is not None and up_val > 0:
            ratio = pr_val / up_val
            if ratio < 0.8:
                delta = "🟢 faster"
            elif ratio > 1.2:
                delta = "🔴 slower"
            else:
                delta = "⚪ ~same"
        else:
            delta = "—"
        lines.append(f"| {key} | {pr_str} | {up_str} | {delta} |")

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
