#!/usr/bin/env python3
"""
Phase B Rule B3: Performance Regression Gate (`scripts/bench_compare.py`).

Runner documentation (S37-R2 / S38-R4):
- Reference host: `bindiego` (AMD EPYC 7B13, 64 logical cores across 4 Zen 3 CCDs)
- Pinned CPU core: self-pinned (`pid = 0` only) via `select_primary_core` (default core 23
  on interior CCD 2 or quietest physical core from read-only `/proc/stat`; override via `ZIPORA_BENCH_PIN_CORE`)
- Compiler flags:
    RUSTFLAGS="-C target-cpu=haswell -C target-feature=+bmi2,+avx2,+popcnt -C codegen-units=1 -C llvm-args=-align-loops=32"
- Exact command:
    make bench_core
    make bench_e2e_search
- Baselines under `benches/baselines/<host>/` are never auto-written by the gate;
  set `UPDATE_BASELINE=1` explicitly when recording or updating a baseline.

Gate rules:
1. Fails (exit 1) if no baseline file exists for the host.
2. Fails (exit 1) if any benchmark present in the baseline is missing from the
   current run or ran with a different `dataset_size`.
3. Compares `min_ns_per_op` (true minimum across trials, selectable via `--metric`)
   against the per-benchmark `max_regression_pct` recorded in the baseline
   (default +5.00%; +9.00% on spec-size DRAM/allocator-bound benchmarks),
   and prints both min and median alongside trial `spread_pct`.
"""

import argparse
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def detect_short_host() -> str:
    try:
        host = Path("/etc/hostname").read_text(encoding="utf-8").strip()
    except OSError:
        host = os.environ.get("HOSTNAME") or socket.gethostname()
    return host.split(".")[0].strip() or "unknown"


def load_json(path: Path, is_baseline: bool = False, host: str = "") -> Dict[str, Any]:
    if not path.exists():
        if is_baseline:
            print(
                f"ERROR: no baseline found for host '{host}' at {path}.\n"
                f"Baselines are never auto-created by the gate. Run with UPDATE_BASELINE=1 to record one explicitly.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"ERROR: benchmark JSON file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compare_benchmarks(
    baseline_data: Dict[str, Any],
    current_data: Dict[str, Any],
    default_max_regression_pct: float,
    override_max_regression_pct: Optional[float],
    abs_noise_floor_ns: float,
    metric: str,
) -> Tuple[int, List[str]]:
    base_list = baseline_data.get("benchmarks", [])
    curr_list = current_data.get("benchmarks", [])
    base_benches = {b["name"]: b for b in base_list}
    curr_benches = {b["name"]: b for b in curr_list}

    metric_field = "min_ns_per_op" if metric == "min" else "median_ns_per_op"

    failures: List[str] = []
    rows: List[Tuple[str, str, float, float, float, float, float, float, float, str]] = []

    # 1. Check all baseline benchmarks in order (catches missing benchmarks!)
    for base in base_list:
        name = base["name"]
        unit = base.get("unit", "ns/op")
        base_min = float(base.get("min_ns_per_op", base["median_ns_per_op"]))
        base_med = float(base["median_ns_per_op"])
        base_val = float(base.get(metric_field, base_med))
        thresh_pct = (
            override_max_regression_pct
            if override_max_regression_pct is not None
            else float(base.get("max_regression_pct", default_max_regression_pct))
        )

        if name not in curr_benches:
            rows.append(
                (name, unit, base_min, 0.0, base_med, 0.0, 0.0, thresh_pct, 0.0, "MISSING")
            )
            failures.append(
                f"{name}: MISSING from current benchmark run (present in baseline)"
            )
            continue

        curr = curr_benches[name]
        base_ds = base.get("dataset_size")
        curr_ds = curr.get("dataset_size")
        if base_ds is not None and curr_ds is not None and int(base_ds) != int(curr_ds):
            failures.append(
                f"{name}: dataset_size mismatch (baseline={base_ds}, current={curr_ds})"
            )

        curr_min = float(curr.get("min_ns_per_op", curr["median_ns_per_op"]))
        curr_med = float(curr["median_ns_per_op"])
        curr_spread = float(curr.get("spread_pct", 0.0))
        curr_val = float(curr.get(metric_field, curr_med))

        abs_diff = curr_val - base_val
        if base_val > 0.0:
            pct_diff = (abs_diff / base_val) * 100.0
        else:
            pct_diff = 0.0

        if pct_diff > thresh_pct and abs_diff > abs_noise_floor_ns:
            status = "REGRESSED"
            failures.append(
                f"{name}: baseline {metric}={base_val:.4f} {unit}, current {metric}={curr_val:.4f} {unit} "
                f"({pct_diff:+.2f}% > +{thresh_pct:.2f}%)"
            )
        elif pct_diff < -thresh_pct and (-abs_diff) > abs_noise_floor_ns:
            status = "IMPROVED"
        else:
            status = "OK"

        rows.append(
            (
                name,
                unit,
                base_min,
                curr_min,
                base_med,
                curr_med,
                curr_spread,
                thresh_pct,
                pct_diff,
                status,
            )
        )

    # 2. Report any newly added benchmarks not yet in baseline
    for curr in curr_list:
        name = curr["name"]
        if name not in base_benches:
            unit = curr.get("unit", "ns/op")
            curr_min = float(curr.get("min_ns_per_op", curr["median_ns_per_op"]))
            curr_med = float(curr["median_ns_per_op"])
            curr_spread = float(curr.get("spread_pct", 0.0))
            thresh_pct = float(curr.get("max_regression_pct", default_max_regression_pct))
            rows.append(
                (
                    name,
                    unit,
                    0.0,
                    curr_min,
                    0.0,
                    curr_med,
                    curr_spread,
                    thresh_pct,
                    0.0,
                    "NEW",
                )
            )

    print(
        f"{'Benchmark':<50} {'Unit':<8} {'Base Min':>10} {'Curr Min':>10} {'Base Med':>10} {'Curr Med':>10} {'Spread%':>8} {'Thresh%':>8} {'Delta%':>9} {'Status':<10}"
    )
    print("-" * 142)
    for (
        name,
        unit,
        base_min,
        curr_min,
        base_med,
        curr_med,
        curr_spread,
        thresh_pct,
        pct_diff,
        status,
    ) in rows:
        print(
            f"{name:<50} {unit:<8} {base_min:>10.4f} {curr_min:>10.4f} {base_med:>10.4f} {curr_med:>10.4f} {curr_spread:>7.2f}% {thresh_pct:>+7.2f}% {pct_diff:>+8.2f}% {status:<10}"
        )

    # Also compare e2e_search summary metrics if present
    if "summary" in baseline_data and "summary" in current_data:
        base_sum = baseline_data["summary"]
        curr_sum = current_data["summary"]
        print("\n=== E2E Search Summary Comparison ===")
        for metric_name, higher_is_better, sum_thresh in [
            ("zipora_qps_top10", True, max(default_max_regression_pct, 9.0)),
            ("zipora_p50_us", False, max(default_max_regression_pct, 9.0)),
            ("zipora_p99_us", False, max(default_max_regression_pct, 9.0)),
            ("zipora_bytes_per_posting", False, default_max_regression_pct),
            ("tantivy_qps_top10", True, max(default_max_regression_pct, 9.0)),
        ]:
            if metric_name in base_sum and metric_name in curr_sum:
                b_val = float(base_sum[metric_name])
                c_val = float(curr_sum[metric_name])
                pct = ((c_val - b_val) / b_val * 100.0) if b_val > 0 else 0.0
                eff_thresh = (
                    override_max_regression_pct
                    if override_max_regression_pct is not None
                    else sum_thresh
                )
                regressed = (
                    (pct < -eff_thresh) if higher_is_better else (pct > eff_thresh)
                )
                status = "REGRESSED" if regressed else "OK"
                print(
                    f"  {metric_name:<30} baseline={b_val:>10.3f}  current={c_val:>10.3f}  ({pct:+.2f}%, thresh={eff_thresh:.2f}%) [{status}]"
                )
                if regressed:
                    failures.append(
                        f"summary.{metric_name}: baseline={b_val:.3f}, current={c_val:.3f} ({pct:+.2f}%)"
                    )

    return len(rows), failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare benchmark JSON output against per-host baseline (Rule B3)."
    )
    parser.add_argument(
        "--suite",
        default="core_bench",
        help="Benchmark suite name (default: core_bench; or e2e_search)",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Host directory under benches/baselines/ (default: auto-detected short hostname)",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Explicit path to baseline JSON file",
    )
    parser.add_argument(
        "--current",
        default=None,
        help="Explicit path to current benchmark JSON file",
    )
    parser.add_argument(
        "--metric",
        choices=["min", "median"],
        default="min",
        help="Statistic used for regression gating (default: min of N trials; or median)",
    )
    parser.add_argument(
        "--max-regression-pct",
        type=float,
        default=None,
        help="Override per-benchmark maximum allowed regression percentage (default: use per-benchmark threshold or 5.0%%)",
    )
    parser.add_argument(
        "--abs-noise-floor-ns",
        type=float,
        default=0.005,
        help="Minimum absolute difference in ns/op required to flag a regression (default: 0.005)",
    )
    args = parser.parse_args()

    host = args.host or detect_short_host()
    baseline_path = (
        Path(args.baseline)
        if args.baseline
        else Path("benches") / "baselines" / host / f"{args.suite}.json"
    )
    current_path = (
        Path(args.current)
        if args.current
        else Path("target") / "bench" / f"{args.suite}.json"
    )

    default_thresh = (
        args.max_regression_pct if args.max_regression_pct is not None else 5.0
    )

    print(
        f"=== Perf Regression Gate (B3) — suite={args.suite}, host={host}, metric={args.metric} ==="
    )
    print(f"  Baseline : {baseline_path}")
    print(f"  Current  : {current_path}")
    print(
        f"  Threshold: +{default_thresh:.2f}% default (per-benchmark thresholds honored unless overridden)\n"
    )

    baseline_data = load_json(baseline_path, is_baseline=True, host=host)
    current_data = load_json(current_path, is_baseline=False, host=host)

    total, failures = compare_benchmarks(
        baseline_data,
        current_data,
        default_thresh,
        args.max_regression_pct,
        args.abs_noise_floor_ns,
        args.metric,
    )

    print("-" * 142)
    if failures:
        print(f"FAIL: {len(failures)} regression/integrity check(s) failed:")
        for msg in failures:
            print(f"  - {msg}")
        return 1

    print(
        f"PASS: all {total} benchmark(s) present and within regression thresholds ({args.metric})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
