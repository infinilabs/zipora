#!/usr/bin/env python3
"""
Phase B Rule B3: Performance Regression Gate (`scripts/bench_compare.py`).

Compares a current benchmark JSON report (`target/bench/core_bench.json` or
`target/bench/e2e_search.json`) against the committed per-host baseline under
`benches/baselines/<host>/` and exits non-zero if any benchmark's median latency
regresses by more than `--max-regression-pct` (default: 5.0%).
"""

import argparse
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def detect_short_host() -> str:
    try:
        host = Path("/etc/hostname").read_text(encoding="utf-8").strip()
    except OSError:
        host = os.environ.get("HOSTNAME") or socket.gethostname()
    return host.split(".")[0].strip() or "unknown"


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        print(f"ERROR: benchmark JSON file not found: {path}", file=sys.stderr)
        sys.exit(2)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compare_benchmarks(
    baseline_data: Dict[str, Any],
    current_data: Dict[str, Any],
    max_regression_pct: float,
    abs_noise_floor_ns: float,
) -> Tuple[int, List[str]]:
    base_benches = {b["name"]: b for b in baseline_data.get("benchmarks", [])}
    curr_benches = {b["name"]: b for b in current_data.get("benchmarks", [])}

    failures: List[str] = []
    rows: List[Tuple[str, str, float, float, float, str]] = []

    for name, curr in curr_benches.items():
        unit = curr.get("unit", "ns/op")
        curr_med = float(curr["median_ns_per_op"])
        if name not in base_benches:
            rows.append((name, unit, 0.0, curr_med, 0.0, "NEW"))
            continue

        base = base_benches[name]
        base_med = float(base["median_ns_per_op"])
        abs_diff = curr_med - base_med
        if base_med > 0.0:
            pct_diff = (abs_diff / base_med) * 100.0
        else:
            pct_diff = 0.0

        # A regression requires both relative increase > max_regression_pct
        # and absolute increase > abs_noise_floor_ns (guards sub-ns quantization).
        if pct_diff > max_regression_pct and abs_diff > abs_noise_floor_ns:
            status = "REGRESSED"
            failures.append(
                f"{name}: baseline={base_med:.4f} {unit}, current={curr_med:.4f} {unit} "
                f"({pct_diff:+.2f}% > +{max_regression_pct:.2f}%)"
            )
        elif pct_diff < -max_regression_pct and (-abs_diff) > abs_noise_floor_ns:
            status = "IMPROVED"
        else:
            status = "OK"

        rows.append((name, unit, base_med, curr_med, pct_diff, status))

    # Print summary table
    print(
        f"{'Benchmark':<52} {'Unit':<8} {'Baseline':>12} {'Current':>12} {'Delta %':>10} {'Status':<10}"
    )
    print("-" * 108)
    for name, unit, base_med, curr_med, pct_diff, status in rows:
        print(
            f"{name:<52} {unit:<8} {base_med:>12.4f} {curr_med:>12.4f} {pct_diff:>+9.2f}% {status:<10}"
        )

    # Also compare e2e_search summary metrics if present
    if "summary" in baseline_data and "summary" in current_data:
        base_sum = baseline_data["summary"]
        curr_sum = current_data["summary"]
        print("\n=== E2E Search Summary Comparison ===")
        for metric, higher_is_better in [
            ("zipora_qps_top10", True),
            ("zipora_p50_us", False),
            ("zipora_p99_us", False),
            ("zipora_bytes_per_posting", False),
        ]:
            if metric in base_sum and metric in curr_sum:
                b_val = float(base_sum[metric])
                c_val = float(curr_sum[metric])
                pct = ((c_val - b_val) / b_val * 100.0) if b_val > 0 else 0.0
                regressed = (
                    (pct < -max_regression_pct)
                    if higher_is_better
                    else (pct > max_regression_pct)
                )
                status = "REGRESSED" if regressed else "OK"
                print(
                    f"  {metric:<30} baseline={b_val:>10.3f}  current={c_val:>10.3f}  ({pct:+.2f}%) [{status}]"
                )
                if regressed:
                    failures.append(
                        f"summary.{metric}: baseline={b_val:.3f}, current={c_val:.3f} ({pct:+.2f}%)"
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
        "--max-regression-pct",
        type=float,
        default=5.0,
        help="Maximum allowed median regression percentage (default: 5.0)",
    )
    parser.add_argument(
        "--abs-noise-floor-ns",
        type=float,
        default=0.05,
        help="Minimum absolute difference in ns/op required to flag a regression (default: 0.05)",
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

    print(f"=== Perf Regression Gate (B3) — suite={args.suite}, host={host} ===")
    print(f"  Baseline : {baseline_path}")
    print(f"  Current  : {current_path}")
    print(f"  Threshold: +{args.max_regression_pct:.2f}% median regression\n")

    baseline_data = load_json(baseline_path)
    current_data = load_json(current_path)

    total, failures = compare_benchmarks(
        baseline_data,
        current_data,
        args.max_regression_pct,
        args.abs_noise_floor_ns,
    )

    print("-" * 108)
    if failures:
        print(
            f"FAIL: {len(failures)}/{total} benchmark(s) regressed beyond +{args.max_regression_pct:.2f}%:"
        )
        for msg in failures:
            print(f"  - {msg}")
        return 1

    print(
        f"PASS: all {total} benchmark(s) within +{args.max_regression_pct:.2f}% median threshold."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
