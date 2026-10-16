#!/usr/bin/env python3
"""
S39-R2: Compare Transparent Hugepages OFF (`PR_SET_THP_DISABLE`, default) vs
ON (`--thp-on` / `ZIPORA_BENCH_THP=on`) across `core_bench` and `e2e_search_bench`.

Usage:
  RUSTFLAGS="-C target-cpu=haswell -C target-feature=+bmi2,+avx2,+popcnt -C codegen-units=1 -C llvm-args=-align-loops=32" \\
    cargo bench --locked --bench core_bench -- --thp-on --output target/bench/core_bench_thp_on.json
  RUSTFLAGS="-C target-cpu=haswell -C target-feature=+bmi2,+avx2,+popcnt -C codegen-units=1 -C llvm-args=-align-loops=32" \\
    cargo bench --locked --bench e2e_search_bench -- --thp-on --output target/bench/e2e_search_thp_on.json
  python3 scripts/measure_thp_impact.py
"""

import argparse
import json
from pathlib import Path


def load_records(path: Path) -> dict:
    data = json.loads(path.read_text())
    return {b["name"]: b for b in data.get("benchmarks", [])}, data


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare THP-off vs THP-on benchmark JSON reports.")
    parser.add_argument(
        "--core-off",
        default="benches/baselines/bindiego/core_bench.json",
        help="Path to THP-off core_bench JSON",
    )
    parser.add_argument(
        "--core-on",
        default="target/bench/core_bench_thp_on.json",
        help="Path to THP-on core_bench JSON",
    )
    parser.add_argument(
        "--e2e-off",
        default="benches/baselines/bindiego/e2e_search.json",
        help="Path to THP-off e2e_search JSON",
    )
    parser.add_argument(
        "--e2e-on",
        default="target/bench/e2e_search_thp_on.json",
        help="Path to THP-on e2e_search JSON",
    )
    args = parser.parse_args()

    core_off_p = Path(args.core_off)
    core_on_p = Path(args.core_on)
    if core_off_p.exists() and core_on_p.exists():
        off_map, _ = load_records(core_off_p)
        on_map, _ = load_records(core_on_p)
        print("### `core_bench` THP-Off (`PR_SET_THP_DISABLE`) vs THP-On (`--thp-on`)\n")
        print("| Benchmark | Dataset Size | THP-Off Min | THP-On Min | THP-On Delta (%) | THP-Off Median | THP-On Median |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for name, off in off_map.items():
            on = on_map.get(name)
            if not on:
                continue
            off_min = off["min_ns_per_op"]
            on_min = on["min_ns_per_op"]
            delta_pct = ((on_min - off_min) / off_min * 100.0) if off_min > 0 else 0.0
            print(
                f"| `{name}` | `{off['dataset_size']}` | `{off_min:.4f}` | `{on_min:.4f}` | `{delta_pct:+.2f}%` | `{off['median_ns_per_op']:.4f}` | `{on['median_ns_per_op']:.4f}` |"
            )
        print()

    e2e_off_p = Path(args.e2e_off)
    e2e_on_p = Path(args.e2e_on)
    if e2e_off_p.exists() and e2e_on_p.exists():
        off_map, off_full = load_records(e2e_off_p)
        on_map, on_full = load_records(e2e_on_p)
        print("### `e2e_search_bench` THP-Off (`PR_SET_THP_DISABLE`) vs THP-On (`--thp-on`)\n")
        print("| Benchmark / Engine | THP-Off Min (`ns/q`) | THP-On Min (`ns/q`) | THP-On Delta (%) | THP-Off QPS | THP-On QPS |")
        print("|---|---:|---:|---:|---:|---:|")
        for name, off in off_map.items():
            on = on_map.get(name)
            if not on:
                continue
            off_min = off["min_ns_per_op"]
            on_min = on["min_ns_per_op"]
            delta_pct = ((on_min - off_min) / off_min * 100.0) if off_min > 0 else 0.0
            off_qps = 1e9 / off_min if off_min > 0 else 0.0
            on_qps = 1e9 / on_min if on_min > 0 else 0.0
            print(
                f"| `{name}` | `{off_min:.2f}` | `{on_min:.2f}` | `{delta_pct:+.2f}%` | `{off_qps:,.1f}` | `{on_qps:,.1f}` |"
            )
        print()
        off_lat = off_full.get("latency_summary", {})
        on_lat = on_full.get("latency_summary", {})
        if off_lat and on_lat:
            print("#### Latency Distribution Summary (THP-Off vs THP-On)\n")
            print("| Engine | Setting | QPS | p50 (`us`) | p95 (`us`) | p99 (`us`) |")
            print("|---|---|---:|---:|---:|---:|")
            for eng in ["zipora_bmw", "tantivy", "uncompressed_baseline"]:
                if eng in off_lat:
                    d = off_lat[eng]
                    print(f"| `{eng}` | `THP-off` (default) | `{d['qps']:,.1f}` | `{d['p50_us']:.2f}` | `{d['p95_us']:.2f}` | `{d['p99_us']:.2f}` |")
                if eng in on_lat:
                    d = on_lat[eng]
                    print(f"| `{eng}` | `THP-on` (`--thp-on`) | `{d['qps']:,.1f}` | `{d['p50_us']:.2f}` | `{d['p95_us']:.2f}` | `{d['p99_us']:.2f}` |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
