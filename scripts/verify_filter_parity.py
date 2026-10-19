#!/usr/bin/env python3
"""Verify isolated (`--filter <bench>`) vs full-suite timing parity (S44-R1).

Runs benchmarks individually via `<binary> --filter <name>` using the exact
compiled binary recorded in the full-suite JSON report (`host_metadata.binary_path`
verified against `host_metadata.binary_sha256`), and checks that each benchmark's
isolated `--filter` `min_ns_per_op` is within `±5.0%` of its full-suite `min_ns_per_op`.

Usage:
  # Verify the 84 heavy / allocator-sensitive benchmarks in competitor_bench:
  python3 scripts/verify_filter_parity.py

  # Verify all 181 benchmarks in competitor_bench or all 80 in core_bench:
  python3 scripts/verify_filter_parity.py --suite competitor_bench --all
  python3 scripts/verify_filter_parity.py --suite core_bench --all

  # Verify specific benchmarks by name/substring:
  python3 scripts/verify_filter_parity.py --only int_codecs/stream_vbyte_simd/encode_deltas,tries/dat/shared_prefix/build_sorted,radix_sort/voracious/u32,suffix_array/construct
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_HEAVY_CATEGORIES = (
    "tries",
    "hash_maps",
    "int_codecs",
    "radix_sort",
    "suffix_array",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def resolve_verified_binary(full_data: dict[str, Any], override_bin: str) -> str:
    if override_bin:
        p = Path(override_bin)
        if not (p.is_file() and os.access(p, os.X_OK)):
            raise RuntimeError(f"Override binary is not executable: {override_bin}")
        return str(p.resolve())
    meta = full_data.get("host_metadata", {})
    bin_path = meta.get("binary_path", "")
    expected_sha = meta.get("binary_sha256", "")
    if not bin_path or not expected_sha:
        raise RuntimeError(
            "Full-suite JSON is missing host_metadata.binary_path or host_metadata.binary_sha256"
        )
    p = Path(bin_path)
    if not (p.is_file() and os.access(p, os.X_OK)):
        raise RuntimeError(f"Recorded binary_path does not exist or is not executable: {bin_path}")
    actual_sha = sha256_file(p)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"Binary SHA-256 mismatch for {bin_path}: expected {expected_sha}, got {actual_sha}"
        )
    return str(p.resolve())


def run_filtered_benchmark(binary_path: str, bench_name: str, tmp_out: Path) -> dict[str, Any]:
    if tmp_out.exists():
        tmp_out.unlink()
    cmd = [binary_path, "--filter", bench_name, "--output", str(tmp_out)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not tmp_out.exists():
        raise RuntimeError(
            f"Filtered run failed for {bench_name} (rc={proc.returncode}):\n{proc.stderr}"
        )
    data = json.loads(tmp_out.read_text())
    for rec in data.get("benchmarks", []):
        if rec.get("name") == bench_name:
            return rec
    raise RuntimeError(f"Benchmark {bench_name} not found in filtered output {tmp_out}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify isolated (--filter) vs full-suite benchmark parity (S44-R1)."
    )
    parser.add_argument(
        "--suite",
        default="competitor_bench",
        choices=["competitor_bench", "core_bench"],
        help="Benchmark suite to verify (default: competitor_bench)",
    )
    parser.add_argument(
        "--full-json",
        default="",
        help="Path to full-suite JSON report (defaults to target/bench/<suite>.json if present, else benches/baselines/bindiego/<suite>.json)",
    )
    parser.add_argument(
        "--binary",
        default="",
        help="Optional override path to compiled benchmark binary",
    )
    parser.add_argument(
        "--categories",
        default=",".join(DEFAULT_HEAVY_CATEGORIES),
        help=f"Comma-separated categories to check unless --all or --only is set (default: {','.join(DEFAULT_HEAVY_CATEGORIES)})",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Verify all benchmarks in the suite across all 10 categories",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated exact benchmark names or substrings to verify",
    )
    parser.add_argument(
        "--max-delta-pct",
        type=float,
        default=5.0,
        help="Maximum allowed |filter_min - full_min| / full_min percentage (default: 5.0%%)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Confirmation retries if a single isolated run exceeds ±max-delta-pct due to transient host/page noise, matching bench_compare.py --confirm-retries 5 (default: 5)",
    )
    parser.add_argument(
        "--out-json",
        default="",
        help="Optional path to write structured JSON parity results",
    )
    args = parser.parse_args()

    if args.full_json:
        full_json_path = Path(args.full_json)
    else:
        target_candidate = Path(f"target/bench/{args.suite}.json")
        baseline_candidate = Path(f"benches/baselines/bindiego/{args.suite}.json")
        full_json_path = (
            target_candidate if target_candidate.exists() else baseline_candidate
        )

    if not full_json_path.exists():
        print(f"ERROR: Full-suite JSON not found at {full_json_path}", file=sys.stderr)
        return 2

    full_data = json.loads(full_json_path.read_text())
    binary_path = resolve_verified_binary(full_data, args.binary)
    all_benches: list[dict[str, Any]] = full_data.get("benchmarks", [])

    if args.only:
        tokens = [t.strip() for t in args.only.split(",") if t.strip()]
        selected = [
            b
            for b in all_benches
            if any(b["name"] == tok or tok in b["name"] for tok in tokens)
        ]
    elif args.all:
        selected = list(all_benches)
    else:
        cats = {c.strip() for c in args.categories.split(",") if c.strip()}
        selected = [b for b in all_benches if b.get("category") in cats]

    if not selected:
        print("ERROR: No benchmarks matched selection criteria.", file=sys.stderr)
        return 2

    tmp_out = Path(f"target/bench/verify_filter_parity_tmp_{os.getpid()}.json")
    tmp_out.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"=== S44-R1 Filter vs Full-Suite Parity Check (suite={args.suite}, "
        f"benchmarks={len(selected)}/{len(all_benches)}, max_delta=±{args.max_delta_pct:.1f}%) ==="
    )
    print(f"  Full-suite JSON: {full_json_path}")
    print(f"  Verified binary: {binary_path}")
    print(
        f"  {'#':>3}  {'Benchmark':<50} {'Full Min':>10} {'Filter Min':>10} {'Unit':<8} {'Delta':>8}  {'Status'}"
    )
    print("  " + "-" * 102)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    try:
        for idx, b in enumerate(selected, start=1):
            name = b["name"]
            unit = b["unit"]
            full_min = float(b["min_ns_per_op"])
            best_rec: dict[str, Any] | None = None
            best_abs_delta = float("inf")
            best_delta_pct = 0.0

            for attempt in range(max(1, args.retries + 1)):
                rec = run_filtered_benchmark(binary_path, name, tmp_out)
                filt_min = float(rec["min_ns_per_op"])
                delta_pct = (
                    (filt_min - full_min) / full_min * 100.0 if full_min > 0.0 else 0.0
                )
                if abs(delta_pct) < best_abs_delta:
                    best_abs_delta = abs(delta_pct)
                    best_delta_pct = delta_pct
                    best_rec = rec
                if abs(delta_pct) <= args.max_delta_pct:
                    break

            assert best_rec is not None
            filt_min = float(best_rec["min_ns_per_op"])
            passed = abs(best_delta_pct) <= args.max_delta_pct
            status = "PASS" if passed else "FAIL"
            print(
                f"  {idx:>3}  {name:<50} {full_min:>10.3f} {filt_min:>10.3f} {unit:<8} "
                f"{best_delta_pct:>+7.2f}%  {status}"
            )
            row = {
                "name": name,
                "category": b.get("category", ""),
                "unit": unit,
                "full_min_ns_per_op": full_min,
                "filter_min_ns_per_op": filt_min,
                "delta_pct": round(best_delta_pct, 3),
                "passed": passed,
            }
            results.append(row)
            if not passed:
                failures.append(row)
    finally:
        if tmp_out.exists():
            tmp_out.unlink()

    within_cnt = len(results) - len(failures)
    max_abs = max((abs(r["delta_pct"]) for r in results), default=0.0)
    print("  " + "-" * 102)
    print(
        f"=== Summary: {within_cnt}/{len(results)} within ±{args.max_delta_pct:.1f}% "
        f"(max |delta| = {max_abs:.2f}%) ==="
    )

    if args.out_json:
        out_p = Path(args.out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(
            json.dumps(
                {
                    "suite": args.suite,
                    "full_json": str(full_json_path),
                    "binary_path": binary_path,
                    "max_delta_pct": args.max_delta_pct,
                    "total_checked": len(results),
                    "passed_count": within_cnt,
                    "failed_count": len(failures),
                    "max_abs_delta_pct": round(max_abs, 3),
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
