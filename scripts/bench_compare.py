#!/usr/bin/env python3
"""
Phase B Rule B3: Performance Regression Gate (`scripts/bench_compare.py`).

Runner documentation (S37-R2 / S38-R4 / S39-R4):
- Reference host: `bindiego` (AMD EPYC 7B13, 64 logical cores across 4 Zen 3 CCDs)
- Pinned CPU core: self-pinned (`pid = 0` only) via `select_primary_core` (default core 23
  on interior CCD 2; override via `ZIPORA_BENCH_PIN_CORE`)
- Compiler flags:
    RUSTFLAGS="-C target-cpu=haswell -C target-feature=+bmi2,+avx2,+popcnt -C codegen-units=1 -C llvm-args=-align-loops=32"
- Exact command:
    make bench_core
    make bench_e2e_search
- Shared-host resilience (S40-R1 Option B / S45-R2):
    When `--confirm-retries K` (`K > 0`, wired as `--confirm-retries 5` in `Makefile`)
    is enabled and one or more Zipora benchmarks exceed their regression threshold on the
    initial pass (with zero structural/integrity errors), `bench_compare.py` waits
    for a quiet host window and re-runs only the failing benchmark(s) via `--filter`
    up to `K` times, failing only if a regression persists across every retry.
    Per-benchmark thresholds remain unchanged (`5.0%` default, `9.0%` DRAM/allocator-bound).
    In `competitor_bench` (`S45-R2`, `S46-R2`), the 101 third-party competitor-only rows
    (95 timing-threshold rows + 6 `enforce_threshold: false` `build_1m` rows) are strictly
    checked for presence and `dataset_size` integrity, while timing-threshold exceedances on
    the 95 third-party rows are reported informationally (`INFO(comp)`) so third-party DRAM
    contention under heavy multi-job host load cannot cause false gate failures (`86` timing-gated
    Zipora rows = `80` `core_bench` rows + `6` Zipora `ef_select`/`cef_select` rows).
- Baselines under `benches/baselines/<host>/` are never auto-written by the gate;
  set `UPDATE_BASELINE=1` (which forwards `-- --save-baseline`) explicitly when recording or updating a baseline.

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
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def is_hex_str(s: str, min_len: int, max_len: int) -> bool:
    return (
        min_len <= len(s) <= max_len
        and all(ch in "0123456789abcdef" for ch in s)
    )


def git_commit_exists(commit: str) -> bool:
    if not is_hex_str(commit, 7, 40):
        return False
    proc = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0


def compute_cargo_lock_sha256(commit: Optional[str] = None) -> Optional[str]:
    if commit is None:
        try:
            return hashlib.sha256(Path("Cargo.lock").read_bytes()).hexdigest().lower()
        except OSError:
            return None
    proc = subprocess.run(
        ["git", "show", f"{commit}:Cargo.lock"],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return hashlib.sha256(proc.stdout).hexdigest().lower()


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


def read_nr_running() -> int:
    try:
        text = Path("/proc/loadavg").read_text(encoding="utf-8").strip()
        parts = text.split()
        if len(parts) >= 4 and "/" in parts[3]:
            return int(parts[3].split("/")[0])
    except (OSError, ValueError):
        pass
    return 1


def read_cpu_stat_totals() -> Optional[Tuple[int, int]]:
    try:
        with Path("/proc/stat").open("r", encoding="utf-8") as f:
            first = f.readline()
        if not first.startswith("cpu "):
            return None
        fields = [int(x) for x in first.split()[1:]]
        if len(fields) < 4:
            return None
        idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
        total = sum(fields)
        return total, idle
    except (OSError, ValueError):
        return None


def sample_busy_cores(window_sec: float = 0.20) -> Tuple[int, float]:
    nr = read_nr_running()
    s0 = read_cpu_stat_totals()
    if s0 is None:
        return nr, float(max(0, nr - 1))
    time.sleep(window_sec)
    s1 = read_cpu_stat_totals()
    nr = read_nr_running()
    if s1 is None:
        return nr, float(max(0, nr - 1))
    dt = max(1, s1[0] - s0[0])
    didle = max(0, s1[1] - s0[1])
    busy_frac = max(0.0, float(dt - didle) / float(dt))
    ncpu = os.cpu_count() or 64
    return nr, busy_frac * ncpu


def wait_for_quiet_host_window(max_wait_sec: float = 35.0) -> Tuple[int, float]:
    nr = read_nr_running()
    if nr <= 5:
        nr2, busy = sample_busy_cores(0.10)
        if nr2 <= 6 and busy <= 6.5:
            return nr2, busy
    deadline = time.monotonic() + max_wait_sec
    best_nr, best_busy = nr, 999.0
    while time.monotonic() < deadline:
        nr, busy = sample_busy_cores(0.20)
        if busy < best_busy:
            best_nr, best_busy = nr, busy
        if nr <= 6 and busy <= 6.5:
            return nr, busy
    return best_nr, best_busy


def resolve_retry_binary(host_metadata: Dict[str, Any]) -> Path:
    recorded_path = str(host_metadata.get("binary_path", "") or "").strip()
    recorded_sha256 = str(host_metadata.get("binary_sha256", "") or "").strip().lower()
    if not recorded_path:
        print(
            "ERROR: missing or empty host_metadata.binary_path in current benchmark JSON; "
            "refusing to run an unverified confirmation retry.",
            file=sys.stderr,
        )
        sys.exit(1)
    if (
        len(recorded_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in recorded_sha256)
    ):
        print(
            "ERROR: missing or invalid host_metadata.binary_sha256 in current benchmark JSON; "
            "refusing to run an unverified confirmation retry.",
            file=sys.stderr,
        )
        sys.exit(1)

    bin_path = Path(recorded_path)
    if not bin_path.is_file() or not os.access(bin_path, os.X_OK):
        print(
            f"ERROR: recorded benchmark binary in host_metadata.binary_path does not exist or is not executable: {bin_path}",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        actual_sha256 = hashlib.sha256(bin_path.read_bytes()).hexdigest().lower()
    except OSError as exc:
        print(
            f"ERROR: failed to read recorded benchmark binary {bin_path}: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
    if actual_sha256 != recorded_sha256:
        print(
            f"ERROR: recorded benchmark binary {bin_path} SHA-256 mismatch "
            f"(expected {recorded_sha256}, found {actual_sha256}); refusing to retry with a different binary.",
            file=sys.stderr,
        )
        sys.exit(1)
    return bin_path


def run_isolated_retry(
    suite: str,
    bench_names: List[str],
    attempt: int,
    total_attempts: int,
    host_metadata: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    target_root = Path(os.environ.get("CARGO_TARGET_DIR", "target"))
    retry_out = target_root / "bench" / f"{suite}_retry_{attempt}.json"
    if retry_out.exists():
        try:
            retry_out.unlink()
        except OSError:
            pass

    compiled_bin = resolve_retry_binary(host_metadata)
    nr, busy = wait_for_quiet_host_window(max_wait_sec=35.0)
    filter_arg = ",".join(bench_names)
    print(
        f"  [confirm-retry {attempt}/{total_attempts}] host nr_running={nr}, busy_cores={busy:.1f}; "
        f"re-running: {filter_arg}",
        flush=True,
    )

    rustflags = host_metadata.get("rustflags")
    env = os.environ.copy()
    if rustflags:
        env["RUSTFLAGS"] = rustflags
    if host_metadata.get("git_commit"):
        env["ZIPORA_BENCH_GIT_COMMIT"] = str(host_metadata["git_commit"])

    cmd = [str(compiled_bin), "--output", str(retry_out)]
    if suite in ("core_bench", "competitor_bench") and bench_names:
        cmd.extend(["--filter", filter_arg])

    proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode != 0 or not retry_out.exists():
        return None
    try:
        with retry_out.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def compare_benchmarks(
    baseline_data: Dict[str, Any],
    current_data: Dict[str, Any],
    default_max_regression_pct: float,
    override_max_regression_pct: Optional[float],
    abs_noise_floor_ns: float,
    metric: str,
    quiet: bool = False,
) -> Tuple[int, List[str], List[str], bool, bool]:
    base_list = baseline_data.get("benchmarks", [])
    curr_list = current_data.get("benchmarks", [])
    base_benches = {b["name"]: b for b in base_list}
    curr_benches = {b["name"]: b for b in curr_list}
    competitor_only_names = {
        row["competitor_bench"]
        for row in baseline_data.get("same_run_comparisons", [])
        if "competitor_bench" in row
    } - {
        row["zipora_bench"]
        for row in baseline_data.get("same_run_comparisons", [])
        if "zipora_bench" in row
    }

    metric_field = "min_ns_per_op" if metric == "min" else "median_ns_per_op"

    failures: List[str] = []
    regressed_names: List[str] = []
    has_timing_regression = False
    has_structural_error = False
    prov_ok = True
    rows: List[Tuple[str, str, float, float, float, float, float, float, float, str]] = []

    # 0. Header-only provenance verification via commit + rustflags + Cargo.lock SHA-256
    # in both clean and dirty trees without disabling timing confirmation retries (S52-R1 / S54-R1).
    suite_name = str(
        baseline_data.get("suite") or current_data.get("suite") or "core_bench"
    ).strip()
    suite_bin = "e2e_search_bench" if suite_name == "e2e_search" else suite_name

    for label, doc in [("baseline", baseline_data), ("current", current_data)]:
        hm = doc.get("host_metadata", {}) or {}
        if len(hm) > 14 or any(k.startswith("row_") for k in hm):
            failures.append(
                f"{label}.host_metadata: unexpected splice summary field present (S52-R1)"
            )
            prov_ok = False
        commit_val = str(hm.get("git_commit", "") or "").strip().lower()
        commit_check = (
            commit_val[:-6]
            if (label == "current" and commit_val.endswith("-dirty"))
            else commit_val
        )
        lock_sha_val = str(hm.get("cargo_lock_sha256", "") or "").strip().lower()
        rustflags_val = str(hm.get("rustflags", "") or "").strip()
        sha_val = str(hm.get("binary_sha256", "") or "").strip().lower()
        bin_path_val = str(hm.get("binary_path", "") or "").strip()
        if not git_commit_exists(commit_check):
            failures.append(
                f"{label}.host_metadata: invalid or unknown git_commit '{commit_val}' (S52-R1/S54-R1)"
            )
            prov_ok = False
        if not is_hex_str(lock_sha_val, 64, 64):
            failures.append(
                f"{label}.host_metadata: invalid or missing cargo_lock_sha256 '{lock_sha_val}' (S54-R1)"
            )
            prov_ok = False
        if not rustflags_val:
            failures.append(
                f"{label}.host_metadata: missing or empty rustflags (S54-R1)"
            )
            prov_ok = False
        if not is_hex_str(sha_val, 64, 64):
            failures.append(
                f"{label}.host_metadata: invalid binary_sha256 '{sha_val}' (S52-R1)"
            )
            prov_ok = False
        bin_p = Path(bin_path_val) if bin_path_val else Path()
        if (
            not bin_path_val
            or bin_p.parent.name != "deps"
            or not bin_p.name.startswith(f"{suite_bin}-")
        ):
            failures.append(
                f"{label}.host_metadata: binary_path '{bin_path_val}' is not a valid Cargo deps/{suite_bin}-* binary (S52-R1)"
            )
            prov_ok = False

        for row in doc.get("benchmarks", []):
            if len(row) > 12:
                failures.append(
                    f"{label}.{row.get('name', '<unknown>')}: unexpected per-row provenance fields present (S52-R1)"
                )
                prov_ok = False
        for cmp_row in doc.get("same_run_comparisons", []):
            if len(cmp_row) > 16:
                failures.append(
                    f"{label}.same_run_comparisons: unexpected per-comparison provenance fields present (S52-R1)"
                )
                prov_ok = False

    base_hm = baseline_data.get("host_metadata", {}) or {}
    curr_hm = current_data.get("host_metadata", {}) or {}
    base_commit = str(base_hm.get("git_commit", "") or "").strip().lower()
    base_lock_sha = str(base_hm.get("cargo_lock_sha256", "") or "").strip().lower()
    curr_lock_sha = str(curr_hm.get("cargo_lock_sha256", "") or "").strip().lower()
    base_rustflags = str(base_hm.get("rustflags", "") or "").strip()
    curr_rustflags = str(curr_hm.get("rustflags", "") or "").strip()
    base_sha = str(base_hm.get("binary_sha256", "") or "").strip().lower()
    base_bin_path = str(base_hm.get("binary_path", "") or "").strip()
    curr_sha = str(curr_hm.get("binary_sha256", "") or "").strip().lower()
    curr_bin_path = str(curr_hm.get("binary_path", "") or "").strip()

    if prov_ok:
        if base_rustflags != curr_rustflags:
            failures.append(
                f"provenance: current rustflags ('{curr_rustflags}') do not match baseline rustflags ('{base_rustflags}') (S54-R1)"
            )

        disk_lock_sha = compute_cargo_lock_sha256(None)
        git_lock_sha = compute_cargo_lock_sha256(base_commit)
        if base_lock_sha != curr_lock_sha:
            failures.append(
                f"provenance: current cargo_lock_sha256 ({curr_lock_sha[:16]}...) "
                f"does not match baseline cargo_lock_sha256 ({base_lock_sha[:16]}...) (S54-R1)"
            )
        if disk_lock_sha is None or base_lock_sha != disk_lock_sha:
            failures.append(
                f"provenance: cargo_lock_sha256 ({base_lock_sha[:16]}...) "
                f"does not match workspace Cargo.lock ({str(disk_lock_sha)[:16]}...) (S54-R1)"
            )
        if git_lock_sha is None or base_lock_sha != git_lock_sha:
            failures.append(
                f"provenance: cargo_lock_sha256 ({base_lock_sha[:16]}...) "
                f"does not match {base_commit}:Cargo.lock ({str(git_lock_sha)[:16]}...) (S54-R1)"
            )

        host_name = str(base_hm.get("host") or detect_short_host()).strip()
        if suite_name != "core_bench":
            core_base_path = (
                Path("benches") / "baselines" / host_name / "core_bench.json"
            )
            if core_base_path.is_file():
                try:
                    core_base_doc = json.loads(
                        core_base_path.read_text(encoding="utf-8")
                    )
                    core_base_hm = core_base_doc.get("host_metadata", {}) or {}
                    core_base_commit = (
                        str(core_base_hm.get("git_commit", "") or "").strip().lower()
                    )
                    core_rustflags = (
                        str(core_base_hm.get("rustflags", "") or "").strip()
                    )
                    core_lock_sha = (
                        str(core_base_hm.get("cargo_lock_sha256", "") or "")
                        .strip()
                        .lower()
                    )
                    if base_commit != core_base_commit:
                        failures.append(
                            f"{suite_name} provenance: baseline git_commit ({base_commit}) "
                            f"does not match core_bench git_commit ({core_base_commit}) (S54-R1)"
                        )
                    if core_rustflags and base_rustflags != core_rustflags:
                        failures.append(
                            f"{suite_name} provenance: baseline rustflags "
                            f"do not match core_bench rustflags (S54-R1)"
                        )
                    if core_lock_sha and base_lock_sha != core_lock_sha:
                        failures.append(
                            f"{suite_name} provenance: baseline cargo_lock_sha256 "
                            f"does not match core_bench cargo_lock_sha256 (S54-R1)"
                        )
                except (OSError, json.JSONDecodeError):
                    pass

        # Verify baseline binary_sha256 matches either the committed HEAD baseline or the on-disk binary at base_bin_path.
        base_sha_verified = False
        head_base_proc = subprocess.run(
            [
                "git",
                "show",
                f"HEAD:benches/baselines/{host_name}/{suite_name}.json",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if head_base_proc.returncode == 0:
            try:
                head_base_doc = json.loads(head_base_proc.stdout)
                head_hm = head_base_doc.get("host_metadata", {}) or {}
                if str(head_hm.get("binary_sha256", "")).strip().lower() == base_sha:
                    base_sha_verified = True
            except json.JSONDecodeError:
                pass
        if not base_sha_verified:
            base_bin = Path(base_bin_path)
            if base_bin.is_file():
                try:
                    actual_base_sha = (
                        hashlib.sha256(base_bin.read_bytes()).hexdigest().lower()
                    )
                    if actual_base_sha == base_sha:
                        base_sha_verified = True
                except OSError:
                    pass
        if not base_sha_verified:
            failures.append(
                f"baseline.host_metadata: binary_sha256 ({base_sha[:16]}...) "
                f"does not match committed HEAD baseline or on-disk binary (S52-R1/S54-R1)"
            )

        curr_bin = Path(curr_bin_path)
        if curr_bin.is_file():
            try:
                actual_curr_sha = (
                    hashlib.sha256(curr_bin.read_bytes()).hexdigest().lower()
                )
                if actual_curr_sha != curr_sha:
                    failures.append(
                        f"current.host_metadata: on-disk binary {curr_bin} SHA-256 ({actual_curr_sha[:16]}...) "
                        f"does not match recorded binary_sha256 ({curr_sha[:16]}...) (S52-R1/S54-R1)"
                    )
            except OSError:
                pass

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
            has_structural_error = True
            continue

        curr = curr_benches[name]
        base_ds = base.get("dataset_size")
        curr_ds = curr.get("dataset_size")
        if base_ds is not None and curr_ds is not None and int(base_ds) != int(curr_ds):
            failures.append(
                f"{name}: dataset_size mismatch (baseline={base_ds}, current={curr_ds})"
            )
            has_structural_error = True

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
            if name in competitor_only_names:
                status = "INFO(comp)"
            else:
                status = "REGRESSED"
                regressed_names.append(name)
                has_timing_regression = True
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

    if not quiet:
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
        if not quiet:
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
                if not quiet:
                    print(
                        f"  {metric_name:<30} baseline={b_val:>10.3f}  current={c_val:>10.3f}  ({pct:+.2f}%, thresh={eff_thresh:.2f}%) [{status}]"
                    )
                if regressed:
                    has_timing_regression = True
                    failures.append(
                        f"summary.{metric_name}: baseline={b_val:.3f}, current={c_val:.3f} ({pct:+.2f}%)"
                    )

    return (
        len(rows),
        failures,
        regressed_names,
        has_timing_regression,
        has_structural_error,
    )


def merge_retry_into_current(
    current_data: Dict[str, Any],
    retry_data: Dict[str, Any],
    metric: str,
) -> bool:
    metric_field = "min_ns_per_op" if metric == "min" else "median_ns_per_op"
    curr_list = current_data.get("benchmarks", [])
    curr_by_name = {b["name"]: i for i, b in enumerate(curr_list)}
    updated = False

    for rb in retry_data.get("benchmarks", []):
        name = rb.get("name")
        if name not in curr_by_name:
            continue
        idx = curr_by_name[name]
        old_b = curr_list[idx]
        old_val = float(old_b.get(metric_field, old_b["median_ns_per_op"]))
        new_val = float(rb.get(metric_field, rb["median_ns_per_op"]))
        if new_val < old_val:
            curr_list[idx] = rb
            updated = True

    if "summary" in current_data and "summary" in retry_data:
        curr_sum = current_data["summary"]
        retry_sum = retry_data["summary"]
        for metric_name, higher_is_better in [
            ("zipora_qps_top10", True),
            ("zipora_p50_us", False),
            ("zipora_p99_us", False),
            ("zipora_bytes_per_posting", False),
            ("tantivy_qps_top10", True),
        ]:
            if metric_name in curr_sum and metric_name in retry_sum:
                old_v = float(curr_sum[metric_name])
                new_v = float(retry_sum[metric_name])
                if (higher_is_better and new_v > old_v) or (
                    not higher_is_better and new_v < old_v
                ):
                    curr_sum[metric_name] = retry_sum[metric_name]
                    updated = True

    return updated


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
    parser.add_argument(
        "--confirm-retries",
        type=int,
        default=0,
        help="Re-run failing benchmarks up to N times to filter transient shared-host contention (default: 0; Makefile sets 5)",
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

    max_retries = max(0, args.confirm_retries)
    (
        total,
        failures,
        regressed_names,
        has_timing_regression,
        has_structural_error,
    ) = compare_benchmarks(
        baseline_data,
        current_data,
        default_thresh,
        args.max_regression_pct,
        args.abs_noise_floor_ns,
        args.metric,
        quiet=(max_retries > 0),
    )

    retries_used = 0
    if max_retries > 0 and has_timing_regression and not has_structural_error:
        host_metadata = current_data.get("host_metadata", {}) or {}
        any_merged = False
        for attempt in range(1, max_retries + 1):
            retries_used = attempt
            print(
                f"=== Shared-Host Confirm-on-Retry ({attempt}/{max_retries}): "
                f"{len(regressed_names) or 1} timing check(s) exceeded threshold ===",
                flush=True,
            )
            for msg in failures:
                print(f"  * {msg}", flush=True)

            if args.suite in ("core_bench", "competitor_bench") and regressed_names:
                # Re-run each failing benchmark in an isolated process so each
                # fits inside a short quiet host window with a clean heap.
                batch_size = 1 if len(regressed_names) <= 20 else 2
                for i in range(0, len(regressed_names), batch_size):
                    chunk = regressed_names[i : i + batch_size]
                    retry_data = run_isolated_retry(
                        args.suite, chunk, attempt, max_retries, host_metadata
                    )
                    if retry_data is not None:
                        if merge_retry_into_current(
                            current_data, retry_data, args.metric
                        ):
                            any_merged = True
            else:
                retry_data = run_isolated_retry(
                    args.suite, regressed_names, attempt, max_retries, host_metadata
                )
                if retry_data is not None:
                    if merge_retry_into_current(current_data, retry_data, args.metric):
                        any_merged = True

            (
                total,
                failures,
                regressed_names,
                has_timing_regression,
                has_structural_error,
            ) = compare_benchmarks(
                baseline_data,
                current_data,
                default_thresh,
                args.max_regression_pct,
                args.abs_noise_floor_ns,
                args.metric,
                quiet=True,
            )
            if not has_timing_regression:
                break

        if any_merged and args.current is None:
            try:
                with current_path.open("w", encoding="utf-8") as f:
                    json.dump(current_data, f, indent=2)
            except OSError:
                pass

    if max_retries > 0:
        total, failures, _, _, _ = compare_benchmarks(
            baseline_data,
            current_data,
            default_thresh,
            args.max_regression_pct,
            args.abs_noise_floor_ns,
            args.metric,
            quiet=False,
        )

    print("-" * 142)
    if failures:
        print(f"FAIL: {len(failures)} regression/integrity check(s) failed:")
        for msg in failures:
            print(f"  - {msg}")
        return 1

    if retries_used > 0:
        print(
            f"PASS: all {total} benchmark(s) present and within regression thresholds ({args.metric}) "
            f"after {retries_used} shared-host confirmation retry(s)."
        )
    else:
        print(
            f"PASS: all {total} benchmark(s) present and within regression thresholds ({args.metric})."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
