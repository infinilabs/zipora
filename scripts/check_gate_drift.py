#!/usr/bin/env python3
"""Check test counts against gate_baseline.txt and prevent test-count drift (Rule B8)."""

import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_FILE = os.path.join(SCRIPT_DIR, 'gate_baseline.txt')

def parse_passed_counts(log_path):
    if not os.path.exists(log_path):
        return []
    counts = []
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = re.search(r'test result:\s+ok\.\s+(\d+)\s+passed', line)
            if m:
                counts.append(int(m.group(1)))
    return counts

def load_baseline():
    baseline = {}
    if not os.path.exists(BASELINE_FILE):
        print(f"Error: baseline file {BASELINE_FILE} not found", file=sys.stderr)
        sys.exit(1)
    with open(BASELINE_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                baseline[k.strip()] = int(v.strip())
    return baseline

def main():
    target_dir = sys.argv[1] if len(sys.argv) > 1 else 'target/sanity'
    baseline = load_baseline()

    debug_log = os.path.join(target_dir, 'debug_tests.log')
    release_log = os.path.join(target_dir, 'release_lib.log')
    doc_log = os.path.join(target_dir, 'doctests.log')

    debug_counts = parse_passed_counts(debug_log)
    release_counts = parse_passed_counts(release_log)
    doc_counts = parse_passed_counts(doc_log)

    lib_debug = debug_counts[0] if len(debug_counts) > 0 else 0
    lib_release = release_counts[0] if len(release_counts) > 0 else 0
    doctests = doc_counts[0] if len(doc_counts) > 0 else 0
    all_targets = sum(debug_counts) + sum(doc_counts)

    current = {
        'lib_debug': lib_debug,
        'lib_release': lib_release,
        'doctests': doctests,
        'all_targets': all_targets,
    }

    print("=== Test Count Drift Guard (Rule B8) ===")
    print(f"{'Metric':<15} {'Measured':<10} {'Baseline':<10} {'Status':<10}")
    print("-" * 48)

    failed = False
    for key in ['lib_debug', 'lib_release', 'doctests', 'all_targets']:
        measured = current[key]
        expected = baseline.get(key, 0)
        if measured < expected:
            status = f"DRIFT (-{expected - measured})"
            failed = True
        elif measured > expected:
            status = f"INCREASE (+{measured - expected})"
        else:
            status = "OK"
        print(f"{key:<15} {measured:<10} {expected:<10} {status:<10}")

    print("-" * 48)

    if failed:
        print("\nFAILURE: Test count has drifted below baseline! See scripts/gate_baseline.txt.", file=sys.stderr)
        sys.exit(1)

    print("Test count drift check: PASS")

if __name__ == '__main__':
    main()
