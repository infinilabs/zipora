#!/usr/bin/env python3
"""Audit production source code for bare .unwrap() and .expect() calls.

Rule D10 / S2-R1:
- Zero bare .unwrap() in non-test production code.
- Production error paths must use Result/Option error handling, never panic on untrusted input.
- Exact ceilings are stored in scripts/audit_baseline.toml and ratchet down.
"""

import argparse
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from audit_common import scan_rust_file, is_test_file

UNWRAP_PATTERN = re.compile(r'\.unwrap\s*\(')
EXPECT_PATTERN = re.compile(r'\.expect\s*\(')

def load_baseline():
    baseline_path = os.path.join(SCRIPT_DIR, 'audit_baseline.toml')
    max_unwraps = 0
    max_expects = 154
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('max_unwraps'):
                    max_unwraps = int(line.split('=')[1].strip())
                elif line.startswith('max_expects'):
                    max_expects = int(line.split('=')[1].strip())
    return max_unwraps, max_expects

def audit_file(filepath):
    unwraps = []
    expects = []
    for item in scan_rust_file(filepath):
        if item['in_test']:
            continue
        line = item['clean_line']
        if UNWRAP_PATTERN.search(line):
            unwraps.append({
                'file': filepath,
                'line': item['line_num'],
                'text': item['raw_line'].strip(),
            })
        if EXPECT_PATTERN.search(line):
            expects.append({
                'file': filepath,
                'line': item['line_num'],
                'text': item['raw_line'].strip(),
            })
    return unwraps, expects

def main():
    max_unwraps_default, max_expects_default = load_baseline()

    parser = argparse.ArgumentParser(description='Audit production unwraps and expects.')
    parser.add_argument('--src', default='src', help='Source directory to scan')
    parser.add_argument('--strict', action='store_true', help='Fail if any unwraps or expects are found')
    parser.add_argument('--max-unwraps', type=int, default=max_unwraps_default, help='Maximum allowed .unwrap() before failing')
    parser.add_argument('--max-expects', type=int, default=max_expects_default, help='Maximum allowed .expect() before failing')
    args = parser.parse_args()

    all_unwraps = []
    all_expects = []

    for root, dirs, files in os.walk(args.src):
        for f in sorted(files):
            if f.endswith('.rs') and not is_test_file(os.path.join(root, f)):
                filepath = os.path.join(root, f)
                unwraps, expects = audit_file(filepath)
                all_unwraps.extend(unwraps)
                all_expects.extend(expects)

    total_unwraps = len(all_unwraps)
    total_expects = len(all_expects)

    print("=== Production Unwrap & Panic Audit (Rule D10 / S2-R1) ===")
    print(f"Total .unwrap() found: {total_unwraps} (ratchet ceiling: {args.max_unwraps})")
    print(f"Total .expect() found: {total_expects} (ratchet ceiling: {args.max_expects})")

    if all_unwraps:
        print(f"\nUnwrap findings ({total_unwraps}):")
        for u in all_unwraps:
            print(f"  {u['file']}:{u['line']}: {u['text']}")

    failed = False
    if args.strict and (total_unwraps > 0 or total_expects > 0):
        print(f"\nFAILURE: --strict mode requires 0 unwraps/expects (found {total_unwraps} unwraps, {total_expects} expects)")
        failed = True
    elif total_unwraps > args.max_unwraps:
        print(f"\nFAILURE: unwraps ({total_unwraps}) exceeds ratchet ceiling ({args.max_unwraps})")
        failed = True
    elif total_expects > args.max_expects:
        print(f"\nFAILURE: expects ({total_expects}) exceeds ratchet ceiling ({args.max_expects})")
        failed = True

    if failed:
        sys.exit(1)

    print("\nAudit status: PASS")

if __name__ == '__main__':
    main()
