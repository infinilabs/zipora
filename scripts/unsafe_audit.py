#!/usr/bin/env python3
"""Audit unsafe code blocks and functions for SAFETY: comments and precondition hygiene.

Rule B7 / Agreement 4 & 5:
- Every unsafe block/fn in non-test code must be preceded by a SAFETY: comment.
- Every debug_assert! inside 5 lines of an unsafe block in a safe pub fn is a failure
  unless the line carries // PROVEN: with a pointer (§8.5).
"""

import argparse
import os
import re
import sys

UNSAFE_PATTERN = re.compile(r'\bunsafe\s*(?:\{|fn\b|impl\b|trait\b)')
SAFETY_PATTERN = re.compile(r'SAFETY:')
DEBUG_ASSERT_PATTERN = re.compile(r'\bdebug_assert!?\(')
PROVEN_PATTERN = re.compile(r'//\s*PROVEN:')
PUB_FN_PATTERN = re.compile(r'\bpub(?:\([^)]*\))?\s+fn\b')

def audit_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    unsafe_sites = []
    precondition_violations = []

    in_test_mod = False
    brace_depth = 0
    test_mod_depth = -1

    current_fn_is_safe_pub = False
    fn_depth = -1

    for idx, line in enumerate(lines):
        line_num = idx + 1
        stripped = line.strip()

        # Track test module scope
        if '#[cfg(test)]' in line:
            in_test_mod = True
            test_mod_depth = brace_depth

        # Track braces
        open_braces = line.count('{')
        close_braces = line.count('}')
        brace_depth += open_braces - close_braces

        if in_test_mod and brace_depth <= test_mod_depth:
            in_test_mod = False
            test_mod_depth = -1

        if in_test_mod:
            continue

        # Skip comment lines
        if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
            continue

        # Track function scope
        if PUB_FN_PATTERN.search(line) and 'unsafe' not in line:
            current_fn_is_safe_pub = True
            fn_depth = brace_depth
        elif current_fn_is_safe_pub and brace_depth < fn_depth:
            current_fn_is_safe_pub = False
            fn_depth = -1

        # Check for unsafe keyword
        if UNSAFE_PATTERN.search(line):
            # Look backwards up to 6 lines for SAFETY:
            has_safety = False
            for prev_idx in range(max(0, idx - 6), idx):
                if SAFETY_PATTERN.search(lines[prev_idx]):
                    has_safety = True
                    break

            unsafe_sites.append({
                'file': filepath,
                'line': line_num,
                'text': stripped,
                'documented': has_safety,
            })

            # Check if inside safe pub fn with debug_assert within 5 lines
            if current_fn_is_safe_pub:
                for check_idx in range(max(0, idx - 5), min(len(lines), idx + 6)):
                    check_line = lines[check_idx]
                    if DEBUG_ASSERT_PATTERN.search(check_line) and not PROVEN_PATTERN.search(check_line):
                        precondition_violations.append({
                            'file': filepath,
                            'line': check_idx + 1,
                            'unsafe_line': line_num,
                            'text': check_line.strip(),
                        })

    return unsafe_sites, precondition_violations

def main():
    parser = argparse.ArgumentParser(description='Audit unsafe code and SAFETY comments.')
    parser.add_argument('--src', default='src', help='Source directory to scan')
    parser.add_argument('--strict', action='store_true', help='Exit with non-zero if undocumented unsafe sites exist')
    parser.add_argument('--max-undocumented', type=int, default=240, help='Maximum allowed undocumented sites before failing')
    parser.add_argument('--max-violations', type=int, default=10, help='Maximum allowed precondition violations before failing')
    args = parser.parse_args()

    all_unsafe = []
    all_violations = []
    module_counts = {}

    for root, dirs, files in os.walk(args.src):
        for f in files:
            if f.endswith('.rs') and f != 'tests.rs':
                filepath = os.path.join(root, f)
                sites, violations = audit_file(filepath)
                all_unsafe.extend(sites)
                all_violations.extend(violations)

                mod = root.replace(args.src, '').strip(os.sep).split(os.sep)[0] or 'root'
                if mod not in module_counts:
                    module_counts[mod] = {'total': 0, 'documented': 0}
                for s in sites:
                    module_counts[mod]['total'] += 1
                    if s['documented']:
                        module_counts[mod]['documented'] += 1

    total = len(all_unsafe)
    documented = sum(1 for s in all_unsafe if s['documented'])
    undocumented = total - documented
    pct = (documented / total * 100) if total > 0 else 100.0

    print(f"=== Unsafe Code Audit ===")
    print(f"Total unsafe sites in non-test code: {total}")
    print(f"Documented (with SAFETY:):           {documented} ({pct:.1f}%)")
    print(f"Undocumented:                        {undocumented} (ceiling: {args.max_undocumented})")
    print(f"Precondition violations:             {len(all_violations)} (ceiling: {args.max_violations})")
    print()

    print("Module breakdown:")
    print(f"{'Module':<20} {'Total':<8} {'Documented':<12} {'Coverage':<10}")
    print("-" * 50)
    for mod in sorted(module_counts.keys(), key=lambda m: module_counts[m]['total'], reverse=True):
        t = module_counts[mod]['total']
        d = module_counts[mod]['documented']
        cov = (d / t * 100) if t > 0 else 100.0
        print(f"{mod:<20} {t:<8} {d:<12} {cov:.1f}%")
    print("-" * 50)

    failed = False
    if args.strict and all_violations:
        print("\nPrecondition Violations (§8.5 - debug_assert near unsafe without // PROVEN:):")
        for v in all_violations:
            print(f"  {v['file']}:{v['line']} (near unsafe at L{v['unsafe_line']}): {v['text']}")
        failed = True
    elif len(all_violations) > args.max_violations:
        print(f"\nFAILURE: precondition violations ({len(all_violations)}) exceeds ceiling ({args.max_violations})")
        failed = True

    if args.strict and undocumented > 0:
        print(f"\nFAILURE: --strict mode requires 0 undocumented unsafe sites (found {undocumented})")
        failed = True
    elif undocumented > args.max_undocumented:
        print(f"\nFAILURE: undocumented sites ({undocumented}) exceeds allowed ceiling ({args.max_undocumented})")
        failed = True

    if failed:
        sys.exit(1)

    print("\nAudit status: PASS")


if __name__ == '__main__':
    main()
