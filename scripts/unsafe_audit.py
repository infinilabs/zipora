#!/usr/bin/env python3
"""Audit unsafe code blocks and functions for SAFETY: comments and precondition hygiene.

Rule B7 / Agreement 4 & 5:
- Every unsafe block/impl/trait in non-test code must be preceded by a SAFETY: comment.
- Every unsafe fn in non-test code must be preceded by a '# Safety' doc section or SAFETY: comment.
- Every debug_assert! inside 5 lines of an unsafe block in a safe pub fn is a failure
  unless the line carries // PROVEN: with a pointer (§8.5).
- Ceilings ratchet down and are stored in scripts/audit_baseline.toml.
"""

import argparse
import os
import re
import sys

# Ensure local script directory is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from audit_common import scan_rust_file, is_test_file

UNSAFE_PATTERN = re.compile(r'\bunsafe\s*(?:\{|fn\b|impl\b|trait\b)')
SAFETY_PATTERN = re.compile(r'\bSAFETY:', re.IGNORECASE)
DOC_SAFETY_PATTERN = re.compile(r'#\s*Safety\b', re.IGNORECASE)
DEBUG_ASSERT_PATTERN = re.compile(r'\bdebug_assert!?\(')
PROVEN_PATTERN = re.compile(r'//\s*PROVEN:')
PUB_FN_PATTERN = re.compile(r'\bpub(?:\([^)]*\))?\s+fn\b')

def load_baseline():
    baseline_path = os.path.join(SCRIPT_DIR, 'audit_baseline.toml')
    max_undoc = 39
    max_viol = 10
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('max_undocumented'):
                    max_undoc = int(line.split('=')[1].strip())
                elif line.startswith('max_precondition_violations'):
                    max_viol = int(line.split('=')[1].strip())
    return max_undoc, max_viol

def audit_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        raw_lines = f.readlines()

    unsafe_sites = []
    precondition_violations = []

    file_scans = list(scan_rust_file(filepath))
    current_fn_is_safe_pub = False
    fn_brace_depth = -1
    brace_stack_len = 0

    for item in file_scans:
        idx = item['line_idx']
        line_num = item['line_num']
        clean = item['clean_line']
        in_test = item['in_test']

        if in_test:
            continue

        # Track pub fn scope
        if PUB_FN_PATTERN.search(clean) and 'unsafe' not in clean:
            current_fn_is_safe_pub = True
            fn_brace_depth = brace_stack_len

        for ch in clean:
            if ch == '{':
                brace_stack_len += 1
            elif ch == '}':
                if brace_stack_len > 0:
                    brace_stack_len -= 1
                if current_fn_is_safe_pub and brace_stack_len <= fn_brace_depth:
                    current_fn_is_safe_pub = False
                    fn_brace_depth = -1

        if clean.endswith(';') and not ('{' in clean) and fn_brace_depth == -1:
            current_fn_is_safe_pub = False

        if UNSAFE_PATTERN.search(clean):
            is_fn = bool(re.search(r'\bunsafe\s+fn\b', clean))
            is_impl = bool(re.search(r'\bunsafe\s+impl\b', clean))

            has_safety = False
            # Check up to 25 lines backwards for safety comments or doc comments
            for prev_idx in range(max(0, idx - 25), idx):
                prev_line = raw_lines[prev_idx].strip()
                if is_fn and DOC_SAFETY_PATTERN.search(prev_line):
                    has_safety = True
                    break
                if SAFETY_PATTERN.search(prev_line):
                    has_safety = True
                    break
                # Boundaries where comments stop applying
                if prev_line.startswith('fn ') or prev_line.startswith('pub fn ') or prev_line.startswith('struct ') or prev_line.startswith('enum '):
                    has_safety = False

            unsafe_sites.append({
                'file': filepath,
                'line': line_num,
                'text': clean,
                'is_fn': is_fn,
                'is_impl': is_impl,
                'documented': has_safety,
            })

            # Check §8.5 precondition: debug_assert within 5 lines of unsafe block in safe pub fn
            if current_fn_is_safe_pub and re.search(r'\bunsafe\s*\{', clean):
                for check_idx in range(max(0, idx - 5), min(len(raw_lines), idx + 6)):
                    chk_line = raw_lines[check_idx]
                    if DEBUG_ASSERT_PATTERN.search(chk_line) and not PROVEN_PATTERN.search(chk_line):
                        precondition_violations.append({
                            'file': filepath,
                            'line': check_idx + 1,
                            'unsafe_line': line_num,
                            'text': chk_line.strip(),
                        })

    return unsafe_sites, precondition_violations

def main():
    max_undoc_default, max_viol_default = load_baseline()

    parser = argparse.ArgumentParser(description='Audit unsafe code and SAFETY comments.')
    parser.add_argument('--src', default='src', help='Source directory to scan')
    parser.add_argument('--strict', action='store_true', help='Exit with non-zero if undocumented unsafe sites or violations exist')
    parser.add_argument('--max-undocumented', type=int, default=max_undoc_default, help='Maximum allowed undocumented sites before failing')
    parser.add_argument('--max-violations', type=int, default=max_viol_default, help='Maximum allowed precondition violations before failing')
    args = parser.parse_args()

    all_unsafe = []
    all_violations = []
    module_counts = {}

    for root, dirs, files in os.walk(args.src):
        for f in sorted(files):
            if f.endswith('.rs') and not is_test_file(os.path.join(root, f)):
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

    print("=== Unsafe Code Audit (Rule B7) ===")
    print(f"Total unsafe sites in non-test code: {total}")
    print(f"Documented (with SAFETY: or # Safety): {documented} ({pct:.1f}%)")
    print(f"Undocumented sites:                  {undocumented} (ratchet ceiling: {args.max_undocumented})")
    print(f"Precondition violations (§8.5):      {len(all_violations)} (ratchet ceiling: {args.max_violations})")
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

    # Always print violations and undocumented sites
    if all_violations:
        print("\nPrecondition Violations (§8.5 - debug_assert near unsafe without // PROVEN:):")
        for v in all_violations:
            print(f"  {v['file']}:{v['line']} (near unsafe at L{v['unsafe_line']}): {v['text']}")

    if undocumented > 0:
        print(f"\nUndocumented unsafe sites ({undocumented}):")
        for s in all_unsafe:
            if not s['documented']:
                kind = 'fn' if s['is_fn'] else ('impl' if s['is_impl'] else 'block')
                print(f"  {s['file']}:{s['line']} [{kind}]: {s['text']}")

    failed = False
    if args.strict:
        if len(all_violations) > 0:
            print(f"\nFAILURE: --strict mode requires 0 precondition violations (found {len(all_violations)})")
            failed = True
        if undocumented > 0:
            print(f"\nFAILURE: --strict mode requires 0 undocumented unsafe sites (found {undocumented})")
            failed = True
    else:
        if len(all_violations) > args.max_violations:
            print(f"\nFAILURE: precondition violations ({len(all_violations)}) exceeds ceiling ({args.max_violations})")
            failed = True
        if undocumented > args.max_undocumented:
            print(f"\nFAILURE: undocumented sites ({undocumented}) exceeds ceiling ({args.max_undocumented})")
            failed = True

    if failed:
        sys.exit(1)

    print("\nAudit status: PASS")

if __name__ == '__main__':
    main()
