#!/usr/bin/env python3
"""Audit API honesty and stub markers in production source code.

Rule D1 / Agreement 8:
- No silent stubs (TODO: implement returning 0/None/Ok, placeholder, simplified, stub).
- make api_honesty greps for TODO: [Ii]mplement, for now, until .* fixed, simplified, placeholder, stub in non-test code.
"""

import argparse
import os
import re
import sys

PATTERNS = [
    (re.compile(r'TODO:\s*[Ii]mplement'), 'TODO: implement'),
    (re.compile(r'\bfor now\b', re.IGNORECASE), 'for now'),
    (re.compile(r'until\s+.*\s+fixed', re.IGNORECASE), 'until ... fixed'),
    (re.compile(r'\bsimplified\b', re.IGNORECASE), 'simplified'),
    (re.compile(r'\bplaceholder\b', re.IGNORECASE), 'placeholder'),
    (re.compile(r'\bstub\b', re.IGNORECASE), 'stub'),
]

def audit_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    hits = []
    in_test_mod = False
    brace_depth = 0
    test_mod_depth = -1

    for idx, line in enumerate(lines):
        line_num = idx + 1

        if '#[cfg(test)]' in line:
            in_test_mod = True
            test_mod_depth = brace_depth

        open_braces = line.count('{')
        close_braces = line.count('}')
        brace_depth += open_braces - close_braces

        if in_test_mod and brace_depth <= test_mod_depth:
            in_test_mod = False
            test_mod_depth = -1

        if in_test_mod:
            continue

        for pattern, label in PATTERNS:
            if pattern.search(line):
                hits.append({
                    'file': filepath,
                    'line': line_num,
                    'label': label,
                    'text': line.strip(),
                })
                break

    return hits

def main():
    parser = argparse.ArgumentParser(description='Audit API honesty markers.')
    parser.add_argument('--src', default='src', help='Source directory to scan')
    parser.add_argument('--strict', action='store_true', help='Fail if any markers are found (exit criteria G2/D1)')
    parser.add_argument('--max-markers', type=int, default=200, help='Maximum allowed markers before failing (ceiling during transition)')
    parser.add_argument('--verbose', action='store_true', help='Print all occurrences')
    args = parser.parse_args()

    all_hits = []
    by_label = {}
    by_file = {}

    for root, dirs, files in os.walk(args.src):
        for f in files:
            if f.endswith('.rs') and f != 'tests.rs':
                filepath = os.path.join(root, f)
                hits = audit_file(filepath)
                all_hits.extend(hits)
                for h in hits:
                    lbl = h['label']
                    by_label[lbl] = by_label.get(lbl, 0) + 1
                    by_file[filepath] = by_file.get(filepath, 0) + 1

    total = len(all_hits)
    print("=== API Honesty Audit ===")
    print(f"Total honesty markers found: {total} (ceiling: {args.max_markers})")
    print("\nBreakdown by marker type:")
    for lbl, count in sorted(by_label.items(), key=lambda x: x[1], reverse=True):
        print(f"  {lbl:<20}: {count}")

    print("\nTop files with markers:")
    for fpath, count in sorted(by_file.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {fpath:<45}: {count}")

    if args.verbose or (args.strict and total > 0):
        print("\nAll findings:")
        for h in all_hits:
            print(f"  {h['file']}:{h['line']} [{h['label']}]: {h['text']}")

    failed = False
    if args.strict and total > 0:
        print(f"\nFAILURE: --strict mode requires 0 honesty markers (found {total})")
        failed = True
    elif total > args.max_markers:
        print(f"\nFAILURE: honesty markers ({total}) exceeds allowed ceiling ({args.max_markers})")
        failed = True

    if failed:
        sys.exit(1)

    print("\nAudit status: PASS")

if __name__ == '__main__':
    main()
