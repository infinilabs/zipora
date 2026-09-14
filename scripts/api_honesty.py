#!/usr/bin/env python3
"""Audit API honesty and stub markers in production source code.

Rule D1 / Agreement 8:
- No silent stubs (TODO: implement returning 0/None/Ok, placeholder, simplified, stub).
- make api_honesty greps for TODO: [Ii]mplement, for now, until .* fixed, simplified, placeholder, stub in non-test code.
- Note: 'simplified' matches both code markers and descriptive prose in doc comments (e.g. algorithmic explanations);
  subsystem owners in Phase C sessions are expected to audit and remove/refine these sites as part of their review pass.
- Ratchet ceiling is stored in scripts/audit_baseline.toml and must not increase.
"""

import argparse
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from audit_common import scan_rust_file, is_test_file

PATTERNS = [
    (re.compile(r'TODO:\s*[Ii]mplement'), 'TODO: implement'),
    (re.compile(r'\bfor now\b', re.IGNORECASE), 'for now'),
    (re.compile(r'until\s+.*\s+fixed', re.IGNORECASE), 'until ... fixed'),
    (re.compile(r'\bsimplified\b', re.IGNORECASE), 'simplified'),
    (re.compile(r'\bplaceholder\b', re.IGNORECASE), 'placeholder'),
    (re.compile(r'\bstub\b', re.IGNORECASE), 'stub'),
]

def load_baseline():
    baseline_path = os.path.join(SCRIPT_DIR, 'audit_baseline.toml')
    max_markers = 182
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('max_markers'):
                    max_markers = int(line.split('=')[1].strip())
    return max_markers

def audit_file(filepath):
    hits = []
    for item in scan_rust_file(filepath):
        if item['in_test']:
            continue
        line = item['raw_line']
        for pattern, label in PATTERNS:
            if pattern.search(line):
                hits.append({
                    'file': filepath,
                    'line': item['line_num'],
                    'label': label,
                    'text': line.strip(),
                })
                break
    return hits

def main():
    max_markers_default = load_baseline()

    parser = argparse.ArgumentParser(description='Audit API honesty markers.')
    parser.add_argument('--src', default='src', help='Source directory to scan')
    parser.add_argument('--strict', action='store_true', help='Fail if any markers are found (exit criteria G2/D1)')
    parser.add_argument('--max-markers', type=int, default=max_markers_default, help='Maximum allowed markers before failing (ceiling during transition)')
    args = parser.parse_args()

    all_hits = []
    by_label = {}
    by_file = {}

    for root, dirs, files in os.walk(args.src):
        for f in sorted(files):
            if f.endswith('.rs') and not is_test_file(os.path.join(root, f)):
                filepath = os.path.join(root, f)
                hits = audit_file(filepath)
                all_hits.extend(hits)
                for h in hits:
                    lbl = h['label']
                    by_label[lbl] = by_label.get(lbl, 0) + 1
                    by_file[filepath] = by_file.get(filepath, 0) + 1

    total = len(all_hits)
    print("=== API Honesty Audit (Rule D1) ===")
    print(f"Total honesty markers found: {total} (ratchet ceiling: {args.max_markers})")
    print("\nBreakdown by marker type:")
    for lbl, count in sorted(by_label.items(), key=lambda x: x[1], reverse=True):
        print(f"  {lbl:<20}: {count}")

    print("\nTop files with markers:")
    for fpath, count in sorted(by_file.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {fpath:<45}: {count}")

    # Always print all findings by default
    print(f"\nAll findings ({total}):")
    for h in all_hits:
        print(f"  {h['file']}:{h['line']} [{h['label']}]: {h['text']}")

    failed = False
    if args.strict and total > 0:
        print(f"\nFAILURE: --strict mode requires 0 honesty markers (found {total})")
        failed = True
    elif total > args.max_markers:
        print(f"\nFAILURE: honesty markers ({total}) exceeds ratchet ceiling ({args.max_markers})")
        failed = True

    if failed:
        sys.exit(1)

    print("\nAudit status: PASS")

if __name__ == '__main__':
    main()
