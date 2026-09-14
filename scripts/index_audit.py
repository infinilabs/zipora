#!/usr/bin/env python3
"""Audit slice indexing vs bounds-checked get() operations in decoder subsystems.

Rule D10.1 / Agreement 12:
- Decoders return Err, never panic.
- In entropy/, compression/, blob_store/, io/var_int*.rs, succinct/*/ load paths,
  direct data[i] / &data[a..b] on caller-supplied bytes is audited against .get(..)
  or explicit bounds guards.
"""

import argparse
import os
import re
import sys

TARGET_DIRECTORIES = [
    'src/entropy',
    'src/compression',
    'src/blob_store',
    'src/io',
    'src/succinct',
]

INDEX_PATTERN = re.compile(r'\b(?:data|buffer|bytes|input|slice|src|buf|in_buf|out_buf)\[[^\]]+\]')
GET_PATTERN = re.compile(r'\.get\(')

def audit_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    index_sites = []
    get_sites = []

    in_test_mod = False
    brace_depth = 0
    test_mod_depth = -1

    for idx, line in enumerate(lines):
        line_num = idx + 1
        stripped = line.strip()

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

        if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
            continue

        if INDEX_PATTERN.search(line):
            index_sites.append({
                'file': filepath,
                'line': line_num,
                'text': stripped,
            })
        if GET_PATTERN.search(line):
            get_sites.append({
                'file': filepath,
                'line': line_num,
                'text': stripped,
            })

    return index_sites, get_sites

def main():
    parser = argparse.ArgumentParser(description='Audit direct slice indexing in decoders.')
    parser.add_argument('--max-index-sites', type=int, default=400, help='Maximum allowed direct indexing sites before failing')
    parser.add_argument('--verbose', action='store_true', help='Print individual indexing sites')
    args = parser.parse_args()

    subsystem_stats = {}
    total_index = 0
    total_get = 0
    all_sites = []

    for target in TARGET_DIRECTORIES:
        if not os.path.exists(target):
            continue
        subsystem_stats[target] = {'index': 0, 'get': 0, 'files': 0}
        for root, dirs, files in os.walk(target):
            for f in files:
                if f.endswith('.rs') and f != 'tests.rs':
                    filepath = os.path.join(root, f)
                    idx_sites, g_sites = audit_file(filepath)
                    count_i = len(idx_sites)
                    count_g = len(g_sites)
                    subsystem_stats[target]['index'] += count_i
                    subsystem_stats[target]['get'] += count_g
                    subsystem_stats[target]['files'] += 1
                    total_index += count_i
                    total_get += count_g
                    all_sites.extend(idx_sites)

    print("=== Decoder Indexing Safety Audit (Rule D10.1) ===")
    print(f"Total direct indexing sites: {total_index} (ceiling: {args.max_index_sites})")
    print(f"Total checked .get() sites:   {total_get}")
    print()
    print(f"{'Subsystem':<20} {'Files':<8} {'Direct Index':<15} {'.get()':<10}")
    print("-" * 55)
    for sub, stats in subsystem_stats.items():
        print(f"{sub:<20} {stats['files']:<8} {stats['index']:<15} {stats['get']:<10}")
    print("-" * 55)

    if args.verbose:
        print("\nDirect indexing sites:")
        for s in all_sites:
            print(f"  {s['file']}:{s['line']}: {s['text']}")

    if total_index > args.max_index_sites:
        print(f"\nFAILURE: direct indexing sites ({total_index}) exceeds ceiling ({args.max_index_sites})")
        sys.exit(1)

    print("\nAudit status: PASS")

if __name__ == '__main__':
    main()
