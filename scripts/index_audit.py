#!/usr/bin/env python3
"""Audit slice indexing vs bounds-checked get() operations in decoder subsystems.

Rule D10.1 / Agreement 12:
- Decoders return Err, never panic on untrusted input.
- In entropy/, compression/, blob_store/, io/var_int*.rs, succinct/*/ load paths,
  direct data[i] / &data[a..b] on caller-supplied bytes is audited against .get(..)
  or explicit bounds guards.
- Exact per-subsystem ceilings are stored in scripts/audit_baseline.toml and ratchet down.
- All remaining sites must have documented justifications in docs/review/index_allowlist.md.
"""

import argparse
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from audit_common import scan_rust_file, is_test_file

INDEX_PATTERN = re.compile(r'\b(?:data|buffer|bytes|input|slice|src|buf|in_buf|out_buf)\[[^\]]+\]')
GET_PATTERN = re.compile(r'\.get(?:_mut)?\(')

TARGETS = [
    ('entropy', ['src/entropy']),
    ('compression', ['src/compression']),
    ('blob_store', ['src/blob_store']),
    ('var_int', ['src/io/var_int.rs', 'src/io/var_int_variants.rs']),
    ('succinct_load', ['src/succinct/elias_fano/basic.rs', 'src/succinct/elias_fano/optimal.rs', 'src/succinct/elias_fano/partitioned.rs']),
]

def load_baseline():
    baseline_path = os.path.join(SCRIPT_DIR, 'audit_baseline.toml')
    ceilings = {
        'entropy': 94,
        'compression': 133,
        'blob_store': 50,
        'var_int': 20,
        'succinct_load': 6,
        'max_total': 303,
    }
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                for k in ceilings.keys():
                    if line.startswith(f'{k} ='):
                        ceilings[k] = int(line.split('=')[1].strip())
    return ceilings

def main():
    ceilings = load_baseline()

    parser = argparse.ArgumentParser(description='Audit direct slice indexing in decoders.')
    parser.add_argument('--strict', action='store_true', help='Fail if any direct indexing site is found')
    args = parser.parse_args()

    root_dir = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    subsystem_stats = {}
    total_index = 0
    total_get = 0
    all_sites = []

    for name, paths in TARGETS:
        subsystem_stats[name] = {'index': 0, 'get': 0, 'files': 0}
        for p in paths:
            full_p = os.path.join(root_dir, p)
            if os.path.isfile(full_p):
                rel_path = os.path.relpath(full_p, root_dir)
                subsystem_stats[name]['files'] += 1
                for item in scan_rust_file(full_p):
                    if not item['in_test']:
                        if INDEX_PATTERN.search(item['clean_line']):
                            subsystem_stats[name]['index'] += 1
                            total_index += 1
                            site = {
                                'file': rel_path,
                                'line': item['line_num'],
                                'subsystem': name,
                                'text': item['clean_line'],
                            }
                            all_sites.append(site)
                        if GET_PATTERN.search(item['clean_line']):
                            subsystem_stats[name]['get'] += 1
                            total_get += 1
            elif os.path.isdir(full_p):
                for r, dirs, files in os.walk(full_p):
                    for f in sorted(files):
                        if f.endswith('.rs') and not is_test_file(os.path.join(r, f)):
                            fp = os.path.join(r, f)
                            rel_path = os.path.relpath(fp, root_dir)
                            subsystem_stats[name]['files'] += 1
                            for item in scan_rust_file(fp):
                                if not item['in_test']:
                                    if INDEX_PATTERN.search(item['clean_line']):
                                        subsystem_stats[name]['index'] += 1
                                        total_index += 1
                                        site = {
                                            'file': rel_path,
                                            'line': item['line_num'],
                                            'subsystem': name,
                                            'text': item['clean_line'],
                                        }
                                        all_sites.append(site)
                                    if GET_PATTERN.search(item['clean_line']):
                                        subsystem_stats[name]['get'] += 1
                                        total_get += 1

    print("=== Decoder Indexing Safety Audit (Rule D10.1) ===")
    print(f"Total direct indexing sites in scoped paths: {total_index} (ratchet ceiling: {ceilings['max_total']})")
    print(f"Total checked .get() sites:                   {total_get}")
    print()
    print(f"{'Subsystem':<20} {'Files':<8} {'Direct Index':<15} {'Ceiling':<10} {'.get()':<10}")
    print("-" * 65)
    for sub, stats in subsystem_stats.items():
        ceil_val = ceilings.get(sub, '-')
        print(f"{sub:<20} {stats['files']:<8} {stats['index']:<15} {ceil_val:<10} {stats['get']:<10}")
    print("-" * 65)

    # Always print all direct indexing sites
    print(f"\nAll direct indexing sites in scope ({total_index}):")
    for s in all_sites:
        print(f"  {s['file']}:{s['line']} [{s['subsystem']}]: {s['text']}")

    failed = False
    if args.strict and total_index > 0:
        print(f"\nFAILURE: --strict mode requires 0 direct indexing sites (found {total_index})")
        failed = True

    if total_index > ceilings['max_total']:
        print(f"\nFAILURE: direct indexing sites ({total_index}) exceeds total ceiling ({ceilings['max_total']})")
        failed = True

    for sub, stats in subsystem_stats.items():
        if sub in ceilings and stats['index'] > ceilings[sub]:
            print(f"\nFAILURE: {sub} indexing sites ({stats['index']}) exceeds subsystem ceiling ({ceilings[sub]})")
            failed = True

    if failed:
        sys.exit(1)

    print("\nAudit status: PASS")

if __name__ == '__main__':
    main()
