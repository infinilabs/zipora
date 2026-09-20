#!/usr/bin/env python3
"""Audit slice indexing vs bounds-checked get() operations in decoder subsystems.

Rule D10.1 / Agreement 12:
- Decoders return Err, never panic on untrusted input.
- In entropy/, compression/, blob_store/, io/var_int*.rs, succinct/*/ load paths,
  direct data[i] / &data[a..b] on caller-supplied bytes is audited against .get(..)
  or explicit bounds guards.
- Provably in-bounds indexing sites on encoder hot paths may be exempted with an
  inline `// D10.1: in-bounds` comment on the indexing line and are audited/ratcheted separately.
- Exact per-subsystem ceilings are stored in scripts/audit_baseline.toml and ratchet down.
"""

import argparse
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from audit_common import scan_rust_file, is_test_file

BASE_IDENTIFIERS = {
    'data', 'buffer', 'bytes', 'input', 'slice', 'src', 'buf',
    'in_buf', 'out_buf', 'hdr', 'header', 'raw_name', 'name_bytes',
}
BUF_DECL_PATTERN = re.compile(
    r'\b(?:let\s+(?:mut\s+)?([a-z_][a-z0-9_]*)\s*(?::\s*&?\[u8\s*;[^\]]+\])?\s*=\s*(?:\[0u8\s*;|\*b"|&?(?:self\.)?(?:data|buffer|bytes|input|slice|src|buf|in_buf|out_buf|hdr|header)\b)|([a-z_][a-z0-9_]*)\s*:\s*&(?:mut\s+)?\[u8\s*;[^\]]+\])'
)
ANY_BRACKET_PATTERN = re.compile(r'(?<![#!:\w])\b[a-z_][a-z0-9_]*\[[^\];]+\]')
GET_PATTERN = re.compile(r'\.get(?:_mut)?\(')
IN_BOUNDS_PATTERN = re.compile(r'//\s*D10\.1:\s*in-bounds')

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
        'max_in_bounds': 0,
    }
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                for k in ceilings.keys():
                    if line.startswith(f'{k} ='):
                        ceilings[k] = int(line.split('=')[1].strip())
    return ceilings

def check_file(full_p, rel_path, name, subsystem_stats, all_sites, in_bounds_sites):
    subsystem_stats[name]['files'] += 1
    total_idx = 0
    total_ib = 0
    total_g = 0
    items = list(scan_rust_file(full_p))
    file_idents = set(BASE_IDENTIFIERS)
    for item in items:
        if not item['in_test']:
            for m in BUF_DECL_PATTERN.finditer(item['clean_line']):
                ident = m.group(1) or m.group(2)
                if ident:
                    file_idents.add(ident)
    ident_alt = '|'.join(sorted(re.escape(i) for i in file_idents))
    file_index_pattern = re.compile(rf'\b(?:{ident_alt})\[[^\]]+\]')

    for item in items:
        if not item['in_test']:
            has_inline_ib = bool(IN_BOUNDS_PATTERN.search(item['raw_line'])) and bool(item['clean_line'].strip())
            has_above_ib = any(
                bool(IN_BOUNDS_PATTERN.search(c[1])) for c in item['comments_above'][-2:]
            )
            is_in_bounds = has_inline_ib or has_above_ib
            has_idx = bool(file_index_pattern.search(item['clean_line'])) or (
                is_in_bounds and bool(ANY_BRACKET_PATTERN.search(item['clean_line']))
            )
            if has_idx:
                site = {
                    'file': rel_path,
                    'line': item['line_num'],
                    'subsystem': name,
                    'text': item['clean_line'],
                }
                if is_in_bounds:
                    subsystem_stats[name]['in_bounds'] += 1
                    total_ib += 1
                    in_bounds_sites.append(site)
                else:
                    subsystem_stats[name]['index'] += 1
                    total_idx += 1
                    all_sites.append(site)
            if GET_PATTERN.search(item['clean_line']):
                subsystem_stats[name]['get'] += 1
                total_g += 1
    return total_idx, total_ib, total_g

def main():
    ceilings = load_baseline()

    parser = argparse.ArgumentParser(description='Audit direct slice indexing in decoders.')
    parser.add_argument('--strict', action='store_true', help='Fail if any direct indexing site is found')
    parser.add_argument('--max-in-bounds', type=int, default=ceilings.get('max_in_bounds', 0), help='Exact expected in-bounds exemptions')
    args = parser.parse_args()

    root_dir = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    subsystem_stats = {}
    total_index = 0
    total_in_bounds = 0
    total_get = 0
    all_sites = []
    in_bounds_sites = []

    for name, paths in TARGETS:
        subsystem_stats[name] = {'index': 0, 'in_bounds': 0, 'get': 0, 'files': 0}
        for p in paths:
            full_p = os.path.join(root_dir, p)
            if os.path.isfile(full_p):
                rel_path = os.path.relpath(full_p, root_dir)
                idx, ib, g = check_file(full_p, rel_path, name, subsystem_stats, all_sites, in_bounds_sites)
                total_index += idx
                total_in_bounds += ib
                total_get += g
            elif os.path.isdir(full_p):
                for r, dirs, files in os.walk(full_p):
                    for f in sorted(files):
                        if f.endswith('.rs') and not is_test_file(os.path.join(r, f)):
                            fp = os.path.join(r, f)
                            rel_path = os.path.relpath(fp, root_dir)
                            idx, ib, g = check_file(fp, rel_path, name, subsystem_stats, all_sites, in_bounds_sites)
                            total_index += idx
                            total_in_bounds += ib
                            total_get += g

    print("=== Decoder Indexing Safety Audit (Rule D10.1) ===")
    print(f"Total direct indexing sites in scoped paths: {total_index} (ratchet ceiling: {ceilings['max_total']})")
    print(f"Total in-bounds exemptions (D10.1: in-bounds): {total_in_bounds} (exact baseline: {args.max_in_bounds})")
    print(f"Total checked .get() sites:                   {total_get}")
    print()
    print(f"{'Subsystem':<20} {'Files':<8} {'Direct Index':<15} {'Ceiling':<10} {'In-Bounds':<12} {'.get()':<10}")
    print("-" * 75)
    for sub, stats in subsystem_stats.items():
        ceil_val = ceilings.get(sub, '-')
        print(f"{sub:<20} {stats['files']:<8} {stats['index']:<15} {ceil_val:<10} {stats['in_bounds']:<12} {stats['get']:<10}")
    print("-" * 75)

    # Always print all direct indexing sites
    print(f"\nAll direct indexing sites in scope ({total_index}):")
    for s in all_sites:
        print(f"  {s['file']}:{s['line']} [{s['subsystem']}]: {s['text']}")

    if in_bounds_sites:
        print(f"\nIn-bounds exemptions ({total_in_bounds}):")
        for s in in_bounds_sites:
            print(f"  {s['file']}:{s['line']} [{s['subsystem']}]: {s['text']}")

    failed = False
    if args.strict and total_index > 0:
        print(f"\nFAILURE: --strict mode requires 0 direct indexing sites (found {total_index})")
        failed = True

    if total_index > ceilings['max_total']:
        print(f"\nFAILURE: direct indexing sites ({total_index}) exceeds total ceiling ({ceilings['max_total']})")
        failed = True

    if total_in_bounds != args.max_in_bounds:
        print(f"\nFAILURE: in-bounds exemptions ({total_in_bounds}) does not match exact baseline ({args.max_in_bounds})")
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
