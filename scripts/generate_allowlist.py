#!/usr/bin/env python3
"""Generate docs/review/index_allowlist.md with per-site justifications for Rule D10.1."""

import os
import sys
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from audit_common import scan_rust_file, is_test_file

INDEX_PATTERN = re.compile(r'\b(?:data|buffer|bytes|input|slice|src|buf|in_buf|out_buf)\[[^\]]+\]')

TARGETS = [
    ('entropy', ['src/entropy']),
    ('compression', ['src/compression']),
    ('blob_store', ['src/blob_store']),
    ('var_int', ['src/io/var_int.rs', 'src/io/var_int_variants.rs']),
    ('succinct_load', ['src/succinct/elias_fano/basic.rs', 'src/succinct/elias_fano/optimal.rs', 'src/succinct/elias_fano/partitioned.rs']),
]

JUSTIFICATIONS = {
    'entropy': 'C1 decoder path: direct slice indexing to be replaced by .get(..).ok_or(..) or proven bounded at loop header during C1 review.',
    'compression': 'Compression decoder path: buffer indexing to be audited against input lengths and replaced with checked access during C1/C2 review.',
    'blob_store': 'C2 blob store path: serialized format buffer access to be audited under G13 binary layout spec and replaced with checked access during C2 review.',
    'var_int': 'C1 varint decode path: multi-byte integer decoding to be checked against slice bounds or replaced with checked access during C1 review.',
    'succinct_load': 'Succinct batch decoder: internal fixed-size 8-element batch buffer bounded by compile-time array size; to be audited during C5 review.',
}

def main():
    root_dir = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    all_sites = []

    for name, paths in TARGETS:
        for p in paths:
            full_p = os.path.join(root_dir, p)
            if os.path.isfile(full_p):
                rel_path = os.path.relpath(full_p, root_dir)
                for item in scan_rust_file(full_p):
                    if not item['in_test'] and INDEX_PATTERN.search(item['clean_line']):
                        all_sites.append((rel_path, item['line_num'], name, item['clean_line']))
            elif os.path.isdir(full_p):
                for r, dirs, files in os.walk(full_p):
                    for f in sorted(files):
                        if f.endswith('.rs') and not is_test_file(os.path.join(r, f)):
                            fp = os.path.join(r, f)
                            rel_path = os.path.relpath(fp, root_dir)
                            for item in scan_rust_file(fp):
                                if not item['in_test'] and INDEX_PATTERN.search(item['clean_line']):
                                    all_sites.append((rel_path, item['line_num'], name, item['clean_line']))

    out_path = os.path.join(root_dir, 'docs', 'review', 'index_allowlist.md')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, 'w', encoding='utf-8') as out:
        out.write('# Decoder Direct Indexing Allowlist and Justifications (Rule D10.1)\n\n')
        out.write('This document inventories all direct slice indexing sites (`data[i]`, `bytes[..]`, `buf[..]`, etc.)\n')
        out.write('in decoder paths across `entropy/`, `compression/`, `blob_store/`, `io/var_int*.rs`, and `succinct/*/` load paths.\n\n')
        out.write('**Rule D10.1 / Agreement 12 Requirement:**\n')
        out.write('- Decoders taking untrusted bytes must return `Err`, never panic on malformed input.\n')
        out.write('- Direct indexing must be replaced by `.get(..).ok_or(..)` or guarded by a bounds proof at the loop header.\n')
        out.write('- Each site in this allowlist is tracked by `scripts/index_audit.py` with exact per-subsystem ratchet ceilings.\n')
        out.write('- During Phase C review sessions (C1, C2, C5), reviewers must eliminate these sites or document proofs.\n\n')
        out.write(f'Total permitted sites: {len(all_sites)}\n\n')
        out.write('| # | Location | Subsystem | Code Snippet | Justification / Phase C Resolution |\n')
        out.write('|---|---|---|---|---|\n')
        for idx, (path, lno, subsys, code) in enumerate(all_sites, 1):
            clean_code = code.replace('|', '\\|')
            just = JUSTIFICATIONS.get(subsys, 'To be audited in Phase C.')
            out.write(f'| {idx} | `{path}:{lno}` | `{subsys}` | `{clean_code}` | {just} |\n')

    print(f'Successfully wrote {len(all_sites)} entries to {out_path}')

if __name__ == '__main__':
    main()
