#!/usr/bin/env python3
"""Generate docs/review/todo_triage.md by inventorying all honesty markers."""

import os
import sys
import re

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

def main():
    hits = []
    src_dir = os.path.join(SCRIPT_DIR, '..', 'src')
    for root, dirs, files in os.walk(src_dir):
        for f in sorted(files):
            if f.endswith('.rs') and not is_test_file(os.path.join(root, f)):
                filepath = os.path.join(root, f)
                rel_path = os.path.relpath(filepath, os.path.join(SCRIPT_DIR, '..'))
                for item in scan_rust_file(filepath):
                    if not item['in_test']:
                        line = item['raw_line']
                        for pat, lbl in PATTERNS:
                            if pat.search(line):
                                hits.append((rel_path, item['line_num'], lbl, line.strip()))
                                break

    out_path = os.path.join(SCRIPT_DIR, '..', 'docs', 'review', 'todo_triage.md')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as out:
        out.write('# API Honesty and Stub Marker Triage (Rule D1)\n\n')
        out.write('This document inventories all 182 stub, placeholder, and honesty markers found in non-test production code,\n')
        out.write('assigning each to its corresponding Phase C review session for implementation, proper error reporting, or removal.\n\n')
        out.write('**Triage Policy (D1 / Agreement 8):**\n')
        out.write('1. **Implement**: Provide the fully working, tested implementation.\n')
        out.write('2. **Unsupported Error**: Return `Err(ZiporaError::unsupported(..))` if the strategy/feature is unconfigured/unsupported.\n')
        out.write('3. **Delete**: Remove the stubbed API if it does not belong in the public surface.\n')
        out.write('4. **Refine Prose**: For `simplified` occurrences in doc comments, audit and replace with exact algorithmic descriptions.\n\n')
        out.write(f'Total markers in non-test code: {len(hits)}\n\n')
        out.write('| # | Location | Marker Type | Snippet | Subsystem / Disposition |\n')
        out.write('|---|---|---|---|---|\n')
        for idx, (path, lno, lbl, text) in enumerate(hits, 1):
            clean_text = text.replace('|', '\\|')
            subsys = path.replace('src/', '').split('/')[0]
            if 'TODO: implement' in lbl:
                disp = f'Phase C ({subsys}): Implement, return Err(unsupported), or delete'
            elif 'simplified' in lbl:
                disp = f'Phase C ({subsys}): Audit doc/code; replace with exact specification'
            elif 'for now' in lbl:
                disp = f'Phase C ({subsys}): Remove temporary fallback; verify production invariant'
            elif 'placeholder' in lbl:
                disp = f'Phase C ({subsys}): Replace placeholder with real structure or clean up'
            elif 'stub' in lbl:
                disp = f'Phase C ({subsys}): Implement or delete stub'
            else:
                disp = f'Phase C ({subsys}): Triage'
            out.write(f'| {idx} | `{path}:{lno}` | `{lbl}` | `{clean_text}` | {disp} |\n')

    print(f'Successfully wrote {len(hits)} rows to {out_path}')

if __name__ == '__main__':
    main()
