#!/usr/bin/env python3
"""Audit production source code for Phase D hygiene gates (D6, D7, D11).

Rules:
- D6: Non-test `allow(dead_code)` must not exceed `max_dead_code_allows` (<= 5)
      and every surviving site must carry a same-line `// reason` comment.
- D7: Zero `println!` or `eprintln!` in non-test production code (`max_prints = 0`).
- D11: Non-test `allow(clippy::should_implement_trait)` and
       `allow(clippy::too_many_arguments)` must not exceed
       `max_clippy_trait_arg_allows` and every surviving site must carry a
       same-line `// reason` comment.
"""

import argparse
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from audit_common import scan_rust_file, is_test_file

DEAD_CODE_RE = re.compile(r'\ballow\s*\(\s*dead_code\s*\)')
PRINT_RE = re.compile(r'\b(?:e?println)\s*!')
CLIPPY_TRAIT_ARG_RE = re.compile(
    r'\ballow\s*\(\s*clippy::(?:should_implement_trait|too_many_arguments)\s*\)'
)


def load_baseline():
    baseline_path = os.path.join(SCRIPT_DIR, 'audit_baseline.toml')
    max_dead_code = 5
    max_prints = 0
    max_clippy_trait_arg = 22
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('max_dead_code_allows'):
                    max_dead_code = int(line.split('=')[1].strip())
                elif line.startswith('max_prints'):
                    max_prints = int(line.split('=')[1].strip())
                elif line.startswith('max_clippy_trait_arg_allows'):
                    max_clippy_trait_arg = int(line.split('=')[1].strip())
    return max_dead_code, max_prints, max_clippy_trait_arg


def has_same_line_reason(raw_line, clean_line):
    idx = raw_line.find('//')
    if idx < 0:
        return False
    reason = raw_line[idx + 2:].strip()
    return len(reason) >= 5


def audit_file(filepath):
    dead_code = []
    dead_code_unjustified = []
    prints = []
    clippy_allows = []
    clippy_unjustified = []

    for item in scan_rust_file(filepath):
        if item['in_test']:
            continue
        clean = item['clean_line']
        raw = item['raw_line'].rstrip('\n')
        if DEAD_CODE_RE.search(clean):
            entry = {
                'file': filepath,
                'line': item['line_num'],
                'text': raw.strip(),
            }
            dead_code.append(entry)
            if not has_same_line_reason(raw, clean):
                dead_code_unjustified.append(entry)

        if PRINT_RE.search(clean):
            prints.append({
                'file': filepath,
                'line': item['line_num'],
                'text': raw.strip(),
            })

        if CLIPPY_TRAIT_ARG_RE.search(clean):
            entry = {
                'file': filepath,
                'line': item['line_num'],
                'text': raw.strip(),
            }
            clippy_allows.append(entry)
            if not has_same_line_reason(raw, clean):
                clippy_unjustified.append(entry)

    return (
        dead_code,
        dead_code_unjustified,
        prints,
        clippy_allows,
        clippy_unjustified,
    )


def main():
    max_dead_default, max_prints_default, max_clippy_default = load_baseline()

    parser = argparse.ArgumentParser(description='Audit D6, D7, and D11 hygiene rules.')
    parser.add_argument('--src', default='src', help='Source directory to scan')
    parser.add_argument('--max-dead-code', type=int, default=max_dead_default)
    parser.add_argument('--max-prints', type=int, default=max_prints_default)
    parser.add_argument('--max-clippy-trait-arg', type=int, default=max_clippy_default)
    args = parser.parse_args()

    all_dead = []
    all_dead_unjustified = []
    all_prints = []
    all_clippy = []
    all_clippy_unjustified = []

    for root, _, files in os.walk(args.src):
        for f in sorted(files):
            filepath = os.path.join(root, f)
            if f.endswith('.rs') and not is_test_file(filepath):
                d, du, p, c, cu = audit_file(filepath)
                all_dead.extend(d)
                all_dead_unjustified.extend(du)
                all_prints.extend(p)
                all_clippy.extend(c)
                all_clippy_unjustified.extend(cu)

    print("=== Phase D Hygiene Audit (Rules D6 / D7 / D11) ===")
    print(
        f"D6 allow(dead_code): {len(all_dead)} "
        f"(ceiling: {args.max_dead_code}, unjustified: {len(all_dead_unjustified)})"
    )
    print(
        f"D7 production println!/eprintln!: {len(all_prints)} "
        f"(ceiling: {args.max_prints})"
    )
    print(
        f"D11 clippy trait/arg allows: {len(all_clippy)} "
        f"(ceiling: {args.max_clippy_trait_arg}, unjustified: {len(all_clippy_unjustified)})"
    )

    failed = False
    if len(all_dead) > args.max_dead_code:
        print(f"\nFAILURE [D6]: allow(dead_code) count ({len(all_dead)}) exceeds ceiling ({args.max_dead_code}):")
        for e in all_dead:
            print(f"  {e['file']}:{e['line']}: {e['text']}")
        failed = True

    if all_dead_unjustified:
        print(f"\nFAILURE [D6]: allow(dead_code) missing same-line // reason ({len(all_dead_unjustified)}):")
        for e in all_dead_unjustified:
            print(f"  {e['file']}:{e['line']}: {e['text']}")
        failed = True

    if len(all_prints) > args.max_prints:
        print(f"\nFAILURE [D7]: production println!/eprintln! count ({len(all_prints)}) exceeds ceiling ({args.max_prints}):")
        for e in all_prints:
            print(f"  {e['file']}:{e['line']}: {e['text']}")
        failed = True

    if len(all_clippy) > args.max_clippy_trait_arg:
        print(
            f"\nFAILURE [D11]: clippy trait/arg allow count ({len(all_clippy)}) "
            f"exceeds ceiling ({args.max_clippy_trait_arg}):"
        )
        for e in all_clippy:
            print(f"  {e['file']}:{e['line']}: {e['text']}")
        failed = True

    if all_clippy_unjustified:
        print(
            f"\nFAILURE [D11]: clippy trait/arg allow missing same-line // reason "
            f"({len(all_clippy_unjustified)}):"
        )
        for e in all_clippy_unjustified:
            print(f"  {e['file']}:{e['line']}: {e['text']}")
        failed = True

    if failed:
        sys.exit(1)

    print("\nAudit status: PASS")


if __name__ == '__main__':
    main()
