#!/usr/bin/env python3
"""Compile and execute all Rust code blocks across README and docs/*.md via rustdoc --test.

Ensures:
1. Every markdown file in README.md, README_cn.md, and docs/*.md runs cleanly under
   `rustdoc --edition=2024 --test` within a strict per-file timeout (120s).
2. Zero failed and zero ignored/no_run/should_panic/compile_fail doctests across all
   markdown files (verified both via fence-token inspection and rustdoc summary counts).
3. Total executed markdown doctests never drop below MIN_MD_DOCTESTS (89).
"""

from concurrent.futures import ThreadPoolExecutor
import glob
import json
import os
import re
import subprocess
import sys

MIN_MD_DOCTESTS = 89
PER_FILE_TIMEOUT_SECS = 120
RESULT_RE = re.compile(
    r"test result:\s+(ok|FAILED)\.\s+(\d+)\s+passed;\s+(\d+)\s+failed;\s+(\d+)\s+ignored;"
)
LIST_MARKER_RE = re.compile(r"^(?:[-*+]|\d+[.)])[ \t]+")
FORBIDDEN_FENCE_ATTRS = {"no_run", "should_panic", "compile_fail", "ignore"}


def strip_md_container_prefixes(line):
    """Strip CommonMark blockquote and list prefixes in O(len(line)) without backtracking."""
    s = line
    while True:
        s = s.lstrip(" \t")
        if s.startswith(">"):
            s = s[1:]
            continue
        m = LIST_MARKER_RE.match(s)
        if m:
            s = s[m.end() :]
            continue
        return s


def scan_forbidden_fences(doc_path):
    """Return list of (line_num, attr, raw_fence_line) for forbidden doctest attributes."""
    violations = []
    in_fence = None  # (fence_char, fence_len)
    with open(doc_path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\r\n")
            s = strip_md_container_prefixes(line)
            if not (s.startswith("```") or s.startswith("~~~")):
                continue
            fence_char = s[0]
            fence_len = len(s) - len(s.lstrip(fence_char))
            rest = s[fence_len:]
            if in_fence is None:
                in_fence = (fence_char, fence_len)
                tokens = [t for t in re.split(r"[^A-Za-z0-9_-]+", rest.strip()) if t]
                for tok in tokens:
                    if tok in FORBIDDEN_FENCE_ATTRS or tok.startswith("ignore-"):
                        violations.append((line_num, tok, line.strip()))
            else:
                open_char, open_len = in_fence
                if fence_char == open_char and fence_len >= open_len and rest.strip() == "":
                    in_fence = None
    return violations


def resolve_zipora_rlib():
    cmd = [
        "cargo",
        "build",
        "--locked",
        "--lib",
        "--all-features",
        "--message-format=json",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr, file=sys.stderr)
        sys.exit(res.returncode)

    rlib_path = None
    for line in res.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("reason") == "compiler-artifact":
            target = msg.get("target", {})
            kinds = target.get("kind", [])
            if target.get("name") == "zipora" and ("lib" in kinds or "rlib" in kinds):
                for fn in msg.get("filenames", []):
                    if fn.endswith(".rlib"):
                        rlib_path = fn
    if not rlib_path or not os.path.exists(rlib_path):
        print("ERROR: failed to locate compiled libzipora.rlib", file=sys.stderr)
        sys.exit(1)
    parent_dir = os.path.dirname(rlib_path)
    deps_dir = (
        parent_dir
        if os.path.basename(parent_dir) == "deps"
        else os.path.join(parent_dir, "deps")
    )
    return rlib_path, deps_dir


def run_markdown_doctest(args):
    doc, rlib_path, deps_dir = args
    fence_violations = scan_forbidden_fences(doc)
    if fence_violations:
        details = "\n".join(
            f"{doc}:{ln}: forbidden doctest fence attribute `{attr}` in `{raw}`"
            for ln, attr, raw in fence_violations
        )
        return doc, 1, 0, len(fence_violations), 0, details

    cmd = [
        "rustdoc",
        "--edition=2024",
        "--test",
        doc,
        "-L",
        deps_dir,
        "--extern",
        f"zipora={rlib_path}",
    ]
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=PER_FILE_TIMEOUT_SECS,
        )
    except subprocess.TimeoutExpired:
        return doc, 124, 0, 1, 0, f"TIMEOUT after {PER_FILE_TIMEOUT_SECS}s"

    passed = 0
    failed = 0
    ignored = 0
    for m in RESULT_RE.finditer(res.stdout):
        passed += int(m.group(2))
        failed += int(m.group(3))
        ignored += int(m.group(4))

    output = (res.stdout + "\n" + res.stderr).strip()
    return doc, res.returncode, passed, failed, ignored, output


def main():
    rlib_path, deps_dir = resolve_zipora_rlib()
    docs = ["README.md", "README_cn.md"] + sorted(glob.glob("docs/*.md"))

    print("=== Markdown Doctests Gate (README.md, README_cn.md, docs/*.md) ===")
    total_passed = 0
    total_failed = 0
    total_ignored = 0
    failed_files = []

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                run_markdown_doctest,
                [(doc, rlib_path, deps_dir) for doc in docs],
            )
        )

    for doc, rc, passed, failed, ignored, output in results:
        status = "OK" if (rc == 0 and failed == 0 and ignored == 0) else "FAIL"
        print(
            f"  {doc:<28} {passed:>3} passed, {failed:>2} failed, {ignored:>2} ignored [{status}]"
        )
        total_passed += passed
        total_failed += failed
        total_ignored += ignored
        if status != "OK":
            failed_files.append((doc, output))

    print("-" * 68)
    print(
        f"Total across {len(docs)} markdown files: "
        f"{total_passed} passed (min: {MIN_MD_DOCTESTS}), "
        f"{total_failed} failed, {total_ignored} ignored"
    )

    if failed_files:
        for doc, output in failed_files:
            print(f"\n--- FAILURE in {doc} ---", file=sys.stderr)
            print(output, file=sys.stderr)
        sys.exit(1)

    if total_ignored > 0:
        print(
            f"FAILURE: {total_ignored} markdown doctests are ignored/no_run (must be 0)",
            file=sys.stderr,
        )
        sys.exit(1)

    if total_passed < MIN_MD_DOCTESTS:
        print(
            f"FAILURE: markdown doctest count ({total_passed}) drifted below minimum ({MIN_MD_DOCTESTS})",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Markdown doctests check: PASS")


if __name__ == "__main__":
    main()
