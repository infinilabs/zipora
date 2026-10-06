#!/usr/bin/env python3
"""Audit review ledger markdown files for citation, commit-table, and identifier accuracy (S21-R3 / S23-R2 / S24-R2).

Usage:
    python3 scripts/audit_ledger.py docs/review/io.md [docs/review/thread_and_concurrency.md ...]
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
from typing import Dict, List, Optional, Set, Tuple

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Citation pattern: `commit:path:line_spec`
CITATION_RE = re.compile(
    r"`([0-9a-f]{7,40}):([a-zA-Z0-9_./-]+):([0-9,\s\-]+)`"
)

# Preceding backticked symbol before a citation: `Symbol` (`commit:path:line`)
SYMBOL_BEFORE_CITATION_RE = re.compile(
    r"`([A-Za-z_][A-Za-z0-9_:]*(?:\(\))?)`\s*\(`([0-9a-f]{7,40}):([a-zA-Z0-9_./-]+):([0-9,\s\-]+)`\)"
)

# Commit table row: | 1 | `commit` | `parent` | YYYY-MM-DD | subject | files | +ins / −del |
COMMIT_ROW_RE = re.compile(
    r"^\|\s*\d+\s*\|\s*`([0-9a-f]{7,40})`\s*\|\s*`([0-9a-f]{7,40})`\s*\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*(.*?)\s*\|\s*(\d+)\s*\|\s*\+(\d+)\s*/\s*[−-](\d+)\s*\|$"
)

IDENT_TOKEN_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
INLINE_CODE_RE = re.compile(r"`([^`]+)`")
TYPE_MEMBER_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)::([A-Za-z_][A-Za-z0-9_]*)(?:\(.*\))?$")

STD_AND_EXTERNAL_WHITELIST: Set[str] = {
    # Rust keywords / primitives / std / core / alloc / libc / nix / tokio / crossbeam / memmap2 / bytemuck
    "Self", "Ok", "Err", "Some", "None", "Box", "Rc", "Arc", "Weak", "Vec", "VecDeque",
    "String", "Option", "Result", "HashMap", "HashSet", "BTreeMap", "BTreeSet", "BinaryHeap",
    "Cell", "RefCell", "UnsafeCell", "OnceLock", "LazyLock", "Mutex", "RwLock", "Condvar",
    "AtomicBool", "AtomicU8", "AtomicI8", "AtomicU16", "AtomicI16", "AtomicU32", "AtomicI32",
    "AtomicU64", "AtomicI64", "AtomicUsize", "AtomicIsize", "AtomicPtr", "Ordering",
    "Relaxed", "Acquire", "Release", "AcqRel", "SeqCst", "ManuallyDrop", "MaybeUninit",
    "NonNull", "PhantomData", "Pin", "Unpin", "Send", "Sync", "Copy", "Clone", "Drop",
    "Default", "Debug", "Display", "PartialEq", "Eq", "PartialOrd", "Ord", "Hash", "Hasher",
    "Read", "Write", "Seek", "BufRead", "SeekFrom", "Cursor", "File", "OpenOptions",
    "Error", "ErrorKind", "IoSlice", "IoSliceMut", "Path", "PathBuf", "Duration", "Instant",
    "SystemTime", "Thread", "JoinHandle", "Builder", "Scope", "Layout", "GlobalAlloc",
    "Any", "TypeId", "Cow", "Borrow", "BorrowMut", "AsRef", "AsMut", "Deref", "DerefMut",
    "Index", "IndexMut", "Iterator", "IntoIterator", "ExactSizeIterator", "DoubleEndedIterator",
    "From", "Into", "TryFrom", "TryInto", "Future", "Poll", "Context", "Waker",
    "Pod", "Zeroable", "AnyBitPattern", "NoUninit", "Mmap", "MmapMut", "MmapOptions",
    "NamedTempFile", "TempDir", "Semaphore", "Notify", "Barrier", "oneshot", "mpsc",
    "BorrowMutError", "BorrowError", " AssertUnwindSafe", "AssertUnwindSafe",
    "u8", "i8", "u16", "i16", "u32", "i32", "u64", "i64", "u128", "i128", "usize", "isize",
    "f32", "f64", "bool", "char", "str", "c_int", "c_void", "c_long", "size_t", "off_t",
    "timespec", "iovec", "sigaction", "siginfo_t",
    "std", "core", "alloc", "libc", "nix", "tokio", "crossbeam", "crossbeam_epoch",
    "Collector", "Global", "push_bag",
    "memmap2", "bytemuck", "tempfile", "rayon", "serde", "zstd", "lz4", "futures",
    "io", "fs", "mem", "ptr", "slice", "sync", "atomic", "thread", "time", "fmt", "ops",
    "cmp", "collections", "any", "cell", "hint", "arch", "x86_64", "aarch64", "unix", "linux",
}


def git_output(args: List[str]) -> Optional[str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


def parse_line_ranges(spec: str) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            s_str, e_str = part.split("-", 1)
            ranges.append((int(s_str.strip()), int(e_str.strip())))
        else:
            ln = int(part)
            ranges.append((ln, ln))
    return ranges


def collect_repo_identifiers(cited_blobs: Dict[Tuple[str, str], List[str]]) -> Set[str]:
    tokens: Set[str] = set(STD_AND_EXTERNAL_WHITELIST)
    for top_file in ["Cargo.toml", "Makefile", "tsan.supp"]:
        p = REPO_ROOT / top_file
        if p.is_file():
            try:
                tokens.update(IDENT_TOKEN_RE.findall(p.read_text(encoding="utf-8", errors="ignore")))
            except OSError:
                pass
    for root_dir in ["src", "tests", "benches", "scripts"]:
        base = REPO_ROOT / root_dir
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in {".rs", ".py", ".toml", ".sh"}:
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                tokens.update(IDENT_TOKEN_RE.findall(text))
    for (_, _), lines in cited_blobs.items():
        for line in lines:
            tokens.update(IDENT_TOKEN_RE.findall(line))
    return tokens


def audit_ledger(md_path: pathlib.Path) -> Tuple[int, int, int, List[str]]:
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    blob_cache: Dict[Tuple[str, str], Optional[List[str]]] = {}
    commit_valid_cache: Dict[str, bool] = {}

    def get_blob_lines(commit: str, rel_path: str) -> Optional[List[str]]:
        key = (commit, rel_path)
        if key not in blob_cache:
            out = git_output(["show", f"{commit}:{rel_path}"])
            blob_cache[key] = out.splitlines() if out is not None else None
        return blob_cache[key]

    def is_valid_commit(commit: str) -> bool:
        if commit not in commit_valid_cache:
            out = git_output(["cat-file", "-t", commit])
            commit_valid_cache[commit] = out is not None and out.strip() == "commit"
        return commit_valid_cache[commit]

    errors: List[str] = []
    citation_count = 0
    commit_row_count = 0
    type_member_count = 0

    in_fence = False
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        # 1. Check commit table rows
        m_row = COMMIT_ROW_RE.match(stripped)
        if m_row:
            commit_row_count += 1
            commit, parent, date, subject, files_s, ins_s, dels_s = m_row.groups()
            meta = git_output(["show", "-s", "--format=%p|%as|%s", commit])
            if meta is None:
                errors.append(f"{md_path}:{idx}: commit `{commit}` in table does not exist")
                continue
            actual_parent, actual_date, actual_subject = meta.strip().split("|", 2)
            if not actual_parent.startswith(parent):
                errors.append(
                    f"{md_path}:{idx}: commit `{commit}` parent mismatch: ledger `{parent}` vs git `{actual_parent}`"
                )
            if actual_date != date:
                errors.append(
                    f"{md_path}:{idx}: commit `{commit}` date mismatch: ledger `{date}` vs git `{actual_date}`"
                )
            if actual_subject != subject:
                errors.append(
                    f"{md_path}:{idx}: commit `{commit}` subject mismatch:\n"
                    f"  ledger: `{subject}`\n"
                    f"  git:    `{actual_subject}`"
                )

        # 2. Check all `commit:path:line_spec` citations
        for m_cit in CITATION_RE.finditer(line):
            citation_count += 1
            commit, rel_path, line_spec = m_cit.groups()
            if not is_valid_commit(commit):
                errors.append(f"{md_path}:{idx}: invalid commit `{commit}` in `{m_cit.group(0)}`")
                continue
            blob_lines = get_blob_lines(commit, rel_path)
            if blob_lines is None:
                errors.append(
                    f"{md_path}:{idx}: path `{rel_path}` not found in commit `{commit}` (`{m_cit.group(0)}`)"
                )
                continue
            total = len(blob_lines)
            try:
                ranges = parse_line_ranges(line_spec)
            except ValueError:
                errors.append(f"{md_path}:{idx}: malformed line spec in `{m_cit.group(0)}`")
                continue
            for start_ln, end_ln in ranges:
                if start_ln < 1 or end_ln < start_ln or end_ln > total:
                    errors.append(
                        f"{md_path}:{idx}: line range {start_ln}-{end_ln} out of bounds (1..={total}) in `{m_cit.group(0)}`"
                    )

        # 3. Check `Symbol` (`commit:path:line`) symbol containment within enclosing item (-90..+15 lines)
        for m_sym in SYMBOL_BEFORE_CITATION_RE.finditer(line):
            sym_raw, commit, rel_path, line_spec = m_sym.groups()
            blob_lines = get_blob_lines(commit, rel_path)
            if not blob_lines:
                continue
            leaf = sym_raw.rstrip("()").split("::")[-1]
            if not leaf or leaf in {"Self", "new"}:
                continue
            ranges = parse_line_ranges(line_spec)
            window_text = []
            for start_ln, end_ln in ranges:
                lo = max(1, start_ln - 90)
                hi = min(len(blob_lines), end_ln + 15)
                window_text.extend(blob_lines[lo - 1 : hi])
            joined = "\n".join(window_text)
            if leaf not in joined:
                errors.append(
                    f"{md_path}:{idx}: symbol `{sym_raw}` (leaf `{leaf}`) not found near `{commit}:{rel_path}:{line_spec}`"
                )

    # 4. Check Type::member pairs in inline code spans against all repo + historical cited blobs
    valid_blobs = {k: v for k, v in blob_cache.items() if v is not None}
    all_tokens = collect_repo_identifiers(valid_blobs)

    in_fence = False
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for m_code in INLINE_CODE_RE.finditer(line):
            span = m_code.group(1).strip()
            m_tm = TYPE_MEMBER_RE.match(span)
            if m_tm:
                type_member_count += 1
                type_name, member_name = m_tm.groups()
                if type_name not in all_tokens:
                    errors.append(
                        f"{md_path}:{idx}: unknown type `{type_name}` in `{span}`"
                    )
                if member_name not in all_tokens:
                    errors.append(
                        f"{md_path}:{idx}: unknown member `{member_name}` in `{span}`"
                    )

    return citation_count, commit_row_count, type_member_count, errors


def main() -> int:
    paths = [pathlib.Path(p) for p in sys.argv[1:]]
    if not paths:
        paths = sorted((REPO_ROOT / "docs" / "review").glob("*.md"))

    any_errors = False
    for path in paths:
        citations, rows, tm_pairs, errors = audit_ledger(path)
        print(
            f"=== Ledger Audit: {path} ===\n"
            f"Citations checked:    {citations}\n"
            f"Commit rows checked:  {rows}\n"
            f"Type::member checked: {tm_pairs}\n"
            f"Errors found:         {len(errors)}"
        )
        for err in errors:
            print(f"  ERROR: {err}")
        if errors:
            any_errors = True
        else:
            print("Ledger audit status: PASS")
    return 1 if any_errors else 0


if __name__ == "__main__":
    sys.exit(main())
