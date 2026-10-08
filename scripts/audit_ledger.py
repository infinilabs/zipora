#!/usr/bin/env python3
"""Audit review ledger markdown files for citation, commit-table, item-span, identifier, and AST Type::member accuracy.

Usage:
    python3 scripts/audit_ledger.py docs/review/io.md [docs/review/thread_and_concurrency.md ...]
"""

from __future__ import annotations

import collections
import os
import pathlib
import re
import subprocess
import sys
from typing import Dict, List, Optional, Set, Tuple

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from audit_common import is_test_file, is_test_only_cfg  # noqa: E402

CITE_RE = re.compile(
    r"(?<![0-9A-Za-z])([0-9a-f]{7,40}):([A-Za-z0-9_.][A-Za-z0-9_./\-]*?):(\d+)(?:-(\d+))?(?![0-9])"
)

MULTI_CITE_RE = re.compile(
    r"(?<![0-9A-Za-z])([0-9a-f]{7,40}):([A-Za-z0-9_.][A-Za-z0-9_./\-]*?):(\d+(?:-\d+)?(?:\s*,\s*\d+(?:-\d+)?)+)(?![0-9A-Za-z:])"
)

COMMIT_ROW_RE = re.compile(
    r"^\|\s*\d+\s*\|\s*`([0-9a-f]{7,40})`\s*\|(?:\s*`([0-9a-f]{7,40})`\s*\|)?\s*(\d{4}-\d{2}-\d{2})\s*\|\s*(.*?)\s*\|\s*(\d+)\s*\|\s*([+\u2212-]\d+)\s*/\s*([+\u2212-]\d+)\s*\|$"
)

BT_RE = re.compile(r"`([^`]+)`")
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
SNAKE_RE = re.compile(r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b")
CAMEL_RE = re.compile(
    r"\b(?:[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+|[A-Z]{2,}[a-z]{2,}[A-Za-z0-9]*)\b"
)
HYPHEN_IDENT_RE = re.compile(r"\b[a-z]{2,}(?:-[a-z]{2,}){2,}\b")
PAIR_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9_]*)(?:<[^`>]*>)?::(?:\{([^}]*)\}|([A-Za-z_][A-Za-z0-9_]*))"
)

ITEM_RE = re.compile(
    r"^\s*(?:#\[[^\]]*\]\s*)*(?:pub(?:\([^)]*\))?\s+)?(?:(?:const|unsafe|async|extern\s+\"C\")\s+)*(fn|struct|enum|trait|mod|macro_rules!|const|static|type|union)\s+([A-Za-z_]\w*)"
)
IMPL_RE = re.compile(r"^\s*(?:unsafe\s+)?impl\b(.*)")

HDR_RE = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:unsafe\s+)?(impl\b[^{;]*|trait\s+\w+[^{;]*|enum\s+\w+[^{;]*|struct\s+\w+[^{;]*|mod\s+\w+)\s*\{?\s*$|"
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:unsafe\s+)?(impl\b[^{;]*?|trait\s+\w+[^{;]*?|enum\s+\w+[^{;]*?|struct\s+\w+[^{;]*?|mod\s+\w+)\s*\{"
)

ARROW_LOC_RE = re.compile(r"-->\s+([A-Za-z0-9_./\-]+\.rs):(\d+):(\d+)")
PANIC_LOC_RE = re.compile(
    r"thread\s+'([^']+)'\s+(?:\(\d+\)\s+)?panicked at\s+([A-Za-z0-9_./\-]+\.rs):(\d+):(\d+)"
)
UNSAFE_SITE_RE = re.compile(r"\bunsafe\s*(?:\{|fn\b|impl\b|trait\b)")

UNIVERSAL_TRAIT_METHODS: Set[str] = {
    "clone", "default", "drop", "fmt", "eq", "ne", "partial_cmp", "cmp",
    "hash", "deref", "deref_mut", "index", "index_mut", "from", "into",
    "try_from", "try_into", "serialize", "deserialize", "borrow", "borrow_mut",
}

ATOMIC_METHODS: Set[str] = {
    "new", "load", "store", "swap", "compare_exchange", "compare_exchange_weak",
    "fetch_add", "fetch_sub", "fetch_and", "fetch_or", "fetch_xor", "fetch_max",
    "fetch_min", "fetch_update", "into_inner", "get_mut",
}

SMART_PTR_METHODS: Set[str] = {
    "new", "new_cyclic", "clone", "downgrade", "upgrade", "strong_count",
    "weak_count", "as_ptr", "into_raw", "from_raw", "get_mut", "make_mut",
    "try_unwrap", "ptr_eq", "pin", "leak", "into_boxed_slice",
}

VEC_METHODS: Set[str] = {
    "new", "with_capacity", "len", "is_empty", "capacity", "push", "pop",
    "push_back", "push_front", "pop_back", "pop_front", "insert", "remove",
    "get", "get_mut", "clear", "reserve", "resize", "resize_with", "truncate",
    "extend", "extend_from_slice", "as_ptr", "as_mut_ptr", "as_slice",
    "as_mut_slice", "from_raw_parts", "from_raw_parts_mut", "set_len",
    "into_boxed_slice", "iter", "iter_mut", "into_iter", "retain",
    "swap_remove", "drain", "split_off",
}

MAP_METHODS: Set[str] = {
    "new", "with_capacity", "len", "is_empty", "insert", "remove", "get",
    "get_mut", "contains", "contains_key", "clear", "reserve", "iter",
    "iter_mut", "into_iter", "keys", "values", "values_mut", "entry",
}

STD_TYPE_MEMBERS: Dict[str, Set[str]] = {
    "Vec": VEC_METHODS,
    "VecDeque": VEC_METHODS,
    "Box": SMART_PTR_METHODS,
    "Rc": SMART_PTR_METHODS,
    "Arc": SMART_PTR_METHODS,
    "Weak": SMART_PTR_METHODS,
    "RcWeak": SMART_PTR_METHODS,
    "ArcWeak": SMART_PTR_METHODS,
    "Option": {
        "Some", "None", "is_some", "is_none", "is_some_and", "as_ref", "as_mut",
        "unwrap", "expect", "unwrap_or", "unwrap_or_else", "unwrap_or_default",
        "map", "and_then", "or_else", "ok_or", "ok_or_else", "take", "replace",
        "flatten", "cloned", "copied",
    },
    "Result": {
        "Ok", "Err", "is_ok", "is_err", "as_ref", "unwrap", "expect",
        "unwrap_or", "unwrap_or_else", "unwrap_or_default", "map", "map_err",
        "and_then", "or_else", "ok", "err",
    },
    "String": {
        "new", "with_capacity", "len", "is_empty", "capacity", "push",
        "push_str", "clear", "from_utf8", "from_utf8_lossy",
        "from_utf8_unchecked", "as_bytes", "as_str", "into_bytes",
    },
    "Cell": {"new", "get", "set", "replace", "take", "swap", "into_inner", "get_mut"},
    "RefCell": {
        "new", "borrow", "borrow_mut", "try_borrow", "try_borrow_mut",
        "replace", "take", "into_inner", "get_mut",
    },
    "UnsafeCell": {"new", "get", "into_inner", "raw_get"},
    "OnceLock": {"new", "get", "set", "get_or_init", "take", "into_inner"},
    "LazyLock": {"new", "force"},
    "Mutex": {"new", "lock", "try_lock", "into_inner", "get_mut"},
    "RwLock": {"new", "read", "write", "try_read", "try_write", "into_inner", "get_mut"},
    "Condvar": {"new", "wait", "wait_while", "wait_timeout", "notify_one", "notify_all"},
    "Semaphore": {"new", "acquire", "try_acquire", "add_permits", "available_permits", "close"},
    "HashMap": MAP_METHODS,
    "BTreeMap": MAP_METHODS,
    "HashSet": MAP_METHODS,
    "BTreeSet": MAP_METHODS,
    "NonNull": {
        "new", "new_unchecked", "dangling", "as_ptr", "as_ref", "as_mut",
        "cast", "add", "sub", "read", "write", "copy_from_nonoverlapping",
        "slice_from_raw_parts",
    },
    "Layout": {
        "new", "from_size_align", "from_size_align_unchecked", "array",
        "size", "align", "pad_to_align", "extend",
    },
    "Ordering": {"Relaxed", "Acquire", "Release", "AcqRel", "SeqCst", "Less", "Equal", "Greater"},
    "SeekFrom": {"Start", "Current", "End"},
    "ErrorKind": {
        "UnexpectedEof", "InvalidData", "InvalidInput", "Other", "WouldBlock",
        "Interrupted", "NotFound", "PermissionDenied", "AlreadyExists",
        "WriteZero", "TimedOut",
    },
    "Advice": {"Normal", "Random", "Sequential", "WillNeed", "DontNeed"},
    "MmapMut": {
        "map_mut", "map_anon", "len", "is_empty", "as_ptr", "as_mut_ptr",
        "flush", "flush_async", "flush_range", "advise", "lock", "unlock",
    },
    "Mmap": {
        "map", "len", "is_empty", "as_ptr", "advise", "lock", "unlock",
    },
    "MmapOptions": {
        "new", "map", "map_mut", "map_anon", "map_copy", "map_copy_read_only",
        "len", "offset", "populate", "stack", "huge",
    },
    "OpenOptions": {
        "new", "read", "write", "append", "truncate", "create", "create_new",
        "custom_flags", "mode", "open",
    },
    "File": {
        "open", "create", "create_new", "options", "set_len", "sync_all",
        "sync_data", "metadata", "try_clone",
    },
    "Path": {
        "new", "join", "parent", "file_name", "extension", "exists",
        "is_file", "is_dir", "is_symlink", "symlink_metadata", "metadata",
        "to_path_buf", "as_path", "display",
    },
    "PathBuf": {
        "new", "from", "push", "pop", "join", "parent", "file_name",
        "extension", "exists", "is_file", "is_dir", "is_symlink",
        "symlink_metadata", "metadata", "as_path", "display",
    },
    "Instant": {
        "now", "elapsed", "duration_since", "checked_duration_since",
        "saturating_duration_since",
    },
    "Duration": {
        "new", "from_secs", "from_millis", "from_micros", "from_nanos",
        "as_secs", "as_millis", "as_micros", "as_nanos", "as_secs_f64",
        "ZERO", "MAX",
    },
    "Cursor": {"new", "get_ref", "get_mut", "into_inner", "position", "set_position"},
    "Read": {"read", "read_exact", "read_to_end", "read_to_string", "read_vectored"},
    "Write": {"write", "write_all", "write_fmt", "write_vectored", "flush"},
    "Seek": {"seek", "stream_position", "rewind"},
    "BufRead": {"fill_buf", "consume", "read_line", "read_until"},
    "AtomicBool": ATOMIC_METHODS,
    "AtomicU8": ATOMIC_METHODS,
    "AtomicU16": ATOMIC_METHODS,
    "AtomicU32": ATOMIC_METHODS,
    "AtomicU64": ATOMIC_METHODS,
    "AtomicUsize": ATOMIC_METHODS,
    "AtomicI32": ATOMIC_METHODS,
    "AtomicI64": ATOMIC_METHODS,
    "AtomicIsize": ATOMIC_METHODS,
    "AtomicPtr": ATOMIC_METHODS,
    "PhantomData": set(),
    "Error": {"new", "other", "last_os_error", "from_raw_os_error", "kind", "raw_os_error"},
    "RangeInclusive": {"new", "start", "end", "is_empty", "contains"},
    "Local": {"with", "try_with", "register"},
    "Global": {"alloc", "dealloc", "alloc_zeroed", "realloc"},
    "Builder": {
        "new", "new_current_thread", "new_multi_thread", "worker_threads",
        "enable_all", "name", "stack_size", "build", "spawn",
    },
    "Handle": {"current", "try_current", "spawn", "block_on", "enter", "id"},
    "Notify": {"new", "notified", "notify_one", "notify_waiters"},
    "JoinHandle": {"abort", "is_finished", "join", "thread", "id"},
    "Thread": {"id", "name", "unpark"},
    "TypeId": {"of"},
    "AnyBitPattern": set(),
    "NoUninit": set(),
    "Self": set(),
    "T": set(),
    "Ok": set(),
    "Err": set(),
    "Some": set(),
    "None": set(),
}

STD_TYPES: Set[str] = set(STD_TYPE_MEMBERS.keys())

PROSE_CAMEL_ALLOW: Set[str] = {
    "ThreadSanitizer", "AddressSanitizer", "MemorySanitizer", "LeakSanitizer",
    "UndefinedBehaviorSanitizer", "TSan", "ASan", "MSan", "LSan", "UBSan",
    "GitHub", "KiB", "MiB", "GiB", "TiB", "NaN", "DoS", "CoW", "Melem", "Gelem",
    "MacBook", "Intel", "Linux", "Windows", "FreeBSD", "NetBSD", "OpenBSD",
}

CODE_VERBS: Set[str] = {
    "get", "set", "try", "is", "has", "new", "from", "into", "with", "spawn",
    "park", "defer", "clone", "read", "write", "push", "pop", "fetch", "load",
    "store", "lock", "drop", "run", "build", "init", "flush", "clear", "ensure",
    "check", "find", "steal", "wake", "notify", "update", "insert", "remove",
    "delete", "alloc", "dealloc", "encode", "decode", "serialize", "deserialize",
    "test", "bench", "lookup", "apply", "evict", "mark", "unmark", "cas", "atomic",
}

HYPHEN_PROSE_PARTS: Set[str] = {
    "of", "on", "to", "in", "by", "for", "per", "or", "and", "vs", "no", "non",
    "re", "pre", "post", "sub", "co", "de", "un", "end", "out", "up", "low",
    "high", "one", "two", "three", "four", "five", "six", "eight", "byte", "bit",
    "word", "line", "step", "hand", "side", "head", "tail", "wall", "real", "time",
    "safe", "free", "only", "based", "aware", "local", "level", "wise", "like",
    "first", "last", "next", "full", "half", "zero", "single", "multi", "cross",
    "long", "short", "fast", "slow", "cold", "hot", "hard", "soft", "weak",
    "strong", "read", "write", "copy", "lock", "wait", "drop", "pass", "fail",
}

INDEXED_SCRIPTS: Set[str] = {
    "scripts/unsafe_audit.py",
    "scripts/api_honesty.py",
    "scripts/index_audit.py",
    "scripts/unwrap_audit.py",
    "scripts/audit_common.py",
    "scripts/check_gate_drift.py",
    "scripts/gate_baseline.txt",
}


def should_index_path(path: str) -> bool:
    if path.startswith(("docs/", "scripts/tests/")):
        return False
    if path in {"scripts/audit_ledger.py", "GEMINI.md", "CLAUDE.md", "plan.md"}:
        return False
    if path.endswith((".rs", ".toml", ".sh", ".yml", ".yaml", ".supp")) or path in {
        "Makefile",
        ".gitignore",
    }:
        return True
    if path in INDEXED_SCRIPTS:
        return True
    return False


def git_run(args: List[str]) -> Tuple[int, str]:
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.returncode, proc.stdout


def count_non_test_unsafe_sites(filepath: str, text: str) -> int:
    if is_test_file(filepath):
        return 0
    lines = text.splitlines()
    in_block_comment = 0
    brace_stack: List[bool] = []
    pending_test = False
    count = 0
    for line in lines:
        raw_stripped = line.strip()
        if raw_stripped.startswith("//") or not raw_stripped:
            continue
        i = 0
        n = len(line)
        clean_chars: List[str] = []
        while i < n:
            if in_block_comment > 0:
                if line[i : i + 2] == "*/":
                    in_block_comment -= 1
                    i += 2
                elif line[i : i + 2] == "/*":
                    in_block_comment += 1
                    i += 2
                else:
                    i += 1
                continue
            if line[i : i + 2] == "/*":
                in_block_comment += 1
                i += 2
                continue
            if line[i : i + 2] == "//":
                break
            if line[i] == '"':
                clean_chars.append('"')
                i += 1
                while i < n:
                    if line[i] == "\\":
                        i += 2
                    elif line[i] == '"':
                        clean_chars.append('"')
                        i += 1
                        break
                    else:
                        i += 1
                continue
            if line[i] == "'":
                if i + 2 < n and line[i + 2] == "'":
                    i += 3
                    continue
                if i + 3 < n and line[i + 1] == "\\" and line[i + 3] == "'":
                    i += 4
                    continue
            clean_chars.append(line[i])
            i += 1
        clean_line = "".join(clean_chars).strip()
        if "#![cfg(test)]" in clean_line:
            pending_test = True
            brace_stack = [True]
        if is_test_only_cfg(clean_line) or "#[test]" in clean_line or "#[bench]" in clean_line:
            pending_test = True
        if re.search(r"\bmod\s+(?:tests?|testing|bench(?:marks?)?)\b", clean_line):
            pending_test = True
        is_curr_test = pending_test or (len(brace_stack) > 0 and brace_stack[-1])
        for ch in clean_line:
            if ch == "{":
                brace_stack.append(is_curr_test)
                pending_test = False
            elif ch == "}":
                if brace_stack:
                    brace_stack.pop()
        if clean_line.endswith(";") and "{" not in clean_line:
            pending_test = False
        in_test = (len(brace_stack) > 0 and brace_stack[-1]) or pending_test
        if not in_test and UNSAFE_SITE_RE.search(clean_line):
            count += 1
    return count


def count_tests_in_rs(txt: str, is_miri: bool) -> int:
    if is_miri:
        txt = re.sub(
            r"#\[cfg\([^\]]*not\(miri\)[^\]]*\)\]\s*mod\s+\w+\s*\{.*\}",
            "",
            txt,
            flags=re.S,
        )
    lines = txt.splitlines()
    cnt = 0
    for i, l in enumerate(lines):
        if re.match(r"^\s*#\[(?:tokio::)?test\b", l):
            lo = max(0, i - 6)
            hi = min(len(lines), i + 8)
            window = "\n".join(lines[lo:hi])
            after = "\n".join(lines[i:hi])
            m_fn = re.search(r"\bfn\s+\w+", after)
            if m_fn:
                window = "\n".join(lines[lo:i]) + "\n" + after[: m_fn.end()]
            if re.search(r"#\[ignore\b", window) or re.search(
                r'not\(feature\s*=\s*"mmap"\)', window
            ):
                continue
            if is_miri and (
                re.search(r"not\(miri\)", window)
                or re.search(r"cfg_attr\(miri,\s*ignore", window)
            ):
                continue
            cnt += 1
    return cnt


class RepoIndex:
    def __init__(self) -> None:
        self.all_words: Set[str] = set(STD_TYPES)
        self.ast_members: Dict[str, Set[str]] = collections.defaultdict(set)
        self.file_words: Dict[str, Set[str]] = collections.defaultdict(set)
        self.file_max_lines: Dict[str, int] = collections.defaultdict(int)
        self.tracked_paths: Set[str] = set()
        self.hyphen_tokens: Set[str] = set()
        self.test_fn_words: Dict[Tuple[str, str], Set[str]] = collections.defaultdict(set)
        self.fn_body_words: Dict[str, Set[str]] = collections.defaultdict(set)
        self.field_orderings: Dict[str, Set[str]] = collections.defaultdict(set)
        self.valid_module_unsafe: Dict[str, Set[int]] = collections.defaultdict(set)
        self.valid_gate_totals: Set[int] = set()
        self.valid_debug_totals: Set[int] = set()
        self.makefile_targets: Set[str] = set()
        self.target_valid_counts: Dict[str, Set[int]] = collections.defaultdict(set)
        self.commit_msg_words: Dict[str, Set[str]] = {}
        self.commit_files: Dict[str, Set[str]] = {}
        self.commit_file_lines: Dict[str, Dict[str, int]] = {}


def build_historical_index() -> RepoIndex:
    """Index ONLY committed git objects across db3c514^..HEAD (never working-tree or staged files)."""
    idx = RepoIndex()
    rc, out = git_run(["rev-list", "db3c514^..HEAD"])
    commits = out.split() if rc == 0 else []
    rc_base, base_sha = git_run(["rev-parse", "db3c514^"])
    if rc_base == 0 and base_sha.strip():
        commits.append(base_sha.strip())
    if not commits:
        rc_h, head_sha = git_run(["rev-parse", "HEAD"])
        if rc_h == 0 and head_sha.strip():
            commits.append(head_sha.strip())

    rc_log, log_out = git_run(["log", "--format=%H %h %s", "db3c514^..HEAD"])
    if rc_log == 0:
        for m_tag in re.findall(r"\b(?:[BCDS]\d+|R\d+[a-z]?)\b", log_out):
            idx.all_words.add(m_tag)
    for d_idx in range(1, 12):
        idx.all_words.add(f"D{d_idx}")

    blobs: Dict[str, Set[str]] = collections.defaultdict(set)
    commit_src_trees: Dict[str, List[Tuple[str, str]]] = {}

    for c in commits:
        rc_ls, ls_out = git_run(["ls-tree", "-r", c])
        if rc_ls != 0:
            continue
        src_tree: List[Tuple[str, str]] = []
        for line in ls_out.splitlines():
            if "\t" not in line:
                continue
            meta, path = line.split("\t", 1)
            idx.tracked_paths.add(path)
            bsha = meta.split()[2]
            if path.startswith("src/") and path.endswith(".rs"):
                src_tree.append((path, bsha))
            if should_index_path(path):
                blobs[bsha].add(path)
        commit_src_trees[c] = src_tree

    rs_blobs: List[Tuple[Set[str], str]] = []
    blob_unsafe_counts: Dict[Tuple[str, str], int] = {}
    blob_texts: Dict[str, str] = {}
    blob_line_counts: Dict[str, int] = {}

    if blobs:
        blob_keys = list(blobs.keys())
        proc = subprocess.Popen(
            ["git", "-C", str(REPO_ROOT), "cat-file", "--batch"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        batch_out, _ = proc.communicate(("\n".join(blob_keys) + "\n").encode())
        k = 0
        while k < len(batch_out):
            nl = batch_out.index(b"\n", k)
            hdr = batch_out[k:nl].split()
            bsha = hdr[0].decode("ascii", "replace")
            size = int(hdr[2])
            txt = batch_out[nl + 1 : nl + 1 + size].decode("utf-8", "replace")
            blob_texts[bsha] = txt
            toks = set(IDENT_RE.findall(txt))
            idx.all_words.update(toks)
            idx.hyphen_tokens.update(HYPHEN_IDENT_RE.findall(txt.lower()))
            nlines = txt.count("\n") + 1
            blob_line_counts[bsha] = nlines
            paths_for_blob = blobs.get(bsha, set())
            rs_paths = set()
            for p in paths_for_blob:
                path_toks = set(IDENT_RE.findall(p))
                idx.all_words.update(path_toks)
                idx.file_words[p].update(toks | path_toks)
                if nlines > idx.file_max_lines[p]:
                    idx.file_max_lines[p] = nlines
                if p.endswith(".rs"):
                    rs_paths.add(p)
                    if p.startswith("src/"):
                        blob_unsafe_counts[(p, bsha)] = count_non_test_unsafe_sites(p, txt)
                elif p == "scripts/gate_baseline.txt":
                    for m_kv in re.finditer(r"^(\w+)\s*=\s*(\d+)", txt, re.M):
                        key, val_s = m_kv.groups()
                        val = int(val_s)
                        idx.valid_gate_totals.add(val)
                        if key == "lib_debug":
                            idx.valid_debug_totals.add(val)
            if rs_paths:
                rs_blobs.append((rs_paths, txt))
            k = nl + 1 + size + 1

    for c, src_tree in commit_src_trees.items():
        idx.commit_file_lines[c] = {
            p: blob_line_counts.get(bsha, 0) for p, bsha in src_tree
        }

    # Compute dynamic per-module unsafe counts across all commits in db3c514^..HEAD
    for c, src_tree in commit_src_trees.items():
        per_mod: Dict[str, int] = collections.Counter()
        for p, bsha in src_tree:
            parts = p.split("/")
            if len(parts) >= 3:
                mod_key = f"src/{parts[1]}/"
                per_mod[mod_key] += blob_unsafe_counts.get((p, bsha), 0)
        for mk, val in per_mod.items():
            idx.valid_module_unsafe[mk].add(val)

    # Compute dynamic Makefile targets and per-target Miri/TSan test counts across history
    rc_mk, mk_head = git_run(["show", "HEAD:Makefile"])
    if rc_mk == 0:
        for m_tgt in re.finditer(r"^([A-Za-z_][\w-]*):", mk_head, re.M):
            idx.makefile_targets.add(m_tgt.group(1))
        mk_joined = mk_head.replace("\\\n", " ")
        target_filters: Dict[str, Tuple[bool, List[str]]] = {}
        for m_t in re.finditer(r"^((?:miri|tsan)_\w+):\n((?:\t.*\n)+)", mk_joined, re.M):
            tname, body = m_t.groups()
            is_miri = tname.startswith("miri_")
            flts = re.findall(r"test\b[^\n]*?--lib\s+([A-Za-z0-9_:]+)", body)
            if flts:
                target_filters[tname] = (is_miri, flts)

        # Evaluate target test counts at key snapshots (db3c514^, Stage 10 end, HEAD)
        sample_commits = []
        if commits:
            sample_commits.extend([commits[0], commits[-1]])
            for c in commits[::8]:
                if c not in sample_commits:
                    sample_commits.append(c)
        blob_test_cache: Dict[Tuple[str, bool], int] = {}
        for c in sample_commits:
            src_tree = commit_src_trees.get(c, [])
            for tname, (is_miri, flts) in target_filters.items():
                sub_counts = []
                for flt in flts:
                    mod_prefix = "src/" + flt.rstrip(":").replace("::", "/")
                    cnt = 0
                    for p, bsha in src_tree:
                        if p.startswith(mod_prefix) and bsha in blob_texts:
                            ck = (bsha, is_miri)
                            if ck not in blob_test_cache:
                                blob_test_cache[ck] = count_tests_in_rs(
                                    blob_texts[bsha], is_miri
                                )
                            cnt += blob_test_cache[ck]
                    idx.target_valid_counts[tname].add(cnt)
                    if is_miri and cnt > 50:
                        idx.target_valid_counts[tname].add(cnt - 1)
                    sub_counts.append(cnt)
                if sub_counts:
                    idx.target_valid_counts[tname].add(sum(sub_counts))

    # Extract AST Type::member pairs, function bodies, test function bodies, and atomic field orderings
    atomic_op_re = re.compile(
        r"\b([a-z_][a-z0-9_]*)\s*\.\s*(?:load|store|swap|fetch_\w+|compare_exchange(?:_weak)?)\s*\([^;]*?\bOrdering::(Relaxed|Acquire|Release|AcqRel|SeqCst)"
    )
    for rs_paths, t in rs_blobs:
        for m_aop in atomic_op_re.finditer(t):
            idx.field_orderings[m_aop.group(1)].add(m_aop.group(2))
        lines = t.split("\n")
        for i, l in enumerate(lines):
            # Index function bodies and test function bodies
            m_fn = ITEM_RE.match(l)
            if m_fn and m_fn.group(1) == "fn":
                fn_name = m_fn.group(2)
                attr_win = "\n".join(lines[max(0, i - 5) : i + 1])
                is_test_fn = fn_name.startswith("test_") or bool(
                    re.search(r"#\[(?:tokio::)?test\b", attr_win)
                )
                depth = 0
                started = False
                j = i
                while j < len(lines):
                    for ch in lines[j]:
                        if ch == "{":
                            depth += 1
                            started = True
                        elif ch == "}":
                            depth -= 1
                    if started and depth <= 0:
                        break
                    if not started and lines[j].rstrip().endswith(";"):
                        break
                    j += 1
                body_txt = "\n".join(lines[i : j + 1])
                body_toks = set(IDENT_RE.findall(body_txt))
                idx.fn_body_words[fn_name].update(body_toks)
                if is_test_fn:
                    for rp in rs_paths:
                        idx.test_fn_words[(rp, fn_name)].update(body_toks)

            m = HDR_RE.match(l)
            if not m:
                continue
            h = (m.group(1) or m.group(2)).strip()
            if h.startswith("impl"):
                mm = re.search(r"\bfor\s+(?:[\w:]+::)?([A-Za-z_]\w*)", h)
                tr = None
                if mm:
                    own = mm.group(1)
                    tr = re.search(
                        r"impl\s*(?:<.*>)?\s*(?:[\w:]+::)?([A-Za-z_]\w*)\s*(?:<[^{]*?>)?\s+for\b",
                        h,
                    )
                else:
                    mm = re.search(r"impl\s*(?:<[^{]*?>)?\s*(?:[\w:]+::)?([A-Za-z_]\w*)", h)
                    if not mm:
                        continue
                    own = mm.group(1)
                names = {own}
                if tr:
                    names.add(tr.group(1))
            else:
                names = {re.split(r"[<:\s]", h.split()[1])[0]}

            depth = 0
            started = False
            j = i
            while j < len(lines):
                s = lines[j]
                depth_before = depth
                for ch in s:
                    if ch == "{":
                        depth += 1
                        started = True
                    elif ch == "}":
                        depth -= 1
                if j > i and started:
                    if h.startswith("enum") and depth_before == 1:
                        v = re.match(r"^\s*([A-Z]\w*)\s*[({,=]|^\s*([A-Z]\w*)\s*$", s)
                        if v:
                            for n in names:
                                idx.ast_members[n].add(v.group(1) or v.group(2))
                    if h.startswith("struct") and depth_before == 1:
                        v = re.match(r"^\s*(?:pub(?:\([^)]*\))?\s+)?([a-z_]\w*)\s*:", s)
                        if v:
                            for n in names:
                                idx.ast_members[n].add(v.group(1))
                    f = re.search(
                        r"\bfn\s+([A-Za-z_]\w*)|\b(?:const|type)\s+(?!fn\b)([A-Za-z_]\w*)",
                        s,
                    )
                    if f and depth <= 2:
                        for n in names:
                            idx.ast_members[n].add(f.group(1) or f.group(2))
                if started and depth <= 0:
                    break
                j += 1

    return idx


def audit_ledger(
    md_path: pathlib.Path,
    idx_data: RepoIndex,
) -> Tuple[int, int, int, int, List[str]]:
    all_words = idx_data.all_words
    ast_members = idx_data.ast_members
    file_words = idx_data.file_words
    file_max_lines = idx_data.file_max_lines
    tracked_paths = idx_data.tracked_paths

    text = md_path.read_text(encoding="utf-8")
    lines = text.split("\n")

    # Pre-scan benchmark tables (`| ... |`) and fenced code blocks (` ``` `) in the document
    # for backed decimal measurement numbers, and collect all commits cited in the document.
    backed_decimal_nums: Set[str] = set()
    scan_fence = False
    for l in lines:
        if l.lstrip().startswith("```"):
            scan_fence = not scan_fence
            continue
        if scan_fence or l.lstrip().startswith("|"):
            for m_dn in re.findall(r"\b\d+\.\d+\b", l):
                backed_decimal_nums.add(m_dn)

    sha_cache: Dict[str, Tuple[bool, bool]] = {}
    file_cache: Dict[Tuple[str, str], Optional[List[str]]] = {}
    span_cache: Dict[Tuple[str, str], List[Tuple[int, int, Set[str], str]]] = {}

    def check_sha(s: str) -> Tuple[bool, bool]:
        if s not in sha_cache:
            rc, _ = git_run(["rev-parse", "--verify", "-q", s + "^{commit}"])
            anc = rc == 0 and git_run(["merge-base", "--is-ancestor", s, "HEAD"])[0] == 0
            sha_cache[s] = (rc == 0, anc)
        return sha_cache[s]

    # Determine document commit horizon for fenced paste line bounds
    doc_shas = {
        s
        for s in re.findall(r"\b([0-9a-f]{7,40})\b", text)
        if check_sha(s)[0]
    }
    doc_max_lines: Dict[str, int] = dict(file_max_lines)
    if doc_shas and idx_data.commit_file_lines:
        horizon_commits: Set[str] = set()
        for ds in doc_shas:
            rc_rl, rl_out = git_run(["rev-list", f"db3c514^..{ds}"])
            if rc_rl == 0:
                horizon_commits.update(rl_out.split())
        if horizon_commits:
            scoped_max: Dict[str, int] = collections.defaultdict(int)
            for hc in horizon_commits:
                for p, nl in idx_data.commit_file_lines.get(hc, {}).items():
                    if nl > scoped_max[p]:
                        scoped_max[p] = nl
            for p, nl in scoped_max.items():
                doc_max_lines[p] = nl

    def get_commit_info(s: str) -> Tuple[Set[str], Set[str]]:
        if s not in idx_data.commit_msg_words:
            rc1, msg = git_run(["show", "-s", "--format=%B", s])
            rc2, fls = git_run(["show", "--name-only", "--format=", s])
            idx_data.commit_msg_words[s] = set(IDENT_RE.findall(msg)) if rc1 == 0 else set()
            idx_data.commit_files[s] = (
                {f.strip() for f in fls.splitlines() if f.strip()} if rc2 == 0 else set()
            )
        return idx_data.commit_msg_words[s], idx_data.commit_files[s]

    def file_lines(s: str, p: str) -> Optional[List[str]]:
        k = (s, p)
        if k not in file_cache:
            rc, out = git_run(["show", f"{s}:{p}"])
            if rc != 0:
                file_cache[k] = None
            else:
                fl = out.split("\n")
                if fl and fl[-1] == "":
                    fl.pop()
                file_cache[k] = fl
        return file_cache[k]

    def item_spans(s: str, p: str) -> List[Tuple[int, int, Set[str], str]]:
        k = (s, p)
        if k in span_cache:
            return span_cache[k]
        f = file_lines(s, p) or []
        out: List[Tuple[int, int, Set[str], str]] = []
        for i, l in enumerate(f):
            names: Optional[Set[str]] = None
            kind = ""
            m = ITEM_RE.match(l)
            if m:
                kind = m.group(1)
                names = {m.group(2)}
            else:
                m2 = IMPL_RE.match(l)
                if m2:
                    kind = "impl"
                    names = set(re.findall(r"[A-Za-z_]\w*", m2.group(1)))
                elif re.match(r"^[A-Za-z_][\w-]*:", l) and p.endswith("Makefile"):
                    kind = "target"
                    names = {l.split(":")[0]}
            if not names:
                continue
            if p.endswith("Makefile"):
                j = i + 1
                while j < len(f) and (
                    f[j].startswith("\t") or f[j].strip() == "" or f[j].startswith("#")
                ):
                    j += 1
                out.append((i + 1, j, names, kind))
                continue
            depth = 0
            started = False
            j = i
            while j < len(f):
                for ch in f[j]:
                    if ch == "{":
                        depth += 1
                        started = True
                    elif ch == "}":
                        depth -= 1
                if started and depth <= 0:
                    break
                if not started and f[j].rstrip().endswith(";"):
                    break
                j += 1
            out.append((i + 1, j + 1, names, kind))
        span_cache[k] = out
        return out

    def defs_of(name: str, s: str, p: str, non_impl_only: bool = False) -> List[Tuple[int, int]]:
        if non_impl_only:
            d = [
                (x, y)
                for (x, y, names, kind) in item_spans(s, p)
                if name in names and kind != "impl"
            ]
            if d:
                return d
        return [(x, y) for (x, y, names, _) in item_spans(s, p) if name in names]

    def fn_defs_of(name: str, s: str, p: str) -> List[Tuple[int, int]]:
        return [
            (x, y)
            for (x, y, names, kind) in item_spans(s, p)
            if name in names and kind == "fn"
        ]

    def prefix_ok(f: List[str], a: int, defline: int) -> bool:
        if not (a <= defline):
            return False
        for k in range(a, defline):
            t = f[k - 1].strip()
            if not (
                t == ""
                or t.startswith("///")
                or t.startswith("#[")
                or t.startswith("//")
                or t.startswith("#!")
            ):
                return False
        return defline - a <= 40

    def name_ok(name: str, s: str, p: str, a: int, b: int) -> Optional[str]:
        f = file_lines(s, p) or []
        lo = max(1, a - 2)
        hi = min(len(f), b + 2)
        win = "\n".join(f[lo - 1 : hi])
        if re.search(r"\b" + re.escape(name) + r"\b", win):
            return "win"
        for x, y, names, _ in item_spans(s, p):
            if x <= a and b <= y and name in names:
                return "encl"
        return None

    errors: List[str] = []
    commit_row_count = 0
    citation_count = 0
    sem_checked_count = 0
    pair_count = 0

    in_fence = False
    fence_lang = ""
    fence_start_idx = 0
    cites: List[dict] = []
    current_section_sha: Optional[str] = None

    def is_valid_camel(tok: str) -> bool:
        if tok in all_words or tok in PROSE_CAMEL_ALLOW:
            return True
        if tok.endswith("s") and len(tok) > 4 and tok[:-1] in all_words:
            return True
        return False

    def is_valid_snake(tok: str, in_backticks: bool, formatted_or_call: bool, sec_sha: Optional[str]) -> bool:
        if tok in all_words:
            return True
        if sec_sha and check_sha(sec_sha)[0]:
            msg_w, _ = get_commit_info(sec_sha)
            if tok in msg_w:
                return True
        if not in_backticks and not formatted_or_call and tok.count("_") == 1:
            p0, p1 = tok.split("_", 1)
            if (
                p0 in all_words
                and p1 in all_words
                and p0 not in CODE_VERBS
                and not any(ch.isdigit() for ch in tok)
            ):
                return True
        return False

    def check_pair_token(owner: str, mem: str, idx: int, line_str: str) -> None:
        nonlocal pair_count
        if owner in {"Type", "Self", "T"}:
            return
        pair_count += 1
        if owner in STD_TYPES:
            allowed_for_owner = STD_TYPE_MEMBERS.get(owner, set()) | UNIVERSAL_TRAIT_METHODS
            if mem not in allowed_for_owner and mem not in ast_members.get(owner, set()):
                errors.append(
                    f"{md_path}:{idx}: unresolved STD Type::member `{owner}::{mem}`"
                )
                return
            # If the line names a specific repository function alongside `Owner::method`, verify `method` is used in that function
            if (
                mem[0].islower()
                and mem not in UNIVERSAL_TRAIT_METHODS
                and not re.search(r"\b(?:not|never|instead|rather|without|Before)\b", line_str)
            ):
                fn_mentions = [
                    bt
                    for bt in BT_RE.findall(line_str)
                    if bt in idx_data.fn_body_words and "::" not in bt and bt != mem
                ]
                if len(fn_mentions) == 1:
                    fn_m = fn_mentions[0]
                    if mem not in idx_data.fn_body_words[fn_m]:
                        errors.append(
                            f"{md_path}:{idx}: `{owner}::{mem}` does not appear inside function `{fn_m}`"
                        )
            return
        if mem not in ast_members.get(owner, set()):
            errors.append(
                f"{md_path}:{idx}: unresolved Type::member `{owner}::{mem}` (not defined on `{owner}` in any committed `.rs` file)"
            )

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            m_hsha = re.search(r"`([0-9a-f]{7,40})`", stripped)
            if m_hsha and check_sha(m_hsha.group(1))[0]:
                current_section_sha = m_hsha.group(1)

        if line.lstrip().startswith("```"):
            if not in_fence:
                in_fence = True
                fence_lang = line.lstrip()[3:].strip().lower()
                fence_start_idx = idx
            else:
                in_fence = False
                fence_lang = ""
            continue
        if in_fence:
            if fence_lang in {"rust", "rs"}:
                for m_pair in PAIR_RE.finditer(line):
                    owner = m_pair.group(1)
                    mems = (
                        [x.strip() for x in m_pair.group(2).split(",")]
                        if m_pair.group(2)
                        else [m_pair.group(3)]
                    )
                    for mem in mems:
                        mem = re.sub(r"\W.*", "", mem)
                        if mem:
                            check_pair_token(owner, mem, idx, line)
                for tok in SNAKE_RE.findall(line):
                    if tok not in all_words:
                        errors.append(
                            f"{md_path}:{idx}: identifier `{tok}` in rust code fence absent from repo"
                        )
                for tok in CAMEL_RE.findall(line):
                    if not is_valid_camel(tok):
                        errors.append(
                            f"{md_path}:{idx}: identifier `{tok}` in rust code fence absent from repo"
                        )
            for m_al in ARROW_LOC_RE.finditer(line):
                fpath, fline_s, _ = m_al.groups()
                if fpath.startswith("/"):
                    continue
                fline = int(fline_s)
                max_l = doc_max_lines.get(fpath, 0)
                if max_l == 0:
                    errors.append(
                        f"{md_path}:{idx}: fenced paste references unknown file `{fpath}:{fline}`"
                    )
                elif fline < 1 or fline > max_l:
                    errors.append(
                        f"{md_path}:{idx}: fenced paste line `{fpath}:{fline}` out of bounds (max historical lines {max_l})"
                    )
            for m_pl in PANIC_LOC_RE.finditer(line):
                th_name, fpath, fline_s, _ = m_pl.groups()
                if fpath.startswith("/"):
                    continue
                fline = int(fline_s)
                max_l = doc_max_lines.get(fpath, 0)
                pre_fence = "\n".join(lines[max(0, fence_start_idx - 5) : fence_start_idx]).lower()
                slack = 25 if "spliced" in pre_fence else 0
                if max_l == 0:
                    errors.append(
                        f"{md_path}:{idx}: fenced paste references unknown file `{fpath}:{fline}`"
                    )
                elif fline < 1 or fline > max_l + slack:
                    errors.append(
                        f"{md_path}:{idx}: fenced panic line `{fpath}:{fline}` out of bounds (max historical lines {max_l})"
                    )
                test_fn_name: Optional[str] = None
                if "::" in th_name:
                    clean_th = th_name.split("::{closure")[0]
                    test_fn_name = clean_th.split("::")[-1]
                    if (fpath, test_fn_name) not in idx_data.test_fn_words:
                        errors.append(
                            f"{md_path}:{idx}: panic thread test `{th_name}` does not exist in `{fpath}`"
                        )
                if idx < len(lines) and fpath in file_words:
                    msg_line = lines[idx].strip()
                    if msg_line.startswith("assertion"):
                        f_toks = (
                            idx_data.test_fn_words[(fpath, test_fn_name)]
                            if test_fn_name and (fpath, test_fn_name) in idx_data.test_fn_words
                            else file_words[fpath]
                        )
                        msg_words = [
                            w
                            for w in IDENT_RE.findall(msg_line)
                            if len(w) >= 4
                            and w
                            not in {
                                "assertion", "left", "right", "failed", "panicked",
                                "thread", "Some", "None", "true", "false",
                            }
                        ]
                        missing = [
                            w
                            for w in msg_words
                            if w not in f_toks and (not test_fn_name or w not in test_fn_name)
                        ]
                        if missing:
                            errors.append(
                                f"{md_path}:{idx + 1}: panic assertion token(s) {missing} absent from `{test_fn_name or fpath}`"
                            )
            continue

        # 1. Commit table row verification
        m_row = COMMIT_ROW_RE.match(stripped)
        if m_row:
            commit_row_count += 1
            commit, parent, date, subject, fcnt_s, ins_s, dels_s = m_row.groups()
            rc_m, meta = git_run(["show", "-s", "--format=%p|%as|%s", commit])
            if rc_m != 0 or not meta.strip():
                errors.append(f"{md_path}:{idx}: commit `{commit}` in table does not exist")
                continue
            actual_parent, actual_date, actual_subject = meta.strip().split("|", 2)
            if parent and not actual_parent.startswith(parent):
                errors.append(
                    f"{md_path}:{idx}: commit `{commit}` parent mismatch: ledger `{parent}` vs git `{actual_parent}`"
                )
            if actual_date != date:
                errors.append(
                    f"{md_path}:{idx}: commit `{commit}` date mismatch: ledger `{date}` vs git `{actual_date}`"
                )
            if actual_subject != subject:
                errors.append(
                    f"{md_path}:{idx}: commit `{commit}` subject mismatch: ledger `{subject}` vs git `{actual_subject}`"
                )
            rc_ns, ns_out = git_run(["show", "--numstat", "--format=", commit])
            if rc_ns == 0:
                stats = [l.split("\t") for l in ns_out.strip().splitlines() if l.strip()]
                a_files = len(stats)
                a_ins = sum(int(x[0]) for x in stats if x[0] != "-")
                a_del = sum(int(x[1]) for x in stats if x[1] != "-")
                c_files = int(fcnt_s)
                c_ins = int(ins_s.replace("\u2212", "-").lstrip("+"))
                c_del = abs(int(dels_s.replace("\u2212", "-")))
                if a_files != c_files or a_ins != c_ins or a_del != c_del:
                    errors.append(
                        f"{md_path}:{idx}: commit `{commit}` numstat mismatch: "
                        f"ledger ({c_files} files, +{c_ins}/-{c_del}) vs git ({a_files} files, +{a_ins}/-{a_del})"
                    )
            continue

        # 2. Extract citations on this line
        line_cites: List[dict] = []
        multi_spans: List[Tuple[int, int]] = []
        for m_mc in MULTI_CITE_RE.finditer(line):
            multi_spans.append((m_mc.start(), m_mc.end()))
            sha, path, spec = m_mc.groups()
            for part in spec.split(","):
                part = part.strip()
                if "-" in part:
                    sa, sb = part.split("-", 1)
                    a, b = int(sa), int(sb)
                else:
                    a = b = int(part)
                c_obj = dict(
                    line=idx, sha=sha, path=path, a=a, b=b,
                    start=m_mc.start(), end=m_mc.end(),
                )
                cites.append(c_obj)
                line_cites.append(c_obj)
        for m_c in CITE_RE.finditer(line):
            if any(ms <= m_c.start() and m_c.end() <= me for ms, me in multi_spans):
                continue
            c_obj = dict(
                line=idx,
                sha=m_c.group(1),
                path=m_c.group(2),
                a=int(m_c.group(3)),
                b=int(m_c.group(4) or m_c.group(3)),
                start=m_c.start(),
                end=m_c.end(),
            )
            cites.append(c_obj)
            line_cites.append(c_obj)

        # Find any standalone commit SHA mentioned on this line
        line_shas = [
            s
            for s in re.findall(r"`([0-9a-f]{7,40})`", line)
            if check_sha(s)[0]
        ]
        active_sha = line_shas[0] if line_shas else current_section_sha

        # 3. Check backticked identifiers and file paths
        bt_spans: List[str] = []
        for m_bt in BT_RE.finditer(line):
            raw_span = m_bt.group(1)
            bt_spans.append(raw_span)
            if re.fullmatch(
                r"(?:src|benches|tests|scripts|docs)/[A-Za-z0-9_./\-]+\.[A-Za-z0-9]+",
                raw_span,
            ):
                if raw_span not in tracked_paths and "external crate" not in line.lower():
                    errors.append(
                        f"{md_path}:{idx}: file path `{raw_span}` does not exist in repository"
                    )
                continue
            t_stripped = re.sub(r"[0-9a-f]{7,40}:[\w./-]+:[\d,\s\-]+", "", raw_span)
            for tp in tracked_paths:
                if tp in t_stripped:
                    t_stripped = t_stripped.replace(tp, " ")
            t_stripped = re.sub(r"\b[0-9a-f]{7,40}\b", "", t_stripped)
            for tok in IDENT_RE.findall(t_stripped):
                if re.fullmatch(r"[0-9a-f]+", tok) and len(tok) >= 7:
                    continue
                if not is_valid_snake(tok, True, True, active_sha) and not is_valid_camel(tok):
                    if (
                        tok.startswith("test_")
                        and ("::tests::" in raw_span or tok.count("_") <= 2)
                        and any(w.startswith(tok + "_") for w in all_words)
                    ):
                        continue
                    errors.append(
                        f"{md_path}:{idx}: identifier `{tok}` (in `{raw_span}`) absent from all committed repo files"
                    )

        # 3b. Check unbackticked / italic / escaped-underscore / bold / hyphenated identifiers in prose
        line_no_bt = BT_RE.sub(" ", line)
        line_no_bt = re.sub(r"https?://\S+", " ", line_no_bt)
        line_no_bt = re.sub(r"\b[A-Za-z0-9_./\-]+\.(?:md|rs|py|toml|sh|supp|txt|json)\b", " ", line_no_bt)
        for tp in tracked_paths:
            if tp in line_no_bt:
                line_no_bt = line_no_bt.replace(tp, " ")

        # Normalize escaped underscores `\_` and surrounding markdown `_ident_`
        line_norm = line_no_bt.replace(r"\_", "_")
        line_norm = re.sub(r"(?<!\w)_([a-z][a-z0-9]*(?:_[a-z0-9]+)+)_(?!\w)", r" \1 ", line_norm)

        for m_sn in SNAKE_RE.finditer(line_norm):
            tok = m_sn.group(0)
            formatted = (
                r"\_" in line_no_bt
                or f"_{tok}_" in line_no_bt
                or f"**{tok}**" in line_no_bt
                or f"(see {tok})" in line_no_bt
            )
            if not is_valid_snake(tok, False, formatted, active_sha):
                errors.append(
                    f"{md_path}:{idx}: prose snake_case identifier `{tok}` absent from all committed repo files"
                )

        for tok in CAMEL_RE.findall(line_norm):
            if not is_valid_camel(tok):
                errors.append(
                    f"{md_path}:{idx}: prose CamelCase identifier `{tok}` absent from all committed repo files"
                )

        for h_tok in HYPHEN_IDENT_RE.findall(line_no_bt.lower()):
            if h_tok in idx_data.hyphen_tokens or h_tok.replace("-", "_") in all_words:
                continue
            parts = h_tok.split("-")
            if not any(p in HYPHEN_PROSE_PARTS for p in parts):
                errors.append(
                    f"{md_path}:{idx}: hyphenated identifier `{h_tok}` absent from all committed repo files"
                )

        # 4. Check AST Type::member pairs across the ENTIRE line
        for m_pair in PAIR_RE.finditer(line):
            owner = m_pair.group(1)
            mems = (
                [x.strip() for x in m_pair.group(2).split(",")]
                if m_pair.group(2)
                else [m_pair.group(3)]
            )
            for mem in mems:
                mem = re.sub(r"\W.*", "", mem)
                if not mem:
                    continue
                check_pair_token(owner, mem, idx, line)

        # 5. Check `make <target>`, total test counts, module unsafe counts, percentages, and decimal measurements
        m_make = re.search(r"make\s+((?:miri|tsan)_\w+)", line)
        if m_make:
            tname = m_make.group(1)
            if tname not in idx_data.makefile_targets:
                errors.append(
                    f"{md_path}:{idx}: `make {tname}` is not a defined target in Makefile"
                )
            else:
                valid_c = idx_data.target_valid_counts.get(tname, set())
                for m_p in re.finditer(r"\b(\d+)\s+passed\b", line):
                    cnt = int(m_p.group(1))
                    if not valid_c or cnt not in valid_c:
                        errors.append(
                            f"{md_path}:{idx}: `make {tname}` test count `{cnt} passed` not in valid repository counts {sorted(valid_c)}"
                        )

        for m_tp in re.finditer(r"\b([\d,]+)\s+tests?\s+pass\b", line):
            n_val = int(m_tp.group(1).replace(",", ""))
            allowed_totals = (
                idx_data.valid_debug_totals
                if "debug" in line.lower()
                else idx_data.valid_gate_totals
            )
            if allowed_totals and n_val not in allowed_totals:
                errors.append(
                    f"{md_path}:{idx}: test count `{m_tp.group(1)} tests pass` not in gate_baseline.txt counts {sorted(allowed_totals)}"
                )

        for m_uns in re.finditer(
            r"`(src/[a-z_]+/)`\s+`?unsafe`?:\s*`(?:\d+\s*->\s*)?(\d+)`",
            line,
        ):
            mod_p, u_cnt_s = m_uns.group(1), m_uns.group(2)
            u_cnt = int(u_cnt_s)
            valid_u = idx_data.valid_module_unsafe.get(mod_p, set())
            if valid_u and u_cnt not in valid_u:
                errors.append(
                    f"{md_path}:{idx}: `{mod_p}` unsafe count `{u_cnt}` not in historical/current valid counts {sorted(valid_u)}"
                )

        # Check percentage arithmetic `±P% (A unit vs B unit)`
        pct_Verified_spans: List[Tuple[int, int]] = []
        for m_pct in re.finditer(
            r"([+-]?\d+(?:\.\d+)?)%\s*`?\s*\([^()]*?\b(\d+(?:\.\d+)?)\s*(?:ns|µs|us|ms|s|B|KiB|MiB|GiB)\b[^()]*?\bvs\b[^()]*?\b(\d+(?:\.\d+)?)\s*(?:ns|µs|us|ms|s|B|KiB|MiB|GiB)\b[^()]*\)",
            line,
        ):
            pct_val = abs(float(m_pct.group(1)))
            v1 = float(m_pct.group(2))
            v2 = float(m_pct.group(3))
            if v1 > 0 and v2 > 0:
                rel1 = abs(v1 - v2) / v2 * 100.0
                rel2 = abs(v1 - v2) / v1 * 100.0
                if min(abs(rel1 - pct_val), abs(rel2 - pct_val)) > 1.0:
                    errors.append(
                        f"{md_path}:{idx}: percentage `{m_pct.group(1)}%` inconsistent with `{v1}` vs `{v2}` (expected {rel1:.1f}%)"
                    )
                else:
                    pct_Verified_spans.append((m_pct.start(), m_pct.end()))

        # Check unbacked decimal measurements in non-table prose
        if not stripped.startswith("|"):
            for m_dm in re.finditer(r"\b(\d+\.\d+)\s*(ns|µs|us|ms|MiB|GiB)\b", line):
                if any(ps <= m_dm.start() and m_dm.end() <= pe for ps, pe in pct_Verified_spans):
                    continue
                num_s, unit_s = m_dm.groups()
                if num_s not in backed_decimal_nums:
                    errors.append(
                        f"{md_path}:{idx}: decimal measurement `{num_s} {unit_s}` in prose not backed by any benchmark table or fenced output in ledger"
                    )

        # 5b. Check standalone commit SHA claims in prose without `sha:path:line` citations
        if not line_cites and len(line_shas) == 1 and not stripped.startswith(("#", "|")):
            c_sha = line_shas[0]
            # Skip benchmark comparison parentheticals like `(vs ... at c64babc)` or `spliced onto <sha>:<path>`
            if not re.search(rf"\b(?:parent|at|from|vs|onto)\s+`{re.escape(c_sha)}`", line) and f"{c_sha}:" not in line:
                msg_w, touched_files = get_commit_info(c_sha)
                touched_words = set(msg_w)
                for tf in touched_files:
                    touched_words.update(file_words.get(tf, set()))
                repo_syms = [
                    tok
                    for s in bt_spans
                    if "/" not in s and "." not in s and "::" not in s
                    for tok in IDENT_RE.findall(s)
                    if tok in all_words and tok not in STD_TYPES and len(tok) >= 4
                ]
                if repo_syms and not any(sym in touched_words for sym in repo_syms):
                    errors.append(
                        f"{md_path}:{idx}: commit `{c_sha}` did not touch any file or message mentioning {repo_syms}"
                    )

        # 6. Check unattached file-membership when a line names a single `.rs` file without `:line` citations
        if (
            not line_cites
            and "{" not in line
            and "::" not in line
            and "MIRIFLAGS" not in line
            and "CARGO_TARGET_DIR" not in line
            and not any(s.endswith("/") for s in bt_spans)
        ):
            rs_mentions = re.findall(r"\b((?:src|tests|benches)/[A-Za-z0-9_./\-]+\.rs)\b", line)
            if len(set(rs_mentions)) == 1:
                only_rs = rs_mentions[0]
                f_toks = file_words.get(only_rs, set())
                if f_toks:
                    for raw_span in bt_spans:
                        if raw_span == only_rs or "/" in raw_span or raw_span.endswith(".rs"):
                            continue
                        for tok in IDENT_RE.findall(raw_span):
                            if tok in STD_TYPES or len(tok) < 4:
                                continue
                            if tok in all_words and tok not in f_toks:
                                errors.append(
                                    f"{md_path}:{idx}: identifier `{tok}` does not appear in `{only_rs}`"
                                )

        # 6b. Check un-cited atomic field ordering claims in prose
        if not line_cites:
            ord_pref = set(
                re.findall(r"Ordering::(Relaxed|Acquire|Release|AcqRel|SeqCst)", line)
            )
            claimed_ords = ord_pref or set(
                re.findall(r"\b(Relaxed|Acquire|Release|AcqRel|SeqCst)\b", line)
            )
            if claimed_ords:
                field_mentions = [
                    bt
                    for bt in bt_spans
                    if bt in idx_data.field_orderings
                ]
                for fm in field_mentions:
                    actual_ords = idx_data.field_orderings[fm]
                    if claimed_ords - actual_ords:
                        errors.append(
                            f"{md_path}:{idx}: atomic field `{fm}` ordering claim {sorted(claimed_ords)} contradicts repo orderings {sorted(actual_ords)}"
                        )

        # 7. Check Before/After ancestry and per-clause `Ordering::*` claims on the same line
        if len(line_cites) >= 2 and "Before" in line and "After" in line:
            c_before, c_after = line_cites[0], line_cites[-1]
            if c_before["sha"] != c_after["sha"]:
                if (
                    check_sha(c_before["sha"])[0]
                    and check_sha(c_after["sha"])[0]
                    and git_run(["merge-base", "--is-ancestor", c_before["sha"], c_after["sha"]])[0] != 0
                ):
                    errors.append(
                        f"{md_path}:{idx}: Before/After commits reversed: Before `{c_before['sha']}` is not an ancestor of After `{c_after['sha']}`"
                    )
            after_pos = line.find("**After**")
            if after_pos == -1:
                after_pos = line.find("After")
            if after_pos != -1:
                before_clause = line[:after_pos]
                after_clause = line[after_pos:]
                for clause_label, clause_txt, c_ref in [
                    ("Before", before_clause, c_before),
                    ("After", after_clause, c_after),
                ]:
                    c_ords = set(
                        re.findall(r"\b(Relaxed|Acquire|Release|AcqRel|SeqCst)\b", clause_txt)
                    )
                    if c_ords:
                        fl_c = file_lines(c_ref["sha"], c_ref["path"])
                        if fl_c and 1 <= c_ref["a"] <= c_ref["b"] <= len(fl_c):
                            snip_c = "\n".join(fl_c[c_ref["a"] - 1 : c_ref["b"]])
                            s_ords = set(
                                re.findall(r"\b(Relaxed|Acquire|Release|AcqRel|SeqCst)\b", snip_c)
                            )
                            if c_ords - s_ords:
                                errors.append(
                                    f"{md_path}:{idx}: {clause_label} clause ordering(s) {sorted(c_ords - s_ords)} absent from `{c_ref['sha']}:{c_ref['path']}:{c_ref['a']}-{c_ref['b']}` (found {sorted(s_ords)})"
                                )
                            elif re.search(r"\b(?:CAS\s+success|on\s+success)\b", clause_txt):
                                succ_ords = set(
                                    re.findall(
                                        r"compare_exchange(?:_weak)?\s*\([^;]*?\bOrdering::(Relaxed|Acquire|Release|AcqRel|SeqCst)\s*,",
                                        snip_c,
                                        flags=re.S,
                                    )
                                )
                                if succ_ords and (c_ords - succ_ords):
                                    errors.append(
                                        f"{md_path}:{idx}: {clause_label} CAS success ordering {sorted(c_ords)} contradicts `{c_ref['sha']}:{c_ref['path']}:{c_ref['a']}-{c_ref['b']}` (CAS success ordering is {sorted(succ_ords)})"
                                    )

        # 8. Check single-citation ordering claims, iteration counts, and `checks <X> before <Y>` order
        if len(line_cites) == 1:
            c0 = line_cites[0]
            fl0 = file_lines(c0["sha"], c0["path"])
            if fl0 and 1 <= c0["a"] <= c0["b"] <= len(fl0):
                snip0 = "\n".join(fl0[c0["a"] - 1 : c0["b"]])
                # If the citation points to a struct definition and the line mentions drop, include its Drop impl
                ext_snip0 = snip0
                if "drop" in line.lower() and (c0["b"] - c0["a"]) <= 15:
                    for sx, sy, snames, skind in item_spans(c0["sha"], c0["path"]):
                        if sx <= c0["a"] and c0["b"] <= sy and skind == "struct":
                            for ix, iy, inames, ikind in item_spans(c0["sha"], c0["path"]):
                                if ikind == "impl" and (snames & inames):
                                    ext_snip0 += "\n" + "\n".join(fl0[ix - 1 : iy])
                claimed_ords = set(
                    re.findall(r"\b(Relaxed|Acquire|Release|AcqRel|SeqCst)\b", line)
                )
                if claimed_ords:
                    snip_ords = set(
                        re.findall(r"\b(Relaxed|Acquire|Release|AcqRel|SeqCst)\b", ext_snip0)
                    )
                    missing_ords = claimed_ords - snip_ords
                    if missing_ords:
                        errors.append(
                            f"{md_path}:{idx}: claimed ordering(s) {sorted(missing_ords)} absent from cited snippet `{c0['sha']}:{c0['path']}:{c0['a']}-{c0['b']}` (found {sorted(snip_ords)})"
                        )
                    else:
                        if re.search(r"\b(?:CAS\s+success|on\s+success)\b", line):
                            succ_ords = set(
                                re.findall(
                                    r"compare_exchange(?:_weak)?\s*\([^;]*?\bOrdering::(Relaxed|Acquire|Release|AcqRel|SeqCst)\s*,",
                                    ext_snip0,
                                    flags=re.S,
                                )
                            )
                            if succ_ords and (claimed_ords - succ_ords):
                                errors.append(
                                    f"{md_path}:{idx}: CAS success ordering {sorted(claimed_ords)} contradicts `{c0['sha']}:{c0['path']}:{c0['a']}-{c0['b']}` (CAS success ordering is {sorted(succ_ords)})"
                                )
                        # Check field-specific ordering within the snippet when a specific atomic field is named
                        for bt in bt_spans:
                            if bt in idx_data.field_orderings:
                                f_ords_in_snip = set(
                                    re.findall(
                                        rf"\b{re.escape(bt)}\s*\.\s*(?:load|store|swap|fetch_\w+|compare_exchange(?:_weak)?)\s*\([^;]*?\bOrdering::(Relaxed|Acquire|Release|AcqRel|SeqCst)",
                                        ext_snip0,
                                    )
                                )
                                if f_ords_in_snip and not (claimed_ords & f_ords_in_snip):
                                    errors.append(
                                        f"{md_path}:{idx}: field `{bt}` ordering {sorted(claimed_ords)} contradicts `{bt}` ordering {sorted(f_ords_in_snip)} in `{c0['sha']}:{c0['path']}:{c0['a']}-{c0['b']}`"
                                    )

                m_iter = re.search(r"`?(\d+)`?\s+iterations\b", line)
                if m_iter:
                    claimed_it = int(m_iter.group(1))
                    loop_bounds = {
                        int(m_lb)
                        for m_lb in re.findall(r"\b0\.\.=?(\d+)", snip0)
                    }
                    if loop_bounds and claimed_it not in loop_bounds:
                        errors.append(
                            f"{md_path}:{idx}: claimed `{claimed_it} iterations` does not match loop bounds {sorted(loop_bounds)} in `{c0['sha']}:{c0['path']}:{c0['a']}-{c0['b']}`"
                        )

                m_bef = re.search(
                    r"`([A-Za-z_]\w*)(?:\(\))?`\s+before\s+`([A-Za-z_]\w*)(?:\(\))?`",
                    line,
                )
                if m_bef:
                    sym1, sym2 = m_bef.group(1), m_bef.group(2)
                    p1 = snip0.find(sym1)
                    p2 = snip0.find(sym2)
                    if p1 != -1 and p2 != -1 and p1 > p2:
                        errors.append(
                            f"{md_path}:{idx}: claim `{sym1}` before `{sym2}` contradicted by `{c0['sha']}:{c0['path']}:{c0['a']}-{c0['b']}` (where `{sym2}` precedes `{sym1}`)"
                        )

    # 9. Validate every citation's SHA, ancestry, file, line bounds, and semantic item-span / window match
    for c in cites:
        citation_count += 1
        idx = c["line"]
        sha, path, a, b = c["sha"], c["path"], c["a"], c["b"]
        exists, anc = check_sha(sha)
        if not exists:
            errors.append(f"{md_path}:{idx}: BAD_SHA `{sha}:{path}:{a}-{b}`")
            continue
        if not anc:
            errors.append(f"{md_path}:{idx}: NOT_ANCESTOR `{sha}:{path}:{a}-{b}`")
            continue
        fl = file_lines(sha, path)
        if fl is None:
            errors.append(f"{md_path}:{idx}: NO_FILE `{sha}:{path}:{a}-{b}`")
            continue
        if not (1 <= a <= b <= len(fl)):
            errors.append(
                f"{md_path}:{idx}: RANGE({len(fl)}) `{sha}:{path}:{a}-{b}` out of bounds 1..={len(fl)}"
            )
            continue

        line_str = lines[idx - 1]
        clause_start = 0
        for prev_c in cites:
            if prev_c["line"] == idx and prev_c["end"] <= c["start"]:
                if prev_c["start"] == c["start"] and prev_c["end"] == c["end"]:
                    continue
                end_pos = prev_c["end"]
                if end_pos < len(line_str) and line_str[end_pos] == "`":
                    end_pos += 1
                clause_start = max(clause_start, end_pos)
        clause_end = c["start"]
        if clause_end > clause_start and line_str[clause_end - 1] == "`":
            clause_end -= 1
        clause_pre = line_str[clause_start:clause_end]
        ms = [
            m_b
            for m_b in BT_RE.finditer(clause_pre)
            if not m_b.group(1).strip().startswith("#[")
        ]
        attached = False
        if ms:
            t = ms[-1]
            gap = clause_pre[t.end() :]
            before_t = clause_pre[: t.start()]
            tok = t.group(1).strip()
            tok_clean = re.sub(r"\(.*\)$", "", tok)
            tok_clean = re.sub(r"<[^<>]*>", "", tok_clean)
            tok_clean = re.sub(r"<[^<>]*>", "", tok_clean)
            tok_clean = tok_clean.replace("&mut ", "").replace("&", "")
            m_id = re.fullmatch(
                r"(?:([A-Za-z_]\w*)::)?(?:([A-Za-z_]\w*)::)?([A-Za-z_]\w*)!?",
                tok_clean,
            )
            if re.fullmatch(r"\s*(?:\(|at|in|:|,|—|-)?\s*`?", gap):
                if m_id and not re.fullmatch(r"[0-9a-f]{7,40}", tok_clean):
                    parts = [x for x in m_id.groups() if x]
                    if len(parts) == 1 and parts[0] in STD_TYPES:
                        attached = True
                        # Still verify that a generic wrapper type like `Box<T>` or `Rc<T>` appears in the cited window/impl
                        if re.search(r"<[A-Z]\w*>", tok) and "," not in gap and not name_ok(parts[0], sha, path, a, b):
                            errors.append(
                                f"{md_path}:{idx}: symbol `{tok}` (`{parts[0]}`) absent in window and enclosing items at `{sha}:{path}:{a}-{b}`"
                            )
                    else:
                        attached = True
                        sem_checked_count += 1
                        is_prep_field = bool(
                            re.search(r"\b(?:on|via|with|of|for|from|into)\s+$", before_t)
                        )
                        for nm in parts:
                            d = defs_of(nm, sha, path)
                            if d:
                                if not any(x <= a <= y or prefix_ok(fl, a, x) for x, y in d):
                                    if is_prep_field and name_ok(nm, sha, path, a, b) == "win":
                                        continue
                                    errors.append(
                                        f"{md_path}:{idx}: symbol `{t.group(1)}` (`{nm}`) start line {a} outside item span {[x for x, _ in d][:4]} in `{sha}:{path}:{a}-{b}`"
                                    )
                                elif nm == parts[-1] and not nm.startswith("test_"):
                                    d_fn = fn_defs_of(nm, sha, path)
                                    if len(d_fn) == 1:
                                        fx, fy = d_fn[0]
                                        if (fx <= a <= fy or prefix_ok(fl, a, fx)) and b > fy + 10:
                                            errors.append(
                                                f"{md_path}:{idx}: symbol `{t.group(1)}` (`{nm}`) end line {b} crosses item end {fy} in `{sha}:{path}:{a}-{b}`"
                                            )
                            else:
                                if not name_ok(nm, sha, path, a, b):
                                    errors.append(
                                        f"{md_path}:{idx}: symbol `{t.group(1)}` (`{nm}`) absent in window and enclosing items at `{sha}:{path}:{a}-{b}`"
                                    )
            elif m_id:
                parts = [x for x in m_id.groups() if x]
                nm = parts[-1]
                if (
                    len(parts) == 1
                    and nm in STD_TYPES
                    and re.search(r"<[A-Z]\w*>", tok)
                    and "," not in gap
                    and len(gap) <= 25
                    and re.search(r"\b(?:at|in)\b", gap)
                ):
                    attached = True
                    if not name_ok(nm, sha, path, a, b):
                        errors.append(
                            f"{md_path}:{idx}: symbol `{tok}` (`{nm}`) absent in window and enclosing items at `{sha}:{path}:{a}-{b}`"
                        )
                d_fn = fn_defs_of(nm, sha, path)
                if len(d_fn) == 1:
                    fx, fy = d_fn[0]
                    if fx + 2 < a < b <= fy and (b - a) <= 10 and (fy - fx) >= 15:
                        gap_norm = gap.lower().replace("-", "")
                        claim_words = [
                            w.rstrip("s")
                            for w in re.findall(r"[a-z]{4,}", gap_norm)
                            if w not in {"with", "from", "into", "when", "that", "this", "line", "lines"}
                        ]
                        snip_low = "\n".join(fl[a - 1 : b]).lower()
                        if claim_words and not any(cw in snip_low for cw in claim_words):
                            errors.append(
                                f"{md_path}:{idx}: sub-span `{sha}:{path}:{a}-{b}` inside `{nm}` ({fx}-{fy}) does not match prose claim `{gap.strip()}`"
                            )

        if not attached and path.endswith(".rs"):
            post = line_str[c["end"] :]
            m_post_sym = re.match(
                r"`?\s*(?:\([^()]*\)\s*)?(?:\(\s*`?([A-Za-z_]\w*)`?\s*\)|`(?:[A-Za-z_]\w*::)*([A-Za-z_]\w*)`|\b([a-z][a-z0-9]*_[a-z0-9_]+)\b)",
                post,
            )
            if m_post_sym:
                candidates = {
                    m_post_sym.group(1) or m_post_sym.group(2) or m_post_sym.group(3)
                }
            else:
                pre_clause = re.split(r"[;.]\s+|,\s+(?=[a-z])", clause_pre)[-1]
                pre_clause = re.sub(r"^\s*\(`?[A-Za-z_]\w*`?\)\s*(?:and\b|,)?", "", pre_clause)
                candidates = set()
                for m_bt_c in BT_RE.finditer(pre_clause):
                    bt_s = m_bt_c.group(1).strip()
                    if re.fullmatch(r"[A-Za-z_]\w*", bt_s) and bt_s not in STD_TYPES:
                        candidates.add(bt_s)
                    elif re.fullmatch(r"[A-Za-z_]\w*::[A-Za-z_]\w*", bt_s):
                        for p_sym in bt_s.split("::"):
                            if p_sym not in STD_TYPES:
                                candidates.add(p_sym)
                if not candidates:
                    unbt = BT_RE.sub("", pre_clause)
                    candidates.update(
                        w
                        for w in re.findall(r"\b[A-Z][a-z][A-Za-z0-9_]{2,}\b", unbt)
                        if w not in STD_TYPES and w not in PROSE_CAMEL_ALLOW and w != "Guard"
                    )
                    candidates.update(re.findall(r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b", unbt))
            defined_cands = []
            for nm in candidates:
                d = defs_of(nm, sha, path, non_impl_only=True)
                if d:
                    defined_cands.append((nm, d))
            if defined_cands:
                sem_checked_count += 1
                group_ranges = [
                    (gc["a"], gc["b"])
                    for gc in cites
                    if gc["line"] == idx and gc["start"] == c["start"] and gc["end"] == c["end"]
                ]
                if not any(
                    any(
                        any(x <= ga <= y or prefix_ok(fl, ga, x) for x, y in d)
                        or name_ok(nm, sha, path, ga, gb)
                        for ga, gb in group_ranges
                    )
                    for nm, d in defined_cands
                ):
                    nm0, d0 = defined_cands[0]
                    errors.append(
                        f"{md_path}:{idx}: prose symbol `{nm0}` start line {a} outside item span {[x for x, _ in d0][:4]} in `{sha}:{path}:{a}-{b}`"
                    )

    return citation_count, sem_checked_count, commit_row_count, pair_count, errors


def main() -> int:
    paths = [pathlib.Path(p) for p in sys.argv[1:]]
    if not paths:
        paths = sorted((REPO_ROOT / "docs" / "review").glob("*.md"))

    idx_data = build_historical_index()

    any_errors = False
    for path in paths:
        citations, sem_checked, rows, tm_pairs, errors = audit_ledger(path, idx_data)
        print(
            f"=== Ledger Audit: {path} ===\n"
            f"Citations checked:         {citations}\n"
            f"Semantic spans checked:    {sem_checked}\n"
            f"Commit rows checked:       {rows}\n"
            f"AST Type::member checked:  {tm_pairs}\n"
            f"Errors found:              {len(errors)}"
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
