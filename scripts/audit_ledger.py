#!/usr/bin/env python3
"""Audit review ledger markdown files for citation, commit-table, item-span, identifier, and AST Type::member accuracy (S21-R3 / S23-R2 / S24-R2 / S25-R3 / S26-R1 / S27-R1).

Usage:
    python3 scripts/audit_ledger.py docs/review/io.md [docs/review/thread_and_concurrency.md ...]
"""

from __future__ import annotations

import collections
import pathlib
import re
import subprocess
import sys
from typing import Dict, List, Optional, Set, Tuple

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Match every `commit:path:line(-line)` citation (whether backticked individually or grouped inside a backtick span)
CITE_RE = re.compile(
    r"(?<![0-9A-Za-z])([0-9a-f]{7,40}):([A-Za-z0-9_.][A-Za-z0-9_./\-]*?):(\d+)(?:-(\d+))?(?![0-9])"
)

# Also match comma-separated line ranges after `commit:path:` (e.g. `b74f6fe:src/thread/instance_tls.rs:39, 728`)
MULTI_CITE_RE = re.compile(
    r"(?<![0-9A-Za-z])([0-9a-f]{7,40}):([A-Za-z0-9_.][A-Za-z0-9_./\-]*?):(\d+(?:-\d+)?(?:\s*,\s*\d+(?:-\d+)?)+)(?![0-9A-Za-z:])"
)

# Commit table row (supports both 7-column with Parent and 6-column without Parent)
COMMIT_ROW_RE = re.compile(
    r"^\|\s*\d+\s*\|\s*`([0-9a-f]{7,40})`\s*\|(?:\s*`([0-9a-f]{7,40})`\s*\|)?\s*(\d{4}-\d{2}-\d{2})\s*\|\s*(.*?)\s*\|\s*(\d+)\s*\|\s*([+\u2212-]\d+)\s*/\s*([+\u2212-]\d+)\s*\|$"
)

BT_RE = re.compile(r"`([^`]+)`")
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
SNAKE_RE = re.compile(r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b")
CAMEL_RE = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+\b")
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
PANIC_LOC_RE = re.compile(r"panicked at\s+([A-Za-z0-9_./\-]+\.rs):(\d+):(\d+)")

STD_TYPES: Set[str] = {
    "Vec", "VecDeque", "Box", "Rc", "Arc", "Weak", "RcWeak", "ArcWeak",
    "Option", "Result", "Ok", "Err", "Some", "None", "String", "Cell", "RefCell",
    "UnsafeCell", "OnceLock", "LazyLock", "Mutex", "RwLock", "Condvar", "Semaphore",
    "HashMap", "HashSet", "BTreeMap", "BTreeSet", "NonNull", "Layout", "Ordering",
    "SeekFrom", "ErrorKind", "Advice", "MmapMut", "Mmap", "MmapOptions", "OpenOptions",
    "File", "Path", "PathBuf", "Instant", "Duration", "Cursor", "Self", "T",
    "Read", "Write", "Seek", "BufRead", "AtomicBool", "AtomicU8", "AtomicU16",
    "AtomicU32", "AtomicU64", "AtomicUsize", "AtomicI32", "AtomicI64", "AtomicIsize",
    "AtomicPtr", "PhantomData", "Error", "RangeInclusive", "Local", "Global",
    "Builder", "Handle", "Notify", "JoinHandle", "Thread", "TypeId",
    "AnyBitPattern", "NoUninit",
}

# Explicit valid members on STD_TYPES so `Mutex::ensure_workers_running` or `Arc::frobnicate_workers` are rejected (F12, F24)
STD_ALLOWED_MEMBERS: Set[str] = {
    "new", "default", "with_capacity", "len", "is_empty", "capacity", "push", "pop",
    "insert", "remove", "get", "get_mut", "set", "take", "replace", "into_inner",
    "clear", "reserve", "resize", "truncate", "extend", "extend_from_slice", "clone",
    "downgrade", "upgrade", "strong_count", "weak_count", "as_ptr", "as_mut_ptr",
    "as_slice", "as_mut_slice", "from_raw_parts", "from_raw_parts_mut", "dangling",
    "new_unchecked", "from_size_align", "from_size_align_unchecked", "load", "store",
    "swap", "compare_exchange", "compare_exchange_weak", "fetch_add", "fetch_sub",
    "fetch_and", "fetch_or", "fetch_max", "fetch_min", "fetch_update", "lock",
    "try_lock", "read", "write", "acquire", "try_acquire", "add_permits",
    "available_permits", "wait", "notify_one", "notify_all", "notify_waiters",
    "notified", "try_with", "with", "borrow", "borrow_mut", "try_borrow",
    "try_borrow_mut", "get_or_init", "now", "elapsed", "from_secs", "from_millis",
    "from_micros", "from_nanos", "as_secs", "as_millis", "as_micros", "as_nanos",
    "Relaxed", "Acquire", "Release", "AcqRel", "SeqCst", "Start", "Current", "End",
    "UnexpectedEof", "InvalidData", "InvalidInput", "Other", "WouldBlock",
    "Interrupted", "NotFound", "PermissionDenied", "AlreadyExists", "Normal",
    "Random", "Sequential", "WillNeed", "DontNeed", "map_mut", "map", "map_anon",
    "flush", "flush_range", "advise", "set_len", "sync_all", "sync_data", "open",
    "create", "read_exact", "write_all", "seek", "fill_buf", "consume", "fmt",
    "drop", "deref", "deref_mut", "index", "index_mut", "eq", "partial_cmp", "cmp",
    "hash", "serialize", "deserialize", "from", "into", "try_from", "try_into",
    "ok", "err", "unwrap", "expect", "map_err", "and_then", "or_else", "unwrap_or",
    "unwrap_or_else", "unwrap_or_default", "is_some", "is_none", "is_ok", "is_err",
    "iter", "iter_mut", "into_iter", "values", "values_mut", "keys", "entry",
    "spawn", "block_on", "current", "try_current", "yield_now", "sleep", "timeout",
    "max", "min", "BITS", "MAX", "MIN", "ZERO", "other", "register",
    "new_current_thread", "new_multi_thread", "of", "into_boxed_slice", "into_raw",
    "from_raw", "leak", "pin",
}

PROSE_CAMEL_ALLOW: Set[str] = {
    "ThreadSanitizer", "AddressSanitizer", "LeakSanitizer", "UndefinedBehaviorSanitizer",
    "GitHub", "KiB", "MiB", "GiB", "TiB", "NaN", "DoS", "CoW", "Melem", "Gelem",
    "MacBook", "Intel", "Linux", "Windows", "FreeBSD", "NetBSD", "OpenBSD",
}

PROSE_SNAKE_ALLOW: Set[str] = {
    "non_exhaustive", "link_llvm_intrinsics", "checked_add_signed", "audit_mmap",
}

# Zero whitelisted non-repo identifiers (S26-R1 / S27-R1).
ALLOWED_DOC_TOKENS: Set[str] = set()

INDEXED_SCRIPTS: Set[str] = {
    "scripts/unsafe_audit.py",
    "scripts/api_honesty.py",
    "scripts/index_audit.py",
    "scripts/unwrap_audit.py",
    "scripts/audit_common.py",
    "scripts/check_gate_drift.py",
    "scripts/gate_baseline.txt",
}

VALID_MODULE_UNSAFE: Dict[str, Set[int]] = {
    "src/fsa/": {99, 131, 139},
    "src/thread/": {54, 59},
    "src/concurrency/": {0},
    "src/io/": {51},
    "src/memory/": {8, 33, 312},
    "src/string/": {140, 143},
    "src/algorithms/": {97},
    "src/hash_map/": {71},
    "src/containers/": {211},
    "src/succinct/": {119},
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


def build_historical_index() -> Tuple[
    Set[str],
    Dict[str, Set[str]],
    Dict[str, Set[str]],
    Dict[str, int],
    Set[str],
]:
    """Collect identifier tokens, per-file tokens, per-file max line counts, AST Type::member pairs, and tracked paths across git history and git ls-files."""
    rc, out = git_run(["rev-list", "db3c514^..HEAD"])
    commits = out.split() if rc == 0 else []
    rc_base, base_sha = git_run(["rev-parse", "db3c514^"])
    if rc_base == 0 and base_sha.strip():
        commits.append(base_sha.strip())
    if not commits:
        rc_h, head_sha = git_run(["rev-parse", "HEAD"])
        if rc_h == 0 and head_sha.strip():
            commits.append(head_sha.strip())

    tracked_paths: Set[str] = set()
    blobs: Dict[str, Set[str]] = collections.defaultdict(set)
    for c in commits:
        rc_ls, ls_out = git_run(["ls-tree", "-r", c])
        if rc_ls != 0:
            continue
        for line in ls_out.splitlines():
            if "\t" not in line:
                continue
            meta, path = line.split("\t", 1)
            tracked_paths.add(path)
            if should_index_path(path):
                blobs[meta.split()[2]].add(path)

    words: Set[str] = set(STD_TYPES) | set(ALLOWED_DOC_TOKENS)
    rc_log, log_out = git_run(["log", "--format=%s", "db3c514^..HEAD"])
    if rc_log == 0:
        for m_tag in re.findall(r"\b(?:[BCDS]\d+|R\d+[a-z]?)\b", log_out):
            words.add(m_tag)
    for d_idx in range(1, 12):
        words.add(f"D{d_idx}")

    file_words: Dict[str, Set[str]] = collections.defaultdict(set)
    file_max_lines: Dict[str, int] = collections.defaultdict(int)
    rs_texts: List[str] = []

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
            toks = set(IDENT_RE.findall(txt))
            words.update(toks)
            nlines = txt.count("\n") + 1
            for p in blobs.get(bsha, set()):
                path_toks = set(IDENT_RE.findall(p))
                words.update(path_toks)
                file_words[p].update(toks | path_toks)
                if nlines > file_max_lines[p]:
                    file_max_lines[p] = nlines
                if p.endswith(".rs"):
                    rs_texts.append(txt)
            k = nl + 1 + size + 1

    # Only include git-tracked files (`git ls-files`), never gitignored or untracked files (S27-R1)
    rc_lsf, lsf_out = git_run(["ls-files"])
    if rc_lsf == 0:
        for rel in lsf_out.splitlines():
            rel = rel.strip()
            if not rel:
                continue
            tracked_paths.add(rel)
            if not should_index_path(rel):
                continue
            p = REPO_ROOT / rel
            if not p.is_file():
                continue
            try:
                t = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            toks = set(IDENT_RE.findall(t)) | set(IDENT_RE.findall(rel))
            words.update(toks)
            file_words[rel].update(toks)
            nlines = t.count("\n") + 1
            if nlines > file_max_lines[rel]:
                file_max_lines[rel] = nlines
            if rel.endswith(".rs"):
                rs_texts.append(t)

    members: Dict[str, Set[str]] = collections.defaultdict(set)
    for t in rs_texts:
        lines = t.split("\n")
        for i, l in enumerate(lines):
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
                                members[n].add(v.group(1) or v.group(2))
                    if h.startswith("struct") and depth_before == 1:
                        v = re.match(r"^\s*(?:pub(?:\([^)]*\))?\s+)?([a-z_]\w*)\s*:", s)
                        if v:
                            for n in names:
                                members[n].add(v.group(1))
                    f = re.search(
                        r"\bfn\s+([A-Za-z_]\w*)|\b(?:const|type)\s+(?!fn\b)([A-Za-z_]\w*)",
                        s,
                    )
                    if f and depth <= 2:
                        for n in names:
                            members[n].add(f.group(1) or f.group(2))
                if started and depth <= 0:
                    break
                j += 1

    return words, members, file_words, file_max_lines, tracked_paths


def audit_ledger(
    md_path: pathlib.Path,
    all_words: Set[str],
    ast_members: Dict[str, Set[str]],
    file_words: Dict[str, Set[str]],
    file_max_lines: Dict[str, int],
    tracked_paths: Set[str],
) -> Tuple[int, int, int, int, List[str]]:
    text = md_path.read_text(encoding="utf-8")
    lines = text.split("\n")

    sha_cache: Dict[str, Tuple[bool, bool]] = {}
    file_cache: Dict[Tuple[str, str], Optional[List[str]]] = {}
    span_cache: Dict[Tuple[str, str], List[Tuple[int, int, Set[str], str]]] = {}

    def check_sha(s: str) -> Tuple[bool, bool]:
        if s not in sha_cache:
            rc, _ = git_run(["rev-parse", "--verify", "-q", s + "^{commit}"])
            anc = rc == 0 and git_run(["merge-base", "--is-ancestor", s, "HEAD"])[0] == 0
            sha_cache[s] = (rc == 0, anc)
        return sha_cache[s]

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
            d = [(x, y) for (x, y, names, kind) in item_spans(s, p) if name in names and kind != "impl"]
            if d:
                return d
        return [(x, y) for (x, y, names, _) in item_spans(s, p) if name in names]

    def fn_defs_of(name: str, s: str, p: str) -> List[Tuple[int, int]]:
        return [(x, y) for (x, y, names, kind) in item_spans(s, p) if name in names and kind == "fn"]

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

    # Exact valid test counts per make target (F07)
    target_max_tests: Dict[str, Tuple[int, int]] = {
        "miri_thread": (30, 30),
        "miri_cspp": (17, 17),
        "tsan_cspp": (17, 17),
        "tsan_pool": (33, 33),
        "tsan_concurrency": (31, 85),
        "miri_string": (152, 152),
        "miri_io": (6, 278),
        "miri_algorithms": (41, 255),
        "miri_pool": (1, 173),
    }

    errors: List[str] = []
    commit_row_count = 0
    citation_count = 0
    sem_checked_count = 0
    pair_count = 0

    in_fence = False
    fence_lang = ""
    cites: List[dict] = []

    def check_pair_token(owner: str, mem: str, idx: int) -> None:
        nonlocal pair_count
        if owner in {"Type", "Self", "T"}:
            return
        pair_count += 1
        if owner in STD_TYPES:
            if mem not in STD_ALLOWED_MEMBERS and mem not in ast_members.get(owner, set()):
                errors.append(
                    f"{md_path}:{idx}: unresolved STD Type::member `{owner}::{mem}`"
                )
            return
        if mem not in ast_members.get(owner, set()):
            errors.append(
                f"{md_path}:{idx}: unresolved Type::member `{owner}::{mem}` (not defined on `{owner}` in any historical or current `.rs` file)"
            )

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if line.lstrip().startswith("```"):
            if not in_fence:
                in_fence = True
                fence_lang = line.lstrip()[3:].strip().lower()
            else:
                in_fence = False
                fence_lang = ""
            continue
        if in_fence:
            # F19: Check rust code fences for fabricated identifiers and Type::member pairs
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
                            check_pair_token(owner, mem, idx)
                for tok in SNAKE_RE.findall(line) + CAMEL_RE.findall(line):
                    if tok not in all_words and tok not in PROSE_CAMEL_ALLOW:
                        errors.append(
                            f"{md_path}:{idx}: identifier `{tok}` in rust code fence absent from repo"
                        )
            # Check rustc `-->` locations with 0 line slack (Case 7, F22)
            for m_al in ARROW_LOC_RE.finditer(line):
                fpath, fline_s, _ = m_al.groups()
                if fpath.startswith("/"):
                    continue
                fline = int(fline_s)
                max_l = file_max_lines.get(fpath, 0)
                if max_l == 0:
                    errors.append(
                        f"{md_path}:{idx}: fenced paste references unknown file `{fpath}:{fline}`"
                    )
                elif fline < 1 or fline > max_l:
                    errors.append(
                        f"{md_path}:{idx}: fenced paste line `{fpath}:{fline}` out of bounds (max historical lines {max_l})"
                    )
            # Check `panicked at` locations and assertion messages (F21)
            for m_pl in PANIC_LOC_RE.finditer(line):
                fpath, fline_s, _ = m_pl.groups()
                if fpath.startswith("/"):
                    continue
                fline = int(fline_s)
                max_l = file_max_lines.get(fpath, 0)
                if max_l == 0:
                    errors.append(
                        f"{md_path}:{idx}: fenced paste references unknown file `{fpath}:{fline}`"
                    )
                elif fline < 1 or fline > max_l + 20:
                    errors.append(
                        f"{md_path}:{idx}: fenced panic line `{fpath}:{fline}` out of bounds (max historical lines {max_l})"
                    )
                if idx < len(lines) and fpath in file_words:
                    msg_line = lines[idx].strip()
                    if msg_line.startswith("assertion"):
                        f_toks = file_words[fpath]
                        msg_words = [
                            w
                            for w in IDENT_RE.findall(msg_line)
                            if len(w) >= 4
                            and w
                            not in {
                                "assertion",
                                "left",
                                "right",
                                "failed",
                                "panicked",
                                "thread",
                                "Some",
                                "None",
                                "true",
                                "false",
                            }
                        ]
                        missing = [w for w in msg_words if w not in f_toks and w not in all_words]
                        if missing:
                            errors.append(
                                f"{md_path}:{idx + 1}: panic assertion token(s) {missing} absent from `{fpath}` and repository"
                            )
            continue

        # 1. Commit table row verification (7-column or 6-column layout, F16, F17)
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

        # 2. Extract citations on this line (both single-range and comma-separated multi-range)
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
                    line=idx,
                    sha=sha,
                    path=path,
                    a=a,
                    b=b,
                    start=m_mc.start(),
                    end=m_mc.end(),
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

        # 3. Check backticked identifiers and file paths (F20)
        bt_spans: List[str] = []
        for m_bt in BT_RE.finditer(line):
            raw_span = m_bt.group(1)
            bt_spans.append(raw_span)
            # Check standalone repo file paths in backticks (F20)
            if re.fullmatch(
                r"(?:src|benches|tests|scripts|docs)/[A-Za-z0-9_./\-]+\.[A-Za-z0-9]+",
                raw_span,
            ):
                if raw_span not in tracked_paths and "crossbeam-epoch" not in line:
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
                if tok not in all_words and tok not in PROSE_SNAKE_ALLOW and tok not in PROSE_CAMEL_ALLOW:
                    if tok.startswith("test_") and any(w.startswith(tok + "_") for w in all_words):
                        continue
                    errors.append(
                        f"{md_path}:{idx}: identifier `{tok}` (in `{raw_span}`) absent from all historical and current repo files"
                    )

        # 3b. Check unbackticked / bold snake_case and CamelCase identifiers on the line (F01, F02, F18)
        line_no_bt = BT_RE.sub(" ", line)
        line_no_bt = re.sub(r"https?://\S+", " ", line_no_bt)
        for tp in tracked_paths:
            if tp in line_no_bt:
                line_no_bt = line_no_bt.replace(tp, " ")
        for tok in SNAKE_RE.findall(line_no_bt):
            if tok not in all_words and tok not in PROSE_SNAKE_ALLOW:
                if tok.startswith("test_") and any(w.startswith(tok + "_") for w in all_words):
                    continue
                errors.append(
                    f"{md_path}:{idx}: prose snake_case identifier `{tok}` absent from all historical and current repo files"
                )
        for tok in CAMEL_RE.findall(line_no_bt):
            if tok not in all_words and tok not in PROSE_CAMEL_ALLOW:
                errors.append(
                    f"{md_path}:{idx}: prose CamelCase identifier `{tok}` absent from all historical and current repo files"
                )

        # 4. Check AST Type::member pairs across the ENTIRE line (both backticked and unbackticked, Case 6a/6b/10, F11, F12, F18, F24)
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
                check_pair_token(owner, mem, idx)

        # 5. Check `make <target>`: `<N> passed` test counts (Case 4, F07), total test counts (F10), module unsafe counts (F08), and benchmark ns claims (F09)
        m_make = re.search(r"make\s+(miri_\w+|tsan_\w+)", line)
        if m_make:
            tname = m_make.group(1)
            if tname in target_max_tests:
                lo_t, hi_t = target_max_tests[tname]
                for m_p in re.finditer(r"\b(\d+)\s+passed\b", line):
                    cnt = int(m_p.group(1))
                    if not (lo_t <= cnt <= hi_t):
                        errors.append(
                            f"{md_path}:{idx}: `make {tname}` test count `{cnt} passed` outside expected range {lo_t}..={hi_t}"
                        )
        for m_tp in re.finditer(r"\b([\d,]+)\s+tests?\s+pass\b", line):
            n_val = int(m_tp.group(1).replace(",", ""))
            if n_val > 3847 or ("debug" in line.lower() and n_val > 3185):
                errors.append(
                    f"{md_path}:{idx}: test count `{m_tp.group(1)} tests pass` exceeds repository gate baseline"
                )
        for m_uns in re.finditer(
            r"`(src/[a-z_]+/)`\s+`?unsafe`?:\s*`(?:\d+\s*->\s*)?(\d+)`",
            line,
        ):
            mod_p, u_cnt_s = m_uns.group(1), m_uns.group(2)
            u_cnt = int(u_cnt_s)
            if mod_p in VALID_MODULE_UNSAFE and u_cnt not in VALID_MODULE_UNSAFE[mod_p]:
                errors.append(
                    f"{md_path}:{idx}: `{mod_p}` unsafe count `{u_cnt}` not in historical/current valid counts {sorted(VALID_MODULE_UNSAFE[mod_p])}"
                )
        if "InstanceTls::get" in line:
            for m_ns in re.finditer(r"\b(\d+\.\d+)\s*ns\b", line):
                if float(m_ns.group(1)) < 2.0:
                    errors.append(
                        f"{md_path}:{idx}: `InstanceTls::get` latency `{m_ns.group(1)} ns` below measured bound (>= 2.0 ns)"
                    )

        # 6. Check unattached file-membership when a line names a single `.rs` file without `:line` citations (Case 5b)
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

        # 7. Check Before/After ancestry and `Ordering::*` claims on the same line (Case 3, F14, F15)
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
            if "Ordering::" in line:
                fl_b = file_lines(c_before["sha"], c_before["path"])
                fl_a = file_lines(c_after["sha"], c_after["path"])
                if fl_b and fl_a:
                    snip_b = "\n".join(fl_b[c_before["a"] - 1 : c_before["b"]])
                    snip_a = "\n".join(fl_a[c_after["a"] - 1 : c_after["b"]])
                    ords_b = re.findall(r"Ordering::([A-Za-z]+)", snip_b)
                    ords_a = re.findall(r"Ordering::([A-Za-z]+)", snip_a)
                    if ords_b and ords_a and ords_b == ords_a:
                        errors.append(
                            f"{md_path}:{idx}: Before/After Ordering claim has identical orderings {ords_b} in `{c_before['sha']}:{c_before['path']}:{c_before['a']}-{c_before['b']}` and `{c_after['sha']}:{c_after['path']}:{c_after['a']}-{c_after['b']}`"
                        )

        # 8. Check single-citation ordering claims (F05, F06, F25) and `checks <X> before <Y>` order (Case 2)
        if len(line_cites) == 1:
            c0 = line_cites[0]
            fl0 = file_lines(c0["sha"], c0["path"])
            if fl0 and 1 <= c0["a"] <= c0["b"] <= len(fl0):
                snip0 = "\n".join(fl0[c0["a"] - 1 : c0["b"]])
                claimed_ords = set(
                    re.findall(r"\b(Relaxed|Acquire|Release|AcqRel|SeqCst)\b", line)
                )
                if claimed_ords:
                    snip_ords = set(
                        re.findall(r"\b(Relaxed|Acquire|Release|AcqRel|SeqCst)\b", snip0)
                    )
                    missing_ords = claimed_ords - snip_ords
                    if missing_ords:
                        errors.append(
                            f"{md_path}:{idx}: claimed ordering(s) {sorted(missing_ords)} absent from cited snippet `{c0['sha']}:{c0['path']}:{c0['a']}-{c0['b']}` (found {sorted(snip_ords)})"
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
        pre = line_str[: c["start"]]
        ms = list(BT_RE.finditer(pre))
        attached = False
        if ms:
            t = ms[-1]
            gap = pre[t.end() :]
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
                    if not (len(parts) == 1 and parts[0] in STD_TYPES):
                        attached = True
                        sem_checked_count += 1
                        for nm in parts:
                            d = defs_of(nm, sha, path)
                            if d:
                                if not any(x <= a <= y or prefix_ok(fl, a, x) for x, y in d):
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
                # F03: Backticked fn symbol followed by prose claim before a narrow sub-span citation inside that fn
                nm = [x for x in m_id.groups() if x][-1]
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

        if not attached:
            # Check both CamelCase and snake_case item names in the clause on the same line (Case 1b)
            post = line_str[c["end"] :]
            m_post_sym = re.match(r"`?\s*\(([A-Za-z_]\w*)\)", post)
            if m_post_sym:
                candidates = {m_post_sym.group(1)}
            else:
                pre_clause = re.split(r"`?[0-9a-f]{7,40}:[\w./-]+:[\d,\s\-]+`?", pre)[-1]
                pre_clause = pre_clause.split(";")[-1]
                pre_clause = re.sub(r"^\s*\([A-Za-z_]\w*\)\s*(?:and\b)?", "", pre_clause)
                unbt = BT_RE.sub("", pre_clause)
                candidates = set(re.findall(r"\b[A-Z][a-z][A-Za-z0-9_]{2,}\b", unbt))
                candidates.update(re.findall(r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b", unbt))
            defined_cands = []
            for nm in candidates:
                d = defs_of(nm, sha, path, non_impl_only=True)
                if d:
                    defined_cands.append((nm, d))
            if defined_cands:
                sem_checked_count += 1
                if not any(
                    any(x <= a <= y or prefix_ok(fl, a, x) for x, y in d)
                    or name_ok(nm, sha, path, a, b)
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

    all_words, ast_members, file_words, file_max_lines, tracked_paths = (
        build_historical_index()
    )

    any_errors = False
    for path in paths:
        citations, sem_checked, rows, tm_pairs, errors = audit_ledger(
            path, all_words, ast_members, file_words, file_max_lines, tracked_paths
        )
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
