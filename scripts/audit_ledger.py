#!/usr/bin/env python3
"""Audit review ledger markdown files for citation, commit-table, item-span, identifier, and AST Type::member accuracy (S21-R3 / S23-R2 / S24-R2 / S25-R3).

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

# Commit table row: | 1 | `commit` | `parent` | YYYY-MM-DD | subject | files | +ins / −del |
COMMIT_ROW_RE = re.compile(
    r"^\|\s*\d+\s*\|\s*`([0-9a-f]{7,40})`\s*\|\s*`([0-9a-f]{7,40})`\s*\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*(.*?)\s*\|\s*(\d+)\s*\|\s*([+\u2212-]\d+)\s*/\s*([+\u2212-]\d+)\s*\|$"
)

BT_RE = re.compile(r"`([^`]+)`")
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
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

STD_TYPES: Set[str] = {
    "Vec", "VecDeque", "Box", "Rc", "Arc", "Weak", "RcWeak", "ArcWeak",
    "Option", "Result", "Ok", "Err", "Some", "None", "String", "Cell", "RefCell",
    "UnsafeCell", "OnceLock", "LazyLock", "Mutex", "RwLock", "Condvar", "Semaphore",
    "HashMap", "HashSet", "BTreeMap", "BTreeSet", "NonNull", "Layout", "Ordering",
    "SeekFrom", "ErrorKind", "Advice", "MmapMut", "Mmap", "MmapOptions", "OpenOptions",
    "File", "Path", "PathBuf", "Instant", "Duration", "Cursor", "Self", "T",
    "Read", "Write", "Seek", "BufRead", "AtomicBool", "AtomicU8", "AtomicU16",
    "AtomicU32", "AtomicU64", "AtomicUsize", "AtomicI32", "AtomicI64", "AtomicIsize",
    "AtomicPtr", "PhantomData", "Error", "RangeInclusive", "Local", "Global", "Collector",
}

ALLOWED_DOC_TOKENS: Set[str] = {
    # Legitimate review-ID prefixes, benchmark names, CLI/Makefile/tool tokens in ledgers
    "DI", "CT", "SP", "RS", "VV", "SB", "D1", "D9", "D10", "B7", "B8",
    "WS1", "WS2", "WS3", "WS5", "WS6", "FP1", "FP2", "PT1", "PT3", "P1", "P2", "P3", "P5",
    "S22", "S23", "S24", "S25", "C9", "C10", "GEMINI", "BASE_MIRIFLAGS", "TFPROBE",
    "io_untrusted_lengths", "asan", "mincore", "yaml", "safe_load", "r4c", "Compiling",
    "golden_ratio_hash_map", "string_optimized_hash_map", "E0599", "taskset",
    "DIRTY_OUTER_", "_STATE", "test_c9_", "scalar_", "neon_", "read_",
    "test_c9_e1_bool_newtype_ub_on_parent", "test_c9_di", "checked_add_signed",
    "test_s23_r1", "test_s23_r3", "vec_rc_shared_100k_de",
    "cspp_lookup_8t", "cspp_lookup_1t", "tls_get_u64", "tls_get_string",
    "load_order", "SharedReadOnly", "steal_amount", "AcquireError", "JoinError", "current_rx",
    "xFFFF_FFFF",
}


def git_run(args: List[str]) -> Tuple[int, str]:
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.returncode, proc.stdout


def build_historical_index() -> Tuple[Set[str], Dict[str, Set[str]]]:
    """Collect all identifier tokens and AST Type::member pairs across historical trees (db3c514^..HEAD) and working tree."""
    rc, out = git_run(["rev-list", "db3c514^..HEAD"])
    commits = out.split() if rc == 0 else []
    rc_base, base_sha = git_run(["rev-parse", "db3c514^"])
    if rc_base == 0 and base_sha.strip():
        commits.append(base_sha.strip())
    if not commits:
        rc_h, head_sha = git_run(["rev-parse", "HEAD"])
        if rc_h == 0 and head_sha.strip():
            commits.append(head_sha.strip())

    blobs: Dict[str, str] = {}
    for c in commits:
        rc_ls, ls_out = git_run(["ls-tree", "-r", c])
        if rc_ls != 0:
            continue
        for line in ls_out.splitlines():
            if "\t" not in line:
                continue
            meta, path = line.split("\t", 1)
            if path.endswith((".rs", ".toml", ".py")) or path == "Makefile":
                blobs[meta.split()[2]] = path

    words: Set[str] = set(STD_TYPES) | set(ALLOWED_DOC_TOKENS)
    rs_texts: List[str] = []

    if blobs:
        proc = subprocess.Popen(
            ["git", "-C", str(REPO_ROOT), "cat-file", "--batch"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        batch_out, _ = proc.communicate(("\n".join(blobs.keys()) + "\n").encode())
        k = 0
        while k < len(batch_out):
            nl = batch_out.index(b"\n", k)
            hdr = batch_out[k:nl].split()
            bsha = hdr[0].decode("ascii", "replace")
            size = int(hdr[2])
            txt = batch_out[nl + 1 : nl + 1 + size].decode("utf-8", "replace")
            words.update(IDENT_RE.findall(txt))
            if blobs.get(bsha, "").endswith(".rs"):
                rs_texts.append(txt)
            k = nl + 1 + size + 1

    # Also include working-tree source files
    for root_dir in ["src", "tests", "benches", "scripts"]:
        base = REPO_ROOT / root_dir
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix in {".rs", ".py", ".toml"}:
                try:
                    t = p.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                words.update(IDENT_RE.findall(t))
                if p.suffix == ".rs":
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

    return words, members


def audit_ledger(
    md_path: pathlib.Path,
    all_words: Set[str],
    ast_members: Dict[str, Set[str]],
) -> Tuple[int, int, int, int, List[str]]:
    text = md_path.read_text(encoding="utf-8")
    lines = text.split("\n")

    sha_cache: Dict[str, Tuple[bool, bool]] = {}
    file_cache: Dict[Tuple[str, str], Optional[List[str]]] = {}
    span_cache: Dict[Tuple[str, str], List[Tuple[int, int, Set[str]]]] = {}

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

    def item_spans(s: str, p: str) -> List[Tuple[int, int, Set[str]]]:
        k = (s, p)
        if k in span_cache:
            return span_cache[k]
        f = file_lines(s, p) or []
        out: List[Tuple[int, int, Set[str]]] = []
        for i, l in enumerate(f):
            names: Optional[Set[str]] = None
            m = ITEM_RE.match(l)
            if m:
                names = {m.group(2)}
            else:
                m2 = IMPL_RE.match(l)
                if m2:
                    names = set(re.findall(r"[A-Za-z_]\w*", m2.group(1)))
                elif re.match(r"^[A-Za-z_][\w-]*:", l) and p.endswith("Makefile"):
                    names = {l.split(":")[0]}
            if not names:
                continue
            if p.endswith("Makefile"):
                j = i + 1
                while j < len(f) and (
                    f[j].startswith("\t") or f[j].strip() == "" or f[j].startswith("#")
                ):
                    j += 1
                out.append((i + 1, j, names))
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
            out.append((i + 1, j + 1, names))
        span_cache[k] = out
        return out

    def defs_of(name: str, s: str, p: str) -> List[Tuple[int, int]]:
        return [(x, y) for (x, y, names) in item_spans(s, p) if name in names]

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
        for x, y, names in item_spans(s, p):
            if x <= a and b <= y and name in names:
                return "encl"
        return None

    errors: List[str] = []
    commit_row_count = 0
    citation_count = 0
    sem_checked_count = 0
    pair_count = 0

    in_fence = False
    cites: List[dict] = []

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        # 1. Commit table row verification (including git show --numstat file/insertion/deletion counts)
        m_row = COMMIT_ROW_RE.match(stripped)
        if m_row:
            commit_row_count += 1
            commit, parent, date, subject, fcnt_s, ins_s, dels_s = m_row.groups()
            rc_m, meta = git_run(["show", "-s", "--format=%p|%as|%s", commit])
            if rc_m != 0 or not meta.strip():
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

        # 2. Extract citations on this line (both single-range and comma-separated multi-range)
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
                cites.append(
                    dict(
                        line=idx,
                        sha=sha,
                        path=path,
                        a=a,
                        b=b,
                        start=m_mc.start(),
                        end=m_mc.end(),
                    )
                )
        for m_c in CITE_RE.finditer(line):
            if any(ms <= m_c.start() and m_c.end() <= me for ms, me in multi_spans):
                continue
            cites.append(
                dict(
                    line=idx,
                    sha=m_c.group(1),
                    path=m_c.group(2),
                    a=int(m_c.group(3)),
                    b=int(m_c.group(4) or m_c.group(3)),
                    start=m_c.start(),
                    end=m_c.end(),
                )
            )

        # 3. Check backticked identifiers and AST Type::member pairs
        for m_bt in BT_RE.finditer(line):
            raw_span = m_bt.group(1)
            # Check all identifier tokens inside backticks against historical + current codebase words
            t_stripped = re.sub(r"[0-9a-f]{7,40}:[\w./-]+:[\d,\s\-]+", "", raw_span)
            t_stripped = re.sub(r"\b[0-9a-f]{7,40}\b", "", t_stripped)
            for tok in IDENT_RE.findall(t_stripped):
                if re.fullmatch(r"[0-9a-f]+", tok) and len(tok) >= 7:
                    continue
                if tok not in all_words:
                    errors.append(
                        f"{md_path}:{idx}: identifier `{tok}` (in `{raw_span}`) absent from all historical and current repo files"
                    )

            # Check Type::member pairs against AST-extracted members
            for m_pair in PAIR_RE.finditer(raw_span):
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
                    pair_count += 1
                    if owner in STD_TYPES:
                        continue
                    if mem not in ast_members.get(owner, set()):
                        errors.append(
                            f"{md_path}:{idx}: unresolved Type::member `{owner}::{mem}` (not defined on `{owner}` in any historical or current `.rs` file)"
                        )

    # 4. Validate every citation's SHA, ancestry, file, line bounds, and semantic item-span / window match
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
            if re.fullmatch(r"\s*(?:\(|at|in|:|,|—|-)?\s*`?", gap):
                tok = t.group(1).strip()
                tok = re.sub(r"\(.*\)$", "", tok)
                tok = re.sub(r"<[^<>]*>", "", tok)
                tok = re.sub(r"<[^<>]*>", "", tok)
                tok = tok.replace("&mut ", "").replace("&", "")
                m_id = re.fullmatch(
                    r"(?:([A-Za-z_]\w*)::)?(?:([A-Za-z_]\w*)::)?([A-Za-z_]\w*)!?",
                    tok,
                )
                if m_id:
                    parts = [x for x in m_id.groups() if x]
                    if not (len(parts) == 1 and len(parts[0]) < 4):
                        attached = True
                        sem_checked_count += 1
                        for nm in parts:
                            d = defs_of(nm, sha, path)
                            if d:
                                if not any(x <= a <= y or prefix_ok(fl, a, x) for x, y in d):
                                    errors.append(
                                        f"{md_path}:{idx}: symbol `{t.group(1)}` (`{nm}`) start line {a} outside item span {[x for x, _ in d][:4]} in `{sha}:{path}:{a}-{b}`"
                                    )
                            else:
                                if not name_ok(nm, sha, path, a, b):
                                    errors.append(
                                        f"{md_path}:{idx}: symbol `{t.group(1)}` (`{nm}`) absent in window and enclosing items at `{sha}:{path}:{a}-{b}`"
                                    )

        if not attached:
            # Also check unbackticked CamelCase item names in the preceding clause on the same line
            pre_clause = re.split(r"`?[0-9a-f]{7,40}:[\w./-]+:[\d,\s\-]+`?", pre)[-1]
            unbt = BT_RE.sub("", pre_clause)
            for nm in re.findall(r"\b[A-Z][a-z][A-Za-z0-9_]{2,}\b", unbt):
                d = defs_of(nm, sha, path)
                if d:
                    sem_checked_count += 1
                    if not (
                        any(x <= a <= y or prefix_ok(fl, a, x) for x, y in d)
                        or name_ok(nm, sha, path, a, b)
                    ):
                        errors.append(
                            f"{md_path}:{idx}: prose symbol `{nm}` start line {a} outside item span {[x for x, _ in d][:4]} in `{sha}:{path}:{a}-{b}`"
                        )

    return citation_count, sem_checked_count, commit_row_count, pair_count, errors


def main() -> int:
    paths = [pathlib.Path(p) for p in sys.argv[1:]]
    if not paths:
        paths = sorted((REPO_ROOT / "docs" / "review").glob("*.md"))

    all_words, ast_members = build_historical_index()

    any_errors = False
    for path in paths:
        citations, sem_checked, rows, tm_pairs, errors = audit_ledger(
            path, all_words, ast_members
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
