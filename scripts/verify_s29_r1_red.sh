#!/usr/bin/env bash
# Reproducible splice-and-run script for S29-R1 RED regression tests on base f37cf61.
# Splices `test_s29_r1_fallback_runtime_mode_*` from current `src/concurrency/work_stealing.rs`
# into a clean temporary worktree at `f37cf61` and runs `cargo test --release --locked --lib`.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_COMMIT="${1:-f37cf61}"
WT_DIR="/tmp/zipora_s29_r1_red_$$"
TARGET_DIR="${CARGO_TARGET_DIR:-/tmp/zipora_s29_r1_red_target}"

cleanup() {
    git -C "$REPO_ROOT" worktree remove --force "$WT_DIR" >/dev/null 2>&1 || rm -rf "$WT_DIR"
}
trap cleanup EXIT

git -C "$REPO_ROOT" worktree add --detach "$WT_DIR" "$BASE_COMMIT" >/dev/null

python3 - "$REPO_ROOT/src/concurrency/work_stealing.rs" "$WT_DIR/src/concurrency/work_stealing.rs" <<'PY'
import sys

src_head, dst_base = sys.argv[1], sys.argv[2]
head_text = open(src_head, encoding="utf-8").read()
base_text = open(dst_base, encoding="utf-8").read()

start_marker = "    fn run_s29_r1_fallback_respawn_mode(mode: char) {"
end_marker = "    #[test]\n    #[ignore]\n    fn bench_s29_r1_executor_timings()"
s = head_text.index(start_marker)
e = head_text.index(end_marker)
snippet = head_text[s:e].rstrip() + "\n}\n"

last_brace = base_text.rfind("}")
spliced = base_text[:last_brace] + "\n" + snippet
with open(dst_base, "w", encoding="utf-8") as f:
    f.write(spliced)
PY

set +e
(
    cd "$WT_DIR"
    CARGO_TARGET_DIR="$TARGET_DIR" cargo test --release --locked --lib test_s29_r1_fallback_runtime_mode_ -- --test-threads=1
)
rc=$?
set -e
echo "exit_code=$rc"
