#!/usr/bin/env bash
# Reproducible splice-and-run RED verification script for C11 (`ea05138`) and S31 (`e91ef17`).
# Keeps all temporary worktrees inside `target/` (never `/tmp`) and cleans up on exit.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WT_DIR="$REPO_ROOT/target/red_verify_wt_$$"
SHARED_TARGET="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"

cleanup() {
    git -C "$REPO_ROOT" worktree remove --force "$WT_DIR" >/dev/null 2>&1 || rm -rf "$WT_DIR"
}
trap cleanup EXIT

git -C "$REPO_ROOT" worktree add --detach "$WT_DIR" ea05138 >/dev/null

python3 - "$REPO_ROOT/src/ffi/c_api.rs" "$WT_DIR/src/ffi/c_api.rs" <<'PY'
import sys

head_src, base_dst = sys.argv[1], sys.argv[2]
head_text = open(head_src, encoding="utf-8").read()
base_text = open(base_dst, encoding="utf-8").read()

start = head_text.index("    #[test]\n    fn test_c11_1_misaligned_and_overflow_pointers_rejected_without_ub()")
end = head_text.index("    #[test]\n    fn test_s31_r4_error_callback_triggering_error_does_not_recurse()")
snippet = "\n    static CALLBACK_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());\n\n" + head_text[start:end] + "\n}\n"

last_brace = base_text.rfind("}")
with open(base_dst, "w", encoding="utf-8") as f:
    f.write(base_text[:last_brace] + snippet)
PY

run_and_assert_red() {
    local commit_label="$1"
    local tname="$2"
    local expected_substr="$3"
    local out_file="$WT_DIR/${tname}.out"

    set +e
    (
        cd "$WT_DIR"
        CARGO_TARGET_DIR="$SHARED_TARGET" cargo test --locked --lib "$tname" --features ffi -- --test-threads=1
    ) >"$out_file" 2>&1
    local rc=$?
    set -e

    cat "$out_file"
    echo "RED check $tname on $commit_label: exit_code=$rc"
    if [[ "$rc" -eq 0 ]]; then
        echo "ERROR: $tname unexpectedly passed on $commit_label!" >&2
        exit 1
    fi
    if ! grep -Fq "Running unittests" "$out_file"; then
        echo "ERROR: $tname failed to compile (test binary never ran) on $commit_label!" >&2
        exit 1
    fi
    if ! grep -Fq "$expected_substr" "$out_file"; then
        echo "ERROR: $tname failed on $commit_label, but output did not contain expected failure '$expected_substr'!" >&2
        exit 1
    fi
}

echo "=== Part 1: C11.1 / C11.2a..c RED checks on ea05138 ==="
run_and_assert_red "ea05138" \
    "test_c11_1_misaligned_and_overflow_pointers_rejected_without_ub" \
    "slice::from_raw_parts_mut requires the pointer to be aligned and non-null"
run_and_assert_red "ea05138" \
    "test_c11_2a_cresult_not_found_mapping_and_blob_store_get" \
    "right: NotFound"
run_and_assert_red "ea05138" \
    "test_c11_2b_set_last_error_preserves_embedded_nul_bytes" \
    "No error information available"
run_and_assert_red "ea05138" \
    "test_c11_2c_error_callback_reentrant_unregister_does_not_deadlock" \
    "C11.2: re-entrant zipora_set_error_callback inside error callback deadlocked on ERROR_CALLBACK"

echo "=== Part 2: S31-R4 / S31-R5 RED checks on e91ef17 ==="
git -C "$WT_DIR" checkout --force e91ef17 >/dev/null

python3 - \
    "$REPO_ROOT/src/ffi/c_api.rs" "$WT_DIR/src/ffi/c_api.rs" \
    "$REPO_ROOT/src/ffi/containers.rs" "$WT_DIR/src/ffi/containers.rs" <<'PY'
import sys

head_capi, base_capi, head_cont, base_cont = sys.argv[1:5]

hc = open(head_capi, encoding="utf-8").read()
bc = open(base_capi, encoding="utf-8").read()
s_r4 = hc.index("    #[test]\n    fn test_s31_r4_error_callback_triggering_error_does_not_recurse()")
with open(base_capi, "w", encoding="utf-8") as f:
    f.write(bc[:bc.rfind("}")] + "\n" + hc[s_r4:])

hcont = open(head_cont, encoding="utf-8").read()
bcont = open(base_cont, encoding="utf-8").read()
s_r5 = hcont.index("    #[test]\n    fn test_s31_r5_memory_pool_new_rejects_invalid_layout_and_documents_max_chunks()")
with open(base_cont, "w", encoding="utf-8") as f:
    f.write(bcont[:bcont.rfind("}")] + "\n" + hcont[s_r5:])
PY

run_and_assert_red "e91ef17" \
    "test_s31_r4_error_callback_triggering_error_does_not_recurse" \
    "overflowed its stack"
run_and_assert_red "e91ef17" \
    "test_s31_r5_memory_pool_new_rejects_invalid_layout_and_documents_max_chunks" \
    "memory_pool_new(usize::MAX, 1) must fail upfront on invalid Layout"

echo "All 6 RED checks verified cleanly with expected runtime failure signatures!"

