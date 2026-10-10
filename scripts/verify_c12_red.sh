#!/usr/bin/env bash
# Reproducible splice-and-run RED verification script for Stage 13 (`C12.1`..`C12.8`) on parent commit `52ae648`.
# Keeps all temporary worktrees inside `target/` (never `/tmp`) and cleans up on exit.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WT_DIR="$REPO_ROOT/target/red_verify_c12_wt_$$"
SHARED_TARGET="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"

cleanup() {
    git -C "$REPO_ROOT" worktree remove --force "$WT_DIR" >/dev/null 2>&1 || rm -rf "$WT_DIR"
}
trap cleanup EXIT

git -C "$REPO_ROOT" worktree add --detach "$WT_DIR" 52ae648 >/dev/null

python3 - "$REPO_ROOT" "$WT_DIR" <<'PY'
import os
import sys

repo_root, wt_dir = sys.argv[1], sys.argv[2]

mod_tests_targets = [
    (
        "src/containers/specialized/int_vec.rs",
        "    #[test]\n    fn test_c12_1_int_vec_from_slice_returns_err_instead_of_aborting_on_invalid_element_width()",
    ),
    (
        "src/system/cpu_features.rs",
        "    #[test]\n    fn test_c12_2_cpu_features_exhaustive_detection_and_single_core_htt()",
    ),
    (
        "src/system/vm_utils.rs",
        "    #[test]\n    fn test_c12_3a_vm_utils_prefetch_provenance_zeroed_alloc_and_advice()",
    ),
    (
        "src/system/process.rs",
        "    #[tokio::test]\n    async fn test_c12_3b_bidirectional_pipe_eof_and_pool_concurrent_kill_while_waiting()",
    ),
    (
        "src/system/profiling.rs",
        "    #[test]\n    fn test_c12_3c_benchmark_suite_zero_iterations_does_not_divide_by_zero()",
    ),
    (
        "src/simd/adaptive.rs",
        "    #[test]\n    fn test_c12_4_adaptive_simd_cache_bucket_poisoning_percentile_bounds_and_zero_duration()",
    ),
    (
        "src/scoring/bm25.rs",
        "    #[test]\n    fn test_c12_5_bm25_zero_tf_zero_norm_and_subnormal_avg_dl_never_nan()",
    ),
    (
        "src/config/nest_louds_trie.rs",
        "    #[test]\n    fn test_c12_6_nest_louds_trie_config_serde_preserves_compression_and_validators_reject_invalid()",
    ),
    (
        "src/statistics/mod.rs",
        "    #[test]\n    fn test_c12_8_statistics_memory_breakdown_overwrite_and_functional_accumulators()",
    ),
]

for rel_path, marker in mod_tests_targets:
    head_src = os.path.join(repo_root, rel_path)
    base_dst = os.path.join(wt_dir, rel_path)
    head_text = open(head_src, encoding="utf-8").read()
    base_text = open(base_dst, encoding="utf-8").read()
    start = head_text.index(marker)
    last_brace = base_text.rfind("}")
    with open(base_dst, "w", encoding="utf-8") as f:
        f.write(base_text[:last_brace] + "\n" + head_text[start:])

cache_rel = "src/cache/simple_tests.rs"
cache_marker = "#[test]\nfn test_c12_7_lru_page_cache_invalidation_recovery_eof_read_writeback_and_zero_shards()"
head_cache = open(os.path.join(repo_root, cache_rel), encoding="utf-8").read()
base_cache = open(os.path.join(wt_dir, cache_rel), encoding="utf-8").read()
start_cache = head_cache.index(cache_marker)
with open(os.path.join(wt_dir, cache_rel), "w", encoding="utf-8") as f:
    f.write(base_cache + "\n" + head_cache[start_cache:])
PY

run_and_assert_red() {
    local commit_label="$1"
    local tname="$2"
    local expected_substr="$3"
    local out_file="$WT_DIR/${tname}.out"

    set +e
    (
        cd "$WT_DIR"
        CARGO_TARGET_DIR="$SHARED_TARGET" cargo test --locked --lib "$tname" -- --test-threads=1
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

echo "=== Stage 13 (C12.1..C12.8) RED checks on 52ae648 ==="
run_and_assert_red "52ae648" \
    "test_c12_1_int_vec_from_slice_returns_err_instead_of_aborting_on_invalid_element_width" \
    "verify(mem::size_of::<T>() >= 1 && mem::size_of::<T>() <= 8) failed: element size 16 must be between 1 and 8 bytes !"
run_and_assert_red "52ae648" \
    "test_c12_2_cpu_features_exhaustive_detection_and_single_core_htt" \
    "CpuFeatures::new() must have SSE2 disabled"
run_and_assert_red "52ae648" \
    "test_c12_3a_vm_utils_prefetch_provenance_zeroed_alloc_and_advice" \
    "attempt to multiply with overflow"
run_and_assert_red "52ae648" \
    "test_c12_3b_bidirectional_pipe_eof_and_pool_concurrent_kill_while_waiting" \
    "BidirectionalPipe::wait() hung because stdin was never closed!"
run_and_assert_red "52ae648" \
    "test_c12_3c_benchmark_suite_zero_iterations_does_not_divide_by_zero" \
    "divide by zero error when dividing duration by scalar"
run_and_assert_red "52ae648" \
    "test_c12_4_adaptive_simd_cache_bucket_poisoning_percentile_bounds_and_zero_duration" \
    "size 8 Scalar selection must not poison cache for size 32 (>= sse2_min_size 16)"
run_and_assert_red "52ae648" \
    "test_c12_5_bm25_zero_tf_zero_norm_and_subnormal_avg_dl_never_nan" \
    "Bm25BatchScorer::score(0, 0) with b=1.0 must be 0.0, got NaN"
run_and_assert_red "52ae648" \
    "test_c12_6_nest_louds_trie_config_serde_preserves_compression_and_validators_reject_invalid" \
    "serde roundtrip dropped compression_algorithm"
run_and_assert_red "52ae648" \
    "test_c12_7_lru_page_cache_invalidation_recovery_eof_read_writeback_and_zero_shards" \
    "second read after invalidate_page must hit cache instead of being permanently poisoned"
run_and_assert_red "52ae648" \
    "test_c12_8_statistics_memory_breakdown_overwrite_and_functional_accumulators" \
    "MemoryBreakdown::total must equal sum of components after overwriting key"

echo "All 10 Stage 13 (C12) RED checks verified cleanly with expected runtime failure signatures!"
