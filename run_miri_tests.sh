#!/bin/bash
# Memory safety testing with Miri
#
# Flags:
#   -Zmiri-disable-stacked-borrows : crossbeam-epoch 0.9.x triggers SB violations
#   -Zmiri-disable-isolation       : secure_pool uses clock_gettime(REALTIME)
#   -Zmiri-ignore-leaks            : epoch-based reclamation doesn't free on exit

set -euo pipefail

BASE_MIRIFLAGS="-Zmiri-disable-stacked-borrows -Zmiri-disable-isolation"
TIMEOUT_PER_TEST=${TIMEOUT_PER_TEST:-300}
COMPREHENSIVE_TIMEOUT=${COMPREHENSIVE_TIMEOUT:-1800}
MODE=${1:-fast}

echo "=== Miri Memory Safety Tests ==="
echo "Mode: $MODE | Timeout: ${TIMEOUT_PER_TEST}s/test"
echo

# Ensure nightly + miri are installed
if ! command -v rustup &> /dev/null; then
    echo "Error: rustup not found."
    exit 1
fi
if ! rustup toolchain list | grep -q nightly; then
    rustup install nightly
fi
if ! rustup component list --toolchain nightly | grep -q "miri.*installed"; then
    rustup +nightly component add miri
fi

# Run a Miri test group.
# $1 = test filter, $2 = timeout, $3 = extra MIRIFLAGS (optional)
run_miri_test() {
    local filter="$1"
    local timeout_s="$2"
    local extra_flags="${3:-}"

    echo -n "  $filter ... "

    local output
    export MIRIFLAGS="$BASE_MIRIFLAGS $extra_flags"
    output=$(timeout "$timeout_s" cargo +nightly miri test --lib "$filter" 2>&1) || true

    # Check for real UB (not leaks)
    if echo "$output" | grep -q 'Undefined Behavior'; then
        echo "FAILED (UB detected)"
        echo "$output" | grep -A2 'Undefined Behavior' | head -10
        return 1
    fi

    # Check pass count
    local passed
    passed=$(echo "$output" | grep -oP '\d+ passed' | head -1 || echo "")

    if [ -n "$passed" ]; then
        echo "ok ($passed)"
        return 0
    fi

    # Timeout
    if echo "$output" | grep -q 'TIMEOUT\|timeout'; then
        echo "TIMEOUT (${timeout_s}s)"
        return 1
    fi

    # Other failure
    echo "FAILED"
    echo "$output" | grep -E 'error:|unsupported' | head -5
    return 1
}

# --- Test suites ---

# Core unsafe: CPUID bypass, SIMD fallbacks, pointer ops, select_in_word, popcount
# These tests have NO crossbeam dependency — strict leak checking applies.
core_unsafe_tests=(
    "algorithms::bit_ops::tests"
    "containers::fast_vec::tests::alignment_tests"
)

# Memory pool tests use crossbeam-epoch LockFreeStack.
# Epoch-based reclamation intentionally leaks on exit → -Zmiri-ignore-leaks.
pool_tests=(
    "memory::secure_pool::tests::test_secure_pool_creation"
    "memory::secure_pool::tests::test_secure_allocation_deallocation"
    "memory::secure_pool::tests::test_chunk_validation"
    "memory::secure_pool::tests::test_pool_reuse"
    "memory::secure_pool::tests::test_memory_access"
    "memory::secure_pool::tests::test_size_classes"
)

# Hash map tests
hash_map_tests=(
    "hash_map::golden_ratio_hash_map::tests"
    "hash_map::string_optimized_hash_map::tests"
    "hash_map::small_hash_map::tests"
    "hash_map::hash_functions::tests"
    "hash_map::gold_hash_map::tests"
)

# Full container tests
container_tests=(
    "containers::fast_vec::tests"
    "containers::specialized"
)

case "$MODE" in
    "fast")
        echo "Fast mode: core unsafe + pool basics"
        ;;
    "full")
        echo "Full mode: core + pools + hash maps + containers"
        ;;
    "comprehensive")
        echo "Comprehensive mode: full suite + all lib tests"
        ;;
    *)
        echo "Usage: $0 [fast|full|comprehensive]"
        exit 1
        ;;
esac
echo

failed=0
total=0

run_group() {
    local group_name="$1"
    shift
    local extra_flags="$1"
    shift
    local tests=("$@")

    echo "--- $group_name ---"
    for filter in "${tests[@]}"; do
        total=$((total + 1))
        if ! run_miri_test "$filter" "$TIMEOUT_PER_TEST" "$extra_flags"; then
            failed=$((failed + 1))
        fi
    done
    echo
}

# Always run core unsafe tests (no leak suppression)
run_group "Core unsafe code" "" "${core_unsafe_tests[@]}"

# Always run pool tests (with leak suppression for crossbeam-epoch)
run_group "Memory pools (crossbeam-epoch)" "-Zmiri-ignore-leaks" "${pool_tests[@]}"

if [ "$MODE" = "full" ] || [ "$MODE" = "comprehensive" ]; then
    run_group "Hash maps" "" "${hash_map_tests[@]}"
    run_group "Containers" "" "${container_tests[@]}"
fi

# Comprehensive: run all lib tests
if [ "$MODE" = "comprehensive" ]; then
    echo "--- All lib tests ---"
    echo -n "  --lib (all) ... "
    total=$((total + 1))
    MIRIFLAGS="$BASE_MIRIFLAGS -Zmiri-ignore-leaks" \
        output=$(timeout "$COMPREHENSIVE_TIMEOUT" cargo +nightly miri test --lib 2>&1) || true
    if echo "$output" | grep -q 'Undefined Behavior'; then
        echo "FAILED (UB detected)"
        failed=$((failed + 1))
    elif echo "$output" | grep -qP '\d+ passed'; then
        passed=$(echo "$output" | grep -oP '\d+ passed' | tail -1)
        echo "ok ($passed)"
    else
        echo "TIMEOUT or error"
        failed=$((failed + 1))
    fi
    echo
fi

echo "=== Results ==="
echo "Mode: $MODE | Groups: $total | Failed: $failed"

if [ $failed -eq 0 ]; then
    echo "All Miri tests passed."
else
    echo "$failed group(s) failed."
fi

exit $failed
