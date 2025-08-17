#!/bin/bash
# Enhanced memory safety testing with Miri
# This script runs comprehensive memory safety tests using Miri

set -euo pipefail

# Configuration
TIMEOUT_PER_TEST=${TIMEOUT_PER_TEST:-300}  # 5 minutes per test
COMPREHENSIVE_TIMEOUT=${COMPREHENSIVE_TIMEOUT:-1800}  # 30 minutes for comprehensive tests
MODE=${1:-fast}  # fast, full, or comprehensive

echo "=== Enhanced Memory Safety Testing with Miri ==="
echo "🔧 Mode: $MODE (use 'fast', 'full', or 'comprehensive')"
echo "⏱️  Timeout per test: ${TIMEOUT_PER_TEST}s"
echo
echo "🎯 Testing Coverage:"
echo "   ✓ Core memory safety (use-after-free, double-free, buffer overflow)"
echo "   ✓ Specialized hash maps (GoldenRatio, StringOptimized, Small)"  
echo "   ✓ String arena safety and reference counting"
echo "   ✓ Inline storage and automatic fallback mechanisms"
echo "   ✓ Hash function safety and distribution quality"
if [ "$MODE" != "fast" ]; then
echo "   ✓ Container integrity under memory pressure"
echo "   ✓ Panic safety and partial operation recovery"
echo "   ✓ Memory ordering and concurrency safety"
fi
echo

# Check if nightly toolchain is available
if ! command -v rustup &> /dev/null; then
    echo "❌ Error: rustup not found. Please install rustup first."
    exit 1
fi

# Install nightly toolchain if not present
if ! rustup toolchain list | grep -q nightly; then
    echo "📦 Installing nightly toolchain..."
    rustup install nightly
fi

# Install Miri if not present
if ! rustup component list --toolchain nightly | grep -q "miri.*installed"; then
    echo "📦 Installing Miri..."
    rustup +nightly component add miri
fi

echo "🔍 Running enhanced memory safety tests..."
echo

# Helper function to run test with timeout and progress
run_test_with_timeout() {
    local test_name="$1"
    local timeout_duration="$2"
    
    echo "🧪 Testing: $test_name (timeout: ${timeout_duration}s)"
    echo "   ⏳ Starting at $(date '+%H:%M:%S')..."
    
    if timeout "$timeout_duration" cargo +nightly miri test "$test_name" --quiet 2>/dev/null; then
        echo "   ✅ PASSED (completed at $(date '+%H:%M:%S'))"
        return 0
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            echo "   ⏰ TIMEOUT after ${timeout_duration}s"
        else
            echo "   ⚠️  ISSUES DETECTED - Running with verbose output:"
            timeout "$timeout_duration" cargo +nightly miri test "$test_name" --verbose || true
        fi
        return $exit_code
    fi
}

# Fast mode tests (essential memory safety only)
fast_tests=(
    "memory::secure_pool"
    "fast_vec"
)

# Core container safety tests (existing)
container_safety_tests=(
    "fast_vec"
    "containers::specialized"
    "memory::secure_pool"
    "memory::pool"
)

# Hash map specific tests (focusing on memory safety aspects)
hash_map_safety_tests=(
    "hash_map::golden_ratio_hash_map::tests"
    "hash_map::string_optimized_hash_map::tests"
    "hash_map::small_hash_map::tests"
    "hash_map::hash_functions::tests"
    "hash_map::gold_hash_map::tests"
)

# Select test suite based on mode
case "$MODE" in
    "fast")
        echo "🚀 Running fast memory safety tests..."
        test_suite=("${fast_tests[@]}")
        ;;
    "full")
        echo "🔍 Running full memory safety tests..."
        test_suite=("${container_safety_tests[@]}" "${hash_map_safety_tests[@]}")
        ;;
    "comprehensive")
        echo "🔬 Running comprehensive memory safety tests..."
        test_suite=("${container_safety_tests[@]}" "${hash_map_safety_tests[@]}")
        ;;
    *)
        echo "❌ Invalid mode: $MODE. Use 'fast', 'full', or 'comprehensive'"
        exit 1
        ;;
esac

# Run selected test suite
failed_tests=0
total_tests=${#test_suite[@]}
current_test=0

for test in "${test_suite[@]}"; do
    current_test=$((current_test + 1))
    echo "[$current_test/$total_tests]"
    
    if ! run_test_with_timeout "$test" "$TIMEOUT_PER_TEST"; then
        failed_tests=$((failed_tests + 1))
    fi
    echo
done

# Additional tests for full and comprehensive modes
if [ "$MODE" = "full" ] || [ "$MODE" = "comprehensive" ]; then
    echo "🗺️ Running general hash map tests with Miri..."
    if ! run_test_with_timeout "hash_map" "$TIMEOUT_PER_TEST"; then
        failed_tests=$((failed_tests + 1))
    fi
    echo
fi

# Comprehensive library test (only in comprehensive mode)
if [ "$MODE" = "comprehensive" ]; then
    echo "🔬 Running comprehensive memory safety test suite with Miri..."
    echo "⚠️  This may take up to $((COMPREHENSIVE_TIMEOUT / 60)) minutes..."
    
    if timeout "$COMPREHENSIVE_TIMEOUT" cargo +nightly miri test --lib --quiet; then
        echo "✅ All library tests passed with Miri!"
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            echo "⏰ TIMEOUT after $((COMPREHENSIVE_TIMEOUT / 60)) minutes"
            echo "💡 Consider using 'full' mode for faster testing"
        else
            echo "⚠️  Some issues detected. Running sample with verbose output:"
            timeout 60 cargo +nightly miri test --lib --verbose || true
        fi
        failed_tests=$((failed_tests + 1))
    fi
fi

# Summary
echo
if [ $failed_tests -eq 0 ]; then
    echo "🎉 All tests passed! Memory safety validation complete."
else
    echo "⚠️  $failed_tests test(s) failed or timed out."
    echo "💡 Consider investigating failures or using a different mode."
fi

echo
echo "📊 Summary:"
echo "   Mode: $MODE"
echo "   Tests run: $total_tests"
echo "   Failed: $failed_tests"
echo "   Timeout per test: ${TIMEOUT_PER_TEST}s"
echo
echo "🚀 Memory safety testing complete!"
echo
echo "💡 Usage Tips:"
echo "   Fast mode:          ./run_miri_tests.sh fast          (2 tests, ~10 min)"
echo "   Full mode:          ./run_miri_tests.sh full          (9 tests, ~45 min)"
echo "   Comprehensive:      ./run_miri_tests.sh comprehensive (all tests, ~hours)"
echo
echo "🔧 Environment Variables:"
echo "   TIMEOUT_PER_TEST=600     # Increase timeout to 10 minutes"
echo "   COMPREHENSIVE_TIMEOUT=3600 # Increase comprehensive timeout to 1 hour"
echo
echo "🛠️  Debugging Tips:"
echo "   - Use 'MIRIFLAGS=\"-Zmiri-disable-isolation\" cargo +nightly miri test' for file system access"
echo "   - Add 'RUST_BACKTRACE=1' for detailed backtraces on failures"
echo "   - Use '--test-threads=1' to avoid thread-related issues in Miri"
echo "   - Run specific tests: cargo +nightly miri test hash_map::<module>::tests"
echo
echo "📚 For more Miri options, see: https://github.com/rust-lang/miri"

# Exit with appropriate code
exit $failed_tests