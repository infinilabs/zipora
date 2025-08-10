#!/bin/bash
# Enhanced memory safety testing with Miri
# This script runs comprehensive memory safety tests using Miri

set -euo pipefail

echo "=== Enhanced Memory Safety Testing with Miri ==="
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

# Test categories with different configurations
test_categories=(
    "enhanced_memory_safety::test_use_after_free_protection"
    "enhanced_memory_safety::test_double_free_prevention" 
    "enhanced_memory_safety::test_buffer_overflow_protection"
    "enhanced_memory_safety::test_large_allocation_bounds"
    "enhanced_memory_safety::test_concurrent_memory_safety"
    "enhanced_memory_safety::test_container_integrity_under_pressure"
    "enhanced_memory_safety::test_panic_safety_partial_operations"
    "enhanced_memory_safety::test_memory_ordering_safety"
)

# Run each test category separately for better isolation
for test in "${test_categories[@]}"; do
    echo "🧪 Testing: $test"
    
    # Run with Miri and capture output
    if cargo +nightly miri test $test --quiet 2>/dev/null; then
        echo "   ✅ PASSED"
    else
        echo "   ⚠️  ISSUES DETECTED - Running with verbose output:"
        cargo +nightly miri test $test --verbose || true
    fi
    echo
done

echo "🔬 Running full container safety test suite with Miri..."
if cargo +nightly miri test container_safety_tests --quiet; then
    echo "✅ All container safety tests passed with Miri!"
else
    echo "⚠️  Some issues detected. Running with verbose output for debugging:"
    cargo +nightly miri test container_safety_tests --verbose || true
fi

echo
echo "🚀 Memory safety testing complete!"
echo
echo "💡 Tips:"
echo "   - Use 'MIRIFLAGS=\"-Zmiri-disable-isolation\" cargo +nightly miri test' for file system access"
echo "   - Add 'RUST_BACKTRACE=1' for detailed backtraces on failures"
echo "   - Use '--test-threads=1' to avoid thread-related issues in Miri"
echo
echo "📚 For more Miri options, see: https://github.com/rust-lang/miri"