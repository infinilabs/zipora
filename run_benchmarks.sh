#!/bin/bash

# Comprehensive Zipora Benchmark Runner
# Runs all performance tests and generates comparison report

set -e

echo "🔬 Zipora Comprehensive Benchmark Suite"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [[ ! -f "Cargo.toml" ]]; then
    print_error "Must be run from zipora project root"
    exit 1
fi

# Create results directory
RESULTS_DIR="benchmark_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
print_status "Results will be saved to: $RESULTS_DIR"

# 1. Build C++ comparison infrastructure
print_status "Building C++ comparison infrastructure..."
if [[ -d "cpp_benchmark" ]]; then
    cd cpp_benchmark
    if ./build.sh; then
        print_success "C++ wrapper built successfully"
        # Test the wrapper
        if ./wrapper_test; then
            print_success "C++ wrapper test passed"
        else
            print_warning "C++ wrapper test failed, continuing with Rust-only benchmarks"
        fi
    else
        print_warning "C++ wrapper build failed, continuing with Rust-only benchmarks"
    fi
    cd ..
else
    print_warning "C++ benchmark directory not found, running Rust-only benchmarks"
fi

# 2. Set up environment
export LD_LIBRARY_PATH="$PWD/cpp_benchmark:$LD_LIBRARY_PATH"
print_status "Environment configured"

# 3. Run individual benchmark suites
BENCHMARKS=(
    "benchmark:Core Performance Suite"
    "rank_select_bench:Rank-Select Operations"
    "specialized_containers_bench:Specialized Containers"
    "memory_pools_bench:Memory Pool Variants"
    "simd_rank_select_bench:SIMD Optimizations"
    "comprehensive_trie_benchmarks:FSA & Trie Performance"
    "fsa_infrastructure_bench:FSA Infrastructure"
    "cache_bench:Cache Performance"
)

print_status "Running benchmark suites..."

for benchmark in "${BENCHMARKS[@]}"; do
    IFS=':' read -r bench_name bench_desc <<< "$benchmark"
    
    print_status "Running $bench_desc ($bench_name)..."
    
    # Run benchmark and save results
    if cargo bench --bench "$bench_name" > "$RESULTS_DIR/${bench_name}.txt" 2>&1; then
        print_success "✅ $bench_desc completed"
    else
        print_warning "⚠️  $bench_desc failed or timed out"
    fi
done

# 4. Try C++ comparison if available
if [[ -f "cpp_benchmark/libzipora_wrapper.so" ]]; then
    print_status "Attempting C++ comparison benchmark..."
    
    # Create temporary cargo config to avoid build script issues
    mkdir -p .cargo_temp
    cat > .cargo_temp/config.toml << EOF
[target.x86_64-unknown-linux-gnu]
rustflags = ["-L", "cpp_benchmark"]
EOF
    
    # Try running the comparison (may fail due to linking issues)
    if CARGO_TARGET_DIR="target_cpp" cargo +stable bench --bench cpp_comparison --config .cargo_temp/config.toml > "$RESULTS_DIR/cpp_comparison.txt" 2>&1; then
        print_success "✅ C++ comparison completed"
    else
        print_warning "⚠️  C++ comparison failed (known linking issues)"
    fi
    
    # Clean up
    rm -rf .cargo_temp
fi

# 5. Generate summary report
print_status "Generating benchmark summary..."

cat > "$RESULTS_DIR/SUMMARY.md" << EOF
# Zipora Benchmark Results

**Generated**: $(date)
**Environment**: $(uname -a)
**Rust Version**: $(rustc --version)

## Benchmark Suites Run

EOF

for benchmark in "${BENCHMARKS[@]}"; do
    IFS=':' read -r bench_name bench_desc <<< "$benchmark"
    if [[ -f "$RESULTS_DIR/${bench_name}.txt" ]]; then
        echo "- ✅ **$bench_desc** ($bench_name)" >> "$RESULTS_DIR/SUMMARY.md"
    else
        echo "- ❌ **$bench_desc** ($bench_name)" >> "$RESULTS_DIR/SUMMARY.md"
    fi
done

echo "" >> "$RESULTS_DIR/SUMMARY.md"
echo "## Key Performance Highlights" >> "$RESULTS_DIR/SUMMARY.md"
echo "" >> "$RESULTS_DIR/SUMMARY.md"

# Extract key metrics from benchmark results
if [[ -f "$RESULTS_DIR/rank_select_bench.txt" ]]; then
    echo "### Rank-Select Performance" >> "$RESULTS_DIR/SUMMARY.md"
    grep -E "(rank1|select1).*time:" "$RESULTS_DIR/rank_select_bench.txt" | head -5 >> "$RESULTS_DIR/SUMMARY.md" || true
    echo "" >> "$RESULTS_DIR/SUMMARY.md"
fi

if [[ -f "$RESULTS_DIR/specialized_containers_bench.txt" ]]; then
    echo "### Specialized Containers" >> "$RESULTS_DIR/SUMMARY.md"
    grep -E "(ValVec32|SmallMap).*time:" "$RESULTS_DIR/specialized_containers_bench.txt" | head -5 >> "$RESULTS_DIR/SUMMARY.md" || true
    echo "" >> "$RESULTS_DIR/SUMMARY.md"
fi

cat >> "$RESULTS_DIR/SUMMARY.md" << EOF

## Files Generated

EOF

for file in "$RESULTS_DIR"/*.txt; do
    if [[ -f "$file" ]]; then
        filename=$(basename "$file")
        echo "- \`$filename\` - $(wc -l < "$file") lines of benchmark output" >> "$RESULTS_DIR/SUMMARY.md"
    fi
done

# 6. Display summary
print_success "Benchmark suite completed!"
echo ""
print_status "Results saved to: $RESULTS_DIR"
print_status "Summary report: $RESULTS_DIR/SUMMARY.md"
echo ""

if [[ -f "$RESULTS_DIR/cpp_comparison.txt" ]]; then
    print_success "✅ C++ comparison data available"
else
    print_warning "⚠️  C++ comparison not available (use individual benchmarks for validation)"
fi

echo ""
print_status "To view results:"
echo "  cat $RESULTS_DIR/SUMMARY.md"
echo "  ls $RESULTS_DIR/"
echo ""
print_status "To update performance documentation:"
echo "  # Use the benchmark data to update docs/PERF_VS_CPP.md"
echo "  # Key files: rank_select_bench.txt, specialized_containers_bench.txt"