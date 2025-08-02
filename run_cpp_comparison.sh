#!/bin/bash

# Comprehensive Rust vs C++ Performance Comparison Runner
# 
# This script automates the complete benchmark comparison process:
# 1. Builds the C++ wrapper library
# 2. Runs comprehensive benchmarks
# 3. Analyzes results and generates reports
# 4. Creates visualizations

set -e

echo "=================================================="
echo "COMPREHENSIVE RUST VS C++ PERFORMANCE COMPARISON"
echo "=================================================="
echo ""

# Check dependencies
echo "🔍 Checking dependencies..."

MISSING_DEPS=0

if ! command -v cargo >/dev/null 2>&1; then
    echo "❌ ERROR: cargo (Rust) is required but not installed"
    MISSING_DEPS=1
fi

if ! command -v cmake >/dev/null 2>&1; then
    echo "❌ ERROR: cmake is required but not installed"
    MISSING_DEPS=1
fi

if ! command -v g++ >/dev/null 2>&1; then
    echo "❌ ERROR: g++ is required but not installed"
    MISSING_DEPS=1
fi

if [ $MISSING_DEPS -eq 1 ]; then
    echo ""
    echo "Please install missing dependencies:"
    echo "  Ubuntu/Debian: sudo apt-get install build-essential cmake cargo"
    echo "  CentOS/RHEL:   sudo yum install gcc-c++ cmake cargo"
    echo "  macOS:         brew install cmake && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

echo "✅ All dependencies found"
echo ""

# Step 1: Build C++ wrapper
echo "🔨 Step 1: Building C++ wrapper library..."
cd cpp_benchmark

if [ ! -f "build.sh" ]; then
    echo "❌ ERROR: build.sh not found in cpp_benchmark directory"
    exit 1
fi

./build.sh

if [ ! -f "libtopling_zip_wrapper.so" ]; then
    echo "❌ ERROR: Failed to build C++ wrapper library"
    exit 1
fi

echo "✅ C++ wrapper library built successfully"
echo ""

# Step 2: Set up environment
echo "🔧 Step 2: Setting up environment..."
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)

# Verify the wrapper works
if [ -f "verify_benchmark" ]; then
    echo "Running wrapper verification..."
    ./verify_benchmark
    echo "✅ Wrapper verification passed"
else
    echo "⚠️  Warning: Wrapper verification executable not found"
fi

cd ..
echo ""

# Step 3: Run Rust benchmarks (warm-up)
echo "🔥 Step 3: Running benchmark warm-up..."
echo "This helps ensure consistent performance measurement..."

cargo bench --bench benchmark > /dev/null 2>&1 || echo "⚠️  Warning: Standard benchmarks failed, continuing..."

echo "✅ Warm-up completed"
echo ""

# Step 4: Run comprehensive comparison benchmarks
echo "🚀 Step 4: Running comprehensive C++ vs Rust benchmarks..."
echo "This may take several minutes depending on your system..."

# Run with both console output and JSON capture
echo "Running benchmarks with console output..."
cargo bench --bench cpp_comparison

echo ""
echo "Running benchmarks with JSON output for analysis..."
cargo bench --bench cpp_comparison -- --output-format=json > benchmark_results.json 2>/dev/null || {
    echo "⚠️  Warning: JSON output failed, running alternative method..."
    cargo bench --bench cpp_comparison 2>&1 | tee benchmark_console.log
    
    # Create a mock JSON structure if the real one failed
    cat > benchmark_results.json << 'EOF'
{
    "benchmarks": [],
    "note": "Results captured in benchmark_console.log"
}
EOF
}

echo "✅ Benchmark execution completed"
echo ""

# Step 5: Analyze results
echo "📊 Step 5: Analyzing results and generating reports..."

cd cpp_benchmark

# Check if we have results to analyze
if [ -f "../benchmark_results.json" ]; then
    # Check if Python is available for analysis
    if command -v python3 >/dev/null 2>&1; then
        echo "Generating comprehensive analysis report..."
        python3 analyze_results.py ../benchmark_results.json --report comprehensive_report.txt
        
        # Try to create visualization
        python3 -c "import matplotlib" >/dev/null 2>&1 && {
            echo "Creating performance visualization..."
            python3 analyze_results.py ../benchmark_results.json --visualize performance_comparison.png --no-display
        } || echo "⚠️  matplotlib not available, skipping visualization"
        
    else
        echo "⚠️  Python3 not available, generating basic report..."
        
        # Create a basic text report
        cat > comprehensive_report.txt << EOF
RUST VS C++ PERFORMANCE COMPARISON REPORT
==========================================
Generated: $(date)

Benchmark execution completed successfully.
Results are available in benchmark_results.json

To generate detailed analysis, install Python 3:
  Ubuntu/Debian: sudo apt-get install python3 python3-pip
  CentOS/RHEL:   sudo yum install python3 python3-pip
  macOS:         brew install python

Then run: python3 analyze_results.py ../benchmark_results.json

For visualization features, also install:
  pip3 install matplotlib numpy

Raw benchmark data:
$(head -50 ../benchmark_results.json 2>/dev/null || echo "JSON data not available")
EOF
    fi
else
    echo "⚠️  Warning: benchmark_results.json not found, creating summary from console output..."
    
    cat > comprehensive_report.txt << EOF
RUST VS C++ PERFORMANCE COMPARISON REPORT
==========================================
Generated: $(date)

Note: Detailed JSON results not available.
Check benchmark_console.log for raw output.

To re-run with full analysis capability:
1. Ensure cargo bench supports --output-format=json
2. Update to latest criterion version
3. Re-run this script

Raw console output available in: benchmark_console.log
EOF
fi

cd ..

echo "✅ Analysis completed"
echo ""

# Step 6: Generate summary
echo "📋 Step 6: Generating execution summary..."

TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
REPORT_FILE="cpp_benchmark/comprehensive_report.txt"
JSON_FILE="benchmark_results.json"
CONSOLE_LOG="benchmark_console.log"

cat > BENCHMARK_SUMMARY.md << EOF
# Rust vs C++ Performance Comparison Results

**Execution Date:** $TIMESTAMP  
**System:** $(uname -a)  
**Rust Version:** $(rustc --version 2>/dev/null || echo "Unknown")  
**C++ Compiler:** $(g++ --version | head -1 2>/dev/null || echo "Unknown")  

## Files Generated

- \`$REPORT_FILE\` - Comprehensive analysis report
- \`$JSON_FILE\` - Raw benchmark data (JSON format)
$([ -f "$CONSOLE_LOG" ] && echo "- \`$CONSOLE_LOG\` - Console output log")
$([ -f "cpp_benchmark/performance_comparison.png" ] && echo "- \`cpp_benchmark/performance_comparison.png\` - Performance visualization")

## Quick Results

$(if [ -f "$REPORT_FILE" ]; then
    echo "\`\`\`"
    head -30 "$REPORT_FILE" 2>/dev/null || echo "Report content not available"
    echo "\`\`\`"
else
    echo "Detailed report not available. Check individual files above."
fi)

## Next Steps

1. **Review Detailed Report**: Open \`$REPORT_FILE\` for comprehensive analysis
2. **Examine Raw Data**: Check \`$JSON_FILE\` for detailed metrics
3. **View Visualization**: Open \`cpp_benchmark/performance_comparison.png\` (if available)

## Re-running Benchmarks

To run specific benchmark categories:
\`\`\`bash
# Vector operations only
cargo bench --bench cpp_comparison vector

# String operations only  
cargo bench --bench cpp_comparison string

# Memory efficiency tests
cargo bench --bench cpp_comparison memory
\`\`\`

## System Configuration

- **CPU Cores:** $(nproc 2>/dev/null || echo "Unknown")
- **Memory:** $(free -h 2>/dev/null | grep Mem | awk '{print $2}' || echo "Unknown")
- **CPU Info:** $(grep "model name" /proc/cpuinfo 2>/dev/null | head -1 | cut -d: -f2 | xargs || echo "Unknown")

EOF

echo "✅ Summary generated: BENCHMARK_SUMMARY.md"
echo ""

# Final summary
echo "🎉 BENCHMARK COMPARISON COMPLETED SUCCESSFULLY!"
echo "================================================"
echo ""
echo "📁 Results Location:"
echo "   Main summary:     BENCHMARK_SUMMARY.md"
echo "   Detailed report:  $REPORT_FILE"
echo "   Raw data:         $JSON_FILE"
if [ -f "cpp_benchmark/performance_comparison.png" ]; then
    echo "   Visualization:    cpp_benchmark/performance_comparison.png"
fi
echo ""

echo "🔍 Key Insights:"
if [ -f "$REPORT_FILE" ]; then
    # Try to extract key insights from the report
    grep -A 5 "OVERALL:" "$REPORT_FILE" 2>/dev/null || echo "   See detailed report for performance analysis"
else
    echo "   Run 'python3 cpp_benchmark/analyze_results.py benchmark_results.json' for detailed analysis"
fi

echo ""
echo "📈 To view full analysis:"
echo "   cat $REPORT_FILE"
echo ""
echo "🔄 To re-run specific benchmarks:"
echo "   cargo bench --bench cpp_comparison [operation_type]"
echo ""
echo "Happy benchmarking! 🚀"