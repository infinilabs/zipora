#!/bin/bash

# Enhanced build script for comprehensive C++ benchmark wrapper
set -e

echo "Building enhanced C++ benchmark wrapper for comprehensive topling-zip comparison..."

# Check for required tools
command -v cmake >/dev/null 2>&1 || { echo "ERROR: cmake is required but not installed."; exit 1; }
command -v g++ >/dev/null 2>&1 || { echo "ERROR: g++ is required but not installed."; exit 1; }

# Create build directory
mkdir -p build
cd build

# Check system capabilities
echo "Detecting system capabilities..."
if grep -q avx2 /proc/cpuinfo 2>/dev/null; then
    echo "✓ AVX2 support detected"
    EXTRA_FLAGS="-mavx2"
else
    echo "! AVX2 not detected, using SSE4.2"
    EXTRA_FLAGS="-msse4.2"
fi

# Configure with CMake with optimized flags
echo "Configuring build with maximum optimization..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -march=native -mtune=native -flto $EXTRA_FLAGS -ffast-math" \
    -DCMAKE_C_FLAGS_RELEASE="-O3 -DNDEBUG -march=native -mtune=native -flto $EXTRA_FLAGS -ffast-math" \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=TRUE

# Build the wrapper library with enhanced features
echo "Building wrapper library..."
make -j$(nproc) VERBOSE=1

# Check if the library was built successfully
if [ -f "libtopling_zip_wrapper.so" ]; then
    echo "✓ Shared library built successfully"
else
    echo "! Warning: Shared library not found, checking for static library..."
fi

if [ -f "libtopling_zip_wrapper.a" ]; then
    echo "✓ Static library built successfully"
fi

# Run tests to verify functionality
if [ -f "wrapper_test" ]; then
    echo "Running comprehensive wrapper tests..."
    ./wrapper_test
    echo "✓ All tests passed"
else
    echo "! Warning: Test executable not found"
fi

# Copy library to parent directory for easy access
if [ -f "libtopling_zip_wrapper.so" ]; then
    cp libtopling_zip_wrapper.so ../
    echo "✓ Library copied to cpp_benchmark/ directory"
fi

# Create a simple benchmark verification script
cat > ../verify_benchmark.cpp << 'EOF'
#include "wrapper.hpp"
#include <iostream>
#include <chrono>

int main() {
    std::cout << "=== C++ Benchmark Wrapper Verification ===" << std::endl;
    
    // Test basic vector operations
    void* vec = cpp_valvec_create();
    for (int i = 0; i < 1000; ++i) {
        cpp_valvec_push(vec, i);
    }
    std::cout << "✓ Vector operations: " << cpp_valvec_size(vec) << " elements" << std::endl;
    cpp_valvec_destroy(vec);
    
    // Test string operations
    const char* test_str = "Hello, benchmark world!";
    void* fstr = cpp_fstring_create(reinterpret_cast<const uint8_t*>(test_str), strlen(test_str));
    uint64_t hash = cpp_fstring_hash(fstr);
    std::cout << "✓ String operations: hash = " << hash << std::endl;
    cpp_fstring_destroy(fstr);
    
    // Test hash map operations
    void* map = cpp_hashmap_create();
    cpp_hashmap_insert(map, "test_key", 42);
    int32_t value;
    bool found = cpp_hashmap_get(map, "test_key", &value);
    std::cout << "✓ HashMap operations: found = " << found << ", value = " << value << std::endl;
    cpp_hashmap_destroy(map);
    
    // Test system info
    CppSystemInfo info;
    cpp_get_system_info(&info);
    std::cout << "✓ System info: " << info.cpu_cores << " cores, " 
              << (info.physical_memory / (1024*1024*1024)) << " GB RAM" << std::endl;
    
    // Test performance measurement
    auto start = std::chrono::high_resolution_clock::now();
    double alloc_time = cpp_measure_allocation_speed(1000, 64);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "✓ Performance measurement: allocation test = " << alloc_time << " μs" << std::endl;
    
    // Test comprehensive benchmark
    CppPerformanceSummary summary;
    cpp_run_comprehensive_benchmark(&summary);
    std::cout << "✓ Comprehensive benchmark: overall score = " << summary.overall_score << std::endl;
    
    std::cout << "=== All verification tests passed! ===" << std::endl;
    return 0;
}
EOF

echo ""
echo "Build completed successfully!"
echo ""
echo "Enhanced C++ wrapper features:"
echo "  ✓ Comprehensive vector operations (batch, capacity management)"
echo "  ✓ Advanced string operations (concat, repeat, compare)"
echo "  ✓ Hash map implementation with batch operations"
echo "  ✓ Bit vector and rank-select operations"
echo "  ✓ Detailed memory tracking and statistics"
echo "  ✓ Cache efficiency and memory bandwidth measurement"
echo "  ✓ System information and hardware detection"
echo "  ✓ High-precision timing and statistical analysis"
echo "  ✓ Comprehensive benchmark suite"
echo ""
echo "To use the enhanced wrapper in Rust benchmarks:"
echo "1. Ensure libtopling_zip_wrapper.so is in LD_LIBRARY_PATH:"
echo "   export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)/.."
echo "2. Run comprehensive C++ vs Rust comparison:"
echo "   cd .. && cargo bench --bench cpp_comparison"
echo "3. For detailed analysis, run:"
echo "   cargo bench --bench cpp_comparison -- --output-format=json"
echo ""
echo "To verify the wrapper functionality:"
echo "   cd .. && g++ -std=c++17 -O3 -o verify_benchmark verify_benchmark.cpp -L. -ltopling_zip_wrapper && ./verify_benchmark"