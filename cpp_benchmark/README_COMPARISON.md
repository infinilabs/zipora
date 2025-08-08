# Comprehensive C++ vs Rust Performance Comparison System

This directory contains a comprehensive benchmark comparison system that enables detailed performance analysis between the Rust `zipora` library and equivalent C++ implementations (including reference libraries when available).

## Overview

The comparison system provides:

- **Head-to-head performance benchmarks** for identical operations
- **Memory usage and allocation tracking** 
- **Cache efficiency analysis**
- **Statistical performance measurement**
- **Comprehensive reporting and visualization**

## Components

### 1. Enhanced C++ Wrapper (`wrapper.hpp` + `wrapper.cpp` + `enhanced_wrapper.cpp`)

A comprehensive C-compatible interface to C++ implementations that provides:

#### Core Data Structures
- **Vector Operations**: `valvec` wrapper with batch operations, capacity management
- **String Operations**: `fstring` wrapper with hashing, searching, manipulation
- **Hash Map Operations**: Comprehensive hash map implementation with batch operations
- **Bit Vector & Rank-Select**: Succinct data structure operations

#### Advanced Features
- **Memory Management**: Detailed tracking, memory pools, allocation statistics
- **Performance Measurement**: High-precision timing, cache analysis, bandwidth measurement
- **System Information**: Hardware detection, CPU capabilities, cache hierarchy
- **Statistical Analysis**: Comprehensive benchmark suite with statistical validation

### 2. Rust Benchmark Suite (`../benches/cpp_comparison.rs`)

A specialized Criterion.rs benchmark suite that:

- Calls C++ functions through FFI for direct comparison
- Uses identical test data and iteration counts
- Measures execution time, memory usage, and allocation counts
- Provides comprehensive coverage of all major operations

### 3. Analysis Tools (`analyze_results.py`)

A Python-based analysis system that:

- Parses Criterion benchmark results
- Generates detailed performance comparison reports  
- Creates comprehensive visualizations
- Provides statistical analysis and performance insights

## Quick Start

### 1. Build the C++ Wrapper

```bash
cd cpp_benchmark
chmod +x build.sh
./build.sh
```

This will:
- Detect system capabilities (AVX2, SSE4.2, etc.)
- Build optimized shared and static libraries
- Run verification tests
- Create a verification executable

### 2. Set Up Environment

```bash
# Add library to path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)

# Verify the wrapper works
g++ -std=c++17 -O3 -o verify_benchmark verify_benchmark.cpp -L. -lzipora_wrapper
./verify_benchmark
```

### 3. Run Comprehensive Benchmarks

```bash
cd ..  # Back to zipora root
cargo bench --bench cpp_comparison
```

For detailed JSON output:
```bash
cargo bench --bench cpp_comparison -- --output-format=json > benchmark_results.json
```

### 4. Analyze Results

```bash
cd cpp_benchmark
python3 analyze_results.py ../benchmark_results.json
```

This generates:
- Comprehensive text report
- Performance visualizations
- Statistical analysis
- Improvement recommendations

## Benchmark Categories

### Vector Operations
- **Creation**: Empty vector vs pre-allocated capacity
- **Push Operations**: Single and batch insertions
- **Access Patterns**: Sequential, random, strided access
- **Memory Efficiency**: Allocation counts, memory usage, fragmentation

### String Operations  
- **Hashing**: Performance across different string sizes
- **Search Operations**: Pattern finding, prefix/suffix matching
- **Manipulation**: Substring creation, concatenation, repetition
- **Memory Management**: String creation and destruction overhead

### Hash Map Operations
- **Insertion**: Single and batch key-value insertions
- **Lookup**: Single and batch key retrieval
- **Memory Usage**: Hash table overhead and efficiency
- **Load Factor**: Performance under different load conditions

### Succinct Data Structures
- **Bit Vector Construction**: Building bit vectors with different patterns
- **Rank-Select Operations**: Query performance for rank1, select1, rank0, select0
- **Memory Overhead**: Space efficiency compared to naive approaches
- **Query Patterns**: Sequential vs random query performance

### Memory Management
- **Allocation Speed**: malloc/new vs custom allocators
- **Deallocation Speed**: free/delete performance
- **Memory Pools**: Pool allocation vs standard allocation
- **Fragmentation Analysis**: Memory layout efficiency

### Cache and System Performance
- **Cache Efficiency**: L1/L2/L3 cache hit rates
- **Memory Bandwidth**: Sequential vs random access patterns
- **SIMD Utilization**: Vectorized operations performance
- **Memory Access Patterns**: Stride analysis and prefetching

## Understanding Results

### Performance Metrics

1. **Throughput**: Operations per second (higher is better)
2. **Latency**: Average time per operation (lower is better)  
3. **Memory Efficiency**: Bytes per operation (lower is better)
4. **Cache Efficiency**: Cache hit ratio (higher is better)

### Comparison Ratios

- **Ratio > 1.0**: Rust is faster
- **Ratio < 1.0**: C++ is faster
- **Ratio ≈ 1.0**: Performance is equivalent

### Report Sections

1. **Summary**: Overall winner and key metrics
2. **Detailed Analysis**: Operation-by-operation breakdown
3. **Performance Insights**: Strengths and improvement areas
4. **Recommendations**: Technology selection guidance

## Advanced Usage

### Custom Test Data

Modify the benchmark parameters in `cpp_comparison.rs`:

```rust
// Test different vector sizes
let sizes = vec![1_000, 10_000, 100_000, 1_000_000];

// Test different string patterns
let test_strings = vec![
    ("Short", "Hello World!"),
    ("Medium", "...".repeat(100)),
    ("Long", "...".repeat(10000)),
];
```

### Memory Pool Testing

Enable detailed memory tracking:

```cpp
// In C++ wrapper
cpp_reset_counters();
CppMemoryStats stats;
cpp_get_memory_stats(&stats);
```

### Cache Analysis

Run cache-specific benchmarks:

```bash
# Focus on cache efficiency
cargo bench --bench cpp_comparison cache_efficiency
```

### Hardware-Specific Optimization

The build system automatically detects and enables:
- **AVX2**: Advanced vector instructions
- **SSE4.2**: SIMD string operations  
- **BMI2**: Bit manipulation instructions
- **LTO**: Link-time optimization

## Troubleshooting

### Build Issues

1. **Missing Dependencies**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install build-essential cmake g++ 
   
   # CentOS/RHEL
   sudo yum install gcc-c++ cmake make
   ```

2. **Reference Library Not Found**:
   - The system automatically falls back to stub implementations
   - Stub implementations provide baseline C++ performance
   - For full comparison, install reference library

3. **Library Linking Issues**:
   ```bash
   # Check library exists
   ls -la libzipora_wrapper.so
   
   # Verify symbols
   nm -D libzipora_wrapper.so | grep cpp_valvec
   ```

### Runtime Issues

1. **FFI Binding Errors**:
   ```bash
   # Ensure library is in path
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/cpp_benchmark
   
   # Test with minimal example
   cd cpp_benchmark && ./verify_benchmark
   ```

2. **Performance Anomalies**:
   - Run multiple iterations: `cargo bench --bench cpp_comparison -- --warm-up-time=10`
   - Check system load: `htop` or `top`
   - Disable CPU frequency scaling: `sudo cpupower frequency-set --governor performance`

### Analysis Issues

1. **Missing Visualization**:
   ```bash
   pip install matplotlib numpy
   ```

2. **JSON Parse Errors**:
   - Ensure Criterion output is valid JSON
   - Use `jq` to validate: `cat benchmark_results.json | jq`

## Performance Optimization Tips

### For Fair Comparison

1. **Use Identical Compiler Flags**:
   - Both Rust and C++ use `-O3 -march=native`
   - Enable LTO for both implementations
   - Use same allocator (system malloc)

2. **Control System Variables**:
   - Disable CPU frequency scaling
   - Close other applications
   - Run multiple iterations and take averages

3. **Use Representative Data**:
   - Test with realistic data sizes
   - Include both synthetic and real-world patterns
   - Test edge cases (empty, very large, etc.)

### For Rust Optimization

1. **Enable All Optimizations**:
   ```toml
   [profile.release]
   lto = true
   codegen-units = 1
   panic = "abort"
   ```

2. **Use SIMD Features**:
   ```bash
   cargo bench --features simd --bench cpp_comparison
   ```

### For C++ Optimization  

1. **Profile-Guided Optimization**:
   - First run: `g++ -fprofile-generate`
   - Second run: `g++ -fprofile-use`

2. **Custom Allocators**:
   - Consider jemalloc or tcmalloc
   - Enable memory pools for frequent allocations

## Contributing

To add new benchmark comparisons:

1. **Add C++ Function**: Implement in `enhanced_wrapper.cpp`
2. **Add Header Declaration**: Update `wrapper.hpp`  
3. **Add Rust FFI Binding**: Update `cpp_comparison.rs`
4. **Add Benchmark**: Create Criterion benchmark function
5. **Update Analysis**: Extend `analyze_results.py` if needed

## References

- [Criterion.rs Documentation](https://docs.rs/criterion/)
- Reference libraries for comparison
- [Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [C++ Optimization Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)

## License

This benchmark comparison system is part of the zipora project and follows the same licensing terms.