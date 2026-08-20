# Container Safety Testing

Memory safety verification for Zipora containers and memory pools:
use-after-free detection, double-free prevention, buffer overflow
protection, and concurrency safety. The tests live in
`tests/container_safety_tests.rs` (module `enhanced_memory_safety`) and are
designed to also run under Miri.

## Test Categories

### 1. Use-After-Free Protection
- **Test**: `test_use_after_free_protection`
- **Purpose**: Validates that memory allocations remain valid during their lifetime
- **Features**: Tests SecureMemoryPool allocation lifecycle and automatic cleanup

### 2. Double-Free Prevention
- **Test**: `test_double_free_prevention`
- **Purpose**: Ensures memory cannot be freed multiple times
- **Features**: Tests RAII patterns and proper deallocation

### 3. Buffer Overflow Protection
- **Test**: `test_buffer_overflow_protection`
- **Purpose**: Validates bounds checking in containers
- **Features**: Tests container boundary validation and string length limits

### 4. Large Allocation Bounds
- **Test**: `test_large_allocation_bounds`
- **Purpose**: Tests memory allocation limits and graceful failure handling
- **Features**: Progressive allocation testing up to system limits

### 5. Concurrent Memory Safety
- **Test**: `test_concurrent_memory_safety`
- **Purpose**: Validates thread-safe memory operations
- **Features**: Multi-threaded allocation/deallocation stress testing

### 6. Container Integrity Under Pressure
- **Test**: `test_container_integrity_under_pressure`
- **Purpose**: Tests container consistency under memory pressure
- **Features**: Mixed container operations with integrity verification

### 7. Panic Safety with Partial Operations
- **Test**: `test_panic_safety_partial_operations`
- **Purpose**: Ensures containers remain valid after panics
- **Features**: Panic recovery testing and container state validation

### 8. Memory Ordering and Data Races
- **Test**: `test_memory_ordering_safety`
- **Purpose**: Validates memory ordering and prevents data races
- **Features**: Producer-consumer patterns with memory ordering verification

## Test Infrastructure

### SafetyTestConfig

```rust
pub struct SafetyTestConfig {
    pub max_threads: usize,               // Maximum concurrent threads
    pub stress_iterations: usize,         // Stress test iteration count
    pub timeout_seconds: u64,             // Test timeout
    pub memory_pressure_size: usize,      // Memory pressure test size
    pub use_after_free_attempts: usize,   // Use-after-free test attempts
    pub buffer_overflow_test_size: usize, // Buffer overflow test size
}
```

### MemoryUsageTracker

Memory tracking with atomic counters: allocation-pattern tracking, leak
detection with configurable thresholds, concurrent-safe measurements.

## Running the Tests

```bash
# Basic safety tests
cargo test container_safety_tests

# Enhanced memory safety tests only
cargo test enhanced_memory_safety

# With release optimizations
cargo test --release container_safety_tests

# Makefile targets
make safety_tests   # container_safety_tests + enhanced_memory_safety
make miri_tests     # enhanced memory safety under Miri
make miri_full      # broader Miri run
```

## Miri Integration

Configuration lives in `.mirirc` and `Cargo.toml` (`[package.metadata.miri]`):
strict provenance, symbolic alignment checks, number-validity checks, and
disabled isolation (tests need filesystem access). SIMD-specific and AVX-512
test files are excluded (Miri has no SIMD intrinsics support).

```bash
# Install Miri (nightly)
rustup +nightly component add miri

# Run all enhanced memory safety tests with Miri
cargo +nightly miri test enhanced_memory_safety

# Run a specific test
cargo +nightly miri test enhanced_memory_safety::test_use_after_free_protection

# Or use the runner script (auto-installs nightly toolchain)
./run_miri_tests.sh
```

## Adversarial Validation Jobs

Beyond these unit-level safety tests, the repo has fuzzing and sanitizer
jobs (run manually/weekly — CI is build-only by policy):

```bash
make fuzz_smoke   # 60s per target, 8 cargo-fuzz targets in fuzz/
make fuzz_soak    # 1h per target, parallel
make miri_core    # uint_vec_min0 + zipora_hash_map + circular_queue + fast_vec under Miri
make miri_cspp    # ConcurrentCsppTrie under Miri (Tree Borrows)
make tsan_cspp    # ConcurrentCsppTrie under ThreadSanitizer
make tsan_pool    # LockFreeMemoryPool stress under ThreadSanitizer
```

The fuzz targets cover blob-store load/build round-trips, Huffman/rANS/FSE
decoding, DoubleArrayTrie insertion, and UintVecMin0 set/get.

## See Also

- `tests/container_safety_tests.rs` — test implementation
- `run_miri_tests.sh` — Miri test runner
- `.mirirc` — Miri configuration
- `fuzz/` — cargo-fuzz targets
