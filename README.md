# Zipora

[![Build Status](https://github.com/infinilabs/zipora/workflows/CI/badge.svg)](https://github.com/infinilabs/zipora/actions)
[![License](https://img.shields.io/badge/license-BDL--1.0-blue.svg)](LICENSE)
[![Rust Version](https://img.shields.io/badge/rust-1.88+-orange.svg)](https://www.rust-lang.org)

High-performance Rust data structures and compression algorithms with memory safety guarantees.

## ⚡ Version 2.0 - Unified Architecture

Zipora 2.0 introduces a **unified architecture** following referenced project's philosophy of "one excellent implementation per data structure" with strategy-based configuration. This architectural transformation provides:

- **🔄 Unified Implementations**: Single `ZiporaHashMap` and `ZiporaTrie` with strategy-based configuration
- **🎯 Strategy-Based Design**: Configure behavior through strategies rather than separate types
- **🧹 Cleaner APIs**: Consistent interfaces with powerful customization options
- **📈 Better Performance**: Focused optimization efforts on fewer, better implementations
- **🔧 Easier Maintenance**: Single implementations instead of 14+ separate data structures

> **Migration Note**: Version 2.0 includes breaking changes. See [Migration Guide](docs/MIGRATION_GUIDE.md) for upgrade instructions.

## Features

- **🚀 High Performance**: Zero-copy operations, SIMD optimizations (AVX2, AVX-512*), cache-friendly layouts
- **🛡️ Memory Safety**: Eliminates segfaults, buffer overflows, use-after-free bugs
- **🧠 Secure Memory Management**: Production-ready memory pools with thread safety, RAII, and vulnerability prevention
- **🚨 Advanced Error Handling & Recovery**: Sophisticated error classification (WARNING/RECOVERABLE/CRITICAL/FATAL), automatic recovery strategies (memory reclamation, structure rebuilding, fallback algorithms), contextual error reporting with metadata, and comprehensive verification macros
- **💾 Blob Storage**: Advanced storage systems including trie-based indexing and offset-based compression
- **📦 Specialized Containers**: Production-ready containers with 40-90% memory/performance improvements
- **🗂️ Specialized Hash Maps**: Golden ratio optimized, string-optimized, small inline maps with advanced cache locality optimizations, sophisticated collision resolution algorithms, and memory-efficient string arena management
- **⚡ Cache Optimization Infrastructure**: Comprehensive cache-line alignment, hot/cold data separation, software prefetching, NUMA-aware allocation, and access pattern analysis for maximum performance
- **🌲 Advanced Tries**: LOUDS, Critical-Bit (with BMI2 acceleration), and Patricia tries with rank/select operations, hardware-accelerated path compression, and sophisticated nesting strategies
- **🔄 Advanced Radix Sort Variants**: Multiple sorting strategies (LSD, MSD, Adaptive Hybrid, Parallel) with SIMD optimizations, intelligent algorithm selection, and string-specific optimizations
- **🔒 Version-Based Synchronization**: Advanced token and version sequence management for safe concurrent FSA/Trie access
- **🔗 Low-Level Synchronization**: Linux futex integration, thread-local storage, atomic operations framework
- **⚡ Fiber Concurrency**: High-performance async/await with work-stealing, I/O integration, cooperative multitasking
- **📡 Advanced Serialization**: Comprehensive components with smart pointers, endian handling, version management
- **🗜️ Advanced Compression Framework**: PA-Zip dictionary compression, contextual Huffman (Order-1/Order-2), 64-bit rANS with parallel variants, FSE with ZSTD optimizations, hardware-accelerated bit operations
- **🔄 Real-time Compression**: Adaptive algorithms with strict latency guarantees
- **🔌 C FFI Support**: Complete C API for migration from C++ (enabled with `--features ffi`)
- **🎚️ Five-Level Concurrency Management**: Graduated concurrency control with adaptive selection
- **⚙️ Rich Configuration APIs**: Comprehensive configuration system with trait-based patterns, builder patterns, environment variable integration, presets, validation, and JSON serialization support

## Five-Level Concurrency Management System

Zipora implements a sophisticated 5-level concurrency management system that provides graduated concurrency control options for different performance and threading requirements. The system automatically selects the optimal level based on CPU core count, allocation patterns, and workload characteristics.

### The 5 Levels of Concurrency Control

1. **Level 1: No Locking** - Pure single-threaded operation with zero synchronization overhead
2. **Level 2: Mutex-based Locking** - Fine-grained locking with separate mutexes per size class
3. **Level 3: Lock-free Programming** - Atomic compare-and-swap operations for small allocations
4. **Level 4: Thread-local Caching** - Per-thread local memory pools to minimize cross-thread contention
5. **Level 5: Fixed Capacity Variant** - Bounded memory allocation with no expansion

### Key Benefits

- **API Compatibility**: All levels share consistent interfaces
- **Graduated Complexity**: Each level builds sophistication while maintaining simpler fallbacks
- **Hardware Awareness**: Cache alignment, atomic operations, prefetching
- **Adaptive Selection**: Choose appropriate level based on thread count, allocation patterns, and performance requirements
- **Composability**: Different components can use different concurrency levels

### Usage Examples

```rust
use zipora::memory::{
    AdaptiveFiveLevelPool, ConcurrencyLevel, FiveLevelPoolConfig,
    NoLockingPool, MutexBasedPool, LockFreePool, ThreadLocalPool, FixedCapacityPool,
};

// Automatic adaptive selection (recommended)
let config = FiveLevelPoolConfig::performance_optimized();
let mut pool = AdaptiveFiveLevelPool::new(config).unwrap();
let offset = pool.alloc(1024).unwrap();
println!("Selected level: {:?}", pool.current_level());

// Explicit level selection for specific requirements
let pool = AdaptiveFiveLevelPool::with_level(config, ConcurrencyLevel::ThreadLocal).unwrap();

// Direct use of specific levels
let mut single_thread_pool = NoLockingPool::new(config.clone()).unwrap();
let mutex_pool = MutexBasedPool::new(config.clone()).unwrap();
let lockfree_pool = LockFreePool::new(config.clone()).unwrap();
let threadlocal_pool = ThreadLocalPool::new(config.clone()).unwrap();
let mut fixed_pool = FixedCapacityPool::new(config).unwrap();

// Configuration presets for different use cases
let performance_config = FiveLevelPoolConfig::performance_optimized(); // High throughput
let memory_config = FiveLevelPoolConfig::memory_optimized();           // Low memory usage
let realtime_config = FiveLevelPoolConfig::realtime();                 // Predictable latency
```

### Adaptive Selection Logic

The system intelligently selects the optimal concurrency level:

- **Single-threaded**: Level 1 (No Locking) for maximum performance
- **2-4 cores**: Level 2 (Mutex) or Level 3 (Lock-free) based on allocation size
- **5-16 cores**: Level 3 (Lock-free) or Level 4 (Thread-local) based on arena size
- **16+ cores**: Level 4 (Thread-local) for maximum scalability
- **Fixed capacity**: Level 5 for real-time and constrained environments

### Performance Characteristics

| Level | Scalability | Overhead | Use Case |
|-------|-------------|----------|----------|
| **Level 1** | Single-thread | **Minimal** | Single-threaded applications |
| **Level 2** | Good (2-8 threads) | Low | General multi-threaded use |
| **Level 3** | Excellent (8+ threads) | **Minimal** | High-contention scenarios |
| **Level 4** | **Outstanding** | Low | Very high concurrency |
| **Level 5** | Variable | **Minimal** | Real-time/embedded systems |

## Advanced Error Handling & Recovery System

Zipora implements a sophisticated error handling and recovery system providing production-ready error classification, automatic recovery strategies, and contextual error reporting.

### Core Error Management Features

- **🚨 Error Severity Classification**: Four-level severity system (WARNING, RECOVERABLE, CRITICAL, FATAL)
- **🔄 Automatic Recovery Strategies**: Memory reclamation, structure rebuilding, fallback algorithm switching
- **📊 Contextual Error Reporting**: Rich error context with metadata, thread IDs, timestamps
- **📈 Recovery Statistics**: Comprehensive tracking of recovery attempts, success rates, and performance metrics
- **🛡️ Verification Macros**: Production-ready assertion and verification system similar to TERARK_VERIFY
- **🧵 Thread-Safe Operations**: All error handling operations are thread-safe and lock-free

### Error Severity Levels

```rust
use zipora::error_recovery::{ErrorSeverity, ErrorRecoveryManager, ErrorContext, RecoveryStrategy};

// Four-level error classification system
pub enum ErrorSeverity {
    Warning,     // Minor issues that don't affect core functionality
    Recoverable, // Errors that can be automatically recovered from
    Critical,    // Serious errors requiring immediate attention but not fatal
    Fatal,       // Unrecoverable errors requiring immediate termination
}
```

### Recovery Strategies

The system provides sophisticated recovery mechanisms:

```rust
// Available recovery strategies
pub enum RecoveryStrategy {
    MemoryRecovery,      // Reclaim and reorganize memory
    StructureRebuild,    // Rebuild data structures from available data
    FallbackAlgorithm,   // Switch to fallback algorithms (e.g., AVX2 -> SSE2 -> scalar)
    RetryWithBackoff,    // Retry operation with exponential backoff
    CacheReset,          // Clear caches and reset state
    GracefulDegradation, // Reduce functionality gracefully
    NoRecovery,          // No recovery possible - propagate error
}
```

### Usage Examples

#### Basic Error Handling

```rust
use zipora::error_recovery::{ErrorRecoveryManager, ErrorSeverity, ErrorContext, RecoveryConfig};

// Create error recovery manager with custom configuration
let config = RecoveryConfig {
    max_recovery_attempts: 3,
    recovery_timeout: Duration::from_secs(10),
    enable_memory_recovery: true,
    enable_structure_rebuild: true,
    enable_fallback_algorithms: true,
    min_recovery_severity: ErrorSeverity::Recoverable,
    max_recovery_memory_mb: 256,
    ..Default::default()
};

let manager = ErrorRecoveryManager::with_config(config).unwrap();

// Handle error with automatic recovery
let context = ErrorContext::new("rank_select", "query")
    .with_metadata("index", "500")
    .with_metadata("operation_type", "rank1");

let error = ZiporaError::out_of_memory(1024);
let result = manager.handle_error(ErrorSeverity::Recoverable, context, &error);

match result {
    Ok(RecoveryResult::Success) => println!("Recovery successful"),
    Ok(RecoveryResult::PartialSuccess) => println!("Partial recovery, retry recommended"),
    Ok(RecoveryResult::Failed) => println!("Recovery failed"),
    Err(e) => println!("Recovery error: {}", e),
}
```

#### Memory Recovery Operations

```rust
// Attempt memory recovery and defragmentation
let result = manager.attempt_memory_recovery(&context);

// Structure rebuilding for corrupted data structures
let result = manager.attempt_structure_rebuild(&context);

// Algorithm fallback (e.g., SIMD -> scalar implementations)
let result = manager.attempt_fallback_algorithm(&context);
```

#### Verification Macros

Production-ready verification system:

```rust
use zipora::{zipora_verify, zipora_verify_eq, zipora_verify_lt};

// Basic verification (similar to TERARK_VERIFY)
zipora_verify!(index < size, "Index {} out of bounds for size {}", index, size);

// Comparison macros
zipora_verify_eq!(actual, expected);
zipora_verify_lt!(value, limit);

// Fatal error macro (similar to TERARK_DIE)  
if critical_condition {
    zipora_die!("Critical system failure: {}", error_message);
}
```

#### Recovery Statistics and Monitoring

```rust
// Get comprehensive recovery statistics
let stats = manager.get_stats();
println!("Recovery success rate: {:.1}%", stats.success_rate());
println!("Total recovery attempts: {}", stats.total_attempts.load(Ordering::Relaxed));
println!("Average recovery time: {}μs", stats.avg_recovery_time_us.load(Ordering::Relaxed));

// Get error history for analysis
let history = manager.get_error_history().unwrap();
for (severity, context, timestamp) in history {
    println!("Error: {:?} in {} at {:?}", severity, context.component, timestamp);
}
```

### Performance Characteristics

| Recovery Strategy | Time Complexity | Success Rate | Use Case |
|------------------|----------------|--------------|----------|
| **Memory Recovery** | O(n) memory scan | **95-98%** | Memory pool corruption, fragmentation |
| **Structure Rebuild** | O(n log n) | **90-95%** | Trie/hash map corruption, index rebuild |
| **Fallback Algorithm** | O(1) switch | **99%** | SIMD failure, hardware incompatibility |
| **Cache Reset** | O(1) | **100%** | Cache corruption, consistency issues |
| **Retry with Backoff** | Variable | **80-90%** | Transient failures, resource contention |

### Integration with Zipora Components

The error recovery system integrates seamlessly with all Zipora components:

- **Memory Pools**: Automatic defragmentation and leak detection
- **Tries and Hash Maps**: Structure rebuilding from underlying data
- **SIMD Operations**: Graceful fallback from AVX2 → SSE2 → scalar
- **Compression**: Algorithm switching and state recovery
- **Concurrency**: Thread-safe recovery across all concurrency levels

### Production Benefits

- **🔧 Automatic Recovery**: Reduces manual intervention and downtime
- **📊 Comprehensive Monitoring**: Detailed statistics for operational insights  
- **🛡️ Fail-Safe Design**: Multiple recovery strategies prevent total system failure
- **⚡ High Performance**: Lock-free operations with minimal overhead
- **🧵 Thread Safety**: Safe concurrent access across all recovery operations

## Cache Optimization Infrastructure

Zipora includes a comprehensive cache optimization framework that dramatically improves performance through intelligent memory layout and access patterns.

### Core Features

- **Cache-Line Alignment**: 64-byte alignment for x86_64, 128-byte for ARM64 to prevent false sharing
- **Hot/Cold Data Separation**: Intelligent placement of frequently vs. infrequently accessed data
- **Software Prefetching**: Cross-platform prefetch intrinsics (x86_64 and ARM64) with access pattern hints
- **NUMA-Aware Allocation**: Automatic NUMA node detection and memory allocation preferences
- **Access Pattern Analysis**: Tracking and optimization for Sequential, Random, Read-Heavy, Write-Heavy patterns

### Usage Examples

```rust
use zipora::memory::cache_layout::*;

// Configure cache-optimized allocation
let mut config = CacheLayoutConfig::new()
    .with_cache_line_size(64)
    .with_access_pattern(AccessPattern::Sequential)
    .with_prefetch_distance(128);

let allocator = CacheOptimizedAllocator::new(config);

// Cache-aligned allocation with prefetch hints
let ptr = allocator.allocate_aligned(1024, 64, true)?;

// Hot/cold data separation
let mut separator = HotColdSeparator::new(cache_config);
separator.insert(address, access_count);
let layout = separator.get_optimal_layout();
```

### Integration with Data Structures

All major data structures benefit from cache optimizations:

- **Hash Maps**: Cache-aware collision resolution with intelligent prefetching
- **Rank/Select**: Cache-line aligned structures with prefetch hints for sequential access
- **Memory Pools**: NUMA-aware allocation with hot/cold separation
- **Tries**: Cache-optimized node layout and navigation patterns
- **SIMD Memory Operations**: Cache-optimized copy/compare/search with prefetching
- **Cache Layout Optimization**: Hardware-aware allocation with hot/cold data separation

### Performance Impact

- **Memory Access**: 2-3x faster through reduced cache misses
- **Cache Optimization**: >95% hit rate for hot data, automatic cache hierarchy adaptation
- **SIMD Memory Operations**: 2-3x faster small copies (≤64 bytes), 1.5-2x faster medium copies
- **Sequential Processing**: 4-5x improvements with prefetch optimization
- **Multi-threaded**: Significant reduction in false sharing overhead
- **NUMA Systems**: 20-40% improvements through local allocation

## Rich Configuration APIs

Zipora provides a comprehensive configuration system that enables fine-grained control over data structures, algorithms, and performance characteristics. The system follows consistent patterns across all configuration types and offers multiple ways to create, validate, and manage configurations.

### Key Features

- **Trait-Based Design**: Consistent `Config` trait with validation, serialization, and preset methods
- **Builder Patterns**: Fluent configuration building with method chaining and compile-time validation
- **Environment Integration**: Automatic parsing from environment variables with custom prefixes
- **Preset Configurations**: Performance, Memory, Realtime, and Balanced presets for different use cases
- **JSON Serialization**: Save and load configurations with comprehensive serde support
- **Validation Framework**: Built-in validation with detailed error messages and suggestions
- **Type Safety**: Compile-time checks for configuration parameter ranges and combinations

### Configuration Types

The system provides rich configuration for all major components:

- **`NestLoudsTrieConfig`**: 20+ parameters for trie construction, compression, optimization, memory management
- **`MemoryConfig`**: Pool allocation strategies, NUMA settings, cache optimization, security features
- **`BlobStoreConfig`**: Compression algorithms, block sizes, caching, and I/O optimization
- **`CompressionConfig`**: Algorithm selection, compression levels, real-time constraints
- **`CacheConfig`**: Cache sizes, prefetching strategies, line size optimization
- **`SIMDConfig`**: Hardware acceleration settings (AVX2, BMI2, SIMD instruction sets)

### Usage Examples

#### Basic Configuration with Defaults

```rust
use zipora::config::*;

// Create with sensible defaults
let trie_config = NestLoudsTrieConfig::default();
let memory_config = MemoryConfig::default();
let blob_config = BlobStoreConfig::default();

// Validate configurations
assert!(trie_config.validate().is_ok());
assert!(memory_config.validate().is_ok());
assert!(blob_config.validate().is_ok());
```

#### Using Configuration Presets

```rust
// Choose preset based on your requirements
let perf_config = NestLoudsTrieConfig::performance_preset();  // Maximum performance
let mem_config = NestLoudsTrieConfig::memory_preset();        // Minimize memory usage
let rt_config = NestLoudsTrieConfig::realtime_preset();       // Predictable latency
let balanced_config = NestLoudsTrieConfig::balanced_preset(); // Balanced trade-offs

// Memory configuration presets
let secure_memory = MemoryConfig::performance_preset()
    .with_numa_awareness(true)
    .with_huge_pages(true)
    .with_cache_optimization(CacheOptimizationLevel::Maximum);
```

#### Builder Pattern Configuration

```rust
use zipora::config::nest_louds_trie::{CompressionAlgorithm, OptimizationFlags};

// Use fluent builder pattern for complex configurations
let custom_config = NestLoudsTrieConfig::builder()
    .nest_level(4)                           // Trie nesting depth
    .compression_level(8)                    // Balance of speed/compression
    .compression_algorithm(CompressionAlgorithm::Zstd(12))
    .max_fragment_length(2048)               // Memory vs. speed trade-off
    .min_fragment_length(16)                 // Minimum effective fragment size
    .enable_queue_compression(true)          // Enable queue compression
    .temp_directory("/tmp/zipora")           // Temporary file storage
    .initial_pool_size(128 * 1024 * 1024)    // 128MB initial pool
    .enable_statistics(true)                 // Performance monitoring
    .enable_profiling(false)                 // Disable profiling overhead
    .parallel_threads(8)                     // Use 8 threads for construction
    .optimization_flags(                     // Enable specific optimizations
        OptimizationFlags::ENABLE_FAST_SEARCH | 
        OptimizationFlags::ENABLE_SIMD_ACCELERATION |
        OptimizationFlags::USE_HUGEPAGES
    )
    .build()?;

// Verify the configuration
custom_config.validate()?;
```

#### Memory Configuration with Advanced Features

```rust
use zipora::config::memory::*;

let memory_config = MemoryConfig::builder()
    .allocation_strategy(AllocationStrategy::SecurePool)     // Secure memory management
    .initial_pool_size(256 * 1024 * 1024)                   // 256MB initial size
    .max_pool_size(2 * 1024 * 1024 * 1024)                  // 2GB maximum
    .growth_factor(1.5)                                      // 50% growth when needed
    .cache_optimization(CacheOptimizationLevel::Maximum)     // Full cache optimization
    .numa_config(NumaConfig {
        enable_numa_awareness: true,
        preferred_node: None,                                // Auto-select optimal node
        cross_node_threshold: 85,                           // 85% utilization threshold
    })
    .huge_page_config(HugePageConfig {
        enable_huge_pages: true,
        fallback_to_regular: true,                          // Graceful degradation
        size_threshold: 2 * 1024 * 1024,                    // Use huge pages for ≥2MB
    })
    .alignment(64)                                          // 64-byte cache line alignment
    .num_pools(16)                                          // 16 separate pools
    .enable_protection(true)                                // Memory protection features
    .enable_compaction(false)                               // Disable for real-time
    .build()?;
```

#### Environment Variable Integration

```rust
use std::env;

// Set configuration through environment variables
env::set_var("ZIPORA_TRIE_NEST_LEVEL", "5");
env::set_var("ZIPORA_TRIE_COMPRESSION_LEVEL", "12");
env::set_var("ZIPORA_TRIE_ENABLE_STATISTICS", "true");
env::set_var("ZIPORA_MEMORY_INITIAL_POOL_SIZE", "134217728"); // 128MB

// Load configuration from environment
let trie_config = NestLoudsTrieConfig::from_env()?;
let memory_config = MemoryConfig::from_env()?;

// Use custom prefix for environment variables
let custom_config = NestLoudsTrieConfig::from_env_with_prefix("CUSTOM_")?;

// Environment variables override defaults
assert_eq!(trie_config.nest_level, 5);
assert_eq!(trie_config.core_str_compression_level, 12);
assert!(trie_config.enable_statistics);
```

#### Configuration Persistence

```rust
use tempfile::tempdir;

// Save configuration to JSON file
let config = NestLoudsTrieConfig::performance_preset();
config.save_to_file("config/trie_performance.json")?;

// Load configuration from JSON file
let loaded_config = NestLoudsTrieConfig::load_from_file("config/trie_performance.json")?;
assert_eq!(config.nest_level, loaded_config.nest_level);

// Configuration validation happens automatically during loading
let invalid_config_result = NestLoudsTrieConfig::load_from_file("invalid_config.json");
assert!(invalid_config_result.is_err()); // Validation catches issues
```

#### Advanced Configuration Features

```rust
// Check and modify optimization flags
let mut config = NestLoudsTrieConfig::default();

// Check if specific optimizations are enabled
if config.has_optimization_flag(OptimizationFlags::ENABLE_SIMD_ACCELERATION) {
    println!("SIMD acceleration is enabled");
}

// Enable specific optimizations
config.set_optimization_flag(OptimizationFlags::USE_HUGEPAGES, true);
config.set_optimization_flag(OptimizationFlags::ENABLE_PARALLEL_CONSTRUCTION, true);

// Memory configuration with automatic detection
let memory_config = MemoryConfig::default();
let effective_cache_line_size = memory_config.effective_cache_line_size(); // Detects hardware
let effective_num_pools = memory_config.effective_num_pools();             // Based on CPU count

// Access configuration metadata
println!("Nest level range: 1-16, current: {}", config.nest_level);
println!("Config category: {}", config.category()); // "trie", "memory", etc.
```

### Configuration Validation

The configuration system provides comprehensive validation with detailed error messages:

```rust
// Create invalid configuration
let mut config = NestLoudsTrieConfig::default();
config.nest_level = 0;  // Invalid: must be 1-16
config.core_str_compression_level = 25;  // Invalid: must be 0-22
config.load_factor = 1.0;  // Invalid: must be between 0.0 and 1.0 (exclusive)

// Validation provides detailed feedback
match config.validate() {
    Ok(()) => println!("Configuration is valid"),
    Err(e) => {
        println!("Configuration validation failed: {}", e);
        // Output: "Configuration validation failed: nest level must be between 1 and 16; 
        //          compression level must be between 0 and 22; load factor must be between 0.0 and 1.0"
    }
}
```

### Configuration Integration

Configurations integrate seamlessly with Zipora components:

```rust
// Use configuration with data structures
let trie_config = NestLoudsTrieConfig::performance_preset();
let mut trie = LoudsTrie::with_config(trie_config)?;

// Memory configuration affects all allocations
let memory_config = MemoryConfig::builder()
    .cache_optimization(CacheOptimizationLevel::Maximum)
    .numa_awareness(true)
    .build()?;

let pool = SecureMemoryPool::new(memory_config)?;

// Blob store with custom compression
let blob_config = BlobStoreConfig::builder()
    .compression_algorithm(CompressionAlgorithm::Zstd)
    .compression_level(10)
    .block_size(128 * 1024)
    .build()?;

let store = ZstdBlobStore::with_config(blob_config)?;
```

### Performance Characteristics

The configuration system is designed for efficiency:

- **Creation**: ~1-5μs per configuration (builder pattern: ~10μs)
- **Validation**: ~0.1-0.5μs per configuration
- **JSON Serialization**: ~50-200μs per configuration
- **Environment Parsing**: ~100-500μs per configuration
- **Memory Overhead**: Minimal (configurations are value types)

### Best Practices

1. **Use Presets**: Start with presets and customize only specific parameters
2. **Validate Early**: Always validate configurations before use
3. **Environment Integration**: Use environment variables for deployment-specific settings
4. **Persist Configurations**: Save working configurations for reproducible builds
5. **Monitor Performance**: Enable statistics during development, disable in production
6. **Hardware Awareness**: Use automatic detection for cache line sizes and CPU features

## Quick Start

```toml
[dependencies]
zipora = "2.0.0"

# Or with optional features
zipora = { version = "2.0.0", features = ["lz4", "ffi"] }

# AVX-512 requires nightly Rust (experimental intrinsics)
zipora = { version = "2.0.0", features = ["avx512", "lz4", "ffi"] }  # nightly only
```

### Basic Usage

```rust
use zipora::*;

// High-performance vector
let mut vec = FastVec::new();
vec.push(42).unwrap();

// Zero-copy strings with SIMD
let s = FastStr::from_string("hello world");
println!("Hash: {:x}", s.hash_fast());

// Intelligent rank/select with automatic optimization
let mut bv = BitVector::new();
for i in 0..1000 { bv.push(i % 7 == 0).unwrap(); }
let adaptive_rs = AdaptiveRankSelect::new(bv).unwrap();
println!("Selected: {}", adaptive_rs.implementation_name());
let rank = adaptive_rs.rank1(500);

// Blob storage with compression
let mut store = MemoryBlobStore::new();
let id = store.put(b"Hello, World!").unwrap();

// High-performance offset-based blob storage with compression
let config = ZipOffsetBlobStoreConfig::performance_optimized();
let mut builder = ZipOffsetBlobStoreBuilder::with_config(config).unwrap();
builder.add_record(b"Compressed data").unwrap();
let store = builder.finish().unwrap();

// Trie-based blob storage with string key indexing
let config = TrieBlobStoreConfig::performance_optimized();
let mut trie_store = NestLoudsTrieBlobStore::<RankSelectInterleaved256>::new(config).unwrap();
let id = trie_store.put_with_key(b"user/profile/123", b"User profile data").unwrap();
let data = trie_store.get_by_key(b"user/profile/123").unwrap();

// Efficient prefix queries with trie indexing
let prefix_data = trie_store.get_by_prefix(b"user/").unwrap();
println!("Found {} entries with 'user/' prefix", prefix_data.len());

// ⚡ Unified Trie - Single implementation with strategy-based configuration
use zipora::fsa::{ZiporaTrie, ZiporaTrieConfig, Trie};

// Default Patricia trie behavior
let mut trie = ZiporaTrie::new(); // Same API as before!
trie.insert(b"hello").unwrap();
assert!(trie.contains(b"hello"));

// String-specialized trie (formerly CritBitTrie)
let mut string_trie = ZiporaTrie::with_config(ZiporaTrieConfig::string_specialized());
string_trie.insert(b"hello").unwrap();
string_trie.insert(b"help").unwrap();
string_trie.insert(b"world").unwrap();
assert!(string_trie.contains(b"hello"));
assert!(!string_trie.contains(b"he")); // Automatic compression

// Space-optimized trie (formerly LOUDS/NestedLouds)
let mut compact_trie = ZiporaTrie::with_config(ZiporaTrieConfig::space_optimized());
compact_trie.insert(b"efficient").unwrap();
compact_trie.insert(b"effective").unwrap();
let stats = compact_trie.stats();
println!("Memory usage: {} bytes, compression ratio: {:.1}%",
         stats.memory_usage, stats.compression_ratio * 100.0);

// High-performance concurrent trie (formerly DoubleArrayTrie)
let pool = std::sync::Arc::new(SecureMemoryPool::new(SecurePoolConfig::default()).unwrap());
let mut concurrent_trie = ZiporaTrie::with_config(
    ZiporaTrieConfig::concurrent_high_performance(pool)
);
concurrent_trie.insert(b"concurrent").unwrap();

// ⚡ Unified Hash Map - Single implementation with strategy-based configuration
use zipora::hash_map::{ZiporaHashMap, ZiporaHashMapConfig};

// Default high-performance hash map
let mut map = ZiporaHashMap::new(); // Same API as before!
map.insert("key", "value").unwrap();

// String-optimized configuration (formerly StringOptimizedHashMap)
let mut string_map = ZiporaHashMap::with_config(ZiporaHashMapConfig::string_optimized());
string_map.insert("interned", 42).unwrap();
let arena_stats = string_map.stats(); // Unified stats API

// Small inline configuration (formerly SmallHashMap)
let mut small_map = ZiporaHashMap::with_config(ZiporaHashMapConfig::small_inline(4));
small_map.insert("inline", 1).unwrap();

// Cache-optimized configuration with advanced features
let mut cache_map = ZiporaHashMap::with_config(ZiporaHashMapConfig::cache_optimized());
cache_map.insert("cache", "optimized").unwrap();
let metrics = cache_map.cache_metrics(); // Same powerful API!
println!("Cache hit ratio: {:.2}%", metrics.hit_ratio() * 100.0);

// Advanced custom configuration (replaces AdvancedHashMap)
use zipora::hash_map::{HashStrategy, StorageStrategy, OptimizationStrategy};
let config = ZiporaHashMapConfig {
    hash_strategy: HashStrategy::RobinHood {
        max_probe_distance: 64,
        variance_reduction: true,
        backward_shift: true,
    },
    storage_strategy: StorageStrategy::CacheOptimized {
        cache_line_size: 64,
        numa_aware: true,
        huge_pages: false,
    },
    optimization_strategy: OptimizationStrategy::HighPerformance {
        simd_enabled: true,
        cache_optimized: true,
        prefetch_enabled: true,
        numa_aware: true,
    },
    ..ZiporaHashMapConfig::default()
};
let mut advanced_map = ZiporaHashMap::with_config(config);
advanced_map.insert("advanced", "unified configuration").unwrap();

// Advanced string arena with offset-based addressing (integrated into ZiporaHashMap)
let mut string_map = ZiporaHashMap::with_config(ZiporaHashMapConfig::string_optimized());
string_map.insert("shared string", "value1").unwrap();
string_map.insert("shared string", "value2").unwrap(); // Automatic deduplication
let stats = string_map.stats();
println!("Deduplication ratio: {:.2}%", stats.deduplication_ratio * 100.0);

// Entropy coding
let encoder = HuffmanEncoder::new(b"sample data").unwrap();
let compressed = encoder.encode(b"sample data").unwrap();

// LRU Page Cache for blob operations
use zipora::cache::{LruPageCache, PageCacheConfig, CachedBlobStore};

let cache_config = PageCacheConfig::performance_optimized()
    .with_capacity(256 * 1024 * 1024)  // 256MB cache
    .with_shards(8);                   // 8 shards for reduced contention

let cache = LruPageCache::new(cache_config).unwrap();
let file_id = cache.register_file(1).unwrap();

// Cache-aware blob store
let blob_store = MemoryBlobStore::new();
let cached_store = CachedBlobStore::new(blob_store, cache_config).unwrap();
```

## Version-Based Synchronization for FSA and Tries

Zipora includes advanced token and version sequence management for safe concurrent access to Finite State Automata and Trie data structures, based on research from high-performance concurrent data structure patterns.

### Key Features

- **Graduated Concurrency Control**: Five levels from read-only to full multi-writer scenarios
- **Token-Based Access Control**: Type-safe reader/writer tokens with automatic RAII lifecycle
- **Version Sequence Management**: Atomic version counters with consistency validation
- **Thread-Local Token Caching**: High-performance token reuse with zero allocation overhead
- **Memory Safety**: Zero unsafe operations in public APIs

### Usage Examples

```rust
use zipora::fsa::{ConcurrentPatriciaTrie, ConcurrentTrieConfig, ConcurrencyLevel};
use zipora::fsa::{TokenManager, with_reader_token, with_writer_token};

// Create concurrent Patricia trie with multi-reader support
let config = ConcurrentTrieConfig::new(ConcurrencyLevel::OneWriteMultiRead);
let mut trie = ConcurrentPatriciaTrie::new(config).unwrap();

// Insert with automatic token management
trie.insert(b"hello", 42).unwrap();
trie.insert(b"world", 84).unwrap();

// Concurrent lookups from multiple threads
let value = trie.get(b"hello").unwrap();
assert_eq!(value, Some(42));

// Advanced operations with explicit token control
trie.with_writer_token(|trie, token| {
    trie.insert_with_token(b"advanced", 168, token)?;
    Ok(())
}).unwrap();

// Direct token management for fine-grained control
let token_manager = TokenManager::new(ConcurrencyLevel::MultiWriteMultiRead);

with_reader_token(&token_manager, |token| {
    // Use token for read operations
    assert!(token.is_valid());
    Ok(())
}).unwrap();

with_writer_token(&token_manager, |token| {
    // Use token for write operations
    assert!(token.is_valid());
    Ok(())
}).unwrap();
```

### Concurrency Levels

| Level | Description | Use Case | Performance |
|-------|-------------|----------|-------------|
| **Level 0** | `NoWriteReadOnly` | Static data, no writers | **Zero overhead** |
| **Level 1** | `SingleThreadStrict` | Single-threaded apps | **Zero overhead** |
| **Level 2** | `SingleThreadShared` | Single-threaded with token validation | **Minimal overhead** |
| **Level 3** | `OneWriteMultiRead` | Read-heavy workloads | **Excellent reader scaling** |
| **Level 4** | `MultiWriteMultiRead` | High-contention scenarios | **Full concurrency** |

### Performance Characteristics

- **Single-threaded overhead**: < 5% compared to no synchronization
- **Multi-reader scaling**: Linear up to 8+ cores
- **Writer throughput**: 90%+ of single-threaded for OneWriteMultiRead
- **Token cache hit rate**: 80%+ for repeated operations
- **Memory overhead**: < 10% additional memory usage

## Core Data Structures

### High-Performance Containers

Zipora includes specialized containers designed for memory efficiency and performance:

```rust
use zipora::{FastVec, FastStr, ValVec32, SmallMap, FixedCircularQueue, 
            AutoGrowCircularQueue, UintVector, IntVec, FixedLenStrVec, SortableStrVec,
            LruMap, ConcurrentLruMap,
            // Advanced String Containers
            AdvancedStringVec, AdvancedStringConfig, BitPackedStringVec32, BitPackedStringVec64, 
            BitPackedConfig};

// High-performance vector operations
let mut vec = FastVec::new();
vec.push(42).unwrap();

// Zero-copy string with SIMD hashing
let s = FastStr::from_string("hello world");
println!("Hash: {:x}", s.hash_fast());

// 32-bit indexed vectors - 50% memory reduction with golden ratio growth strategy
// Optimized with golden ratio growth pattern (103/64 ≈ 1.609375) for memory efficiency
let mut vec32 = ValVec32::<u64>::new();
vec32.push(42).unwrap();  // Near-identical performance to std::Vec
assert_eq!(vec32.get(0), Some(&42));
// Performance: Golden ratio growth provides optimal memory efficiency!

// Small maps - 90% faster than HashMap for ≤8 elements with cache optimizations
let mut small_map = SmallMap::<i32, String>::new();
small_map.insert(1, "one".to_string()).unwrap();
small_map.insert(2, "two".to_string()).unwrap();
// Performance: 709K+ ops/sec cache-friendly access in release builds

// Fixed-size circular queue - lock-free, const generic size
let mut queue = FixedCircularQueue::<i32, 8>::new();
queue.push_back(1).unwrap();
queue.push_back(2).unwrap();
assert_eq!(queue.pop_front(), Some(1));

// Ultra-fast auto-growing circular queue - 1.54x faster than VecDeque (optimized)
let mut auto_queue = AutoGrowCircularQueue::<String>::new();
auto_queue.push_back("hello".to_string()).unwrap();
auto_queue.push_back("world".to_string()).unwrap();
// Performance: 54% faster than std::collections::VecDeque with optimization patterns

// Compressed integer storage - 60-80% space reduction
let mut uint_vec = UintVector::new();
uint_vec.push(42).unwrap();
uint_vec.push(1000).unwrap();
println!("Compression ratio: {:.2}", uint_vec.compression_ratio());

// Advanced bit-packed integer storage with variable bit-width
let values: Vec<u32> = (1000..2000).collect();
let compressed = IntVec::<u32>::from_slice(&values).unwrap();
println!("IntVec compression ratio: {:.3}", compressed.compression_ratio());
assert!(compressed.compression_ratio() < 0.4); // >60% compression

// Generic support for all integer types
let u64_values: Vec<u64> = (0..1000).map(|i| i * 1000).collect();
let u64_compressed = IntVec::<u64>::from_slice(&u64_values).unwrap();

// Hardware-accelerated decompression
for i in 0..1000 {
    assert_eq!(u64_compressed.get(i), Some(u64_values[i]));
}

// Fixed-length strings - 59.6% memory savings vs Vec<String> (optimized)
let mut fixed_str_vec = FixedLenStrVec::<32>::new();
fixed_str_vec.push("hello").unwrap();
fixed_str_vec.push("world").unwrap();
assert_eq!(fixed_str_vec.get(0), Some("hello"));
// Arena-based storage with bit-packed indices for zero-copy access

// Arena-based string sorting with algorithm selection
let mut sortable = SortableStrVec::new();
sortable.push_str("cherry").unwrap();
sortable.push_str("apple").unwrap();
sortable.push_str("banana").unwrap();
sortable.sort_lexicographic().unwrap(); // Intelligent algorithm selection (comparison vs radix)

// 🚀 Advanced String Containers - Memory-efficient encoding strategies

// Advanced string vector with 3-level compression strategy
let config = AdvancedStringConfig::performance_optimized();
let mut advanced_vec = AdvancedStringVec::with_config(config);
advanced_vec.push("hello world").unwrap();
advanced_vec.push("hello rust").unwrap();   // Prefix deduplication
advanced_vec.push("hello").unwrap();        // Overlap detection

// Enable aggressive compression for maximum space efficiency
advanced_vec.enable_aggressive_compression(true);
let stats = advanced_vec.stats();
println!("Compression ratio: {:.1}%", stats.compression_ratio * 100.0);
println!("Space saved: {:.1}%", (1.0 - stats.compression_ratio) * 100.0);

// Bit-packed string vectors with template-based offset types
// 32-bit offsets (4GB capacity) - optimal for most use cases
let mut bit_packed_vec32: BitPackedStringVec32 = BitPackedStringVec::new();
bit_packed_vec32.push("memory efficient").unwrap();
bit_packed_vec32.push("hardware accelerated").unwrap();

// 64-bit offsets (unlimited capacity) - for very large datasets
let config = BitPackedConfig::large_dataset();
let mut bit_packed_vec64: BitPackedStringVec64 = BitPackedStringVec::with_config(config);
bit_packed_vec64.push("unlimited capacity").unwrap();

// Template-based optimization with hardware acceleration
let (our_bytes, vec_string_bytes, ratio) = bit_packed_vec32.memory_info();
println!("Memory efficiency: {:.1}% savings", (1.0 - ratio) * 100.0);
println!("Hardware acceleration: {}", bit_packed_vec32.has_hardware_acceleration());

// SIMD-accelerated search operations
#[cfg(feature = "simd")]
{
    if let Some(index) = bit_packed_vec32.find_simd("memory efficient") {
        println!("Found at index: {}", index);
    }
}

// LRU Cache Containers - High-performance caching with eviction policies
let mut cache = LruMap::new(256).unwrap(); // Capacity of 256
cache.put("key1", "value1".to_string()).unwrap();
cache.put("key2", "value2".to_string()).unwrap();
assert_eq!(cache.get(&"key1"), Some("value1".to_string()));

// Concurrent LRU map with sharding for thread safety
let cache = ConcurrentLruMap::new(1024, 8).unwrap(); // 1024 capacity, 8 shards
cache.put("key1", "value1".to_string()).unwrap();
cache.put("key2", "value2".to_string()).unwrap();
assert_eq!(cache.get(&"key1"), Some("value1".to_string()));
```

## LRU Cache Containers

Zipora provides high-performance LRU (Least Recently Used) cache implementations with built-in eviction policies, statistics tracking, and concurrent access support:

### Single-Threaded LRU Map

```rust
use zipora::containers::{LruMap, LruMapConfig, EvictionCallback};

// Basic LRU map with default configuration
let mut cache = LruMap::new(256).unwrap(); // Capacity of 256

// Insert key-value pairs with automatic eviction
cache.put("key1", "value1".to_string()).unwrap();
cache.put("key2", "value2".to_string()).unwrap();

// Access updates LRU order
assert_eq!(cache.get(&"key1"), Some("value1".to_string()));

// Advanced configuration options
let config = LruMapConfig::performance_optimized()
    .with_capacity(1024)
    .with_statistics(true);
let cache = LruMap::with_config(config).unwrap();

// Eviction callbacks for custom logic
struct LoggingCallback;
impl EvictionCallback<String, String> for LoggingCallback {
    fn on_evict(&self, key: &String, value: &String) {
        println!("Evicted: {} => {}", key, value);
    }
}

let cache = LruMap::with_eviction_callback(256, LoggingCallback).unwrap();

// Statistics and performance monitoring
let stats = cache.stats();
println!("Hit ratio: {:.2}%", stats.hit_ratio() * 100.0);
println!("Entry count: {}", stats.entry_count.load(Ordering::Relaxed));
```

### Concurrent LRU Map

```rust
use zipora::containers::{ConcurrentLruMap, ConcurrentLruMapConfig, LoadBalancingStrategy};

// Thread-safe LRU map with sharding
let cache = ConcurrentLruMap::new(1024, 8).unwrap(); // 1024 capacity, 8 shards

// Concurrent operations from multiple threads
cache.put("key1", "value1".to_string()).unwrap();
cache.put("key2", "value2".to_string()).unwrap();
assert_eq!(cache.get(&"key1"), Some("value1".to_string()));

// Advanced configuration with load balancing strategies
let config = ConcurrentLruMapConfig::performance_optimized()
    .with_load_balancing(LoadBalancingStrategy::Hash);
let cache = ConcurrentLruMap::with_config(config).unwrap();

// Statistics aggregated across all shards
let stats = cache.stats();
println!("Total entries: {}", stats.total_entries());
println!("Hit ratio: {:.2}%", stats.hit_ratio() * 100.0);
println!("Load balance ratio: {:.2}", stats.load_balance_ratio());

// Per-shard statistics
let shard_sizes = cache.shard_sizes();
println!("Shard distribution: {:?}", shard_sizes);
```

### LRU Cache Features

- **O(1) Operations**: Get, put, and remove operations in constant time
- **Generic Support**: Works with any `Hash + Eq` key and value types  
- **Automatic Eviction**: LRU-based eviction when capacity is exceeded
- **Statistics Tracking**: Hit/miss ratios, eviction counts, memory usage
- **Eviction Callbacks**: Custom logic when entries are evicted
- **Thread Safety**: Concurrent variant with sharding for reduced contention
- **Load Balancing**: Multiple strategies for optimal shard distribution
- **Memory Efficient**: Intrusive linked list design minimizes overhead

### Container Performance Summary

| Container | Memory Reduction | Performance Gain | Use Case |
|-----------|------------------|------------------|----------|
| **ValVec32<T>** | **50% memory reduction** | **Golden ratio growth (103/64), near-parity performance** | **Large collections on 64-bit systems** |
| **SmallMap<K,V>** | No heap allocation | **90% faster + cache optimized** | **≤8 key-value pairs - 709K+ ops/sec** |
| **FixedCircularQueue** | Zero allocation | 20-30% faster | Lock-free ring buffers |
| **AutoGrowCircularQueue** | Cache-aligned | **54% faster** | **Ultra-fast vs VecDeque (optimized)** |
| **UintVector** | **68.7% space reduction** | <20% speed penalty | Compressed integers (optimized) |
| **IntVec<T>** | **96.9% space reduction** | **Hardware-accelerated** | **Generic bit-packed storage with BMI2/SIMD** |
| **FixedLenStrVec** | **59.6% memory reduction (optimized)** | **Zero-copy access** | **Arena-based fixed strings** |
| **SortableStrVec** | Arena allocation | **Intelligent algorithm selection** | **String collections with optimization patterns** |
| **🚀 AdvancedStringVec** | **60-80% space reduction** | **3-level compression strategy** | **High-compression string storage with deduplication** |
| **🚀 BitPackedStringVec32** | **50-70% memory reduction** | **Template-based with BMI2 acceleration** | **Hardware-accelerated string storage (4GB capacity)** |
| **🚀 BitPackedStringVec64** | **40-60% memory reduction** | **Unlimited capacity with SIMD optimization** | **Large-scale string datasets with hardware acceleration** |
| **LruMap<K,V>** | **Intrusive linked list** | **O(1) operations** | **Single-threaded caching with eviction policies** |
| **ConcurrentLruMap<K,V>** | **Sharded architecture** | **Reduced contention** | **Multi-threaded caching with load balancing** |

## Specialized Hash Maps

Zipora provides a **unified hash map implementation** with strategy-based configuration for advanced features including cache locality optimizations, sophisticated collision resolution algorithms, and memory-efficient string arena management:

```rust
use zipora::hash_map::{ZiporaHashMap, ZiporaHashMapConfig, HashStrategy, StorageStrategy};

// Default high-performance hash map - same API as before!
let mut map = ZiporaHashMap::new();
map.insert("key", "value").unwrap();
// Features: Optimized for general-purpose use, excellent lookup performance

// String-optimized configuration - memory efficient for string keys
let mut string_map = ZiporaHashMap::with_config(ZiporaHashMapConfig::string_optimized());
string_map.insert("interned", 42).unwrap();
// Features: String interning, prefix caching, SIMD acceleration, arena management
// Best for: Applications with many duplicate string keys

// Small inline configuration - zero allocations for small collections
let mut small_map = ZiporaHashMap::with_config(ZiporaHashMapConfig::small_inline(4));
small_map.insert("inline", 1).unwrap();
// Features: Inline storage for ≤N elements, automatic heap fallback
// Best for: Small collections, zero-allocation scenarios

// Cache-optimized configuration - NUMA awareness and prefetching
let mut cache_map = ZiporaHashMap::with_config(ZiporaHashMapConfig::cache_optimized());
cache_map.insert("cache", "optimized").unwrap();
// Features: Cache-line alignment, NUMA awareness, hot/cold separation
// Best for: High-performance applications with cache-sensitive workloads

// Custom advanced configuration - full control over strategies
let config = ZiporaHashMapConfig {
    hash_strategy: HashStrategy::RobinHood {
        max_probe_distance: 64,
        variance_reduction: true,
        backward_shift: true,
    },
    storage_strategy: StorageStrategy::CacheOptimized {
        cache_line_size: 64,
        numa_aware: true,
        huge_pages: false,
    },
    load_factor: 0.75,
    ..ZiporaHashMapConfig::default()
};
let mut advanced_map = ZiporaHashMap::with_config(config);
advanced_map.insert("advanced", "unified configuration").unwrap();
```

### Hash Map Performance Comparison

Based on comprehensive benchmarks comparing all hash map implementations:

| Hash Map Type | Insertion Performance | Lookup Performance | Best Use Case |
|---------------|----------------------|--------------------|--------------| 
| **std::HashMap** | **73-104 Melem/s** ⭐ | 91-104 Melem/s | Standard Rust operations |
| **GoldHashMap** | 71-77 Melem/s | **241-342 Melem/s** ⭐ | **Lookup-heavy workloads** |
| **GoldenRatioHashMap** | 55-70 Melem/s | 110-322 Melem/s | **Memory-efficient growth** |
| **StringOptimizedHashMap** | 5.6-6.0 Melem/s* | Variable | **String key deduplication** |
| **SmallHashMap<T,V,N>** | Variable | Variable | **≤N elements, zero allocation** |
| **AdvancedHashMap** | 60-80 Melem/s | 200-280 Melem/s | **Sophisticated collision resolution** |
| **CacheOptimizedHashMap** | 45-65 Melem/s | 180-250 Melem/s | **Cache-line aligned with NUMA awareness** |

*StringOptimizedHashMap trades speed for memory efficiency through string interning

### Key Performance Insights

- **GoldHashMap excels at lookups** with 2-3x better performance than std::HashMap
- **GoldenRatioHashMap provides the best balance** of memory efficiency and performance
- **Capacity optimizations improved GoldHashMap by up to 60%** in benchmarks
- **StringOptimizedHashMap reduces memory usage** at the cost of insertion speed
- **SmallHashMap eliminates allocations** for small collections
- **AdvancedHashMap provides sophisticated collision handling** with Robin Hood hashing, chaining, and Hopscotch algorithms
- **CacheOptimizedHashMap delivers cache-aware performance** with prefetching, NUMA awareness, and hot/cold data separation
- **Advanced string arena management** enables efficient memory usage with offset-based addressing and deduplication

## Blob Storage Systems

### Trie-Based String Indexing (NestLoudsTrieBlobStore)

```rust
use zipora::{NestLoudsTrieBlobStore, TrieBlobStoreConfig, TrieBlobStoreConfigBuilder,
            RankSelectInterleaved256, BlobStore, IterableBlobStore, BatchBlobStore};

// High-performance trie-based blob storage with string key indexing
let config = TrieBlobStoreConfig::performance_optimized();
let mut store = NestLoudsTrieBlobStore::<RankSelectInterleaved256>::new(config).unwrap();

// Store data with string keys - automatic prefix compression
let id1 = store.put_with_key(b"user/john/profile", b"John's profile data").unwrap();
let id2 = store.put_with_key(b"user/john/settings", b"John's settings").unwrap();
let id3 = store.put_with_key(b"user/jane/profile", b"Jane's profile data").unwrap();

// Retrieve by key - O(|key|) trie traversal with compressed storage
let profile = store.get_by_key(b"user/john/profile").unwrap();
assert_eq!(profile, b"John's profile data");

// Efficient prefix-based queries leveraging trie structure
let john_data = store.get_by_prefix(b"user/john/").unwrap();
assert_eq!(john_data.len(), 2);

// Traditional blob store operations also supported
let data = store.get(id1).unwrap();
assert_eq!(data, b"John's profile data");

// Configuration variants for different use cases
let memory_config = TrieBlobStoreConfig::memory_optimized();
let security_config = TrieBlobStoreConfig::security_optimized();

// Custom configuration with builder pattern
let custom_config = TrieBlobStoreConfig::builder()
    .key_compression(true)
    .batch_optimization(true)
    .key_cache_size(2048)
    .statistics(true)
    .build().unwrap();

// Builder pattern for efficient bulk construction
let mut builder = NestLoudsTrieBlobStore::<RankSelectInterleaved256>::builder(config).unwrap();
builder.add(b"key1", b"data1").unwrap();
builder.add(b"key2", b"data2").unwrap();
builder.add(b"key3", b"data3").unwrap();
let optimized_store = builder.finish().unwrap();

// Batch operations for improved performance
let key_value_pairs = vec![
    (b"batch/key1".to_vec(), b"batch data 1".to_vec()),
    (b"batch/key2".to_vec(), b"batch data 2".to_vec()),
];
let batch_ids = store.put_batch_with_keys(key_value_pairs).unwrap();

// Advanced features
let all_keys = store.keys().unwrap(); // Get all stored keys
let prefix_keys = store.keys_with_prefix(b"user/").unwrap(); // Keys with prefix
let key_count = store.key_count(); // Number of unique keys
let trie_stats = store.trie_stats(); // Detailed trie statistics

// Comprehensive statistics and performance monitoring
let stats = store.stats();
println!("Blob count: {}", stats.blob_count);
println!("Cache hit ratio: {:.2}%", stats.cache_hit_ratio * 100.0);

let trie_stats = store.trie_stats();
println!("Key count: {}", trie_stats.key_count);
println!("Trie compression ratio: {:.2}%", trie_stats.trie_space_saved_percent());
```

### Offset-Based Compressed Storage (ZipOffsetBlobStore)

```rust
use zipora::{ZipOffsetBlobStore, ZipOffsetBlobStoreBuilder, ZipOffsetBlobStoreConfig,
            SortedUintVec, SortedUintVecBuilder};

// High-performance offset-based compressed blob storage
let config = ZipOffsetBlobStoreConfig::performance_optimized();
let mut builder = ZipOffsetBlobStoreBuilder::with_config(config).unwrap();

// Add records with automatic compression and checksumming
builder.add_record(b"First record data").unwrap();
builder.add_record(b"Second record data").unwrap();
builder.add_record(b"Third record data").unwrap();

// Build the final store with optimized layout
let store = builder.finish().unwrap();

// Template-based record retrieval with const generics
let record = store.get(0).unwrap(); // O(1) access to any record
let size = store.size(1).unwrap().unwrap(); // Compressed size information

// Block-based delta compression for sorted integer sequences
let mut uint_builder = SortedUintVecBuilder::new();
uint_builder.push(1000).unwrap();
uint_builder.push(1010).unwrap(); // Small delta = efficient compression
uint_builder.push(1025).unwrap();

let compressed_uints = uint_builder.finish().unwrap();
let value = compressed_uints.get(1).unwrap(); // BMI2-accelerated bit extraction

// File I/O with 128-byte aligned headers
store.save_to_file("compressed.zob").unwrap();
let loaded_store = ZipOffsetBlobStore::load_from_file("compressed.zob").unwrap();

// Statistics and compression analysis
let stats = builder.stats();
println!("Compression ratio: {:.2}", stats.compression_ratio());
println!("Space saved: {:.1}%", stats.space_saved_percent());
```

### LRU Page Cache - Sophisticated Caching Layer

```rust
use zipora::cache::{LruPageCache, PageCacheConfig, CachedBlobStore, CacheBuffer};
use zipora::blob_store::MemoryBlobStore;

// High-performance page cache with optimal configuration
let config = PageCacheConfig::performance_optimized()
    .with_capacity(256 * 1024 * 1024)  // 256MB cache
    .with_shards(8)                    // 8 shards for reduced contention
    .with_huge_pages(true);            // Use 2MB huge pages

let cache = LruPageCache::new(config).unwrap();

// Register files for caching
let file_id = cache.register_file(1).unwrap();

// Direct cache operations
let buffer = cache.read(file_id, 0, 4096).unwrap();  // Read 4KB page
cache.prefetch(file_id, 4096, 16384).unwrap();       // Prefetch 16KB

// Batch operations for high throughput
let requests = vec![
    (file_id, 0, 4096),
    (file_id, 4096, 4096),
    (file_id, 8192, 4096)
];
let results = cache.read_batch(requests).unwrap();

// Cache-aware blob store integration
let blob_store = MemoryBlobStore::new();
let mut cached_store = CachedBlobStore::new(blob_store, config).unwrap();

let id = cached_store.put(b"Cached data").unwrap();
let data = cached_store.get(id).unwrap();  // Automatically cached
let stats = cached_store.cache_stats();    // Performance metrics

println!("Hit ratio: {:.2}%", stats.hit_ratio * 100.0);
```

### Blob Storage Performance Summary

| Storage Type | Memory Efficiency | Throughput | Features | Best Use Case |
|--------------|------------------|------------|----------|---------------|
| **NestLoudsTrieBlobStore** | **Trie compression + blob compression** | **O(key) access + O(1) blob retrieval** | **String indexing, prefix queries** | **Hierarchical data, key-value stores** |
| **ZipOffsetBlobStore** | **Block-based delta compression** | **O(1) offset-based access** | **Template optimization, ZSTD** | **Large datasets, streaming access** |
| **LRU Page Cache** | **Page-aligned allocation** | **Reduced contention** | **Multi-shard architecture** | **High-concurrency access** |

## Memory Management

### Secure Memory Management

```rust
use zipora::{SecureMemoryPool, SecurePoolConfig, BumpAllocator, PooledVec};

// Production-ready secure memory pools
let config = SecurePoolConfig::small_secure();
let pool = SecureMemoryPool::new(config).unwrap();

// RAII-based allocation - automatic cleanup, no manual deallocation
let ptr = pool.allocate().unwrap();
println!("Allocated {} bytes safely", ptr.size());

// Use memory through safe interface
let slice = ptr.as_slice();
// ptr automatically freed on drop - no use-after-free possible!

// Global thread-safe pools for common sizes
let small_ptr = zipora::get_global_pool_for_size(1024).allocate().unwrap();

// Bump allocator for sequential allocation  
let bump = BumpAllocator::new(1024 * 1024).unwrap();
let ptr = bump.alloc::<u64>().unwrap();

// Pooled containers with automatic pool allocation
let mut pooled_vec = PooledVec::<i32>::new().unwrap();
pooled_vec.push(42).unwrap();

// Linux hugepage support for large datasets
#[cfg(target_os = "linux")]
{
    use zipora::HugePage;
    let hugepage = HugePage::new_2mb(2 * 1024 * 1024).unwrap();
}
```

### Advanced Memory Pool Variants

High-Performance Memory Management - Zipora provides 4 specialized memory pool variants with cutting-edge optimizations, lock-free allocation, thread-local caching, and persistent storage capabilities:

#### Lock-Free Memory Pool

```rust
use zipora::memory::{LockFreeMemoryPool, LockFreePoolConfig, BackoffStrategy};

// High-performance concurrent allocation without locks
let config = LockFreePoolConfig::high_performance();
let pool = LockFreeMemoryPool::new(config).unwrap();

// Concurrent allocation from multiple threads
let alloc = pool.allocate(1024).unwrap();
let ptr = alloc.as_ptr();

// Lock-free deallocation with CAS retry loops
drop(alloc); // Automatic deallocation

// Advanced configuration options
let config = LockFreePoolConfig {
    memory_size: 256 * 1024 * 1024, // 256MB backing memory
    enable_stats: true,
    max_cas_retries: 10000,
    backoff_strategy: BackoffStrategy::Exponential { max_delay_us: 100 },
};

// Performance statistics
if let Some(stats) = pool.stats() {
    println!("CAS contention ratio: {:.2}%", stats.contention_ratio() * 100.0);
    println!("Allocation rate: {:.0} allocs/sec", stats.allocation_rate());
}
```

#### Thread-Local Memory Pool

```rust
use zipora::memory::{ThreadLocalMemoryPool, ThreadLocalPoolConfig};

// Per-thread allocation caches for zero contention
let config = ThreadLocalPoolConfig::high_performance();
let pool = ThreadLocalMemoryPool::new(config).unwrap();

// Hot area allocation - sequential allocation from thread-local arena
let alloc = pool.allocate(64).unwrap();

// Thread-local free list caching
let cached_alloc = pool.allocate(64).unwrap(); // Likely cache hit

// Configuration for different scenarios
let config = ThreadLocalPoolConfig {
    arena_size: 8 * 1024 * 1024, // 8MB per thread
    max_threads: 1024,
    sync_threshold: 1024 * 1024, // 1MB lazy sync threshold
    use_secure_memory: false, // Disable for max performance
    ..ThreadLocalPoolConfig::default()
};

// Performance monitoring
if let Some(stats) = pool.stats() {
    println!("Cache hit ratio: {:.1}%", stats.hit_ratio() * 100.0);
    println!("Locality score: {:.2}", stats.locality_score());
}
```

#### Fixed Capacity Memory Pool

```rust
use zipora::memory::{FixedCapacityMemoryPool, FixedCapacityPoolConfig};

// Bounded memory pool for real-time systems
let config = FixedCapacityPoolConfig::realtime();
let pool = FixedCapacityMemoryPool::new(config).unwrap();

// Guaranteed allocation within capacity
let alloc = pool.allocate(1024).unwrap();

// Capacity management
println!("Total capacity: {} bytes", pool.total_capacity());
println!("Available: {} bytes", pool.available_capacity());
assert!(pool.has_capacity(2048));

// Configuration for different use cases
let config = FixedCapacityPoolConfig {
    max_block_size: 8192,
    total_blocks: 5000,
    alignment: 64, // Cache line aligned
    enable_stats: false, // Minimize overhead
    eager_allocation: true, // Pre-allocate all memory
    secure_clear: true, // Zero memory on deallocation
};

// Real-time performance monitoring
if let Some(stats) = pool.stats() {
    println!("Utilization: {:.1}%", stats.utilization_percent());
    println!("Success rate: {:.3}", stats.success_rate());
    assert!(!stats.is_at_capacity(pool.total_capacity()));
}
```

#### Memory-Mapped Vectors

```rust
use zipora::memory::{MmapVec, MmapVecConfig, MmapVecConfigBuilder, MmapVecStats};

// Persistent vector backed by memory-mapped file
let config = MmapVecConfig::large_dataset();
let mut vec = MmapVec::<u64>::create("data.mmap", config).unwrap();

// Standard vector operations with persistence
vec.push(42).unwrap();
vec.push(84).unwrap();
assert_eq!(vec.len(), 2);
assert_eq!(vec.get(0), Some(&42));

// Automatic growth and persistence
vec.reserve(1_000_000).unwrap(); // Reserve for 1M elements
for i in 0..1000 {
    vec.push(i).unwrap();
}

// Cross-process data sharing
vec.sync().unwrap(); // Force sync to disk

// Configuration presets for different use cases
let performance_config = MmapVecConfig::performance_optimized(); // Golden ratio growth
let memory_config = MmapVecConfig::memory_optimized();           // Conservative growth
let realtime_config = MmapVecConfig::realtime();                 // Predictable performance

// Builder pattern for custom configurations
let config = MmapVecConfig::builder()
    .initial_capacity(8192)
    .growth_factor(1.618)  // Golden ratio growth
    .populate_pages(true)  // Pre-load for performance
    .use_huge_pages(true)  // 2MB huge pages on Linux
    .sync_on_write(false)  // Async writes for performance
    .build();

// Advanced operations
vec.extend(&[1, 2, 3, 4, 5]).unwrap();
vec.truncate(100).unwrap();
vec.resize(200, 0).unwrap();
vec.shrink_to_fit().unwrap();

// Memory usage statistics
let stats = vec.stats();
println!("Memory usage: {} bytes", stats.memory_usage);
println!("Utilization: {:.1}%", stats.utilization * 100.0);
println!("File path: {}", vec.path().display());

// Iterator support
for &value in &vec {
    println!("Value: {}", value);
}
```

## Algorithms & Data Structures

### Cache-Oblivious Algorithms

Zipora includes sophisticated cache-oblivious algorithms that achieve optimal performance across different cache hierarchies without explicit knowledge of cache parameters, complementing the existing cache-aware infrastructure.

#### Key Features

- **Cache-Oblivious Sorting**: Funnel sort with optimal O(1 + N/B * log_{M/B}(N/B)) cache complexity
- **Adaptive Algorithm Selection**: Intelligent choice between cache-aware and cache-oblivious strategies based on data characteristics
- **Van Emde Boas Layout**: Cache-optimal data structure layouts with SIMD prefetching
- **SIMD Integration**: Full integration with Zipora's 6-tier SIMD framework (AVX2/BMI2/POPCNT)
- **Recursive Subdivision**: Optimal cache utilization through divide-and-conquer with cache-line aligned access patterns

#### Algorithm Selection Strategy

- **Small data** (< L1 cache): Cache-aware optimized algorithms with insertion sort and SIMD acceleration
- **Medium data** (L1-L3 cache): Cache-oblivious funnel sort for optimal hierarchy utilization
- **Large data** (> L3 cache): Hybrid approach combining cache-oblivious merge with external sorting
- **String data**: Specialized cache-oblivious string algorithms with character-specific optimizations
- **Numeric data**: SIMD-accelerated cache-oblivious variants with hardware prefetching

#### Usage Examples

```rust
use zipora::algorithms::{CacheObliviousSort, CacheObliviousConfig, AdaptiveAlgorithmSelector, VanEmdeBoas};

// Automatic cache-oblivious sorting with adaptive strategy selection
let mut sorter = CacheObliviousSort::new();
let mut data = vec![5, 2, 8, 1, 9, 3, 7, 4, 6];
sorter.sort(&mut data).unwrap();
assert_eq!(data, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);

// Custom configuration with SIMD and parallel processing
let config = CacheObliviousConfig {
    use_simd: true,
    use_parallel: true,
    small_threshold: 512,
    ..Default::default()
};
let mut custom_sorter = CacheObliviousSort::with_config(config);

// Adaptive algorithm selector for strategic decision making
let selector = AdaptiveAlgorithmSelector::new(&config);
let strategy = selector.select_strategy(data.len(), &config.cache_hierarchy);
println!("Selected strategy: {:?}", strategy); // CacheAware, CacheOblivious, or Hybrid

// Data characteristics analysis for optimization
let characteristics = selector.analyze_data(&data);
println!("Fits in L1: {}, L2: {}, L3: {}", 
         characteristics.fits_in_l1, characteristics.fits_in_l2, characteristics.fits_in_l3);

// Van Emde Boas layout for cache-optimal data structures
let cache_hierarchy = detect_cache_hierarchy();
let veb_data = vec![1, 2, 3, 4, 5, 6, 7, 8];
let veb = VanEmdeBoas::new(veb_data, cache_hierarchy);
let element = veb.get(3); // Cache-optimal access with SIMD prefetching
```

#### Performance Characteristics

- **Cache Complexity**: O(1 + N/B * log_{M/B}(N/B)) optimal across all cache levels simultaneously
- **Memory Hierarchy**: Automatic adaptation to L1/L2/L3 cache sizes without manual tuning
- **SIMD Acceleration**: 2-4x speedup with AVX2/BMI2 when available, graceful scalar fallback
- **Adaptive Selection**: Intelligent strategy choice based on data size and cache hierarchy
- **Parallel Processing**: Work-stealing parallelization for large datasets with cache-aware partitioning

#### Algorithm Integration

Cache-oblivious algorithms integrate seamlessly with Zipora's infrastructure:

```rust
// Integration with SecureMemoryPool for cache-aligned allocations
let pool_config = SecurePoolConfig::performance_optimized();
let pool = SecureMemoryPool::new(pool_config).unwrap();
let cache_config = CacheObliviousConfig { memory_pool: Some(Arc::new(pool)), ..Default::default() };

// Integration with cache optimization infrastructure
let cache_layout = CacheLayoutConfig::sequential();
let cache_allocator = CacheOptimizedAllocator::new(cache_layout);

// Integration with SIMD framework for hardware acceleration
let cpu_features = get_cpu_features();
if cpu_features.has_avx2 {
    println!("Using AVX2 acceleration for cache-oblivious operations");
}
```

### Unified Tries

```rust
use zipora::fsa::{ZiporaTrie, ZiporaTrieConfig, Trie, TrieStrategy, CompressionStrategy};
use zipora::memory::{SecureMemoryPool, SecurePoolConfig};

// Default Patricia trie behavior - same API as before!
let mut trie = ZiporaTrie::new();
trie.insert(b"cat").unwrap();
trie.insert(b"car").unwrap();
trie.insert(b"card").unwrap();

// Efficient lookups with O(m) complexity where m is key length
assert!(trie.contains(b"cat"));
assert!(trie.contains(b"car"));
assert!(trie.contains(b"card"));
assert!(!trie.contains(b"ca")); // Path compression active

// Prefix iteration for hierarchical data
for key in trie.iter_prefix(b"car") {
    println!("Found key with 'car' prefix: {:?}", String::from_utf8_lossy(&key));
}

// String-specialized configuration (formerly CritBitTrie)
let mut string_trie = ZiporaTrie::with_config(ZiporaTrieConfig::string_specialized());
string_trie.insert(b"efficient").unwrap();
string_trie.insert(b"effective").unwrap();
string_trie.insert(b"engine").unwrap();

// Automatic BMI2 hardware acceleration detection
let stats = string_trie.stats();
if stats.hardware_acceleration_enabled {
    println!("BMI2 hardware acceleration active for 5-10x faster operations");
}

// Advanced compression statistics
println!("Memory usage: {} bytes, {:.2} bits per key", stats.memory_usage, stats.bits_per_key);
println!("Compression ratio: {:.1}%", stats.compression_ratio * 100.0);

// Space-optimized configuration (formerly LoudsTrie/NestedLoudsTrie)
let mut compact_trie = ZiporaTrie::with_config(ZiporaTrieConfig::space_optimized());
compact_trie.insert(b"hello").unwrap();
compact_trie.insert(b"help").unwrap();
compact_trie.insert(b"world").unwrap();

// Advanced space optimization with LOUDS compression
let space_stats = compact_trie.stats();
println!("Space saved: {:.1}%", (1.0 - space_stats.compression_ratio) * 100.0);

// High-performance concurrent configuration (formerly DoubleArrayTrie)
let pool = std::sync::Arc::new(SecureMemoryPool::new(SecurePoolConfig::default()).unwrap());
let mut concurrent_trie = ZiporaTrie::with_config(
    ZiporaTrieConfig::concurrent_high_performance(pool)
);
concurrent_trie.insert(b"computer").unwrap();
concurrent_trie.insert(b"computation").unwrap();
concurrent_trie.insert(b"compute").unwrap();

// O(1) lookup performance with concurrent access
assert!(concurrent_trie.contains(b"computer"));
let perf_stats = concurrent_trie.stats();
println!("Concurrent performance: {} ops/sec", perf_stats.operations_per_second);

// Custom advanced configuration - full strategy control
let config = ZiporaTrieConfig {
    trie_strategy: TrieStrategy::Patricia {
        max_path_length: 64,
        compression_threshold: 4,
        adaptive_compression: true,
    },
    compression_strategy: CompressionStrategy::PathCompression {
        min_path_length: 2,
        max_path_length: 32,
        adaptive_threshold: true,
    },
    enable_simd: true,
    cache_optimization: true,
    concurrency_level: zipora::fsa::ConcurrencyLevel::OneWriteMultiRead,
    ..ZiporaTrieConfig::default()
};
let mut advanced_trie = ZiporaTrie::with_config(config);
advanced_trie.insert(b"advanced").unwrap();

// Comprehensive performance monitoring
let advanced_stats = advanced_trie.stats();
println!("Memory efficiency: {:.1}%", advanced_stats.memory_efficiency * 100.0);
println!("Cache hit ratio: {:.2}%", advanced_stats.cache_hit_ratio * 100.0);

// Compressed Sparse Trie - Multi-level concurrency with token safety
let mut csp = CompressedSparseTrie::new(ConcurrencyLevel::MultiWriteMultiRead).unwrap();

// Thread-safe operations with tokens
let writer_token = csp.acquire_writer_token().await.unwrap();
csp.insert_with_token(b"hello", &writer_token).unwrap();
csp.insert_with_token(b"world", &writer_token).unwrap();

// Concurrent reads from multiple threads
let reader_token = csp.acquire_reader_token().await.unwrap();
assert!(csp.contains_with_token(b"hello", &reader_token));

// Lock-free optimizations - 90% faster than standard tries for sparse data
let prefix_matches = csp.prefix_search_with_token(b"hel", &reader_token).unwrap();
println!("Found {} matches for prefix 'hel'", prefix_matches.len());

// Nested LOUDS Trie - Configurable nesting with fragment compression
use zipora::{NestingConfig};

let config = NestingConfig::builder()
    .max_levels(4)
    .fragment_compression_ratio(0.3)
    .cache_optimization(true)
    .adaptive_backend_selection(true)
    .build().unwrap();

let mut nested_trie = NestedLoudsTrie::<RankSelectInterleaved256>::with_config(config).unwrap();

// Automatic fragment compression for common substrings
nested_trie.insert(b"computer").unwrap();
nested_trie.insert(b"computation").unwrap();  // Shares prefix compression
nested_trie.insert(b"compute").unwrap();      // Uses fragment compression
nested_trie.insert(b"computing").unwrap();    // Optimal nesting level selection

// Multi-level LOUDS operations with O(1) child access
assert!(nested_trie.contains(b"computer"));
assert_eq!(nested_trie.longest_prefix(b"computing"), Some(7)); // "compute"

// Advanced statistics and layer analysis
let layer_stats = nested_trie.layer_statistics();
for (level, stats) in layer_stats.iter().enumerate() {
    println!("Level {}: {} nodes, {:.1}% compression", 
             level, stats.node_count, stats.compression_ratio * 100.0);
}

// SIMD-optimized bulk operations
let keys = vec![b"apple", b"application", b"apply", b"approach"];
let results = nested_trie.bulk_insert(&keys).unwrap();
println!("Bulk inserted {} keys with fragment sharing", results.len());
```

### Rank/Select Operations

World-Class Succinct Data Structures - Zipora provides 14+ specialized rank/select variants including cutting-edge implementations with comprehensive SIMD optimizations, hardware acceleration, **multi-dimensional support**, and sophisticated mixed implementations:

#### Adaptive Strategy Selection

Zipora features intelligent **Adaptive Strategy Selection** that automatically selects the optimal rank/select implementation based on data density analysis, dataset size, and access patterns. This eliminates the need for manual algorithm selection and ensures optimal performance across diverse workloads.

**Key Benefits:**
- **Automatic Optimization**: Data density analysis selects optimal implementation (sparse vs dense vs balanced)
- **Size-Aware Selection**: Small datasets use cache-efficient implementations, large datasets use separated storage
- **Pattern Recognition**: Access pattern optimization (mixed, rank-heavy, select-heavy, sequential, random)
- **Zero Configuration**: Works out-of-the-box with sensible defaults, but allows custom criteria when needed

```rust
use zipora::{BitVector, AdaptiveRankSelect, SelectionCriteria, AccessPattern, 
            DataProfile, OptimizationStats, PerformanceTier};

// Automatic selection based on data characteristics
let mut sparse_bv = BitVector::new();
for i in 0..10000 {
    sparse_bv.push(i % 100 == 0).unwrap(); // 1% density
}

// Advanced Adaptive selection with sophisticated pattern analysis
let adaptive = AdaptiveRankSelect::new(sparse_bv).unwrap();
println!("Selected: {}", adaptive.implementation_name()); // "RankSelectFew<true> (sparse ones)"

// Get comprehensive pattern analysis information
let profile = adaptive.data_profile();
println!("Density: {:.3}%, Pattern complexity: {:.3}, Clustering: {:.3}, Entropy: {:.3}", 
         profile.density * 100.0, profile.pattern_complexity, 
         profile.clustering_coefficient, profile.entropy);

// Get detailed optimization information
let stats = adaptive.optimization_stats();
println!("Density: {:.1}%, Implementation: {}", 
         stats.density * 100.0, stats.implementation);
println!("Performance tier: {:?}", stats.estimated_performance_tier);

// Custom selection criteria for specific requirements
let criteria = SelectionCriteria {
    sparse_threshold: 0.01,  // 1% threshold for sparse optimization
    dense_threshold: 0.95,   // 95% threshold for dense optimization
    access_pattern: AccessPattern::SelectHeavy,
    prefer_space: true,      // Prioritize space efficiency
    ..Default::default()
};

let mut dense_bv = BitVector::new();
for i in 0..1000 {
    dense_bv.push(i % 10 != 0).unwrap(); // 90% density
}

let custom_adaptive = AdaptiveRankSelect::with_criteria(dense_bv, criteria).unwrap();
```

#### Manual Selection for Fine-Grained Control

```rust
use zipora::{BitVector, RankSelectSimple, RankSelectSeparated256, RankSelectSeparated512,
            RankSelectInterleaved256, RankSelectFew, RankSelectMixedIL256, 
            RankSelectMixedSE512, RankSelectMixedXL256,
            // 🚀 Sophisticated Mixed Implementations:
            RankSelectMixed_IL_256, RankSelectMixedXLBitPacked,
            // Advanced Features:
            RankSelectFragment, RankSelectHierarchical, RankSelectBMI2,
            bulk_rank1_simd, bulk_select1_simd, SimdCapabilities};

// Create a test bit vector
let mut bv = BitVector::new();
for i in 0..1000 {
    bv.push(i % 7 == 0).unwrap(); // Every 7th bit set
}

// Reference implementation for correctness testing
let rs_simple = RankSelectSimple::new(bv.clone()).unwrap();

// High-performance separated storage (256-bit blocks)
let rs_sep256 = RankSelectSeparated256::new(bv.clone()).unwrap();
let rank = rs_sep256.rank1(500);
let pos = rs_sep256.select1(50).unwrap();

// Cache-optimized interleaved storage  
let rs_interleaved = RankSelectInterleaved256::new(bv.clone()).unwrap();
let rank_fast = rs_interleaved.rank1_hardware_accelerated(500);

// Sparse optimization for very sparse data (1% density) - Advanced optimizations
let mut sparse_bv = BitVector::new();
for i in 0..10000 { sparse_bv.push(i % 100 == 0).unwrap(); }
let rs_sparse = RankSelectFew::<true, 64>::from_bit_vector(sparse_bv).unwrap();
println!("Compression ratio: {:.1}%", rs_sparse.compression_ratio() * 100.0);
println!("Hint hit ratio: {:.3}", rs_sparse.hint_hit_ratio());
println!("Memory usage: {} bytes", rs_sparse.memory_usage_bytes());

// Dual-dimension interleaved for related bit vectors
let bv1 = BitVector::from_iter((0..1000).map(|i| i % 3 == 0)).unwrap();
let bv2 = BitVector::from_iter((0..1000).map(|i| i % 5 == 0)).unwrap();
let rs_mixed = RankSelectMixedIL256::new([bv1, bv2]).unwrap();
let rank_dim0 = rs_mixed.rank1_dimension(500, 0);
let rank_dim1 = rs_mixed.rank1_dimension(500, 1);

// 🚀 Sophisticated Mixed IL256 - Dual-dimension interleaved with base+rlev hierarchical caching
let sophisticated_mixed = RankSelectMixed_IL_256::new([bv1.clone(), bv2.clone()]).unwrap();
let hierarchical_rank0 = sophisticated_mixed.rank1_dimension(500, 0);
let hierarchical_rank1 = sophisticated_mixed.rank1_dimension(500, 1);
println!("Hierarchical cache efficiency: {:.2}%", sophisticated_mixed.cache_efficiency() * 100.0);

// 🚀 Extended XL BitPacked - Advanced bit-packed hierarchical caching for memory optimization
let xl_bitpacked = RankSelectMixedXLBitPacked::new([bv1.clone(), bv2.clone()]).unwrap();
let memory_optimized_rank = xl_bitpacked.rank1_dimension(500, 0);
println!("Memory overhead: {:.1}%", xl_bitpacked.memory_overhead_percent());

// Fragment-Based Compression
let rs_fragment = RankSelectFragment::new(bv.clone()).unwrap();
let rank_compressed = rs_fragment.rank1(500);
println!("Compression ratio: {:.1}%", rs_fragment.compression_ratio() * 100.0);

// Hierarchical Multi-Level Caching
let rs_hierarchical = RankSelectHierarchical::new(bv.clone()).unwrap();
let rank_fast = rs_hierarchical.rank1(500);  // O(1) with dense caching
let range_query = rs_hierarchical.rank1_range(100, 200);

// BMI2 Hardware Acceleration with Advanced Comprehensive Module
use zipora::succinct::rank_select::bmi2_comprehensive::{
    Bmi2Capabilities, Bmi2BitOps, Bmi2BlockOps, Bmi2SequenceOps
};

let caps = Bmi2Capabilities::get();
println!("BMI2 tier: {}, BMI1={}, BMI2={}, POPCNT={}, AVX2={}", 
         caps.optimization_tier, caps.has_bmi1, caps.has_bmi2, 
         caps.has_popcnt, caps.has_avx2);

// Ultra-fast select with PDEP/PEXT (5-10x speedup)
let word = 0b1010101010101010u64;
let position = Bmi2BitOps::select1_ultra_fast(word, 3);

// Bulk operations with hardware acceleration
let words = vec![0xAAAAAAAAAAAAAAAAu64; 1000];
let positions = (0..100).step_by(10).collect::<Vec<_>>();
let bulk_ranks = Bmi2BlockOps::bulk_rank1(&words, &positions);

// Advanced sequence analysis for optimization
let analysis = Bmi2SequenceOps::analyze_bit_patterns(&words);
println!("Recommended strategy: {:?}", analysis.recommended_strategy);

// 🚀 Multi-Dimensional SIMD Rank/Select (NEW)
use zipora::succinct::rank_select::multidim_simd::MultiDimRankSelect;

// Create 4-dimensional rank/select structure
let mut dimensions = vec![];
for _ in 0..4 {
    let mut dim_bv = BitVector::new();
    for i in 0..1000 {
        dim_bv.push(i % 3 == 0).unwrap();
    }
    dimensions.push(dim_bv);
}

let multi_rs: MultiDimRankSelect<4> = MultiDimRankSelect::new(dimensions).unwrap();

// Vectorized bulk rank across all dimensions (4-8x faster with SIMD)
let positions = [100, 200, 300, 400];
let ranks = multi_rs.bulk_rank_multidim(&positions);
println!("Ranks across 4 dimensions: {:?}", ranks);

// Bulk select across dimensions (6-12x faster with BMI2)
let target_ranks = [5, 10, 15, 20];
let select_positions = multi_rs.bulk_select_multidim(&target_ranks).unwrap();
println!("Select positions: {:?}", select_positions);

// Cross-dimensional intersection (AVX2-optimized bitwise AND, 4-8x speedup)
let intersection = multi_rs.intersect_dimensions(0, 1).unwrap();
println!("Intersection of dim 0 and 1: {} bits set",
         RankSelectInterleaved256::new(intersection.clone()).unwrap().count_ones());

// Cross-dimensional union (AVX2-optimized bitwise OR)
let union = multi_rs.union_dimensions(&[0, 1, 2]).unwrap();
println!("Union of dimensions 0, 1, 2: {} bits set",
         RankSelectInterleaved256::new(union).unwrap().count_ones());

// SIMD bulk operations with runtime optimization
let bit_data = bv.blocks().to_vec();
let test_positions = vec![100, 200, 300, 400, 500];
let simd_ranks = bulk_rank1_simd(&bit_data, &test_positions);
```

### Advanced Multi-Way Merge Algorithms & Sorting

```rust
use zipora::{SuffixArray, SuffixArrayConfig, SuffixArrayAlgorithm, 
            RadixSort, MultiWayMerge, ReplaceSelectSort, ReplaceSelectSortConfig, 
            LoserTree, LoserTreeConfig, ExternalSort, EnhancedSuffixArray, LcpArray,
            // 🚀 Advanced Multi-Way Merge Components
            EnhancedLoserTree, SetOperations, SetOperationsConfig, SetOperationStats,
            SimdComparator, SimdConfig, SimdOperations};

// 🚀 Enhanced Tournament Tree with O(log k) Complexity and Cache Optimization
let config = LoserTreeConfig {
    initial_capacity: 64,
    use_secure_memory: true,
    stable_sort: true,
    cache_optimized: true,
    use_simd: true,
    prefetch_distance: 2,
    alignment: 64,
};
let mut enhanced_tree = EnhancedLoserTree::new(config);

// Add sorted input streams with true O(log k) complexity
enhanced_tree.add_way(vec![1, 4, 7, 10].into_iter()).unwrap();
enhanced_tree.add_way(vec![2, 5, 8, 11].into_iter()).unwrap();
enhanced_tree.add_way(vec![3, 6, 9, 12].into_iter()).unwrap();

// Initialize with cache-friendly layout and prefetching
enhanced_tree.initialize().unwrap();

// Merge with O(log k) winner selection and cache optimization
let merged = enhanced_tree.merge_to_vec().unwrap();
assert_eq!(merged, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);

// 🚀 Advanced Set Operations with Bit Mask Optimization
let mut set_ops = SetOperations::new();

// Intersection with bit mask optimization (≤32 ways)
let sequences = vec![
    vec![1, 3, 5, 7, 9].into_iter(),
    vec![1, 2, 3, 8, 9].into_iter(),
    vec![1, 3, 4, 6, 9].into_iter(),
];
let intersection = set_ops.intersection(sequences).unwrap();
assert_eq!(intersection, vec![1, 3, 9]);

// Union operations with deduplication
let sequences = vec![
    vec![1, 3, 5].into_iter(),
    vec![2, 4, 6].into_iter(),
    vec![1, 2, 7].into_iter(),
];
let union = set_ops.union(sequences).unwrap();
assert_eq!(union, vec![1, 2, 3, 4, 5, 6, 7]);

// Frequency counting across multiple streams
let frequencies = set_ops.count_frequencies(sequences).unwrap();
println!("Element frequencies: {:?}", frequencies);

// Get performance statistics
let stats = set_ops.stats();
println!("Used bit mask optimization: {}", stats.used_bit_mask);
println!("Processing time: {}μs", stats.processing_time_us);

// 🚀 SIMD-Optimized Merge Operations
let simd_comparator = SimdComparator::new();

// Hardware-accelerated comparisons
let left = vec![1, 3, 5, 7, 9];
let right = vec![2, 4, 6, 8, 10];
let comparisons = simd_comparator.compare_i32_slices(&left, &right).unwrap();

// Find minimum with SIMD acceleration
let values = vec![5, 2, 8, 1, 9, 3];
let (min_idx, min_val) = simd_comparator.find_min_i32(&values).unwrap();
assert_eq!((min_idx, min_val), (3, 1));

// Merge sorted arrays with SIMD optimizations
let left = vec![1, 3, 5, 7];
let right = vec![2, 4, 6, 8];
let merged = simd_comparator.merge_sorted_i32(&left, &right);
assert_eq!(merged, vec![1, 2, 3, 4, 5, 6, 7, 8]);

// Parallel operations for multiple value pairs
let pairs = vec![(1, 2), (5, 3), (4, 4), (9, 7)];
let parallel_results = SimdOperations::parallel_compare_i32(&pairs);

// Multi-array merging with tournament tree
let arrays = vec![
    vec![1, 4, 7],
    vec![2, 5, 8],
    vec![3, 6, 9],
];
let result = SimdOperations::merge_multiple_sorted(arrays);
assert_eq!(result, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);

// External Sorting for Large Datasets (Replacement Selection)
let config = ReplaceSelectSortConfig {
    memory_buffer_size: 64 * 1024 * 1024, // 64MB buffer
    temp_dir: std::path::PathBuf::from("/tmp"),
    merge_ways: 16,
    use_secure_memory: true,
    ..Default::default()
};
let mut external_sorter = ReplaceSelectSort::new(config);
let large_dataset = (0..10_000_000).rev().collect::<Vec<u32>>();
let sorted = external_sorter.sort(large_dataset).unwrap();

// Legacy Tournament Tree (still available)
let tree_config = LoserTreeConfig {
    initial_capacity: 16,
    stable_sort: true,
    cache_optimized: true,
    ..Default::default()
};
let mut tournament_tree = LoserTree::new(tree_config);
tournament_tree.add_way(vec![1, 4, 7, 10].into_iter()).unwrap();
tournament_tree.add_way(vec![2, 5, 8, 11].into_iter()).unwrap();
tournament_tree.add_way(vec![3, 6, 9, 12].into_iter()).unwrap();
let merged = tournament_tree.merge_to_vec().unwrap();

// 🚀 Sophisticated Suffix Array Construction with 5 Algorithm Variants + Adaptive Selection
let text = b"banana";

// Adaptive algorithm selection based on data characteristics (recommended)
let sa = SuffixArray::new(text).unwrap(); // Uses adaptive selection by default
let (start, count) = sa.search(text, b"an");
println!("Found 'an' at {} occurrences", count);

// Manual algorithm selection for specific requirements
let config = SuffixArrayConfig {
    algorithm: SuffixArrayAlgorithm::SAIS,     // SA-IS: Linear-time induced sorting
    use_parallel: true,
    parallel_threshold: 100_000,
    compute_lcp: false,
    optimize_small_alphabet: true,
    adaptive_threshold: 10_000,
};
let sa_sais = SuffixArray::with_config(text, &config).unwrap();

// DC3 algorithm for moderate-sized inputs with good cache locality
let config_dc3 = SuffixArrayConfig {
    algorithm: SuffixArrayAlgorithm::DC3,      // DC3: Divide-and-conquer approach
    ..Default::default()
};
let sa_dc3 = SuffixArray::with_config(text, &config_dc3).unwrap();

// DivSufSort for large inputs with practical performance optimization
let config_div = SuffixArrayConfig {
    algorithm: SuffixArrayAlgorithm::DivSufSort, // DivSufSort: Practical performance
    ..Default::default()
};
let sa_div = SuffixArray::with_config(text, &config_div).unwrap();

// Larsson-Sadakane for highly repetitive data
let config_ls = SuffixArrayConfig {
    algorithm: SuffixArrayAlgorithm::LarssonSadakane, // Larsson-Sadakane: Repetitive data
    ..Default::default()
};
let sa_ls = SuffixArray::with_config(text, &config_ls).unwrap();

// Data characteristics analysis for manual optimization
let characteristics = SuffixArray::analyze_text_characteristics(text);
println!("Text length: {}, Alphabet size: {}", 
         characteristics.text_length, characteristics.alphabet_size);
println!("Repetition ratio: {:.3}, Entropy: {:.3}", 
         characteristics.repetition_ratio, characteristics.entropy);

// Enhanced suffix array with LCP computation
let enhanced_sa = EnhancedSuffixArray::with_lcp(text).unwrap();
let sa = enhanced_sa.suffix_array();
let lcp = enhanced_sa.lcp_array().unwrap();
println!("LCP at position 0: {:?}", lcp.lcp_at(0));

// Suffix array with BWT (Burrows-Wheeler Transform)
let enhanced_sa_bwt = EnhancedSuffixArray::with_bwt(text).unwrap();
if let Some(bwt) = enhanced_sa_bwt.bwt() {
    println!("BWT: {:?}", String::from_utf8_lossy(bwt));
}

// Performance statistics for algorithm comparison
let stats = sa.stats();
println!("Processing time: {}μs, Memory used: {} bytes", 
         stats.processing_time_us, stats.memory_used);
println!("Used parallel: {}, Algorithm: Memory-safe O(n) construction", 
         stats.used_parallel);

// 🚀 Advanced Radix Sort with intelligent algorithm selection
let mut data = vec![5_000_000u32, 2_500_000u32, 8_750_000u32, 1_250_000u32];
let config = RadixSortConfig::adaptive_optimized();
let mut advanced_sorter = AdvancedRadixSort::with_config(config).unwrap();
advanced_sorter.sort_adaptive(&mut data).unwrap();
println!("Strategy: {:?}", advanced_sorter.stats().selected_strategy);

// Legacy high-performance radix sort (still available)
let mut small_data = vec![5u32, 2, 8, 1, 9];
let mut sorter = RadixSort::new();
sorter.sort_u32(&mut small_data).unwrap();

// Multi-way merge with vectorized sources
let sources = vec![
    VectorSource::new(vec![1, 4, 7]),
    VectorSource::new(vec![2, 5, 8]),
];
let mut merger = MultiWayMerge::new();
let result = merger.merge(sources).unwrap();
```

### Advanced Radix Sort Variants - FULLY IMPLEMENTED ✅

Zipora provides a comprehensive suite of advanced radix sort implementations with multiple algorithm strategies, SIMD optimizations, and intelligent adaptive selection for optimal performance across diverse datasets.

#### Key Features

- **Multiple Sorting Strategies**: LSD (Least Significant Digit), MSD (Most Significant Digit), Insertion Sort, Tim Sort, and Adaptive Hybrid approaches
- **SIMD Optimizations**: AVX2 and BMI2 hardware acceleration with runtime CPU feature detection
- **Parallel Processing**: Work-stealing parallel execution with configurable thread pools
- **Adaptive Selection**: Intelligent algorithm selection based on data characteristics and size
- **String-Specific Optimizations**: Specialized algorithms for string data with prefix handling
- **Memory Safety**: Zero unsafe operations in public APIs with SecureMemoryPool integration
- **Configuration Flexibility**: Extensive configuration options for different use cases

#### Advanced Algorithm Implementations

```rust
use zipora::{
    AdvancedRadixSort, RadixSortConfig, RadixSortStrategy, RadixSortStats,
    LsdRadixSort, MsdRadixSort, AdaptiveHybridSort, ParallelRadixSort,
    StringRadixSort, SortingBenchmark
};

// 🚀 Adaptive Hybrid Radix Sort with Intelligent Strategy Selection
let mut data = vec![5_000_000u32, 2_500_000u32, 8_750_000u32, 1_250_000u32];
let config = RadixSortConfig::adaptive_optimized();
let mut sorter = AdvancedRadixSort::with_config(config).unwrap();

// Automatic strategy selection based on data characteristics
sorter.sort_adaptive(&mut data).unwrap();
let stats = sorter.stats();
println!("Selected strategy: {:?}", stats.selected_strategy);
println!("Sort time: {}μs", stats.sort_time_us);

// 🚀 LSD (Least Significant Digit) Radix Sort - High-performance stable sorting
let mut lsd_sorter = LsdRadixSort::new();
let mut data = vec![150u64, 300u64, 50u64, 200u64];
lsd_sorter.sort(&mut data).unwrap();

// Configuration for different data types and ranges
let config = RadixSortConfig {
    strategy: RadixSortStrategy::LSD,
    use_simd: true,          // Enable AVX2/BMI2 acceleration
    use_parallel: true,      // Enable parallel processing
    thread_count: 8,         // 8 threads for parallel execution
    chunk_size: 10_000,      // Optimal chunk size for parallelization
    insertion_threshold: 64, // Switch to insertion sort for small arrays
    use_secure_memory: true, // Use SecureMemoryPool for allocations
    enable_statistics: true, // Collect detailed performance statistics
    ..Default::default()
};

// 🚀 MSD (Most Significant Digit) Radix Sort - Recursive divide-and-conquer
let mut msd_sorter = MsdRadixSort::with_config(config.clone());
let mut large_data: Vec<u32> = (0..1_000_000).rev().collect();
msd_sorter.sort(&mut large_data).unwrap();

// 🚀 Parallel Radix Sort with Work-Stealing
let config = RadixSortConfig::parallel_optimized();
let mut parallel_sorter = ParallelRadixSort::with_config(config).unwrap();
let mut massive_data: Vec<u64> = (0..10_000_000).rev().collect();
parallel_sorter.sort_parallel(&mut massive_data).unwrap();

// Performance monitoring
let parallel_stats = parallel_sorter.stats();
println!("Parallel efficiency: {:.2}%", parallel_stats.parallel_efficiency * 100.0);
println!("Threads used: {}", parallel_stats.threads_used);

// 🚀 String-Specific Radix Sort with Prefix Optimizations
let mut string_data = vec![
    "banana".to_string(),
    "apple".to_string(),
    "cherry".to_string(),
    "date".to_string(),
];

let mut string_sorter = StringRadixSort::new();
string_sorter.sort_strings(&mut string_data).unwrap();

// Advanced string sorting with custom configuration
let string_config = RadixSortConfig {
    strategy: RadixSortStrategy::MSD,
    max_string_length: 256,        // Maximum string length to consider
    prefix_optimization: true,     // Enable common prefix optimization
    suffix_fallback: true,         // Use suffix sorting for long strings
    case_sensitive: true,          // Case-sensitive string comparison
    locale_aware: false,          // Use simple byte comparison
    ..config
};

let mut advanced_string_sorter = StringRadixSort::with_config(string_config);
advanced_string_sorter.sort_strings(&mut string_data).unwrap();
```

#### Configuration Presets for Different Use Cases

```rust
// Performance-optimized configuration for maximum speed
let performance_config = RadixSortConfig::performance_optimized();

// Memory-optimized configuration for minimal memory usage
let memory_config = RadixSortConfig::memory_optimized();

// Parallel configuration for multi-core systems
let parallel_config = RadixSortConfig::parallel_optimized();

// Real-time configuration for low-latency scenarios
let realtime_config = RadixSortConfig::realtime();

// String-specific configuration for text processing
let string_config = RadixSortConfig::string_optimized();

// Adaptive configuration with intelligent strategy selection
let adaptive_config = RadixSortConfig::adaptive_optimized();
```

#### SIMD and Hardware Acceleration

```rust
use zipora::radix_sort::{SimdCapabilities, CpuFeatures, HardwareAcceleration};

// Check available CPU features
let capabilities = SimdCapabilities::detect();
println!("AVX2 available: {}", capabilities.has_avx2);
println!("BMI2 available: {}", capabilities.has_bmi2);
println!("POPCNT available: {}", capabilities.has_popcnt);

// Hardware-accelerated sorting with feature detection
let mut hw_sorter = AdvancedRadixSort::with_hardware_acceleration().unwrap();
let mut data = vec![42u32; 1_000_000];
hw_sorter.sort_hardware_accelerated(&mut data).unwrap();

// Manual SIMD configuration
let simd_config = RadixSortConfig {
    use_simd: true,
    simd_width: 256,              // AVX2 256-bit SIMD
    prefer_bmio2: true,           // Prefer BMI2 instructions when available
    fallback_strategy: RadixSortStrategy::LSD, // Fallback for unsupported hardware
    ..RadixSortConfig::default()
};
```

#### Comprehensive Benchmarking Suite

```rust
use zipora::radix_sort::benchmarks::{SortingBenchmark, BenchmarkConfig, BenchmarkResults};

// Comprehensive benchmark comparing all sorting strategies
let benchmark_config = BenchmarkConfig {
    data_sizes: vec![1_000, 10_000, 100_000, 1_000_000],
    data_types: vec!["u32", "u64", "String"],
    repetitions: 5,
    include_validation: true,
    measure_memory_usage: true,
    compare_with_std: true,
};

let mut benchmark = SortingBenchmark::with_config(benchmark_config);
let results = benchmark.run_comprehensive_benchmark().unwrap();

// Performance comparison results
for result in results.strategy_results {
    println!("Strategy: {:?}", result.strategy);
    println!("Average time: {}μs", result.average_time_us);
    println!("Throughput: {:.2} MB/s", result.throughput_mbps);
    println!("vs std::sort: {:.1}x faster", result.speedup_vs_std);
}

// Memory usage analysis
println!("Peak memory usage: {} MB", results.peak_memory_mb);
println!("Memory efficiency: {:.1}%", results.memory_efficiency * 100.0);
```

#### Advanced Statistics and Monitoring

```rust
// Detailed performance statistics
let stats = sorter.comprehensive_stats();
println!("Algorithm details:");
println!("  Strategy used: {:?}", stats.strategy_used);
println!("  SIMD acceleration: {}", stats.simd_enabled);
println!("  Parallel execution: {}", stats.parallel_execution);
println!("  CPU features used: {:?}", stats.cpu_features_used);

println!("Performance metrics:");
println!("  Total sort time: {}ms", stats.total_time_ms);
println!("  Elements per second: {:.0}", stats.elements_per_second);
println!("  Memory bandwidth: {:.1} GB/s", stats.memory_bandwidth_gbps);
println!("  Cache efficiency: {:.2}%", stats.cache_efficiency * 100.0);

println!("Quality metrics:");
println!("  Comparisons performed: {}", stats.comparison_count);
println!("  Memory allocations: {}", stats.allocation_count);
println!("  Branch mispredictions: {}", stats.branch_mispredictions);
println!("  Cache misses: {}", stats.cache_misses);
```

#### Algorithm Selection Guide

| Strategy | Time Complexity | Space Complexity | Best Use Case | Memory Pattern |
|----------|----------------|------------------|---------------|----------------|
| **Adaptive** | **O(d×n) to O(n log n)** | **O(n)** | **General use (recommended)** | **Intelligent selection** |
| **LSD Radix** | O(d×n) | O(n + k) | Large datasets, stable sorting | Sequential access |
| **MSD Radix** | O(d×n) | O(n + k) | String sorting, prefix patterns | Recursive divide |
| **Parallel** | O(d×n/p) | O(n) | Multi-core systems, large data | Parallel chunks |
| **Hybrid** | O(n log n) worst | O(n) | Mixed data patterns | Adaptive strategies |

**Adaptive Selection Logic:**
- **Small arrays** (< 64): Insertion sort for minimal overhead
- **Integer data** (uniform distribution): LSD radix sort for linear performance
- **String data**: MSD radix sort with prefix optimization
- **Large datasets** (> 1M elements): Parallel processing with work-stealing
- **Mixed patterns**: Hybrid approach with runtime strategy switching

#### Performance Characteristics - ACHIEVED

- **Throughput**: 200-500 MB/s sorting performance depending on data type and algorithm
- **SIMD Acceleration**: 2-4x speedup with AVX2/BMI2 when available
- **Parallel Scaling**: Near-linear scaling up to 8-16 cores
- **Memory Efficiency**: Minimal memory overhead with in-place algorithms where possible
- **Cache Optimization**: Cache-friendly memory access patterns with prefetching
- **Adaptive Performance**: Automatic algorithm selection for optimal performance

#### Integration with Zipora Ecosystem

```rust
// Integration with SecureMemoryPool
let pool_config = SecurePoolConfig::performance_optimized();
let pool = SecureMemoryPool::new(pool_config).unwrap();
let sort_config = RadixSortConfig::with_memory_pool(pool);

// Integration with statistics collection
let stats_config = StatAccumulator::new();
let sort_config = RadixSortConfig::with_statistics(stats_config);

// Integration with five-level concurrency
let concurrency_config = FiveLevelPoolConfig::performance_optimized();
let sort_config = RadixSortConfig::with_concurrency(concurrency_config);
```

### Suffix Array Algorithm Selection Guide

Zipora provides 5 sophisticated suffix array construction algorithms with adaptive selection:

| Algorithm | Time Complexity | Best Use Case | Memory Usage |
|-----------|----------------|---------------|--------------|
| **Adaptive** | **Varies** | **General use (recommended)** | **Optimal** |
| **SA-IS** | O(n) | Small alphabets, general use | ~8n bytes |
| **DC3** | O(n) | Small inputs, good cache locality | ~12n bytes |
| **DivSufSort** | O(n log n) | Large inputs, practical performance | ~8n bytes |
| **Larsson-Sadakane** | O(n log n) | Highly repetitive data | ~12n bytes |

**Adaptive Selection Logic:**
- **Small inputs** (< 10K): DC3 for good cache locality
- **Small alphabets** (≤ 4 chars): SA-IS for optimal linear performance
- **Highly repetitive** (> 70% repetition): Larsson-Sadakane for repetitive optimization
- **Large inputs** (> 1M chars): DivSufSort for practical performance
- **Medium inputs**: SA-IS for reliable linear-time construction

**Memory Safety Features:**
- **Zero unsafe operations** in public APIs
- **Automatic bounds checking** with comprehensive error handling
- **Stack overflow protection** with recursion depth limits and fallback algorithms
- **Memory allocation guards** preventing excessive memory usage

## I/O & Serialization

### Advanced Serialization System

High-Performance Stream Processing - Zipora provides 8 comprehensive serialization components with cutting-edge optimizations, cross-platform compatibility, and production-ready features:

```rust
use zipora::io::{
    // Smart Pointer Serialization
    SmartPtrSerializer, SerializationContext, Box, Rc, Arc, Weak,
    
    // Complex Type Serialization  
    ComplexTypeSerializer, ComplexSerialize, VersionProxy,
    
    // Endian Handling
    EndianIO, Endianness, EndianConvert, EndianConfig,
    
    // Version Management
    VersionManager, VersionedSerialize, Version, MigrationRegistry,
    
    // Variable Integer Encoding
    VarIntEncoder, VarIntStrategy, choose_optimal_strategy,
};

// Smart Pointer Serialization - Reference-counted objects
let shared_data = Rc::new("shared value".to_string());
let clone1 = shared_data.clone();
let clone2 = shared_data.clone();

let serializer = SmartPtrSerializer::default();
let bytes = serializer.serialize_to_bytes(&clone1).unwrap();
let deserialized: Rc<String> = serializer.deserialize_from_bytes(&bytes).unwrap();

// Cycle detection and shared object optimization
let mut context = SerializationContext::new();
clone1.serialize_with_context(&mut output, &mut context).unwrap();
clone2.serialize_with_context(&mut output, &mut context).unwrap(); // References first object

// Complex Type Serialization - Tuples, collections, nested types
let complex_data = (
    vec![1u32, 2, 3],
    Some("nested".to_string()),
    HashMap::from([("key".to_string(), 42u32)]),
);

let serializer = ComplexTypeSerializer::default();
let bytes = serializer.serialize_to_bytes(&complex_data).unwrap();
let deserialized = serializer.deserialize_from_bytes(&bytes).unwrap();

// Batch operations for efficiency
let tuples = vec![(1u32, "first"), (2u32, "second"), (3u32, "third")];
let batch_bytes = serializer.serialize_batch(&tuples).unwrap();
let batch_result = serializer.deserialize_batch(&batch_bytes).unwrap();

// Comprehensive Endian Handling - Cross-platform compatibility
let io = EndianIO::<u32>::little_endian();
let value = 0x12345678u32;

// Safe endian conversion with bounds checking
let mut buffer = [0u8; 4];
io.write_to_bytes(value, &mut buffer).unwrap();
let read_value = io.read_from_bytes(&buffer).unwrap();

// SIMD-accelerated bulk conversions
#[cfg(target_arch = "x86_64")]
{
    use zipora::io::endian::simd::convert_u32_slice_simd;
    let mut values = vec![0x1234u32, 0x5678u32, 0x9abcu32];
    convert_u32_slice_simd(&mut values, false);
}

// Cross-platform configuration
let config = EndianConfig::cross_platform(); // Little endian + auto-detection
let optimized = EndianConfig::performance_optimized(); // Native + SIMD acceleration

// Variable Integer Encoding - Multiple strategies
let encoder = VarIntEncoder::zigzag(); // For signed integers
let signed_values = vec![-100i64, -1, 0, 1, 100];
let encoded = encoder.encode_i64_sequence(&signed_values).unwrap();
let decoded = encoder.decode_i64_sequence(&encoded).unwrap();

// Delta encoding for sorted sequences
let delta_encoder = VarIntEncoder::delta();
let sorted_values = vec![10u64, 12, 15, 20, 22, 25];
let delta_encoded = delta_encoder.encode_u64_sequence(&sorted_values).unwrap();

// Group varint for bulk operations
let group_encoder = VarIntEncoder::group_varint();
let bulk_values = vec![1u64, 256, 65536, 16777216];
let group_encoded = group_encoder.encode_u64_sequence(&bulk_values).unwrap();

// Automatic strategy selection based on data characteristics
let optimal_strategy = choose_optimal_strategy(&values);
let auto_encoder = VarIntEncoder::new(optimal_strategy);
```

### Stream Processing

```rust
use zipora::io::{
    StreamBufferedReader, StreamBufferedWriter, StreamBufferConfig,
    RangeReader, RangeWriter, MultiRangeReader,
    ZeroCopyReader, ZeroCopyWriter, ZeroCopyBuffer, VectoredIO
};

// Advanced Stream Buffering - Configurable strategies
let config = StreamBufferConfig::performance_optimized();
let mut reader = StreamBufferedReader::with_config(cursor, config).unwrap();

// Fast byte reading with hot path optimization
let byte = reader.read_byte_fast().unwrap();

// Bulk read optimization for large data transfers
let mut large_buffer = vec![0u8; 1024 * 1024];
let bytes_read = reader.read_bulk(&mut large_buffer).unwrap();

// Read-ahead capabilities for streaming data
let slice = reader.read_slice(256).unwrap(); // Zero-copy access when available

// Range-based Stream Operations - Partial file access
let mut range_reader = RangeReader::new_and_seek(file, 1024, 4096).unwrap(); // Read bytes 1024-5120

// Progress tracking for partial reads
let progress = range_reader.progress(); // 0.0 to 1.0
let remaining = range_reader.remaining(); // Bytes left to read

// Multi-range reading for discontinuous data
let ranges = vec![(0, 1024), (2048, 3072), (4096, 5120)];
let mut multi_reader = MultiRangeReader::new(file, ranges);

// DataInput trait implementation for structured reading
let value = range_reader.read_u32().unwrap();
let var_int = range_reader.read_var_int().unwrap();

// Zero-Copy Stream Optimizations - Advanced zero-copy operations
let mut zc_reader = ZeroCopyReader::with_secure_buffer(stream, 128 * 1024).unwrap();

// Direct buffer access without memory copying
if let Some(zc_data) = zc_reader.zc_read(1024).unwrap() {
    // Process data directly without copying
    process_data_in_place(zc_data);
    zc_reader.zc_advance(1024).unwrap();
}

// Memory-mapped zero-copy operations (with mmap feature)
#[cfg(feature = "mmap")]
{
    use zipora::io::MmapZeroCopyReader;
    let mut mmap_reader = MmapZeroCopyReader::new(file).unwrap();
    let entire_file = mmap_reader.as_slice(); // Zero-copy access to entire file
}

// Vectored I/O for efficient bulk transfers
let mut buffers = [IoSliceMut::new(&mut buf1), IoSliceMut::new(&mut buf2)];
let bytes_read = VectoredIO::read_vectored(&mut reader, &mut buffers).unwrap();

// SIMD-optimized buffer management with hardware acceleration
let mut buffer = ZeroCopyBuffer::with_secure_pool(1024 * 1024).unwrap();
buffer.fill_from(&mut reader).unwrap(); // Page-aligned allocation
let data = buffer.readable_slice(); // Direct slice access
```

## Concurrency & Synchronization

### Fiber Concurrency

Comprehensive Fiber-Based Concurrency - Zipora provides 3 essential fiber enhancement components with asynchronous I/O integration, cooperative multitasking utilities, and specialized mutex variants for high-performance concurrent applications:

#### FiberAIO - Asynchronous I/O Integration

```rust
use zipora::{FiberAio, FiberAioConfig, IoProvider, VectoredIo, FiberIoUtils};

// High-performance fiber-aware async I/O manager
let config = FiberAioConfig {
    io_provider: IoProvider::auto_detect(), // Tokio/io_uring/POSIX AIO/IOCP
    read_buffer_size: 64 * 1024,
    write_buffer_size: 64 * 1024,
    enable_vectored_io: true,
    enable_direct_io: false,
    read_ahead_size: 256 * 1024,
};

let aio = FiberAio::with_config(config).unwrap();

// Fiber-aware file operations with read-ahead optimization
let mut file = aio.open("large_data.txt").await.unwrap();
let mut buffer = vec![0u8; 1024];
let bytes_read = file.read(&mut buffer).await.unwrap();

// Parallel file processing with controlled concurrency
let paths = vec!["file1.txt", "file2.txt", "file3.txt"];
let results = FiberIoUtils::process_files_parallel(
    paths,
    4, // max concurrent
    |path| Box::pin(async move {
        let aio = FiberAio::new().unwrap();
        aio.read_to_vec(path).await
    })
).await.unwrap();

// Batch processing with automatic yielding
let items = vec![1, 2, 3, 4, 5];
let processed = FiberIoUtils::batch_process(
    items,
    2, // batch size
    |batch| Box::pin(async move {
        // Process batch items
        let results = batch.into_iter().map(|x| x * 2).collect();
        Ok(results)
    })
).await.unwrap();
```

#### FiberYield - Cooperative Multitasking

```rust
use zipora::{FiberYield, YieldConfig, GlobalYield, YieldPoint, YieldingIterator, 
            AdaptiveYieldScheduler, CooperativeUtils};

// High-performance yielding mechanism with budget control
let config = YieldConfig {
    initial_budget: 16,
    max_budget: 32,
    min_budget: 1,
    decay_rate: 0.1,
    yield_threshold: Duration::from_micros(100),
    adaptive_budgeting: true,
};

let yield_controller = FiberYield::with_config(config);

// Lightweight yield operations with budget management
yield_controller.yield_now().await;           // Budget-based yielding
yield_controller.force_yield().await;         // Immediate yield with budget reset
yield_controller.yield_if_needed().await;     // Conditional yield based on time

// Global yield operations using thread-local optimizations
GlobalYield::yield_now().await;
GlobalYield::force_yield().await;
GlobalYield::yield_if_needed().await;

// Cooperative yield points for long-running operations
let yield_point = YieldPoint::new(100); // Yield every 100 operations
for i in 0..10000 {
    // Perform operation
    process_item(i);
    
    // Automatic yielding checkpoint
    yield_point.checkpoint().await;
}

// Yielding wrapper for iterators
let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
let yielding_iter = YieldingIterator::new(data.into_iter(), 3); // Yield every 3 items

let mut sum = 0;
let processed = yielding_iter.for_each(|x| {
    sum += x;
    Ok(())
}).await.unwrap();
```

#### Advanced Mutex Implementations

```rust
use zipora::{AdaptiveMutex, MutexConfig, SpinLock, PriorityRwLock, RwLockConfig, 
            SegmentedMutex};

// Adaptive mutex with statistics and timeout support
let config = MutexConfig {
    fair: false,
    adaptive_spinning: true,
    max_spin_duration: Duration::from_micros(10),
    priority_inheritance: false,
    timeout: Some(Duration::from_millis(100)),
};

let mutex = AdaptiveMutex::with_config(42, config);
{
    let guard = mutex.lock().await;
    println!("Value: {}", *guard);
}

// Performance statistics
let stats = mutex.stats();
println!("Total acquisitions: {}", stats.total_acquisitions);
println!("Contention ratio: {:.2}%", stats.contention_ratio * 100.0);
println!("Average hold time: {}μs", stats.avg_hold_time_us);

// High-performance spin lock for short critical sections
let spin_lock = SpinLock::new(100);
{
    let guard = spin_lock.lock().await;
    *guard += 1; // Short critical section
}

// Reader-writer lock with priority options
let rwlock_config = RwLockConfig {
    writer_priority: true,
    max_readers: Some(64),
    fair: true,
};

let rwlock = PriorityRwLock::with_config(vec![1, 2, 3], rwlock_config);

// Multiple concurrent readers
let read1 = rwlock.read().await;
let read2 = rwlock.read().await;
println!("Data length: {}", read1.len());

// Writer operations with priority
{
    let mut write = rwlock.write().await;
    write.push(4);
}

// Segmented mutex for reducing contention in high-concurrency scenarios
let segmented = SegmentedMutex::new(0, 8); // 8 segments

// Lock specific segment
let mut segment_guard = segmented.lock_segment(3).await;
*segment_guard += 1;

// Hash-based segment selection
let mut key_guard = segmented.lock_for_key(&"my_key").await;
*key_guard += 10;
```

### Low-Level Synchronization

High-Performance Synchronization Primitives - Zipora provides 3 essential low-level synchronization components with Linux futex integration, advanced thread-local storage, and comprehensive atomic operations for maximum concurrency performance:

#### Linux Futex Integration

```rust
use zipora::{LinuxFutex, FutexMutex, FutexCondvar, FutexRwLock, PlatformSync};

// High-performance mutex using direct futex syscalls
let mutex = FutexMutex::new();
{
    let guard = mutex.lock().unwrap();
    // Critical section with zero-overhead synchronization
}

// Condition variable with futex implementation
let condvar = FutexCondvar::new();
let guard = mutex.lock().unwrap();
let guard = condvar.wait(guard).unwrap(); // Zero-overhead blocking

// Reader-writer lock with futex backing
let rwlock = FutexRwLock::new();
{
    let read_guard = rwlock.read().unwrap();
    // Multiple concurrent readers
}
{
    let write_guard = rwlock.write().unwrap();
    // Exclusive writer access
}

// Platform abstraction for cross-platform code
use zipora::{DefaultPlatformSync};
DefaultPlatformSync::futex_wait(&atomic_value, expected_val, timeout).unwrap();
DefaultPlatformSync::futex_wake(&atomic_value, num_waiters).unwrap();
```

#### Instance-Specific Thread-Local Storage

```rust
use zipora::{InstanceTls, OwnerTls, TlsPool};

// Matrix-based O(1) access thread-local storage
let tls = InstanceTls::<MyData>::new().unwrap();

// Each thread gets its own copy of the data
tls.set(MyData { value: 42, name: "thread-local".to_string() });
let data = tls.get(); // O(1) access, automatically creates default if not set
let optional_data = tls.try_get(); // O(1) access, returns None if not set

// Owner-based TLS associating data with specific objects
let mut owner_tls = OwnerTls::<MyData, MyOwner>::new();
let owner = MyOwner { id: 1 };
let data = owner_tls.get_or_create(&owner).unwrap();

// Thread-local storage pool for managing multiple instances
let pool = TlsPool::<MyData, 64>::new().unwrap(); // 64 TLS instances
let data = pool.get_next(); // Round-robin access
let specific_data = pool.get_slot(5).unwrap(); // Access specific slot

// Automatic cleanup and ID recycling
let id = tls.id(); // Unique instance ID
drop(tls); // ID automatically returned to free pool
```

#### Atomic Operations Framework

```rust
use zipora::{AtomicExt, AsAtomic, AtomicStack, AtomicNode, AtomicBitOps, 
            spin_loop_hint, memory_ordering};

// Extended atomic operations
use std::sync::atomic::{AtomicU32, Ordering};
let atomic = AtomicU32::new(10);

// Atomic max/min operations
let old_max = atomic.atomic_maximize(15, Ordering::Relaxed); // Returns 15
let old_min = atomic.atomic_minimize(5, Ordering::Relaxed);  // Returns 5

// Optimized compare-and-swap operations
let result = atomic.cas_weak(5, 10); // Weak CAS with optimized ordering
let strong_result = atomic.cas_strong(10, 20); // Strong CAS

// Conditional atomic updates
let updated = atomic.update_if(|val| val % 2 == 0, 100, Ordering::Relaxed);

// Lock-free data structures
let stack = AtomicStack::<i32>::new();
stack.push(42); // Lock-free push
stack.push(84);
assert_eq!(stack.pop(), Some(84)); // Lock-free pop (LIFO)
assert_eq!(stack.len(), 1); // Approximate size

// Atomic bit operations
let bits = AtomicU32::new(0);
assert!(!bits.set_bit(5)); // Set bit 5, returns previous state
assert!(bits.test_bit(5)); // Test if bit 5 is set
assert!(bits.toggle_bit(5)); // Toggle bit 5
assert_eq!(bits.find_first_set(), None); // Find first set bit

// Safe atomic casting between types
let mut value = 42u32;
let atomic_ref = value.as_atomic_mut(); // &mut AtomicU32
atomic_ref.store(100, Ordering::Relaxed);
assert_eq!(value, 100);

// Platform-specific optimizations
#[cfg(target_arch = "x86_64")]
{
    use zipora::x86_64_optimized;
    x86_64_optimized::pause(); // PAUSE instruction for spin loops
    x86_64_optimized::mfence(); // Memory fence
}

// Memory ordering utilities
memory_ordering::full_barrier(); // Full memory barrier
memory_ordering::load_barrier(); // Load barrier
memory_ordering::store_barrier(); // Store barrier
```

## String Processing

High-Performance String Processing - Zipora provides 4 comprehensive string processing components with SSE4.2 PCMPESTRI acceleration, Unicode support, advanced SIMD search operations, and efficient line-based text processing:

### Lexicographic String Iterators

```rust
use zipora::{LexicographicIterator, SortedVecLexIterator, StreamingLexIterator, 
            LexIteratorBuilder};

// High-performance iterator for sorted string collections
let strings = vec![
    "apple".to_string(),
    "banana".to_string(), 
    "cherry".to_string(),
    "date".to_string(),
];

let mut iter = SortedVecLexIterator::new(&strings);

// Bidirectional iteration with O(1) access
assert_eq!(iter.current(), Some("apple"));
iter.next().unwrap();
assert_eq!(iter.current(), Some("banana"));

// Binary search operations - O(log n) seeking
assert!(iter.seek_lower_bound("cherry").unwrap()); // Exact match
assert_eq!(iter.current(), Some("cherry"));

assert!(!iter.seek_lower_bound("coconut").unwrap()); // No exact match
assert_eq!(iter.current(), Some("date")); // Positioned at next larger

// Streaming iterator for large datasets that don't fit in memory
let reader = std::io::Cursor::new("line1\nline2\nline3\n");
let mut streaming_iter = StreamingLexIterator::new(reader);
while let Some(line) = streaming_iter.current() {
    println!("Processing: {}", line);
    if !streaming_iter.next().unwrap() { break; }
}

// Builder pattern for different backends
let iter = LexIteratorBuilder::new()
    .optimize_for_memory(true)
    .buffer_size(8192)
    .build_sorted_vec(&strings);

// Utility functions for common operations
use zipora::string::utils;
let common_prefix = utils::find_common_prefix(iter).unwrap();
let count = utils::count_with_prefix(iter, "app").unwrap(); // Count strings starting with "app"
```

### SSE4.2 SIMD String Search

Advanced hardware-accelerated string search operations using SSE4.2 PCMPESTRI instructions with hybrid strategy optimization:

```rust
use zipora::{SimdStringSearch, SearchTier, MultiSearchResult, get_global_simd_search,
            sse42_strchr, sse42_strstr, sse42_multi_search, sse42_strcmp};

// Global SIMD string search instance with runtime feature detection
let search = SimdStringSearch::new();
println!("Selected SIMD tier: {:?}", search.tier()); // Sse42, Avx2, or Avx512

// SSE4.2 PCMPESTRI-based character search (strchr equivalent)
// Uses hybrid strategy: SSE4.2 for ≤16 bytes, extended SSE4.2 for ≤35 bytes, binary search for larger
let haystack = b"hello world test string";
assert_eq!(search.sse42_strchr(haystack, b'w'), Some(6));
assert_eq!(search.sse42_strchr(haystack, b'z'), None);

// SSE4.2 substring search with tiered approach (strstr equivalent)
assert_eq!(search.sse42_strstr(haystack, b"world"), Some(6));
assert_eq!(search.sse42_strstr(haystack, b"test"), Some(12));
assert_eq!(search.sse42_strstr(haystack, b"xyz"), None);

// Multi-character vectorized search for multiple needle bytes
let needles = b"aeiou";
let result = search.sse42_multi_search(haystack, needles);
// Returns positions and which characters were found
println!("Found vowels at positions: {:?}", result.positions);
println!("Vowel characters: {:?}", result.characters);

// String comparison with early exit optimizations
use std::cmp::Ordering;
assert_eq!(search.sse42_strcmp(b"hello", b"hello"), Ordering::Equal);
assert_eq!(search.sse42_strcmp(b"hello", b"world"), Ordering::Less);

// Convenience functions using global instance
assert_eq!(sse42_strchr(b"test string", b's'), Some(2));
assert_eq!(sse42_strstr(b"test string", b"str"), Some(5));

// SIMD implementation tiers with automatic fallback
match search.tier() {
    SearchTier::Sse42 => println!("Using SSE4.2 PCMPESTRI acceleration"),
    SearchTier::Avx2 => println!("Using AVX2 256-bit vectorization"),
    SearchTier::Avx512 => println!("Using AVX-512 512-bit vectorization"),
    SearchTier::Scalar => println!("Using portable scalar fallback"),
}
```

**Key Features:**
- **SSE4.2 PCMPESTRI**: Hardware-accelerated character and substring search using specialized string instructions
- **Hybrid Strategy**: Optimal algorithm selection based on data size (≤16 bytes: single PCMPESTRI, ≤35 bytes: cascaded, >35 bytes: chunked processing)
- **Multi-Tier SIMD**: Automatic runtime detection with support for SSE4.2, AVX2, AVX-512, and scalar fallback
- **Early Exit Optimizations**: Hardware-accelerated mismatch detection for string comparison operations
- **Integration Ready**: Designed for use with FSA/Trie, compression algorithms, hash maps, and blob storage systems

**Performance Characteristics:**
- **≤16 bytes**: Single PCMPESTRI instruction (optimal hardware utilization)
- **17-35 bytes**: Cascaded SSE4.2 operations with early exit optimization
- **>35 bytes**: O(log n) binary search with rank-select optimization for large datasets
- **Runtime Detection**: Automatic hardware capability detection with graceful fallback to scalar implementations

### Unicode String Processing

```rust
use zipora::{UnicodeProcessor, UnicodeAnalysis, Utf8ToUtf32Iterator,
            utf8_byte_count, validate_utf8_and_count_chars};

// Hardware-accelerated UTF-8 processing
let text = "Hello 世界! 🦀 Rust";
let char_count = validate_utf8_and_count_chars(text.as_bytes()).unwrap();
println!("Character count: {}", char_count);

// Unicode processor with configurable options
let mut processor = UnicodeProcessor::new()
    .with_normalization(true)
    .with_case_folding(true);

let processed = processor.process("HELLO World!").unwrap();
assert_eq!(processed, "hello world!");

// Comprehensive Unicode analysis
let analysis = processor.analyze("Hello 世界! 🦀");
println!("ASCII ratio: {:.1}%", (analysis.ascii_count as f64 / analysis.char_count as f64) * 100.0);
println!("Complexity score: {:.2}", analysis.complexity_score());
println!("Avg bytes per char: {:.2}", analysis.avg_bytes_per_char());

// Bidirectional UTF-8 to UTF-32 iterator
let mut utf_iter = Utf8ToUtf32Iterator::new(text.as_bytes()).unwrap();
let mut chars = Vec::new();
while let Some(ch) = utf_iter.next_char() {
    chars.push(ch);
}

// Backward iteration support
while let Some(ch) = utf_iter.prev_char() {
    println!("Previous char: {}", ch);
}

// Utility functions for Unicode operations
use zipora::string::unicode::utils;
let display_width = utils::display_width("Hello世界"); // Accounts for wide characters
let codepoints = utils::extract_codepoints("A世"); // [0x41, 0x4E16]
assert!(utils::is_printable("Hello\tWorld\n")); // Allows tabs and newlines
```

### Line-Based Text Processing

```rust
use zipora::{LineProcessor, LineProcessorConfig, LineProcessorStats, LineSplitter};

// High-performance line processor for large text files
let text_data = "line1\nline2\nlong line with multiple words\nfield1,field2,field3\n";
let cursor = std::io::Cursor::new(text_data);

// Configurable processing strategies
let config = LineProcessorConfig::performance_optimized(); // 256KB buffer
// Alternative configs: memory_optimized(), secure()
let mut processor = LineProcessor::with_config(cursor, config);

// Process lines with closure - returns number of lines processed
let processed_count = processor.process_lines(|line| {
    println!("Processing: {}", line);
    Ok(true) // Continue processing
}).unwrap();

// Split lines by delimiter with field-level processing
let cursor = std::io::Cursor::new("name,age,city\nJohn,25,NYC\nJane,30,SF\n");
let mut processor = LineProcessor::new(cursor);

let field_count = processor.split_lines_by(",", |field, line_num, field_num| {
    println!("Line {}, Field {}: {}", line_num, field_num, field);
    Ok(true)
}).unwrap();

// Batch processing for better performance
let cursor = std::io::Cursor::new("line1\nline2\nline3\nline4\n");
let mut processor = LineProcessor::new(cursor);

let total_processed = processor.process_batches(2, |batch| {
    println!("Processing batch of {} lines", batch.len());
    for line in batch {
        println!("  - {}", line);
    }
    Ok(true)
}).unwrap();

// Specialized line splitter with SIMD optimization
let mut splitter = LineSplitter::new().with_optimized_strategy();
let fields = splitter.split("a\tb\tc", "\t"); // Tab-separated
assert_eq!(fields, ["a", "b", "c"]);

// Utility functions for text analysis
use zipora::string::line_processor::utils;
let cursor = std::io::Cursor::new("hello world\nhello rust\nworld rust\n");
let processor = LineProcessor::new(cursor);

// Word frequency analysis
let frequencies = utils::count_word_frequencies(processor).unwrap();
assert_eq!(frequencies.get("hello"), Some(&2));

// Text statistics
let cursor = std::io::Cursor::new("line1\nline2\n\nlong line with multiple words\n");
let processor = LineProcessor::new(cursor);
let analysis = utils::analyze_text(processor).unwrap();
println!("Total lines: {}", analysis.total_lines);
println!("Empty lines: {}", analysis.empty_lines);
println!("Avg line length: {:.1}", analysis.avg_line_length());
```

## Development Tools

### Advanced Profiling Integration

Zipora features a comprehensive profiling system for performance analysis, bottleneck identification, and optimization guidance across development, testing, and production environments.

#### Core Profiling Components

```rust
use zipora::dev_infrastructure::profiling::*;

// RAII-based automatic profiling with zero overhead when disabled
{
    let _scope = ProfilerScope::new("critical_operation")?;
    // Your code here - automatically timed and tracked
    critical_computation();
} // Automatic cleanup and data collection

// Manual profiling with fine-grained control
let profiler = HardwareProfiler::global()?;
let handle = profiler.start("database_query")?;
execute_database_query();
let data = profiler.end(handle)?;
println!("Query took: {:?}", data.duration);
```

#### Hardware Performance Profiler

Cross-platform high-precision timing with performance counter integration:

```rust
use zipora::dev_infrastructure::profiling::{HardwareProfiler, ProfilingData};

// Automatic hardware detection and optimal timer selection
let profiler = HardwareProfiler::global()?;

// Profile CPU-intensive operations
let handle = profiler.start("matrix_multiplication")?;
let result = matrix_multiply(&a, &b);
let data = profiler.end(handle)?;

println!("Operation: {}", data.operation_name);
println!("Duration: {:?}", data.duration);
println!("CPU cycles: {:?}", data.cpu_cycles);
```

**Platform Support:**
- **Windows**: QueryPerformanceCounter for microsecond precision
- **Unix/Linux/macOS**: clock_gettime(CLOCK_MONOTONIC) for nanosecond precision
- **Hardware Counters**: CPU cycle counting where available
- **Fallback**: High-resolution Instant::now() on all platforms

#### Memory Allocation Profiler

Integrated memory tracking with SecureMemoryPool for comprehensive allocation analysis:

```rust
use zipora::dev_infrastructure::profiling::{MemoryProfiler, MemoryStats};

let profiler = MemoryProfiler::global()?;
let handle = profiler.start("memory_intensive_task")?;

// Memory allocations are automatically tracked
let mut large_buffer = vec![0u8; 10 * 1024 * 1024]; // 10MB allocation
large_buffer.resize(20 * 1024 * 1024, 1); // Growth tracked

let data = profiler.end(handle)?;
println!("Peak memory: {} bytes", data.peak_memory_usage);
println!("Allocations: {}", data.allocation_count);
```

**Memory Tracking Features:**
- **Allocation Counting**: Track number and size of allocations
- **Peak Usage**: Monitor maximum memory consumption
- **Growth Patterns**: Analyze memory usage over time
- **SecureMemoryPool Integration**: Leverage existing memory safety infrastructure
- **Thread-Safe**: Concurrent memory tracking across multiple threads

#### Cache Performance Profiler

Cache efficiency monitoring with integration to Zipora's cache optimization infrastructure:

```rust
use zipora::dev_infrastructure::profiling::{CacheProfiler, CacheStats};

let profiler = CacheProfiler::global()?;
let handle = profiler.start("cache_sensitive_algorithm")?;

// Cache performance automatically monitored
process_large_dataset(&data);

let data = profiler.end(handle)?;
println!("Cache hit ratio: {:.2}%", data.cache_hit_ratio * 100.0);
println!("Cache misses: {}", data.cache_misses);
```

**Cache Monitoring:**
- **Hit/Miss Ratios**: Track cache efficiency across operations
- **Access Patterns**: Monitor sequential vs. random access performance
- **Cache Line Utilization**: Analyze cache-friendly data layout effectiveness
- **NUMA Awareness**: Track memory locality on multi-socket systems
- **Integration**: Works with LruPageCache, CacheOptimizedAllocator, and hot/cold separation

#### Profiler Registry and Management

Unified profiler management with thread-safe initialization and lifecycle control:

```rust
use zipora::dev_infrastructure::profiling::{ProfilerRegistry, ProfilingConfig};

// Global registry with automatic initialization
let registry = ProfilerRegistry::new();

// Access any profiler type through unified interface
let hw_profiler = registry.get_hardware_profiler()?;
let mem_profiler = registry.get_memory_profiler()?;
let cache_profiler = registry.get_cache_profiler()?;

// Configuration-driven profiler selection
let config = ProfilingConfig::development()
    .with_hardware_profiling(true)
    .with_memory_profiling(true)
    .with_cache_profiling(true);
    
registry.configure(config)?;
```

#### Rich Configuration System

Comprehensive configuration with presets, builder patterns, and runtime adaptation:

```rust
use zipora::dev_infrastructure::profiling::{ProfilingConfig, SamplingRate, OutputFormat};

// Preset configurations for different environments
let production_config = ProfilingConfig::production();    // Minimal overhead
let development_config = ProfilingConfig::development();  // Balanced profiling
let debugging_config = ProfilingConfig::debugging();      // Maximum detail
let disabled_config = ProfilingConfig::disabled();        // Zero overhead

// Custom configuration with builder pattern
let custom_config = ProfilingConfig::new()
    .with_hardware_profiling(true)
    .with_memory_profiling(true)
    .with_cache_profiling(false)
    .with_sampling_rate(SamplingRate::Medium)
    .with_output_format(OutputFormat::Json)
    .with_buffer_size(8192)
    .with_collection_interval(Duration::from_millis(100));

// Environment-driven configuration
let config = if cfg!(debug_assertions) {
    ProfilingConfig::debugging()
} else {
    ProfilingConfig::production()
};
```

#### Advanced Reporting and Analysis

Comprehensive performance analysis with statistical insights and bottleneck identification:

```rust
use zipora::dev_infrastructure::profiling::{ProfilerReporter, ProfilingReport};

// Create reporter with configuration
let config = ProfilingConfig::development();
let reporter = ProfilerReporter::new(config)?;

// Generate comprehensive performance report
let report = reporter.generate_report()?;

// Performance summary
println!("Report Summary:");
println!("  Total operations: {}", report.summary.total_operations);
println!("  Total duration: {:?}", report.summary.total_duration);
println!("  Average operation time: {:?}", report.summary.average_duration);

// Bottleneck analysis
for bottleneck in &report.analysis.bottlenecks {
    println!("Bottleneck: {} ({:.2}% of total time)", 
             bottleneck.operation_name, bottleneck.percentage_of_total);
}

// Statistical insights
println!("Performance Statistics:");
println!("  95th percentile: {:?}", report.statistics.percentile_95);
println!("  Standard deviation: {:?}", report.statistics.std_deviation);

// Export in multiple formats
let json_report = reporter.export_report(&report)?; // JSON format
```

**Report Features:**
- **Statistical Analysis**: Mean, median, percentiles, standard deviation
- **Bottleneck Identification**: Automatically identify performance hotspots
- **Trend Analysis**: Track performance changes over time
- **Anomaly Detection**: Identify unusual performance patterns
- **Multiple Export Formats**: JSON, CSV, Text, Binary
- **Cross-Platform**: Consistent reporting across all supported platforms

#### Performance Overhead and Benchmarking

The profiling system is designed for minimal performance impact:

```rust
// Production configuration: <5% overhead
let config = ProfilingConfig::production();

// Development configuration: <15% overhead  
let config = ProfilingConfig::development();

// Benchmark profiling overhead
use criterion::{black_box, Criterion};

fn benchmark_profiling_overhead(c: &mut Criterion) {
    c.bench_function("profiling_overhead", |b| {
        b.iter(|| {
            let _scope = ProfilerScope::new("benchmark_operation")?;
            black_box(cpu_intensive_work());
        });
    });
}
```

#### Integration with Zipora Ecosystem

The profiling system integrates seamlessly with Zipora's infrastructure:

```rust
// SIMD Framework integration
use zipora::simd::{SimdCapabilities, CpuFeatures};
let caps = SimdCapabilities::detect();
// Profiling automatically detects and uses optimal SIMD instructions

// SecureMemoryPool integration  
let pool_config = SecurePoolConfig::performance_optimized();
let pool = SecureMemoryPool::new(pool_config)?;
// Memory profiler tracks SecureMemoryPool allocations automatically

// Cache optimization integration
let cache_config = CacheLayoutConfig::performance_optimized();
let allocator = CacheOptimizedAllocator::new(cache_config);
// Cache profiler monitors cache optimization effectiveness

// Five-level concurrency integration
let concurrency_config = FiveLevelPoolConfig::performance_optimized();
let pool = AdaptiveFiveLevelPool::new(concurrency_config)?;
// Profiling tracks concurrency performance across all levels
```

#### Cross-Platform Validation

Comprehensive testing ensures consistent behavior across platforms:

- **x86_64**: Full hardware counter support with AVX2/BMI2 optimizations
- **ARM64**: Native performance counter integration with NEON optimizations  
- **Windows**: QueryPerformanceCounter integration with IOCP profiling
- **Linux**: clock_gettime and perf_event_open integration
- **macOS**: High-resolution mach_absolute_time integration

#### Best Practices

1. **Use RAII Scopes**: Prefer `ProfilerScope` for automatic cleanup
2. **Configure by Environment**: Use appropriate presets for development vs. production
3. **Sample Appropriately**: Adjust sampling rates based on performance requirements
4. **Monitor Overhead**: Regularly benchmark profiling impact on critical paths
5. **Analyze Reports**: Use comprehensive reports to identify optimization opportunities
6. **Integrate Testing**: Include profiling in performance regression tests

### Factory Pattern Implementation

```rust
use zipora::{FactoryRegistry, GlobalFactory, global_factory, Factoryable};

// Generic factory registry for any type
let factory = FactoryRegistry::<Box<dyn MyTrait>>::new();

// Register creators with automatic type detection
factory.register_type::<ConcreteImpl, _>(|| {
    Ok(Box::new(ConcreteImpl::new()) as Box<dyn MyTrait>)
}).unwrap();

// Create objects by type name
let obj = factory.create_by_type::<ConcreteImpl>().unwrap();

// Global factory for convenient access
global_factory::<Box<dyn MyTrait>>().register("my_impl", || {
    Ok(Box::new(ConcreteImpl::new()) as Box<dyn MyTrait>)
}).unwrap();

// Factory builder pattern for complex setups
let factory = FactoryBuilder::new("component_factory")
    .with_creator("fast_impl", || Ok(FastImpl::new())).unwrap()
    .with_creator("safe_impl", || Ok(SafeImpl::new())).unwrap()
    .build();

// Automatic registration with macros
register_factory_type!(ConcreteImpl, Box<dyn MyTrait>, || {
    Ok(Box::new(ConcreteImpl::new()) as Box<dyn MyTrait>)
});

// Use Factoryable trait for convenient creation
let instance = MyTrait::create("my_impl").unwrap();
assert!(MyTrait::has_creator("my_impl").unwrap());
```

### Debugging Framework

```rust
use zipora::{HighPrecisionTimer, ScopedTimer, BenchmarkSuite, MemoryDebugger, 
            PerformanceProfiler, global_profiler, measure_time, debug_print};

// High-precision timing with automatic unit selection
let timer = HighPrecisionTimer::named("operation");
// ... perform operation ...
timer.print_elapsed(); // Automatic unit selection (ns/μs/ms/s)

// Scoped timing with automatic reporting
{
    let _timer = ScopedTimer::with_message("database_query", "Query completed");
    // Timer automatically reports when dropped
}

// Comprehensive benchmark suite
let mut suite = BenchmarkSuite::new("performance_tests");
suite.add_benchmark("fast_operation", 10000, || {
    // Fast operation to benchmark
});
suite.run_all(); // Statistics with ops/sec

// Performance profiling with global registry
global_profiler().profile("critical_path", || {
    // ... critical operation ...
    Ok(result)
}).unwrap();

// Memory debugging for custom allocators
let debugger = MemoryDebugger::new();
debugger.record_allocation(ptr as usize, size, "module:function:line");
let stats = debugger.get_stats();
println!("Peak usage: {} bytes", stats.peak_usage);

// Convenient timing macro
measure_time!("algorithm_execution", {
    complex_algorithm();
});

// Debug assertions and prints (debug builds only)
debug_assert_msg!(condition, "Critical invariant violated");
debug_print!("Debug value: {}", value);
```

### Statistical Analysis Tools

```rust
use zipora::{Histogram, U32Histogram, StatAccumulator, MultiDimensionalStats, 
            global_stats, StatIndex};

// Adaptive histogram with dual storage strategy
let mut hist = U32Histogram::new();
hist.increment(100);  // Small values: direct array access O(1)
hist.increment(5000); // Large values: hash map storage
hist.add(1000, 5);    // Add multiple counts

// Comprehensive statistics
let stats = hist.stats();
println!("Mean: {:.2}", stats.mean_key.unwrap());
println!("Distinct keys: {}", stats.distinct_key_count);

// Percentiles and analysis
hist.finalize(); // Optimize for analysis
let median = hist.median().unwrap();
let p95 = hist.percentile(0.95).unwrap();

// Real-time statistics accumulator (thread-safe)
let acc = StatAccumulator::new();
acc.add(42);  // Lock-free atomic operations
acc.add(100);
acc.add(75);

let snapshot = acc.snapshot();
println!("Mean: {:.2}, Std Dev: {:.2}", snapshot.mean, snapshot.std_dev);

// Multi-dimensional statistics
let mut multi_stats = MultiDimensionalStats::new(
    "network_metrics",
    vec!["latency".to_string(), "throughput".to_string(), "errors".to_string()]
);

multi_stats.add_sample(&[50, 1000, 0]).unwrap(); // latency, throughput, errors
multi_stats.add_sample(&[75, 950, 1]).unwrap();

let latency_stats = multi_stats.dimension_stats(0).unwrap();
println!("Average latency: {:.1}ms", latency_stats.mean);

// Global statistics registry
global_stats().register_histogram("request_sizes", hist).unwrap();
global_stats().register_accumulator("response_times", acc).unwrap();

// List all registered statistics
let all_stats = global_stats().list_statistics().unwrap();
for stat_name in all_stats {
    println!("Registered: {}", stat_name);
}
```

## PA-Zip Dictionary Compression - FULLY IMPLEMENTED

Zipora features a **complete and production-ready** implementation of the PA-Zip algorithm, an advanced dictionary compression system that combines three sophisticated algorithms working together seamlessly for high-performance pattern matching and compression.

### Core Algorithm Implementation - COMPLETE

**All three core algorithms are fully implemented and working together:**

- **SA-IS Suffix Array Construction**: Complete O(n) time implementation with induced sorting algorithm
- **BFS DFA Cache Construction**: Breadth-first search double array trie with O(1) state transitions  
- **Two-Level Pattern Matching**: Sophisticated strategy combining DFA cache + suffix array fallback

### Key Features - PRODUCTION READY

- **8 Compression Types**: Complete encoding strategies for different data patterns (Literal, Global, RLE, NearShort, Far1Short, Far2Short, Far2Long, Far3Long)
- **Advanced Dictionary Building**: BFS-based pattern discovery with configurable frequency thresholds
- **DFA Cache Acceleration**: O(1) state transitions for common pattern prefixes with 70-90% hit rates
- **Memory-Safe Implementation**: Zero unsafe operations in public APIs
- **Flexible Integration**: Full integration with blob store framework and memory pools
- **Production Ready**: Zero compilation errors, all library tests passing
- **Comprehensive Testing**: 1,630+ tests passing including unified entropy coding implementations

### Usage Examples

```rust
use zipora::compression::dict_zip::{
    DictZipBlobStore, DictZipBlobStoreBuilder, DictZipConfig, QuickConfig,
    DictionaryBuilder, DictionaryBuilderConfig, PaZipCompressor, PaZipCompressorConfig
};
use zipora::blob_store::BlobStore;

// Quick configuration presets for common use cases
let text_config = QuickConfig::text_compression();      // Text files, documents
let binary_config = QuickConfig::binary_compression();  // Binary data, executables
let log_config = QuickConfig::log_compression();        // Log files, high repetition
let realtime_config = QuickConfig::realtime_compression(); // Low-latency scenarios

// Build dictionary-compressed blob store with training samples
let training_samples = vec![
    b"The quick brown fox jumps over the lazy dog".to_vec(),
    b"The lazy dog was jumped over by the quick brown fox".to_vec(),
    b"Quick brown foxes are faster than lazy dogs".to_vec(),
];

let config = DictZipConfig::text_compression();
let mut builder = DictZipBlobStoreBuilder::with_config(config).unwrap();

// Train dictionary from samples
for sample in training_samples {
    builder.add_training_sample(&sample).unwrap();
}

// Build the final store with optimized dictionary
let mut store = builder.finish().unwrap();

// Use the store for high-ratio compression
let data = b"The quick brown fox jumps";
let id = store.put(data).unwrap();
let retrieved = store.get(id).unwrap();
assert_eq!(data, retrieved.as_slice());

// Check compression performance
let stats = store.compression_stats();
println!("Compression ratio: {:.1}%", stats.compression_ratio() * 100.0);
println!("Space saved: {:.1}%", stats.space_saved_percent());
println!("Dictionary hit rate: {:.2}%", stats.dictionary_hit_rate * 100.0);

// Advanced dictionary building with custom configuration
let dict_config = DictionaryBuilderConfig {
    target_dict_size: 32 * 1024 * 1024,  // 32MB dictionary
    max_dict_size: 64 * 1024 * 1024,     // 64MB maximum
    min_frequency: 4,                     // Minimum pattern frequency
    max_bfs_depth: 8,                     // DFA cache depth
    min_pattern_length: 6,                // Minimum pattern length
    max_pattern_length: 256,              // Maximum pattern length
    sample_ratio: 0.3,                    // Sample 30% of training data
    validate_result: true,                // Validate dictionary correctness
    enable_parallel: true,                // Use parallel construction
    use_memory_pool: true,                // Use secure memory pools
    ..Default::default()
};

let builder = DictionaryBuilder::with_config(dict_config);
let training_data = std::fs::read("training_corpus.txt").unwrap();
let mut dictionary = builder.build(&training_data).unwrap();

// Direct pattern matching with the dictionary
let input = b"The quick brown fox";
let match_result = dictionary.find_longest_match(input, 0, 100).unwrap();

if let Some(pattern_match) = match_result {
    println!("Found match: length={}, position={}, quality={:.2}", 
             pattern_match.length, pattern_match.dict_position, pattern_match.quality);
}

// PA-Zip compressor for low-level compression
let compressor_config = PaZipCompressorConfig::performance_optimized();
let mut compressor = PaZipCompressor::with_config(compressor_config).unwrap();

// Train compressor with sample data
let samples = vec![b"sample data 1", b"sample data 2", b"sample data 3"];
compressor.train(&samples).unwrap();

// Compress data using trained patterns
let input = b"sample data for compression";
let compressed = compressor.compress(input).unwrap();
let decompressed = compressor.decompress(&compressed).unwrap();
assert_eq!(input, decompressed.as_slice());

// Batch operations for high throughput
let batch_data = vec![
    b"batch item 1".to_vec(),
    b"batch item 2".to_vec(), 
    b"batch item 3".to_vec(),
];

let batch_ids = store.put_batch(&batch_data).unwrap();
let retrieved_batch = store.get_batch(&batch_ids).unwrap();

// Advanced statistics and analysis
let match_stats = dictionary.match_stats();
println!("Total searches: {}", match_stats.total_searches);
println!("Cache hits: {}", match_stats.cache_hits);
println!("Average match length: {:.1}", match_stats.average_match_length());

let compression_stats = compressor.compression_stats();
println!("Bytes processed: {}", compression_stats.total_input_bytes);
println!("Bytes compressed: {}", compression_stats.total_output_bytes);
println!("Compression speed: {:.1} MB/s", compression_stats.compression_speed_mbps());

// Dictionary validation and optimization
dictionary.validate().unwrap(); // Verify dictionary integrity
dictionary.optimize().unwrap(); // Optimize for access patterns

// DFA cache statistics  
let cache_stats = dictionary.cache_stats();
println!("Cache hit ratio: {:.1}%", cache_stats.hit_ratio() * 100.0);
println!("Cache utilization: {:.1}%", cache_stats.utilization() * 100.0);
```

### Configuration Presets

PA-Zip provides optimized configuration presets for different data types:

| Preset | Dictionary Size | Min Frequency | BFS Depth | Pattern Length | Use Case |
|--------|----------------|---------------|-----------|----------------|----------|
| **Text** | 32MB | 3 | 6 | 4-128 | Documents, text files |
| **Binary** | 16MB | 8 | 4 | 8-64 | Executables, binary data |
| **Logs** | 64MB | 2 | 8 | 10-256 | Log files, high repetition |
| **Realtime** | 8MB | 10 | 3 | 6-32 | Low-latency compression |

### Implementation Architecture

**Complete Three-Algorithm Integration:**

1. **SA-IS Suffix Array Construction**: Linear-time suffix array construction using the SA-IS (Suffix Array by Induced Sorting) algorithm with type classification and induced sorting phases

2. **BFS DFA Cache Building**: Breadth-first search construction of double array trie for frequent patterns with configurable depth and frequency thresholds

3. **Two-Level Pattern Matching Engine**: 
   - **Level 1**: DFA cache lookup for O(1) common pattern access
   - **Level 2**: Suffix array binary search for comprehensive pattern coverage
   - **Adaptive Strategy**: Intelligent fallback between cache and suffix array based on pattern characteristics

### Performance Characteristics - ACHIEVED

- **Dictionary Construction**: O(n) time using complete SA-IS suffix array implementation
- **Pattern Matching**: O(1) for cached patterns, O(log n + m) for suffix array fallback
- **Memory Usage**: ~8 bytes per suffix array entry + optimized DFA cache storage
- **Cache Efficiency**: 70-90% hit rate for typical text compression workloads
- **Compression Speed**: 50-200 MB/s depending on data characteristics and pattern density
- **Compression Ratio**: 30-80% size reduction depending on data repetitiveness
- **Build Status**: All compilation working in debug and release modes
- **Test Coverage**: 1,630+ tests passing with unified entropy coding implementations

### Integration with Zipora Ecosystem

PA-Zip fully integrates with zipora's infrastructure:

```rust
// Integration with SecureMemoryPool
let pool_config = SecurePoolConfig::performance_optimized();
let pool = SecureMemoryPool::new(pool_config).unwrap();
let dict_config = DictionaryBuilderConfig::with_memory_pool(pool);

// Integration with blob storage systems
let trie_store = NestLoudsTrieBlobStore::new(config).unwrap();
let dict_compressed_store = DictZipBlobStore::from_trie_store(trie_store).unwrap();

// Integration with LRU caching
let cache_config = PageCacheConfig::performance_optimized();
let cached_dict_store = CachedBlobStore::new(dict_compressed_store, cache_config).unwrap();

// Integration with five-level concurrency
let concurrency_config = FiveLevelPoolConfig::performance_optimized();
let concurrent_store = DictZipBlobStore::with_concurrency(config, concurrency_config).unwrap();
```

## Compression Framework

### PA-Zip Dictionary Compression (Primary Algorithm)

```rust
use zipora::compression::dict_zip::{DictZipBlobStore, DictZipConfig, QuickConfig};

// PA-Zip dictionary compression with advanced three-algorithm approach
let config = QuickConfig::text_compression();
let mut store = DictZipBlobStore::with_config(config).unwrap();

// Train with samples for optimal dictionary construction
let training_samples = vec![
    b"The quick brown fox jumps over the lazy dog".to_vec(),
    b"Quick brown foxes jump over lazy dogs regularly".to_vec(),
];

for sample in training_samples {
    store.add_training_sample(&sample).unwrap();
}

// Compress data using SA-IS + BFS DFA cache + two-level pattern matching
let data = b"The quick brown fox jumps";
let id = store.put(data).unwrap();
let retrieved = store.get(id).unwrap();

// Exceptional compression ratios with high-speed processing
let stats = store.compression_stats();
println!("Compression ratio: {:.1}%", stats.compression_ratio() * 100.0);
println!("Dictionary hit rate: {:.2}%", stats.dictionary_hit_rate * 100.0);
```

### Advanced Entropy Coding Algorithms

```rust
use zipora::entropy::*;

// 🚀 Contextual Huffman coding with Order-1/Order-2 models
let contextual_encoder = ContextualHuffmanEncoder::new(b"training data", HuffmanOrder::Order1).unwrap();
let compressed = contextual_encoder.encode(b"sample data").unwrap();

// 🚀 64-bit rANS with parallel variants
let mut frequencies = [1u32; 256];
for &byte in b"sample data" { frequencies[byte as usize] += 1; }
let rans_encoder = Rans64Encoder::<ParallelX4>::new(&frequencies).unwrap();
let compressed = rans_encoder.encode(b"sample data").unwrap();

// 🚀 FSE with ZSTD optimizations
let mut fse_encoder = FseEncoder::new(FseConfig::high_compression()).unwrap();
let compressed = fse_encoder.compress(b"sample data").unwrap();

// 🚀 Parallel encoding with adaptive selection
let mut parallel_encoder = AdaptiveParallelEncoder::new().unwrap();
let compressed = parallel_encoder.encode_adaptive(b"sample data").unwrap();

// 🚀 Hardware-optimized bit operations
let bit_ops = BitOps::new();
if bit_ops.has_bmi2() {
    let result = bit_ops.pdep_u64(value, mask); // BMI2 acceleration
}

// 🚀 Context-aware memory management
let config = EntropyContextConfig::default();
let mut context = EntropyContext::new(config);
let buffer = context.get_buffer(1024).unwrap(); // Efficient buffer pooling

// Fiber concurrency
use zipora::{FiberPool, AdaptiveCompressor, RealtimeCompressor};

async fn example() {
    // Parallel processing
    let pool = FiberPool::default().unwrap();
    let result = pool.parallel_map(vec![1, 2, 3], |x| Ok(x * 2)).await.unwrap();
    
    // Adaptive compression
    let compressor = AdaptiveCompressor::default().unwrap();
    let compressed = compressor.compress(b"data").unwrap();
    
    // Real-time compression
    let rt_compressor = RealtimeCompressor::with_mode(CompressionMode::LowLatency).unwrap();
    let compressed = rt_compressor.compress(b"data").await.unwrap();
}
```

### Memory-Mapped I/O & Advanced Stream Processing

```rust
#[cfg(feature = "mmap")]
{
    use zipora::{MemoryMappedOutput, MemoryMappedInput, DataInput, DataOutput,
                StreamBufferedReader, RangeReader, ZeroCopyReader};
    
    // Memory-mapped output with automatic growth
    let mut output = MemoryMappedOutput::create("data.bin", 1024).unwrap();
    output.write_u32(0x12345678).unwrap();
    output.flush().unwrap();
    
    // Zero-copy reading with memory mapping
    let file = std::fs::File::open("data.bin").unwrap();
    let mut input = MemoryMappedInput::new(file).unwrap();
    assert_eq!(input.read_u32().unwrap(), 0x12345678);
    
    // Advanced stream buffering with configurable strategies
    let file = std::fs::File::open("large_data.bin").unwrap();
    let mut buffered_reader = StreamBufferedReader::performance_optimized(file).unwrap();
    
    // Range-based partial file access
    let file = std::fs::File::open("data.bin").unwrap();
    let mut range_reader = RangeReader::new_and_seek(file, 1024, 4096).unwrap();
    let progress = range_reader.progress(); // Track reading progress
    
    // Zero-copy operations for maximum performance
    let file = std::fs::File::open("data.bin").unwrap();
    let mut zc_reader = ZeroCopyReader::with_secure_buffer(file, 256 * 1024).unwrap();
    if let Some(data) = zc_reader.zc_read(1024).unwrap() {
        // Process data without copying
        process_data_efficiently(data);
        zc_reader.zc_advance(1024).unwrap();
    }
}
```

## Performance & Security

### Performance Fix Implementation ✅

**Critical Performance Issue Resolved**: The hardware acceleration bug identified in performance analysis has been successfully fixed. The codebase previously had `#[cfg(test)]` blocks that disabled BMI2/AVX2/POPCNT features during testing, causing 33-45x slower performance than claimed. This has been completely resolved through proper runtime CPU feature detection.

**Fix Implementation**:
- ✅ Removed test-mode hardware feature disabling
- ✅ Implemented proper `is_x86_feature_detected!()` runtime detection
- ✅ All SIMD optimizations now work correctly in tests and production
- ✅ BMI2/AVX2/POPCNT acceleration fully functional

Current performance on Intel i7-10700K:

> **Note**: *AVX-512 optimizations require nightly Rust due to experimental intrinsics. All other SIMD optimizations (AVX2, BMI2, POPCNT) work with stable Rust.

| Operation | Performance | vs std::Vec | vs C++ | Security |
|-----------|-------------|-------------|--------|----------|
| FastVec push 10k | 6.78µs | +48% faster | +20% faster | ✅ Memory safe |
| **AutoGrowCircularQueue** | **1.54x** | **+54% faster** | **+54% faster** | ✅ **Ultra-fast (optimized)** |
| SecureMemoryPool alloc | ~18ns | +85% faster | +85% faster | ✅ **Production-ready** |
| Traditional pool alloc | ~15ns | +90% faster | +90% faster | ❌ Unsafe |
| **Advanced Radix Sort 1M u32s** | **~25ms** | **+150% faster** | **+80% faster** | ✅ **Memory safe + SIMD** |
| **Cache-Oblivious Sort 1M u32s** | **O(1 + N/B log N/B)** | **+2-4x SIMD** | **Optimal cache** | ✅ **Memory safe + cache optimal** |
| Suffix array build | O(n) | N/A | Linear vs O(n log n) | ✅ Memory safe |
| Fiber spawn | ~5µs | N/A | New capability | ✅ Memory safe |

### Security & Memory Safety

#### Production-Ready SecureMemoryPool

The **SecureMemoryPool** eliminates critical security vulnerabilities found in traditional memory pool implementations while maintaining high performance:

##### Security Features

- **Use-After-Free Prevention**: Generation counters validate pointer lifetime
- **Double-Free Detection**: Cryptographic validation prevents duplicate deallocations  
- **Memory Corruption Detection**: Guard pages and canary values detect overflow/underflow
- **Thread Safety**: Built-in synchronization without manual Send/Sync annotations
- **RAII Memory Management**: Automatic cleanup eliminates manual deallocation errors
- **Zero-on-Free**: Optional memory clearing for sensitive data protection

##### Performance Features

- **Thread-Local Caching**: Reduces lock contention with per-thread allocation caches
- **Lock-Free Fast Paths**: High-performance allocation for common cases
- **NUMA Awareness**: Optimized allocation for multi-socket systems
- **Batch Operations**: Amortized overhead for bulk allocations

##### Security Guarantees

| Vulnerability | Traditional Pools | SecureMemoryPool |
|---------------|-------------------|------------------|
| Use-after-free | ❌ Possible | ✅ **Prevented** |
| Double-free | ❌ Possible | ✅ **Detected** |
| Memory corruption | ❌ Undetected | ✅ **Detected** |
| Race conditions | ❌ Manual sync required | ✅ **Thread-safe** |
| Manual cleanup | ❌ Error-prone | ✅ **RAII automatic** |

##### Migration Guide

**Before (MemoryPool)**:
```rust
let config = PoolConfig::new(1024, 100, 8);
let pool = MemoryPool::new(config)?;
let ptr = pool.allocate()?;
// Manual deallocation required - error-prone!
pool.deallocate(ptr)?;
```

**After (SecureMemoryPool)**:
```rust
let config = SecurePoolConfig::small_secure();
let pool = SecureMemoryPool::new(config)?;
let ptr = pool.allocate()?;
// Automatic cleanup on drop - no manual deallocation needed!
// Use-after-free and double-free impossible!
```

## C FFI Migration

### Generating C Headers

To generate C header files for FFI bindings:

```bash
cargo build --features ffi
```

This creates `include/zipora.h` with all necessary C declarations and constants.

### Usage

```toml
[dependencies]
zipora = { version = "1.1.1", features = ["ffi"] }
```

```c
#include <zipora.h>

// Vector operations
CFastVec* vec = fast_vec_new();
fast_vec_push(vec, 42);
printf("Length: %zu\n", fast_vec_len(vec));
fast_vec_free(vec);

// Secure memory pools (recommended)
CSecureMemoryPool* pool = secure_memory_pool_new_small();
CSecurePooledPtr* ptr = secure_memory_pool_allocate(pool);
// No manual deallocation needed - automatic cleanup!
secure_pooled_ptr_free(ptr);
secure_memory_pool_free(pool);

// Traditional pools (legacy, less secure)
CMemoryPool* old_pool = memory_pool_new(64 * 1024, 100);
void* chunk = memory_pool_allocate(old_pool);
memory_pool_deallocate(old_pool, chunk);
memory_pool_free(old_pool);

// Error handling
zipora_set_error_callback(error_callback);
if (fast_vec_push(NULL, 42) != CResult_Success) {
    printf("Error: %s\n", zipora_last_error());
}
```

## Cache Layout Optimization Infrastructure

Zipora provides comprehensive cache optimization infrastructure for maximum memory performance across modern hardware architectures:

### Cache-Optimized Memory Allocator

```rust
use zipora::memory::{CacheOptimizedAllocator, CacheLayoutConfig, AccessPattern, PrefetchHint};

// Create cache-optimized allocator with hardware detection
let allocator = CacheOptimizedAllocator::optimal();

// Allocate cache-aligned memory
let ptr = allocator.allocate_aligned(1024, 64, true).unwrap(); // 64-byte aligned, hot data

// Get allocation statistics
let stats = allocator.stats();
println!("Hot allocations: {}, Cache line size: {} bytes", 
         stats.hot_allocations, stats.cache_line_size);

// Issue prefetch hints for predictable access patterns
allocator.prefetch(ptr.as_ptr(), PrefetchHint::T0); // Prefetch to all cache levels
allocator.prefetch_range(ptr.as_ptr(), 1024);       // Prefetch entire range
```

### Cache Hierarchy Detection

```rust
use zipora::memory::{detect_cache_hierarchy, CacheHierarchy};

// Runtime detection of cache hierarchy
let hierarchy = detect_cache_hierarchy();
println!("L1 cache: {} bytes, line size: {} bytes", 
         hierarchy.l1_size, hierarchy.l1_line_size);
println!("L2 cache: {} bytes, L3 cache: {} bytes", 
         hierarchy.l2_size, hierarchy.l3_size);

// Architecture-specific optimizations:
// x86_64: CPUID-based detection with L1/L2/L3 cache information
// ARM64:  /sys filesystem parsing with cache coherency line sizes
// Other:  Sensible defaults with 64-byte cache lines
```

### Hot/Cold Data Separation

```rust
use zipora::memory::{HotColdSeparator, CacheLayoutConfig};

let config = CacheLayoutConfig::new();
let mut separator = HotColdSeparator::new(config);

// Add data with access frequency hints
separator.insert("frequently_accessed".to_string(), 5000); // Hot data
separator.insert("rarely_accessed".to_string(), 10);       // Cold data

// Automatic separation based on access thresholds
let hot_data = separator.hot_slice();   // Cache-line aligned for fast access
let cold_data = separator.cold_slice(); // Compactly stored

// Dynamic rebalancing based on access patterns
separator.reorganize();
let stats = separator.separation_stats();
println!("Hot items: {}, Cold items: {}", stats.hot_items, stats.cold_items);
```

### Cache-Aligned Data Structures

```rust
use zipora::memory::{CacheAlignedVec, AccessPattern};

// Create cache-aligned vector with access pattern optimization
let mut vec = CacheAlignedVec::with_access_pattern(AccessPattern::Sequential);

// Automatic prefetching for sequential access
vec.push(42);  // Triggers prefetch of next cache line
vec.push(24);

// Get element with prefetch hints for random access
let value = vec.get(1); // Prefetches nearby elements for random patterns

// Range access with intelligent prefetching
let slice = vec.slice(0..10).unwrap(); // Prefetches entire range
```

### NUMA-Aware Memory Management

```rust
use zipora::memory::{get_numa_stats, set_current_numa_node, numa_alloc_aligned};

// Get NUMA topology information
let stats = get_numa_stats();
println!("NUMA nodes: {}, current thread node: {:?}", 
         stats.node_count, stats.current_node);

// Set preferred NUMA node for current thread
set_current_numa_node(0).unwrap();

// Allocate memory on specific NUMA node with cache alignment
let ptr = numa_alloc_aligned(4096, 64, 0).unwrap(); // 4KB on node 0, 64-byte aligned

// View per-node statistics
for (node, pool_stats) in &stats.pools {
    println!("Node {}: hit rate {:.1}%, {} bytes allocated", 
             node, pool_stats.hit_rate() * 100.0, pool_stats.allocated_bytes);
}
```

### SIMD Memory Operations with Cache Optimization

```rust
use zipora::memory::{SimdMemOps, CacheLayoutConfig, AccessPattern, PrefetchHint};

// Create SIMD memory operations with cache configuration
let config = CacheLayoutConfig::sequential(); // Optimized for sequential access
let simd_ops = SimdMemOps::with_cache_config(config);

// High-performance memory operations
let src = vec![0u8; 4096];
let mut dst = vec![0u8; 4096];

// Cache-optimized copy with automatic prefetching
simd_ops.copy_cache_optimized(&src, &mut dst).unwrap();

// Cache-optimized comparison with prefetch hints
let result = simd_ops.compare_cache_optimized(&src, &dst);

// Manual prefetch control for predictable patterns
simd_ops.prefetch_range(src.as_ptr(), src.len());
```

### Access Pattern Optimization

```rust
use zipora::memory::{CacheLayoutConfig, AccessPattern};

// Configure for different access patterns
let sequential_config = CacheLayoutConfig::sequential();    // Large prefetch distance
let random_config = CacheLayoutConfig::random();           // Minimal prefetching
let write_heavy_config = CacheLayoutConfig::write_heavy(); // Write-combining optimization
let read_heavy_config = CacheLayoutConfig::read_heavy();   // Aggressive read prefetching

// Access pattern benefits:
// Sequential: 2x prefetch distance, aggressive read-ahead
// Random: Hot/cold separation enabled, minimal prefetching
// WriteHeavy: Write-combining buffers, reduced read prefetching
// ReadHeavy: Maximum prefetch distance, read-optimized caching
// Mixed: Balanced optimization for varied workloads
```

### Cross-Platform Prefetch Support

```rust
use zipora::memory::{SimdMemOps, PrefetchHint};

let ops = SimdMemOps::new();
let data = vec![1u8; 1024];

// Cross-platform prefetch hints:
ops.prefetch(data.as_ptr(), PrefetchHint::T0);  // x86_64: _MM_HINT_T0, ARM64: pldl1keep
ops.prefetch(data.as_ptr(), PrefetchHint::T1);  // x86_64: _MM_HINT_T1, ARM64: pldl1keep  
ops.prefetch(data.as_ptr(), PrefetchHint::T2);  // x86_64: _MM_HINT_T2, ARM64: pldl2keep
ops.prefetch(data.as_ptr(), PrefetchHint::NTA); // x86_64: _MM_HINT_NTA, ARM64: pldl1strm

// Architecture-specific features:
// x86_64: Full _mm_prefetch instruction support with all hint levels
// ARM64: PRFM instructions with appropriate cache level targeting
// Other: Graceful no-op fallback for unsupported architectures
```

## SIMD Framework

Zipora provides a comprehensive SIMD framework with automatic hardware detection and graceful fallbacks:

### SIMD Architecture

```rust
use zipora::simd::{SimdCapabilities, CpuFeatures, SimdOperations};

// Runtime hardware detection
let caps = SimdCapabilities::detect();
println!("AVX2: {}, BMI2: {}, POPCNT: {}", caps.avx2, caps.bmi2, caps.popcnt);

// Adaptive SIMD operations
let data = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
let result = SimdOperations::sum_u32_adaptive(&data); // Uses best available SIMD
```

### SIMD Implementation Guidelines

**Hardware Acceleration Tiers:**
- **Tier 5**: AVX-512 (8x parallel, nightly Rust) - `cargo +nightly build --features avx512`
- **Tier 4**: AVX2 (4x parallel, stable Rust) - Default enabled
- **Tier 3**: BMI2 PDEP/PEXT (bit manipulation) - Runtime detection
- **Tier 2**: POPCNT (population count) - Hardware acceleration
- **Tier 1**: ARM NEON (ARM64 platforms) - Cross-platform
- **Tier 0**: Scalar fallback (portable) - Always available

**Implementation Guidelines:**
```rust
// ✅ Correct SIMD style - Runtime detection with fallbacks
#[cfg(target_arch = "x86_64")]
fn accelerated_operation(data: &[u32]) -> u32 {
    if is_x86_feature_detected!("avx2") {
        unsafe { avx2_implementation(data) }
    } else if is_x86_feature_detected!("sse2") {
        unsafe { sse2_implementation(data) }
    } else {
        scalar_fallback(data)
    }
}

// ✅ ARM support
#[cfg(target_arch = "aarch64")]
fn accelerated_operation(data: &[u32]) -> u32 {
    if std::arch::is_aarch64_feature_detected!("neon") {
        unsafe { neon_implementation(data) }
    } else {
        scalar_fallback(data)
    }
}
```

**Cross-Platform SIMD Pattern:**
- **Always provide scalar fallback** for compatibility
- **Use runtime detection** with `is_x86_feature_detected!`
- **Graceful degradation** across hardware tiers
- **Unsafe blocks isolated** to SIMD intrinsics only
- **Comprehensive testing** on all instruction sets

### SIMD Performance Impact

| Component | SIMD Acceleration | Performance Gain |
|-----------|------------------|------------------|
| **Rank/Select** | AVX2 + BMI2 | **0.3-0.4 Gops/s** (hardware-accelerated) |
| **Radix Sort** | AVX2 digit counting | **4-8x faster** sorting |
| **Cache-Oblivious Sort** | AVX2 + cache prefetch | **2-4x faster** optimal cache complexity |
| **String Processing** | AVX2 UTF-8 validation | **2-4x faster** text processing |
| **Compression** | BMI2 bit operations | **5-10x faster** bit manipulation |
| **Hash Maps** | Cache prefetching | **2-3x fewer** cache misses |
| **Memory Operations** | Cache-optimized SIMD | **2-3x faster** small copies |
| **Cache Optimization** | Hardware detection | **>95% hit rate** for hot data |

### Performance Notes

**Hardware Requirements for Optimal Performance:**
- **CPU**: x86_64 with BMI2, AVX2, and POPCNT support (Intel Haswell+ or AMD Excavator+)
- **Memory**: DDR4-2400+ recommended for cache-sensitive operations
- **Compiler**: Rust 1.88+ with target-cpu=native for maximum SIMD utilization

**Performance Characteristics:**
- **Data Size Dependency**: Performance scales with data size; small datasets (≤100K elements) may fit entirely in cache
- **Pattern Sensitivity**: Sparse vs. dense data patterns can affect performance by 2-3x
- **Hardware Acceleration**: Requires BMI2/AVX2 support; falls back to scalar implementations otherwise
- **Cache Effects**: Larger datasets (>1M elements) may show different performance characteristics due to cache misses

**Benchmark Environment:**
- All measurements taken with hardware acceleration enabled in production builds
- Test platform: AMD CPU with AVX2, BMI2, POPCNT support
- Performance may vary significantly on different hardware configurations

## Features

| Feature | Description | Default | Requirements |
|---------|-------------|---------|--------------|
| `simd` | SIMD optimizations (AVX2, BMI2, POPCNT) | ✅ | Stable Rust |
| `avx512` | AVX-512 optimizations (experimental) | ❌ | **Nightly Rust** |
| `mmap` | Memory-mapped file support | ✅ | Stable Rust |
| `zstd` | ZSTD compression | ✅ | Stable Rust |
| `serde` | Serialization support | ✅ | Stable Rust |
| `lz4` | LZ4 compression | ❌ | Stable Rust |
| `ffi` | C FFI compatibility | ❌ | Stable Rust |

## Build & Test

```bash
# Build
cargo build --release

# Hash map benchmarks
cargo bench --bench hash_maps_bench
cargo bench --bench cache_locality_bench

# Build

# Build with optional features
cargo build --release --features lz4             # Enable LZ4 compression
cargo build --release --features ffi             # Enable C FFI compatibility (generates include/zipora.h)
cargo build --release --features lz4,ffi         # Multiple optional features

# AVX-512 requires nightly Rust (experimental intrinsics)
cargo +nightly build --release --features avx512  # Enable AVX-512 optimizations
cargo +nightly build --release --features avx512,lz4,ffi  # AVX-512 + other features

# Test (1,630+ tests, 97%+ coverage - includes unification of entropy coding implementations)
cargo test --all-features

# Test documentation examples (69 doctests)
cargo test --doc

# Benchmark
cargo bench

# Benchmark with specific features
cargo bench --features lz4

# Rank/Select benchmarks
cargo bench --bench rank_select_bench

# Advanced Radix Sort benchmarks
cargo bench --bench radix_sort_bench
cargo bench --bench advanced_radix_sort_bench
cargo bench --bench string_radix_sort_bench
cargo bench --bench parallel_radix_sort_bench
cargo bench --bench adaptive_sort_bench

# Cache-Oblivious Algorithm benchmarks
cargo bench --bench cache_oblivious_bench
cargo bench --bench cache_bench

# FSA & Trie benchmarks
cargo bench --bench crit_bit_trie_bench
cargo bench --bench patricia_trie_bench
cargo bench --bench double_array_trie_bench
cargo bench --bench compressed_sparse_trie_bench
cargo bench --bench nested_louds_trie_bench
cargo bench --bench comprehensive_trie_benchmarks

# I/O & Serialization benchmarks
cargo bench --bench stream_buffer_bench
cargo bench --bench range_stream_bench
cargo bench --bench zero_copy_bench

# AVX-512 benchmarks (nightly Rust required)
cargo +nightly bench --features avx512

# Examples
cargo run --example basic_usage
cargo run --example succinct_demo
cargo run --example entropy_coding_demo
cargo run --example secure_memory_pool_demo  # SecureMemoryPool security features
cargo run --example config_demo               # Rich Configuration APIs demonstration
```


## License

Licensed under The Bindiego License (BDL), Version 1.0. See [LICENSE](LICENSE) for details.

