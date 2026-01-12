# Configuration APIs

Zipora provides a comprehensive configuration system that enables fine-grained control over data structures, algorithms, and performance characteristics.

## Key Features

- **Trait-Based Design**: Consistent `Config` trait with validation, serialization, and preset methods
- **Builder Patterns**: Fluent configuration building with method chaining and compile-time validation
- **Environment Integration**: Automatic parsing from environment variables with custom prefixes
- **Preset Configurations**: Performance, Memory, Realtime, and Balanced presets for different use cases
- **JSON Serialization**: Save and load configurations with comprehensive serde support
- **Validation Framework**: Built-in validation with detailed error messages and suggestions
- **Type Safety**: Compile-time checks for configuration parameter ranges and combinations

## Configuration Types

The system provides rich configuration for all major components:

- **`NestLoudsTrieConfig`**: 20+ parameters for trie construction, compression, optimization, memory management
- **`MemoryConfig`**: Pool allocation strategies, NUMA settings, cache optimization, security features
- **`BlobStoreConfig`**: Compression algorithms, block sizes, caching, and I/O optimization
- **`CompressionConfig`**: Algorithm selection, compression levels, real-time constraints
- **`CacheConfig`**: Cache sizes, prefetching strategies, line size optimization
- **`SIMDConfig`**: Hardware acceleration settings (AVX2, BMI2, SIMD instruction sets)

## Usage Examples

### Basic Configuration with Defaults

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

### Using Configuration Presets

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

### Builder Pattern Configuration

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

### Memory Configuration with Advanced Features

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
        size_threshold: 2 * 1024 * 1024,                    // Use huge pages for >=2MB
    })
    .alignment(64)                                          // 64-byte cache line alignment
    .num_pools(16)                                          // 16 separate pools
    .enable_protection(true)                                // Memory protection features
    .enable_compaction(false)                               // Disable for real-time
    .build()?;
```

### Environment Variable Integration

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

### Configuration Persistence

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
        // Output: "nest level must be between 1 and 16;
        //          compression level must be between 0 and 22; load factor must be between 0.0 and 1.0"
    }
}
```

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

### Performance Impact

- **Memory Access**: 2-3x faster through reduced cache misses
- **Cache Optimization**: >95% hit rate for hot data, automatic cache hierarchy adaptation
- **SIMD Memory Operations**: 2-3x faster small copies (<=64 bytes), 1.5-2x faster medium copies
- **Sequential Processing**: 4-5x improvements with prefetch optimization
- **Multi-threaded**: Significant reduction in false sharing overhead
- **NUMA Systems**: 20-40% improvements through local allocation

## Best Practices

1. **Use Presets**: Start with presets and customize only specific parameters
2. **Validate Early**: Always validate configurations before use
3. **Environment Integration**: Use environment variables for deployment-specific settings
4. **Persist Configurations**: Save working configurations for reproducible builds
5. **Monitor Performance**: Enable statistics during development, disable in production
6. **Hardware Awareness**: Use automatic detection for cache line sizes and CPU features
