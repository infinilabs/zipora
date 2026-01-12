# Specialized Hash Maps

Zipora provides **two production-grade hash map implementations** with different optimization strategies.

## ZiporaHashMap - Strategy-Based Unified Implementation

**Unified hash map** with strategy-based configuration for advanced features including cache locality optimizations, sophisticated collision resolution algorithms, and memory-efficient string arena management.

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
// Features: Inline storage for <=N elements, automatic heap fallback
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

## GoldHashMap - Link-Based High-Performance Hash Table

**Production-grade hash table** inspired by Terark's gold_hash_map, featuring link-based collision resolution, configurable link types (u32/u64), optional hash caching, and efficient freelist management.

```rust
use zipora::hash_map::{GoldHashMap, GoldHashMapConfig, IterationStrategy};

// Basic usage with default configuration (u32 links, 0.7 load factor)
let mut map = GoldHashMap::<String, i32>::new();
map.insert("hello".to_string(), 42);
assert_eq!(map.get(&"hello".to_string()), Some(&42));

// Update existing entry
map.insert("hello".to_string(), 100);
assert_eq!(map.get(&"hello".to_string()), Some(&100));

// Remove entries - efficient freelist management
map.remove(&"hello".to_string());
assert_eq!(map.len(), 0);

// Small map preset - enables hash caching for better performance
let mut small_map = GoldHashMap::<i32, String>::with_config(
    GoldHashMapConfig::small()
);
for i in 0..100 {
    small_map.insert(i, format!("value_{}", i));
}
assert!(small_map.is_hash_cached()); // Hash caching enabled

// Large map preset - uses u64 links, enables auto GC
let mut large_map = GoldHashMap::<i32, Vec<u8>, u64>::with_config(
    GoldHashMapConfig::large()
);
// Handles millions of entries efficiently

// High-churn workload - optimized for frequent insert/delete
let mut churn_map = GoldHashMap::<String, Vec<u8>>::with_config(
    GoldHashMapConfig::high_churn()
);
// Lower load factor (0.6), auto GC enabled, efficient slot reuse

// Safe iteration - skips deleted entries (default)
for (key, value) in map.iter() {
    println!("{}: {}", key, value);
}

// Fast iteration - no validity checks (use only if no deletions)
for (key, value) in map.iter_fast() {
    // Direct array access, maximum performance
}

// Custom configuration - full control
let config = GoldHashMapConfig {
    initial_capacity: 1024,
    load_factor: 0.8,
    enable_hash_cache: true,
    enable_auto_gc: false,
    enable_freelist_reuse: true,
    default_iteration_strategy: IterationStrategy::Safe,
};
let mut custom_map = GoldHashMap::<i64, String>::with_config(config);

// Runtime hash caching toggle
custom_map.set_hash_caching(false);  // Disable to save memory
assert!(!custom_map.is_hash_cached());

// Compact deleted entries (invalidates indices but improves iteration)
custom_map.revoke_deleted();  // Manual GC
assert_eq!(custom_map.deleted_count(), 0);

// Check capacity and load factor
println!("Capacity: {}", custom_map.capacity());
println!("Load factor: {:.2}", custom_map.load_factor());
```

## Hash Map Performance Comparison

Based on comprehensive benchmarks comparing all hash map implementations:

| Hash Map Type | Insertion | Lookup | Best Use Case |
|---------------|-----------|--------|---------------|
| **std::HashMap** | **73-104 Melem/s** | 91-104 Melem/s | Standard Rust operations |
| **GoldHashMap** | 71-77 Melem/s | **241-342 Melem/s** | **Lookup-heavy workloads** |
| **GoldenRatioHashMap** | 55-70 Melem/s | 110-322 Melem/s | **Memory-efficient growth** |
| **StringOptimizedHashMap** | 5.6-6.0 Melem/s* | Variable | **String key deduplication** |
| **SmallHashMap<T,V,N>** | Variable | Variable | **<=N elements, zero allocation** |
| **AdvancedHashMap** | 60-80 Melem/s | 200-280 Melem/s | **Sophisticated collision resolution** |
| **CacheOptimizedHashMap** | 45-65 Melem/s | 180-250 Melem/s | **Cache-line aligned with NUMA** |

*StringOptimizedHashMap trades speed for memory efficiency through string interning

## Key Performance Insights

- **GoldHashMap excels at lookups** with 2-3x better performance than std::HashMap
- **Link-based collision resolution** provides excellent cache locality and predictable performance
- **Configurable link types** (u32/u64) allow memory vs capacity tradeoff
- **Hash caching** reduces recomputation overhead for small to medium maps
- **Freelist management** enables efficient slot reuse in high-churn workloads
- **Auto GC** prevents memory fragmentation in long-running applications
- **GoldenRatioHashMap provides the best balance** of memory efficiency and performance
- **StringOptimizedHashMap reduces memory usage** at the cost of insertion speed
- **SmallHashMap eliminates allocations** for small collections
- **AdvancedHashMap provides sophisticated collision handling** with Robin Hood hashing, chaining, and Hopscotch algorithms
- **CacheOptimizedHashMap delivers cache-aware performance** with prefetching, NUMA awareness, and hot/cold data separation
- **Advanced string arena management** enables efficient memory usage with offset-based addressing and deduplication

## String Arena and Deduplication

```rust
// Advanced string arena with offset-based addressing (integrated into ZiporaHashMap)
let mut string_map = ZiporaHashMap::with_config(ZiporaHashMapConfig::string_optimized());
string_map.insert("shared string", "value1").unwrap();
string_map.insert("shared string", "value2").unwrap(); // Automatic deduplication
let stats = string_map.stats();
println!("Deduplication ratio: {:.2}%", stats.deduplication_ratio * 100.0);

// Cache metrics for cache-optimized maps
let mut cache_map = ZiporaHashMap::with_config(ZiporaHashMapConfig::cache_optimized());
cache_map.insert("cache", "optimized").unwrap();
let metrics = cache_map.cache_metrics();
println!("Cache hit ratio: {:.2}%", metrics.hit_ratio() * 100.0);
```
