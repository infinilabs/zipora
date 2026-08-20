# Specialized Hash Maps

Zipora provides **two production-grade hash map implementations** with different optimization strategies.

## ZiporaHashMap - Strategy-Based Unified Implementation

**Unified hash map** with strategy-based configuration for advanced features including cache locality optimizations, sophisticated collision resolution algorithms, and memory-efficient string arena management.

```rust
use zipora::hash_map::{ZiporaHashMap, ZiporaHashMapConfig, HashStrategy, HashStorageStrategy};

// Default high-performance hash map - constructors return Result
let mut map = ZiporaHashMap::new().unwrap();
map.insert("key", "value").unwrap();
// Features: Optimized for general-purpose use, excellent lookup performance

// String-optimized configuration - memory efficient for string keys
let mut string_map = ZiporaHashMap::with_config(ZiporaHashMapConfig::string_optimized()).unwrap();
string_map.insert("interned", 42).unwrap();
// Features: String interning, prefix caching, SIMD acceleration, arena management
// Best for: Applications with many duplicate string keys

// Small inline configuration - zero allocations for small collections
let mut small_map = ZiporaHashMap::with_config(ZiporaHashMapConfig::small_inline(4)).unwrap();
small_map.insert("inline", 1).unwrap();
// Features: Inline storage for <=N elements, automatic heap fallback
// Best for: Small collections, zero-allocation scenarios

// Cache-optimized configuration - NUMA awareness and prefetching
let mut cache_map = ZiporaHashMap::with_config(ZiporaHashMapConfig::cache_optimized()).unwrap();
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
    storage_strategy: HashStorageStrategy::CacheOptimized {
        cache_line_size: 64,
        numa_aware: true,
        huge_pages: false,
    },
    load_factor: 0.75,
    ..ZiporaHashMapConfig::default()
};
let mut advanced_map = ZiporaHashMap::with_config(config).unwrap();
advanced_map.insert("advanced", "unified configuration").unwrap();
```

## GoldHashMap - Link-Based High-Performance Hash Table

**Production-grade hash table** featuring link-based collision resolution, configurable link types (u32/u64), optional hash caching, and efficient freelist management.

```rust
use zipora::hash_map::{GoldHashMap, GoldHashMapConfig, IterationStrategy};

// Basic usage with default configuration (u32 links, 0.7 load factor)
// insert/remove return Result
let mut map = GoldHashMap::<String, i32>::new();
map.insert("hello".to_string(), 42).unwrap();
assert_eq!(map.get(&"hello".to_string()), Some(&42));

// Update existing entry
map.insert("hello".to_string(), 100).unwrap();
assert_eq!(map.get(&"hello".to_string()), Some(&100));

// Remove entries - efficient freelist management
map.remove(&"hello".to_string()).unwrap();
assert_eq!(map.len(), 0);

// Small map preset - enables hash caching for better performance
let mut small_map = GoldHashMap::<i32, String>::with_config(
    GoldHashMapConfig::small()
);
for i in 0..100 {
    small_map.insert(i, format!("value_{}", i)).unwrap();
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

| Implementation / Configuration | Optimization Focus | Best Use Case |
|--------------------------------|--------------------|---------------|
| **ZiporaHashMap** (default) | Robin Hood probing, general purpose | Standard workloads |
| **ZiporaHashMap** + `string_optimized()` | String interning, arena storage, SIMD string ops | String key deduplication |
| **ZiporaHashMap** + `small_inline(n)` | Inline storage for <=n elements | Small collections, zero allocation |
| **ZiporaHashMap** + `cache_optimized()` | Cache-line alignment, NUMA awareness, prefetching | Cache-sensitive workloads |
| **GoldHashMap** | Link-based collision resolution, hash caching | Lookup-heavy workloads |

Measured, up-to-date benchmark numbers are maintained in [PERFORMANCE.md](PERFORMANCE.md).

## Key Performance Insights

- **GoldHashMap targets lookup-heavy workloads**: link-based collision resolution provides good cache locality and predictable probe behavior
- **Configurable link types** (u32/u64) allow memory vs capacity tradeoff
- **Hash caching** reduces recomputation overhead for small to medium maps
- **Freelist management** enables efficient slot reuse in high-churn workloads
- **Auto GC** prevents memory fragmentation in long-running applications
- **`ZiporaHashMapConfig::string_optimized()` reduces memory usage** through string interning and arena management, at the cost of insertion speed
- **`ZiporaHashMapConfig::small_inline(n)` eliminates allocations** for small collections
- **`ZiporaHashMapConfig::cache_optimized()` delivers cache-aware performance** with prefetching, NUMA awareness, and hot/cold data separation
- **Custom `HashStrategy`/`HashStorageStrategy` combinations** (e.g. Robin Hood with backward shift) give full control over collision handling

## String Arena and Deduplication

```rust
// Advanced string arena with offset-based addressing (integrated into ZiporaHashMap)
let mut string_map = ZiporaHashMap::with_config(ZiporaHashMapConfig::string_optimized()).unwrap();
string_map.insert("shared string", "value1").unwrap();
string_map.insert("shared string", "value2").unwrap(); // Automatic deduplication
let stats = string_map.stats();
println!("Insertions: {}, lookups: {}, collisions: {}",
    stats.insertions, stats.lookups, stats.collisions);

// Cache metrics for cache-optimized maps
let mut cache_map = ZiporaHashMap::with_config(ZiporaHashMapConfig::cache_optimized()).unwrap();
cache_map.insert("cache", "optimized").unwrap();
let metrics = cache_map.cache_metrics();
println!("Cache hit ratio: {:.2}%", metrics.hit_ratio() * 100.0);
```
