# Specialized Hash Maps

Zipora provides **two production-grade hash map implementations** with different optimization strategies.

## ZiporaHashMap - Strategy-Based Unified Implementation

**Unified hash map** with SwissTable 16-way SSE2 group probing by default and inline small-map storage (`small_inline`) for small collections.

```rust
use zipora::hash_map::{ZiporaHashMap, ZiporaHashMapConfig, HashStrategy, HashStorageStrategy};

// Default high-performance hash map (SwissTable 16-way SSE2 group probing)
let mut map: ZiporaHashMap<&str, &str> = ZiporaHashMap::new().unwrap();
map.insert("key", "value").unwrap();
assert_eq!(map.get(&"key"), Some(&"value"));

// Pre-allocated capacity
let mut sized_map: ZiporaHashMap<&str, i32> = ZiporaHashMap::with_capacity(64).unwrap();
sized_map.insert("sized", 42).unwrap();

// Small inline configuration - inline storage for <=N elements before spilling to StandardTable
let mut small_map: ZiporaHashMap<&str, i32> =
    ZiporaHashMap::with_config(ZiporaHashMapConfig::small_inline(4)).unwrap();
small_map.insert("inline", 1).unwrap();
assert_eq!(small_map.get(&"inline"), Some(&1));

// Custom configuration with Standard storage and custom load factor
let config = ZiporaHashMapConfig {
    hash_strategy: HashStrategy::RobinHood {
        max_probe_distance: 64,
        variance_reduction: true,
        backward_shift: true,
    },
    storage_strategy: HashStorageStrategy::Standard {
        initial_capacity: 64,
        growth_factor: 2.0,
    },
    load_factor: 0.75,
    ..ZiporaHashMapConfig::default()
};
let mut advanced_map: ZiporaHashMap<&str, &str> = ZiporaHashMap::with_config(config).unwrap();
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
| **ZiporaHashMap** (default) | SwissTable 16-way SSE2 group probing | Standard workloads |
| **ZiporaHashMap** + `small_inline(n)` | Inline storage for <=n elements | Small collections, zero allocation |
| **GoldHashMap** | Link-based collision resolution, hash caching | Lookup-heavy workloads |

Measured, up-to-date benchmark numbers are maintained in [PERFORMANCE.md](PERFORMANCE.md).

## Key Performance Insights

- **`ZiporaHashMap` (default) uses SwissTable 16-way SSE2 group probing**: 1-byte H2 control tags (`ctrl: FastVec<u8>`) + unboxed `(K, V)` slots match `hashbrown`/`std` memory footprint while supporting pluggable strategies
- **GoldHashMap targets lookup-heavy workloads**: link-based collision resolution provides good cache locality and predictable probe behavior
- **Configurable link types** (u32/u64) allow memory vs capacity tradeoff
- **Hash caching** reduces recomputation overhead for small to medium maps
- **Freelist management** enables efficient slot reuse in high-churn workloads
- **Auto GC** prevents memory fragmentation in long-running applications
- **`ZiporaHashMapConfig::small_inline(n)` eliminates heap allocations** for small collections before spilling to the standard table
- **Custom `HashStrategy`/`HashStorageStrategy` combinations** allow tuning initial capacity and load factor

## Runtime Statistics and Cache Metrics

```rust
use zipora::hash_map::{ZiporaHashMap, ZiporaHashMapConfig};

// Track insertions, lookups, and collisions
let mut map: ZiporaHashMap<&str, &str> = ZiporaHashMap::new().unwrap();
map.insert("shared string", "value1").unwrap();
map.insert("shared string", "value2").unwrap(); // Overwrites existing key
assert_eq!(map.get(&"shared string"), Some(&"value2"));
let stats = map.stats();
println!("Insertions: {}, lookups: {}, collisions: {}",
    stats.insertions, stats.lookups, stats.collisions);

// Cache metrics tracking on inline/standard maps
let mut small_map: ZiporaHashMap<&str, &str> =
    ZiporaHashMap::with_config(ZiporaHashMapConfig::small_inline(8)).unwrap();
small_map.insert("cache", "tracked").unwrap();
let metrics = small_map.cache_metrics();
println!("Cache hit ratio: {:.2}%", metrics.hit_ratio() * 100.0);
```
