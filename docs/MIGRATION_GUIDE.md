# Zipora Unification Migration Guide

This guide helps you migrate from the legacy multiple-implementation approach to the new unified, strategy-based implementations inspired by referenced project's architecture.

## Overview

**Before (Legacy)**: 14+ separate data structure implementations
**After (Unified)**: 2 core implementations with strategy-based configuration

## Benefits of Migration

- **Reduced maintenance burden**: Single implementations to maintain instead of 14+
- **Consistent optimization**: SIMD, cache, and concurrency features applied uniformly
- **Easier testing**: Comprehensive test coverage on fewer, better implementations
- **Better performance**: Focused optimization efforts yield higher quality
- **Cleaner API**: Configuration over separate classes

## Hash Map Migration

### Quick Reference

| Legacy Implementation | Unified Replacement |
|----------------------|-------------------|
| `GoldHashMap::new()` | `ZiporaHashMap::new()` |
| `GoldenRatioHashMap::new()` | `ZiporaHashMap::with_config(ZiporaHashMapConfig::default())` |
| `StringOptimizedHashMap::new()` | `ZiporaHashMap::with_config(ZiporaHashMapConfig::string_optimized())` |
| `SmallHashMap::<K, V, N>::new()` | `ZiporaHashMap::with_config(ZiporaHashMapConfig::small_inline(N))` |
| `CacheOptimizedHashMap::new()` | `ZiporaHashMap::with_config(ZiporaHashMapConfig::cache_optimized())` |
| `AdvancedHashMap::with_strategy(s)` | `ZiporaHashMap::with_config(custom_config)` |

### Detailed Migration Examples

#### Basic Hash Map
```rust
// BEFORE
use zipora::hash_map::GoldHashMap;
let mut map = GoldHashMap::new();
map.insert("key", "value").unwrap();

// AFTER
use zipora::hash_map::ZiporaHashMap;
let mut map = ZiporaHashMap::new(); // Same API!
map.insert("key", "value").unwrap();
```

#### String-Optimized Hash Map
```rust
// BEFORE
use zipora::hash_map::StringOptimizedHashMap;
let mut map = StringOptimizedHashMap::new();
map.insert("hello", 42).unwrap();
let stats = map.string_arena_stats();

// AFTER
use zipora::hash_map::{ZiporaHashMap, ZiporaHashMapConfig};
let mut map = ZiporaHashMap::with_config(ZiporaHashMapConfig::string_optimized());
map.insert("hello", 42).unwrap();
let stats = map.stats(); // Unified stats API
```

#### Small Inline Hash Map
```rust
// BEFORE
use zipora::hash_map::SmallHashMap;
let mut map: SmallHashMap<i32, String, 4> = SmallHashMap::new();
map.insert(1, "one".to_string()).unwrap();
assert!(map.is_inline());

// AFTER
use zipora::hash_map::{ZiporaHashMap, ZiporaHashMapConfig};
let mut map = ZiporaHashMap::with_config(ZiporaHashMapConfig::small_inline(4));
map.insert(1, "one".to_string()).unwrap();
// Check storage type via config or stats
```

#### Cache-Optimized Hash Map
```rust
// BEFORE
use zipora::hash_map::CacheOptimizedHashMap;
let mut map = CacheOptimizedHashMap::new();
map.insert("key", "value").unwrap();
let metrics = map.cache_metrics();

// AFTER
use zipora::hash_map::{ZiporaHashMap, ZiporaHashMapConfig};
let mut map = ZiporaHashMap::with_config(ZiporaHashMapConfig::cache_optimized());
map.insert("key", "value").unwrap();
let metrics = map.cache_metrics(); // Same API!
```

#### Advanced Configuration
```rust
// BEFORE
use zipora::hash_map::{AdvancedHashMap, CollisionStrategy};
let strategy = CollisionStrategy::RobinHood { /* config */ };
let mut map = AdvancedHashMap::with_collision_strategy(strategy);

// AFTER
use zipora::hash_map::{ZiporaHashMap, ZiporaHashMapConfig, HashStrategy};
let config = ZiporaHashMapConfig {
    hash_strategy: HashStrategy::RobinHood {
        max_probe_distance: 64,
        variance_reduction: true,
        backward_shift: true,
    },
    // ... other configuration
    ..ZiporaHashMapConfig::default()
};
let mut map = ZiporaHashMap::with_config(config);
```

## Trie Migration

### Quick Reference

| Legacy Implementation | Unified Replacement |
|----------------------|-------------------|
| `PatriciaTrie::new()` | `ZiporaTrie::new()` |
| `CritBitTrie::new()` | `ZiporaTrie::with_config(ZiporaTrieConfig::string_specialized())` |
| `DoubleArrayTrie::new()` | `ZiporaTrie::with_config(ZiporaTrieConfig::concurrent_high_performance(pool))` |
| `LoudsTrie::new()` | `ZiporaTrie::with_config(ZiporaTrieConfig::space_optimized())` |
| `NestedLoudsTrie::new()` | `ZiporaTrie::with_config(ZiporaTrieConfig::space_optimized())` |
| `CompressedSparseTrie::new()` | `ZiporaTrie::with_config(ZiporaTrieConfig::sparse_optimized())` |

### Detailed Migration Examples

#### Basic Patricia Trie
```rust
// BEFORE
use zipora::fsa::{PatriciaTrie, Trie};
let mut trie = PatriciaTrie::new();
trie.insert(b"hello").unwrap();
assert!(trie.contains(b"hello"));

// AFTER
use zipora::fsa::{ZiporaTrie, Trie};
let mut trie = ZiporaTrie::new(); // Same API!
trie.insert(b"hello").unwrap();
assert!(trie.contains(b"hello"));
```

#### Critical-Bit Trie for Strings
```rust
// BEFORE
use zipora::fsa::{CritBitTrie, Trie};
let mut trie = CritBitTrie::new();
trie.insert(b"string").unwrap();
trie.insert(b"structure").unwrap();

// AFTER
use zipora::fsa::{ZiporaTrie, ZiporaTrieConfig, Trie};
let mut trie = ZiporaTrie::with_config(ZiporaTrieConfig::string_specialized());
trie.insert(b"string").unwrap();
trie.insert(b"structure").unwrap();
```

#### Space-Optimized LOUDS Trie
```rust
// BEFORE
use zipora::fsa::{NestedLoudsTrie, NestingConfig, Trie};
use zipora::RankSelectInterleaved256;

let config = NestingConfig::builder()
    .max_levels(4)
    .fragment_compression_ratio(0.3)
    .build()?;
let mut trie = NestedLoudsTrie::<RankSelectInterleaved256>::with_config(config)?;

// AFTER
use zipora::fsa::{ZiporaTrie, ZiporaTrieConfig, Trie};
let mut trie = ZiporaTrie::with_config(ZiporaTrieConfig::space_optimized());
// Nesting and compression configured automatically
```

#### High-Performance Concurrent Trie
```rust
// BEFORE
use zipora::fsa::{DoubleArrayTrie, DoubleArrayTrieConfig, Trie};
use zipora::memory::{SecureMemoryPool, SecurePoolConfig};

let pool = std::sync::Arc::new(SecureMemoryPool::new(SecurePoolConfig::default()));
let config = DoubleArrayTrieConfig {
    use_memory_pool: true,
    enable_simd: true,
    cache_aligned: true,
    // ...
};
let mut trie = DoubleArrayTrie::with_config(config);

// AFTER
use zipora::fsa::{ZiporaTrie, ZiporaTrieConfig, Trie};
use zipora::memory::{SecureMemoryPool, SecurePoolConfig};

let pool = std::sync::Arc::new(SecureMemoryPool::new(SecurePoolConfig::default()));
let mut trie = ZiporaTrie::with_config(
    ZiporaTrieConfig::concurrent_high_performance(pool)
);
// SIMD, cache alignment, and concurrency configured automatically
```

#### Advanced Custom Configuration
```rust
// AFTER (New capabilities)
use zipora::fsa::{ZiporaTrie, ZiporaTrieConfig, TrieStrategy, CompressionStrategy};

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
    // ...
};
let mut trie = ZiporaTrie::with_config(config);
```

## Common Migration Patterns

### 1. Configuration Builder Pattern
```rust
// Create a custom configuration for specific needs
use zipora::hash_map::{ZiporaHashMapConfig, HashStrategy, StorageStrategy, OptimizationStrategy};

let config = ZiporaHashMapConfig {
    hash_strategy: HashStrategy::RobinHood {
        max_probe_distance: 32,
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
    load_factor: 0.7,
    ..ZiporaHashMapConfig::default()
};

let map = ZiporaHashMap::with_config(config);
```

### 2. Memory Pool Integration
```rust
use zipora::memory::{SecureMemoryPool, SecurePoolConfig};
use zipora::hash_map::{ZiporaHashMapConfig, StorageStrategy};

let pool = std::sync::Arc::new(SecureMemoryPool::new(SecurePoolConfig::default()));
let config = ZiporaHashMapConfig {
    storage_strategy: StorageStrategy::PoolAllocated {
        pool: pool.clone(),
        chunk_size: 1024,
    },
    ..ZiporaHashMapConfig::default()
};

let map = ZiporaHashMap::with_config(config);
```

### 3. Type Aliases for Compatibility
```rust
// Create type aliases for common configurations
use zipora::hash_map::{ZiporaHashMap, ZiporaHashMapConfig};
use zipora::fsa::{ZiporaTrie, ZiporaTrieConfig};

type FastHashMap<K, V> = ZiporaHashMap<K, V>;
type StringHashMap<V> = ZiporaHashMap<String, V>;
type InlineHashMap<K, V> = ZiporaHashMap<K, V>; // With small_inline config

type FastTrie = ZiporaTrie;
type StringTrie = ZiporaTrie; // With string_specialized config
type CompactTrie = ZiporaTrie; // With space_optimized config

// Usage
let mut map: FastHashMap<String, i32> = ZiporaHashMap::new();
let mut trie: StringTrie = ZiporaTrie::with_config(ZiporaTrieConfig::string_specialized());
```

## Breaking Changes

### API Changes
1. **Configuration over separate types**: Use `with_config()` instead of different struct types
2. **Unified statistics**: All implementations now use consistent stats APIs
3. **Strategy-based customization**: Advanced configuration through strategy enums

### Import Changes
```rust
// BEFORE
use zipora::hash_map::{GoldHashMap, StringOptimizedHashMap, SmallHashMap};
use zipora::fsa::{PatriciaTrie, CritBitTrie, DoubleArrayTrie};

// AFTER
use zipora::hash_map::{ZiporaHashMap, ZiporaHashMapConfig};
use zipora::fsa::{ZiporaTrie, ZiporaTrieConfig};
```

### Deprecation Timeline
- **v0.8.0**: Unified implementations introduced, legacy implementations deprecated
- **v0.9.0**: Legacy implementations will show stronger deprecation warnings
- **v1.0.0**: Legacy implementations will be removed entirely

## Migration Checklist

### Phase 1: Update Imports
- [ ] Replace legacy hash map imports with `ZiporaHashMap`
- [ ] Replace legacy trie imports with `ZiporaTrie`
- [ ] Add configuration imports (`ZiporaHashMapConfig`, `ZiporaTrieConfig`)

### Phase 2: Update Construction
- [ ] Replace `TypedHashMap::new()` with `ZiporaHashMap::with_config(config)`
- [ ] Replace `TypedTrie::new()` with `ZiporaTrie::with_config(config)`
- [ ] Use appropriate configuration functions (e.g., `cache_optimized()`, `string_specialized()`)

### Phase 3: Update Configuration
- [ ] Migrate custom configurations to strategy enums
- [ ] Update memory pool integrations
- [ ] Review and optimize configuration choices

### Phase 4: Test and Validate
- [ ] Ensure compilation succeeds
- [ ] Run existing tests
- [ ] Validate performance characteristics
- [ ] Update documentation and examples

## Performance Notes

### Expected Performance Changes
- **Same or better performance**: Unified implementations preserve all optimizations
- **More consistent performance**: Same optimizations applied across all use cases
- **Better scalability**: Focused optimization efforts on fewer implementations

### Benchmarking
```rust
// Use the same APIs for benchmarking
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use zipora::hash_map::ZiporaHashMap;

fn benchmark_insertion(c: &mut Criterion) {
    c.bench_function("unified_hash_map_insert", |b| {
        b.iter(|| {
            let mut map = ZiporaHashMap::new();
            for i in 0..1000 {
                map.insert(black_box(i), black_box(i * 2)).unwrap();
            }
        })
    });
}
```

## Getting Help

### Compilation Errors
1. Check import statements
2. Verify configuration syntax
3. Ensure feature flags are correct

### Runtime Issues
1. Review configuration choices
2. Check memory pool setup
3. Validate strategy parameters

### Performance Concerns
1. Profile with unified implementations
2. Experiment with different configurations
3. Use built-in metrics and statistics

## Examples Repository

See `examples/migration/` for complete working examples of:
- Basic hash map migration
- Advanced trie configuration
- Memory pool integration
- Custom strategy implementation
- Performance comparison tests

## Advanced Usage

### Custom Strategy Implementation
```rust
use zipora::hash_map::strategy_traits::CollisionResolutionStrategy;

struct CustomStrategy;

impl<K, V> CollisionResolutionStrategy<K, V> for CustomStrategy {
    // Implement custom collision resolution logic
    // ...
}

// Use with unified hash map
let strategy = CustomStrategy;
// Configure with strategy traits for advanced usage
```

### Integration with Existing Code
```rust
// Gradual migration approach
#[cfg(feature = "legacy")]
use zipora::hash_map::GoldHashMap as HashMap;

#[cfg(not(feature = "legacy"))]
use zipora::hash_map::ZiporaHashMap as HashMap;

// Use HashMap throughout codebase
let mut map = HashMap::new();
```

This migration guide ensures a smooth transition from Zipora's multiple-implementation approach to the unified, referenced project-inspired architecture while preserving all advanced optimizations and improving maintainability.