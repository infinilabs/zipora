# Zipora

[![Build Status](https://github.com/infinilabs/zipora/workflows/CI/badge.svg)](https://github.com/infinilabs/zipora/actions)
[![License](https://img.shields.io/badge/license-BDL--1.0-blue.svg)](LICENSE)
[![Rust Version](https://img.shields.io/badge/rust-1.88+-orange.svg)](https://www.rust-lang.org)

High-performance Rust data structures and compression algorithms with memory safety guarantees.

## Features

- **🚀 High Performance**: Zero-copy operations, SIMD optimizations (AVX2, AVX-512*), cache-friendly layouts
- **🛡️ Memory Safety**: Eliminates segfaults, buffer overflows, use-after-free bugs
- **🧠 Secure Memory Management**: Production-ready memory pools with thread safety, RAII, and vulnerability prevention
- **💾 Blob Storage**: Advanced storage systems including trie-based indexing and offset-based compression
- **📦 Specialized Containers**: Production-ready containers with 40-90% memory/performance improvements
- **🗂️ Specialized Hash Maps**: Golden ratio optimized, string-optimized, and small inline maps with superior performance
- **🌲 Advanced Tries**: LOUDS, Critical-Bit (with BMI2 acceleration), and Patricia tries with rank/select operations, hardware-accelerated path compression, and sophisticated nesting strategies
- **🔒 Version-Based Synchronization**: Advanced token and version sequence management for safe concurrent FSA/Trie access
- **🔗 Low-Level Synchronization**: Linux futex integration, thread-local storage, atomic operations framework
- **⚡ Fiber Concurrency**: High-performance async/await with work-stealing, I/O integration, cooperative multitasking
- **📡 Advanced Serialization**: Comprehensive components with smart pointers, endian handling, version management
- **🗜️ Advanced Compression Framework**: PA-Zip dictionary compression, contextual Huffman (Order-1/Order-2), 64-bit rANS with parallel variants, FSE with ZSTD optimizations, hardware-accelerated bit operations
- **🔄 Real-time Compression**: Adaptive algorithms with strict latency guarantees
- **🔌 C FFI Support**: Complete C API for migration from C++
- **🎚️ Five-Level Concurrency Management**: Graduated concurrency control with adaptive selection

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

## Quick Start

```toml
[dependencies]
zipora = "1.1.1"

# Or with optional features
zipora = { version = "1.1.1", features = ["lz4", "ffi"] }

# AVX-512 requires nightly Rust (experimental intrinsics)
zipora = { version = "1.1.1", features = ["avx512", "lz4", "ffi"] }  # nightly only
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

// Advanced tries
let mut trie = LoudsTrie::new();
trie.insert(b"hello").unwrap();
assert!(trie.contains(b"hello"));

// Critical Bit Trie - space-efficient radix tree with binary decisions
let mut crit_bit = CritBitTrie::new();
crit_bit.insert(b"hello").unwrap();
crit_bit.insert(b"help").unwrap();
crit_bit.insert(b"world").unwrap();
assert!(crit_bit.contains(b"hello"));
assert!(crit_bit.contains(b"help"));
assert!(!crit_bit.contains(b"he")); // Prefix compression

// Space-Optimized Critical Bit Trie with BMI2 hardware acceleration
let mut optimized_trie = SpaceOptimizedCritBitTrie::new();
optimized_trie.insert(b"efficient").unwrap();
optimized_trie.insert(b"effective").unwrap();
let stats = optimized_trie.stats();
println!("Memory usage: {} bytes, {} bits per key", stats.memory_usage, stats.bits_per_key);

// Patricia Trie with hardware acceleration
let mut patricia = PatriciaTrie::new();
patricia.insert(b"hello").unwrap();
patricia.insert(b"help").unwrap();
assert!(patricia.contains(b"hello"));
assert!(patricia.contains(b"help"));
assert!(!patricia.contains(b"he")); // Path compression

// Hash maps - multiple specialized implementations
let mut map = GoldHashMap::new();
map.insert("key", "value").unwrap();

// Golden ratio optimized hash map (15-20% better memory efficiency)
let mut golden_map = GoldenRatioHashMap::new();
golden_map.insert("optimal", "growth").unwrap();

// String-optimized hash map with interning (memory efficient for string keys)
let mut string_map = StringOptimizedHashMap::new();
string_map.insert("interned", 42).unwrap();

// Small hash map with inline storage (zero allocations for ≤N elements)
let mut small_hash_map: SmallHashMap<&str, i32, 4> = SmallHashMap::new();
small_hash_map.insert("inline", 1).unwrap();

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

Zipora provides four specialized hash map implementations designed for different use cases and performance characteristics:

```rust
use zipora::{GoldHashMap, GoldenRatioHashMap, StringOptimizedHashMap, SmallHashMap};

// Original high-performance hash map (existing)
let mut gold_map = GoldHashMap::new();
gold_map.insert("key", "value").unwrap();
// Best for: General-purpose use, excels at lookups (241-342 Melem/s)

// Golden ratio optimized hash map (NEW - 15-20% better memory efficiency)
let mut golden_map = GoldenRatioHashMap::new();
golden_map.insert("optimal", "growth").unwrap();
// Features: Golden ratio growth strategy (1.618x), FaboHashCombine hash function
// Best for: Memory-constrained applications, sustained performance (55-70 Melem/s)

// String-optimized hash map with interning (NEW - memory efficient for strings)
let mut string_map = StringOptimizedHashMap::new();
string_map.insert("interned", 42).unwrap();
// Features: String interning, prefix caching, SIMD acceleration ready
// Best for: Applications with many duplicate string keys, memory optimization

// Small hash map with inline storage (NEW - zero allocations for small maps)
let mut small_hash_map: SmallHashMap<&str, i32, 4> = SmallHashMap::new();
small_hash_map.insert("inline", 1).unwrap();
// Features: Inline storage for ≤N elements, automatic heap fallback
// Best for: Small collections, zero-allocation scenarios
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

*StringOptimizedHashMap trades speed for memory efficiency through string interning

### Key Performance Insights

- **GoldHashMap excels at lookups** with 2-3x better performance than std::HashMap
- **GoldenRatioHashMap provides the best balance** of memory efficiency and performance
- **Capacity optimizations improved GoldHashMap by up to 60%** in benchmarks
- **StringOptimizedHashMap reduces memory usage** at the cost of insertion speed
- **SmallHashMap eliminates allocations** for small collections

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

### Advanced Tries

```rust
use zipora::{CritBitTrie, SpaceOptimizedCritBitTrie, PatriciaTrie, DoubleArrayTrie, 
            CompressedSparseTrie, NestedLoudsTrie, ConcurrencyLevel, ReaderToken, 
            WriterToken, RankSelectInterleaved256, PatriciaConfig};

// Critical Bit Trie - Space-efficient radix tree with binary critical bit decisions
let mut crit_bit = CritBitTrie::new();
crit_bit.insert(b"cat").unwrap();
crit_bit.insert(b"car").unwrap();
crit_bit.insert(b"card").unwrap();

// Efficient lookups with O(m) complexity where m is key length
assert!(crit_bit.contains(b"cat"));
assert!(crit_bit.contains(b"car"));
assert!(crit_bit.contains(b"card"));
assert!(!crit_bit.contains(b"ca")); // Prefix compression means partial matches aren't found

// Prefix iteration for hierarchical data
for key in crit_bit.iter_prefix(b"car") {
    println!("Found key with 'car' prefix: {:?}", String::from_utf8_lossy(&key));
}

// Space-Optimized Critical Bit Trie with BMI2 hardware acceleration
let mut optimized_trie = SpaceOptimizedCritBitTrie::optimized_for_strings().unwrap();
optimized_trie.insert(b"efficient").unwrap();
optimized_trie.insert(b"effective").unwrap();
optimized_trie.insert(b"engine").unwrap();

// Check BMI2 acceleration status
if optimized_trie.bmi2_acceleration_enabled() {
    println!("BMI2 hardware acceleration is active for 5-10x faster operations");
}

// Advanced space optimization statistics
let stats = optimized_trie.stats();
let (compression_ratio, avg_key_len, cache_miss_est) = optimized_trie.compression_stats();
println!("Memory usage: {} bytes, {:.2} bits per key", stats.memory_usage, stats.bits_per_key);
println!("Compression ratio: {:.1}%, Average key length: {:.1}", 
         compression_ratio * 100.0, avg_key_len);

// Runtime optimization for access patterns
optimized_trie.optimize().unwrap();

// Patricia Trie - Sophisticated radix tree with advanced concurrency and hardware acceleration
let mut patricia = PatriciaTrie::new();
patricia.insert(b"hello").unwrap();
patricia.insert(b"help").unwrap();
patricia.insert(b"world").unwrap();

// O(m) lookup performance with path compression and SIMD acceleration
assert!(patricia.contains(b"hello"));
assert!(patricia.contains(b"help"));
assert!(!patricia.contains(b"he")); // Path compression means partial matches aren't found

// Advanced configuration with hardware optimizations
let config = PatriciaConfig::performance_optimized(); // BMI2 + SIMD + concurrency
let mut optimized_trie = PatriciaTrie::with_config(config);

// Concurrent operations with token-based safety
let config = PatriciaConfig {
    concurrency_level: ConcurrencyLevel::OneWriteMultiRead,
    use_bmi2: true,        // Hardware acceleration with BMI2 instructions
    use_simd: true,        // SIMD string comparisons
    alignment: 64,         // Cache-line alignment for performance
    ..Default::default()
};
let concurrent_trie = PatriciaTrie::with_config(config);
let read_token = concurrent_trie.acquire_read_token();
let write_token = concurrent_trie.acquire_write_token();

// Thread-safe operations
concurrent_trie.lookup_with_token(b"hello", &read_token);
concurrent_trie.insert_with_token(b"new_key", &write_token).unwrap();

// Prefix iteration with lexicographic ordering
for key in patricia.iter_prefix(b"hel") {
    println!("Found key: {:?}", key);
}

// Comprehensive statistics and performance monitoring
let stats = patricia.stats();
println!("Memory usage: {} bytes", stats.memory_usage);
println!("Keys: {}, Nodes: {}", stats.num_keys, stats.num_states);
println!("Bits per key: {:.2}", stats.bits_per_key);
println!("Path compression ratio: {:.1}%", (1.0 - stats.avg_depth / stats.max_depth as f64) * 100.0);

// Double Array Trie - Constant-time O(1) state transitions
let mut dat = DoubleArrayTrie::new();
dat.insert(b"computer").unwrap();
dat.insert(b"computation").unwrap();
dat.insert(b"compute").unwrap();

// O(1) lookup performance - 2-3x faster than hash maps for dense key sets
assert!(dat.contains(b"computer"));
assert_eq!(dat.num_keys(), 3);
let stats = dat.get_statistics();
println!("Memory usage: {} bytes per key", stats.memory_usage / stats.num_keys);

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

World-Class Succinct Data Structures - Zipora provides 14 specialized rank/select variants including 6 cutting-edge implementations with comprehensive SIMD optimizations, hardware acceleration, multi-dimensional support, and sophisticated mixed implementations:

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

// SIMD bulk operations with runtime optimization
let bit_data = bv.blocks().to_vec();
let test_positions = vec![100, 200, 300, 400, 500];
let simd_ranks = bulk_rank1_simd(&bit_data, &test_positions);
```

### Sorting & Search Algorithms

```rust
use zipora::{SuffixArray, RadixSort, MultiWayMerge, 
            ReplaceSelectSort, ReplaceSelectSortConfig, LoserTree, LoserTreeConfig,
            ExternalSort, EnhancedSuffixArray, LcpArray};

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

// Tournament Tree for Efficient K-Way Merging
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

// Advanced Suffix Arrays with SA-IS Algorithm (Linear Time)
let text = b"banana";
let enhanced_sa = EnhancedSuffixArray::with_lcp(text).unwrap();
let sa = enhanced_sa.suffix_array();
let (start, count) = sa.search(text, b"an");
let lcp = enhanced_sa.lcp_array().unwrap();

// Existing high-performance algorithms
let mut data = vec![5u32, 2, 8, 1, 9];
let mut sorter = RadixSort::new();
sorter.sort_u32(&mut data).unwrap();

// Multi-way merge with vectorized sources
let sources = vec![
    VectorSource::new(vec![1, 4, 7]),
    VectorSource::new(vec![2, 5, 8]),
];
let mut merger = MultiWayMerge::new();
let result = merger.merge(sources).unwrap();
```

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

High-Performance String Processing - Zipora provides 3 comprehensive string processing components with Unicode support, hardware acceleration, and efficient line-based text processing:

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

### Performance

Current performance on Intel i7-10700K:

> **Note**: *AVX-512 optimizations require nightly Rust due to experimental intrinsics. All other SIMD optimizations (AVX2, BMI2, POPCNT) work with stable Rust.

| Operation | Performance | vs std::Vec | vs C++ | Security |
|-----------|-------------|-------------|--------|----------|
| FastVec push 10k | 6.78µs | +48% faster | +20% faster | ✅ Memory safe |
| **AutoGrowCircularQueue** | **1.54x** | **+54% faster** | **+54% faster** | ✅ **Ultra-fast (optimized)** |
| SecureMemoryPool alloc | ~18ns | +85% faster | +85% faster | ✅ **Production-ready** |
| Traditional pool alloc | ~15ns | +90% faster | +90% faster | ❌ Unsafe |
| Radix sort 1M u32s | ~45ms | +60% faster | +40% faster | ✅ Memory safe |
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

# Build with optional features
cargo build --release --features lz4             # Enable LZ4 compression
cargo build --release --features ffi             # Enable C FFI compatibility
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
```


## License

Licensed under The Bindiego License (BDL), Version 1.0. See [LICENSE](LICENSE) for details.

