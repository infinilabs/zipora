# Core Data Structures & Containers

Zipora includes specialized containers designed for memory efficiency and performance.

## High-Performance Containers

```rust
use zipora::{FastVec, FastStr, ValVec32, SmallMap, FixedCircularQueue,
            AutoGrowCircularQueue, UintVector, IntVec, FixedLenStrVec, SortableStrVec};
use zipora::containers::{LruMap, ConcurrentLruMap}; // not re-exported at crate root

// High-performance vector operations
let mut vec = FastVec::new();
vec.push(42).unwrap();

// Zero-copy string with SIMD hashing
let s = FastStr::from_string("hello world");
println!("Hash: {:x}", s.hash_fast());

// 32-bit indexed vectors - 50% memory reduction with golden ratio growth strategy
let mut vec32 = ValVec32::<u64>::new();
vec32.push(42).unwrap();
assert_eq!(vec32.get(0), Some(&42));

// Small maps - 3.8x faster than HashMap for <=8 elements
let mut small_map = SmallMap::<i32, String>::new();
small_map.insert(1, "one".to_string()).unwrap();
small_map.insert(2, "two".to_string()).unwrap();

// Fixed-size circular queue - lock-free, const generic size
let mut queue = FixedCircularQueue::<i32, 8>::new();
queue.push_back(1).unwrap();
queue.push_back(2).unwrap();
assert_eq!(queue.pop_front(), Some(1));

// Auto-growing circular queue - 1.17x faster than VecDeque
let mut auto_queue = AutoGrowCircularQueue::<String>::new();
auto_queue.push_back("hello".to_string()).unwrap();
auto_queue.push_back("world".to_string()).unwrap();

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

// Fixed-length strings - arena-based, no per-string heap allocation
// (FixedStr16Vec: 7.8x faster than Vec<String>)
let mut fixed_str_vec = FixedLenStrVec::<32>::new();
fixed_str_vec.push("hello").unwrap();
fixed_str_vec.push("world").unwrap();
assert_eq!(fixed_str_vec.get(0), Some("hello"));

// Arena-based string sorting with algorithm selection
let mut sortable = SortableStrVec::new();
sortable.push_str("cherry").unwrap();
sortable.push_str("apple").unwrap();
sortable.push_str("banana").unwrap();
sortable.sort_lexicographic().unwrap();
```

## LRU Cache Containers

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

// Advanced configuration: start from a preset, then set public fields
// (presets: performance_optimized(), memory_optimized(), security_optimized())
let mut config = LruMapConfig::performance_optimized();
config.capacity = 1024;
config.enable_statistics = true;
let cache: LruMap<String, String> = LruMap::with_config(config).unwrap();

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

// Advanced configuration: start from a preset, then set public fields.
// LoadBalancingStrategy::Hash is the only strategy (key-stable shard routing).
let mut config = ConcurrentLruMapConfig::performance_optimized();
config.load_balancing = LoadBalancingStrategy::Hash;
let cache: ConcurrentLruMap<String, String> = ConcurrentLruMap::with_config(config).unwrap();

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
- **Shard Load Balancing**: Hash-based shard distribution for reduced lock contention and cache coherence
- **Memory Efficient**: Intrusive linked list design minimizes overhead

## Container Performance Summary

| Container | Memory Reduction | Performance Gain | Use Case |
|-----------|------------------|------------------|----------|
| **ValVec32<T>** | **50%** | Golden ratio growth, near-parity | Large collections on 64-bit |
| **SmallMap<K,V>** | No heap allocation | **3.8x vs HashMap** | <=8 key-value pairs |
| **FixedCircularQueue** | Zero allocation | 20-30% faster | Lock-free ring buffers |
| **AutoGrowCircularQueue** | Cache-aligned | **1.17x vs VecDeque** | Ultra-fast vs VecDeque |
| **UintVector** | **68.7%** | <20% speed penalty | Compressed integers |
| **IntVec<T>** | Variable bit-width packing | **Hardware-accelerated** | Generic bit-packed storage |
| **FixedLenStrVec** | Arena-based, no per-string alloc | **7.8x vs Vec<String>** (FixedStr16Vec) | Arena-based fixed strings |
| **SortableStrVec** | Arena allocation | **Intelligent selection** | String collections |
| **LruMap<K,V>** | Intrusive linked list | **O(1) operations** | Single-threaded caching |
| **ConcurrentLruMap<K,V>** | Sharded architecture | **Reduced contention** | Multi-threaded caching |

Detailed, benchmarked numbers are maintained in [PERFORMANCE.md](PERFORMANCE.md).

## Unified Tries

```rust
use zipora::fsa::{ZiporaTrie, ZiporaTrieConfig, Trie, TrieStrategy, TrieCompressionStrategy};
use zipora::memory::{SecureMemoryPool, SecurePoolConfig};

// Default Patricia trie behavior
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

// Structural and memory statistics
let stats = string_trie.stats();
println!("States: {}, keys: {}, transitions: {}",
    stats.num_states, stats.num_keys, stats.num_transitions);
println!("Max depth: {}, avg depth: {:.2}", stats.max_depth, stats.avg_depth);
println!("Memory usage: {} bytes, {:.2} bits per key", stats.memory_usage, stats.bits_per_key);

// Space-optimized trie (formerly LOUDS/NestedLouds)
let mut compact_trie = ZiporaTrie::with_config(ZiporaTrieConfig::space_optimized());
compact_trie.insert(b"efficient").unwrap();

// High-performance concurrent trie (formerly DoubleArrayTrie)
let pool = SecureMemoryPool::new(SecurePoolConfig::default()).unwrap(); // already returns Arc<SecureMemoryPool>
let mut concurrent_trie = ZiporaTrie::with_config(
    ZiporaTrieConfig::concurrent_high_performance(pool)
);
concurrent_trie.insert(b"concurrent").unwrap();
```
