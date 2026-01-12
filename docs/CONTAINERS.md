# Core Data Structures & Containers

Zipora includes specialized containers designed for memory efficiency and performance.

## High-Performance Containers

```rust
use zipora::{FastVec, FastStr, ValVec32, SmallMap, FixedCircularQueue,
            AutoGrowCircularQueue, UintVector, IntVec, FixedLenStrVec, SortableStrVec,
            LruMap, ConcurrentLruMap,
            AdvancedStringVec, AdvancedStringConfig, BitPackedStringVec32, BitPackedStringVec64,
            BitPackedConfig};

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

// Small maps - 90% faster than HashMap for <=8 elements
let mut small_map = SmallMap::<i32, String>::new();
small_map.insert(1, "one".to_string()).unwrap();
small_map.insert(2, "two".to_string()).unwrap();

// Fixed-size circular queue - lock-free, const generic size
let mut queue = FixedCircularQueue::<i32, 8>::new();
queue.push_back(1).unwrap();
queue.push_back(2).unwrap();
assert_eq!(queue.pop_front(), Some(1));

// Auto-growing circular queue - 1.54x faster than VecDeque
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

// Fixed-length strings - 59.6% memory savings vs Vec<String>
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

## Advanced String Containers

```rust
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

## Container Performance Summary

| Container | Memory Reduction | Performance Gain | Use Case |
|-----------|------------------|------------------|----------|
| **ValVec32<T>** | **50%** | Golden ratio growth, near-parity | Large collections on 64-bit |
| **SmallMap<K,V>** | No heap allocation | **90% faster** | <=8 key-value pairs |
| **FixedCircularQueue** | Zero allocation | 20-30% faster | Lock-free ring buffers |
| **AutoGrowCircularQueue** | Cache-aligned | **54% faster** | Ultra-fast vs VecDeque |
| **UintVector** | **68.7%** | <20% speed penalty | Compressed integers |
| **IntVec<T>** | **96.9%** | **Hardware-accelerated** | Generic bit-packed storage |
| **FixedLenStrVec** | **59.6%** | **Zero-copy access** | Arena-based fixed strings |
| **SortableStrVec** | Arena allocation | **Intelligent selection** | String collections |
| **AdvancedStringVec** | **60-80%** | **3-level compression** | High-compression strings |
| **BitPackedStringVec32** | **50-70%** | **BMI2 acceleration** | Hardware-accelerated strings |
| **BitPackedStringVec64** | **40-60%** | **SIMD optimization** | Large-scale string datasets |
| **LruMap<K,V>** | Intrusive linked list | **O(1) operations** | Single-threaded caching |
| **ConcurrentLruMap<K,V>** | Sharded architecture | **Reduced contention** | Multi-threaded caching |

## Unified Tries

```rust
use zipora::fsa::{ZiporaTrie, ZiporaTrieConfig, Trie, TrieStrategy, CompressionStrategy};
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

// Automatic BMI2 hardware acceleration detection
let stats = string_trie.stats();
if stats.hardware_acceleration_enabled {
    println!("BMI2 hardware acceleration active for 5-10x faster operations");
}

// Advanced compression statistics
println!("Memory usage: {} bytes, {:.2} bits per key", stats.memory_usage, stats.bits_per_key);
println!("Compression ratio: {:.1}%", stats.compression_ratio * 100.0);

// Space-optimized trie (formerly LOUDS/NestedLouds)
let mut compact_trie = ZiporaTrie::with_config(ZiporaTrieConfig::space_optimized());
compact_trie.insert(b"efficient").unwrap();

// High-performance concurrent trie (formerly DoubleArrayTrie)
let pool = std::sync::Arc::new(SecureMemoryPool::new(SecurePoolConfig::default()).unwrap());
let mut concurrent_trie = ZiporaTrie::with_config(
    ZiporaTrieConfig::concurrent_high_performance(pool)
);
concurrent_trie.insert(b"concurrent").unwrap();
```
