# Blob Storage Systems

Zipora provides 8 specialized blob storage implementations for different use cases.

## Trie-Based String Indexing (NestLoudsTrieBlobStore)

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
let all_keys = store.keys().unwrap();
let prefix_keys = store.keys_with_prefix(b"user/").unwrap();
let key_count = store.key_count();
let trie_stats = store.trie_stats();

// Comprehensive statistics and performance monitoring
let stats = store.stats();
println!("Blob count: {}", stats.blob_count);
println!("Cache hit ratio: {:.2}%", stats.cache_hit_ratio * 100.0);
println!("Trie compression ratio: {:.2}%", trie_stats.trie_space_saved_percent());
```

## Offset-Based Compressed Storage (ZipOffsetBlobStore)

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

## LRU Page Cache

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

## Specialized Blob Stores

### Zero-Length Blob Store

```rust
use zipora::{ZeroLengthBlobStore, BlobStore};

// Optimized storage for zero-length blobs (empty records)
// O(1) memory overhead regardless of record count
let mut store = ZeroLengthBlobStore::new();

// Add empty records efficiently
let id1 = store.put(b"").unwrap();
let id2 = store.put(&[]).unwrap();
let id3 = store.put(b"").unwrap();

// All get operations return empty vectors
assert_eq!(store.get(id1).unwrap(), b"");
assert!(store.contains(id2));
assert_eq!(store.len(), 3);

// Perfect for sparse indexes, placeholder records, or bitmap storage
```

### Simple Zip Blob Store

```rust
use zipora::{SimpleZipBlobStore, SimpleZipConfig, SimpleZipConfigBuilder, BlobStore};

// Fragment-based compression with HashMap deduplication
let config = SimpleZipConfig::builder()
    .delimiters(vec![b'\n', b' ', b'\t'])  // Split at whitespace
    .min_fragment_len(3)
    .max_fragment_len(64)
    .enable_deduplication(true)
    .build().unwrap();

let records = vec![
    b"GET /api/users HTTP/1.1".to_vec(),
    b"GET /api/posts HTTP/1.1".to_vec(),
    b"POST /api/users HTTP/1.1".to_vec(),
];

let store = SimpleZipBlobStore::build_from(records, config).unwrap();

// Retrieve records efficiently
let id = 0;
let data = store.get(id).unwrap();
assert_eq!(data, b"GET /api/users HTTP/1.1");

// Ideal for datasets with shared substrings (logs, JSON, configuration files)
let stats = store.stats();
println!("Deduplication saved: {:.1}% space",
         (1.0 - stats.average_size / stats.total_size as f64) * 100.0);
```

### Mixed-Length Blob Store

```rust
use zipora::{MixedLenBlobStore, BlobStore};

// Hybrid storage for datasets with mixed fixed/variable-length records
let records = vec![
    b"FIXED".to_vec(),     // 5 bytes (common length)
    b"FIXED".to_vec(),     // 5 bytes
    b"FIXED".to_vec(),     // 5 bytes
    b"VARIABLE LENGTH".to_vec(),  // Different length
    b"FIXED".to_vec(),     // 5 bytes
];

let store = MixedLenBlobStore::build_from(records, 5).unwrap();

// Automatic rank/select bitmap distinguishes fixed from variable
let id = 0;
let data = store.get(id).unwrap();
assert_eq!(data, b"FIXED");

// Best for datasets where >=50% records share same length
let stats = store.stats();
println!("Fixed-length ratio: {:.1}%",
         stats.blob_count as f64 / store.len() as f64 * 100.0);
```

### ZReorderMap - RLE Reordering Utility

```rust
use zipora::{ZReorderMap, ZReorderMapBuilder};
use tempfile::NamedTempFile;

// Build a reorder map with ascending sequences (sign = 1)
let temp_file = NamedTempFile::new().unwrap();
let path = temp_file.path();

{
    let mut builder = ZReorderMapBuilder::new(path, 1_000_000, 1).unwrap();
    // Push reordering values - consecutive sequences are automatically compressed
    for i in 0..500_000 {
        builder.push(i).unwrap();          // Sequence 1: 0..500_000
    }
    for i in 1_000_000..1_500_000 {
        builder.push(i).unwrap();          // Sequence 2: 1_000_000..1_500_000
    }
    builder.finish().unwrap();
}

// Read back the mapping
let map = ZReorderMap::open(path).unwrap();
println!("Total elements: {}", map.size());

// Iterate through reordering values
for (idx, value) in map.enumerate() {
    if idx < 10 {
        println!("Index {} -> Value {}", idx, value);
    }
}

// Compression efficiency: 1M elements compressed from 8MB to ~100 bytes
// Perfect for optimizing blob store access patterns
```

## Blob Storage Performance Summary

| Storage Type | Memory Efficiency | Throughput | Features | Best Use Case |
|--------------|------------------|------------|----------|---------------|
| **NestLoudsTrieBlobStore** | Trie + blob compression | O(key) + O(1) retrieval | String indexing, prefix queries | Hierarchical data |
| **ZipOffsetBlobStore** | Block-based delta compression | O(1) offset-based | Template optimization, ZSTD | Large datasets |
| **ZeroLengthBlobStore** | O(1) overhead | O(1) all operations | Bitmap-only storage | Sparse indexes |
| **SimpleZipBlobStore** | Fragment deduplication | O(1) indexed access | Delimiter-based splitting | Logs, JSON |
| **MixedLenBlobStore** | Rank/select hybrid | O(1) bitmap + vector | Fixed/variable separation | Mixed-length datasets |
| **ZReorderMap** | RLE compression | O(1) amortized sequential | 5-byte values + var_uint | Blob store reordering |
| **LRU Page Cache** | Page-aligned allocation | Reduced contention | Multi-shard architecture | High-concurrency access |
