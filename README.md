# Zipora

[中文](README_cn.md)

[![Build Status](https://github.com/infinilabs/zipora/workflows/CI/badge.svg)](https://github.com/infinilabs/zipora/actions)
[![License](https://img.shields.io/badge/license-BDL--1.0-blue.svg)](LICENSE)
[![Rust Version](https://img.shields.io/badge/rust-1.88+-orange.svg)](https://www.rust-lang.org)

High-performance Rust data structures and compression algorithms with memory safety guarantees.

## Key Features

- **High Performance**: Zero-copy operations, SIMD optimizations (AVX2, AVX-512), cache-friendly layouts
- **Memory Safety**: 99.8% unsafe block documentation coverage, all production unsafe blocks annotated with `// SAFETY:` comments
- **Secure Memory Management**: Production-ready memory pools with thread safety and RAII
- **Blob Storage**: 8 specialized stores with trie-based indexing and compression
- **Succinct Data Structures**: 12 rank/select variants, Rank9 (Vigna 2008), Elias-Fano / Partitioned / DP-Optimal Partitioned Elias-Fano, HybridPostingList (auto-select encoding), AMD-safe PDEP with `has_fast_bmi2`
- **Specialized Containers**: 13+ containers (VecTrbSet/Map, MinimalSso, SortedUintVec, LruMap, etc.)
- **Hash Maps**: Golden ratio optimized, string-optimized, cache-optimized implementations
- **Advanced Tries**: Double-Array (DoubleArrayTrie, XOR transitions), LOUDS, Critical-Bit (BMI2), Patricia tries with rank/select, NestTrieDawg
- **Compression**: PA-Zip, Huffman O0/O1/O2, FSE, rANS, ZSTD integration
- **C FFI Support**: Complete C API (`--features ffi`)

## Quick Start

```toml
[dependencies]
zipora = "3.1.1"

# With C FFI bindings
zipora = { version = "3.1.1", features = ["ffi"] }

# AVX-512 (nightly only)
zipora = { version = "3.1.1", features = ["avx512"] }
```

### Basic Usage

```rust
use zipora::*;

// High-performance vector
let mut vec = FastVec::new();
vec.push(42).unwrap();

// Zero-copy strings with SIMD hashing
let s = FastStr::from_string("hello world");
println!("Hash: {:x}", s.hash_fast());

// Intelligent rank/select with automatic optimization
let mut bv = BitVector::new();
for i in 0..1000 { bv.push(i % 7 == 0).unwrap(); }
let adaptive_rs = AdaptiveRankSelect::new(bv).unwrap();
let rank = adaptive_rs.rank1(500);

// Unified Trie - Strategy-based configuration
use zipora::fsa::{ZiporaTrie, ZiporaTrieConfig, Trie};

let mut trie = ZiporaTrie::new();
trie.insert(b"hello").unwrap();
assert!(trie.contains(b"hello"));

// Unified Hash Map - Strategy-based configuration
use zipora::hash_map::{ZiporaHashMap, ZiporaHashMapConfig};

let mut map = ZiporaHashMap::new();
map.insert("key", "value").unwrap();

// Blob storage with compression
let config = ZipOffsetBlobStoreConfig::performance_optimized();
let mut builder = ZipOffsetBlobStoreBuilder::with_config(config).unwrap();
builder.add_record(b"Compressed data").unwrap();
let store = builder.finish().unwrap();

// Entropy coding
let encoder = HuffmanEncoder::new(b"sample data").unwrap();
let compressed = encoder.encode(b"sample data").unwrap();

// String utilities
use zipora::string::{join_str, hex_encode, hex_decode, words, decimal_strcmp};
let joined = join_str(", ", &["hello", "world"]);
assert_eq!(joined, "hello, world");
```

## Documentation

### Core Components
- **[Containers](docs/CONTAINERS.md)** - Specialized containers (FastVec, ValVec32, IntVec, LruMap, etc.)
- **[Hash Maps](docs/HASH_MAPS.md)** - ZiporaHashMap, GoldHashMap with strategy-based configuration
- **[Blob Storage](docs/BLOB_STORAGE.md)** - 8 blob store variants with trie indexing and compression
- **[Memory Management](docs/MEMORY_MANAGEMENT.md)** - SecureMemoryPool, MmapVec, five-level pools

### Algorithms & Processing
- **[Algorithms](docs/ALGORITHMS.md)** - Radix sort, suffix arrays, set operations, cache-oblivious algorithms, SIMD popcount
- **[Compression](docs/COMPRESSION.md)** - PA-Zip, Huffman, FSE, rANS, real-time compression
- **[String Processing](docs/STRING_PROCESSING.md)** - SIMD string operations, pattern matching

### System Architecture
- **[Concurrency](docs/CONCURRENCY.md)** - Pipeline processing, work-stealing, parallel trie building
- **[Error Handling](docs/ERROR_HANDLING.md)** - Error classification, automatic recovery strategies
- **[Configuration](docs/CONFIGURATION.md)** - Rich configuration APIs, presets, validation
- **[SIMD Framework](docs/SIMD.md)** - 6-tier SIMD with AVX2/BMI2/POPCNT support

### Integration
- **[I/O & Serialization](docs/IO_SERIALIZATION.md)** - Stream processing, endian handling, varint encoding
- **[C FFI](docs/FFI.md)** - C API for interoperability

### Reference
- **[Porting Status](docs/PORTING_STATUS.md)** - Feature implementation status

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `simd` | Yes | SIMD optimizations (AVX2, SSE4.2) |
| `mmap` | Yes | Memory-mapped file support |
| `zstd` | Yes | ZSTD compression |
| `serde` | Yes | Serialization support (serde, serde_json, bincode) |
| `lz4` | Yes | LZ4 compression |
| `async` | Yes | Async runtime (tokio) for concurrency, pipeline, real-time compression |
| `ffi` | No | C FFI bindings |
| `avx512` | No | AVX-512 (nightly only) |
| `nightly` | No | Nightly-only optimizations |

## Build & Test

```bash
# Build (default features)
cargo build --release

# Build with all features including FFI
cargo build --release --all-features

# Test
cargo test --lib

# Sanity check (all feature combinations, debug + release)
make sanity

# Benchmark (release only)
cargo bench

# Lint
cargo clippy --all-targets --all-features -- -D warnings
```

## Verified Performance

> **Test Machine**: AMD EPYC 7B13 (Zen 3), 64 vCPUs, 117 GB RAM, AVX2/BMI2/POPCNT, rustc 1.91.1, Linux 6.17.
> Results vary across hardware — Intel may differ on BMI2 (native vs microcode), ARM lacks x86 SIMD paths.
> Run `cargo bench` to reproduce on your own hardware.

### Trie / Term Dictionary (DoubleArrayTrie)

| Operation (5000 terms) | Time | Per-op |
|------------------------|------|--------|
| Lookup hit | 103 µs | 20.6 ns/lookup |
| Lookup miss | 19 µs | 3.8 ns/lookup |
| Prefix search (5 queries) | 14 µs | 2.8 µs/query |
| Insert (incremental) | 967 µs | 193 ns/term |

XOR transitions, terminal bit in NInfo, unsafe `get_unchecked` — 3 ops, 1 branch per transition.
Supports arbitrary binary keys including `\x00` bytes.

### BitVector (scatter + popcount)

| Operation (1M bits) | Zipora | Scalar Vec\<u64\> | Ratio |
|---------------------|--------|-------------------|-------|
| Scatter + popcount (20×5K docs) | **1.08 ms** | 1.35 ms | **0.80x (faster)** |
| Allocation (`with_size(1M, false)`) | **155 µs** | 247 µs | **0.63x (faster)** |
| Popcount only (50% density) | 9.25 µs | 9.26 µs | Tied |

`alloc_zeroed` (calloc), zero-copy `from_blocks`, SIMD `popcount_slice` (AVX-512 / POPCNT / AVX2 / NEON).

### popcount_slice (SIMD population count)

| Slice size | Throughput | Rate |
|------------|-----------|------|
| 16 words (128B) | 4.4 ns | 3.7 Gwords/s |
| 781 words (6KB, engine union buffer) | 150 ns | 5.2 Gwords/s |
| 10K words (80KB) | 1.9 µs | 5.4 Gwords/s |

Multi-tier dispatch: AVX-512 VPOPCNTDQ → hardware POPCNT → AVX2 vpshufb → NEON → scalar.
Used internally by `BitVector::count_ones()` and available as `zipora::algorithms::popcount_slice`.

### Succinct Data Structures

| Operation | Zipora | Baseline | Speedup |
|-----------|--------|----------|---------|
| Rank1 query (100K bits) | 192 ns | — | ~5.2 Gops/s |
| Select1 query (100K bits) | 5.4 ms / 100K queries | — | ~18.5 Mops/s |
| Bulk rank (SIMD, 50K) | 8.4 µs | 84.1 µs (individual) | **10x** |
| Bulk bitwise ops (SIMD, 50K) | 3.1 µs | 128.4 µs (individual) | **41x** |
| Range set (SIMD, 50K) | 3.2 µs | 17.9 µs (individual) | **5.6x** |

### Containers vs std

| Operation | Zipora | std | Ratio |
|-----------|--------|-----|-------|
| ValVec32 push (100K) | 119 µs | 120 µs | 1.0x |
| ValVec32 random access (100K) | 706 ns | 729 ns | **0.97x** |
| ValVec32 iteration (10K) | 778 ns | 783 ns | 1.0x |
| ValVec32 bulk extend (100K) | 21.8 µs | 28.7 µs | **0.76x** |
| SmallMap insert+lookup (8 keys) | 444 ns | 805 ns (HashMap) | **1.8x** |
| SmallMap lookup-intensive | 36.9 µs | 141.7 µs (HashMap) | **3.8x** |
| CircularQueue push+pop (100K) | 326 µs | 381 µs (VecDeque) | **0.86x** |
| FixedStr16Vec push (100K) | 755 µs | 5,906 µs (Vec\<String\>) | **7.8x** |
| SortableStrVec sort (5K) | 390 µs | 448 µs (Vec\<String\>) | **1.15x** |

### Entropy Coding (65KB input)

| Algorithm | Entropy 0.5 | Entropy 2.0 | Entropy 6.0 |
|-----------|-------------|-------------|-------------|
| Huffman O0 | 1,124 µs | 1,235 µs | 1,720 µs |
| Huffman O1 (x1 stream) | 188 µs | 173 µs | 188 µs |
| rANS64 | 405 µs | 351 µs | 426 µs |

### Cache (LRU vs HashMap)

| Operation | LruMap | HashMap | Note |
|-----------|--------|---------|------|
| Hot get (cap=64, 10K ops) | 5.7 µs | 152 µs | **26x** faster (hot-set fits in cache) |
| Hot get (cap=1024, 10K ops) | 94.6 µs | 152 µs | **1.6x** faster |
| Insert (cap=64, 10K ops) | 1,897 µs | 1,177 µs | 0.62x (eviction overhead) |

## Dependencies

Minimal dependency footprint by design:
- **Core**: `bytemuck`, `thiserror`, `log`, `ahash`, `rayon`, `libc`, `once_cell`, `raw-cpuid`
- **Default**: `memmap2` (mmap), `zstd`, `lz4_flex`, `serde`/`serde_json`/`bincode`, `tokio` (async)
- **Optional**: `cbindgen` (ffi)
- **Removed**: `crossbeam-utils`, `parking_lot`, `uuid`, `num_cpus`, `async-trait`, `futures` (all replaced with std or eliminated)

## Building a Search Engine with Zipora

Zipora provides the core building blocks for high-performance search engines: succinct posting lists, compressed document storage, trie-based term dictionaries, SIMD-accelerated query processing, and multi-threaded indexing pipelines.

### Architecture Overview

```
 Documents                    Query
     |                          |
     v                          v
 [Tokenizer]              [Query Parser]
     |                          |
     v                          v
 [Term Dictionary]  --->  [Term Lookup]        ZiporaTrie / DoubleArrayTrie
     |                          |
     v                          v
 [Inverted Index]  --->  [Posting Lists]       UintVecMin0 / SortedUintVec / BitVector
     |                          |
     v                          v
 [Document Store]  --->  [Doc Retrieval]       DictZipBlobStore / MixedLenBlobStore
     |                          |
     v                          v
 [Compression]            [Ranking]            HuffmanEncoder / Rans64Encoder
```

### 1. Term Dictionary (Trie-based)

Use `DoubleArrayTrie` (double-array trie with XOR transitions) for maximum performance — 8 bytes per state with O(1) transitions per byte. Supports arbitrary binary keys including `\x00` bytes. For large vocabularies, it's 3-5x more memory-efficient than `HashMap<String, u32>` while providing faster lookups.

```rust
use zipora::DoubleArrayTrie;

// Build term dictionary during indexing
let mut dict = DoubleArrayTrie::new();

for term in terms.iter() {
    dict.insert(term.as_bytes()).unwrap();
}

// Query-time lookup: O(|key|) with O(1) per-byte transitions
assert!(dict.contains(b"search"));

// For key-value storage (term → term_id)
// DoubleArrayTrieMap<V> requires V: MapValue (configurable sentinel for zero-cost Option<V> elimination)
// Built-in impls: i32 (MIN), u32 (MAX), i64 (MIN), u64 (MAX), usize (MAX)
use zipora::DoubleArrayTrieMap;
let mut term_ids: DoubleArrayTrieMap<u32> = DoubleArrayTrieMap::new();
for (term_id, term) in terms.iter().enumerate() {
    term_ids.insert(term.as_bytes(), term_id as u32).unwrap();
}
let id = term_ids.get(b"search");
```

`DoubleArrayTrieMap<V>` uses the `MapValue` trait with a compile-time sentinel constant instead of `Option<V>`, halving the values array memory footprint for primitive types (e.g., 4 bytes vs 8 bytes per slot for `i32`). The sentinel is monomorphized to a single `cmp` instruction — zero runtime cost.

For alternative trie strategies (LOUDS, Patricia, CritBit), use `ZiporaTrie` with explicit config. For compressed term storage with prefix sharing, use `NestLoudsTrieBlobStore`.

### 2. Inverted Index (Posting Lists)

Choose the right container based on posting list characteristics:

```rust
use zipora::containers::{UintVecMin0, ZipIntVec};
use zipora::blob_store::SortedUintVec;
use zipora::BitVector;

// Option A: UintVecMin0 — variable-width packed integers (2-58 bits per value)
// Best for: medium-length posting lists with bounded doc IDs
let mut postings = UintVecMin0::new();
for doc_id in matching_docs {
    postings.push(doc_id);
}
// Access: postings.get(i) — O(1), cache-friendly sequential layout

// Option B: SortedUintVec — delta + block compression for sorted doc IDs
// Best for: long posting lists (60-80% space reduction vs raw u32)

// Option C: BitVector + RankSelect — bitmap representation
// Best for: high-frequency terms (>10% of docs), boolean queries
let mut bitmap = BitVector::new();
for i in 0..num_docs {
    bitmap.push(doc_ids.contains(&i)).unwrap();
}
```

### 3. Boolean Query Processing (Set Operations)

SIMD-accelerated set operations on posting lists — **up to 41x faster** than element-by-element processing for bitwise operations.

```rust
use zipora::algorithms::set_ops::{
    multiset_intersection,   // AND queries
    multiset_union,          // OR queries
    multiset_difference,     // NOT queries
    multiset_fast_intersection, // adaptive: picks best algo by size ratio
};

// AND query: "rust" AND "search"
let result = multiset_intersection(&postings_rust, &postings_search);

// For skewed sizes (one term rare, one common), use adaptive intersection
// Automatically picks linear merge vs binary search based on |A|/|B| ratio
let result = multiset_fast_intersection(&rare_term, &common_term);

// Bulk bitwise on rank/select bitvectors (41x faster with SIMD)
use zipora::AdaptiveRankSelect;
let rs = AdaptiveRankSelect::new(bitmap).unwrap();
let rank = rs.rank1(doc_id);   // count docs before this ID — O(1)
let pos = rs.select1(rank);    // find N-th matching doc — O(log n)
```

### 4. Document Storage (Compressed Blob Stores)

Store and retrieve documents with dictionary compression (PA-Zip):

```rust
use zipora::DictZipBlobStore;
use zipora::blob_store::{MixedLenBlobStore, PlainBlobStore, BlobStore};

// DictZipBlobStore: best compression for similar documents (web pages, logs)
// Learns a shared dictionary from training data, then compresses each record
let store = DictZipBlobStore::builder()
    .build_from_records(&documents)
    .unwrap();

// Retrieve: zero-copy access via mmap
let doc = store.get(doc_id).unwrap();

// MixedLenBlobStore: optimal for mixed fixed/variable-length records
// Automatically selects storage strategy based on record size distribution

// PlainBlobStore: uncompressed, fastest retrieval for hot data
```

### 5. Entropy Coding (Posting List Compression)

Compress posting list deltas with Huffman or rANS:

```rust
use zipora::HuffmanEncoder;
use zipora::Rans64Encoder;

// Huffman O0: simple, fast encoding (1.1 µs per 65KB)
let encoder = HuffmanEncoder::new(&training_data).unwrap();
let compressed = encoder.encode(&delta_encoded_postings).unwrap();

// Huffman O1: context-aware, better compression for structured data
// Particularly effective for posting list deltas with skewed distributions

// rANS: highest compression ratio, slightly slower
let rans = Rans64Encoder::new(&training_data).unwrap();
let compressed = rans.encode(&data).unwrap();
```

### 6. Multi-threaded Indexing

Parallelize index building with rayon and zipora's pipeline processing:

```rust
use rayon::prelude::*;
use zipora::algorithms::MultiWayMerge;

// Parallel document processing: each thread builds a segment
let segments: Vec<_> = document_batches
    .par_iter()
    .map(|batch| {
        let mut segment_index = SegmentIndex::new();
        for doc in batch {
            let terms = tokenize(doc);
            for term in terms {
                segment_index.add(term, doc.id);
            }
        }
        segment_index
    })
    .collect();

// Merge segments using k-way merge (loser tree)
use zipora::EnhancedLoserTree;
// EnhancedLoserTree provides O(log k) per element for k-way merge
// Ideal for merging sorted posting lists from parallel index segments
```

For async pipeline processing (requires `async` feature):

```rust
use zipora::Pipeline;
// Pipeline stages: parse → tokenize → index → compress → flush
// Each stage runs concurrently with work-stealing load balancing
```

### 7. Memory-Mapped Index Files

Serve large indices directly from disk without loading into RAM:

```rust
use zipora::memory::MmapVec;

// Memory-map an index file — OS manages paging
let index: MmapVec<u32> = MmapVec::open("postings.idx").unwrap();

// Random access is backed by the page cache
let doc_id = index[position];

// For blob stores, use mmap-backed storage
// DictZipBlobStore and NestLoudsTrieBlobStore support mmap natively
```

### 8. Query Result Caching

LRU cache for frequently accessed posting lists — **26x faster** hot-set retrieval vs HashMap:

```rust
use zipora::containers::specialized::LruMap;

// Cache hot posting lists
let mut cache: LruMap<String, Vec<u32>> = LruMap::new(1024);

fn get_postings(term: &str, cache: &mut LruMap<String, Vec<u32>>) -> Vec<u32> {
    if let Some(cached) = cache.get(term) {
        return cached.clone(); // 26x faster than HashMap for hot keys
    }
    let postings = load_from_disk(term);
    cache.insert(term.to_string(), postings.clone());
    postings
}
```

### 9. String Processing for Tokenization

```rust
use zipora::SortableStrVec;
use zipora::string::{decimal_strcmp, words};

// Arena-based string storage: 7.8x faster than Vec<String> for push (100K strings)
let mut terms = SortableStrVec::new();
for token in document.split_whitespace() {
    terms.push(token);
}
terms.sort(); // In-place sort, 1.15x faster than Vec<String>::sort

// For small lookup tables (field names, stop words), SmallMap is 3.8x faster
use zipora::SmallMap;
let mut stop_words = SmallMap::new();
stop_words.insert("the", true);
stop_words.insert("and", true);
```

### Component Selection Guide

| Search Engine Component | Zipora Type | When to Use |
|------------------------|-------------|-------------|
| Term dictionary | `DoubleArrayTrie` | Default choice, 8 bytes/state, XOR transitions |
| Term dictionary (alternatives) | `ZiporaTrie` | LOUDS/Patricia/CritBit via config |
| Short posting lists | `UintVecMin0` | Variable-width, <1M doc IDs |
| Long posting lists | `SortedUintVec` | Delta-compressed sorted IDs |
| Compressed posting lists | `HybridPostingList` | Auto-selects: Dense/EF/Partitioned/Optimal by list size |
| Rank/Select (large bitvecs) | `Rank9` | O(1) rank, O(log n) select, 25% overhead, hardware-independent |
| Boolean posting lists | `BitVector` + `AdaptiveRankSelect` | High-frequency terms, bitwise ops |
| AND/OR/NOT queries | `set_ops::multiset_*` | Sorted posting list intersection |
| Bulk bitwise queries | SIMD rank/select | 10-41x faster than scalar |
| Document storage | `DictZipBlobStore` | Best compression for similar docs |
| Document storage (fast) | `PlainBlobStore` | Uncompressed, fastest retrieval |
| Posting compression | `HuffmanEncoder` | Fast encode/decode |
| Posting compression | `Rans64Encoder` | Best compression ratio |
| Query cache | `LruMap` | 26x faster hot-set access |
| Small lookups | `SmallMap` | 3.8x faster for ≤8 keys |
| String storage | `SortableStrVec` / `FixedStr16Vec` | Arena-based, 7.8x vs Vec\<String\> |
| Index files | `MmapVec` | Disk-backed, OS-managed paging |
| Segment merge | `MultiWayMerge` / `EnhancedLoserTree` | K-way merge of sorted lists |
| Parallel indexing | `rayon` + `Pipeline` | Multi-threaded segment building |

## License

Business Source License 1.0 - See [LICENSE](LICENSE) for details.
