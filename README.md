# Zipora

[![Build Status](https://github.com/infinilabs/zipora/workflows/CI/badge.svg)](https://github.com/infinilabs/zipora/actions)
[![License](https://img.shields.io/badge/license-BDL--1.0-blue.svg)](LICENSE)
[![Rust Version](https://img.shields.io/badge/rust-1.88+-orange.svg)](https://www.rust-lang.org)

High-performance Rust data structures and compression algorithms with memory safety guarantees.

## Key Features

- **High Performance**: Zero-copy operations, SIMD optimizations (AVX2, AVX-512), cache-friendly layouts
- **Memory Safety**: Eliminates segfaults, buffer overflows, use-after-free bugs
- **Secure Memory Management**: Production-ready memory pools with thread safety and RAII
- **Blob Storage**: 8 specialized stores with trie-based indexing and compression
- **Specialized Containers**: 13+ containers with 40-90% memory/performance improvements
- **Hash Maps**: Golden ratio optimized, string-optimized, cache-optimized implementations
- **Advanced Tries**: LOUDS, Critical-Bit (BMI2), Patricia tries with rank/select
- **Compression**: PA-Zip, Huffman O0/O1/O2, FSE, rANS, ZSTD integration
- **Five-Level Concurrency**: Graduated control from single-thread to lock-free
- **C FFI Support**: Complete C API for migration from C++ (`--features ffi`)

> **Migration Note**: Version 2.0+ includes breaking changes. See [Migration Guide](docs/MIGRATION_GUIDE.md).

## Quick Start

```toml
[dependencies]
zipora = "2.1.2"

# With optional features
zipora = { version = "2.1.2", features = ["lz4", "ffi"] }

# AVX-512 (nightly only)
zipora = { version = "2.1.2", features = ["avx512"] }
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

// String-specialized trie
let mut string_trie = ZiporaTrie::with_config(ZiporaTrieConfig::string_specialized());
string_trie.insert(b"hello").unwrap();

// Unified Hash Map - Strategy-based configuration
use zipora::hash_map::{ZiporaHashMap, ZiporaHashMapConfig};

let mut map = ZiporaHashMap::new();
map.insert("key", "value").unwrap();

// String-optimized configuration
let mut string_map = ZiporaHashMap::with_config(ZiporaHashMapConfig::string_optimized());
string_map.insert("interned", 42).unwrap();

// Cache-optimized configuration
let mut cache_map = ZiporaHashMap::with_config(ZiporaHashMapConfig::cache_optimized());
cache_map.insert("cache", "optimized").unwrap();

// Blob storage with compression
let config = ZipOffsetBlobStoreConfig::performance_optimized();
let mut builder = ZipOffsetBlobStoreBuilder::with_config(config).unwrap();
builder.add_record(b"Compressed data").unwrap();
let store = builder.finish().unwrap();

// LRU Page Cache
use zipora::cache::{LruPageCache, PageCacheConfig};
let cache_config = PageCacheConfig::performance_optimized()
    .with_capacity(256 * 1024 * 1024);
let cache = LruPageCache::new(cache_config).unwrap();

// Entropy coding
let encoder = HuffmanEncoder::new(b"sample data").unwrap();
let compressed = encoder.encode(b"sample data").unwrap();

// String utilities
use zipora::string::{join_str, hex_encode, hex_decode, words, decimal_strcmp};

// Join strings efficiently
let joined = join_str(", ", &["hello", "world"]);
assert_eq!(joined, "hello, world");

// Hex encoding/decoding
let hex = hex_encode(b"Hello");
assert_eq!(hex, "48656c6c6f");
let bytes = hex_decode("48656c6c6f").unwrap();
assert_eq!(bytes, b"Hello");

// Word iteration
let word_list: Vec<_> = words(b"hello, world!").collect();
assert_eq!(word_list.len(), 2);

// Numeric string comparison
use std::cmp::Ordering;
assert_eq!(decimal_strcmp("100", "99"), Some(Ordering::Greater));
```

## Documentation

### Core Components
- **[Containers](docs/CONTAINERS.md)** - Specialized containers (FastVec, ValVec32, IntVec, LruMap, etc.)
- **[Hash Maps](docs/HASH_MAPS.md)** - ZiporaHashMap, GoldHashMap with strategy-based configuration
- **[Blob Storage](docs/BLOB_STORAGE.md)** - 8 blob store variants with trie indexing and compression
- **[Memory Management](docs/MEMORY_MANAGEMENT.md)** - SecureMemoryPool, MmapVec, five-level pools

### Algorithms & Processing
- **[Algorithms](docs/ALGORITHMS.md)** - Radix sort, suffix arrays, set operations, cache-oblivious algorithms
- **[Compression](docs/COMPRESSION.md)** - PA-Zip, Huffman, FSE, rANS, real-time compression
- **[String Processing](docs/STRING_PROCESSING.md)** - SIMD string operations, pattern matching

### System Architecture
- **[Concurrency](docs/CONCURRENCY.md)** - Five-level concurrency, version-based synchronization
- **[Error Handling](docs/ERROR_HANDLING.md)** - Error classification, automatic recovery strategies
- **[Configuration](docs/CONFIGURATION.md)** - Rich configuration APIs, presets, validation
- **[SIMD Framework](docs/SIMD.md)** - 6-tier SIMD with AVX2/BMI2/POPCNT support

### Integration
- **[I/O & Serialization](docs/IO_SERIALIZATION.md)** - Stream processing, endian handling, varint encoding
- **[C FFI](docs/FFI.md)** - C API for migration from C++
- **[Migration Guide](docs/MIGRATION_GUIDE.md)** - Upgrade from version 1.x

### Performance Reports
- **[Performance vs C++](docs/PERF_VS_CPP.md)** - Benchmark comparisons
- **[Porting Status](docs/PORTING_STATUS.md)** - Feature parity status

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `simd` | Yes | SIMD optimizations (AVX2, SSE4.2) |
| `mmap` | Yes | Memory-mapped file support |
| `zstd` | Yes | ZSTD compression |
| `serde` | Yes | Serialization support |
| `lz4` | No | LZ4 compression |
| `ffi` | No | C FFI bindings |
| `avx512` | No | AVX-512 (nightly only) |

## Build & Test

```bash
# Build
cargo build --release

# Test
cargo test --all-features

# Benchmark
cargo bench

# Lint
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
```

## Performance Targets

| Component | Target |
|-----------|--------|
| Rank/Select | 0.3-0.4 Gops/s with BMI2 |
| Radix Sort | 4-8x vs comparison sorts |
| SIMD Memory | 4-12x bulk operations |
| Cache Hit | >95% with prefetching |

## License

Business Source License 1.0 - See [LICENSE](LICENSE) for details.
