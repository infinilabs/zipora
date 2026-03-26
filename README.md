# Zipora

[![Build Status](https://github.com/infinilabs/zipora/workflows/CI/badge.svg)](https://github.com/infinilabs/zipora/actions)
[![License](https://img.shields.io/badge/license-BDL--1.0-blue.svg)](LICENSE)
[![Rust Version](https://img.shields.io/badge/rust-1.88+-orange.svg)](https://www.rust-lang.org)

High-performance Rust data structures and compression algorithms with memory safety guarantees.

## Key Features

- **High Performance**: Zero-copy operations, SIMD optimizations (AVX2, AVX-512), cache-friendly layouts
- **Memory Safety**: 99.8% unsafe block documentation coverage, all production unsafe blocks annotated with `// SAFETY:` comments
- **Secure Memory Management**: Production-ready memory pools with thread safety and RAII
- **Blob Storage**: 8 specialized stores with trie-based indexing and compression
- **Succinct Data Structures**: 12 rank/select variants
- **Specialized Containers**: 13+ containers (VecTrbSet/Map, MinimalSso, SortedUintVec, LruMap, etc.)
- **Hash Maps**: Golden ratio optimized, string-optimized, cache-optimized implementations
- **Advanced Tries**: LOUDS, Critical-Bit (BMI2), Patricia tries with rank/select, NestTrieDawg
- **Compression**: PA-Zip, Huffman O0/O1/O2, FSE, rANS, ZSTD integration
- **C FFI Support**: Complete C API for migration from C++ (`--features ffi`)

## Quick Start

```toml
[dependencies]
zipora = "2.1.4"

# With C FFI bindings
zipora = { version = "2.1.4", features = ["ffi"] }

# AVX-512 (nightly only)
zipora = { version = "2.1.4", features = ["avx512"] }
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
- **[Algorithms](docs/ALGORITHMS.md)** - Radix sort, suffix arrays, set operations, cache-oblivious algorithms
- **[Compression](docs/COMPRESSION.md)** - PA-Zip, Huffman, FSE, rANS, real-time compression
- **[String Processing](docs/STRING_PROCESSING.md)** - SIMD string operations, pattern matching

### System Architecture
- **[Concurrency](docs/CONCURRENCY.md)** - Pipeline processing, work-stealing, parallel trie building
- **[Error Handling](docs/ERROR_HANDLING.md)** - Error classification, automatic recovery strategies
- **[Configuration](docs/CONFIGURATION.md)** - Rich configuration APIs, presets, validation
- **[SIMD Framework](docs/SIMD.md)** - 6-tier SIMD with AVX2/BMI2/POPCNT support

### Integration
- **[I/O & Serialization](docs/IO_SERIALIZATION.md)** - Stream processing, endian handling, varint encoding
- **[C FFI](docs/FFI.md)** - C API for migration from C++

### Performance Reports
- **[Performance vs C++](docs/PERF_VS_CPP.md)** - Benchmark comparisons
- **[Porting Status](docs/PORTING_STATUS.md)** - Feature parity status

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

| Component | Measured |
|-----------|----------|
| Rank/Select | 0.53 Gops/s (BMI2) |
| Huffman O1 | 2.1-2.6x speedup with fast symbol table |
| Radix Sort | 4-8x vs comparison sorts |
| SIMD Memory | 4-12x bulk operations |
| ValVec32 push | 0.79-0.87x vs std::Vec |
| ValVec32 random access | 1.0x vs std::Vec |
| ValVec32 iteration | 8.1% faster than std::Vec |

## Dependencies

Minimal dependency footprint by design:
- **Core**: `bytemuck`, `thiserror`, `log`, `ahash`, `rayon`, `libc`, `once_cell`, `raw-cpuid`
- **Default**: `memmap2` (mmap), `zstd`, `lz4_flex`, `serde`/`serde_json`/`bincode`, `tokio` (async)
- **Optional**: `cbindgen` (ffi)
- **Removed**: `crossbeam-utils`, `parking_lot`, `uuid`, `num_cpus`, `async-trait`, `futures` (all replaced with std or eliminated)

## License

Business Source License 1.0 - See [LICENSE](LICENSE) for details.
